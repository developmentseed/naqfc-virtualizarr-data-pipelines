"""Backfill tests.

Two layers, deliberately separated:

* the fork/merge/promote helpers, which are dataset-independent and run against
  `StubProcessor` so they stay fast and offline;
* the NAQFC store shape and its region writes, which need real GRIB and are
  marked `network`.
"""

import pathlib
import pickle

import icechunk
import numpy as np
import pytest
import xarray as xr
import zarr
from stub_processor import StubProcessor
from virtualizarr_processor import backfill, naqfc
from virtualizarr_processor.processor import Processor

# --- generic fork / merge / promote mechanics (offline) --------------------


def test_backfill_repo_has_main_branch(backfill_repo: icechunk.Repository) -> None:
    assert "main" in backfill_repo.list_branches()


def _worker(shared_fork_bytes: bytes, keys: list[str]) -> bytes:
    processor = StubProcessor()
    child = pickle.loads(shared_fork_bytes).fork()
    for key in keys:
        assert processor.process_backfill_file(key, child)
    return pickle.dumps(child)


def test_full_backfill_round_trip(backfill_repo: icechunk.Repository) -> None:
    processor = StubProcessor()
    processor.initialize_backfill_store(backfill_repo)

    shared = backfill.create_fork(backfill_repo)
    child_a = _worker(shared, ["0", "1", "2"])
    child_b = _worker(shared, ["3", "4", "5"])

    tip = backfill.merge_and_commit(
        backfill_repo, [child_a, child_b], message="backfill commit"
    )
    assert isinstance(tip, str) and tip

    arr = zarr.open_group(backfill_repo.readonly_session("backfill").store, mode="r")[
        "foo"
    ]
    expected = np.arange(6)[:, None, None]
    assert (np.asarray(arr[:]) == expected).all()

    backfill.promote(backfill_repo)
    arr_main = zarr.open_group(backfill_repo.readonly_session("main").store, mode="r")[
        "foo"
    ]
    assert (np.asarray(arr_main[:]) == expected).all()


def test_open_backfill_repo_local_filesystem(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ICECHUNK_BUCKET", raising=False)
    monkeypatch.setenv("ICECHUNK_LOCAL_PATH", str(tmp_path / "repo"))
    processor = Processor()

    repo = processor.open_backfill_repo()

    assert isinstance(repo, icechunk.Repository)
    assert "main" in repo.list_branches()
    # main must have a resolvable tip so initialize_backfill_store can branch off it
    assert repo.lookup_branch("main")


# --- NAQFC store shape (offline, synthetic reference cube) -----------------


def _fake_reference_cube() -> xr.Dataset:
    """A stand-in for one parsed cycle, shaped like `cycle_cube` output.

    `full_shape_skeleton` copies array metadata rather than reconstructing it,
    so it never inspects the codec and a plain BytesCodec stands in for
    gribberish here. That keeps the extent arithmetic testable without GRIB.
    """
    from virtualizarr.manifests import ChunkManifest, ManifestArray
    from zarr.codecs import BytesCodec
    from zarr.core.dtype import parse_data_type
    from zarr.core.metadata import ArrayV3Metadata

    ny, nx = 4, 5
    zdtype = parse_data_type(np.dtype("float64"), zarr_format=3)
    metadata = ArrayV3Metadata(
        shape=(1, 3, ny, nx),
        data_type=zdtype,
        chunk_grid={
            "name": "regular",
            "configuration": {"chunk_shape": (1, 1, ny, nx)},
        },
        chunk_key_encoding={"name": "default"},
        fill_value=zdtype.default_scalar(),
        codecs=[BytesCodec()],
        attributes={},
        dimension_names=naqfc.DIMS,
        storage_transformers=None,
    )
    ma = ManifestArray(
        chunkmanifest=ChunkManifest({}, shape=(1, 3, 1, 1)), metadata=metadata
    )
    return xr.Dataset(
        {naqfc.VARIABLE: xr.Variable(naqfc.DIMS, ma)},
        coords={
            "reference_time": (
                "reference_time",
                [np.datetime64("2025-01-01T06", "ns")],
            ),
            "lead": ("lead", np.arange(1, 4, dtype="int32")),
            "y": ("y", np.arange(ny, dtype="float64")),
            "x": ("x", np.arange(nx, dtype="float64")),
        },
    )


def test_skeleton_declares_full_extent_with_no_references() -> None:
    refs = naqfc.cycle_reference_times("2025-01-01", "2025-01-03", ("06", "12"))
    lead = naqfc.lead_axis(3)

    skeleton = naqfc.full_shape_skeleton(refs, lead, _fake_reference_cube())
    array = skeleton[naqfc.VARIABLE].data

    assert array.shape == (6, 3, 4, 5)
    assert array.metadata.chunk_grid.chunk_shape == (1, 1, 4, 5)
    assert array.metadata.dimension_names == naqfc.DIMS
    # extent only: every chunk arrives later via a region write
    assert len(array.manifest.items()) == 0


def test_skeleton_reference_time_axis_matches_inventory() -> None:
    args = ("2025-01-01", "2025-01-03", ("06", "12"))
    refs = naqfc.cycle_reference_times(*args)

    skeleton = naqfc.full_shape_skeleton(
        refs, naqfc.lead_axis(3), _fake_reference_cube()
    )

    assert skeleton.sizes["reference_time"] == len(naqfc.cycle_urls(*args))
    assert np.array_equal(skeleton.reference_time.values, refs)


def test_skeleton_pins_absolute_time_units() -> None:
    """Left to inference, a single-valued first write yields
    `days since <that cycle>`, and every later value reads back days off."""
    refs = naqfc.cycle_reference_times("2025-01-01", "2025-01-02", ("06", "12"))

    skeleton = naqfc.full_shape_skeleton(
        refs, naqfc.lead_axis(3), _fake_reference_cube()
    )

    for name in ("reference_time", "valid_time"):
        assert skeleton[name].encoding["units"] == "hours since 1970-01-01"


def test_skeleton_valid_time_is_derived() -> None:
    refs = naqfc.cycle_reference_times("2025-01-01", "2025-01-01", ("06", "12"))
    lead = naqfc.lead_axis(3)

    skeleton = naqfc.full_shape_skeleton(refs, lead, _fake_reference_cube())

    assert skeleton.valid_time.dims == ("reference_time", "lead")
    assert skeleton.valid_time.values[1, 2] == refs[1] + np.timedelta64(3, "h")


# --- NAQFC region writes (network) -----------------------------------------


@pytest.mark.network
def test_initialize_backfill_store_creates_full_shape(
    naqfc_repo: icechunk.Repository, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(naqfc, "START", "2025-01-01")
    monkeypatch.setattr(naqfc, "END", "2025-01-01")

    snapshot = Processor().initialize_backfill_store(naqfc_repo)

    assert isinstance(snapshot, str) and snapshot
    assert "backfill" in naqfc_repo.list_branches()
    root = zarr.open_group(naqfc_repo.readonly_session("backfill").store, mode="r")
    arr = root[naqfc.VARIABLE]
    assert arr.shape == (2, naqfc.LEAD_HOURS, 1025, 1473)
    assert arr.chunks == (1, 1, 1025, 1473)  # one GRIB message per chunk
    assert arr.dtype == np.dtype("float64")


@pytest.mark.network
def test_two_cycles_write_disjoint_regions(
    naqfc_repo: icechunk.Repository, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 06z and 12z runs of a day share 66 valid times. Keyed by
    (reference_time, lead) they occupy disjoint regions, so two forks merge
    without conflict and both runs' views survive."""
    monkeypatch.setattr(naqfc, "START", "2025-01-01")
    monkeypatch.setattr(naqfc, "END", "2025-01-01")
    processor = Processor()
    processor.initialize_backfill_store(naqfc_repo)
    urls = naqfc.cycle_urls("2025-01-01", "2025-01-01", ("06", "12"))

    shared = backfill.create_fork(naqfc_repo)
    children = []
    for url in urls:
        child = pickle.loads(shared).fork()
        assert processor.process_backfill_file(url, child)
        children.append(pickle.dumps(child))
    backfill.merge_and_commit(naqfc_repo, children, message="two cycles")
    backfill.promote(naqfc_repo)

    cube = xr.open_zarr(
        naqfc_repo.readonly_session("main").store, consolidated=False, zarr_format=3
    )
    point = cube[naqfc.VARIABLE].isel(y=500, x=700).load()
    for row in range(2):
        assert int(np.count_nonzero(~np.isnan(point.isel(reference_time=row)))) == (
            naqfc.LEAD_HOURS
        )
    assert (
        len(np.intersect1d(cube.valid_time.values[0], cube.valid_time.values[1])) == 66
    )
