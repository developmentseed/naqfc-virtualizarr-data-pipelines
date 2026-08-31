"""Processor-level tests for the NAQFC implementation.

Everything that can be checked without reading GRIB is checked offline: the
protocol conformance, the cycle enumeration that the store axis and the
inventory both derive from, and the cube reshaping rules. The round trips that
genuinely need bytes from NOAA's bucket are marked `network`.
"""

import functools
import pathlib
from datetime import datetime, timedelta, timezone

import icechunk
import numpy as np
import pytest
import xarray as xr
from icechunk import Repository
from virtualizarr_processor import naqfc
from virtualizarr_processor.processor import (
    Processor,
    sort_reference_time,
    store_reference_times,
    write_plan,
)
from virtualizarr_processor.typing import VirtualizarrProcessor


def protocol_type_check(processor: VirtualizarrProcessor) -> None:
    assert processor


def test_follows_protocol() -> None:
    protocol_type_check(processor=Processor())


# --- cycle enumeration -----------------------------------------------------


def test_cycle_url_matches_published_layout() -> None:
    from datetime import date

    assert naqfc.cycle_url(date(2025, 6, 1), "06") == (
        "s3://noaa-nws-naqfc-pds/AQMv7/CS/20250601/06/"
        "aqm.t06z.ave_1hr_o3.20250601.227.grib2"
    )


def test_cycles_are_chronological_and_complete() -> None:
    urls = naqfc.cycle_urls("2025-01-01", "2025-12-31", ("06", "12"))
    assert len(urls) == 730  # 365 days x 2 cycles
    assert urls == sorted(urls)
    assert len(set(urls)) == len(urls)


def test_reference_times_align_with_inventory_order() -> None:
    """The store's reference_time axis and the inventory must describe the same
    cycles. region="auto" aligns by coordinate value, so an inventory cycle
    missing from the axis fails loudly -- but an axis row missing from the
    inventory is silent, leaving an empty row nothing reports."""
    args = ("2025-01-01", "2025-01-04", ("06", "12"))
    urls = naqfc.cycle_urls(*args)
    refs = naqfc.cycle_reference_times(*args)

    assert len(urls) == len(refs)
    for url, ref in zip(urls, refs):
        day, cycle = url.split("/")[-3], url.split("/")[-2]
        assert ref == np.datetime64(
            f"{day[:4]}-{day[4:6]}-{day[6:]}T{cycle}:00:00", "ns"
        )


def test_no_valid_time_coordinate_is_written() -> None:
    """valid_time is exactly reference_time + lead, so the store doesn't carry
    it -- a 2-D redundant coordinate would have to be kept consistent on every
    append. Consumers derive it instead."""
    assert not hasattr(naqfc, "valid_time_grid")
    assert "valid_time" not in naqfc.STATIC_COORDS


def test_leap_day_included() -> None:
    urls = naqfc.cycle_urls("2024-02-28", "2024-03-01", ("06",))
    assert any("20240229" in u for u in urls)
    assert len(urls) == 3


def test_grid_derives_from_domain() -> None:
    """Grid and domain are not independent -- a mismatched pair builds URLs that
    resolve to nothing, so the grid follows the domain unless overridden."""
    assert naqfc.grid_for_domain("CS") == "227"  # Lambert conformal
    assert naqfc.grid_for_domain("AK") == "198"  # polar stereographic
    assert naqfc.grid_for_domain("HI") == "196"  # Mercator
    assert naqfc.grid_for_domain("CS", "999") == "999"  # explicit override wins


def test_other_domains_and_products_build_urls() -> None:
    """One deployment per (domain, product); the URL pattern follows both."""
    ak = naqfc.cycle_urls("2025-06-01", "2025-06-01", ("06",), domain="AK", grid="198")[
        0
    ]
    assert ak.endswith("AQMv7/AK/20250601/06/aqm.t06z.ave_1hr_o3.20250601.198.grib2")

    hi_pm25 = naqfc.cycle_urls(
        "2025-06-01",
        "2025-06-01",
        ("06",),
        domain="HI",
        grid="196",
        product="ave_1hr_pm25",
    )[0]
    assert hi_pm25.endswith(
        "AQMv7/HI/20250601/06/aqm.t06z.ave_1hr_pm25.20250601.196.grib2"
    )


# --- repository plumbing ---------------------------------------------------


def test_open_repo_local_filesystem(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ICECHUNK_BUCKET", raising=False)
    monkeypatch.setenv("ICECHUNK_LOCAL_PATH", str(tmp_path / "repo"))

    repo = Processor().initialize_repo()

    assert isinstance(repo, Repository)
    assert "main" in repo.list_branches()


def test_repo_authorizes_source_bucket(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Virtual chunks live in a bucket we do not own; without a container and
    credentials for it, every read fails at dereference time."""
    monkeypatch.delenv("ICECHUNK_BUCKET", raising=False)
    monkeypatch.setenv("ICECHUNK_LOCAL_PATH", str(tmp_path / "repo"))

    repo = Processor().initialize_repo()
    containers = repo.config.virtual_chunk_containers

    assert f"s3://{naqfc.BUCKET}/" in containers


def test_garbage_collect(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ICECHUNK_BUCKET", raising=False)
    monkeypatch.setenv("ICECHUNK_LOCAL_PATH", str(tmp_path / "repo"))

    gcs = Processor().garbage_collect(
        expiry_time=datetime.now(timezone.utc) - timedelta(days=2)
    )

    assert isinstance(gcs, icechunk.GCSummary)


# --- cube shaping (network) ------------------------------------------------


@pytest.mark.network
def test_cycle_cube_is_two_dimensional_in_time() -> None:
    cube = naqfc.cycle_cube(naqfc.cycle_urls("2025-01-01", "2025-01-01", ("06",))[0])

    assert cube[naqfc.VARIABLE].dims == naqfc.DIMS
    assert cube.sizes["reference_time"] == 1
    assert cube.sizes["lead"] == naqfc.LEAD_HOURS
    assert cube.reference_time.values[0] == np.datetime64("2025-01-01T06:00:00", "ns")
    assert list(cube.lead.values) == list(range(1, naqfc.LEAD_HOURS + 1))
    # per-cycle provenance must not become a store-wide attribute
    assert "reference_date" not in cube[naqfc.VARIABLE].attrs


@pytest.mark.network
def test_region_cube_drops_shared_grid_coords() -> None:
    """Every worker writes the same grid coordinates otherwise, and Icechunk
    treats concurrent writes to one chunk as a conflict even when the bytes
    agree."""
    cube = naqfc.cycle_cube(naqfc.cycle_urls("2025-01-01", "2025-01-01", ("06",))[0])
    region = naqfc.region_cube(cube)

    for dropped in naqfc.STATIC_COORDS:
        assert dropped not in region.variables
    assert naqfc.VARIABLE in region
    # the coordinates region="auto" resolves the target slice from must survive
    assert "reference_time" in region.coords
    assert "lead" in region.coords


@pytest.mark.network
def test_truncated_cycle_is_rejected() -> None:
    """A short run would shift every later lead index, so it must fail loudly
    rather than write a misaligned region."""
    url = naqfc.cycle_urls("2025-01-01", "2025-01-01", ("06",))[0]

    with pytest.raises(ValueError, match="lead steps, expected"):
        naqfc.cycle_cube(url, lead_hours=99)


# --- forward write mode ----------------------------------------------------


def synthetic_cube(reference_time: str) -> xr.Dataset:
    """A cycle cube's shape without the GRIB: same dims, coords and variable
    name as `naqfc.cycle_cube`, so the write-mode rules can be checked offline."""
    lead = np.arange(1, 4, dtype="int32")
    return xr.Dataset(
        {naqfc.VARIABLE: (naqfc.DIMS, np.zeros((1, len(lead), 2, 2), dtype="float32"))},
        coords={
            "reference_time": (
                "reference_time",
                np.array([reference_time], dtype="datetime64[ns]"),
            ),
            "lead": ("lead", lead),
            "y": ("y", np.arange(2)),
            "x": ("x", np.arange(2)),
        },
    )


def test_write_plan_creates_when_the_store_is_empty() -> None:
    """The first file of a forward-only deployment has no array to write into,
    so the write has to create one."""
    plan = write_plan(synthetic_cube("2025-01-01T06:00:00"), None)

    assert plan.mode == "create"
    assert plan.kwargs == {}


def test_write_plan_appends_a_cycle_off_the_end_of_the_axis() -> None:
    existing = np.array(["2025-01-01T06:00:00"], dtype="datetime64[ns]")

    plan = write_plan(synthetic_cube("2025-01-01T12:00:00"), existing)

    assert plan.mode == "append"
    assert plan.kwargs == {"append_dim": "reference_time"}
    # the grid coordinates ride along: with no reference_time dimension they
    # leave the already-written copies alone rather than duplicating them
    assert "y" in plan.cube.coords


def test_write_plan_regions_a_cycle_already_on_the_axis() -> None:
    """Appending a reference_time the store already carries would store the
    cycle twice and leave the axis non-monotonic, so it is written in place."""
    existing = np.array(
        ["2025-01-01T06:00:00", "2025-01-01T12:00:00"], dtype="datetime64[ns]"
    )

    plan = write_plan(synthetic_cube("2025-01-01T12:00:00"), existing)

    assert plan.mode == "region"
    assert plan.kwargs == {"region": "auto"}
    # region="auto" resolves a slice per dimension, so variables that have none
    # of the region's dimensions cannot come along
    for dropped in naqfc.STATIC_COORDS:
        assert dropped not in plan.cube.variables
    assert "reference_time" in plan.cube.coords and "lead" in plan.cube.coords


def test_write_plan_matches_across_datetime_resolutions() -> None:
    """A store's axis decodes at whatever resolution its units imply, which need
    not be the cube's nanoseconds; the cycle is still the same cycle."""
    existing = np.array(["2025-01-01T06:00:00"], dtype="datetime64[s]")

    assert write_plan(synthetic_cube("2025-01-01T06:00:00"), existing).mode == "region"
    assert write_plan(synthetic_cube("2025-01-02T06:00:00"), existing).mode == "append"


@pytest.mark.network
def test_reprocessing_a_cycle_rewrites_its_row(naqfc_repo: Repository) -> None:
    """End to end over three writes: create, re-deliver the same cycle, then a
    new one. The re-delivery must leave one row, not two."""
    processor = Processor()
    session = processor.initialize_session(naqfc_repo)
    first, second = naqfc.cycle_urls("2025-01-01", "2025-01-01", ("06", "12"))

    assert processor.process_file(first, session)  # create
    assert processor.process_file(first, session)  # same cycle -> region write
    assert processor.process_file(second, session)  # new cycle -> append

    cube = xr.open_zarr(session.store, consolidated=False, zarr_format=3)
    assert cube.sizes["reference_time"] == 2
    assert list(cube.reference_time.values) == [
        np.datetime64("2025-01-01T06:00:00", "ns"),
        np.datetime64("2025-01-01T12:00:00", "ns"),
    ]


# --- forward write modes against a real store ------------------------------


@pytest.fixture
def local_repo(tmp_path: pathlib.Path) -> Repository:
    """A repo whose virtual chunks are local files.

    The processor's own repo points its virtual chunk container at NOAA's
    bucket, so writing anything into it needs real GRIB. This one stands in for
    it, letting the whole create/region/append cycle run offline.
    """
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    prefix = f"file://{chunks}/"
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(prefix, icechunk.local_filesystem_store(chunks))
    )
    return icechunk.Repository.open_or_create(
        storage=icechunk.local_filesystem_storage(str(tmp_path / "repo")),
        config=config,
        authorize_virtual_chunk_access={
            prefix: icechunk.credentials.LocalFileSystemAccess
        },
    )


def virtual_cycle_cube(
    chunks: pathlib.Path, reference_time: str, value: float, n_lead: int = 3
) -> xr.Dataset:
    """What `naqfc.cycle_cube` produces, with local bytes standing in for GRIB.

    Same dims, coords, encoding and one-message-per-chunk grid; only the chunks
    point at a local file instead of a range inside a NAQFC object.
    """
    from virtualizarr.manifests import ChunkManifest, ManifestArray
    from zarr.codecs import BytesCodec
    from zarr.core.dtype import parse_data_type
    from zarr.core.metadata import ArrayV3Metadata

    ny = nx = 2
    data = np.full((n_lead, ny, nx), value, dtype="float32")
    chunk_bytes = data[0].nbytes
    # Keyed by value as well as cycle: two cubes for the same cycle with
    # different values must not share a backing file, or writing the second
    # rewrites bytes the first's references already point at.
    path = chunks / f"{reference_time.replace(':', '')}-{value:g}.bin"
    path.write_bytes(data.tobytes())

    dtype = parse_data_type(data.dtype, zarr_format=3)
    array = ManifestArray(
        chunkmanifest=ChunkManifest(
            {
                f"0.{lead}.0.0": {
                    "path": f"file://{path}",
                    "offset": lead * chunk_bytes,
                    "length": chunk_bytes,
                }
                for lead in range(n_lead)
            }
        ),
        metadata=ArrayV3Metadata(
            shape=(1, n_lead, ny, nx),
            data_type=dtype,
            chunk_grid={
                "name": "regular",
                "configuration": {"chunk_shape": (1, 1, ny, nx)},
            },
            chunk_key_encoding={"name": "default"},
            fill_value=dtype.default_scalar(),
            codecs=[BytesCodec()],
            attributes={},
            dimension_names=naqfc.DIMS,
            storage_transformers=None,
        ),
    )
    cube = xr.Dataset(
        {naqfc.VARIABLE: xr.Variable(naqfc.DIMS, array)},
        coords={
            "reference_time": (
                "reference_time",
                np.array([reference_time], dtype="datetime64[ns]"),
            ),
            "lead": ("lead", np.arange(1, n_lead + 1, dtype="int32")),
            "y": ("y", np.arange(ny)),
            "x": ("x", np.arange(nx)),
        },
    )
    return naqfc.pin_time_encoding(cube)


def write_cycle(store: object, cube: xr.Dataset) -> str:
    plan = write_plan(cube, store_reference_times(store))
    plan.cube.vz.to_icechunk(store, **plan.kwargs)
    return plan.mode


def test_forward_writes_create_then_append_then_region(
    local_repo: Repository, tmp_path: pathlib.Path
) -> None:
    """The three modes in the order a deployment meets them, against a real
    store: the first cycle creates it, the next extends the axis, and a cycle
    already on the axis is rewritten in place rather than duplicated."""
    chunks = tmp_path / "chunks"
    session = local_repo.writable_session("main")

    assert (
        write_cycle(
            session.store, virtual_cycle_cube(chunks, "2025-01-01T06:00:00", 1.0)
        )
        == "create"
    )
    assert (
        write_cycle(
            session.store, virtual_cycle_cube(chunks, "2025-01-01T12:00:00", 2.0)
        )
        == "append"
    )
    assert (
        write_cycle(
            session.store, virtual_cycle_cube(chunks, "2025-01-01T06:00:00", 3.0)
        )
        == "region"
    )

    cube = xr.open_zarr(session.store, consolidated=False, zarr_format=3)

    assert list(cube.reference_time.values) == [
        np.datetime64("2025-01-01T06:00:00", "ns"),
        np.datetime64("2025-01-01T12:00:00", "ns"),
    ]
    # the re-delivered 06z cycle replaced its row; the 12z row is untouched
    assert (cube[naqfc.VARIABLE].isel(reference_time=0).values == 3.0).all()
    assert (cube[naqfc.VARIABLE].isel(reference_time=1).values == 2.0).all()
    # and the grid coordinates survived a region write that could not carry them
    assert list(cube.y.values) == [0, 1] and list(cube.x.values) == [0, 1]


# --- putting an out-of-order axis back in order -----------------------------


def axis_of(session: icechunk.Session) -> list[str]:
    store = xr.open_zarr(session.store, consolidated=False, zarr_format=3)
    return [str(t) for t in store.reference_time.values]


def test_sorting_an_ordered_axis_is_a_no_op(
    local_repo: Repository, tmp_path: pathlib.Path
) -> None:
    """`reindex_array` visits every chunk even when the permutation is the
    identity, and dirties the session doing it, so the ordered case -- which is
    almost every batch -- has to short-circuit before reaching it."""
    chunks = tmp_path / "chunks"
    session = local_repo.writable_session("main")
    cube = functools.partial(virtual_cycle_cube, chunks)

    write_cycle(session.store, cube("2025-01-01T06:00:00", 1.0))
    write_cycle(session.store, cube("2025-01-01T12:00:00", 2.0))
    Processor().commit_processed_files(session)

    session = local_repo.writable_session("main")
    assert sort_reference_time(session) == 0
    assert not session.has_uncommitted_changes


def test_cycles_appended_out_of_order_are_sorted_before_the_commit(
    local_repo: Repository, tmp_path: pathlib.Path
) -> None:
    """The whole point: a batch arrives jumbled, every cycle is appended where
    it falls, and the snapshot that lands is in order with each row holding its
    own forecast."""
    chunks = tmp_path / "chunks"
    session = local_repo.writable_session("main")
    cube = functools.partial(virtual_cycle_cube, chunks)

    assert write_cycle(session.store, cube("2025-01-02T06:00:00", 3.0)) == "create"
    assert write_cycle(session.store, cube("2025-01-01T06:00:00", 1.0)) == "append"
    assert write_cycle(session.store, cube("2025-01-01T12:00:00", 2.0)) == "append"

    # mid-session the axis is in arrival order, which is exactly why the sort
    # has to happen before the commit rather than after it
    assert axis_of(session) == [
        "2025-01-02T06:00:00.000000000",
        "2025-01-01T06:00:00.000000000",
        "2025-01-01T12:00:00.000000000",
    ]

    Processor().commit_processed_files(session)

    store = xr.open_zarr(
        local_repo.readonly_session("main").store, consolidated=False, zarr_format=3
    )
    assert [str(t) for t in store.reference_time.values] == [
        "2025-01-01T06:00:00.000000000",
        "2025-01-01T12:00:00.000000000",
        "2025-01-02T06:00:00.000000000",
    ]
    # each row carries its own cycle's data, not just a sorted coordinate
    for row, written in enumerate((1.0, 2.0, 3.0)):
        assert (store[naqfc.VARIABLE].isel(reference_time=row).values == written).all()


def test_a_cycle_older_than_the_whole_store_is_accepted(
    local_repo: Repository, tmp_path: pathlib.Path
) -> None:
    """Zarr can only grow an axis at its end, so a cycle older than everything
    in the store has nowhere to be inserted. Appending it and sorting afterwards
    puts it at row 0 regardless -- the case a placement-based write has to
    refuse outright."""
    chunks = tmp_path / "chunks"
    cube = functools.partial(virtual_cycle_cube, chunks)

    session = local_repo.writable_session("main")
    write_cycle(session.store, cube("2025-06-01T06:00:00", 5.0))
    Processor().commit_processed_files(session)

    session = local_repo.writable_session("main")
    assert write_cycle(session.store, cube("2025-01-01T06:00:00", 1.0)) == "append"
    Processor().commit_processed_files(session)

    store = xr.open_zarr(
        local_repo.readonly_session("main").store, consolidated=False, zarr_format=3
    )
    assert [str(t) for t in store.reference_time.values] == [
        "2025-01-01T06:00:00.000000000",
        "2025-06-01T06:00:00.000000000",
    ]
    assert (store[naqfc.VARIABLE].isel(reference_time=0).values == 1.0).all()
    assert (store[naqfc.VARIABLE].isel(reference_time=1).values == 5.0).all()


def test_every_committed_snapshot_is_ordered(
    local_repo: Repository, tmp_path: pathlib.Path
) -> None:
    """Commit by commit, not just at the end: a reader taking any snapshot must
    never see an axis in arrival order."""
    chunks = tmp_path / "chunks"
    cube = functools.partial(virtual_cycle_cube, chunks)
    arrivals = [
        ("2025-01-02T12:00:00", 4.0),
        ("2025-01-01T06:00:00", 1.0),
        ("2025-01-02T06:00:00", 3.0),
        ("2025-01-01T12:00:00", 2.0),
    ]

    for stamp, value in arrivals:
        session = local_repo.writable_session("main")
        write_cycle(session.store, cube(stamp, value))
        Processor().commit_processed_files(session)

        committed = axis_of(local_repo.readonly_session("main"))
        assert committed == sorted(committed), f"unordered after {stamp}"


def test_a_duplicate_cycle_is_refused_rather_than_guessed_at(
    local_repo: Repository, tmp_path: pathlib.Path
) -> None:
    """`write_plan` region-writes a cycle already on the axis, so a duplicate
    means something bypassed it. No sort can make the axis strictly increasing,
    and choosing a copy to drop would lose a forecast."""
    chunks = tmp_path / "chunks"
    session = local_repo.writable_session("main")
    cube = functools.partial(virtual_cycle_cube, chunks)

    write_cycle(session.store, cube("2025-01-01T06:00:00", 1.0))
    # bypass write_plan to force the duplicate a correct caller cannot create
    cube("2025-01-01T06:00:00", 9.0).vz.to_icechunk(
        session.store, append_dim="reference_time"
    )

    with pytest.raises(ValueError, match="duplicate cycles"):
        sort_reference_time(session)
