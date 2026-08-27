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


# --- the cycle schedule ----------------------------------------------------


def test_schedule_starts_after_the_stores_last_row() -> None:
    """The last row is itself a cycle, so a store ending at 06z that receives
    that day's 12z extends by exactly one row -- not by the whole day."""
    schedule = naqfc.schedule_between(
        np.datetime64("2025-01-01T06:00:00", "ns"),
        np.datetime64("2025-01-01T12:00:00", "ns"),
    )

    assert list(schedule) == [np.datetime64("2025-01-01T12:00:00", "ns")]


def test_schedule_spans_the_irregular_overnight_gap() -> None:
    """06z to 12z is 6 hours and 12z to the next 06z is 18, so a 6-hourly grid
    would invent 00z and 18z rows that no file could ever fill."""
    schedule = naqfc.schedule_between(
        np.datetime64("2025-01-01T06:00:00", "ns"),
        np.datetime64("2025-01-03T06:00:00", "ns"),
    )

    assert [str(t) for t in schedule] == [
        "2025-01-01T12:00:00.000000000",
        "2025-01-02T06:00:00.000000000",
        "2025-01-02T12:00:00.000000000",
        "2025-01-03T06:00:00.000000000",
    ]


def test_schedule_follows_the_configured_cycle_hours() -> None:
    schedule = naqfc.schedule_between(
        np.datetime64("2025-01-01T00:00:00", "ns"),
        np.datetime64("2025-01-02T00:00:00", "ns"),
        cycle_hours=("00", "06", "12", "18"),
    )

    assert len(schedule) == 4
    assert [t.astype("datetime64[h]").astype(object).hour for t in schedule] == [
        6,
        12,
        18,
        0,
    ]


def test_an_unscheduled_cycle_hour_is_refused() -> None:
    """Reindexing onto a schedule that excludes the cycle would drop its data
    silently, so an off-schedule hour has to fail loudly instead."""
    naqfc.check_scheduled(np.datetime64("2025-01-01T06:00:00", "ns"))

    with pytest.raises(ValueError, match="not a scheduled cycle"):
        naqfc.check_scheduled(np.datetime64("2025-01-01T18:00:00", "ns"))


def test_a_cycle_off_the_hour_is_refused() -> None:
    with pytest.raises(ValueError, match="not a scheduled cycle"):
        naqfc.check_scheduled(np.datetime64("2025-01-01T06:30:00", "ns"))


# --- placing a cycle on the axis -------------------------------------------


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


def test_write_plan_appends_the_next_scheduled_cycle_as_one_row() -> None:
    """Nothing was skipped, so nothing is reserved."""
    existing = np.array(["2025-01-01T06:00:00"], dtype="datetime64[ns]")

    plan = write_plan(synthetic_cube("2025-01-01T12:00:00"), existing)

    assert plan.mode == "append"
    assert plan.kwargs == {"append_dim": "reference_time"}
    assert plan.cube.sizes["reference_time"] == 1
    # the grid coordinates ride along: with no reference_time dimension they
    # leave the already-written copies alone rather than duplicating them
    assert "y" in plan.cube.coords


def test_write_plan_reserves_a_row_for_every_cycle_it_skipped() -> None:
    """The reserved rows are the whole point: a cycle still in flight gets an
    address now, so when it lands it is a region write instead of an
    out-of-order append."""
    existing = np.array(["2025-01-01T06:00:00"], dtype="datetime64[ns]")

    plan = write_plan(synthetic_cube("2025-01-02T06:00:00"), existing)

    assert plan.mode == "append"
    assert [str(t) for t in plan.cube.reference_time.values] == [
        "2025-01-01T12:00:00.000000000",
        "2025-01-02T06:00:00.000000000",
    ]
    # and the arriving cycle is in the row that belongs to it, not the first one
    assert not np.isnan(plan.cube[naqfc.VARIABLE].values[1]).any()
    assert np.isnan(plan.cube[naqfc.VARIABLE].values[0]).all()


def test_write_plan_keeps_the_time_encoding_through_the_reindex() -> None:
    """reindex does not carry encoding over, and a block written without it
    infers `days since <first row>` and reads back wrong by a factor of 24."""
    existing = np.array(["2025-01-01T06:00:00"], dtype="datetime64[ns]")

    plan = write_plan(synthetic_cube("2025-01-02T06:00:00"), existing)

    assert plan.cube["reference_time"].encoding["units"] == "hours since 1970-01-01"


def test_write_plan_refuses_a_cycle_older_than_the_store() -> None:
    """Zarr grows an axis only at its end."""
    existing = np.array(["2025-01-01T12:00:00"], dtype="datetime64[ns]")

    with pytest.raises(ValueError, match="older than the store"):
        write_plan(synthetic_cube("2025-01-01T06:00:00"), existing)


def test_write_plan_refuses_a_cycle_in_a_gap_no_row_was_reserved_for() -> None:
    """What a store appended to before schedule alignment looks like: the 12z
    row was never created, and Zarr cannot insert one in the middle."""
    existing = np.array(
        ["2025-01-01T06:00:00", "2025-01-02T06:00:00"], dtype="datetime64[ns]"
    )

    with pytest.raises(ValueError, match="no row of its own"):
        write_plan(synthetic_cube("2025-01-01T12:00:00"), existing)


def test_write_plan_refuses_to_reserve_more_rows_than_the_limit() -> None:
    """A file whose reference_date parses but is decades out would otherwise
    declare millions of empty rows to reach itself."""
    existing = np.array(["2025-01-01T06:00:00"], dtype="datetime64[ns]")

    with pytest.raises(ValueError, match="cycle limit"):
        write_plan(synthetic_cube("2035-01-01T06:00:00"), existing)


def test_write_plan_refuses_an_unscheduled_cycle_before_touching_the_store() -> None:
    with pytest.raises(ValueError, match="not a scheduled cycle"):
        write_plan(synthetic_cube("2025-01-01T18:00:00"), None)


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
    point at a local file instead of a range inside a NAQFC object. The fill
    value is NaN, as a decoded GRIB float field's is, so a row with no chunk
    reference is distinguishable from a row of real zeros.
    """
    from virtualizarr.manifests import ChunkManifest, ManifestArray
    from virtualizarr.manifests.utils import create_v3_array_metadata

    ny = nx = 2
    data = np.full((n_lead, ny, nx), value, dtype="float32")
    chunk_bytes = data[0].nbytes
    path = chunks / f"{reference_time.replace(':', '')}-{value:g}.bin"
    path.write_bytes(data.tobytes())

    array = ManifestArray(
        metadata=create_v3_array_metadata(
            shape=(1, n_lead, ny, nx),
            data_type=data.dtype,
            chunk_shape=(1, 1, ny, nx),
            fill_value=np.nan,
            dimension_names=naqfc.DIMS,
        ),
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


def write_cycle(session: icechunk.Session, cube: xr.Dataset) -> str:
    """One pass of what `process_file` does, minus the GRIB read."""
    plan = write_plan(cube, store_reference_times(session.store))
    plan.cube.vz.to_icechunk(session.store, **plan.kwargs)
    return plan.mode


def test_a_cycle_delivered_late_lands_in_the_row_reserved_for_it(
    local_repo: Repository, tmp_path: pathlib.Path
) -> None:
    """The whole point, against a real store: 12z is delivered after the next
    day's 06z, and the axis still comes out in order with every row correct."""
    chunks = tmp_path / "chunks"
    session = local_repo.writable_session("main")
    cube = functools.partial(virtual_cycle_cube, chunks)

    assert write_cycle(session, cube("2025-01-01T06:00:00", 1.0)) == "create"
    assert write_cycle(session, cube("2025-01-02T06:00:00", 3.0)) == "append"
    # the skipped 12z now has a row waiting, so its late arrival is a region
    # write rather than an append that would put it after the 2nd of January
    assert write_cycle(session, cube("2025-01-01T12:00:00", 2.0)) == "region"

    store = xr.open_zarr(session.store, consolidated=False, zarr_format=3)
    axis = store.reference_time.values

    assert list(axis) == sorted(axis)
    assert [str(t) for t in axis] == [
        "2025-01-01T06:00:00.000000000",
        "2025-01-01T12:00:00.000000000",
        "2025-01-02T06:00:00.000000000",
    ]
    for row, written in enumerate((1.0, 2.0, 3.0)):
        assert (store[naqfc.VARIABLE].isel(reference_time=row).values == written).all()
    # the grid coordinates survived a region write that could not carry them
    assert list(store.y.values) == [0, 1] and list(store.x.values) == [0, 1]


def test_a_reserved_row_costs_nothing_until_its_cycle_arrives(
    local_repo: Repository, tmp_path: pathlib.Path
) -> None:
    """A reserved row is a real reference_time over an empty manifest: it reads
    as the fill value and stores no chunk at all, which is also how ops can tell
    a cycle has not landed without fetching a byte."""
    chunks = tmp_path / "chunks"
    session = local_repo.writable_session("main")
    cube = functools.partial(virtual_cycle_cube, chunks)

    write_cycle(session, cube("2025-01-01T06:00:00", 1.0))
    write_cycle(session, cube("2025-01-02T06:00:00", 3.0))

    store = xr.open_zarr(session.store, consolidated=False, zarr_format=3)
    assert store.sizes["reference_time"] == 3
    assert np.isnan(store[naqfc.VARIABLE].isel(reference_time=1).values).all()

    array = f"/{naqfc.VARIABLE}"
    assert session.chunk_type(array, [1, 0, 0, 0]) == icechunk.ChunkType.uninitialized
    assert session.chunk_type(array, [0, 0, 0, 0]) == icechunk.ChunkType.virtual
    assert session.chunk_type(array, [2, 0, 0, 0]) == icechunk.ChunkType.virtual


def test_a_reserved_row_survives_a_commit_and_reopen(
    local_repo: Repository, tmp_path: pathlib.Path
) -> None:
    """Committed and read back through a fresh session, so the reserved row is
    a property of the store rather than of one uncommitted write."""
    chunks = tmp_path / "chunks"
    session = local_repo.writable_session("main")
    cube = functools.partial(virtual_cycle_cube, chunks)

    write_cycle(session, cube("2025-01-01T06:00:00", 1.0))
    session.commit("create")
    session = local_repo.writable_session("main")
    write_cycle(session, cube("2025-01-02T06:00:00", 3.0))
    session.commit("append over the gap")

    session = local_repo.writable_session("main")
    assert write_cycle(session, cube("2025-01-01T12:00:00", 2.0)) == "region"
    session.commit("fill the reserved row")

    store = xr.open_zarr(
        local_repo.readonly_session("main").store, consolidated=False, zarr_format=3
    )
    assert [str(t) for t in store.reference_time.values] == [
        "2025-01-01T06:00:00.000000000",
        "2025-01-01T12:00:00.000000000",
        "2025-01-02T06:00:00.000000000",
    ]
    assert (store[naqfc.VARIABLE].isel(reference_time=1).values == 2.0).all()


@pytest.mark.network
def test_out_of_order_cycles_round_trip_through_process_file(
    naqfc_repo: Repository,
) -> None:
    """The same journey through the real path: GRIB from NOAA's bucket, cycles
    handed to `process_file` in the wrong order."""
    processor = Processor()
    session = processor.initialize_session(naqfc_repo)
    first, second, third = naqfc.cycle_urls("2025-01-01", "2025-01-02", ("06", "12"))[
        :3
    ]

    assert processor.process_file(first, session)  # 01 Jan 06z -> create
    assert processor.process_file(third, session)  # 02 Jan 06z -> append, reserves
    assert processor.process_file(second, session)  # 01 Jan 12z -> region

    store = xr.open_zarr(session.store, consolidated=False, zarr_format=3)
    axis = store.reference_time.values

    assert list(axis) == sorted(axis)
    assert [str(t) for t in axis] == [
        "2025-01-01T06:00:00.000000000",
        "2025-01-01T12:00:00.000000000",
        "2025-01-02T06:00:00.000000000",
    ]
    array = f"/{naqfc.VARIABLE}"
    assert all(
        session.chunk_type(array, [row, 0, 0, 0]) == icechunk.ChunkType.virtual
        for row in range(3)
    )
