"""Processor-level tests for the NAQFC implementation.

Everything that can be checked without reading GRIB is checked offline: the
protocol conformance, the cycle enumeration that the store axis and the
inventory both derive from, and the cube reshaping rules. The round trips that
genuinely need bytes from NOAA's bucket are marked `network`.
"""

import pathlib
from datetime import datetime, timedelta, timezone

import icechunk
import numpy as np
import pytest
from icechunk import Repository
from virtualizarr_processor import naqfc
from virtualizarr_processor.processor import Processor
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
