"""NAQFC AQMv7 dataset conventions: file layout, cycle axis, and cube shaping.

This module is the single source of truth for the ``(reference_time, lead)``
axis. It matters that it is: ``initialize_backfill_store`` never sees the
inventory -- it is handed only a Repository -- so it reconstructs the
``reference_time`` axis from configuration while the workers consume a
separately generated inventory. The two have to describe the same set of cycles.

``region="auto"`` aligns by coordinate *value*, so the two failure modes are
asymmetric:

* a cycle in the inventory but **not** in the axis fails loudly, with
  ``KeyError: Not all values of coordinate 'reference_time' ... were found``.
  Nothing lands in the wrong row.
* a cycle in the axis but **not** in the inventory is silent -- that row simply
  stays empty, and only reading it back reveals the hole.

So both sides go through here: ``initialize_backfill_store`` calls
``cycle_reference_times()`` and ``scripts/generate_inventory.py`` calls
``cycle_urls()``, which keeps the extent and the file list in step.

Layout on the public bucket::

    s3://noaa-nws-naqfc-pds/AQMv7/CS/<YYYYMMDD>/<HH>/
        aqm.t<HH>z.<product>.<YYYYMMDD>.227.grib2

Heavy imports (virtualizarr, gribberish, xarray) are deferred into the functions
that need them so the inventory script can import the path/axis helpers without
pulling the parser stack.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    import numpy as np
    import xarray as xr

# --- dataset configuration -------------------------------------------------
# One deployment serves one (domain, product) pair. Defaults describe the 2025
# AQMv7 CONUS hourly-ozone archive; every value is overridable by environment
# variable so AK/HI and other products can be deployed as separate stacks
# against separate stores. The CDK forwards these into every Lambda.
#
# The SNS subscription filter is configured separately, as NAQFC_KEY_PATTERN in
# StackSettings, and is NOT derived from these values -- keep the two in step by
# hand when retargeting a deployment.
#
# Read at import: set them in the Lambda environment (via .env -> StackSettings),
# not at runtime.

# NCEP grid IDs are a property of the domain, not an independent choice. Setting
# one without the other yields URLs that resolve to nothing, so the grid is
# derived unless explicitly overridden.
GRID_BY_DOMAIN = {"CS": "227", "AK": "198", "HI": "196"}


def grid_for_domain(domain: str, explicit: str | None = None) -> str:
    """The NCEP grid ID that appears in a domain's filenames."""
    if explicit:
        return explicit
    return GRID_BY_DOMAIN.get(domain, "")


BUCKET = os.environ.get("NAQFC_DATA_BUCKET", "noaa-nws-naqfc-pds")
SOURCE_REGION = os.environ.get("NAQFC_SOURCE_REGION", "us-east-1")
COLLECTION = os.environ.get("NAQFC_COLLECTION", "AQMv7")
DOMAIN = os.environ.get("NAQFC_DOMAIN", "CS")
GRID = grid_for_domain(DOMAIN, os.environ.get("NAQFC_GRID"))
PRODUCT = os.environ.get("NAQFC_PRODUCT", "ave_1hr_o3")
CYCLES = tuple(os.environ.get("NAQFC_CYCLES", "06,12").split(","))

# The backfill extent. `initialize_backfill_store` builds the store's
# reference_time axis from these and the inventory script defaults to them, so
# changing them moves both together -- which is the point. Nothing downstream
# re-checks that the two agree.
START = os.environ.get("NAQFC_START", "2025-01-01")
END = os.environ.get("NAQFC_END", "2025-12-31")

# Forecast length, which varies by product: ave_1hr_o3 runs to +72 h, ave_8hr_o3
# to +65. A cycle that disagrees is rejected rather than padded (see
# `cycle_cube`), so this must match the configured product.
LEAD_HOURS = int(os.environ.get("NAQFC_LEAD_HOURS", "72"))

# The variable name the GRIB parser assigns to the decoded field. Ozone products
# decode to `ozcon`, pm25 to `pmtf`, so this moves with the product.
VARIABLE = os.environ.get("NAQFC_VARIABLE", "ozcon")

DIMS = ("reference_time", "lead", "y", "x")

# Coordinates that describe the grid rather than a cycle. They are written once
# by the initializer and MUST be dropped before a region write: every worker's
# fork would otherwise write byte-identical chunks to the same coordinate
# array, and Icechunk treats concurrent writes to one chunk as a merge
# conflict regardless of whether the bytes agree.
STATIC_COORDS = ("y", "x", "latitude", "longitude", "spatial_ref")


# --- cycle enumeration -----------------------------------------------------


def _daterange(start: date, end: date) -> Iterator[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def days(start: str | None = None, end: str | None = None) -> list[date]:
    """Each calendar day in the range, inclusive."""
    start = START if start is None else start
    end = END if end is None else end
    return list(_daterange(date.fromisoformat(start), date.fromisoformat(end)))


# Defaults below resolve from the module constants at call time rather than
# being bound into the signature, where Python would capture their values once
# at import and ignore any later change to the constant.
def cycles(
    start: str | None = None,
    end: str | None = None,
    cycle_hours: tuple[str, ...] | None = None,
) -> list[tuple[date, str]]:
    """Every (day, cycle-hour) pair in the range, chronologically ordered.

    This ordering defines the store's `reference_time` axis and the inventory
    order; the backfill partitioner slices the inventory into contiguous chunks,
    so consecutive cycles land in adjacent rows.
    """
    ordered = sorted(CYCLES if cycle_hours is None else cycle_hours)
    return [(day, cycle) for day in days(start, end) for cycle in ordered]


def cycle_url(
    day: date,
    cycle: str,
    bucket: str | None = None,
    collection: str | None = None,
    domain: str | None = None,
    product: str | None = None,
    grid: str | None = None,
) -> str:
    """The `s3://` URL of one cycle's product file."""
    bucket = BUCKET if bucket is None else bucket
    collection = COLLECTION if collection is None else collection
    domain = DOMAIN if domain is None else domain
    product = PRODUCT if product is None else product
    grid = GRID if grid is None else grid
    return (
        f"s3://{bucket}/{collection}/{domain}/{day:%Y%m%d}/{cycle}/"
        f"aqm.t{cycle}z.{product}.{day:%Y%m%d}.{grid}.grib2"
    )


def cycle_urls(
    start: str | None = None,
    end: str | None = None,
    cycle_hours: tuple[str, ...] | None = None,
    **kwargs: str | None,
) -> list[str]:
    """Every cycle URL in the range, in `reference_time` order."""
    return [
        cycle_url(day, cycle, **kwargs)
        for day, cycle in cycles(start, end, cycle_hours)
    ]


def cycle_reference_times(
    start: str | None = None,
    end: str | None = None,
    cycle_hours: tuple[str, ...] | None = None,
) -> "np.ndarray":
    """The store's `reference_time` axis, aligned index-for-index with
    `cycle_urls()` over the same arguments."""
    import numpy as np

    return np.array(
        [
            np.datetime64(f"{day:%Y-%m-%d}T{cycle}:00:00", "ns")
            for day, cycle in cycles(start, end, cycle_hours)
        ],
        dtype="datetime64[ns]",
    )


def lead_axis(lead_hours: int | None = None) -> "np.ndarray":
    """The `lead` axis: +1 h .. +lead_hours."""
    import numpy as np

    return np.arange(
        1, (LEAD_HOURS if lead_hours is None else lead_hours) + 1, dtype="int32"
    )


# NB: the store deliberately carries no `valid_time` coordinate. It is exactly
# `reference_time + lead`, so writing it would store 2-D redundant data that has
# to be kept consistent on every append. Consumers derive it in one line:
#
#     valid_time = cube.reference_time + cube.lead.astype("timedelta64[h]")


# --- virtual dataset construction ------------------------------------------


def registry() -> Any:
    """An object-store registry for anonymous reads of the public bucket."""
    from obspec_utils.registry import ObjectStoreRegistry
    from obstore.store import S3Store

    store = S3Store(BUCKET, region=SOURCE_REGION, skip_signature=True)
    return ObjectStoreRegistry({f"s3://{BUCKET}": store})


def parser() -> Any:
    """The GRIB parser. `collapse_groups=True` folds the single-variable file
    into one root dataset."""
    from gribberish.virtualizarr import GribberishParser  # type: ignore[import-untyped]

    return GribberishParser(collapse_groups=True)


def open_cycle(url: str, reg: Any = None, prs: Any = None) -> "xr.Dataset":
    """Open one cycle file virtually, as published: dims (time, y, x).

    Reads only GRIB metadata sections -- no data bytes. NAQFC publishes no
    `.idx` sidecars, so this still transfers the whole object (~86 MB for
    hourly ozone) to scan message headers.
    """
    from virtualizarr import open_virtual_dataset

    return open_virtual_dataset(url, registry=reg or registry(), parser=prs or parser())


def pin_time_encoding(cube: "xr.Dataset") -> "xr.Dataset":
    """Pin absolute units on `reference_time`.

    Without this, a store whose first-written `reference_time` holds a single
    value gets `units="days since <that cycle>"` inferred, and every later
    value is written under those units and reads back wildly wrong.
    """
    if "reference_time" in cube.coords:
        cube["reference_time"].encoding.update(
            units="hours since 1970-01-01", dtype="int64"
        )
    return cube


def cycle_cube(
    url: str,
    reg: Any = None,
    prs: Any = None,
    lead_hours: int | None = None,
) -> "xr.Dataset":
    """One cycle as a (reference_time, lead, y, x) virtual dataset.

    The published `time` axis is *valid* time, and successive cycles re-forecast
    most of the same hours with different values. Splitting it into
    (reference_time, lead) makes every element unique by construction, so the
    06z and 12z runs of a day occupy disjoint regions instead of colliding.
    """
    import numpy as np

    lead_hours = LEAD_HOURS if lead_hours is None else lead_hours
    vds = open_cycle(url, reg, prs)

    ref = np.datetime64(vds[VARIABLE].attrs["reference_date"][:19], "ns")
    valid = vds["time"].values
    lead = ((valid - ref) / np.timedelta64(1, "h")).astype("int32")

    if len(lead) != lead_hours:
        # A truncated run would shift every subsequent lead index, so refuse it
        # rather than write a misaligned region. Naming the URL matters: the
        # worker otherwise reports only a generic failure for a batch of files.
        raise ValueError(
            f"{url} has {len(lead)} lead steps, expected {lead_hours}; "
            f"refusing to write a misaligned region"
        )

    cube = (
        vds.rename({"time": "lead"})
        .assign_coords(lead=("lead", lead))
        .expand_dims(reference_time=[ref])
    )
    # Per-cycle provenance; leaving it on would label the whole store with
    # whichever cycle happened to be written first.
    cube[VARIABLE].attrs.pop("reference_date", None)
    return pin_time_encoding(cube)


def region_cube(cube: "xr.Dataset") -> "xr.Dataset":
    """Reduce a cycle cube to what a region write may touch.

    Keeps the data variable and the two index coordinates `region="auto"` needs
    to resolve the target slice; drops the grid coordinates, which are written
    once by the initializer and would otherwise collide between forks.
    """
    return cube.drop_vars([c for c in STATIC_COORDS if c in cube.variables])


def full_shape_skeleton(
    reference_times: "np.ndarray",
    lead: "np.ndarray",
    reference_cube: "xr.Dataset",
) -> "xr.Dataset":
    """The whole store at final shape, with no chunk references.

    Array metadata (dtype, fill value, and critically the `gribberish` codec) is
    lifted from a real cycle rather than reconstructed, so the skeleton cannot
    drift from what the workers actually write into it. The manifest is empty:
    this declares extent only, and every chunk arrives later via a region write.
    """
    import numpy as np
    import xarray as xr
    from virtualizarr.manifests import ChunkManifest, ManifestArray
    from zarr.core.metadata import ArrayV3Metadata

    source = reference_cube[VARIABLE].data
    ny, nx = source.metadata.shape[-2:]
    shape = (len(reference_times), len(lead), ny, nx)
    chunks = (1, 1, ny, nx)  # one GRIB message per chunk

    spec = source.metadata.to_dict()
    spec["shape"] = list(shape)
    spec["chunk_grid"] = {
        "name": "regular",
        "configuration": {"chunk_shape": list(chunks)},
    }
    spec["dimension_names"] = list(DIMS)

    empty = ManifestArray(
        chunkmanifest=ChunkManifest({}, shape=(len(reference_times), len(lead), 1, 1)),
        metadata=ArrayV3Metadata.from_dict(spec),
    )

    skeleton = xr.Dataset(
        {VARIABLE: xr.Variable(DIMS, empty, attrs=reference_cube[VARIABLE].attrs)},
        coords={
            "reference_time": ("reference_time", reference_times),
            "lead": ("lead", np.asarray(lead)),
            # Grid coordinates carried straight over: y/x are real arrays,
            # latitude/longitude/spatial_ref stay virtual references into the
            # cycle they came from.
            **{
                name: reference_cube[name]
                for name in ("y", "x", "latitude", "longitude", "spatial_ref")
                if name in reference_cube.coords
            },
        },
        attrs=reference_cube.attrs,
    )
    return pin_time_encoding(skeleton)
