"""NAQFC AQMv7 dataset conventions: file layout, cycle axis, and cube shaping.

This module is the single source of truth for the ``(reference_time, lead)``
axis. It matters that it is: ``initialize_backfill_store`` never sees the
inventory -- it is handed only a Repository -- so it must reconstruct the
``reference_time`` axis independently. If the axis it builds and the inventory
the workers consume were derived separately, a drift between them would not
raise; ``region="auto"`` would resolve each cycle against whatever axis the
store happens to have and land the write in the wrong row.

So both sides go through here: ``initialize_backfill_store`` calls
``cycle_reference_times()`` and ``scripts/generate_inventory.py`` calls
``cycle_urls()``, and the two are the same enumeration in the same order.

Layout on the public bucket::

    s3://noaa-nws-naqfc-pds/AQMv7/CS/<YYYYMMDD>/<HH>/
        aqm.t<HH>z.<product>.<YYYYMMDD>.227.grib2

Heavy imports (virtualizarr, gribberish, xarray) are deferred into the functions
that need them so the inventory script can import the path/axis helpers without
pulling the parser stack.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING, Any, Iterator, cast

if TYPE_CHECKING:
    import numpy as np
    import xarray as xr

# --- dataset configuration -------------------------------------------------
# The 2025 AQMv7 CONUS hourly-ozone archive. This repo is a per-dataset fork of
# the pipeline template, so the dataset it serves is a property of the code
# rather than of the deployment.

BUCKET = "noaa-nws-naqfc-pds"
SOURCE_REGION = "us-east-1"
COLLECTION = "AQMv7"
DOMAIN = "CS"  # CONUS; AK and HI are separate grids
GRID = "227"  # NCEP grid ID: 5 km Lambert conformal over CONUS
PRODUCT = "ave_1hr_o3"
CYCLES = ("06", "12")

# The backfill extent. `initialize_backfill_store` builds the store's
# reference_time axis from these and the inventory script defaults to them, so
# changing them here moves both together -- which is the point. Nothing
# downstream re-checks that the two agree.
START = "2025-01-01"
END = "2025-12-31"

# Forecast length. AQMv7 runs this product to +72 h uniformly; a cycle that
# disagrees is rejected rather than padded (see `cycle_cube`). Changing PRODUCT
# to ave_8hr_o3 means changing this to 65 as well.
LEAD_HOURS = 72

# gribberish decodes the ozone products via NCEP local parameter (0, 14, 193).
# The pm25 and max_* products raise inside the parser -- see the support table
# at the end of the source notebook -- so they are rejected up front with a
# useful message instead of an opaque parser failure inside a worker.
VARIABLE = "ozcon"
SUPPORTED_PRODUCTS = frozenset(
    {"ave_1hr_o3", "ave_1hr_o3_bc", "ave_8hr_o3", "ave_8hr_o3_bc"}
)

DIMS = ("reference_time", "lead", "y", "x")

# Coordinates that describe the grid rather than a cycle. They are written once
# by the initializer and MUST be dropped before a region write: every worker's
# fork would otherwise write byte-identical chunks to the same coordinate
# array, and Icechunk treats concurrent writes to one chunk as a merge
# conflict regardless of whether the bytes agree.
STATIC_COORDS = ("y", "x", "latitude", "longitude", "spatial_ref", "valid_time")


class UnsupportedProductError(ValueError):
    """Raised for a product gribberish cannot decode."""


def check_product(product: str | None = None) -> None:
    """Fail fast on a product the GRIB parser cannot read."""
    product = PRODUCT if product is None else product
    if product not in SUPPORTED_PRODUCTS:
        raise UnsupportedProductError(
            f"{product!r} is not decodable by gribberish; supported: "
            f"{sorted(SUPPORTED_PRODUCTS)}. The pm25 and max_* products need "
            f"kerchunk's scan_grib instead."
        )


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


def object_key_wildcard(
    collection: str | None = None,
    domain: str | None = None,
    product: str | None = None,
    grid: str | None = None,
) -> str:
    """An S3-key wildcard matching exactly the files this pipeline can ingest.

    Used as the SNS subscription filter so the forward queue receives only the
    one product the store holds. The topic carries every NWS air quality
    product -- all model versions, all domains, PM2.5 and smoke as well as
    ozone -- so without a filter the consumer wakes for files it must discard.

    The dots around the product are load-bearing: `ave_1hr_o3` is a prefix of
    `ave_1hr_o3_bc`, so a bare `*ave_1hr_o3*` would also admit the
    bias-corrected product, which is a different variable and does not belong
    in this array.
    """
    collection = COLLECTION if collection is None else collection
    domain = DOMAIN if domain is None else domain
    product = PRODUCT if product is None else product
    grid = GRID if grid is None else grid
    # Two wildcards, within SNS's limit of three per pattern. The first spans
    # <date>/<cycle>/aqm.t<HH>z, the second the date repeated in the filename.
    return f"{collection}/{domain}/*.{product}.*.{grid}.grib2"


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


def valid_time_grid(reference_times: "np.ndarray", lead: "np.ndarray") -> "np.ndarray":
    """The 2-D `valid_time` coordinate, derived rather than read.

    valid = reference_time + lead hours, which holds for every cycle, so the
    initializer can write this in full without touching a single GRIB file.
    """
    grid = (
        reference_times[:, None]
        + lead.astype("timedelta64[h]").astype("timedelta64[ns]")[None, :]
    )
    # cast: pre-commit runs mypy without numpy, so the arithmetic is Any there
    # and warn_return_any flags a bare return.
    return cast("np.ndarray", grid)


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
    """Pin absolute units on the time coordinates.

    Without this, a store whose first-written `reference_time` holds a single
    value gets `units="days since <that cycle>"` inferred, and every later
    value is written under those units and reads back wildly wrong.
    """
    for name in ("reference_time", "valid_time"):
        if name in cube.coords:
            cube[name].encoding.update(units="hours since 1970-01-01", dtype="int64")
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
        .assign_coords(valid_time=(("reference_time", "lead"), valid[None, :]))
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
            "valid_time": (
                ("reference_time", "lead"),
                valid_time_grid(reference_times, lead),
            ),
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
