"""VirtualizarrProcessor for NOAA NAQFC AQMv7 surface-ozone GRIB2.

Builds an Icechunk store dimensioned ``(reference_time, lead, y, x)`` from the
public ``noaa-nws-naqfc-pds`` bucket. Nothing is copied: each GRIB message
becomes one virtual chunk, range-requested from NOAA's bucket on read and
decoded by the ``gribberish`` zarr codec.

The two-dimensional time layout is what makes concurrent writes safe. NAQFC
cycles overlap heavily in valid time -- the 06z and 12z runs of a day share 66
of their 72 hours -- so a flat time axis would have consecutive cycles writing
the same chunks with different values. Keyed by ``(reference_time, lead)``
instead, every cycle owns a disjoint region and the fork/merge backfill has no
conflicts to resolve.

Forward processing does not assume cycles arrive in order. SNS fans out the 06z
and 12z notifications within minutes of each other and promises nothing about
which lands first, so appending whatever turns up leaves ``reference_time``
non-monotonic. Each arrival is instead placed against the cycle schedule: one
past the end of the store extends the axis over every cycle still in flight,
reserving their rows, and a straggler is written into the row already waiting
for it. `write_plan` is where that decision is made.

Dataset layout, extent, and the cycle enumeration live in `naqfc`.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import icechunk
from icechunk import ForkSession, Repository, Session

from virtualizarr_processor import naqfc

if TYPE_CHECKING:
    import numpy as np
    import xarray as xr

logger = logging.getLogger(__name__)

BACKFILL_BRANCH = "backfill"


def source_prefix() -> str:
    """URL prefix of the bucket holding the virtual chunks. Resolved per call so
    it tracks `naqfc.BUCKET` rather than freezing it at import."""
    return f"s3://{naqfc.BUCKET}/"


class WritePlan(NamedTuple):
    """How one cycle gets written: `plan.cube.vz.to_icechunk(store, **plan.kwargs)`."""

    mode: str  # "create" | "region" | "append"
    cube: "xr.Dataset"
    kwargs: dict[str, Any]


def store_reference_times(store: Any) -> "np.ndarray | None":
    """The `reference_time` axis a store already holds, or None if it holds no
    data yet.

    Re-read for every file rather than cached: an append earlier in the same
    batch adds a row that the next file has to see. A session reads its own
    uncommitted writes, so this stays correct mid-batch.
    """
    import xarray as xr

    try:
        cube = xr.open_zarr(store, consolidated=False, zarr_format=3)
    except Exception:
        # No group at all yet: a forward-only deployment before its first file.
        return None
    if naqfc.VARIABLE not in cube.variables or "reference_time" not in cube.coords:
        return None
    return cast("np.ndarray", cube["reference_time"].values)


def _reserve(cube: "xr.Dataset", schedule: "np.ndarray") -> "xr.Dataset":
    """Place one cycle on `schedule`, reserving a row per cycle not yet here.

    Rows the cycle does not cover get no chunk manifest entries at all, so they
    cost nothing to store and read back as the array's fill value. They exist to
    hold an address: a straggler that lands later is a region write into the row
    already waiting for it rather than an append out of order.

    Deliberately *not* `cube.reindex(reference_time=schedule)`, which is the
    obvious spelling and the one the prototype used. xarray builds its reindex
    indexer over the array's element shape, and at NAQFC's grid that is
    72 x 1473 x 1025 per row -- ~1 GB of transient allocation to reserve a
    single row, which kills a 2 GB Lambda before it writes anything. The work
    belongs on the chunk grid instead: one entry per GRIB message, 72 per cycle.
    """
    import numpy as np
    import xarray as xr
    from virtualizarr.manifests import ChunkManifest, ManifestArray
    from zarr.core.metadata import ArrayV3Metadata

    variable = cube[naqfc.VARIABLE]
    array = variable.data
    row = int(np.flatnonzero(schedule == cube["reference_time"].values[0])[0])

    spec = array.metadata.to_dict()
    spec["shape"] = [len(schedule), *array.shape[1:]]
    padded = ManifestArray(
        chunkmanifest=ChunkManifest(
            {
                ".".join([str(row), *key.split(".")[1:]]): entry
                for key, entry in array.manifest.dict().items()
            },
            shape=(len(schedule), *array.manifest.shape_chunk_grid[1:]),
        ),
        metadata=ArrayV3Metadata.from_dict(spec),
    )

    # Every other coordinate spans lead/y/x only, so it carries over untouched --
    # which is also why none of them needs a row reserving.
    coords: dict[Any, Any] = {
        name: cube[name]
        for name in cube.coords
        if "reference_time" not in cube[name].dims
    }
    coords["reference_time"] = ("reference_time", schedule)
    return naqfc.pin_time_encoding(
        xr.Dataset(
            {naqfc.VARIABLE: xr.Variable(naqfc.DIMS, padded, attrs=variable.attrs)},
            coords=coords,
            attrs=cube.attrs,
        )
    )


def write_plan(cube: "xr.Dataset", existing: "np.ndarray | None") -> WritePlan:
    """Choose how to write one cycle, given the `reference_time` axis it meets.

    Files arrive out of order, so appending whatever turns up leaves the axis
    non-monotonic. Instead the axis is treated as the *schedule*: where a cycle
    lands relative to it decides the write.

    * **region** -- the cycle is already on the axis. Its row exists, either
      reserved by an earlier arrival that skipped over it or already filled by
      this same cycle, and is written in place. A re-delivered notification
      therefore rewrites its own row rather than adding a duplicate; the
      references are identical, so it is idempotent.
    * **append** -- the cycle is past the last row, so the axis is extended to
      reach it. Every scheduled cycle in between gets a reserved row, which is
      what lets a straggler land in `region` mode later instead of being
      appended out of order.
    * **create** -- there is no array yet, the first file of a forward-only
      deployment.

    Two cases have no write at all, because Zarr can only grow an axis at its
    end: a cycle older than the store's first row, and one that falls inside the
    axis on no row (a gap left by the append-anything behaviour that predates
    schedule alignment). Both raise rather than write somewhere wrong.

    A region write goes through `naqfc.region_cube`, which drops the grid
    coordinates: they carry no `reference_time` dimension, so `region="auto"`
    has no slice to resolve for them. An append keeps them, where having no
    `reference_time` dimension means the already-written copies are left alone.
    """
    reference_time = cube["reference_time"].values[0]
    naqfc.check_scheduled(reference_time)

    if existing is None:
        return WritePlan("create", cube, {})

    if bool((existing == reference_time).any()):
        return WritePlan("region", naqfc.region_cube(cube), {"region": "auto"})

    # max(), not the last element: a store written before schedule alignment may
    # already be non-monotonic, and extending from anything but its high-water
    # mark would write reference_time values it already holds.
    last, first = existing.max(), existing.min()

    if reference_time > last:
        schedule = naqfc.schedule_between(last, reference_time)
        if len(schedule) > naqfc.MAX_RESERVED_CYCLES:
            raise ValueError(
                f"cycle {reference_time} is {len(schedule)} cycles past the "
                f"store's last row ({last}), over the {naqfc.MAX_RESERVED_CYCLES}"
                " cycle limit; reaching it would reserve that many empty rows. "
                "Raise NAQFC_MAX_RESERVED_CYCLES if the store really is that far "
                "behind, or check the file's reference_date."
            )
        return WritePlan(
            "append", _reserve(cube, schedule), {"append_dim": "reference_time"}
        )

    if reference_time < first:
        raise ValueError(
            f"cycle {reference_time} is older than the store, which starts at "
            f"{first}; Zarr cannot prepend, so extending the archive backwards "
            f"is a rebuild rather than an ingest"
        )

    raise ValueError(
        f"cycle {reference_time} falls inside the store's axis ({first} .. "
        f"{last}) but has no row of its own; Zarr cannot insert one in the "
        f"middle. The axis has a gap that was never reserved, which is what a "
        f"store appended to before schedule alignment looks like -- filling this "
        f"cycle in means rebuilding it."
    )


class Processor:
    # --- repository plumbing ------------------------------------------------

    def _storage(self) -> Any:
        """Icechunk storage: S3 in a deployed Lambda, local filesystem in tests.

        ICECHUNK_BUCKET  - set => S3 (IAM credentials from the environment)
        ICECHUNK_PREFIX  - key prefix (must be non-empty to create a new repo)
        ICECHUNK_REGION  - S3 region
        ICECHUNK_LOCAL_PATH - filesystem repo path when no bucket is set
        """
        bucket = os.environ.get("ICECHUNK_BUCKET")
        if bucket:
            return icechunk.s3_storage(
                bucket=bucket,
                prefix=os.environ.get("ICECHUNK_PREFIX"),
                region=os.environ.get("ICECHUNK_REGION"),
                from_env=True,
            )
        return icechunk.local_filesystem_storage(os.environ["ICECHUNK_LOCAL_PATH"])

    def _open(self) -> Repository:
        """Open (or create) the repo, wired to read virtual chunks from NOAA.

        The chunks live in a bucket we do not own, so Icechunk needs an explicit
        virtual chunk container for it plus credentials -- anonymous here, since
        the NAQFC bucket is public.
        """
        prefix = source_prefix()
        config = icechunk.RepositoryConfig.default()
        config.set_virtual_chunk_container(
            icechunk.VirtualChunkContainer(
                prefix, icechunk.s3_store(region=naqfc.SOURCE_REGION)
            )
        )
        return icechunk.Repository.open_or_create(
            storage=self._storage(),
            config=config,
            authorize_virtual_chunk_access=icechunk.containers_credentials(
                {prefix: icechunk.s3_anonymous_credentials()}
            ),
        )

    def initialize_repo(self) -> Repository:
        return self._open()

    def open_backfill_repo(self) -> Repository:
        return self._open()

    # --- backfill -----------------------------------------------------------

    def initialize_backfill_store(self, repo: Repository) -> str:
        """Declare the store at full extent on a `backfill` branch and commit.

        Reads exactly one cycle, for two things it cannot invent: the grid
        coordinates, and the `ozcon` array metadata carrying the gribberish
        codec. Everything else -- the reference_time axis and the lead axis --
        is computed from configuration, which is why this never needs the
        inventory.

        Commits and leaves nothing pending: workers fork from this snapshot, and
        a fork is only mergeable if its base is a clean committed snapshot.
        """
        reference_times = naqfc.cycle_reference_times()
        lead = naqfc.lead_axis()
        urls = naqfc.cycle_urls()
        logger.info(
            "Initializing backfill store: %d cycles x %d leads, reference cycle %s",
            len(reference_times),
            len(lead),
            urls[0],
        )

        reference_cube = naqfc.cycle_cube(urls[0])
        skeleton = naqfc.full_shape_skeleton(reference_times, lead, reference_cube)

        repo.create_branch(BACKFILL_BRANCH, repo.lookup_branch("main"))
        session = repo.writable_session(BACKFILL_BRANCH)
        skeleton.vz.to_icechunk(session.store)
        return cast(
            str,
            session.commit(
                f"Initialize {naqfc.PRODUCT} backfill shape: "
                f"{len(reference_times)} cycles x {len(lead)} leads"
            ),
        )

    def process_backfill_file(self, file_key: str, fork: ForkSession) -> bool:
        """Write one cycle into its region of the fork. Must not commit."""
        try:
            cube = naqfc.region_cube(naqfc.cycle_cube(file_key))
            cube.vz.to_icechunk(fork.store, region="auto")
            return True
        except Exception:
            # The worker only reports a generic failure, so the real cause has
            # to be logged here or it is lost. Transient object-store throttling
            # under backfill concurrency shows up here too.
            logger.exception("process_backfill_file failed for %s", file_key)
            return False

    # --- forward processing -------------------------------------------------

    def initialize_session(self, repo: Repository) -> Session:
        return repo.writable_session("main")

    def process_file(self, file_key: str, session: Session) -> bool:
        """Write one cycle into `main`, wherever on the axis it belongs.

        Cycles are delivered in whatever order SNS happens to fan them out, so
        this does not simply append: `write_plan` places each one against the
        store's schedule, extending the axis over any cycle still in flight and
        writing a straggler into the row already reserved for it.

        A cycle that cannot be placed at all -- older than the store, or in a
        gap no row was reserved for -- is logged and reported as a failure
        rather than written somewhere wrong.
        """
        try:
            cube = naqfc.cycle_cube(file_key)
            plan = write_plan(cube, store_reference_times(session.store))
            logger.info(
                "%s: %s write to main (%d row(s))",
                file_key,
                plan.mode,
                plan.cube.sizes["reference_time"],
            )
            plan.cube.vz.to_icechunk(session.store, **plan.kwargs)
            return True
        except Exception:
            logger.exception("process_file failed for %s", file_key)
            return False

    def commit_processed_files(self, session: Session) -> str:
        return str(session.commit(message=f"Update {session.snapshot_id}"))

    # --- maintenance --------------------------------------------------------

    def garbage_collect(self, expiry_time: datetime) -> icechunk.GCSummary:
        repo = self._open()
        repo.expire_snapshots(older_than=expiry_time)
        return repo.garbage_collect(delete_object_older_than=expiry_time)
