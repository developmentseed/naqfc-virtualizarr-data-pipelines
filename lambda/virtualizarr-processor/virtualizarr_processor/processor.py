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


def write_plan(cube: "xr.Dataset", existing: "np.ndarray | None") -> WritePlan:
    """Choose between creating, region-writing, and appending one cycle.

    The deciding question is whether the store's `reference_time` axis already
    carries this cycle:

    * **region** -- it does, so the row exists and must be written in place.
      An append would add a second row with the same coordinate value, leaving
      the axis non-monotonic and the cycle stored twice. This is the normal case
      after a backfill: the store is declared at its full extent up front, so
      every cycle inside that extent already has a (possibly empty) row waiting,
      and it is also how a re-delivered notification lands harmlessly.
    * **append** -- it does not, which is the forward case: a cycle past the
      declared axis extends it by one row.
    * **create** -- there is no array at all yet, the first file of a
      forward-only deployment, where the write has to create the store.

    A region write goes through `naqfc.region_cube`, which drops the grid
    coordinates: they carry no `reference_time` dimension, so `region="auto"`
    has no slice to resolve for them. An append keeps them, where having no
    `reference_time` dimension means the already-written copies are left alone.
    """
    if existing is None:
        return WritePlan("create", cube, {})
    reference_time = cube["reference_time"].values[0]
    if bool((existing == reference_time).any()):
        return WritePlan("region", naqfc.region_cube(cube), {"region": "auto"})
    return WritePlan("append", cube, {"append_dim": "reference_time"})


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
        """Write one cycle into `main`, region-writing or appending as needed.

        Which of the two depends on the file: a cycle whose `reference_time` is
        already on the store's axis is written in place, one that is not extends
        the axis by a row. `write_plan` holds the reasoning.
        """
        try:
            cube = naqfc.cycle_cube(file_key)
            plan = write_plan(cube, store_reference_times(session.store))
            logger.info("%s: %s write to main", file_key, plan.mode)
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
