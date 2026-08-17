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
from typing import Any, cast

import icechunk
from icechunk import ForkSession, Repository, Session

from virtualizarr_processor import naqfc

logger = logging.getLogger(__name__)

BACKFILL_BRANCH = "backfill"


def source_prefix() -> str:
    """URL prefix of the bucket holding the virtual chunks. Resolved per call so
    it tracks `naqfc.BUCKET` rather than freezing it at import."""
    return f"s3://{naqfc.BUCKET}/"


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

    def _has_data(self, session: Session) -> bool:
        """Whether the store already holds the data array."""
        import zarr

        try:
            return naqfc.VARIABLE in zarr.open_group(session.store, mode="r")
        except Exception:
            # No group at all yet: a forward-only deployment before its first file.
            return False

    def process_file(self, file_key: str, session: Session) -> bool:
        """Add one cycle to `main` as a `reference_time` row.

        Appends, except for the very first cycle of a forward-only deployment,
        where there is no array to append to yet and the write has to create it.
        After a backfill the store already exists, so this always appends.

        Unlike the backfill path this keeps the grid coordinates on the cube:
        they carry no `reference_time` dimension, so an append leaves the
        already-written copies alone rather than duplicating them.
        """
        try:
            cube = naqfc.cycle_cube(file_key)
            append = {"append_dim": "reference_time"} if self._has_data(session) else {}
            cube.vz.to_icechunk(session.store, **append)  # type: ignore[arg-type]
            return True
        except Exception:
            logger.exception("process_file failed for %s", file_key)
            return False

    def commit_processed_files(self, session: Session) -> str:
        return str(session.commit(message=f"Append to {session.snapshot_id}"))

    # --- maintenance --------------------------------------------------------

    def garbage_collect(self, expiry_time: datetime) -> icechunk.GCSummary:
        repo = self._open()
        repo.expire_snapshots(older_than=expiry_time)
        return repo.garbage_collect(delete_object_older_than=expiry_time)
