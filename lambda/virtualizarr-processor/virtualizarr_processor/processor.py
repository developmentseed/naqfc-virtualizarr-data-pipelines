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
which lands first, so a cycle is appended wherever it turns up and the axis is
put back into chronological order by `sort_reference_time` before the batch
commits. Nothing is committed out of order: the sort runs inside the same
session as the appends, so the intermediate state is never visible.

Dataset layout, extent, and the cycle enumeration live in `naqfc`.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, NamedTuple, cast

import icechunk
import numpy as np
import xarray as xr
import zarr
from icechunk import ForkSession, Repository, Session

from virtualizarr_processor import naqfc

logger = logging.getLogger(__name__)

BACKFILL_BRANCH = "backfill"


def source_prefix() -> str:
    """URL prefix of the bucket holding the virtual chunks. Resolved per call so
    it tracks `naqfc.BUCKET` rather than freezing it at import."""
    return f"s3://{naqfc.BUCKET}/"


class WritePlan(NamedTuple):
    """How one cycle gets written: `plan.cube.vz.to_icechunk(store, **plan.kwargs)`."""

    mode: str  # "create" | "region" | "append"
    cube: xr.Dataset
    kwargs: dict[str, Any]


def store_reference_times(store: Any) -> np.ndarray | None:
    """The `reference_time` axis a store already holds, or None if it holds no
    data yet.

    Re-read for every file rather than cached: an append earlier in the same
    batch adds a row that the next file has to see. A session reads its own
    uncommitted writes, so this stays correct mid-batch.
    """
    try:
        cube = xr.open_zarr(store, consolidated=False, zarr_format=3)
    except Exception:
        # No group at all yet: a forward-only deployment before its first file.
        return None
    if naqfc.VARIABLE not in cube.variables or "reference_time" not in cube.coords:
        return None
    return cast(np.ndarray, cube["reference_time"].values)


def write_plan(cube: xr.Dataset, existing: np.ndarray | None) -> WritePlan:
    """Choose between creating, region-writing, and appending one cycle.

    The deciding question is whether the store's `reference_time` axis already
    carries this cycle:

    * **region** -- it does, so the row exists and is written in place. An
      append would add a second row with the same coordinate value, and a
      duplicated cycle is something no later sort can repair. This is the normal
      case after a backfill, where the store is declared at its full extent up
      front so every cycle inside that extent already has a row waiting, and it
      is also how a re-delivered notification lands harmlessly.
    * **append** -- it does not, so the axis grows by a row. Where that row
      lands chronologically does not matter: a cycle older than the store's last
      row, or one that fills a gap in the middle, is appended at the end just
      the same, and `sort_reference_time` moves it into place before the batch
      commits. Zarr can only grow an axis at its end, and this is what lets an
      out-of-order arrival be accepted anyway rather than refused.
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


def sort_reference_time(session: Session) -> int:
    """Put the `reference_time` axis back in chronological order, in place.

    Cycles are delivered in whatever order SNS happens to fan them out, so
    `write_plan` appends a late arrival at the end of the axis rather than
    refusing it, and this straightens the axis out afterwards. Run inside the
    same session as those appends -- see `Processor.commit_processed_files` --
    so the out-of-order state is never committed and no reader ever sees it.

    The work is done on the chunk grid, not on the data: `reindex_array` moves
    each virtual chunk reference to the row its cycle belongs in, and the
    coordinate array is permuted to match. Nothing is materialized and no chunk
    bytes are fetched, so the cost is proportional to the number of chunk
    references rather than to the size of the store.

    Returns the number of rows that moved, which is 0 when the axis was already
    ordered -- the overwhelmingly common case, and worth short-circuiting:
    `reindex_array` visits every chunk even when the permutation is the
    identity, and dirties the session doing it.
    """
    try:
        cube = xr.open_zarr(session.store, consolidated=False, zarr_format=3)
    except Exception:
        return 0  # nothing written yet; there is no axis to sort
    if "reference_time" not in cube.coords:
        return 0

    times = cube["reference_time"].values
    order = np.argsort(times, kind="stable")

    if np.unique(times).size != len(times):
        # `write_plan` region-writes a cycle already on the axis rather than
        # appending it, so a duplicate means something bypassed that. Sorting
        # cannot make the axis strictly increasing, and guessing which copy to
        # keep would silently drop data.
        raise ValueError(
            "reference_time contains duplicate cycles; sorting cannot make it "
            "strictly increasing"
        )
    if np.array_equal(order, np.arange(len(times))):
        return 0

    # Maps an old row index to the row it belongs in once sorted; `order` is its
    # inverse, which is what `reindex_array` needs to clear vacated positions.
    old_to_new = np.empty_like(order)
    old_to_new[order] = np.arange(len(order))

    # Everything is checked before anything is moved: a half-permuted array is
    # far worse than a refused one.
    temporal: list[tuple[str, int]] = []
    for name, variable in cube.variables.items():
        if name == "reference_time" or "reference_time" not in variable.dims:
            continue
        axis = variable.dims.index("reference_time")
        array = zarr.open_array(session.store, path=str(name), mode="r", zarr_format=3)
        if array.chunks[axis] != 1:
            # One row per chunk is what makes a row a movable unit. A coarser
            # chunking would need chunks split before they could be reordered.
            raise ValueError(
                f"{name!r} has reference_time chunk size {array.chunks[axis]}; "
                f"expected 1, so its rows cannot be moved independently"
            )
        temporal.append((str(name), axis))

    for name, axis in temporal:

        def forward(position: Any, *, axis: int = axis) -> Any:
            target = list(position)
            target[axis] = int(old_to_new[position[axis]])
            return target

        def backward(position: Any, *, axis: int = axis) -> Any:
            source = list(position)
            source[axis] = int(order[position[axis]])
            return source

        # `backward` is not optional: `reindex_array` only visits chunks that
        # exist, so without it a row vacated by the permutation keeps whatever
        # was there before.
        session.reindex_array(f"/{name}", forward, backward)

    # The raw encoded values, permuted directly. Going through xarray here would
    # re-infer the time encoding rather than overwrite the existing coordinate,
    # and a block written under the wrong units reads back wrong by a factor of
    # 24.
    coordinate = zarr.open_array(
        session.store, path="reference_time", mode="r+", zarr_format=3
    )
    coordinate[:] = np.asarray(coordinate[:])[order]

    return int((order != np.arange(len(order))).sum())


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
        """Straighten the axis, then commit the batch as one snapshot.

        The sort belongs here rather than in the handler because it has to
        happen on every commit without exception: an appended-but-unsorted axis
        that reached a snapshot would be exactly the non-monotonic store this
        is meant to prevent, and there is no second chance to fix it once
        readers have the snapshot.
        """
        moved = sort_reference_time(session)
        if moved:
            logger.info("Sorted %d reference_time row(s) into place", moved)
        return str(session.commit(message=f"Update {session.snapshot_id}"))

    # --- maintenance --------------------------------------------------------

    def garbage_collect(self, expiry_time: datetime) -> icechunk.GCSummary:
        repo = self._open()
        repo.expire_snapshots(older_than=expiry_time)
        return repo.garbage_collect(delete_object_older_than=expiry_time)
