#!/usr/bin/env python
"""Generate a backfill input inventory for NOAA NAQFC AQMv7 GRIB2 files.

The NAQFC public bucket lays files out as::

    s3://noaa-nws-naqfc-pds/AQMv7/<domain>/<YYYYMMDD>/<HH>/
        aqm.t<HH>z.<product>.<YYYYMMDD>.<grid>.grib2

which is fully determined by (date, cycle, product), so the inventory is
synthesized from the pattern rather than listed from S3. Pass ``--verify`` to
confirm each key actually exists before writing.

The output is a JSON array of ``s3://`` URIs, sorted chronologically by
(date, cycle) -- the backfill partitioner slices the inventory into contiguous
chunks and each chunk becomes one commit against a disjoint region, so the
ordering here determines the region layout.

One inventory file is written per domain x product, matching the deployment
model: one stack per (domain, product) pair, each with its own store. The grid
follows the domain (CS=227 Lambert, AK=198 polar stereographic, HI=196
Mercator), so selecting a domain is enough.

Usage:
    uv run scripts/generate_inventory.py
    uv run scripts/generate_inventory.py --verify \
        --upload s3://my-icechunk-bucket/inventory/
    uv run scripts/generate_inventory.py --domains AK HI \
        --products ave_1hr_o3 ave_1hr_pm25
    uv run scripts/generate_inventory.py --start 2025-06-01 --end 2025-06-30 \
        --domains CS --products ave_1hr_o3
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any

# The URL pattern and cycle enumeration come from the processor package, not
# from a copy here. `initialize_backfill_store` builds the store's
# reference_time axis from the same functions, and a divergence between the two
# would not raise -- it would place region writes in the wrong rows.
from virtualizarr_processor import naqfc

DEFAULT_PRODUCTS = (naqfc.PRODUCT,)
DEFAULT_DOMAINS = (naqfc.DOMAIN,)

# Matches BACKFILL_PARTITION_SIZE's default in cdk/settings.py; used only to
# report how many commits the inventory will produce.
DEFAULT_PARTITION_SIZE = 500


def day_prefix(day: date, bucket: str, collection: str, domain: str) -> str:
    """Return the ``s3://`` prefix holding every cycle for one day."""
    return f"s3://{bucket}/{collection}/{domain}/{day:%Y%m%d}/"


def s3_client() -> Any:
    """Return an unsigned S3 client -- the NAQFC bucket is public."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def list_existing(prefixes: list[str], workers: int) -> set[str]:
    """List every object under the given ``s3://`` prefixes concurrently."""
    client = s3_client()

    def one(prefix: str) -> list[str]:
        bucket, _, key_prefix = prefix[len("s3://") :].partition("/")
        found = []
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
            for obj in page.get("Contents", []):
                found.append(f"s3://{bucket}/{obj['Key']}")
        return found

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return {uri for batch in pool.map(one, prefixes) for uri in batch}


def drop_missing(uris: list[str], existing: set[str]) -> list[str]:
    """Drop URIs with no matching object, reporting what was removed.

    Takes an already-listed set rather than doing the listing: one day prefix
    holds every product for that domain, so the listing is shared across
    products instead of repeated per product.
    """
    present = [uri for uri in uris if uri in existing]
    missing = [uri for uri in uris if uri not in existing]
    if missing:
        print(f"  {len(missing)} missing file(s) dropped:")
        for uri in missing[:20]:
            print(f"    {uri}")
        if len(missing) > 20:
            print(f"    ... and {len(missing) - 20} more")
    else:
        print("  all files present")
    return present


def upload(local: Path, destination: str) -> str:
    """Upload the inventory, returning the resulting object URI.

    Signed, unlike the read path -- the destination is the caller's own bucket.
    """
    import boto3

    bucket, _, key_prefix = destination[len("s3://") :].partition("/")
    key = f"{key_prefix.rstrip('/')}/{local.name}" if key_prefix else local.name
    boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=local.read_bytes())
    return f"s3://{bucket}/{key}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--start", default=naqfc.START, help="YYYY-MM-DD inclusive")
    parser.add_argument("--end", default=naqfc.END, help="YYYY-MM-DD inclusive")
    parser.add_argument("--bucket", default=naqfc.BUCKET)
    parser.add_argument("--collection", default=naqfc.COLLECTION, help="e.g. AQMv7")
    parser.add_argument(
        "--domains",
        nargs="+",
        default=list(DEFAULT_DOMAINS),
        help=f"one or more of {sorted(naqfc.GRID_BY_DOMAIN)}; one inventory is "
        f"written per domain x product",
    )
    parser.add_argument(
        "--grid",
        help="NCEP grid ID. Normally omitted -- it follows the domain "
        f"({', '.join(f'{d}={g}' for d, g in sorted(naqfc.GRID_BY_DOMAIN.items()))}). "
        f"Setting it applies to every --domains value, so only use it with one.",
    )
    parser.add_argument("--cycles", nargs="+", default=list(naqfc.CYCLES))
    parser.add_argument("--products", nargs="+", default=list(DEFAULT_PRODUCTS))
    parser.add_argument("--out-dir", type=Path, default=Path("inventory"))
    parser.add_argument(
        "--verify",
        action="store_true",
        help="list the bucket and drop keys that do not exist (one paginated "
        "call per day). A file missing at run time fails its whole partition.",
    )
    parser.add_argument(
        "--upload",
        metavar="S3_URI",
        help="also upload to this s3:// prefix, where the backfill Lambdas can read it",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--partition-size", type=int, default=DEFAULT_PARTITION_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if start > end:
        raise SystemExit(f"--start {start} is after --end {end}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cycles = tuple(args.cycles)
    days = naqfc.days(args.start, args.end)

    for domain in args.domains:
        # Grid and domain are not independent (CS=227 Lambert, AK=198 polar
        # stereographic, HI=196 Mercator). Deriving it here is what keeps
        # --domains AK from silently emitting CONUS-grid URLs that match no
        # object -- a mistake --verify would catch but plain generation would not.
        grid = naqfc.grid_for_domain(domain, args.grid)

        # One listing per day serves every product in that domain, so verify
        # once per domain rather than once per product.
        existing = None
        if args.verify:
            prefixes = [
                day_prefix(day, args.bucket, args.collection, domain) for day in days
            ]
            print(f"\n{domain}: verifying against {len(prefixes)} day prefix(es)...")
            existing = list_existing(prefixes, args.workers)

        for product in args.products:
            print(f"\n{domain} / {product} (grid {grid}):")
            uris = naqfc.cycle_urls(
                args.start,
                args.end,
                cycles,
                bucket=args.bucket,
                collection=args.collection,
                domain=domain,
                product=product,
                grid=grid,
            )
            print(f"  {len(uris)} file(s) from the pattern")

            if existing is not None:
                uris = drop_missing(uris, existing)

            if not uris:
                print("  nothing to write, skipping")
                continue

            name = (
                f"naqfc_{args.collection.lower()}_{domain.lower()}_{product}"
                f"_{start:%Y%m%d}_{end:%Y%m%d}.json"
            )
            local = args.out_dir / name
            local.write_text(json.dumps(uris, indent=2))

            partitions = -(-len(uris) // args.partition_size)
            print(
                f"  wrote {local} ({len(uris)} files, {partitions} partitions/commits)"
            )
            print(f"  first: {uris[0]}")
            print(f"  last:  {uris[-1]}")

            inventory_uri = upload(local, args.upload) if args.upload else None
            if inventory_uri:
                print(f"  uploaded to {inventory_uri}")
            print(
                "\n  ./scripts/start_backfill.sh <execution-name> "
                f"{inventory_uri or '<inventory-uri>'}"
            )


if __name__ == "__main__":
    main()
