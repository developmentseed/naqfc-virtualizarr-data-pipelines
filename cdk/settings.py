import os
from typing import Any, Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings

print("STAGE from env:", os.getenv("STAGE"))


def include_trailing_slash(value: Any) -> Any:
    """Make sure the value includes a trailing slash if str"""
    if isinstance(value, str):
        return value.rstrip("/") + "/"
    return value


# NCEP grid ID per NAQFC domain. Grid and domain are not independent -- CS is a
# 5 km Lambert conformal, AK polar stereographic, HI Mercator -- and a
# mismatched pair yields object keys that match nothing.
GRID_BY_DOMAIN = {"CS": "227", "AK": "198", "HI": "196"}


class StackSettings(BaseSettings):
    PROJECT_NAME: str = "virtualizarr-data-pipelines"
    STACK_NAME: str = "virtualizarr-data-pipelines"
    STAGE: Literal["dev", "prod"]
    # Optional: when blank, app.py falls back to CDK_DEFAULT_ACCOUNT (the account
    # of the active AWS credentials) so synth/deploy still resolve an environment.
    ACCOUNT_ID: str | None = None
    ACCOUNT_REGION: str = "us-east-1"
    ICECHUNK_BUCKET_NAME: str = "icechunk-outuput"
    ICECHUNK_BUCKET: str | None = None
    # Common key prefix for every output written by this deployment. Backfill
    # artifacts are placed directly below it; ICECHUNK_PREFIX is relative to it.
    S3_PREFIX: str | None = None
    # Dataset-specific suffix for the Icechunk repo. Icechunk >=2.1.0 refuses to
    # create a repo at the bucket root, so S3_PREFIX or ICECHUNK_PREFIX must be
    # non-empty to bootstrap a new store. Passed into Lambda as the combined path.
    ICECHUNK_PREFIX: str | None = None
    DATA_BUCKET_NAME: str | None = None
    PROJECT: str = "virtualizarr-data-pipelines"
    SNS_TOPIC: str | None = None
    MAX_CONCURRENCY: int = 50
    SQS_BATCH_SIZE: int = 10

    # ARN of the Secrets Manager secret holding Earthdata {username, password}.
    # Optional: required only for reading protected GES DISC granules. When unset,
    # no secret resource or IAM grant is created.
    EARTHDATA_SECRET_ARN: str | None = None

    # Freguency in days to run garbage collection.
    GARBAGE_COLLECTION_FREQUENCY: int | None = None

    VPC_ID: str | None = None
    # AWS Batch cluster reference to SSM parameter describing the AMI _or_ the AMI ID
    # If using SSM to resolve the AMI ID, prefix with `resolve:ssm`.
    # MCP_AMI_ID: str = "resolve:ssm:/mcp/amis/aml2023-ecs"
    AMI_ID: str = (
        "resolve:ssm:/aws/service/ecs/optimized-ami/amazon-linux-2/recommended/image_id"
    )

    # Cluster scaling max
    BATCH_MAX_VCPU: int = 10

    # Backfill (partitioned fork/merge) pipeline
    BACKFILL_ENABLED: bool = False
    BACKFILL_PARTITION_SIZE: int = 500
    BACKFILL_MAX_ITEMS_PER_BATCH: int = 10
    BACKFILL_MAX_CONCURRENCY: int = 50

    # --- NAQFC dataset selection -------------------------------------------
    # One stack serves one (domain, product) pair. Deploy AK, HI and CONUS, or
    # o3 and pm25, as separate stacks with distinct STACK_NAME and
    # ICECHUNK_PREFIX. These are forwarded into every Lambda and also drive the
    # SNS subscription filter, so a stack cannot subscribe to one product while
    # its processor expects another.
    NAQFC_DATA_BUCKET: str = "noaa-nws-naqfc-pds"
    NAQFC_SOURCE_REGION: str = "us-east-1"
    NAQFC_COLLECTION: str = "AQMv7"
    NAQFC_DOMAIN: Literal["CS", "AK", "HI"] = "CS"
    # NCEP grid ID appearing in the filenames. Resolved from the domain when
    # unset (see the validator below); set only for a grid not in that map.
    NAQFC_GRID: str | None = None
    NAQFC_PRODUCT: str = "ave_1hr_o3"
    # Variable name the GRIB parser assigns to the decoded field; moves with
    # the product (o3 decodes to `ozcon`, pm25 to `pmtf`).
    NAQFC_VARIABLE: str = "ozcon"
    NAQFC_CYCLES: str = "06,12"
    # SNS subscription filter on the S3 object key. Set this to match the files
    # this stack should ingest -- it is not derived from the fields above, so
    # keep the two in step by hand: a filter admitting files the processor is
    # not configured for wastes consumer invocations, and one that admits
    # nothing leaves the queue silently empty.
    #
    # Glob-style, NOT a regular expression: '*' is the only metacharacter and
    # everything else matches literally. SNS allows at most three wildcards.
    # The dots around the product matter -- `ave_1hr_o3` is a prefix of
    # `ave_1hr_o3_bc`, so `*ave_1hr_o3*` would also admit the bias-corrected
    # product.
    NAQFC_KEY_PATTERN: str = "AQMv7/CS/*.ave_1hr_o3.*.227.grib2"
    # Backfill extent. The store's reference_time axis is built from these, and
    # scripts/generate_inventory.py defaults to them, so they must match the
    # inventory a backfill actually runs against.
    NAQFC_START: str = "2025-01-01"
    NAQFC_END: str = "2025-12-31"
    # Forecast length, which varies by product (ave_1hr_o3 = 72, ave_8hr_o3 = 65).
    NAQFC_LEAD_HOURS: int = 72

    # Forward SQS consumer. `None` resolves in the validator below:
    #   backfill enabled  -> default disabled (bootstrap via backfill, enable later)
    #   backfill disabled -> default enabled  (normal forward-only deployment)
    FORWARD_QUEUE_ENABLED: bool | None = None

    @property
    def s3_key_prefix(self) -> str | None:
        """Return the normalized global S3 key prefix."""
        return self.S3_PREFIX.strip("/") if self.S3_PREFIX else None

    @property
    def icechunk_storage_prefix(self) -> str | None:
        """Return the global and dataset-specific prefixes as one S3 key prefix."""
        return (
            "/".join(
                prefix.strip("/")
                for prefix in (self.S3_PREFIX, self.ICECHUNK_PREFIX)
                if prefix and prefix.strip("/")
            )
            or None
        )

    @model_validator(mode="after")
    def _validate_prefixes(self) -> "StackSettings":
        """Keep the Icechunk prefix relative to the global output prefix."""
        icechunk_prefix = (self.ICECHUNK_PREFIX or "").strip("/")
        if self.s3_key_prefix and (
            icechunk_prefix == self.s3_key_prefix
            or icechunk_prefix.startswith(f"{self.s3_key_prefix}/")
        ):
            raise ValueError("ICECHUNK_PREFIX must be relative to S3_PREFIX")
        return self

    @model_validator(mode="after")
    def _resolve_naqfc_grid(self) -> "StackSettings":
        """Fill NAQFC_GRID from the domain when it was not set explicitly."""
        if self.NAQFC_GRID is None:
            self.NAQFC_GRID = GRID_BY_DOMAIN[self.NAQFC_DOMAIN]
        return self

    @model_validator(mode="after")
    def _resolve_forward_queue_enabled(self) -> "StackSettings":
        if self.FORWARD_QUEUE_ENABLED is None:
            self.FORWARD_QUEUE_ENABLED = not self.BACKFILL_ENABLED
        return self
