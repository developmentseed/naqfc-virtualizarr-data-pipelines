from collections.abc import Iterator
from unittest.mock import MagicMock

import boto3
import pytest
from moto import mock_aws
from stub_processor import StubProcessor

BUCKET = "test-backfill-bucket"

# Handler modules that construct a Processor at invocation time.
HANDLER_MODULES = ("init", "fork", "worker", "reduce", "promote")


@pytest.fixture(autouse=True)
def stub_processor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the handlers against the synthetic processor.

    These tests cover handler wiring -- fork artifacts in and out of S3, event
    shapes, the serial-partition sequence -- none of which depends on the
    dataset. Pointing them at the real NAQFC processor would make every case
    download GRIB from a public bucket to assert something unrelated to GRIB.
    """
    for name in HANDLER_MODULES:
        module = __import__(f"backfill_handlers.{name}", fromlist=["Processor"])
        monkeypatch.setattr(module, "Processor", StubProcessor)


@pytest.fixture()
def s3_bucket() -> Iterator[str]:
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield BUCKET


@pytest.fixture()
def lambda_context() -> MagicMock:
    """A stand-in Lambda context.

    powertools' @logger.inject_lambda_context reads context.function_name etc.
    at invocation, so handlers cannot be called with None; a MagicMock supplies
    any attribute (matching the existing tests/test_handler.py convention).
    """
    return MagicMock()
