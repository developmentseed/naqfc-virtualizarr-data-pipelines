"""The SNS subscription filter that gates the forward queue.

The NewNWSAirQualityObject topic announces every NWS air quality object, so the
filter is the only thing keeping other model versions, domains, and pollutants
out of the queue. A filter that is too loose wastes consumer invocations on
files the processor rejects; one that is too tight starves the queue silently.
Both failure modes are invisible until you inspect a deployed subscription, so
the pattern is pinned here.
"""

import fnmatch

import aws_cdk as cdk
import pytest
from aws_cdk.assertions import Template
from settings import StackSettings
from stack import VirtualizarrSqsStack
from virtualizarr_processor import naqfc

TOPIC_ARN = "arn:aws:sns:us-east-1:709902155096:NewNWSAirQualityObject"


def _subscription() -> dict:
    settings = StackSettings(
        STAGE="dev",
        ACCOUNT_ID="111111111111",
        ICECHUNK_BUCKET_NAME="ice-test",
        DATA_BUCKET_NAME="data-test",
        SNS_TOPIC=TOPIC_ARN,
    )
    app = cdk.App()
    stack = VirtualizarrSqsStack(
        app,
        settings.STACK_NAME,
        settings=settings,
        env={"account": settings.ACCOUNT_ID, "region": settings.ACCOUNT_REGION},
    )
    subs = Template.from_stack(stack).find_resources("AWS::SNS::Subscription")
    assert len(subs) == 1, subs
    return next(iter(subs.values()))["Properties"]


def test_filter_scope_is_message_body() -> None:
    """S3 event notifications set no message attributes, so the default
    attribute scope would match nothing. Payload scope is also what permits the
    nested Records/s3/object/key path."""
    assert _subscription()["FilterPolicyScope"] == "MessageBody"


def test_filter_targets_the_s3_object_key() -> None:
    policy = _subscription()["FilterPolicy"]

    assert policy == {
        "Records": {
            "s3": {"object": {"key": [{"wildcard": naqfc.object_key_wildcard()}]}}
        }
    }


def test_raw_message_delivery_preserved() -> None:
    """The consumer parses the S3 event directly; SNS envelope wrapping would
    break its Records lookup."""
    assert _subscription()["RawMessageDelivery"] is True


# --- what the pattern actually admits --------------------------------------

CYCLE = "AQMv7/CS/20250601/06/aqm.t06z"

ACCEPTED = [
    f"{CYCLE}.ave_1hr_o3.20250601.227.grib2",
    "AQMv7/CS/20251231/12/aqm.t12z.ave_1hr_o3.20251231.227.grib2",
]

REJECTED = [
    # bias-corrected: a different variable, and the reason the product is
    # bracketed by dots rather than matched as a bare substring
    f"{CYCLE}.ave_1hr_o3_bc.20250601.227.grib2",
    # other pollutants and averaging periods
    f"{CYCLE}.ave_8hr_o3.20250601.227.grib2",
    f"{CYCLE}.ave_1hr_pm25.20250601.227.grib2",
    f"{CYCLE}.max_1hr_o3.20250601.227.grib2",
    # other domains: different projection and grid, unreadable by the parser
    "AQMv7/AK/20250601/06/aqm.t06z.ave_1hr_o3.20250601.198.grib2",
    "AQMv7/HI/20250601/06/aqm.t06z.ave_1hr_o3.20250601.196.grib2",
    # other model versions
    "AQMv6/CS/20250601/06/aqm.t06z.ave_1hr_o3.20250601.227.grib2",
    "AQMv7_suppl/CS/20250601/06/aqm.t06z.ave_1hr_o3.20250601.227.grib2",
    # other collections on the same topic
    "RAP_Smoke/CS/20250601/06/aqm.t06z.1hr_SfcSmoke.20250601.227.grib2",
]


@pytest.mark.parametrize("key", ACCEPTED)
def test_pattern_accepts(key: str) -> None:
    assert fnmatch.fnmatchcase(key, naqfc.object_key_wildcard())


@pytest.mark.parametrize("key", REJECTED)
def test_pattern_rejects(key: str) -> None:
    assert not fnmatch.fnmatchcase(key, naqfc.object_key_wildcard())


def test_pattern_within_sns_wildcard_limit() -> None:
    """SNS allows at most three wildcards per pattern."""
    assert naqfc.object_key_wildcard().count("*") <= 3


def test_inventory_keys_match_the_filter() -> None:
    """Backfill and forward processing must agree on what belongs in the store:
    every URL the inventory generator emits should be one the filter admits."""
    for url in naqfc.cycle_urls("2025-01-01", "2025-01-03"):
        key = url.split(f"{naqfc.BUCKET}/", 1)[1]
        assert fnmatch.fnmatchcase(key, naqfc.object_key_wildcard()), key
