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

# The shipped default, matching the CONUS ozone deployment.
DEFAULT_PATTERN = StackSettings.model_fields["NAQFC_KEY_PATTERN"].default


def _template(**overrides: object) -> Template:
    settings = StackSettings(
        STAGE="dev",
        ACCOUNT_ID="111111111111",
        ICECHUNK_BUCKET_NAME="ice-test",
        DATA_BUCKET_NAME="data-test",
        SNS_TOPIC=TOPIC_ARN,
        **overrides,
    )
    app = cdk.App()
    stack = VirtualizarrSqsStack(
        app,
        settings.STACK_NAME,
        settings=settings,
        env={"account": settings.ACCOUNT_ID, "region": settings.ACCOUNT_REGION},
    )
    return Template.from_stack(stack)


def _subscription(**overrides: object) -> dict:
    subs = _template(**overrides).find_resources("AWS::SNS::Subscription")
    assert len(subs) == 1, subs
    return next(iter(subs.values()))["Properties"]


def _wildcard(**overrides: object) -> str:
    policy = _subscription(**overrides)["FilterPolicy"]
    return policy["Records"]["s3"]["object"]["key"][0]["wildcard"]


def test_filter_scope_is_message_body() -> None:
    """S3 event notifications set no message attributes, so the default
    attribute scope would match nothing. Payload scope is also what permits the
    nested Records/s3/object/key path."""
    assert _subscription()["FilterPolicyScope"] == "MessageBody"


def test_filter_targets_the_s3_object_key() -> None:
    policy = _subscription()["FilterPolicy"]

    assert policy == {
        "Records": {"s3": {"object": {"key": [{"wildcard": DEFAULT_PATTERN}]}}}
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
    assert fnmatch.fnmatchcase(key, DEFAULT_PATTERN)


@pytest.mark.parametrize("key", REJECTED)
def test_pattern_rejects(key: str) -> None:
    assert not fnmatch.fnmatchcase(key, DEFAULT_PATTERN)


def test_pattern_within_sns_wildcard_limit() -> None:
    """SNS allows at most three wildcards per pattern."""
    assert DEFAULT_PATTERN.count("*") <= 3


def test_inventory_keys_match_the_filter() -> None:
    """The filter is configured by hand, not derived from the dataset settings,
    so nothing enforces that it admits what a backfill would process. This is
    the check that catches a drift between the two for the default deployment.
    """
    for url in naqfc.cycle_urls("2025-01-01", "2025-01-03"):
        key = url.split(f"{naqfc.BUCKET}/", 1)[1]
        assert fnmatch.fnmatchcase(key, DEFAULT_PATTERN), key


# --- per-deployment configuration ------------------------------------------


def _processor_envs(template: Template) -> list[dict]:
    """Environments of the Lambdas that open the Icechunk store.

    Excludes CDK's own custom-resource providers, which carry no processor
    config and correctly know nothing about the dataset.
    """
    envs = [
        fn["Properties"].get("Environment", {}).get("Variables", {})
        for fn in template.find_resources("AWS::Lambda::Function").values()
    ]
    return [e for e in envs if "ICECHUNK_BUCKET" in e]


def test_dataset_selection_reaches_every_processor_lambda() -> None:
    """The filter alone is not enough -- the processor reads the same values
    from its environment, and a Lambda left on defaults would parse an AK file
    against a CONUS-shaped store."""
    envs = _processor_envs(
        _template(
            NAQFC_DOMAIN="AK",
            NAQFC_PRODUCT="ave_1hr_pm25",
            NAQFC_VARIABLE="pmtf",
            NAQFC_LEAD_HOURS=48,
        )
    )
    assert envs, "no processor Lambdas synthesized"

    for env in envs:
        assert env["NAQFC_DOMAIN"] == "AK"
        assert env["NAQFC_GRID"] == "198"  # derived, never set explicitly
        assert env["NAQFC_PRODUCT"] == "ave_1hr_pm25"
        assert env["NAQFC_VARIABLE"] == "pmtf"
        assert env["NAQFC_LEAD_HOURS"] == "48"


def test_backfill_handlers_get_the_same_selection() -> None:
    """The init handler builds the store's reference_time axis from the extent,
    so the backfill Lambdas need it too, not just the forward consumer."""
    envs = _processor_envs(
        _template(
            BACKFILL_ENABLED=True,
            NAQFC_DOMAIN="HI",
            NAQFC_START="2026-01-01",
            NAQFC_END="2026-06-30",
            NAQFC_CYCLES="06",
        )
    )
    # forward consumer + six backfill handlers
    assert len(envs) >= 7, f"expected the backfill handlers, got {len(envs)}"
    for env in envs:
        assert env["NAQFC_DOMAIN"] == "HI"
        assert env["NAQFC_GRID"] == "196"
        assert env["NAQFC_START"] == "2026-01-01"
        assert env["NAQFC_END"] == "2026-06-30"
        assert env["NAQFC_CYCLES"] == "06"


# --- pattern configuration --------------------------------------------------


def test_key_pattern_is_configured_not_derived() -> None:
    """The filter is an explicit setting: changing the dataset selection does
    not move it, which is why the two must be kept in step by hand."""
    assert _wildcard(NAQFC_DOMAIN="AK", NAQFC_PRODUCT="ave_1hr_pm25") == DEFAULT_PATTERN
    assert _wildcard(NAQFC_KEY_PATTERN="AQMv7/AK/*.ave_1hr_pm25.*.198.grib2") == (
        "AQMv7/AK/*.ave_1hr_pm25.*.198.grib2"
    )


def test_grid_still_resolves_from_domain() -> None:
    """Grid stays derived -- it is dataset config, not the filter."""
    for domain, grid in (("CS", "227"), ("AK", "198"), ("HI", "196")):
        envs = _processor_envs(_template(NAQFC_DOMAIN=domain))
        assert envs and all(e["NAQFC_GRID"] == grid for e in envs)


def test_explicit_grid_overrides_the_domain_default() -> None:
    envs = _processor_envs(_template(NAQFC_DOMAIN="AK", NAQFC_GRID="999"))
    assert envs and all(e["NAQFC_GRID"] == "999" for e in envs)
