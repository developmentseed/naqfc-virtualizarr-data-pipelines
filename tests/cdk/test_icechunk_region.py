"""ICECHUNK_REGION reaches the handlers only when it is set.

The store's bucket may live outside the region the stack deploys into. Pinning
the region at synth time would make that unexpressible, so an unset setting
leaves the variable out of the Lambda environment entirely and icechunk resolves
the region from the running handler instead.

This is also the reason backfill artifacts get a bucket of their own: the Step
Functions item reader takes no region and assumes the stack's, so the manifests
cannot follow the store somewhere else.
"""

import aws_cdk as cdk
from aws_cdk.assertions import Template
from settings import StackSettings
from stack import VirtualizarrSqsStack


def _lambda_environments(region: str | None, *, backfill: bool = False) -> list[dict]:
    """Every Lambda's environment in a stack synthesized with this region."""
    kwargs = dict(
        STAGE="dev",
        ACCOUNT_ID="111111111111",
        ICECHUNK_BUCKET_NAME="ice-test",
        DATA_BUCKET_NAME="data-test",
        BACKFILL_ENABLED=backfill,
    )
    if region is not None:
        kwargs["ICECHUNK_REGION"] = region
    settings = StackSettings(**kwargs)
    app = cdk.App()
    stack = VirtualizarrSqsStack(
        app,
        settings.STACK_NAME,
        settings=settings,
        env={"account": settings.ACCOUNT_ID, "region": settings.ACCOUNT_REGION},
    )
    template = Template.from_stack(stack)
    return [
        fn["Properties"].get("Environment", {}).get("Variables", {})
        for fn in template.find_resources("AWS::Lambda::Function").values()
    ]


def test_the_setting_defaults_to_unset() -> None:
    settings = StackSettings(STAGE="dev", ACCOUNT_ID="111111111111")
    assert settings.ICECHUNK_REGION is None


def test_an_unset_region_reaches_no_handler() -> None:
    """Not merely absent from the config: absent from the Lambda environment, so
    icechunk resolves it at runtime rather than being pinned to the deploy
    region."""
    environments = _lambda_environments(None)

    assert environments, "expected at least one Lambda in the stack"
    assert all("ICECHUNK_REGION" not in env for env in environments)
    # the bucket still reaches them; only the region is left open
    assert any("ICECHUNK_BUCKET" in env for env in environments)


def test_a_set_region_reaches_every_handler_that_opens_the_store() -> None:
    environments = _lambda_environments("us-west-2")
    carrying_bucket = [env for env in environments if "ICECHUNK_BUCKET" in env]

    assert carrying_bucket
    assert all(env.get("ICECHUNK_REGION") == "us-west-2" for env in carrying_bucket)


def test_the_backfill_handlers_carry_it_too() -> None:
    """The backfill Lambdas build their environment in the pipeline construct
    rather than from processor_env, so they are a separate path to the same
    setting."""
    environments = _lambda_environments("eu-central-1", backfill=True)
    carrying_bucket = [env for env in environments if "ICECHUNK_BUCKET" in env]

    # forward handlers plus the six backfill handlers
    assert len(carrying_bucket) > 6
    assert all(env.get("ICECHUNK_REGION") == "eu-central-1" for env in carrying_bucket)


def test_an_unset_region_reaches_no_backfill_handler_either() -> None:
    environments = _lambda_environments(None, backfill=True)

    assert all("ICECHUNK_REGION" not in env for env in environments)
