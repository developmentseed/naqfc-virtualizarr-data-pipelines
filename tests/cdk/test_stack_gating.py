import aws_cdk as cdk
from aws_cdk.assertions import Match, Template
from settings import StackSettings
from stack import VirtualizarrSqsStack


def _template(
    *,
    backfill: bool,
    forward: bool | None = None,
    icechunk_bucket: str | None = None,
) -> Template:
    kwargs = dict(
        STAGE="dev",
        ACCOUNT_ID="111111111111",
        ICECHUNK_BUCKET_NAME="ice-test",
        DATA_BUCKET_NAME="data-test",
        BACKFILL_ENABLED=backfill,
    )
    if forward is not None:
        kwargs["FORWARD_QUEUE_ENABLED"] = forward
    if icechunk_bucket is not None:
        kwargs["ICECHUNK_BUCKET"] = icechunk_bucket
    settings = StackSettings(**kwargs)
    app = cdk.App()
    stack = VirtualizarrSqsStack(
        app,
        settings.STACK_NAME,
        settings=settings,
        env={"account": settings.ACCOUNT_ID, "region": settings.ACCOUNT_REGION},
    )
    return Template.from_stack(stack)


def _synth(enabled: bool) -> Template:
    return _template(backfill=enabled)


def test_backfill_disabled_creates_no_state_machine() -> None:
    _synth(False).resource_count_is("AWS::StepFunctions::StateMachine", 0)


def test_backfill_enabled_creates_state_machine() -> None:
    template = _synth(True)
    template.resource_count_is("AWS::StepFunctions::StateMachine", 1)
    template.resource_count_is("AWS::CloudFormation::CustomResource", 0)


def test_existing_icechunk_bucket_must_match_stack_region() -> None:
    template = _template(backfill=True, icechunk_bucket="existing-bucket")

    template.has_resource_properties(
        "AWS::CloudFormation::CustomResource",
        Match.object_like(
            {
                "BucketName": "existing-bucket",
                "ExpectedRegion": "us-east-1",
            }
        ),
    )
    template.has_resource_properties(
        "AWS::Lambda::Function",
        Match.object_like({"Timeout": 30}),
    )
    template.has_resource_properties(
        "AWS::IAM::Policy",
        Match.object_like(
            {
                "PolicyDocument": {
                    "Statement": Match.array_with(
                        [
                            Match.object_like(
                                {
                                    "Action": "s3:GetBucketLocation",
                                    "Resource": Match.any_value(),
                                }
                            )
                        ]
                    )
                }
            }
        ),
    )


def test_s3_prefix_scopes_the_icechunk_store() -> None:
    settings = StackSettings(
        STAGE="dev",
        ACCOUNT_ID="111111111111",
        ICECHUNK_BUCKET_NAME="ice-test",
        DATA_BUCKET_NAME="data-test",
        S3_PREFIX="naqfc",
        ICECHUNK_PREFIX="aqmv7/o3_conus",
    )
    app = cdk.App()
    stack = VirtualizarrSqsStack(
        app,
        settings.STACK_NAME,
        settings=settings,
        env={"account": settings.ACCOUNT_ID, "region": settings.ACCOUNT_REGION},
    )

    Template.from_stack(stack).has_resource_properties(
        "AWS::Lambda::Function",
        Match.object_like(
            {
                "Environment": {
                    "Variables": {
                        "ICECHUNK_PREFIX": "naqfc/aqmv7/o3_conus",
                        "ICECHUNK_REGION": "us-east-1",
                    }
                }
            }
        ),
    )


def test_forward_queue_enabled_when_backfill_off() -> None:
    t = _template(backfill=False)
    t.resource_count_is("AWS::Lambda::EventSourceMapping", 1)
    t.has_resource_properties("AWS::Lambda::EventSourceMapping", {"Enabled": True})


def test_forward_queue_disabled_when_backfill_on() -> None:
    t = _template(backfill=True)
    t.resource_count_is("AWS::Lambda::EventSourceMapping", 1)
    t.has_resource_properties("AWS::Lambda::EventSourceMapping", {"Enabled": False})


def test_forward_queue_explicit_enable_with_backfill_on() -> None:
    t = _template(backfill=True, forward=True)
    t.resource_count_is("AWS::Lambda::EventSourceMapping", 1)
    t.has_resource_properties("AWS::Lambda::EventSourceMapping", {"Enabled": True})


def _resource_ids(template: Template) -> str:
    return " ".join(template.to_json()["Resources"].keys()).lower()


def test_backfill_disabled_creates_initialize_lambda() -> None:
    assert "initializeicechunk" in _resource_ids(_template(backfill=False))


def test_backfill_enabled_skips_initialize_lambda() -> None:
    assert "initializeicechunk" not in _resource_ids(_template(backfill=True))
