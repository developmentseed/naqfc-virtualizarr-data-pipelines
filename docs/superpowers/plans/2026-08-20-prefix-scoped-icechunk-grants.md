# Prefix-Scoped Icechunk S3 Grants Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace every bucket-wide `icechunk_bucket.grant_read_write(...)` with grants scoped to the deployment's own key prefixes, so no stack's roles can write another stack's Icechunk store.

**Architecture:** One shared helper (`grant_prefixed_read_write`) emits the two spec-mandated statements (object read/write on `{prefix}/*`, `s3:ListBucket` with a `StringLike s3:prefix` condition) and falls back to the current bucket-wide grant when no prefix is configured. `cdk/stack.py` uses it for the processor Lambda, initialize Lambda, and GC job role (store prefix only). `cdk/stack_constructs/backfill_pipeline.py` uses it for the six backfill Lambdas — the five repo-opening handlers get the store prefix **plus** the backfill run prefix (`{s3_prefix}/backfill` — verified in source: fork writes `fork.pkl`, worker writes child forks, reduce lists/reads them there, all outside the store prefix), while `partition` gets only the run prefix plus a separate narrow `s3:GetObject` on a new `INVENTORY_PREFIX` setting.

**Tech Stack:** AWS CDK v2 (Python), pydantic-settings, pytest with `aws_cdk.assertions.Template`, uv, ruff, mypy.

**Spec:** `/workspace/context/next-task.md`

## Global Constraints

- Work on branch `claude/prefix-scoped-grants` (created in Task 1 from `feat/s3-prefix`). Never `git push` (denied in sandbox); never deploy.
- Run everything from the repo root: `/workspace/repos/naqfc-virtualizarr-data-pipelines`.
- Exact statement shapes from the spec: actions `s3:GetObject, s3:PutObject, s3:DeleteObject, s3:AbortMultipartUpload` on `arn:...:bucket/{prefix}/*`; `s3:ListBucket` on the bucket ARN with condition `StringLike: {"s3:prefix": ["{prefix}/*"]}`. Empty `icechunk_storage_prefix` ⇒ keep the current bucket-wide `grant_read_write` fallback.
- `cdk/` is mypy-strict (`disallow_untyped_defs`, `warn_return_any`; mypy `files = ["lambda", "cdk"]` — tests are NOT type-checked). Ruff: line-length 88, py312, rules E/F/I, double quotes.
- No `cdk` CLI in this sandbox: synth with `uv run --env-file .env_pm25_conus python cdk/app.py` (identical to what `cdk.json` runs). It needs no AWS credentials for the pm25 env files (GC/VPC lookup is disabled in them).
- `uv run` may rewrite `uv.lock`; never commit `uv.lock` churn — `git checkout -- uv.lock` if it shows as modified.
- Known accepted gap (document in the final commit, do NOT try to fix): CDK's `S3JsonItemReader` auto-grants the Step Functions state-machine role a bucket-wide read-only `s3:GetObject` on the Icechunk bucket. It is CDK-managed, read-only, and cannot overwrite any store; scoping it has no clean CDK knob. "No bucket-wide object grants" in verification is therefore asserted for **write-capable** actions (`s3:Put*`, `s3:Delete*`).
- The default STACK_NAME in tests is `virtualizarr-data-pipelines`; IAM policy logical IDs are matched by lowercase substring (e.g. `processmessageslambda`, `initializeicechunk`, `partitionfn`).

---

### Task 1: Grant helper + test plumbing

**Files:**
- Create: `cdk/stack_constructs/grants.py`
- Modify: `cdk/stack_constructs/__init__.py`
- Modify: `tests/cdk/conftest.py`
- Test: `tests/cdk/test_icechunk_grants.py` (new)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `grant_prefixed_read_write(grantee: iam.IGrantable, bucket: s3.IBucket, prefixes: Sequence[str | None]) -> None` — exported from `stack_constructs`; filters falsy/blank prefixes; empty result ⇒ `bucket.grant_read_write(grantee)`.
  - conftest helpers (plain functions, importable as `from conftest import ...` because `tests/cdk` has no `__init__.py` and pytest's prepend import mode puts the dir on `sys.path`): `resolve_joins(value)`, `iam_statements(template, role_marker=None)`, `actions_of(stmt)`, `resources_of(stmt)`.

- [ ] **Step 1: Create the branch**

```bash
cd /workspace/repos/naqfc-virtualizarr-data-pipelines
git checkout -b claude/prefix-scoped-grants
```

- [ ] **Step 2: Add template-rendering helpers to `tests/cdk/conftest.py`**

Append to the existing file (keep the existing `sys.path` lines at the top):

```python
from typing import Any, Iterator

from aws_cdk.assertions import Template


def resolve_joins(value: Any) -> Any:
    """Render Fn::Join nodes to strings; non-string parts become "<REF>".

    Mirrors the flattening idiom already used by _state_machine_asl in
    test_backfill_pipeline.py, applied to a whole template dict.
    """
    if isinstance(value, dict):
        if set(value) == {"Fn::Join"}:
            sep, parts = value["Fn::Join"]
            resolved = [resolve_joins(p) for p in parts]
            return sep.join(p if isinstance(p, str) else "<REF>" for p in resolved)
        return {k: resolve_joins(v) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_joins(v) for v in value]
    return value


def iam_statements(
    template: Template, role_marker: str | None = None
) -> Iterator[dict[str, Any]]:
    """Yield rendered statements from AWS::IAM::Policy resources, optionally
    only from policies whose logical id contains role_marker (case-insensitive)."""
    for logical_id, res in template.to_json()["Resources"].items():
        if res["Type"] != "AWS::IAM::Policy":
            continue
        if role_marker and role_marker.lower() not in logical_id.lower():
            continue
        yield from resolve_joins(res["Properties"]["PolicyDocument"]["Statement"])


def actions_of(stmt: dict[str, Any]) -> list[str]:
    action = stmt.get("Action", [])
    return action if isinstance(action, list) else [action]


def resources_of(stmt: dict[str, Any]) -> list[str]:
    resource = stmt.get("Resource", [])
    return resource if isinstance(resource, list) else [resource]
```

- [ ] **Step 3: Write the failing helper test**

Create `tests/cdk/test_icechunk_grants.py`:

```python
import aws_cdk as cdk
from aws_cdk import aws_iam as iam
from aws_cdk import aws_s3 as s3
from aws_cdk.assertions import Template
from conftest import actions_of, iam_statements, resources_of

STORE_OBJECTS = "arn:<REF>:s3:::ice-test/naqfc/aqmv7/o3_conus/*"
BUCKET_ARN = "arn:<REF>:s3:::ice-test"
BUCKET_WIDE = "arn:<REF>:s3:::ice-test/*"
WRITE_ACTIONS = [
    "s3:GetObject",
    "s3:PutObject",
    "s3:DeleteObject",
    "s3:AbortMultipartUpload",
]


def test_grant_helper_scopes_and_falls_back() -> None:
    from stack_constructs import grant_prefixed_read_write

    app = cdk.App()
    stack = cdk.Stack(
        app, "T", env=cdk.Environment(account="111111111111", region="us-east-1")
    )
    bucket = s3.Bucket.from_bucket_name(stack, "B", "ice-test")
    scoped = iam.Role(
        stack, "ScopedRole", assumed_by=iam.ServicePrincipal("lambda.amazonaws.com")
    )
    wide = iam.Role(
        stack, "WideRole", assumed_by=iam.ServicePrincipal("lambda.amazonaws.com")
    )
    grant_prefixed_read_write(scoped, bucket, ["naqfc/aqmv7/o3_conus", None, ""])
    grant_prefixed_read_write(wide, bucket, [None])
    template = Template.from_stack(stack)

    scoped_stmts = list(iam_statements(template, "scopedrole"))
    assert any(
        actions_of(s) == WRITE_ACTIONS and resources_of(s) == [STORE_OBJECTS]
        for s in scoped_stmts
    )
    assert any(
        actions_of(s) == ["s3:ListBucket"]
        and resources_of(s) == [BUCKET_ARN]
        and s["Condition"] == {"StringLike": {"s3:prefix": ["naqfc/aqmv7/o3_conus/*"]}}
        for s in scoped_stmts
    )
    assert not any(BUCKET_WIDE in resources_of(s) for s in scoped_stmts)

    wide_stmts = list(iam_statements(template, "widerole"))
    assert any(
        BUCKET_WIDE in resources_of(s) and "s3:DeleteObject*" in actions_of(s)
        for s in wide_stmts
    )
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `uv run pytest tests/cdk/test_icechunk_grants.py -v`
Expected: FAIL with `ImportError: cannot import name 'grant_prefixed_read_write'`

- [ ] **Step 5: Implement the helper**

Create `cdk/stack_constructs/grants.py`:

```python
"""Shared IAM grant helpers for the Icechunk bucket."""

from collections.abc import Sequence

from aws_cdk import aws_iam as iam
from aws_cdk import aws_s3 as s3


def grant_prefixed_read_write(
    grantee: iam.IGrantable,
    bucket: s3.IBucket,
    prefixes: Sequence[str | None],
) -> None:
    """Grant object read/write and listing on ``bucket`` under ``prefixes``.

    Six stacks share this bucket, one key prefix each; scoping the grants keeps
    one stack's roles from writing another stack's store. With no usable prefix
    the pre-existing bucket-wide grant is kept as the fallback.
    """
    keys = [p.strip("/") for p in prefixes if p and p.strip("/")]
    if not keys:
        bucket.grant_read_write(grantee)
        return
    principal = grantee.grant_principal
    principal.add_to_principal_policy(
        iam.PolicyStatement(
            actions=[
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:AbortMultipartUpload",
            ],
            resources=[bucket.arn_for_objects(f"{key}/*") for key in keys],
        )
    )
    principal.add_to_principal_policy(
        iam.PolicyStatement(
            actions=["s3:ListBucket"],
            resources=[bucket.bucket_arn],
            conditions={"StringLike": {"s3:prefix": [f"{key}/*" for key in keys]}},
        )
    )
```

Update `cdk/stack_constructs/__init__.py` to:

```python
from .aws_batch_infra import BatchInfra
from .aws_batch_job import BatchJob
from .backfill_pipeline import BackfillPipeline
from .grants import grant_prefixed_read_write

__all__ = [
    "BackfillPipeline",
    "BatchInfra",
    "BatchJob",
    "grant_prefixed_read_write",
]
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `uv run pytest tests/cdk/test_icechunk_grants.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git checkout -- uv.lock 2>/dev/null || true
git add cdk/stack_constructs/grants.py cdk/stack_constructs/__init__.py tests/cdk/conftest.py tests/cdk/test_icechunk_grants.py
git commit -m "feat: add prefix-scoped S3 grant helper for the Icechunk bucket

Emits object read/write on {prefix}/* plus s3:ListBucket conditioned on
s3:prefix, falling back to the bucket-wide grant when no prefix is set.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: `INVENTORY_PREFIX` setting

**Files:**
- Modify: `cdk/settings.py`
- Test: `tests/cdk/test_settings.py`

**Interfaces:**
- Consumes: existing `StackSettings.s3_key_prefix` property.
- Produces: `StackSettings.INVENTORY_PREFIX: str | None = None` field and `StackSettings.inventory_prefix -> str` property (explicit value stripped of slashes, else `"{s3_key_prefix}/inventory"`, else `"inventory"`). Task 4 passes it into `BackfillPipeline`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/cdk/test_settings.py` (pass `INVENTORY_PREFIX=None` explicitly so a stray env var can't leak into the default cases):

```python
def test_inventory_prefix_defaults_under_s3_prefix() -> None:
    s = StackSettings(STAGE="dev", S3_PREFIX="naqfc", INVENTORY_PREFIX=None)
    assert s.inventory_prefix == "naqfc/inventory"


def test_inventory_prefix_without_s3_prefix() -> None:
    s = StackSettings(STAGE="dev", S3_PREFIX=None, INVENTORY_PREFIX=None)
    assert s.inventory_prefix == "inventory"


def test_inventory_prefix_explicit_overrides_and_strips() -> None:
    s = StackSettings(STAGE="dev", S3_PREFIX="naqfc", INVENTORY_PREFIX="/custom/inv/")
    assert s.inventory_prefix == "custom/inv"
```

If `test_settings.py` does not already import `StackSettings`, add `from settings import StackSettings` at the top.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/cdk/test_settings.py -v -k inventory`
Expected: FAIL (`ValidationError` for the unknown field or `AttributeError: inventory_prefix`)

- [ ] **Step 3: Implement**

In `cdk/settings.py`, add the field after `ICECHUNK_PREFIX` (line ~39):

```python
    # Key prefix in the Icechunk bucket where backfill inventories are uploaded
    # (see README: s3://<bucket>/naqfc/inventory/). The backfill partition
    # Lambda is granted read on this prefix only.
    INVENTORY_PREFIX: str | None = None
```

and the property next to `icechunk_storage_prefix` (after line ~129):

```python
    @property
    def inventory_prefix(self) -> str:
        """Key prefix the backfill partition Lambda may read inventories from."""
        if self.INVENTORY_PREFIX:
            return self.INVENTORY_PREFIX.strip("/")
        return "/".join(p for p in (self.s3_key_prefix, "inventory") if p)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/cdk/test_settings.py -v`
Expected: PASS (all, including pre-existing tests)

- [ ] **Step 5: Commit**

```bash
git checkout -- uv.lock 2>/dev/null || true
git add cdk/settings.py tests/cdk/test_settings.py
git commit -m "feat: add INVENTORY_PREFIX setting defaulting to {S3_PREFIX}/inventory

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Scope the backfill pipeline grants

**Files:**
- Modify: `cdk/stack_constructs/backfill_pipeline.py`
- Test: `tests/cdk/test_backfill_pipeline.py`

**Interfaces:**
- Consumes: `grant_prefixed_read_write` from Task 1 (import from `.grants` inside the package, not from `stack_constructs`, to avoid a circular import through `__init__`).
- Produces: `BackfillPipeline.__init__` gains keyword-only param `inventory_prefix: str | None = None`. Task 4 passes `settings.inventory_prefix` for it.

Prefix facts verified in source (do not re-derive):
- Run prefix = `{s3_prefix}/backfill/{execution}/` (`_build_state_machine`, line ~119). partition writes manifests there; fork writes `forks/{id}/in/fork.pkl`; worker reads `fork_in_uri` and writes `{forks_out_prefix}{uuid}.pkl`; reduce lists+reads `forks/{id}/out/` (`fork_store.list_forks` needs `s3:ListBucket`). So repo handlers need BOTH the store prefix and `{s3_prefix}/backfill`.
- partition never opens the repo (`_REPO_ACTIONS` excludes it): it needs only the run prefix plus inventory read — not the store prefix.

- [ ] **Step 1: Write the failing tests**

Append to `tests/cdk/test_backfill_pipeline.py` (add `from conftest import actions_of, iam_statements, resources_of` to the imports):

```python
STORE = "arn:<REF>:s3:::ice-test/naqfc/aqmv7/o3_conus/*"
RUN = "arn:<REF>:s3:::ice-test/naqfc/backfill/*"
INVENTORY = "arn:<REF>:s3:::ice-test/naqfc/inventory/*"


def _scoped_template() -> Template:
    app = cdk.App()
    stack = cdk.Stack(
        app,
        "TestStack",
        env=cdk.Environment(account="111111111111", region="us-east-1"),
    )
    bucket = s3.Bucket.from_bucket_name(stack, "IceBucket", "ice-test")
    BackfillPipeline(
        stack,
        "Backfill",
        icechunk_bucket=bucket,
        icechunk_prefix="naqfc/aqmv7/o3_conus",
        s3_prefix="naqfc",
        inventory_prefix="naqfc/inventory",
        data_bucket_name="my-data-bucket",
        partition_size=500,
        max_items_per_batch=10,
        max_concurrency=50,
    )
    return Template.from_stack(stack)


def test_partition_reads_inventory_and_writes_run_prefix_only() -> None:
    stmts = list(iam_statements(_scoped_template(), "partitionfn"))
    all_resources = [r for s in stmts for r in resources_of(s)]
    assert INVENTORY in all_resources
    assert STORE not in all_resources
    writes = [s for s in stmts if "s3:PutObject" in actions_of(s)]
    assert writes and all(resources_of(s) == [RUN] for s in writes)


def test_repo_lambdas_scoped_to_store_and_run_prefixes() -> None:
    template = _scoped_template()
    for action in ["init", "fork", "worker", "reduce", "promote"]:
        stmts = list(iam_statements(template, f"{action}fn"))
        writes = [s for s in stmts if "s3:PutObject" in actions_of(s)]
        assert writes, action
        assert all(resources_of(s) == [STORE, RUN] for s in writes), action
        assert any(
            "s3:ListBucket" in actions_of(s)
            and s.get("Condition")
            == {
                "StringLike": {
                    "s3:prefix": ["naqfc/aqmv7/o3_conus/*", "naqfc/backfill/*"]
                }
            }
            for s in stmts
        ), action


def test_no_icechunk_prefix_keeps_bucket_wide_grant() -> None:
    stmts = list(iam_statements(_template(), "initfn"))
    assert any(
        "s3:DeleteObject*" in actions_of(s)
        and any(r.endswith("/*") for r in resources_of(s))
        for s in stmts
    )
```

(`_template()` is the existing module helper that builds the pipeline with `icechunk_prefix=None`.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/cdk/test_backfill_pipeline.py -v`
Expected: the three new tests FAIL (`TypeError: ... unexpected keyword argument 'inventory_prefix'` for the first two; the third may already pass — that is fine, it pins current behavior). Pre-existing tests PASS.

- [ ] **Step 3: Implement**

In `cdk/stack_constructs/backfill_pipeline.py`:

1. Add the import: `from .grants import grant_prefixed_read_write` (after the `constructs` import).
2. Add the constructor param (after `icechunk_prefix: str | None,`): `inventory_prefix: str | None = None,`.
3. Before the `for action in _ACTIONS:` loop, compute the run prefix (keep in step with `_build_state_machine`, which formats `s3://{bucket}/{s3_prefix}/backfill/{execution}/`):

```python
        # Backfill scratch space (partition manifests + pickled forks) lives
        # under {s3_prefix}/backfill/, outside the store prefix; keep this in
        # step with run_prefix in _build_state_machine.
        run_key_prefix = f"{s3_prefix}/backfill" if s3_prefix else "backfill"
```

4. In the loop, replace `icechunk_bucket.grant_read_write(fn)` with:

```python
            if not icechunk_prefix:
                icechunk_bucket.grant_read_write(fn)
            elif action == "partition":
                # partition never opens the repo: it reads the inventory and
                # writes partition manifests under the run prefix.
                grant_prefixed_read_write(fn, icechunk_bucket, [run_key_prefix])
            else:
                grant_prefixed_read_write(
                    fn, icechunk_bucket, [icechunk_prefix, run_key_prefix]
                )
```

5. After the loop (next to the existing `data_policy` block), add the narrow inventory grant:

```python
        if icechunk_prefix and inventory_prefix:
            self.functions["partition"].add_to_role_policy(
                iam.PolicyStatement(
                    actions=["s3:GetObject"],
                    resources=[
                        icechunk_bucket.arn_for_objects(
                            f"{inventory_prefix.strip('/')}/*"
                        )
                    ],
                )
            )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/cdk/test_backfill_pipeline.py -v`
Expected: PASS (all, including pre-existing tests)

- [ ] **Step 5: Commit**

```bash
git checkout -- uv.lock 2>/dev/null || true
git add cdk/stack_constructs/backfill_pipeline.py tests/cdk/test_backfill_pipeline.py
git commit -m "feat: prefix-scope backfill Lambda grants on the Icechunk bucket

Repo-opening handlers get the store prefix plus the {s3_prefix}/backfill
run prefix (fork pickles and partition manifests live there); partition
gets only the run prefix plus a narrow GetObject on the inventory prefix.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Scope the stack's grants (processor, initialize, GC)

**Files:**
- Modify: `cdk/stack.py`
- Test: `tests/cdk/test_icechunk_grants.py`

**Interfaces:**
- Consumes: `grant_prefixed_read_write` (Task 1), `settings.inventory_prefix` (Task 2), `BackfillPipeline(inventory_prefix=...)` (Task 3).
- Produces: nothing new for later tasks.

- [ ] **Step 1: Write the failing tests**

Append to `tests/cdk/test_icechunk_grants.py`:

```python
import pytest
from settings import StackSettings
from stack import VirtualizarrSqsStack


def _stack_template(**overrides: object) -> Template:
    kwargs: dict[str, object] = dict(
        STAGE="dev",
        ACCOUNT_ID="111111111111",
        ICECHUNK_BUCKET="ice-test",
        DATA_BUCKET_NAME="data-test",
        S3_PREFIX="naqfc",
        ICECHUNK_PREFIX="aqmv7/o3_conus",
        INVENTORY_PREFIX=None,
    )
    kwargs.update(overrides)
    settings = StackSettings(**{k: v for k, v in kwargs.items() if v is not None})
    app = cdk.App()
    stack = VirtualizarrSqsStack(
        app,
        settings.STACK_NAME,
        settings=settings,
        env={"account": settings.ACCOUNT_ID, "region": settings.ACCOUNT_REGION},
    )
    return Template.from_stack(stack)


@pytest.mark.parametrize("marker", ["processmessageslambda", "initializeicechunk"])
def test_stack_lambdas_scoped_to_store_prefix(marker: str) -> None:
    stmts = list(iam_statements(_stack_template(), marker))
    writes = [s for s in stmts if "s3:PutObject" in actions_of(s)]
    assert writes and all(resources_of(s) == [STORE_OBJECTS] for s in writes)
    assert any(
        "s3:ListBucket" in actions_of(s)
        and s.get("Condition")
        == {"StringLike": {"s3:prefix": ["naqfc/aqmv7/o3_conus/*"]}}
        for s in stmts
    )


@pytest.mark.parametrize("backfill", [False, True])
def test_no_bucket_wide_writes_remain_when_prefix_set(backfill: bool) -> None:
    template = _stack_template(BACKFILL_ENABLED=backfill)
    for stmt in iam_statements(template):
        if any(a.startswith(("s3:Put", "s3:Delete")) for a in actions_of(stmt)):
            assert BUCKET_WIDE not in resources_of(stmt)


def test_backfill_partition_gets_inventory_grant_from_settings() -> None:
    template = _stack_template(BACKFILL_ENABLED=True)
    stmts = list(iam_statements(template, "partitionfn"))
    assert any(
        resources_of(s) == ["arn:<REF>:s3:::ice-test/naqfc/inventory/*"]
        and actions_of(s) == ["s3:GetObject"]
        for s in stmts
    )


def test_no_prefix_keeps_bucket_wide_grant() -> None:
    template = _stack_template(S3_PREFIX=None, ICECHUNK_PREFIX=None)
    stmts = list(iam_statements(template, "processmessageslambda"))
    assert any(
        BUCKET_WIDE in resources_of(s) and "s3:DeleteObject*" in actions_of(s)
        for s in stmts
    )


def test_gc_role_scoped_to_store_prefix() -> None:
    template = _stack_template(GARBAGE_COLLECTION_FREQUENCY=2, VPC_ID="vpc-12345")
    stmts = list(iam_statements(template, "gcjobtaskrole"))
    writes = [s for s in stmts if "s3:PutObject" in actions_of(s)]
    assert writes and all(resources_of(s) == [STORE_OBJECTS] for s in writes)
```

Fallback note for the GC test only: `Vpc.from_lookup` in a unit test synthesizes a dummy VPC when lookup context is missing. If it instead raises in this environment, delete `test_gc_role_scoped_to_store_prefix` and note in the final commit that the GC path is covered by the shared-helper test (the GC call site is the identical one-line helper call).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/cdk/test_icechunk_grants.py -v`
Expected: the new tests FAIL (`test_no_prefix_keeps_bucket_wide_grant` may already pass — fine, it pins the fallback); Task 1's helper test still PASSES.

- [ ] **Step 3: Implement**

In `cdk/stack.py`:

1. Change the constructs import (line 50) to:
   `from stack_constructs import BackfillPipeline, BatchInfra, BatchJob, grant_prefixed_read_write`
2. Replace line 267 `self.icechunk_bucket.grant_read_write(self.process_messages_lambda)` with:

```python
        grant_prefixed_read_write(
            self.process_messages_lambda,
            self.icechunk_bucket,
            [settings.icechunk_storage_prefix],
        )
```

3. Replace line 297 `self.icechunk_bucket.grant_read_write(self.initialize_icechunk_lambda)` with:

```python
            grant_prefixed_read_write(
                self.initialize_icechunk_lambda,
                self.icechunk_bucket,
                [settings.icechunk_storage_prefix],
            )
```

4. Replace line 373 `self.icechunk_bucket.grant_read_write(self.gc_job.role)` with:

```python
            grant_prefixed_read_write(
                self.gc_job.role,
                self.icechunk_bucket,
                [settings.icechunk_storage_prefix],
            )
```

5. In the `BackfillPipeline(...)` call (line ~396), add after `icechunk_prefix=settings.icechunk_storage_prefix,`:

```python
                inventory_prefix=settings.inventory_prefix,
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/cdk -v`
Expected: PASS (all cdk tests, new and pre-existing)

- [ ] **Step 5: Commit**

```bash
git checkout -- uv.lock 2>/dev/null || true
git add cdk/stack.py tests/cdk/test_icechunk_grants.py
git commit -m "feat: prefix-scope processor, initialize, and GC grants on the Icechunk bucket

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Full verification + real-env synth check

**Files:**
- No source changes expected (fixes only if verification fails).

**Interfaces:**
- Consumes: everything above.
- Produces: verified branch `claude/prefix-scoped-grants`, ready for the human to export/push and deploy.

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest`
Expected: PASS (includes `tests/backfill_handlers` and `tests/backfill_mechanics`, untouched by this change)

- [ ] **Step 2: Lint, format, and type-check**

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Expected: all clean. If `ruff format --check` flags only files this branch did not touch, leave them; fix any file this branch modified.

- [ ] **Step 3: Synth the pm25 conus stack from its real env file**

```bash
uv run --env-file .env_pm25_conus python cdk/app.py
```

Expected: exits 0 and writes `cdk.out/naqfc-pm25-conus.template.json`. (No `cdk` CLI in the sandbox; `cdk.json` runs exactly this command.)

- [ ] **Step 4: Assert no bucket-wide write grants remain in the synthesized template**

```bash
uv run python - <<'EOF'
import json, sys

sys.path.insert(0, "tests/cdk")
from conftest import actions_of, resolve_joins, resources_of

tmpl = json.load(open("cdk.out/naqfc-pm25-conus.template.json"))
bad = []
for logical_id, res in tmpl["Resources"].items():
    if res["Type"] != "AWS::IAM::Policy":
        continue
    for stmt in resolve_joins(res["Properties"]["PolicyDocument"]["Statement"]):
        if any(a.startswith(("s3:Put", "s3:Delete")) for a in actions_of(stmt)):
            wide = [
                r
                for r in resources_of(stmt)
                if r.endswith(":s3:::airquality-data-store-develop/*")
            ]
            if wide:
                bad.append((logical_id, stmt))
assert not bad, bad
print("OK: no bucket-wide write grants on the Icechunk bucket")
scoped = json.dumps(tmpl)
assert "airquality-data-store-develop/naqfc/aqmv7/pm25_conus/*" in scoped
assert "naqfc/inventory/*" in scoped
print("OK: store-prefix and inventory-prefix grants present")
EOF
```

Expected: both `OK` lines print. (The state-machine role keeps a CDK-managed read-only `s3:GetObject` on the bucket — expected, see Global Constraints.)

- [ ] **Step 5: Clean up and final commit**

```bash
rm -rf cdk.out
git checkout -- uv.lock 2>/dev/null || true
git status --short   # expect empty
```

If any verification step required source fixes, commit them now with a message explaining what and why. Then record the verification summary in an empty commit:

```bash
git commit --allow-empty -m "chore: verify prefix-scoped grants (pytest, ruff, mypy, pm25 synth)

uv run pytest, ruff check, ruff format --check, and mypy all pass.
uv run --env-file .env_pm25_conus python cdk/app.py synthesizes; the
template has no bucket-wide s3:Put*/s3:Delete* grants on the Icechunk
bucket for any role. Known remaining bucket-wide READ: the Step
Functions Distributed Map item-reader grant (CDK-managed s3:GetObject,
cannot overwrite stores).

Because this change only narrows existing identity-policy statements,
the pm25 cdk deploys should not prompt for broad IAM approval, and
re-deploying the o3 stacks picks up the tightened policies.

Post-deploy runbook (from the spec, runs on the host, not here): after
the first pm25 deploy, run one backfill and one forward-processing
message end-to-end before deploying the rest, to catch any missed S3
action at runtime.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Self-review notes

- Spec coverage: three `stack.py` call sites → Task 4; backfill Lambdas → Task 3; two-statement shape + fallback → Task 1; `INVENTORY_PREFIX` field with `{S3_PREFIX}/inventory` default → Task 2; "check which prefix manifest writes land under" → resolved in Task 3 preamble (run prefix `{s3_prefix}/backfill`, granted to partition AND to fork/worker/reduce, which the spec's partition-only framing missed but its end-to-end runbook anticipated); template tests + synth check + lint/type → Tasks 1–5; branch + commit summary → Tasks 1–5. Deploys and the end-to-end runtime check are host-side and explicitly recorded in the final commit message.
- Deliberate deviation from a literal spec reading: the SM item-reader's bucket-wide read-only GetObject stays (no clean CDK scope knob; cannot overwrite stores); "no bucket-wide object grants" is enforced for write-capable actions.
