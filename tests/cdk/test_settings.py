from settings import StackSettings


def test_backfill_settings_defaults() -> None:
    settings = StackSettings(STAGE="dev", ACCOUNT_ID="111111111111")
    assert settings.BACKFILL_ENABLED is False
    assert settings.BACKFILL_PARTITION_SIZE == 500
    assert settings.BACKFILL_MAX_ITEMS_PER_BATCH == 10
    assert settings.BACKFILL_MAX_CONCURRENCY == 50


def test_bucket_prefixes_are_normalized_independently() -> None:
    settings = StackSettings(
        STAGE="dev",
        ACCOUNT_ID="111111111111",
        ICECHUNK_PREFIX="/naqfc/aqmv7/o3_conus/",
        BACKFILL_PREFIX="/naqfc/",
    )

    assert settings.icechunk_storage_prefix == "naqfc/aqmv7/o3_conus"
    assert settings.backfill_key_prefix == "naqfc"


def test_forward_queue_enabled_defaults_on_when_backfill_off() -> None:
    settings = StackSettings(STAGE="dev", ACCOUNT_ID="111111111111")
    assert settings.FORWARD_QUEUE_ENABLED is True


def test_forward_queue_enabled_defaults_off_when_backfill_on() -> None:
    settings = StackSettings(
        STAGE="dev", ACCOUNT_ID="111111111111", BACKFILL_ENABLED=True
    )
    assert settings.FORWARD_QUEUE_ENABLED is False


def test_forward_queue_enabled_explicit_value_is_honored() -> None:
    settings = StackSettings(
        STAGE="dev",
        ACCOUNT_ID="111111111111",
        BACKFILL_ENABLED=True,
        FORWARD_QUEUE_ENABLED=True,
    )
    assert settings.FORWARD_QUEUE_ENABLED is True


def test_forward_queue_disabled_explicit_with_backfill_off() -> None:
    settings = StackSettings(
        STAGE="dev",
        ACCOUNT_ID="111111111111",
        FORWARD_QUEUE_ENABLED=False,
    )
    assert settings.FORWARD_QUEUE_ENABLED is False


def test_inventory_prefix_defaults_under_backfill_prefix() -> None:
    s = StackSettings(STAGE="dev", BACKFILL_PREFIX="naqfc", INVENTORY_PREFIX=None)
    assert s.inventory_prefix == "naqfc/inventory"


def test_inventory_prefix_without_backfill_prefix() -> None:
    s = StackSettings(STAGE="dev", BACKFILL_PREFIX=None, INVENTORY_PREFIX=None)
    assert s.inventory_prefix == "inventory"


def test_inventory_prefix_explicit_overrides_and_strips() -> None:
    s = StackSettings(
        STAGE="dev", BACKFILL_PREFIX="naqfc", INVENTORY_PREFIX="/custom/inv/"
    )
    assert s.inventory_prefix == "custom/inv"
