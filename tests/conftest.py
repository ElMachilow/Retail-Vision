import os
import tempfile

import pytest


@pytest.fixture(autouse=True)
def isolated_sqlite(monkeypatch):
    tmp_dir = tempfile.mkdtemp(prefix="tesis-test-")
    db_path = os.path.join(tmp_dir, "test.db")
    monkeypatch.setenv("SQLITE_PATH", db_path)
    monkeypatch.setenv("RECOGNITION_IMAGE_DIR", os.path.join(tmp_dir, "recognition-images"))
    from app.core import config as config_module
    from app.api import dependencies as dependencies_module
    from app.db import connection as connection_module

    config_module.get_settings.cache_clear()
    dependencies_module._cached_repository.cache_clear()
    dependencies_module._cached_recognition_repository.cache_clear()
    dependencies_module._cached_inventory_repository.cache_clear()
    dependencies_module._cached_suggestion_service.cache_clear()
    dependencies_module._cached_categorizer.cache_clear()
    connection_module._initialized.clear()
    yield
    config_module.get_settings.cache_clear()
    dependencies_module._cached_repository.cache_clear()
    dependencies_module._cached_recognition_repository.cache_clear()
    dependencies_module._cached_inventory_repository.cache_clear()
    dependencies_module._cached_suggestion_service.cache_clear()
    dependencies_module._cached_categorizer.cache_clear()
    connection_module._initialized.clear()
