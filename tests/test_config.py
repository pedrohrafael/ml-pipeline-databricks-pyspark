from src import config


def test_environment_flag():
    assert config.ENV in {"LOCAL", "DATABRICKS"}


def test_paths_defined():
    assert config.DATA_RAW_PATH is not None
