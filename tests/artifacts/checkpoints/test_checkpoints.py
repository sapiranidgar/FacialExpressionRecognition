from artifacts.checkpoints import CHECKPOINTS_DIR


def test_checkpoints_dir_points_at_its_own_directory():
    assert CHECKPOINTS_DIR.name == "checkpoints"
    assert CHECKPOINTS_DIR.is_dir()
