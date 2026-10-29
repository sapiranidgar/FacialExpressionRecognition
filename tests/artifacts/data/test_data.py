from artifacts.data import DATA_DIR


def test_data_dir_points_at_its_own_directory():
    assert DATA_DIR.name == "data"
    assert DATA_DIR.is_dir()
