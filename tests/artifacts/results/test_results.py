from artifacts.results import RESULTS_DIR


def test_results_dir_points_at_its_own_directory():
    assert RESULTS_DIR.name == "results"
    assert RESULTS_DIR.is_dir()
