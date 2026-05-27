"""Tests for foam_list_times — listing time directories."""

import pytest

from pyfoam.tools.foam_list_times import foam_list_times


@pytest.fixture
def case_dir(tmp_path):
    """Create a minimal OpenFOAM case with time directories."""
    for t in ["0", "0.001", "0.01", "0.1", "1", "10"]:
        (tmp_path / t).mkdir()
    # Non-time directories should be ignored
    (tmp_path / "constant").mkdir()
    (tmp_path / "system").mkdir()
    # Non-numeric files should be ignored
    (tmp_path / "log").touch()
    return tmp_path


class TestListAllTimes:
    """Default behaviour: return all time directories."""

    def test_returns_all_numeric_dirs(self, case_dir):
        times = foam_list_times(case_dir)
        assert times == [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]

    def test_sorted_ascending(self, case_dir):
        times = foam_list_times(case_dir)
        assert times == sorted(times)

    def test_ignores_non_numeric_dirs(self, case_dir):
        times = foam_list_times(case_dir)
        # "constant", "system" should not appear
        assert all(isinstance(t, float) for t in times)
        assert len(times) == 6

    def test_ignores_files(self, case_dir):
        times = foam_list_times(case_dir)
        assert 10.0 in times  # "10" is a dir, not confused with "log"

    def test_empty_case(self, tmp_path):
        """Case with no time directories returns empty list."""
        tmp_path.mkdir(exist_ok=True)
        (tmp_path / "constant").mkdir()
        assert foam_list_times(tmp_path) == []


class TestLatestTime:
    """Selector 'latestTime'."""

    def test_returns_largest(self, case_dir):
        times = foam_list_times(case_dir, time_selector="latestTime")
        assert times == [10.0]

    def test_empty_case(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        assert foam_list_times(tmp_path, time_selector="latestTime") == []


class TestFirstTime:
    """Selector 'firstTime'."""

    def test_returns_smallest(self, case_dir):
        times = foam_list_times(case_dir, time_selector="firstTime")
        assert times == [0.0]

    def test_empty_case(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        assert foam_list_times(tmp_path, time_selector="firstTime") == []


class TestNumericSelector:
    """Numeric time_selector: return times <= value."""

    def test_float_selector(self, case_dir):
        times = foam_list_times(case_dir, time_selector=0.1)
        assert times == [0.0, 0.001, 0.01, 0.1]

    def test_int_selector(self, case_dir):
        times = foam_list_times(case_dir, time_selector=1)
        assert times == [0.0, 0.001, 0.01, 0.1, 1.0]

    def test_exact_match(self, case_dir):
        times = foam_list_times(case_dir, time_selector=0.001)
        assert times == [0.0, 0.001]

    def test_below_all(self, case_dir):
        times = foam_list_times(case_dir, time_selector=-1.0)
        assert times == []

    def test_above_all(self, case_dir):
        times = foam_list_times(case_dir, time_selector=100.0)
        assert times == [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]


class TestErrors:
    """Error handling."""

    def test_nonexistent_path(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            foam_list_times(tmp_path / "nonexistent")

    def test_not_a_directory(self, tmp_path):
        f = tmp_path / "file.txt"
        f.touch()
        with pytest.raises(FileNotFoundError):
            foam_list_times(f)

    def test_unknown_string_selector(self, case_dir):
        with pytest.raises(ValueError, match="Unknown time_selector"):
            foam_list_times(case_dir, time_selector="badSelector")


class TestDecimalNames:
    """Directories with decimal point names (e.g. 0.005)."""

    def test_decimal_directories(self, tmp_path):
        for t in ["0", "0.005", "0.5", "1"]:
            (tmp_path / t).mkdir()
        times = foam_list_times(tmp_path)
        assert times == [0.0, 0.005, 0.5, 1.0]
