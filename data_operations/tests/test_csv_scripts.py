import pytest
from unittest.mock import patch, mock_open, MagicMock
from data_operations.preprocessing.add_headers import DataParser


@pytest.fixture
def data_parser():
    return DataParser(encoding="ascii")


@pytest.fixture
def mock_files():
    return ["test1.csv", "test2.csv"]


@pytest.fixture
def headers():
    return ["Header1", "Header2", "Header3"]


def test_get_csv_files(data_parser, mock_files):
    with patch("os.listdir") as mock_listdir, patch("os.path.join") as mock_join:
        mock_listdir.return_value = mock_files
        mock_join.side_effect = lambda directory, filename: f"{directory}/{filename}"
        directory = "mock_directory"
        csv_files = data_parser.get_csv_files(directory)

        assert len(csv_files) == 2
        assert f"{directory}/test1.csv" in csv_files
        assert f"{directory}/test2.csv" in csv_files


def test_read_data(data_parser):
    mock_file_data = "line1\nline2\n"
    with patch("builtins.open", new_callable=mock_open, read_data=mock_file_data):
        result = data_parser._read_data("test.csv")

        assert result == ["line1\n", "line2\n"]


def test_merge_data(data_parser, headers):
    previous_data = ["line1\n", "line2\n"]
    with patch("builtins.open", new_callable=mock_open) as mock_file:
        data_parser._merge_data("test.csv", previous_data, headers)

        mock_file().write.assert_called_once_with("Header1,Header2,Header3\r\n")
        mock_file().writelines.assert_called_once_with(previous_data)


def test_prepend_headers(data_parser, mock_files, headers):
    with patch.object(
        data_parser, "_read_data", return_value=["line1\n", "line2\n"]
    ), patch.object(data_parser, "_merge_data") as mock_merge_data:
        data_parser.prepend_headers(mock_files, headers)

        data_parser._read_data.assert_any_call("test1.csv")
        data_parser._read_data.assert_any_call("test2.csv")
        assert mock_merge_data.call_count == 2


def test_prepend_headers_skip(data_parser, mock_files, headers):
    with patch.object(
        data_parser, "_read_data", return_value=["Header1,Header2,Header3\r\n"]
    ), patch.object(data_parser, "logger") as mock_logger:
        data_parser.prepend_headers(mock_files, headers)

        assert mock_logger.info.call_count == 2


if __name__ == "__main__":
    pytest.main()
