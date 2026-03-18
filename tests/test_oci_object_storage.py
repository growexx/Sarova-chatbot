import os
import tempfile
import pytest
import pandas as pd
from unittest.mock import MagicMock

from code_modules.oci_object_storage import OCIObjectStorageClient


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def mock_config(monkeypatch):
    fake_config = {
        "DEFAULT": {
            "user": "user",
            "key_file": "key.pem",
            "fingerprint": "fp",
            "tenancy": "tenancy",
        },
        "BUCKET": {
            "namespace": "ns",
            "bucket_name": "bucket",
            "compartment_id": "compartment",
            "region": "eu-frankfurt-1",
        },
    }

    mock_parser = MagicMock()
    mock_parser.__getitem__.side_effect = lambda k: fake_config[k]

    monkeypatch.setattr("configparser.ConfigParser", lambda: mock_parser)


@pytest.fixture
def mock_oci_client(monkeypatch):
    mock_client = MagicMock()

    monkeypatch.setattr(
        "oci.object_storage.ObjectStorageClient",
        lambda config: mock_client
    )

    return mock_client


@pytest.fixture
def client(mock_config, mock_oci_client):
    return OCIObjectStorageClient()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def sample_chunk_generator():
    yield pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    yield pd.DataFrame({"col1": [3], "col2": ["c"]})


def empty_chunk_generator():
    if False:
        yield


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def test_init_loads_config(mock_config, mock_oci_client):
    client = OCIObjectStorageClient()
    assert client.bucket_info["namespace"] == "ns"
    assert client.bucket_info["bucket_name"] == "bucket"


def test_stream_chunks_to_excel_success(client, mock_oci_client):
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        local_path = tmp.name

    result = client.stream_chunks_to_excel_and_upload(
        sample_chunk_generator(),
        local_file_path=local_path,
        bucket_folder_name="test-folder/"
    )

    assert result is True
    assert mock_oci_client.put_object.called


def test_stream_chunks_to_excel_empty_generator(client, mock_oci_client):
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        local_path = tmp.name

    result = client.stream_chunks_to_excel_and_upload(
        empty_chunk_generator(),
        local_file_path=local_path,
        bucket_folder_name="test-folder/"
    )

    assert result is True
    assert mock_oci_client.put_object.called


def test_stream_chunks_to_excel_upload_failure(client, mock_oci_client):
    mock_oci_client.put_object.side_effect = Exception("Upload failed")

    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        local_path = tmp.name

    with pytest.raises(Exception):
        client.stream_chunks_to_excel_and_upload(
            sample_chunk_generator(),
            local_file_path=local_path,
            bucket_folder_name="test-folder/"
        )


def test_stream_chunks_temp_file_cleanup(client, mock_oci_client):
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        local_path = tmp.name

    client.stream_chunks_to_excel_and_upload(
        sample_chunk_generator(),
        local_file_path=local_path,
        bucket_folder_name="test-folder/"
    )

    # File should be deleted in finally block
    assert not os.path.exists(local_path)


def test_create_par_url(client, mock_oci_client):
    mock_response = MagicMock()
    mock_response.data.access_uri = "/p/some-uri"

    mock_oci_client.create_preauthenticated_request.return_value = mock_response

    par_url = client._create_par_url("folder/file.xlsx")

    assert "objectstorage.eu-frankfurt-1.oraclecloud.com" in par_url
    assert "/p/some-uri" in par_url
    assert mock_oci_client.create_preauthenticated_request.called


import pytest
from unittest.mock import patch, MagicMock, mock_open


@patch("code_modules.oci_object_storage.open", new_callable=mock_open, read_data=b"data")
def test_put_file_in_bucket_folder_success(mock_file):
    # Arrange
    client = OCIObjectStorageClient()

    # Mock bucket info
    client.bucket_info = {
        "namespace": "test_namespace",
        "bucket_name": "test_bucket"
    }

    # Mock OCI client
    mock_response = MagicMock()
    client.client = MagicMock()
    client.client.put_object.return_value = mock_response

    # Act
    result = client.put_file_in_bucket_folder(
        "path/to/test_file.txt",
        "folder"
    )

    # Assert
    mock_file.assert_called_once_with("path/to/test_file.txt", "rb")

    client.client.put_object.assert_called_once()
    
    args = client.client.put_object.call_args[0]

    assert args[0] == "test_namespace"
    assert args[1] == "test_bucket"
    assert args[2] == "folder/test_file.txt"   # ✅ object name check

    assert result == mock_response

@patch("code_modules.oci_object_storage.open", new_callable=mock_open, read_data=b"data")
def test_put_file_object_name_extraction(mock_file):
    client = OCIObjectStorageClient()

    client.bucket_info = {
        "namespace": "ns",
        "bucket_name": "bucket"
    }

    client.client = MagicMock()
    client.client.put_object.return_value = "ok"

    client.put_file_in_bucket_folder(
        "/deep/path/myfile.csv",
        "data-folder"
    )

    args = client.client.put_object.call_args[0]

    assert args[2] == "data-folder/myfile.csv"

@patch("code_modules.oci_object_storage.open", side_effect=FileNotFoundError)
def test_put_file_file_not_found(mock_file):
    client = OCIObjectStorageClient()

    with pytest.raises(FileNotFoundError):
        client.put_file_in_bucket_folder("invalid/path.txt", "folder")