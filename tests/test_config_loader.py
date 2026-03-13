import pytest
from unittest.mock import MagicMock, patch

from config_loader import load_adw_config, ADWConfig


@patch("config_loader.ConfigParser")
def test_load_adw_config_success(mock_config_parser):
    mock_parser = MagicMock()
    mock_config_parser.return_value = mock_parser

    # Mock parser.get() return values
    mock_parser.get.side_effect = lambda section, key: {
        ("ADW", "config_dir"): "/opt/oracle/config",
        ("ADW", "wallet_loc"): "/opt/oracle/wallet",
        ("ADW", "wallet_pw"): "wallet123",
        ("ADW", "dsn"): "adb_high",
        ("ADW", "USERNAME"): "admin",
        ("ADW", "PASSWORD"): "secret",
    }[(section, key)]

    result = load_adw_config("fake_path.ini")

    assert isinstance(result, ADWConfig)
    assert result.config_dir == "/opt/oracle/config"
    assert result.wallet_loc == "/opt/oracle/wallet"
    assert result.wallet_pw == "wallet123"
    assert result.dsn == "adb_high"
    assert result.username == "admin"
    assert result.password == "secret"

    mock_parser.read.assert_called_once_with("fake_path.ini")


@patch("config_loader.ConfigParser")
def test_load_adw_config_missing_key_raises_exception(mock_config_parser):
    mock_parser = MagicMock()
    mock_config_parser.return_value = mock_parser

    mock_parser.get.side_effect = Exception("Missing config key")

    with pytest.raises(Exception):
        load_adw_config("invalid.ini")
