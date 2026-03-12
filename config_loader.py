"""
ADW Configuration Loader Module.

This module is responsible for loading Oracle Autonomous Data Warehouse (ADW)
connection details from a configuration file and exposing them as a strongly
typed dataclass.

It provides a clean separation between configuration parsing logic and
application code, making it easier to test, maintain, and validate using
tools like SonarQube.
"""
from dataclasses import dataclass
from configparser import ConfigParser

@dataclass
class ADWConfig:
    """
    Dataclass representing ADW connection configuration.

    Attributes:
        config_dir (str): Directory containing Oracle network configuration files.
        wallet_loc (str): Location of the ADW wallet.
        wallet_pw (str): Password for the ADW wallet.
        dsn (str): Database service name (TNS alias).
        username (str): Database username.
        password (str): Database password.
    """
    config_dir: str
    wallet_loc: str
    wallet_pw: str
    dsn: str
    username: str
    password: str


def load_adw_config(path: str = "config.ini") -> ADWConfig:
    """
    Loads ADW configutration from a file.
    """
    parser = ConfigParser()
    parser.read(path)

    return ADWConfig(
        config_dir=parser.get("ADW", "config_dir"),
        wallet_loc=parser.get("ADW", "wallet_loc"),
        wallet_pw=parser.get("ADW", "wallet_pw"),
        dsn=parser.get("ADW", "dsn"),
        username=parser.get("ADW", "USERNAME"),
        password=parser.get("ADW", "PASSWORD"),
    )
