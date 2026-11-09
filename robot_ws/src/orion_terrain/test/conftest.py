"""Shared test helpers; Gazebo tests under test/sim/ only run with ORION_RUN_SIM=1."""

import os
from pathlib import Path

import pytest

SIM_DIR = Path(__file__).parent / 'sim'


@pytest.hookimpl(tryfirst=True)
def pytest_ignore_collect(collection_path, config):
    """
    Skip test/sim (boots Gazebo) unless ORION_RUN_SIM=1; colcon test must stay headless.

    tryfirst: launch_testing's own tryfirst hook returns False (not None) for every path,
    which would otherwise end the firstresult hook chain and defeat collect_ignore.
    """
    if os.environ.get('ORION_RUN_SIM') == '1':
        return None
    path = Path(collection_path)
    return True if path == SIM_DIR or SIM_DIR in path.parents else None


class RecordingLogger:
    """Stand-in for a logging.Logger / rclpy logger that keeps the warnings it gets."""

    def __init__(self):
        self.messages = []

    def warning(self, text):
        self.messages.append(text)

    def count(self, fragment):
        return sum(fragment in m for m in self.messages)
