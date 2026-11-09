"""Gazebo launch tests under test/sim/ only run with ORION_RUN_SIM=1 (see test/sim/README)."""

import os

collect_ignore_glob = [] if os.environ.get('ORION_RUN_SIM') == '1' else ['sim/*']


class RecordingLogger:
    """Stand-in for a logging.Logger / rclpy logger that keeps the warnings it gets."""

    def __init__(self):
        self.messages = []

    def warning(self, text):
        self.messages.append(text)

    def count(self, fragment):
        return sum(fragment in m for m in self.messages)
