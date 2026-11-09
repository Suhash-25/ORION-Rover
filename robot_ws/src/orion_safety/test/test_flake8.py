import os

from ament_flake8.main import main_with_errors
import pytest

# Shared ORION lint rules (ament defaults minus Q000), see robot_ws/flake8.ini.
CONFIG = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'flake8.ini')


@pytest.mark.flake8
@pytest.mark.linter
def test_flake8():
    rc, errors = main_with_errors(argv=['--config', CONFIG])
    assert rc == 0, 'Found %d code style errors / warnings:\n' % len(errors) + '\n'.join(errors)
