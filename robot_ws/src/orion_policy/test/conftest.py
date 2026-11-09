"""Make orion_learning importable when it is not pip-installed (CI: pip install -e learning)."""

import os
import sys

try:
    import orion_learning  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'learning')))
sys.path.insert(0, os.path.dirname(__file__))

# colcon tests packages in parallel: each package that starts rclpy nodes in its unit tests
# needs its own ROS domain or their topics and services cross (orion_policy 75,
# orion_experiments 95).
os.environ['ROS_DOMAIN_ID'] = '75'
