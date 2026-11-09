"""Own ROS domain for the rclpy node tests (colcon runs packages in parallel)."""

import os

# orion_policy 75, orion_experiments 95, orion_safety 96: a /clock or topic from another
# package's test must not reach these nodes.
os.environ['ROS_DOMAIN_ID'] = '96'
