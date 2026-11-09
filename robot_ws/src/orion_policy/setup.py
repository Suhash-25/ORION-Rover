from glob import glob

from setuptools import find_packages
from setuptools import setup

package_name = 'orion_policy'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'requirements.txt']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Suhas',
    maintainer_email='suhasacharya0525@gmail.com',
    description='ORION controller runtime (B0, ONNX policy, critic gating, escape).',
    license='MIT',
    # colcon only finds pytest through extras_require here (tests_require is ignored).
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'policy_node = orion_policy.policy_node:main',
        ],
    },
)
