from glob import glob

from setuptools import find_packages
from setuptools import setup

package_name = 'orion_experiments'

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
    description='ORION session plans, trial runner and pre-registered analysis.',
    license='MIT',
    # colcon only finds pytest through extras_require here (tests_require is ignored).
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'trial_runner = orion_experiments.trial_runner:main',
            'plan_session = orion_experiments.protocol:main',
            'analyse_trials = orion_experiments.analysis:main',
            'phase0_power = orion_experiments.power:main',
            'imu_check = orion_experiments.imu_check:main',
        ],
    },
)
