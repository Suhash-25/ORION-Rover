from glob import glob

from setuptools import find_packages
from setuptools import setup

package_name = 'orion_safety'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Suhas',
    maintainer_email='suhasacharya0525@gmail.com',
    description='ORION safety supervisor: tilt, current, heartbeat, geofence, stuck, e-stop.',
    license='MIT',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'safety_node = orion_safety.safety_node:main',
        ],
    },
)
