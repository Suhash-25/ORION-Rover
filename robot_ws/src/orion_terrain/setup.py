from glob import glob

from setuptools import find_packages, setup

package_name = 'orion_terrain'

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
    description='ORION terrain estimator (slip, body speed, sinkage proxy).',
    license='MIT',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'terrain_node = orion_terrain.terrain_node:main',
            'fit_effort_model = orion_terrain.fit_effort_model:main',
        ],
    },
)
