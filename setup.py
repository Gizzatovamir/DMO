from setuptools import find_packages, setup
import os
import glob

package_name = 'dmo'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob.glob('launch/*.py')),
        (os.path.join('share', package_name), glob.glob('dmo/*.py')),
        (os.path.join('share', package_name), glob.glob('rviz/*.rviz')),
        (os.path.join('share', package_name, "bag"), glob.glob('bag/*.db3')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='amir',
    maintainer_email='gizzatovamir777@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "lidar_detection_node = dmo.lidar_detection:main",
        ],
    },
)
