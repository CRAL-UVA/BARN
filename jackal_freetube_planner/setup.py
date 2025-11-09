from setuptools import setup

package_name = 'jackal_freetube_planner'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='Donggen Li',
    maintainer_email='eaz7wk@virginia.edu',
    description='Free-space motion tube planner for Jackal (ROS2)',
    license='BSD-3-Clause',
    entry_points={
        'console_scripts': [
            'fixed_granular_ros2 = jackal_freetube_planner.fixed_granular_ros2:main',
        ],
    },
)
