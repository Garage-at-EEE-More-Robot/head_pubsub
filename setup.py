from setuptools import find_packages, setup

package_name = 'head_pubsub'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/models', ['yolov11m-face.pt', 'yolov12n-face.pt']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='redranger',
    maintainer_email='frentzenseow001@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'face_tracker_node = head_pubsub.face_tracker_node:main',
        ],
    },
)
