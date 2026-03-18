# head_pubsub

## Overview
`head_pubsub` is a ROS 2 Python package for camera-based face tracking. It runs YOLO face detection and publishes head control topics for downstream servo/micro-ROS controllers.

## Repository Structure
- `head_pubsub/face_tracker_node.py` – main face tracker node (YOLO + servo command publishing)
- `head_pubsub/camera_publisher.py` – camera image publisher node
- `head_pubsub/camera_subscriber.py` – camera subscriber + face tracking path
- `setup.py`, `package.xml` – package metadata and entry points
- `test/` – lint and style tests