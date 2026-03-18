#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')

        # Create a publisher for the 'camera/image' topic
        self.publisher = self.create_publisher(Image, 'camera/image', 10)  # 10 is the QoS value

        # Set up a timer to publish images at a fixed rate
        self.timer = self.create_timer(0.001, self.timer_callback)  # Run every 0.1 seconds (10 Hz)
        
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(0)  # Open the first camera device

        if not self.cap.isOpened():
            self.get_logger().error("Failed to open the camera.")
            exit(1)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert the OpenCV frame to a ROS Image message
            image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            
            # Set the timestamp to current time
            now = self.get_clock().now()
            image_msg.header.stamp = now.to_msg()
            
            # Publish the image message to the 'camera/image' topic
            self.publisher.publish(image_msg)

def main(args=None):
    rclpy.init(args=args)
    camera_publisher = CameraPublisher()

    # Spin the node to keep it running and handle the timer callbacks
    rclpy.spin(camera_publisher)

    camera_publisher.destroy_node()
    cv2.destroyAllWindows()  # Close any OpenCV windows
    rclpy.shutdown()

if __name__ == '__main__':
    main()
