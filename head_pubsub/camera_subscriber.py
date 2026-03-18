#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String
import cv2
from ultralytics import YOLO
import time
from builtin_interfaces.msg import Time as ROSTime
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
import torch

class FaceTracker:
    def __init__(self, yolo_model_path):
        self.model = YOLO(yolo_model_path)
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            self.device = 'cuda'
            self.model.to('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = 'cpu'
            print("CUDA not available, using CPU")
        self.frame_width = 0
        self.frame_height = 0
        self.mid_x = 0
        self.mid_y = 0
        self.bridge = CvBridge()
        self.model_inference_time = 0  # Track model inference time

    def process_frame(self, frame):
        """
        Process a single frame: detect faces and calculate offset from center.
        Draw bounding boxes on the frame.
        :param frame: The image frame in OpenCV format (BGR).
        :return: A tuple (x offset, y offset, processed frame with bounding boxes).
        """
        # Get frame dimensions
        self.frame_height, self.frame_width, _ = frame.shape
        self.mid_x = self.frame_width // 2
        self.mid_y = self.frame_height // 2

        # Run YOLO face detection with timing
        yolo_start_time = time.time()
        results = self.model.predict(frame, verbose=False, conf=0.6, iou=0.5)[0]
        yolo_end_time = time.time()
        self.model_inference_time = yolo_end_time - yolo_start_time

        boxes = results.boxes.xywhn  # Normalized center_x, center_y, width, height

        # Default no-movement
        offset_x, offset_y = 0, 0

        if len(boxes) > 0:
            # Take the first detected face for tracking
            x_center_norm, y_center_norm, width_norm, height_norm = boxes[0]
            x_center = int(x_center_norm * self.frame_width)
            y_center = int(y_center_norm * self.frame_height)

            # Calculate offset from center
            offset_x = self.mid_x - x_center
            offset_y = self.mid_y - y_center

            # Draw bounding box
            x1 = int(x_center - width_norm * self.frame_width / 2)
            y1 = int(y_center - height_norm * self.frame_height / 2)
            x2 = int(x_center + width_norm * self.frame_width / 2)
            y2 = int(y_center + height_norm * self.frame_height / 2)
            color = (0, 255, 0)  # Green color for bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.line(frame, (self.mid_x, self.mid_y), (x_center, y_center), (255, 0, 0), 2)
            cv2.circle(frame, (x_center, y_center), 5, (0, 0, 255), -1)

        return offset_x, offset_y, frame


class FaceDetectionSubscriber(Node):
    def __init__(self):
        super().__init__('face_detection_subscriber')

        # Change QoS profile to prioritize newest messages
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1  # Only keep most recent message
        )

        # Create a subscriber to the 'camera/image' topic
        self.subscription = self.create_subscription(
            Image, 'camera/image', self.image_callback, qos
        )

        # Create a publisher for the custom message to send x/y directions
        self.publisher_x = self.create_publisher(String, 'face_direction_x', 10)
        self.publisher_y = self.create_publisher(String, 'face_direction_y', 10)
        
        self.face_direction_x_msg = String()
        self.face_direction_y_msg = String()

        # Initialize FaceTracker with YOLO model path
        self.face_tracker = FaceTracker("yolov11m-face.pt")
        
        # Timing variables
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.avg_fps = 0
        self.prev_timestamp = None
        self.transport_delay = 0  # Time from camera capture to callback

    def image_callback(self, msg):
        """
        Callback function for the image subscription. Processes the image,
        calculates the offsets, and publishes the result.
        :param msg: The incoming image message.
        """
        try:
            # Timing: Start of callback processing
            callback_start_time = time.time()
            
            # Calculate transport delay (time from camera capture to this callback)
            # Extract timestamp from ROS message header
            msg_sec = msg.header.stamp.sec
            msg_nanosec = msg.header.stamp.nanosec
            msg_timestamp = msg_sec + (msg_nanosec / 1e9)
            
            # Calculate transport delay if we have a valid timestamp
            if msg_timestamp > 0:
                self.transport_delay = callback_start_time - msg_timestamp
            
            # Calculate FPS (frames per second)
            current_time = time.time()
            elapsed = current_time - self.last_frame_time
            self.last_frame_time = current_time
            
            # Update rolling average FPS
            self.frame_count += 1
            if self.frame_count > 10:  # Start averaging after 10 frames
                if self.avg_fps == 0:
                    self.avg_fps = 1.0 / elapsed if elapsed > 0 else 0
                else:
                    self.avg_fps = 0.9 * self.avg_fps + 0.1 * (1.0 / elapsed if elapsed > 0 else 0)

            # Convert ROS Image message to OpenCV format
            frame = self.face_tracker.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Process the frame and get the x, y offset values
            processing_start = time.time()
            offset_x, offset_y, processed_frame = self.face_tracker.process_frame(frame)
            processing_time = time.time() - processing_start

            # Determine the direction for x and y
            self.face_direction_x_msg.data = "+" if offset_x < -50 else "-" if offset_x > 50 else "0"
            self.face_direction_y_msg.data = "+" if offset_y < -50 else "-" if offset_y > 50 else "0"

            # Publish the direction message
            self.publisher_x.publish(self.face_direction_x_msg)
            self.publisher_y.publish(self.face_direction_y_msg)
            
            # Calculate total callback processing time
            total_callback_time = time.time() - callback_start_time
            
            # Add timing information to the image
            timing_info = [
                f"FPS: {self.avg_fps:.1f}",
                f"Transport Delay: {self.transport_delay*1000:.1f} ms",
                f"YOLO Time: {self.face_tracker.model_inference_time*1000:.1f} ms",
                f"Processing Time: {processing_time*1000:.1f} ms",
                f"Total Callback: {total_callback_time*1000:.1f} ms",
                f"Frame-to-Frame Time: {(elapsed*1000):.1f} ms",
            ]
            
            y_pos = 30
            for info in timing_info:
                cv2.putText(processed_frame, info, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_pos += 25
            
            # Display the processed frame with bounding boxes and timing info
            cv2.imshow("Face Detection", processed_frame)
            cv2.waitKey(1)  # Required for OpenCV window updates

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = FaceDetectionSubscriber()

    rclpy.spin(node)

    # Gracefully shutdown and close OpenCV windows
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()