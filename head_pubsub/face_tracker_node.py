#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32
from geometry_msgs.msg import Point
import cv2
import time
from ultralytics import YOLO
import torch
import numpy as np
import gc

class FaceTrackerNode(Node):
    def __init__(self):
        super().__init__('face_tracker_node')
        
        # Add headless mode parameter (no GUI display)
        self.declare_parameter('headless', False)
        self.headless = self.get_parameter('headless').value
        
        # Initialize camera - try multiple indices
        self.cap = None
        for camera_index in [6, 7]:  # Try /dev/video0, /dev/video6, /dev/video7
            self.get_logger().info(f"Trying to open camera at index {camera_index}...")
            self.cap = cv2.VideoCapture(camera_index)
            if self.cap.isOpened():
                self.get_logger().info(f"Successfully opened camera at index {camera_index}")
                break
            else:
                self.cap.release()
        
        if self.cap is None or not self.cap.isOpened():
            self.get_logger().error("Failed to open camera at any index!")
            self.camera_available = False
            return
        
        self.camera_available = True
            
        # Set camera properties for consistent performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 100)
        
        # Initialize YOLO face detection model
        self.get_logger().info("Loading YOLO face detection model...")
        
        # Clear any existing GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            print(f"Cleared GPU memory. Available memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        else:
            print("CUDA not available, using CPU")
            
        # Load model on CPU first to avoid memory issues
        pkg_share = get_package_share_directory('head_pubsub')
        model_path = os.path.join(pkg_share, 'models', 'yolov12n-face.pt')
        self.get_logger().info(f'Loading YOLO model from: {model_path}')
        self.model = YOLO(model_path)
        
        # Determine device to use
        use_gpu = True

        if use_gpu and torch.cuda.is_available():
            try:
                self.device = 'cuda'
                self.model.to('cuda')
                self.use_fp16 = True  # Enable FP16 on Jetson - faster + less memory
                self.get_logger().info(f"Using GPU: {torch.cuda.get_device_name(0)} (FP16)")

                # Warm up GPU to avoid slow first inference
                self.get_logger().info("Warming up GPU...")
                dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                for _ in range(3):
                    self.model.predict(dummy_frame, verbose=False, device=self.device, half=self.use_fp16)
                self.get_logger().info("GPU warmup complete!")

            except Exception as e:
                self.get_logger().warn(f"Failed to use GPU: {e}, using CPU")
                self.device = 'cpu'
                self.model.to('cpu')
                self.use_fp16 = False
        else:
            self.device = 'cpu'
            self.use_fp16 = False
            self.get_logger().info("Using CPU for inference (reliable and stable)")
        
        # Publishers for direct servo angle control
        self.servo_x_angle_publisher = self.create_publisher(Int32, 'servo_x_angle', 10)
        self.servo_y_angle_publisher = self.create_publisher(Int32, 'servo_y_angle', 10)
        
        # Keep existing publishers for backward compatibility
        self.offset_publisher = self.create_publisher(Point, 'face_offset', 10)
        self.direction_x_publisher = self.create_publisher(String, 'face_direction_x', 10)
        self.direction_y_publisher = self.create_publisher(String, 'face_direction_y', 10)
        self.offset_x_publisher = self.create_publisher(Int32, 'face_offset_x', 10)
        self.offset_y_publisher = self.create_publisher(Int32, 'face_offset_y', 10)
        
        # Timer for processing frames - SLOWER for stability
        self.timer = self.create_timer(0.000001, self.process_frame)  # 10 Hz instead of 30 Hz
        
        # Frame properties
        self.frame_width = 640
        self.frame_height = 480
        self.center_x = self.frame_width // 2
        self.center_y = self.frame_height // 2
        
        # Servo configuration for MOVING CAMERA setup
        self.servo_x_range = [70, 170]   # X servo angle range
        self.servo_y_range = [20, 190]   # Y servo angle range
        
        # Current servo positions - start at center
        self.current_servo_x = 120.0     # Use float for precise calculations
        self.current_servo_y = 100.0
        
        # MUCH more conservative control parameters
        self.deadzone_x = 80             # LARGER deadzone to prevent oscillation
        self.deadzone_y = 60             # LARGER deadzone to prevent oscillation
        self.movement_scale_x = 0.02     # MUCH smaller movement scale
        self.movement_scale_y = 0.015    # MUCH smaller movement scale
        self.max_movement_step = 10       # MUCH smaller maximum step
        
        # Tracking variables
        self.frame_count = 0
        self.last_time = time.time()
        self.face_lost_frames = 0
        self.max_lost_frames = 50        # Longer before returning to center
        
        # Movement damping to prevent oscillation
        self.last_movement_x = 0
        self.last_movement_y = 0
        self.movement_damping = 0.2      # Reduce movement if changing direction
        
        if self.headless:
            self.get_logger().info("Face Tracker Node initialized with STABLE control (HEADLESS mode - no GUI)")
        else:
            self.get_logger().info("Face Tracker Node initialized with STABLE control!")
        
    def process_frame(self):
        """
        Main processing function with STABLE control for moving camera
        """
        if not self.camera_available:
            return
            
        # Capture frame
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to capture frame")
            return
            
        # Update frame properties
        self.frame_height, self.frame_width = frame.shape[:2]
        self.center_x = self.frame_width // 2
        self.center_y = self.frame_height // 2
        
        # Time the inference
        inference_start = time.time()
        
        # Run YOLO face detection
        try:
            results = self.model.predict(frame, verbose=False, conf=0.6, iou=0.5, 
                                       device=self.device, half=self.use_fp16)[0]
        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")
            return
        
        inference_time = time.time() - inference_start
        
        # Calculate face position and servo adjustments
        face_detected, face_x, face_y = self.get_face_position(results, frame)
        
        if face_detected:
            self.face_lost_frames = 0
            
            # Calculate offset from center
            offset_x = face_x - self.center_x
            offset_y = face_y - self.center_y
            
            # Only move if WELL outside deadzone (prevent oscillation)
            movement_x = 0
            movement_y = 0
            
            if abs(offset_x) > self.deadzone_x:
                # FIXED: Move servo in SAME direction as face offset to track the face
                # If face is to the right (+offset_x), move servo right (+movement)
                # If face is to the left (-offset_x), move servo left (-movement)
                movement_x = offset_x * self.movement_scale_x  # REMOVED the negative sign
                
                # Apply damping if changing direction
                if (movement_x > 0 and self.last_movement_x < 0) or (movement_x < 0 and self.last_movement_x > 0):
                    movement_x *= self.movement_damping
                
                # Clip to maximum step
                movement_x = np.clip(movement_x, -self.max_movement_step, self.max_movement_step)
                
                # Apply movement
                self.current_servo_x += movement_x
                self.last_movement_x = movement_x
            else:
                self.last_movement_x = 0
            
            if abs(offset_y) > self.deadzone_y:
                # FIXED: Move servo in SAME direction as face offset to track the face
                # If face is down (+offset_y), move servo down (+movement)
                # If face is up (-offset_y), move servo up (-movement)
                movement_y = offset_y * self.movement_scale_y  # REMOVED the negative sign
                
                # Apply damping if changing direction
                if (movement_y > 0 and self.last_movement_y < 0) or (movement_y < 0 and self.last_movement_y > 0):
                    movement_y *= self.movement_damping
                
                # Clip to maximum step
                movement_y = np.clip(movement_y, -self.max_movement_step, self.max_movement_step)
                
                # Apply movement
                self.current_servo_y += movement_y
                self.last_movement_y = movement_y
            else:
                self.last_movement_y = 0
                
        else:
            # Face lost - increment counter
            self.face_lost_frames += 1
            
            # Very slowly return to center position if face lost for too long
            if self.face_lost_frames > self.max_lost_frames:
                center_x, center_y = 120.0, 110.0
                self.current_servo_x += (center_x - self.current_servo_x) * 0.01  # Very slow return
                self.current_servo_y += (center_y - self.current_servo_y) * 0.01
        
        # Constrain servo positions to valid ranges
        self.current_servo_x = np.clip(self.current_servo_x, self.servo_x_range[0], self.servo_x_range[1])
        self.current_servo_y = np.clip(self.current_servo_y, self.servo_y_range[0], self.servo_y_range[1])
        
        # Publish servo commands and other data
        self.publish_servo_commands()
        self.publish_legacy_data(face_detected, face_x if face_detected else self.center_x, 
                                face_y if face_detected else self.center_y)
        
        # Display frame with annotations
        self.display_frame(frame, face_detected, face_x if face_detected else self.center_x, 
                         face_y if face_detected else self.center_y, inference_time)
        
        # Update statistics
        self.frame_count += 1
        if self.frame_count % 30 == 0:  # Print stats every 30 frames (3 seconds at 10fps)
            current_time = time.time()
            fps = 30 / (current_time - self.last_time)
            self.get_logger().info(f"Tracking at {fps:.1f} FPS | Servo X: {self.current_servo_x:.1f}° Y: {self.current_servo_y:.1f}° | Lost frames: {self.face_lost_frames}")
            self.last_time = current_time
    
    def get_face_position(self, results, frame):
        """
        Extract face position from YOLO results
        
        Returns:
            tuple: (face_detected, face_x, face_y)
        """
        boxes = results.boxes
        
        if boxes is None or len(boxes) == 0:
            return False, self.center_x, self.center_y
        
        # Get the first (most confident) detection
        box = boxes.xywhn[0]  # Normalized coordinates [center_x, center_y, width, height]
        
        # Convert to pixel coordinates
        face_x = int(box[0] * self.frame_width)
        face_y = int(box[1] * self.frame_height)
        
        # Draw visualization
        self.draw_face_detection(frame, box, face_x, face_y)
        
        return True, face_x, face_y
    
    def publish_servo_commands(self):
        """
        Publish direct servo angle commands
        """
        servo_x_msg = Int32()
        servo_y_msg = Int32()
        servo_x_msg.data = int(round(self.current_servo_x))
        servo_y_msg.data = int(round(self.current_servo_y))
        
        self.servo_x_angle_publisher.publish(servo_x_msg)
        self.servo_y_angle_publisher.publish(servo_y_msg)
    
    def publish_legacy_data(self, face_detected, face_x, face_y):
        """
        Publish legacy offset and direction data for backward compatibility
        """
        if face_detected:
            offset_x = face_x - self.center_x
            offset_y = face_y - self.center_y
        else:
            offset_x, offset_y = 0, 0
        
        # Publish offset data
        point_msg = Point()
        point_msg.x = float(offset_x)
        point_msg.y = float(offset_y)
        point_msg.z = 0.0
        self.offset_publisher.publish(point_msg)
        
        # Publish integer offsets
        offset_x_msg = Int32()
        offset_y_msg = Int32()
        offset_x_msg.data = int(offset_x)
        offset_y_msg.data = int(offset_y)
        self.offset_x_publisher.publish(offset_x_msg)
        self.offset_y_publisher.publish(offset_y_msg)
        
        # Publish direction commands based on deadzone
        direction_x_msg = String()
        direction_y_msg = String()
        
        if abs(offset_x) < self.deadzone_x:
            direction_x_msg.data = "0"
        elif offset_x > 0:
            direction_x_msg.data = "+"
        else:
            direction_x_msg.data = "-"
            
        if abs(offset_y) < self.deadzone_y:
            direction_y_msg.data = "0"
        elif offset_y > 0:
            direction_y_msg.data = "+"
        else:
            direction_y_msg.data = "-"
            
        self.direction_x_publisher.publish(direction_x_msg)
        self.direction_y_publisher.publish(direction_y_msg)
    
    def draw_face_detection(self, frame, box, face_x, face_y):
        """
        Draw face detection visualization with deadzone
        """
        # Convert normalized box to pixel coordinates
        width_norm, height_norm = box[2], box[3]
        box_width = int(width_norm * self.frame_width)
        box_height = int(height_norm * self.frame_height)
        
        x1 = face_x - box_width // 2
        y1 = face_y - box_height // 2
        x2 = face_x + box_width // 2
        y2 = face_y + box_height // 2
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (face_x, face_y), 5, (0, 0, 255), -1)
        cv2.circle(frame, (self.center_x, self.center_y), 5, (255, 0, 0), -1)
        cv2.line(frame, (self.center_x, self.center_y), (face_x, face_y), (255, 0, 0), 2)
        
        # Draw deadzone rectangle
        deadzone_x1 = self.center_x - self.deadzone_x
        deadzone_y1 = self.center_y - self.deadzone_y
        deadzone_x2 = self.center_x + self.deadzone_x
        deadzone_y2 = self.center_y + self.deadzone_y
        cv2.rectangle(frame, (deadzone_x1, deadzone_y1), (deadzone_x2, deadzone_y2), (255, 255, 0), 1)
        
        # Draw crosshairs
        cv2.line(frame, (self.center_x - 20, self.center_y), (self.center_x + 20, self.center_y), (255, 255, 255), 1)
        cv2.line(frame, (self.center_x, self.center_y - 20), (self.center_x, self.center_y + 20), (255, 255, 255), 1)
    
    def display_frame(self, frame, face_detected, face_x, face_y, inference_time):
        """
        Display frame with stabilization information (only if not headless)
        """
        if self.headless:
            return  # Skip all GUI display in headless mode
        
        offset_x = face_x - self.center_x
        offset_y = face_y - self.center_y
        
        # Add text overlay
        info_text = [
            f"Face: {'DETECTED' if face_detected else 'SEARCHING'}",
            f"Device: {self.device.upper()} {'FP16' if self.use_fp16 else 'FP32'}",
            f"Servo X: {self.current_servo_x:.1f}°",
            f"Servo Y: {self.current_servo_y:.1f}°",
            f"Face offset: ({offset_x:+4.0f}, {offset_y:+4.0f})",
            f"Deadzone: ±{self.deadzone_x}px, ±{self.deadzone_y}px",
            f"Last move: X={self.last_movement_x:.2f}, Y={self.last_movement_y:.2f}",
            f"Lost frames: {self.face_lost_frames}",
            f"Inference: {inference_time*1000:.1f}ms"
        ]
        
        # Draw text on frame
        y_pos = 30
        for text in info_text:
            cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            y_pos += 18
        
        # Show stabilization status
        if face_detected:
            in_deadzone = abs(offset_x) <= self.deadzone_x and abs(offset_y) <= self.deadzone_y
            status = "LOCKED" if in_deadzone else "TRACKING"
            color = (0, 255, 0) if in_deadzone else (0, 255, 255)
        else:
            status = "SEARCHING"
            color = (0, 0, 255)
            
        cv2.putText(frame, status, (frame.shape[1] - 150, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow("Face Tracker - STABLE", frame)
        cv2.waitKey(1)
    
    def destroy_node(self):
        """
        Cleanup when node is destroyed
        """
        if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
            self.cap.release()
        
        if not self.headless:
            cv2.destroyAllWindows()
            
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = None
    
    try:
        node = FaceTrackerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutting down face tracker...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if node is not None and not node.headless:
            cv2.destroyAllWindows()
        if node is not None:
            node.destroy_node()
        try:
            rclpy.shutdown()
        except:
            pass  # Already shutdown, ignore error

if __name__ == '__main__':
    main()
