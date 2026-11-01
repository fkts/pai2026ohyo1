import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2

class YoloCamera(Node):
    def __init__(self):
        super().__init__('yolo_camera')
        self.bridge = CvBridge()
        self.model = YOLO('/workspace/yolo11n.pt')
        self.sub = self.create_subscription(
            Image, '/front_camera', self.callback,
            qos_profile_sensor_data
        )
        self.frame_count = 0
        self.get_logger().info('YOLO camera started')

    def callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding='bgr8'
        )
        results = self.model(
            frame, imgsz=320, conf=0.25, verbose=False
        )
        self.frame_count += 1

        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                name = self.model.names[cls]
                conf = float(box.conf[0])
                self.get_logger().info(
                    f'Detected: {name}, confidence={conf:.2f}'
                )
        elif self.frame_count % 10 == 0:
            self.get_logger().info('No objects detected')

        image = results[0].plot()
        cv2.imwrite('/workspace/yolo_live.jpg', image)

def main():
    rclpy.init()
    node = YoloCamera()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
