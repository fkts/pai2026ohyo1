import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np


class AutoDriveYOLO(Node):
    def __init__(self):
        super().__init__('auto_drive_yolo')
        self.bridge = CvBridge()
        self.model = YOLO('/workspace/yolo11n.pt')
        self.sub = self.create_subscription(
            Image, '/front_camera', self.callback,
            qos_profile_sensor_data
        )
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.lane_widths = [334, 645, 1050]
        self.frame_count = 0
        self.get_logger().info('AutoDrive YOLO started')

    def callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        h, w = frame.shape[:2]

        results = self.model(frame, imgsz=320, conf=0.4, verbose=False)
        person_detected = any(
            self.model.names[int(box.cls[0])] == 'person'
            for box in results[0].boxes
        )

        output = results[0].plot()

        if person_detected:
            self.cmd_pub.publish(Twist())
            cv2.imwrite('/workspace/yolo_drive.jpg', output)
            self.get_logger().info('Person detected: STOP')
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv, np.array([10, 30, 20]), np.array([50, 255, 255])
        )
        mask = cv2.medianBlur(mask, 5)

        bands = [(0.48, 0.54), (0.54, 0.60), (0.60, 0.68)]
        centers = []

        for i, (top, bottom) in enumerate(bands):
            band = mask[int(h * top):int(h * bottom), :]
            left = band[:, :w // 2]
            right = band[:, w // 2:]

            _, lx = np.where(left > 0)
            _, rx = np.where(right > 0)

            left_x = float(np.mean(lx)) if len(lx) > 30 else None
            right_x = (
                float(np.mean(rx)) + w // 2
                if len(rx) > 30 else None
            )

            if left_x is not None and right_x is not None:
                center = (left_x + right_x) / 2
            elif left_x is not None:
                center = left_x + self.lane_widths[i] / 2
            elif right_x is not None:
                center = right_x - self.lane_widths[i] / 2
            else:
                center = None

            centers.append(center)

        far, middle, near = centers
        valid = []

        if far is not None:
            valid.append((far, 0.5))
        if middle is not None:
            valid.append((middle, 0.3))

        cmd = Twist()

        if valid:
            target = sum(c * weight for c, weight in valid)
            total_weight = sum(weight for c, weight in valid)
            target /= total_weight
            error = target - w / 2
            cmd.linear.x = 1.0
            cmd.angular.z = -0.006 * error

        self.cmd_pub.publish(cmd)
        cv2.imwrite('/workspace/yolo_drive.jpg', output)

        self.frame_count += 1
        if self.frame_count % 10 == 0:
            self.get_logger().info(
                f'YOLO running, far={far}, middle={middle}'
            )


def main():
    rclpy.init()
    node = AutoDriveYOLO()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
