import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
import cv2
import numpy as np


class AutoDriveCV(Node):
    def __init__(self):
        super().__init__('auto_drive_cv')

        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            Image, '/front_camera', self.callback, 10)

        self.cmd_pub = self.create_publisher(
            Twist, '/cmd_vel', 10)

        self.last_log = self.get_clock().now()

        # 黄色ラインから推定するコース幅
        self.lane_width = 448

    def callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        h, w = frame.shape[:2]

        # 黄色をHSV色空間で抽出
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower = np.array([10, 30, 20])
        upper = np.array([50, 255, 255])

        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.medianBlur(mask, 5)

        debug = frame.copy()
        debug[mask > 0] = (0, 0, 255)
        cv2.imwrite('/workspace/yellow_detection.jpg', debug)

        # 画像を遠方・中間・手前の3領域に分割
        bands = [
            (0.48, 0.54),
            (0.54, 0.60),
            (0.60, 0.68)
        ]

        centers = []
        detections = []

        lane_widths = [334, 645, 1050]

        for i, (top, bottom) in enumerate(bands):
            y1 = int(h * top)
            y2 = int(h * bottom)

            band = mask[y1:y2, :]
            left = band[:, :w // 2]
            right = band[:, w // 2:]

            ly, lx = np.where(left > 0)
            ry, rx = np.where(right > 0)

            left_x = None
            right_x = None

            if len(lx) > 30:
                left_x = float(np.mean(lx))

            if len(rx) > 30:
                right_x = float(np.mean(rx)) + w // 2

            # 左右両方のラインがある場合
            if left_x is not None and right_x is not None:
                center = (left_x + right_x) / 2

            # 左側のラインだけの場合
            elif left_x is not None:
                center = left_x + lane_widths[i] / 2

            # 右側のラインだけの場合
            elif right_x is not None:
                center = right_x - lane_widths[i] / 2

            else:
                center = None

            detections.append(
                f'{top:.2f}: L={left_x}, R={right_x}'
            )
            centers.append(center)

        far, middle, near = centers

        # 遠方を重視して進行方向を推定
        valid = []

        if far is not None:
            valid.append((far, 0.5))

        if middle is not None:
            valid.append((middle, 0.3))

        if valid:
            target = sum(c * weight for c, weight in valid)
            total_weight = sum(weight for c, weight in valid)
            target /= total_weight

            error = target - w / 2

            cmd = Twist()
            cmd.linear.x = 1.0
            cmd.angular.z = -0.006 * error

            self.cmd_pub.publish(cmd)

        else:
            error = None
            self.cmd_pub.publish(Twist())

        now = self.get_clock().now()

        if (now - self.last_log).nanoseconds > 1_000_000_000:
            self.get_logger().info(
                f'far={far}, middle={middle}, near={near}, '
                f'error={error}, detections={detections}'
            )
            self.last_log = now


def main():
    rclpy.init()
    node = AutoDriveCV()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
