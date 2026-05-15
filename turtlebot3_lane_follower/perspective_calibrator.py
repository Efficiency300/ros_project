#!/usr/bin/env python3
#
# Interactive perspective-transform calibrator.
#
# Two OpenCV windows open side-by-side:
#   Left  — raw camera image with 4 draggable corner points
#   Right — bird's-eye warp result (updates live as you drag)
#
# Controls:
#   Click + drag   move a corner point
#   s              save perspective_src to lane_detection.yaml
#   r              reload values from YAML (discard unsaved changes)
#   q / ESC        quit
#
# Point order matches lane_detection.yaml / lane_detector_node.py:
#   BL (green)   — bottom-left
#   BR (red)     — bottom-right
#   TR (magenta) — top-right
#   TL (cyan)    — top-left

import os
import threading

import cv2
import numpy as np
import rclpy
import rclpy.executors
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

# ── visual constants ──────────────────────────────────────────────────────────
_COLOURS = [
    (0,   255,   0),   # BL  green
    (0,   0,   255),   # BR  red
    (255,  0,  255),   # TR  magenta
    (255, 255,   0),   # TL  cyan
]
_LABELS    = ['BL', 'BR', 'TR', 'TL']
_SNAP_PX   = 20   # pixels — max distance to grab a point
_WIN_CAM   = 'Perspective Calibrator  (drag points)'
_WIN_WARP  = "Bird's-Eye View  (live)"
_WIN_MASK  = 'Yellow Mask  (HSV result)'
_WIN_HSV   = 'HSV Tuner'


# ── YAML helpers ──────────────────────────────────────────────────────────────

def _yaml_path() -> str:
    share = get_package_share_directory('turtlebot3_lane_follower')
    return os.path.join(share, 'param', 'lane_detection.yaml')


def _read_pts(path: str):
    """Return [[BL],[BR],[TR],[TL]] parsed from perspective_src line."""
    with open(path) as f:
        for line in f:
            if 'perspective_src' in line and '[' in line:
                vals = list(map(int,
                    line.split('[')[1].split(']')[0].split(',')))
                return [[vals[0], vals[1]], [vals[2], vals[3]],
                        [vals[4], vals[5]], [vals[6], vals[7]]]
    return None


def _read_hsv(path: str) -> dict:
    """Return dict with yellow_h/s/v_min/max from YAML."""
    keys = ['yellow_h_min', 'yellow_h_max',
            'yellow_s_min', 'yellow_s_max',
            'yellow_v_min', 'yellow_v_max']
    result = {'yellow_h_min': 20, 'yellow_h_max': 35,
              'yellow_s_min': 100, 'yellow_s_max': 255,
              'yellow_v_min': 100, 'yellow_v_max': 255}
    with open(path) as f:
        for line in f:
            for k in keys:
                if line.strip().startswith(k + ':'):
                    try:
                        result[k] = int(line.split(':')[1].strip())
                    except ValueError:
                        pass
    return result


def _write_yaml(path: str, pts: list, hsv: dict) -> None:
    """Write perspective_src and all HSV keys back to YAML in-place."""
    flat = [pts[0][0], pts[0][1], pts[1][0], pts[1][1],
            pts[2][0], pts[2][1], pts[3][0], pts[3][1]]
    replacements = {
        'perspective_src': f'    perspective_src: {flat}\n',
        'yellow_h_min':    f'    yellow_h_min: {hsv["yellow_h_min"]}\n',
        'yellow_h_max':    f'    yellow_h_max: {hsv["yellow_h_max"]}\n',
        'yellow_s_min':    f'    yellow_s_min: {hsv["yellow_s_min"]}\n',
        'yellow_s_max':    f'    yellow_s_max: {hsv["yellow_s_max"]}\n',
        'yellow_v_min':    f'    yellow_v_min: {hsv["yellow_v_min"]}\n',
        'yellow_v_max':    f'    yellow_v_max: {hsv["yellow_v_max"]}\n',
    }
    with open(path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        for key, new_line in replacements.items():
            if stripped.startswith(key + ':'):
                lines[i] = new_line
                break
    with open(path, 'w') as f:
        f.writelines(lines)


# ── ROS 2 node ────────────────────────────────────────────────────────────────

class PerspectiveCalibrator(Node):

    def __init__(self):
        super().__init__('perspective_calibrator')
        self._yaml = _yaml_path()

        self.pts = _read_pts(self._yaml) or \
                   [[60, 240], [260, 240], [190, 155], [130, 155]]
        self._hsv = _read_hsv(self._yaml)

        self._frame: np.ndarray | None = None
        self._lock  = threading.Lock()
        self._drag  = -1
        self._status = ''
        self._status_frames = 0

        self._sub = self.create_subscription(
            Image, '/camera/image_raw', self._img_cb, qos_profile_sensor_data)

        for win in (_WIN_CAM, _WIN_WARP, _WIN_MASK, _WIN_HSV):
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(_WIN_CAM, self._mouse_cb)
        self._create_trackbars()

        self.get_logger().info('Calibrator ready  —  YAML: ' + self._yaml)
        self.get_logger().info('  Drag trapezoid points in window 1.')
        self.get_logger().info('  Adjust HSV sliders until mask shows solid white lines.')
        self.get_logger().info('  Press "s" to save both to lane_detection.yaml.')

    # ── trackbars ─────────────────────────────────────────────────────────

    def _create_trackbars(self) -> None:
        h = self._hsv
        cv2.createTrackbar('H  min', _WIN_HSV, h['yellow_h_min'], 179, lambda v: None)
        cv2.createTrackbar('H  max', _WIN_HSV, h['yellow_h_max'], 179, lambda v: None)
        cv2.createTrackbar('S  min', _WIN_HSV, h['yellow_s_min'], 255, lambda v: None)
        cv2.createTrackbar('S  max', _WIN_HSV, h['yellow_s_max'], 255, lambda v: None)
        cv2.createTrackbar('V  min', _WIN_HSV, h['yellow_v_min'], 255, lambda v: None)
        cv2.createTrackbar('V  max', _WIN_HSV, h['yellow_v_max'], 255, lambda v: None)

    def _read_trackbars(self) -> dict:
        return {
            'yellow_h_min': cv2.getTrackbarPos('H  min', _WIN_HSV),
            'yellow_h_max': cv2.getTrackbarPos('H  max', _WIN_HSV),
            'yellow_s_min': cv2.getTrackbarPos('S  min', _WIN_HSV),
            'yellow_s_max': cv2.getTrackbarPos('S  max', _WIN_HSV),
            'yellow_v_min': cv2.getTrackbarPos('V  min', _WIN_HSV),
            'yellow_v_max': cv2.getTrackbarPos('V  max', _WIN_HSV),
        }

    # ── image callback ────────────────────────────────────────────────────

    def _img_cb(self, msg: Image) -> None:
        try:
            if msg.encoding.lower() == 'nv21':
                yuv = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    (msg.height * 3 // 2, msg.width))
                frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
            else:
                from cv_bridge import CvBridge
                frame = CvBridge().imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # frame = cv2.rotate(frame, cv2.ROTATE_180)
        except Exception as exc:
            self.get_logger().error(str(exc))
            return
        with self._lock:
            self._frame = frame

    # ── mouse callback ────────────────────────────────────────────────────

    def _mouse_cb(self, event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            dists = [np.hypot(x - p[0], y - p[1]) for p in self.pts]
            idx = int(np.argmin(dists))
            if dists[idx] < _SNAP_PX:
                self._drag = idx
        elif event == cv2.EVENT_MOUSEMOVE and self._drag >= 0:
            self.pts[self._drag] = [x, y]
        elif event == cv2.EVENT_LBUTTONUP:
            self._drag = -1

    # ── drawing helpers ───────────────────────────────────────────────────

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        vis = frame.copy()

        # trapezoid outline  BL→BR→TR→TL→BL
        poly = np.array(self.pts, dtype=np.int32)
        cv2.polylines(vis, [poly], isClosed=True, color=(0, 255, 255), thickness=2)

        # corner points + labels
        for pt, col, lbl in zip(self.pts, _COLOURS, _LABELS):
            cv2.circle(vis, tuple(pt), 8, col, -1)
            cv2.circle(vis, tuple(pt), 8, (255, 255, 255), 1)
            cv2.putText(vis, lbl, (pt[0] + 10, pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

        # key hints
        cv2.putText(vis, 's=save   r=reload   q=quit',
                    (5, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # status message (shown for a few frames after save)
        if self._status:
            cv2.putText(vis, self._status, (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 1)
        return vis

    def _warp(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        src = np.float32(self.pts)
        dst = np.float32([[0, h], [w, h], [w, 0], [0, 0]])
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(frame, M, (w, h))

    def _yellow_mask(self, warped: np.ndarray, hsv_vals: dict) -> np.ndarray:
        """Return colour-annotated mask: detected pixels as cyan on dim background."""
        lower = np.array([hsv_vals['yellow_h_min'],
                          hsv_vals['yellow_s_min'],
                          hsv_vals['yellow_v_min']], dtype=np.uint8)
        upper = np.array([hsv_vals['yellow_h_max'],
                          hsv_vals['yellow_s_max'],
                          hsv_vals['yellow_v_max']], dtype=np.uint8)
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        vis = (warped * 0.3).astype(np.uint8)
        vis[mask == 255] = (0, 220, 220)   # cyan where yellow detected

        h, w = vis.shape[:2]
        pct = 100.0 * np.count_nonzero(mask) / mask.size
        cv2.putText(vis, f'detected: {pct:.1f}%',
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis,
                    f'H [{hsv_vals["yellow_h_min"]}-{hsv_vals["yellow_h_max"]}]  '
                    f'S [{hsv_vals["yellow_s_min"]}-{hsv_vals["yellow_s_max"]}]  '
                    f'V [{hsv_vals["yellow_v_min"]}-{hsv_vals["yellow_v_max"]}]',
                    (5, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        return vis

    # ── GUI tick (called from main loop) ─────────────────────────────────

    def tick(self) -> bool:
        with self._lock:
            frame = self._frame.copy() if self._frame is not None else None

        if frame is None:
            cv2.waitKey(50)
            return True

        hsv_vals = self._read_trackbars()
        warped   = self._warp(frame)

        cv2.imshow(_WIN_CAM,  self._draw_overlay(frame))
        cv2.imshow(_WIN_WARP, warped)
        cv2.imshow(_WIN_MASK, self._yellow_mask(warped, hsv_vals))

        if self._status_frames > 0:
            self._status_frames -= 1
            if self._status_frames == 0:
                self._status = ''

        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), 27):
            return False
        elif key == ord('s'):
            _write_yaml(self._yaml, self.pts, hsv_vals)
            flat = [c for pt in self.pts for c in pt]
            self._status = 'Saved!'
            self._status_frames = 60
            self.get_logger().info(f'Saved perspective_src: {flat}')
            self.get_logger().info(
                f'Saved HSV: H[{hsv_vals["yellow_h_min"]}-{hsv_vals["yellow_h_max"]}] '
                f'S[{hsv_vals["yellow_s_min"]}-{hsv_vals["yellow_s_max"]}] '
                f'V[{hsv_vals["yellow_v_min"]}-{hsv_vals["yellow_v_max"]}]'
            )
        elif key == ord('r'):
            pts = _read_pts(self._yaml)
            if pts:
                self.pts = pts
            hsv = _read_hsv(self._yaml)
            cv2.setTrackbarPos('H  min', _WIN_HSV, hsv['yellow_h_min'])
            cv2.setTrackbarPos('H  max', _WIN_HSV, hsv['yellow_h_max'])
            cv2.setTrackbarPos('S  min', _WIN_HSV, hsv['yellow_s_min'])
            cv2.setTrackbarPos('S  max', _WIN_HSV, hsv['yellow_s_max'])
            cv2.setTrackbarPos('V  min', _WIN_HSV, hsv['yellow_v_min'])
            cv2.setTrackbarPos('V  max', _WIN_HSV, hsv['yellow_v_max'])
            self._status = 'Reloaded'
            self._status_frames = 60
            self.get_logger().info('Reloaded from YAML.')

        return True


# ── entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = PerspectiveCalibrator()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.01)
            if not node.tick():
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
