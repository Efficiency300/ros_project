#!/usr/bin/env python3
#
# Copyright 2026 - TurtleBot3 Yellow Lane Follower
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import deque

import cv2
import numpy as np

from enum import IntEnum

from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Int8


class ZoneState(IntEnum):
    NORMAL     = 0   # following yellow lines normally
    WHITE_ZONE = 1   # white dashed lines detected, yellow absent
    TURN_LEFT  = 2   # white zone ended, controller should turn left
    PARKING    = 3   # red marker dominates view → stop on it


class LaneDetectorNode(Node):
    """
    Detects two yellow lane lines from the TurtleBot3 camera using:
    - Bird's-eye view perspective warp
    - CLAHE lighting normalisation
    - Yellow HSV colour masking
    - Morphological noise removal
    - Prior-based sliding window tracking
    - 2nd-order polynomial lane fitting
    - Adaptive lookahead midpoint calculation
    - Exponential smoothing on the output error

    Both lane lines are yellow; they are separated spatially (left/right
    halves of the warped image) rather than by colour.

    Published Topics
    ----------------
    /lane/center_error  (std_msgs/Float32)
        Normalised lateral error [-1, 1].  0 = centred.
        NaN published when the lane is completely lost.
    /lane/image_projected   (sensor_msgs/Image)  [debug only]
    /lane/image_yellow_mask (sensor_msgs/Image)  [debug only]
    /lane/image_lane_debug  (sensor_msgs/Image)  [debug only]

    Parameters (see param/lane_detection.yaml for defaults and tuning notes)
    ----------
    resize_factor, debug_mode,
    perspective_src (8 ints, original-image coords: BL,BR,TR,TL x,y pairs),
    yellow_h/s/v_min/max,
    n_windows, window_margin, min_pixels, min_lane_pixels, min_valid_windows,
    prior_expiry_frames, alpha_smooth, base_lookahead, cold_start_frames
    """

    def __init__(self):
        super().__init__('lane_detector_node')

        # ── parameters ────────────────────────────────────────────────────
        self.declare_parameter('resize_factor', 0.5)
        self.declare_parameter('debug_mode', False)

        # perspective_src: 8 integers defining the trapezoid in the
        # ORIGINAL (pre-resize) camera image, order:
        #   bottom-left-x, bottom-left-y,
        #   bottom-right-x, bottom-right-y,
        #   top-right-x, top-right-y,
        #   top-left-x, top-left-y
        # Default tuned for TurtleBot3 Burger default camera position, 640x480.
        # MUST be recalibrated using /lane/image_lane_debug + rqt.
        self.declare_parameter('perspective_src', [
            120, 480,
            520, 480,
            380, 310,
            260, 310,
        ])

        # Yellow HSV range
        self.declare_parameter('yellow_h_min', 20)
        self.declare_parameter('yellow_h_max', 35)
        self.declare_parameter('yellow_s_min', 100)
        self.declare_parameter('yellow_s_max', 255)
        self.declare_parameter('yellow_v_min', 100)
        self.declare_parameter('yellow_v_max', 255)

        # Sliding window
        self.declare_parameter('n_windows', 10)
        self.declare_parameter('window_margin', 60)
        self.declare_parameter('min_pixels', 50)
        self.declare_parameter('min_lane_pixels', 200)
        self.declare_parameter('min_valid_windows', 4)
        self.declare_parameter('prior_expiry_frames', 10)

        # Error control
        self.declare_parameter('alpha_smooth', 0.5)
        self.declare_parameter('base_lookahead', 0.6)
        self.declare_parameter('cold_start_frames', 5)

        # Camera blind zone compensation.
        self.declare_parameter('bottom_crop_fraction', 0.0)

        # ── White line / special zone detection ───────────────────────────
        # White lines in HSV: any hue, low saturation, high brightness.
        self.declare_parameter('white_s_max', 50)
        self.declare_parameter('white_v_min', 160)
        # yellow_lost_thresh: yellow pixel count below this → yellow "gone"
        self.declare_parameter('yellow_lost_thresh', 400)
        # white_found_thresh: white pixel count above this → white "present"
        self.declare_parameter('white_found_thresh', 200)
        # Consecutive frames required to confirm zone entry / exit
        self.declare_parameter('white_zone_entry_frames', 6)
        self.declare_parameter('white_zone_exit_frames',  8)

        # ── Red parking marker detection ──────────────────────────────────
        # Red wraps around H=0/179 so two HSV ranges are needed.
        self.declare_parameter('red_h1_max', 10)
        self.declare_parameter('red_h2_min', 160)
        self.declare_parameter('red_s_min', 120)
        self.declare_parameter('red_v_min', 80)
        # red pixel count above which red marker is considered "in view"
        self.declare_parameter('red_found_thresh', 1500)
        # consecutive frames required to confirm parking entry
        self.declare_parameter('parking_entry_frames', 5)
        # frames at start during which red is ignored (avoid latching on the
        # start marker right under the robot).
        self.declare_parameter('parking_arm_frames', 80)

        # ── Polynomial sanity (reject "lane goes sideways" fits) ──────────
        # Reject a fit if the lane's tangent at the bottom is more horizontal
        # than this many image-x-pixels per image-y-pixel.  Stops the loop /
        # wrap-around backwards-driving symptom.
        self.declare_parameter('max_bottom_slope', 4.0)

        self._read_parameters()

        # ── state ─────────────────────────────────────────────────────────
        self.bridge = CvBridge()
        self.frame_count = 0

        # Window seeds (in resized-image coords)
        self.prev_left_x = None
        self.prev_right_x = None

        # Best available lane polynomials (may be stale)
        self.prev_left_poly = None
        self.prev_right_poly = None

        # Staleness counters – when either reaches prior_expiry_frames
        # the corresponding prior is discarded so fallback logic activates
        self.left_stale_frames = 0
        self.right_stale_frames = 0

        # Total frames with no usable poly for both lanes
        self.total_lost_frames = 0

        # Rolling average of valid lane widths (resized px)
        self.lane_width_history = deque(maxlen=30)

        # Smoothed error state
        self.smoothed_error = 0.0

        # Zone state machine
        self.zone_state        = ZoneState.NORMAL
        self._white_entry_cnt  = 0   # consecutive frames with white detected
        self._no_line_cnt      = 0   # consecutive frames with no lines (in WHITE_ZONE)
        self._red_entry_cnt    = 0   # consecutive frames with red marker visible

        # ── publishers ────────────────────────────────────────────────────
        qos = QoSProfile(depth=10)
        self.error_pub = self.create_publisher(Float32, '/lane/center_error', qos)
        dbg_qos = QoSProfile(depth=1)
        self.img_proj_pub = self.create_publisher(Image, '/lane/image_projected', dbg_qos)
        self.img_mask_pub = self.create_publisher(Image, '/lane/image_yellow_mask', dbg_qos)
        self.img_dbg_pub = self.create_publisher(Image, '/lane/image_lane_debug', dbg_qos)

        self.zone_pub = self.create_publisher(Int8, '/lane/zone_state',
                                               QoSProfile(depth=10))

        # ── subscriber (best-effort, depth=1 → always latest frame) ──────
        self.img_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info('LaneDetectorNode started')
        self.get_logger().info(
            f'  resize_factor={self.resize_factor}  debug_mode={self.debug_mode}'
        )

    # ─────────────────────────────────────────────────────────────────────
    # Parameter helpers
    # ─────────────────────────────────────────────────────────────────────

    def _read_parameters(self):
        self.resize_factor = self.get_parameter('resize_factor').value
        self.debug_mode = self.get_parameter('debug_mode').value

        src_flat = list(self.get_parameter('perspective_src').value)
        # perspective_src is in original image coords; we scale by resize_factor
        # so the warp is applied after resize.
        self.perspective_src_orig = np.float32([
            [src_flat[0], src_flat[1]],
            [src_flat[2], src_flat[3]],
            [src_flat[4], src_flat[5]],
            [src_flat[6], src_flat[7]],
        ])

        self.yellow_lower = np.array([
            self.get_parameter('yellow_h_min').value,
            self.get_parameter('yellow_s_min').value,
            self.get_parameter('yellow_v_min').value,
        ], dtype=np.uint8)
        self.yellow_upper = np.array([
            self.get_parameter('yellow_h_max').value,
            self.get_parameter('yellow_s_max').value,
            self.get_parameter('yellow_v_max').value,
        ], dtype=np.uint8)

        self.n_windows = int(self.get_parameter('n_windows').value)
        self.window_margin = int(self.get_parameter('window_margin').value)
        self.min_pixels = int(self.get_parameter('min_pixels').value)
        self.min_lane_pixels = int(self.get_parameter('min_lane_pixels').value)
        self.min_valid_windows = int(self.get_parameter('min_valid_windows').value)
        self.prior_expiry_frames = int(self.get_parameter('prior_expiry_frames').value)

        self.alpha_smooth = float(self.get_parameter('alpha_smooth').value)
        self.base_lookahead = float(self.get_parameter('base_lookahead').value)
        self.cold_start_frames = int(self.get_parameter('cold_start_frames').value)
        self.bottom_crop_fraction = float(
            self.get_parameter('bottom_crop_fraction').value)

        self.white_s_max           = int(self.get_parameter('white_s_max').value)
        self.white_v_min           = int(self.get_parameter('white_v_min').value)
        self.yellow_lost_thresh    = int(self.get_parameter('yellow_lost_thresh').value)
        self.white_found_thresh    = int(self.get_parameter('white_found_thresh').value)
        self.white_zone_entry_frames = int(
            self.get_parameter('white_zone_entry_frames').value)
        self.white_zone_exit_frames  = int(
            self.get_parameter('white_zone_exit_frames').value)

        self.red_h1_max = int(self.get_parameter('red_h1_max').value)
        self.red_h2_min = int(self.get_parameter('red_h2_min').value)
        self.red_s_min  = int(self.get_parameter('red_s_min').value)
        self.red_v_min  = int(self.get_parameter('red_v_min').value)
        self.red_found_thresh    = int(self.get_parameter('red_found_thresh').value)
        self.parking_entry_frames = int(self.get_parameter('parking_entry_frames').value)
        self.parking_arm_frames   = int(self.get_parameter('parking_arm_frames').value)
        self.max_bottom_slope = float(self.get_parameter('max_bottom_slope').value)

    # ─────────────────────────────────────────────────────────────────────
    # Main image callback
    # ─────────────────────────────────────────────────────────────────────

    def image_callback(self, msg: Image):
        try:
            if msg.encoding.lower() == 'nv21':
                yuv = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    (msg.height * 3 // 2, msg.width))
                frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
            else:
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'cv_bridge: {exc}')
            return

        # frame = cv2.rotate(frame, cv2.ROTATE_180)

        self.frame_count += 1

        # ── 1. Resize ──────────────────────────────────────────────────────
        if self.resize_factor != 1.0:
            frame = cv2.resize(
                frame, None,
                fx=self.resize_factor, fy=self.resize_factor,
                interpolation=cv2.INTER_LINEAR,
            )
        h, w = frame.shape[:2]

        # ── 2. CLAHE on L channel (lighting normalisation) ─────────────────
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_ch = clahe.apply(l_ch)
        frame_eq = cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2BGR)

        # ── 3. Perspective warp (bird's-eye view) ──────────────────────────
        # Scale src points from original-image coords to resized-image coords
        src = self.perspective_src_orig * self.resize_factor
        dst = np.float32([
            [0,   h],
            [w,   h],
            [w,   0],
            [0,   0],
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(frame_eq, M, (w, h))

        # ── 4. Yellow HSV mask ─────────────────────────────────────────────
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)

        # ── 5. Morphological opening (remove small specular blobs) ─────────
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # ── 5b. Bottom crop (camera blind zone compensation) ──────────────
        crop_rows = int(h * self.bottom_crop_fraction)
        if crop_rows > 0:
            mask[-crop_rows:, :] = 0
        effective_h = h - crop_rows

        # ── 5c. White line mask + zone-state pixel counts ──────────────────
        yellow_px = int(np.count_nonzero(mask))
        hsv_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(
            hsv_warped,
            np.array([0,   0,              self.white_v_min], dtype=np.uint8),
            np.array([179, self.white_s_max, 255           ], dtype=np.uint8),
        )
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        if crop_rows > 0:
            white_mask[-crop_rows:, :] = 0
        white_px = int(np.count_nonzero(white_mask))

        # Red (wraps around H=0/179, so OR of two ranges).
        red_lower1 = np.array([0,                 self.red_s_min, self.red_v_min], dtype=np.uint8)
        red_upper1 = np.array([self.red_h1_max,   255,            255           ], dtype=np.uint8)
        red_lower2 = np.array([self.red_h2_min,   self.red_s_min, self.red_v_min], dtype=np.uint8)
        red_upper2 = np.array([179,               255,            255           ], dtype=np.uint8)
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv_warped, red_lower1, red_upper1),
            cv2.inRange(hsv_warped, red_lower2, red_upper2),
        )
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        if crop_rows > 0:
            red_mask[-crop_rows:, :] = 0
        red_px = int(np.count_nonzero(red_mask))

        # In WHITE_ZONE: combine yellow + white so either colour is tracked
        if self.zone_state == ZoneState.WHITE_ZONE:
            mask = cv2.bitwise_or(mask, white_mask)

        # ── 6. Find lane base positions ────────────────────────────────────
        left_x_base, right_x_base = self._find_lane_base(mask, w, effective_h)

        # ── 7. Sliding window search ────────────────────────────────────────
        left_cents, left_pixels, left_valid_wins = self._sliding_windows(
            mask, effective_h, left_x_base)
        right_cents, right_pixels, right_valid_wins = self._sliding_windows(
            mask, effective_h, right_x_base)

        # ── 8. Polynomial fitting ───────────────────────────────────────────
        left_poly, left_fresh = self._fit_polynomial(
            left_cents, left_pixels, left_valid_wins,
            self.prev_left_poly, effective_h)
        right_poly, right_fresh = self._fit_polynomial(
            right_cents, right_pixels, right_valid_wins,
            self.prev_right_poly, effective_h)

        # ── 9. Update state from fresh detections ──────────────────────────
        if left_fresh:
            self.left_stale_frames = 0
            self.prev_left_poly = left_poly
            self.prev_left_x = int(np.clip(
                np.polyval(left_poly, h * 0.9), 0, w - 1))
        else:
            self.left_stale_frames += 1
            if self.left_stale_frames > self.prior_expiry_frames:
                self.prev_left_poly = None
                self.prev_left_x = None

        if right_fresh:
            self.right_stale_frames = 0
            self.prev_right_poly = right_poly
            self.prev_right_x = int(np.clip(
                np.polyval(right_poly, h * 0.9), 0, w - 1))
        else:
            self.right_stale_frames += 1
            if self.right_stale_frames > self.prior_expiry_frames:
                self.prev_right_poly = None
                self.prev_right_x = None

        # ── 10. Update rolling lane-width history ──────────────────────────
        if left_fresh and right_fresh:
            ref_y = h * 0.8
            lx = np.polyval(left_poly, ref_y)
            rx = np.polyval(right_poly, ref_y)
            width_px = rx - lx
            if 10 < width_px < w * 0.95:
                self.lane_width_history.append(float(width_px))

        # ── 11. Single-lane fallback ────────────────────────────────────────
        # Only activates when one poly is completely gone (expired prior).
        # Synthesise the missing lane from the visible one ± average lane width.
        if left_poly is None and right_poly is not None:
            left_poly = self._synthesise_lane(right_poly, -1)
        elif right_poly is None and left_poly is not None:
            right_poly = self._synthesise_lane(left_poly, +1)

        # ── 12. Compute and publish error ──────────────────────────────────
        error_msg = Float32()

        if left_poly is not None and right_poly is not None:
            self.total_lost_frames = 0

            # Initial lookahead (base fraction of image height from top)
            base_y = h * self.base_lookahead
            lx0 = np.polyval(left_poly, base_y)
            rx0 = np.polyval(right_poly, base_y)
            mid_x0 = (lx0 + rx0) / 2.0
            coarse_error = (mid_x0 - w / 2.0) / (w / 2.0)

            # Adaptive lookahead: shorten when the error is large (in a turn)
            adaptive_y = h * self.base_lookahead * (1.0 - 0.4 * abs(coarse_error))
            adaptive_y = float(np.clip(adaptive_y, h * 0.3, h * 0.9))

            lx = np.polyval(left_poly, adaptive_y)
            rx = np.polyval(right_poly, adaptive_y)
            mid_x = (lx + rx) / 2.0

            raw_error = float(np.clip((mid_x - w / 2.0) / (w / 2.0), -1.0, 1.0))

            # Exponential smoothing
            self.smoothed_error = (
                self.alpha_smooth * raw_error
                + (1.0 - self.alpha_smooth) * self.smoothed_error
            )
            error_msg.data = self.smoothed_error

        else:
            self.total_lost_frames += 1
            if self.total_lost_frames > 10:
                error_msg.data = float('nan')
                self.get_logger().warn(
                    'Lane completely lost – publishing NaN error', throttle_duration_sec=2.0)
            else:
                # Hold last smoothed error for a few frames (brief occlusion)
                error_msg.data = self.smoothed_error

        self.error_pub.publish(error_msg)

        # ── 13. Zone state machine + publish ──────────────────────────────
        self._update_zone_state(
            yellow_px, white_px, red_px, left_poly, right_poly)
        zone_msg = Int8()
        zone_msg.data = int(self.zone_state)
        self.zone_pub.publish(zone_msg)

        # ── 14. Debug images (only when debug_mode=True) ───────────────────
        if self.debug_mode:
            self._publish_debug(
                warped, mask,
                left_poly, right_poly,
                left_cents, right_cents,
                h, w,
            )

    # ─────────────────────────────────────────────────────────────────────
    # Lane detection helpers
    # ─────────────────────────────────────────────────────────────────────

    def _find_lane_base(self, mask, w, h):
        """
        Return (left_x_base, right_x_base) for seeding sliding windows.

        For the first cold_start_frames frames, and whenever a prior seed
        is unavailable, use a histogram of the bottom third of the mask.
        Otherwise reuse previous frame's seed x positions.
        """
        use_histogram = (
            self.frame_count <= self.cold_start_frames
            or self.prev_left_x is None
            or self.prev_right_x is None
        )

        if use_histogram:
            histogram = np.sum(
                mask[int(h * 2 / 3):, :], axis=0
            ).astype(np.float32)
            mid = w // 2

            left_half = histogram[:mid]
            right_half = histogram[mid:]

            if np.any(left_half > 0):
                left_x = int(np.argmax(left_half))
            else:
                left_x = mid // 2  # safe default: quarter-width

            if np.any(right_half > 0):
                right_x = mid + int(np.argmax(right_half))
            else:
                right_x = mid + mid // 2  # safe default: three-quarter-width

            # Update priors so tracking starts from here
            self.prev_left_x = left_x
            self.prev_right_x = right_x
            return left_x, right_x

        return self.prev_left_x, self.prev_right_x

    def _sliding_windows(self, mask, h, x_base):
        """
        Slide n_windows windows from the bottom of the image upward, tracking
        yellow lane pixels.

        Returns
        -------
        centroids       : list of (y_center, x_center) for every window
        total_pixels    : total white pixels found across all windows
        valid_windows   : number of windows that exceeded min_pixels threshold
        """
        window_height = h // self.n_windows
        nonzero_y, nonzero_x = mask.nonzero()
        nonzero_y = np.asarray(nonzero_y)
        nonzero_x = np.asarray(nonzero_x)

        current_x = int(x_base)
        centroids = []
        total_pixels = 0
        valid_windows = 0

        for i in range(self.n_windows):
            y_low = h - (i + 1) * window_height
            y_high = h - i * window_height
            x_low = max(0, current_x - self.window_margin)
            x_high = min(mask.shape[1] - 1, current_x + self.window_margin)

            good = np.where(
                (nonzero_y >= y_low) & (nonzero_y < y_high)
                & (nonzero_x >= x_low) & (nonzero_x < x_high)
            )[0]

            if len(good) >= self.min_pixels:
                current_x = int(np.mean(nonzero_x[good]))
                total_pixels += len(good)
                valid_windows += 1

            centroids.append(((y_low + y_high) // 2, current_x))

        return centroids, total_pixels, valid_windows

    def _fit_polynomial(self, centroids, total_pixels, valid_windows,
                        prev_poly, h):
        """
        Fit a 2nd-order polynomial x = f(y) through the sliding window centroids.

        Returns
        -------
        (poly, is_fresh)
            poly      : np.ndarray of shape (3,) or None if no data at all
            is_fresh  : True if fitted from this frame's data
        """
        if total_pixels >= self.min_lane_pixels and valid_windows >= self.min_valid_windows:
            y_vals = np.array([c[0] for c in centroids], dtype=np.float32)
            x_vals = np.array([c[1] for c in centroids], dtype=np.float32)
            try:
                poly = np.polyfit(y_vals, x_vals, 2)
                # Sanity: reject fits where the lane runs nearly horizontally
                # at the bottom of the image — that means the sliding window
                # latched onto a loop's far side, which would drive the robot
                # backwards. Slope = dx/dy at y = h.
                slope_bottom = 2.0 * poly[0] * h + poly[1]
                if abs(slope_bottom) > self.max_bottom_slope:
                    # Treat as failed fit; fall through to prior.
                    pass
                else:
                    return poly, True
            except (np.linalg.LinAlgError, Exception):
                pass  # fall through to prior

        if prev_poly is not None:
            return prev_poly, False

        return None, False

    def _synthesise_lane(self, known_poly, direction):
        """
        Estimate the missing lane polynomial by shifting the known one
        laterally by the average lane width.  direction=+1 shifts right,
        direction=-1 shifts left.
        """
        if len(self.lane_width_history) == 0:
            return None
        avg_width = float(np.mean(self.lane_width_history))
        synth = known_poly.copy()
        synth[2] += direction * avg_width  # shift the constant term (x-intercept)
        return synth

    # ─────────────────────────────────────────────────────────────────────
    # Zone state machine
    # ─────────────────────────────────────────────────────────────────────

    def _update_zone_state(self, yellow_px, white_px, red_px,
                            left_poly, right_poly):
        any_line = (left_poly is not None or right_poly is not None)

        # Red marker has highest priority once armed: as soon as enough red
        # pixels are seen for several consecutive frames, switch to PARKING
        # and stay there. The arm-frame guard prevents the start marker from
        # latching parking immediately.
        if (self.zone_state != ZoneState.PARKING
                and self.frame_count > self.parking_arm_frames
                and red_px > self.red_found_thresh):
            self._red_entry_cnt += 1
            if self._red_entry_cnt >= self.parking_entry_frames:
                self.zone_state = ZoneState.PARKING
                self.get_logger().info(
                    f'Zone → PARKING  (red_px={red_px})')
                return
        else:
            self._red_entry_cnt = 0

        if self.zone_state == ZoneState.NORMAL:
            # Enter WHITE_ZONE when yellow disappears and white appears
            if yellow_px < self.yellow_lost_thresh and white_px > self.white_found_thresh:
                self._white_entry_cnt += 1
                if self._white_entry_cnt >= self.white_zone_entry_frames:
                    self.zone_state = ZoneState.WHITE_ZONE
                    self._no_line_cnt = 0
                    self.get_logger().info(
                        f'Zone → WHITE_ZONE  (y={yellow_px} w={white_px})')
            else:
                self._white_entry_cnt = 0

        elif self.zone_state == ZoneState.WHITE_ZONE:
            # If yellow comes back strongly, abort — false positive
            if yellow_px >= self.yellow_lost_thresh:
                self.zone_state = ZoneState.NORMAL
                self._white_entry_cnt = 0
                self._no_line_cnt = 0
                self.get_logger().info('Zone → NORMAL  (yellow returned)')
                return

            # White zone ended when ALL lines disappear for long enough
            if not any_line:
                self._no_line_cnt += 1
                if self._no_line_cnt >= self.white_zone_exit_frames:
                    self.zone_state = ZoneState.TURN_LEFT
                    self.get_logger().info('Zone → TURN_LEFT  (lines gone after white zone)')
            else:
                self._no_line_cnt = 0

        elif self.zone_state == ZoneState.TURN_LEFT:
            # Controller drives the turn; detector waits for yellow to return
            if yellow_px >= self.yellow_lost_thresh:
                self.zone_state = ZoneState.NORMAL
                self._white_entry_cnt = 0
                self._no_line_cnt = 0
                self.get_logger().info('Zone → NORMAL  (yellow re-acquired after turn)')

        # PARKING is terminal — no exit transitions.

    # ─────────────────────────────────────────────────────────────────────
    # Debug visualisation
    # ─────────────────────────────────────────────────────────────────────

    def _publish_debug(self, warped, mask, left_poly, right_poly,
                       left_cents, right_cents, h, w):
        # Bird's-eye view
        self.img_proj_pub.publish(
            self.bridge.cv2_to_imgmsg(warped, encoding='bgr8'))

        # Yellow mask
        self.img_mask_pub.publish(
            self.bridge.cv2_to_imgmsg(
                cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), encoding='bgr8'))

        # Annotated debug image
        debug = warped.copy()
        plot_y = np.linspace(0, h - 1, h)

        # Draw left lane (blue) and right lane (red) polynomial curves
        for poly, colour in ((left_poly, (255, 80, 80)), (right_poly, (80, 80, 255))):
            if poly is not None:
                xs = np.polyval(poly, plot_y).astype(np.int32)
                for yi, xi in zip(plot_y.astype(np.int32), xs):
                    if 0 <= xi < w:
                        cv2.circle(debug, (xi, yi), 2, colour, -1)

        # Draw sliding window centroid dots
        for y, x in left_cents:
            cv2.circle(debug, (x, y), 4, (255, 180, 0), -1)
        for y, x in right_cents:
            cv2.circle(debug, (x, y), 4, (0, 180, 255), -1)

        # Draw midpoint marker at adaptive lookahead y
        if left_poly is not None and right_poly is not None:
            coarse_y = h * self.base_lookahead
            lx0 = np.polyval(left_poly, coarse_y)
            rx0 = np.polyval(right_poly, coarse_y)
            coarse_err = ((lx0 + rx0) / 2.0 - w / 2.0) / (w / 2.0)
            apy = int(np.clip(
                h * self.base_lookahead * (1.0 - 0.4 * abs(coarse_err)),
                h * 0.3, h * 0.9,
            ))
            lx = int(np.polyval(left_poly, apy))
            rx = int(np.polyval(right_poly, apy))
            mid = (lx + rx) // 2
            cv2.circle(debug, (mid, apy), 10, (0, 255, 0), -1)
            cv2.circle(debug, (w // 2, apy), 10, (0, 255, 255), 2)  # ideal centre

        # Image centre line (yellow dashed reference)
        for yi in range(0, h, 10):
            cv2.line(debug, (w // 2, yi), (w // 2, yi + 5), (0, 220, 220), 1)

        # Error readout
        cv2.putText(
            debug,
            f'err={self.smoothed_error:+.3f}',
            (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
        )
        left_state = 'OK' if left_poly is not None else 'LOST'
        right_state = 'OK' if right_poly is not None else 'LOST'
        cv2.putText(
            debug,
            f'L:{left_state}  R:{right_state}',
            (5, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1,
        )
        zone_colours = {
            ZoneState.NORMAL:     (180, 180, 180),
            ZoneState.WHITE_ZONE: (255, 255,   0),
            ZoneState.TURN_LEFT:  (0,   80,  255),
            ZoneState.PARKING:    (0,    0,  255),
        }
        cv2.putText(
            debug,
            f'ZONE:{self.zone_state.name}',
            (5, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            zone_colours.get(self.zone_state, (200, 200, 200)), 1,
        )

        self.img_dbg_pub.publish(
            self.bridge.cv2_to_imgmsg(debug, encoding='bgr8'))


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectorNode()
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
