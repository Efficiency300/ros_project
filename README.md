# TurtleBot3 Yellow Lane Follower

ROS 2 Humble package for autonomous lane following on a custom track where **both lane lines are yellow**.

- **Robot**: TurtleBot3 Burger with camera add-on (default mounting position)
- **Camera**: `/camera/image_raw` published by `turtlebot3_bringup/camera.launch.py`
- **Algorithm**: Bird's-eye view → yellow HSV mask → sliding window tracking → PID control

---

## 1. Prerequisites

Install dependencies on the **Remote PC** (and/or on the robot if running nodes there):

```bash
sudo apt install \
  ros-humble-cv-bridge \
  ros-humble-image-transport \
  ros-humble-vision-opencv \
  python3-opencv \
  python3-numpy
```

---

## 2. Build

```bash
cd ~/turtlebot3_ws
colcon build --packages-select turtlebot3_lane_follower
source install/setup.bash
```

Rebuild after every edit to `lane_detector_node.py` or `lane_controller_node.py`:

```bash
colcon build --packages-select turtlebot3_lane_follower --symlink-install
```

> `--symlink-install` lets Python file edits take effect without rebuilding.

---

## 3. Bringup (on the Robot)

Always start the robot and camera first on the TurtleBot3:

```bash
# Terminal 1 – on the TurtleBot3 (SSH)
ros2 launch turtlebot3_bringup robot.launch.py
```

```bash
# Terminal 2 – on the TurtleBot3 (SSH), if camera is not included in robot.launch.py
ros2 launch turtlebot3_bringup camera.launch.py
```

Verify the camera is publishing:

```bash
ros2 topic hz /camera/image_raw
# Expected: ~30 Hz
```

---

## 4. Calibration (run once, before first autonomous drive)

### 4.1 Start the calibration launch

Run on the **Remote PC** (detector runs here to offload the RPi):

```bash
ros2 launch turtlebot3_lane_follower lane_detection.launch.py
```

This starts the lane detector with `debug_mode=true`, publishing three image topics.

### 4.2 Open rqt for visualisation

```bash
rqt
```

Go to **Plugins → Visualization → Image View** and open **three** image view panels.

| Panel | Topic | What to look for |
|---|---|---|
| 1 | `/lane/image_projected` | Bird's-eye view of the track |
| 2 | `/lane/image_yellow_mask` | Yellow lines = **solid white**, background = **black** |
| 3 | `/lane/image_lane_debug` | Blue = left lane, Red = right lane, Green dot = midpoint |

### 4.3 Tune the perspective transform

Edit `param/lane_detection.yaml` → `perspective_src`:

```yaml
perspective_src: [120, 480, 520, 480, 380, 310, 260, 310]
#                 ^BL-x BL-y  BR-x BR-y  TR-x TR-y  TL-x TL-y
#                 (coordinates in the ORIGINAL 640x480 camera image)
```

**Goal**: in `/lane/image_projected`, both lane lines should appear as near-vertical,
near-parallel strips. The trapezoid must tightly enclose the track visible ahead.

After each edit, rebuild and relaunch:

```bash
colcon build --packages-select turtlebot3_lane_follower --symlink-install
# No rebuild needed with --symlink-install, just restart the node:
ros2 launch turtlebot3_lane_follower lane_detection.launch.py
```

### 4.4 Tune the yellow HSV range

Edit `param/lane_detection.yaml` → `yellow_h_min/max`, `yellow_s_min/max`, `yellow_v_min/max`.

```yaml
yellow_h_min: 20    # OpenCV H range: 0–179
yellow_h_max: 35    # Typical yellow: ~22–30; widen if lines are dim
yellow_s_min: 100
yellow_s_max: 255
yellow_v_min: 100
yellow_v_max: 255
```

**Goal**: in `/lane/image_yellow_mask`, both lane lines are solid white with no holes.
Background, floor and reflections are black.

**Quick override without editing the file** (for live testing):

```bash
ros2 param set /lane_detector_node yellow_h_min 18
ros2 param set /lane_detector_node yellow_h_max 38
```

> Note: `ros2 param set` changes take effect on the **next frame** but are **not saved** to YAML.
> Write confirmed values back to `param/lane_detection.yaml` manually.

### 4.5 Verify lane debug overlay

Watch `/lane/image_lane_debug`:
- Coloured dots (window centroids) should track along each lane line
- The **green dot** (midpoint) should sit between the two lines
- The **cyan dashed line** is the image centre — green dot should be close to it when centred
- Text in top-left shows `err=+0.xxx` (near 0 when robot is centred)
- Text shows `L:OK  R:OK` when both lanes are detected

---

## 5. Autonomous Driving

Place the robot **roughly centred** between the two lane lines, then:

```bash
# Full autonomous run (no debug overhead — best performance)
ros2 launch turtlebot3_lane_follower lane_following.launch.py
```

```bash
# Autonomous run WITH live debug images (for monitoring, ~40 ms extra latency)
ros2 launch turtlebot3_lane_follower lane_following.launch.py debug_mode:=true
```

To **stop the robot** at any time:

```bash
# Ctrl+C in the launch terminal, OR:
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist '{}'
```

---

## 6. Debugging Commands

### 6.1 Monitor the lane error in real time

```bash
ros2 topic echo /lane/center_error
# 0.0  = perfectly centred
# +1.0 = robot at far right edge → turns left
# -1.0 = robot at far left edge  → turns right
# nan  = lane completely lost     → robot stops
```

### 6.2 Monitor cmd_vel output

```bash
ros2 topic echo /cmd_vel
# Watch linear.x (should reduce in turns) and angular.z (PID output)
```

### 6.3 Monitor topic frequencies

```bash
ros2 topic hz /camera/image_raw        # should be ~30 Hz
ros2 topic hz /lane/center_error       # should match camera rate
ros2 topic hz /cmd_vel                 # should match lane error rate
```

### 6.4 Check all active nodes and topics

```bash
ros2 node list
ros2 topic list
```

### 6.5 View all current parameters of the detector

```bash
ros2 param list /lane_detector_node
ros2 param dump /lane_detector_node      # prints all values
```

### 6.6 Live-tune PID gains without restarting

```bash
ros2 param set /lane_controller_node kp 1.5
ros2 param set /lane_controller_node kd 0.4
ros2 param set /lane_controller_node base_speed 0.12
```

> Parameters set this way reset to YAML defaults on next launch.
> Write confirmed values to `param/lane_controller.yaml`.

### 6.7 Run only the detector (no controller — robot stays still)

Useful to verify detection without the robot moving:

```bash
ros2 run turtlebot3_lane_follower lane_detector_node \
  --ros-args --params-file ~/turtlebot3_ws/src/turtlebot3_lane_follower/param/lane_detection.yaml \
  -p debug_mode:=true
```

### 6.8 Run only the controller (feed a manual error to test PID)

```bash
ros2 run turtlebot3_lane_follower lane_controller_node \
  --ros-args --params-file ~/turtlebot3_ws/src/turtlebot3_lane_follower/param/lane_controller.yaml

# In another terminal, publish a test error:
ros2 topic pub /lane/center_error std_msgs/msg/Float32 '{data: 0.3}'
# Robot should slowly turn left to correct
```

### 6.9 Inspect a single debug image from the command line

```bash
ros2 run image_tools showimage --ros-args -r image:=/lane/image_lane_debug
```

### 6.10 Record a bag for offline analysis

```bash
ros2 bag record \
  /camera/image_raw \
  /lane/center_error \
  /lane/image_lane_debug \
  /cmd_vel \
  -o lane_debug_bag
```

Replay and re-inspect:

```bash
ros2 bag play lane_debug_bag --loop
rqt   # then open Image View → /lane/image_lane_debug
```

---

## 7. PID Tuning Guide

Start with the defaults in `lane_controller.yaml` and adjust one parameter at a time.

| Symptom | Cause | Fix |
|---|---|---|
| Robot barely reacts, drifts off | `kp` too low | Increase `kp` (+0.2 steps) |
| Robot weaves / oscillates | `kp` too high | Reduce `kp`; or increase `kd` |
| Robot overshoots apex of turns | `kd` too low | Increase `kd` (+0.1 steps) |
| Robot jerks/chatters even on straights | `kd` too high | Reduce `kd` |
| Persistent offset on long straights | Steady-state error | Add small `ki` (0.02–0.05) |
| Robot oscillates after correcting offset | `ki` too high | Reduce `ki`; increase `ki_max` clamp |
| Robot too slow in straights | `base_speed` too low | Increase `base_speed` (+0.02 m/s steps) |
| Robot cuts corners / crosses outer line | `base_speed` too high | Reduce `base_speed`; increase `speed_reduction_factor` |

**Recommended tuning order**: `kp` → `kd` → `base_speed` → `ki` (only if needed)

---

## 8. Detection Tuning Guide

| Symptom | Cause | Fix |
|---|---|---|
| Lines not visible in mask | HSV range too narrow | Widen `yellow_h_max` or reduce `yellow_s_min` / `yellow_v_min` |
| Lots of noise/blobs in mask | HSV range too wide | Narrow the range; increase `yellow_s_min` |
| Bird's-eye view looks skewed | `perspective_src` wrong | Re-tune the 8 trapezoid points |
| Lanes appear too close together in bird's-eye | Trapezoid base too narrow | Increase x spread of bottom-left/bottom-right points |
| Lane detection drops out on curves | `window_margin` too small | Increase `window_margin` (e.g. 80) |
| Detector jumps to wrong lane | `prior_expiry_frames` too low | Increase `prior_expiry_frames` (e.g. 15) |
| Error is noisy / robot oscillates without apparent cause | Smoothing too low | Increase `alpha_smooth` toward 0.7 |
| Robot reacts sluggishly to curves | Smoothing too high | Reduce `alpha_smooth` toward 0.3 |

---

## 9. Topic & Parameter Reference

### Topics

| Topic | Type | Description |
|---|---|---|
| `/camera/image_raw` | `sensor_msgs/Image` | Raw camera input (subscribed) |
| `/lane/center_error` | `std_msgs/Float32` | Lateral error [-1, 1], NaN = lost |
| `/lane/image_projected` | `sensor_msgs/Image` | Bird's-eye warp (debug) |
| `/lane/image_yellow_mask` | `sensor_msgs/Image` | Yellow HSV binary mask (debug) |
| `/lane/image_lane_debug` | `sensor_msgs/Image` | Annotated overlay (debug) |
| `/cmd_vel` | `geometry_msgs/Twist` | Velocity command (published) |

### Key Parameters — Detector

| Parameter | Default | Effect |
|---|---|---|
| `resize_factor` | `0.5` | `0.5` = 320×240; lower = faster but less detail |
| `debug_mode` | `false` | `true` enables 3 debug image topics |
| `perspective_src` | 8 ints | Trapezoid in original 640×480 image — **must calibrate** |
| `yellow_h_min/max` | `20/35` | Hue range for yellow — **must calibrate** |
| `window_margin` | `60` | Sliding window half-width (resized px) |
| `prior_expiry_frames` | `10` | Frames before stale prior is discarded |
| `alpha_smooth` | `0.5` | Error smoothing (lower=reactive, higher=smooth) |
| `base_lookahead` | `0.6` | Lookahead fraction (shorter=reactive, longer=anticipatory) |

### Key Parameters — Controller

| Parameter | Default | Effect |
|---|---|---|
| `kp` | `1.2` | Proportional gain |
| `kd` | `0.3` | Derivative gain (dampens oscillation, anticipates turns) |
| `ki` | `0.0` | Integral gain (fixes steady-state offset) |
| `base_speed` | `0.10` | Forward speed (m/s) on straights |
| `speed_reduction_factor` | `1.5` | How much to slow in turns |
| `safety_timeout` | `2.0` | Seconds before safety stop |
