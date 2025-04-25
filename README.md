# Cobot-C1 Telekinesis

This Repo is a modified version of Frankmocap to perform Monocular camera based Robot Telekinesis with Cobot-C1 robotic arm

For getting more information about Frankmocap and installation instructions:

See [FRANKMOCAP.md](docs/FRANKMOCAP.md)

**NOTE:** This submodule is designed to be used in conjunction with MRL Cobo Sim ROS package

## Key Challenges

1. **No depth data or camera intrinsics**: Hard to determine how far the human wrist is from the camera.
2. **No camera-to-robot calibration**: Cannot directly map image-space positions to robot-space.
3. **Different body structures**: Human and robot body geometries vary greatly.

---

## Core Insight

Instead of mapping **absolute wrist positions**, the system maps **relative transformations**:

> The **relative transformation between the human wrist and their torso** is transferred to the **robot wrist relative to the robot torso**.

This assumption holds well in practice and simplifies cross-domain retargeting.

---

## Method Overview

### 1. **Body Pose Estimation**
- Crop of the human body is passed to **FrankMocap** to estimate SMPL-X parameters:
  - `β_b ∈ ℝ¹⁰`: Body shape
  - `θ_b ∈ ℝ⁴⁵`: Joint rotations (24 joints)
  - `φ_b ∈ ℝ³`: Global orientation
- Output: Full 3D body mesh, including wrist position and orientation.

### 2. **Coordinate Frame Definition**

| Entity          | Frame Origin            | Axes Definition                          |
|----------------|-------------------------|------------------------------------------|
| Human torso    | Human’s pelvic area      | x → front, y → left, z → up              |
| Robot torso    | 15cm below base frame*   | x → front, y → left, z → up              |
| Wrist (both)   | Wrist joint             | x → out of palm, y → toward thumb, z → toward middle fingertip |

##### \* Can be adjusted further
---

### 3. **Transformation Extraction**
- Traverse **human kinematic chain** from torso to wrist (using SMPL-X) to compute the **relative pose** (rotation + translation).
- This relative wrist pose is then applied to the **robot wrist**, assuming robot torso as the origin.

> This sidesteps the need for global alignment and allows a "floating camera" approach.

---
### 4. Gripper State Estimation
- Calculated using the distance between the thumb and the tip of the index finger.

- A sliding window filter is applied to ensure robustness against transient outliers.

---
### 5. Command Transmission
- **Transformation Matrix** and **Gripper State** is packed and sent to a ROS Node using a **UDP Socket** for further processing and execution.
- A local socket is used to make it easier to communicate between Frankmocap running in **conda** environment and **ROS** environment.
- This ensures low-latency communication for real-time robot control.

---
### 6. **Post-processing**
To ensure stability and smoothness:
- **Low-pass filtering** using an **Kalman filter**: (To add mathematical formulation)
  ```
  P_EMA = α * P_new + (1 - α) * P_EMA
  ```
  - Typical α = 0.25 for balance between responsiveness and noise smoothing.

- **Transformation** is applied to the Robot torso frame to get the target robot wrist pose

---

### 5. **Inverse Kinematics**
- Final smoothed wrist target pose is converted into **joint angles** using analytical inverse kinematcs equations for Cobot-C1

---

### 6. **Motion Execution**
- Telekinesis motion is executed by pulishing **JointState** msg on a rostopic using a tuned **PD controller**

## Result

This method enables fluid, real-time teleoperation of the robot arm using only a monocular camera and passive 3D body estimation, while ensuring:
- No collisions
- Realistic and responsive motion
- Generalization across users and environments

## How to Run

## Modifications to Frankmocap
Since our application required fast close to realtime telekinesis operation, some changes had to be made to Frankmocap pipeline
* As we only require pose information about some specific joints only, all other extra computations have been removed
* Using copy-paste type hand and body pose integration method for getting higher FPS as EFT base based integration optimization is computationly expensive and requires precomputed OpenPose Keypoints to work.
* Added mixed precision and gpu synchronization for *hand bbox detection*, can be found [Here](https://github.com/GigabyteZX1/frankmocap_mrl/blob/opt/handmocap/hand_bbox_detector.py#L114)
* Added vectorized calculations for *body bbox detection*
* Removed OpenCV visualization, using glViewer visualization with toggleable mesh 
* **telekinesis_utils.py** script is added to facilitate connection with cobot, different conversions and transformation matrix formation.
* Added extra CLI argument to facilitate ease of use and debugging in [demo_options.py](https://github.com/GigabyteZX1/frankmocap_mrl/blob/opt/demo/demo_options.py)

## Cobot Telekinesis ROS Node

## Current Pipeline


## Future Improvements

## References
The main idea behind this project is taken from this paper
```
@misc{sivakumar2022robotictelekinesislearningrobotic,
      title={Robotic Telekinesis: Learning a Robotic Hand Imitator by Watching Humans on Youtube}, 
      author={Aravind Sivakumar and Kenneth Shaw and Deepak Pathak},
      year={2022},
      eprint={2202.10448},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2202.10448}, 
}
```