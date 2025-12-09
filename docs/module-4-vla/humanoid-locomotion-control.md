# Humanoid Locomotion and Control

Humanoid locomotion represents one of the most challenging problems in robotics, requiring sophisticated control systems to achieve stable, efficient, and human-like movement. This section covers the principles and implementation of humanoid locomotion using Isaac Sim and Isaac ROS.

## Introduction to Humanoid Locomotion

Humanoid locomotion involves the complex control of multi-degree-of-freedom systems to achieve stable movement patterns similar to human walking. Unlike wheeled robots, humanoid robots must manage balance, foot placement, and dynamic stability simultaneously.

### Key Challenges in Humanoid Locomotion

1. **Balance Control**: Maintaining center of mass within support polygon
2. **Step Planning**: Determining optimal foot placement
3. **Gait Generation**: Creating stable walking patterns
4. **Terrain Adaptation**: Adjusting to different surfaces and obstacles
5. **Dynamic Stability**: Managing momentum during movement

### Types of Humanoid Gait

- **Static Gait**: Center of mass always within support polygon
- **Dynamic Gait**: Uses momentum to maintain balance
- **Passive Dynamic**: Exploits mechanical dynamics for efficient walking
- **Adaptive Gait**: Adjusts to terrain and disturbances in real-time

## Center of Mass (CoM) Control

### Zero Moment Point (ZMP) Theory

The Zero Moment Point is a critical concept in humanoid locomotion:

```
ZMP = (Σ(mi * (xi * g - x''i * mi)) / Σ(mi * g - z''i * mi), Σ(mi * (yi * g - y''i * mi)) / Σ(mi * g - z''i * mi))
```

Where:
- mi = mass of point i
- xi, yi = position coordinates
- x''i, y''i = acceleration
- g = gravitational acceleration
- z''i = vertical acceleration

### Center of Pressure (CoP) vs ZMP

- **Center of Pressure (CoP)**: Point where the ground reaction force acts
- **Zero Moment Point (ZMP)**: Point where net moment of active forces equals moment of passive forces

For stable walking, ZMP must remain within the support polygon defined by the feet.

## Walking Pattern Generation

### Preview Control Method

Preview control uses future reference trajectories to generate stable walking patterns:

```python
import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp

class PreviewController:
    def __init__(self, dt=0.01, preview_horizon=2.0):
        self.dt = dt
        self.preview_horizon = preview_horizon
        self.preview_steps = int(preview_horizon / dt)

    def compute_reference_trajectory(self, start_pos, goal_pos, walk_speed=0.3):
        """Compute reference trajectory for walking"""
        # Calculate distance to goal
        dist = np.sqrt((goal_pos[0] - start_pos[0])**2 + (goal_pos[1] - start_pos[1])**2)

        # Generate trajectory points
        steps = int(dist / (walk_speed * self.dt))
        x_traj = np.linspace(start_pos[0], goal_pos[0], steps)
        y_traj = np.linspace(start_pos[1], goal_pos[1], steps)

        # Add small sinusoidal variation for natural movement
        t = np.arange(len(x_traj)) * self.dt
        y_variation = 0.02 * np.sin(2 * np.pi * t * 0.5)  # Small vertical movement

        return np.column_stack([x_traj, y_traj, y_variation])

    def generate_footsteps(self, com_trajectory, step_length=0.3, step_width=0.2):
        """Generate footstep locations based on CoM trajectory"""
        footsteps = []
        current_left = [0, step_width/2, 0]  # Start with left foot
        current_right = [0, -step_width/2, 0]  # Start with right foot

        for i, com_pos in enumerate(com_trajectory):
            # Alternate steps based on phase
            if i % (int(0.8 / self.dt) * 2) < int(0.8 / self.dt):  # Left foot phase
                # Place left foot
                target_x = com_pos[0] + step_length / 2
                target_y = com_pos[1] + step_width / 2
                current_left = [target_x, target_y, 0]
            else:  # Right foot phase
                # Place right foot
                target_x = com_pos[0] + step_length / 2
                target_y = com_pos[1] - step_width / 2
                current_right = [target_x, target_y, 0]

            # Add footstep to trajectory
            if i % int(0.8 / self.dt) == 0:  # Add step every half cycle
                if i % (int(0.8 / self.dt) * 2) < int(0.8 / self.dt):
                    footsteps.append(('left', current_left))
                else:
                    footsteps.append(('right', current_right))

        return footsteps
```

### Linear Inverted Pendulum Model (LIPM)

The Linear Inverted Pendulum Model simplifies humanoid balance:

```python
class LIPMController:
    def __init__(self, com_height=0.8, gravity=9.81):
        self.com_height = com_height
        self.gravity = gravity
        self.omega = np.sqrt(gravity / com_height)

    def compute_zmp_from_com(self, com_pos, com_vel, com_acc):
        """Compute ZMP from CoM position, velocity, and acceleration"""
        zmp_x = com_pos[0] - (com_acc[0] / self.gravity) * self.com_height
        zmp_y = com_pos[1] - (com_acc[1] / self.gravity) * self.com_height
        return [zmp_x, zmp_y, 0]

    def compute_com_from_zmp(self, zmp_pos, com_pos_prev, com_vel_prev):
        """Compute CoM position from desired ZMP using LIPM"""
        dt = 0.01  # Control timestep

        # LIPM dynamics: com_ddot = omega^2 * (com - zmp)
        com_acc = self.omega**2 * (np.array(com_pos_prev[:2]) - np.array(zmp_pos[:2]))

        # Integrate to get new CoM position and velocity
        new_com_vel = com_vel_prev[:2] + com_acc * dt
        new_com_pos = com_pos_prev[:2] + new_com_vel * dt + 0.5 * com_acc * dt**2

        return [new_com_pos[0], new_com_pos[1], self.com_height], new_com_vel
```

## Isaac ROS Humanoid Control Integration

### Isaac ROS Control Packages

Isaac ROS provides specialized packages for humanoid control:

```yaml
# Isaac ROS Control configuration
isaac_ros_control:
  ros__parameters:
    update_rate: 100  # Hz
    controller_manager:
      ros__parameters:
        use_sim_time: true
        controller_names:
          - joint_state_broadcaster
          - left_leg_controller
          - right_leg_controller
          - torso_controller
          - head_controller

left_leg_controller:
  ros__parameters:
    type: position_controllers/JointGroupPositionController
    joints:
      - left_hip_roll
      - left_hip_yaw
      - left_hip_pitch
      - left_knee
      - left_ankle_pitch
      - left_ankle_roll

right_leg_controller:
  ros__parameters:
    type: position_controllers/JointGroupPositionController
    joints:
      - right_hip_roll
      - right_hip_yaw
      - right_hip_pitch
      - right_knee
      - right_ankle_pitch
      - right_ankle_roll
```

### Humanoid Balance Controller

```cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <Eigen/Dense>

class HumanoidBalanceController : public rclcpp::Node
{
public:
    HumanoidBalanceController() : Node("humanoid_balance_controller")
    {
        // Subscriptions
        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "joint_states", 10,
            std::bind(&HumanoidBalanceController::jointStateCallback, this, std::placeholders::_1)
        );

        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "imu/data", 10,
            std::bind(&HumanoidBalanceController::imuCallback, this, std::placeholders::_1)
        );

        // Publishers
        target_joints_pub_ = this->create_publisher<sensor_msgs::msg::JointState>(
            "target_joint_positions", 10
        );

        // Timer for control loop
        control_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10),  // 100 Hz control
            std::bind(&HumanoidBalanceController::controlLoop, this)
        );

        // Initialize balance controller parameters
        com_height_ = this->declare_parameter("com_height", 0.8);
        control_gain_ = this->declare_parameter("control_gain", 10.0);
        max_correction_ = this->declare_parameter("max_correction", 0.1);

        RCLCPP_INFO(this->get_logger(), "Humanoid Balance Controller initialized");
    }

private:
    void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        // Store current joint states
        current_joint_positions_ = msg->position;
        current_joint_velocities_ = msg->velocity;
        current_joint_efforts_ = msg->effort;
        has_joint_state_ = true;
    }

    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        // Store IMU data for balance feedback
        imu_orientation_ = msg->orientation;
        imu_angular_velocity_ = msg->angular_velocity;
        imu_linear_acceleration_ = msg->linear_acceleration;
        has_imu_data_ = true;
    }

    void controlLoop()
    {
        if (!has_joint_state_ || !has_imu_data_) {
            return;  // Wait for sensor data
        }

        // Calculate current center of mass
        auto com_pos = calculateCenterOfMass(current_joint_positions_);
        auto com_vel = calculateCenterOfMassVelocity(current_joint_positions_, current_joint_velocities_);

        // Calculate desired ZMP based on target trajectory
        auto desired_zmp = calculateDesiredZMP();

        // Calculate current ZMP from IMU and kinematics
        auto current_zmp = calculateCurrentZMP(com_pos, com_vel, imu_linear_acceleration_);

        // Compute balance error
        double x_error = desired_zmp.x - current_zmp.x;
        double y_error = desired_zmp.y - current_zmp.y;

        // Generate corrective joint commands using PID control
        auto correction_commands = computeBalanceCorrection(x_error, y_error);

        // Apply corrections to joint targets
        auto target_joints = applyBalanceCorrections(current_joint_positions_, correction_commands);

        // Publish target joint positions
        publishTargetJoints(target_joints);
    }

    geometry_msgs::msg::Point calculateCenterOfMass(const std::vector<double>& joint_positions)
    {
        // Calculate CoM based on joint positions and link masses
        // This is a simplified implementation - in practice, use URDF info
        geometry_msgs::msg::Point com;

        // For a simplified model, assume CoM is at fixed height
        // and calculate horizontal position based on joint angles
        double com_x = 0.0, com_y = 0.0;

        // Simplified CoM calculation (would use full kinematic model in practice)
        for (size_t i = 0; i < joint_positions.size(); ++i) {
            // Weight each joint contribution based on its position in the kinematic chain
            com_x += joint_positions[i] * 0.01;  // Simplified weighting
            com_y += joint_positions[i] * 0.01;  // Simplified weighting
        }

        com.x = com_x;
        com.y = com_y;
        com.z = com_height_;

        return com;
    }

    geometry_msgs::msg::Point calculateCurrentZMP(
        const geometry_msgs::msg::Point& com_pos,
        const geometry_msgs::msg::Point& com_vel,
        const geometry_msgs::msg::Vector3& linear_acc)
    {
        // Calculate ZMP from CoM and acceleration data
        // ZMP_x = CoM_x - (CoM_z * CoM_acc_x) / g
        // ZMP_y = CoM_y - (CoM_z * CoM_acc_y) / g

        geometry_msgs::msg::Point zmp;
        zmp.x = com_pos.x - (com_pos.z * linear_acc.x) / gravity_;
        zmp.y = com_pos.y - (com_pos.z * linear_acc.y) / gravity_;
        zmp.z = 0.0;  // ZMP is on the ground plane

        return zmp;
    }

    std::vector<double> computeBalanceCorrection(double x_error, double y_error)
    {
        // Simple PD controller for balance correction
        static double prev_x_error = 0, prev_y_error = 0;
        static double integral_x_error = 0, integral_y_error = 0;

        // PID parameters
        double kp = 100.0;  // Proportional gain
        double ki = 10.0;   // Integral gain
        double kd = 50.0;   // Derivative gain

        // Update error integrals
        integral_x_error += x_error * dt_;
        integral_y_error += y_error * dt_;

        // Calculate derivatives
        double dx_error = (x_error - prev_x_error) / dt_;
        double dy_error = (y_error - prev_y_error) / dt_;

        // Compute control outputs
        double x_control = kp * x_error + ki * integral_x_error + kd * dx_error;
        double y_control = kp * y_error + ki * integral_y_error + kd * dy_error;

        // Limit control outputs
        x_control = std::max(-max_correction_, std::min(max_correction_, x_control));
        y_control = std::max(-max_correction_, std::min(max_correction_, y_control));

        // Convert to joint space corrections
        // This would involve inverse kinematics in a real implementation
        std::vector<double> corrections(num_joints_, 0.0);

        // Simplified mapping - in practice, use full inverse kinematics
        corrections[left_hip_roll_idx_] = x_control * 0.1;
        corrections[right_hip_roll_idx_] = -x_control * 0.1;
        corrections[left_ankle_roll_idx_] = -x_control * 0.2;
        corrections[right_ankle_roll_idx_] = x_control * 0.2;

        corrections[left_hip_pitch_idx_] = y_control * 0.1;
        corrections[right_hip_pitch_idx_] = y_control * 0.1;
        corrections[left_ankle_pitch_idx_] = -y_control * 0.2;
        corrections[right_ankle_pitch_idx_] = -y_control * 0.2;

        prev_x_error = x_error;
        prev_y_error = y_error;

        return corrections;
    }

    void publishTargetJoints(const std::vector<double>& target_positions)
    {
        auto msg = sensor_msgs::msg::JointState();
        msg.header.stamp = this->now();
        msg.name = joint_names_;  // Would be initialized with actual joint names
        msg.position = target_positions;

        target_joints_pub_->publish(msg);
    }

    // Subscriptions
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;

    // Publishers
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr target_joints_pub_;

    // Timer
    rclcpp::TimerBase::SharedPtr control_timer_;

    // State variables
    std::vector<double> current_joint_positions_;
    std::vector<double> current_joint_velocities_;
    std::vector<double> current_joint_efforts_;
    geometry_msgs::msg::Quaternion imu_orientation_;
    geometry_msgs::msg::Vector3 imu_angular_velocity_;
    geometry_msgs::msg::Vector3 imu_linear_acceleration_;

    bool has_joint_state_ = false;
    bool has_imu_data_ = false;

    // Balance control parameters
    double com_height_;
    double control_gain_;
    double max_correction_;
    double gravity_ = 9.81;
    double dt_ = 0.01;  // Control timestep

    // Joint indices (would be initialized based on actual robot)
    int left_hip_roll_idx_ = 0;
    int right_hip_roll_idx_ = 1;
    // ... other joint indices
};
```

## Isaac Sim Humanoid Simulation

### Creating Humanoid Robots in Isaac Sim

```python
# Isaac Sim humanoid robot setup
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
import numpy as np

class IsaacHumanoidRobot:
    def __init__(self, prim_path="/World/HumanoidRobot", name="humanoid"):
        self.prim_path = prim_path
        self.name = name
        self.world = World()

        # Add humanoid robot to stage
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            print("Could not find Isaac Sim assets. Please check your Isaac Sim installation.")
            return

        # Use a humanoid robot asset (example with Carter robot - replace with actual humanoid)
        robot_asset_path = assets_root_path + "/Isaac/Robots/Humanoid/humanoid_instanceable.usd"
        add_reference_to_stage(usd_path=robot_asset_path, prim_path=prim_path)

        # Create articulation view for the robot
        self.robot_articulation = ArticulationView(prim_path=prim_path + "/base_link", name=name + "_view")
        self.world.add_articulation(self.robot_articulation)

    def initialize_robot(self):
        """Initialize robot in simulation"""
        self.world.reset()

        # Set default joint positions for standing pose
        default_positions = np.array([
            0.0,  # left_hip_roll
            0.0,  # left_hip_yaw
            0.0,  # left_hip_pitch
            0.0,  # left_knee
            0.0,  # left_ankle_pitch
            0.0,  # left_ankle_roll
            0.0,  # right_hip_roll
            0.0,  # right_hip_yaw
            0.0,  # right_hip_pitch
            0.0,  # right_knee
            0.0,  # right_ankle_pitch
            0.0,  # right_ankle_roll
            # ... add other joints
        ])

        self.robot_articulation.set_joint_positions(default_positions)

    def move_to_standing_pose(self):
        """Move robot to neutral standing pose"""
        standing_positions = np.array([
            0.0,   # left_hip_roll
            0.0,   # left_hip_yaw
            0.0,   # left_hip_pitch (neutral)
            0.0,   # left_knee (straight)
            0.0,   # left_ankle_pitch
            0.0,   # left_ankle_roll
            0.0,   # right_hip_roll
            0.0,   # right_hip_yaw
            0.0,   # right_hip_pitch (neutral)
            0.0,   # right_knee (straight)
            0.0,   # right_ankle_pitch
            0.0,   # right_ankle_roll
        ])

        self.robot_articulation.set_joint_positions(standing_positions)

    def execute_walk_cycle(self, step_phase, step_length=0.3, step_height=0.05):
        """Execute a single step in walking cycle"""
        # Calculate joint positions based on step phase (0 to 1)
        # This is a simplified walking pattern - in practice, use proper gait generation

        # Left leg trajectory
        left_knee_angle = np.sin(step_phase * 2 * np.pi) * 0.3  # Knee bend
        left_ankle_pitch = -left_knee_angle * 0.5  # Compensate for knee movement

        # Right leg trajectory (opposite phase)
        right_phase = (step_phase + 0.5) % 1.0
        right_knee_angle = np.sin(right_phase * 2 * np.pi) * 0.3
        right_ankle_pitch = -right_knee_angle * 0.5

        # Hip adjustments for balance
        pelvis_roll = np.sin(step_phase * 2 * np.pi) * 0.1  # Shift weight

        target_positions = np.array([
            pelvis_roll,  # left_hip_roll
            0.0,          # left_hip_yaw
            0.0,          # left_hip_pitch
            left_knee_angle,      # left_knee
            left_ankle_pitch,     # left_ankle_pitch
            0.0,          # left_ankle_roll
            -pelvis_roll, # right_hip_roll (opposite to maintain balance)
            0.0,          # right_hip_yaw
            0.0,          # right_hip_pitch
            right_knee_angle,     # right_knee
            right_ankle_pitch,    # right_ankle_pitch
            0.0,          # right_ankle_roll
        ])

        self.robot_articulation.set_joint_positions(target_positions)

    def get_robot_state(self):
        """Get current robot state including joint positions, velocities, and CoM"""
        joint_positions = self.robot_articulation.get_joint_positions()
        joint_velocities = self.robot_articulation.get_joint_velocities()

        # Calculate center of mass position and velocity
        com_position = self.calculate_center_of_mass(joint_positions)
        com_velocity = self.calculate_center_of_mass_velocity(joint_positions, joint_velocities)

        robot_state = {
            'joint_positions': joint_positions,
            'joint_velocities': joint_velocities,
            'com_position': com_position,
            'com_velocity': com_velocity,
            'base_position': self.robot_articulation.get_world_poses()[0][0],
            'base_orientation': self.robot_articulation.get_world_poses()[1][0]
        }

        return robot_state

    def calculate_center_of_mass(self, joint_positions):
        """Calculate center of mass position from joint configuration"""
        # Simplified CoM calculation - in practice, use URDF mass properties
        # This would involve forward kinematics and mass-weighted averaging

        # For now, return a simplified estimate
        return np.array([0.0, 0.0, 0.8])  # Approximate CoM height for humanoid

    def calculate_center_of_mass_velocity(self, joint_positions, joint_velocities):
        """Calculate center of mass velocity"""
        # Simplified CoM velocity calculation
        return np.array([0.0, 0.0, 0.0])
```

## Gait Planning and Execution

### Walking Pattern Generator

```python
class WalkingPatternGenerator:
    def __init__(self, step_length=0.3, step_width=0.2, step_height=0.05, step_duration=0.8):
        self.step_length = step_length
        self.step_width = step_width
        self.step_height = step_height
        self.step_duration = step_duration
        self.dt = 0.01  # Control timestep

    def generate_walk_trajectory(self, distance, direction='forward'):
        """Generate complete walk trajectory for given distance"""
        # Calculate number of steps needed
        step_count = int(distance / self.step_length)

        trajectory = []

        for step_num in range(step_count):
            # Generate single step trajectory
            step_trajectory = self.generate_single_step(step_num % 2 == 0)  # Alternate feet
            trajectory.extend(step_trajectory)

        return trajectory

    def generate_single_step(self, use_left_foot=True):
        """Generate trajectory for a single step"""
        steps_per_phase = int(self.step_duration / self.dt / 4)  # 4 phases per step

        trajectory = []

        # Phase 1: Preparation (lift foot slightly)
        for i in range(steps_per_phase):
            t = i / steps_per_phase
            lift_amount = self.step_height * 0.3 * t  # Gentle lift

            joint_positions = self.calculate_step_joints(
                phase=t,
                foot_lift=lift_amount,
                swing_foot=use_left_foot
            )
            trajectory.append(joint_positions)

        # Phase 2: Swing (move foot forward)
        for i in range(steps_per_phase):
            t = i / steps_per_phase
            forward_progress = self.step_length * t
            foot_lift = self.step_height * (1 - (t - 0.5)**2)  # Parabolic lift

            joint_positions = self.calculate_step_joints(
                phase=t,
                forward_progress=forward_progress,
                foot_lift=foot_lift,
                swing_foot=use_left_foot
            )
            trajectory.append(joint_positions)

        # Phase 3: Landing (lower foot)
        for i in range(steps_per_phase):
            t = i / steps_per_phase
            foot_lift = self.step_height * (1 - t)  # Lower foot
            forward_progress = self.step_length

            joint_positions = self.calculate_step_joints(
                phase=t,
                forward_progress=forward_progress,
                foot_lift=foot_lift,
                swing_foot=use_left_foot
            )
            trajectory.append(joint_positions)

        # Phase 4: Stabilization (adjust for balance)
        for i in range(steps_per_phase):
            t = i / steps_per_phase
            joint_positions = self.calculate_stance_joints(use_left_foot)
            trajectory.append(joint_positions)

        return trajectory

    def calculate_step_joints(self, phase, forward_progress=0, foot_lift=0, swing_foot=True):
        """Calculate joint positions for step execution"""
        # This would implement inverse kinematics for foot placement
        # For simplicity, return a basic pattern

        if swing_foot:  # Left foot is swinging
            # Move left foot forward and lift it
            left_knee = np.clip(foot_lift * 3, 0, 0.5)  # Knee bend for lifting
            left_ankle = -foot_lift  # Compensate ankle for lift
            right_knee = 0.0  # Right leg straight
            right_ankle = 0.0
        else:  # Right foot is swinging
            right_knee = np.clip(foot_lift * 3, 0, 0.5)
            right_ankle = -foot_lift
            left_knee = 0.0
            left_ankle = 0.0

        # Hip adjustments for balance during step
        hip_adjustment = foot_lift * 0.2 if foot_lift > 0 else 0.0

        joint_positions = np.zeros(12)  # Assuming 12 leg joints
        joint_positions[3] = left_knee    # left_knee
        joint_positions[4] = left_ankle   # left_ankle_pitch
        joint_positions[9] = right_knee   # right_knee
        joint_positions[10] = right_ankle # right_ankle_pitch

        if swing_foot:
            joint_positions[2] = hip_adjustment   # left_hip_pitch (raise opposite hip)
            joint_positions[8] = -hip_adjustment  # right_hip_pitch (lower swing hip)
        else:
            joint_positions[2] = -hip_adjustment  # left_hip_pitch (lower swing hip)
            joint_positions[8] = hip_adjustment   # right_hip_pitch (raise opposite hip)

        return joint_positions

    def calculate_stance_joints(self, stance_foot_is_left):
        """Calculate joint positions for stable stance"""
        # Return neutral standing position with slight knee bend
        joint_positions = np.zeros(12)
        joint_positions[3] = 0.1  # Slight knee bend for stability
        joint_positions[9] = 0.1  # Slight knee bend for stability
        return joint_positions
```

## ROS 2 Integration for Humanoid Control

### Humanoid Controller Manager

```cpp
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/twist.h>
#include <control_msgs/msg/joint_trajectory_controller_state.hpp>

class HumanoidControllerManager : public rclcpp::Node
{
public:
    HumanoidControllerManager() : Node("humanoid_controller_manager")
    {
        // Initialize humanoid-specific controllers
        initializeControllers();

        // Create subscribers for different command types
        cmd_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "cmd_vel", 10,
            std::bind(&HumanoidControllerManager::cmdVelCallback, this, std::placeholders::_1)
        );

        joint_command_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
            "joint_group_position_controller/commands", 10,
            std::bind(&HumanoidControllerManager::jointCommandCallback, this, std::placeholders::_1)
        );

        // Publishers for robot state
        robot_state_pub_ = this->create_publisher<sensor_msgs::msg::JointState>(
            "robot_state", 10
        );

        // Timer for control loop
        control_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10),  // 100 Hz
            std::bind(&HumanoidControllerManager::controlLoop, this)
        );

        RCLCPP_INFO(this->get_logger(), "Humanoid Controller Manager initialized");
    }

private:
    void cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
    {
        // Convert velocity command to walking pattern
        requested_linear_vel_ = msg->linear.x;
        requested_angular_vel_ = msg->angular.z;

        // Generate appropriate gait pattern based on velocity request
        generateWalkPattern(requested_linear_vel_, requested_angular_vel_);
    }

    void jointCommandCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
    {
        // Handle direct joint position commands
        target_joint_positions_ = msg->data;
        control_mode_ = JOINT_POSITION_CONTROL;
    }

    void controlLoop()
    {
        // Get current robot state
        auto current_state = getCurrentRobotState();

        // Apply appropriate control based on mode
        std::vector<double> commands;

        switch (control_mode_) {
            case WALK_CONTROL:
                commands = generateWalkCommands(current_state);
                break;
            case JOINT_POSITION_CONTROL:
                commands = generateJointPositionCommands(current_state);
                break;
            case BALANCE_CONTROL:
                commands = generateBalanceCommands(current_state);
                break;
            default:
                commands = std::vector<double>(num_joints_, 0.0);
                break;
        }

        // Apply commands to robot
        sendJointCommands(commands);

        // Publish robot state
        publishRobotState(current_state);
    }

    std::vector<double> generateWalkCommands(const RobotState& state)
    {
        // Generate walking pattern based on requested velocities
        std::vector<double> commands(num_joints_, 0.0);

        // Calculate step parameters based on requested velocities
        double step_frequency = calculateStepFrequency(requested_linear_vel_);
        double step_length = calculateStepLength(requested_linear_vel_);
        double turn_compensation = calculateTurnCompensation(requested_angular_vel_);

        // Generate walking pattern
        auto walk_pattern = walking_generator_.generateWalkingPattern(
            step_frequency, step_length, turn_compensation, state
        );

        // Convert to joint commands
        commands = walk_pattern.toJointCommands();

        return commands;
    }

    void initializeControllers()
    {
        // Initialize different controller types:
        // - Joint position controllers for each limb
        // - Balance controller for CoM management
        // - Walking pattern generator
        // - Impedance controllers for compliant behavior

        walking_generator_.initialize();
        balance_controller_.initialize();
        impedance_controllers_.initialize();
    }

    enum ControlMode {
        WALK_CONTROL,
        JOINT_POSITION_CONTROL,
        BALANCE_CONTROL,
        TRAJECTORY_CONTROL
    };

    ControlMode control_mode_ = WALK_CONTROL;

    // Subscriptions
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr joint_command_sub_;

    // Publishers
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr robot_state_pub_;

    // Timer
    rclcpp::TimerBase::SharedPtr control_timer_;

    // State variables
    double requested_linear_vel_ = 0.0;
    double requested_angular_vel_ = 0.0;
    std::vector<double> target_joint_positions_;
    std::vector<std::string> joint_names_;

    // Controllers
    WalkingPatternGenerator walking_generator_;
    BalanceController balance_controller_;
    ImpedanceControllerManager impedance_controllers_;

    // Constants
    const size_t num_joints_ = 28;  // Typical humanoid has ~28+ joints
};
```

## Advanced Locomotion Patterns

### Different Gait Types

```python
class GaitController:
    def __init__(self):
        self.current_gait = "walking"
        self.gait_parameters = {
            "walking": {
                "step_length": 0.3,
                "step_height": 0.05,
                "step_duration": 0.8,
                "stance_ratio": 0.6
            },
            "trotting": {
                "step_length": 0.4,
                "step_height": 0.1,
                "step_duration": 0.4,
                "stance_ratio": 0.4
            },
            "crawling": {
                "step_length": 0.1,
                "step_height": 0.02,
                "step_duration": 1.0,
                "stance_ratio": 0.8
            }
        }

    def switch_gait(self, gait_type):
        """Switch between different locomotion gaits"""
        if gait_type in self.gait_parameters:
            self.current_gait = gait_type
            self.update_gait_parameters(self.gait_parameters[gait_type])
            return True
        return False

    def generate_gait_pattern(self, velocity, terrain_type="flat"):
        """Generate gait pattern based on velocity and terrain"""
        gait_params = self.gait_parameters[self.current_gait]

        if terrain_type == "rough":
            # Adjust parameters for rough terrain
            gait_params["step_height"] *= 1.5
            gait_params["step_length"] *= 0.7
            gait_params["step_duration"] *= 1.2

        elif terrain_type == "stairs":
            # Special pattern for stair climbing
            return self.generate_stair_climbing_pattern(velocity)

        elif terrain_type == "narrow":
            # Adjust for narrow passages
            gait_params["step_width"] = 0.1  # Narrower steps

        # Generate appropriate pattern based on current gait
        if self.current_gait == "walking":
            return self.generate_walking_pattern(velocity, gait_params)
        elif self.current_gait == "trotting":
            return self.generate_trotting_pattern(velocity, gait_params)
        elif self.current_gait == "crawling":
            return self.generate_crawling_pattern(velocity, gait_params)

    def generate_stair_climbing_pattern(self, velocity):
        """Generate special pattern for stair climbing"""
        # Stair climbing requires specific foot placement and balance
        # This is a simplified implementation
        pattern = []

        # Approach step
        approach_joints = self.calculate_stance_joints()
        pattern.extend([approach_joints] * 10)  # Hold approach position

        # Lift swing foot to step height
        for i in range(20):  # 0.2 seconds at 100Hz
            t = i / 20.0
            lift_amount = 0.15 * np.sin(np.pi * t)  # Lift to step height
            joints = self.calculate_stance_joints()
            # Apply lift to swing foot (simplified)
            pattern.append(joints)

        # Move swing foot forward over step
        for i in range(20):
            t = i / 20.0
            forward_amount = 0.3 * t  # Move forward by step depth
            joints = self.calculate_stance_joints()
            pattern.append(joints)

        # Lower swing foot to next step
        for i in range(20):
            t = i / 20.0
            lower_amount = 0.15 * (1 - np.cos(np.pi * t))  # Smooth lowering
            joints = self.calculate_stance_joints()
            pattern.append(joints)

        return pattern

    def adapt_to_terrain(self, terrain_analysis):
        """Adapt gait based on terrain analysis from perception system"""
        # Analyze terrain characteristics
        slope = terrain_analysis.get('slope', 0)
        step_height = terrain_analysis.get('step_height', 0)
        surface_roughness = terrain_analysis.get('roughness', 0)
        obstacles = terrain_analysis.get('obstacles', [])

        # Select appropriate gait based on terrain
        if slope > 15:  # Steep incline
            self.switch_gait("crawling")
        elif step_height > 0.1:  # Significant height changes
            # Use special climbing gait
            pass
        elif surface_roughness > 0.5:  # Very rough terrain
            self.switch_gait("crawling")
            self.reduce_step_length(0.5)
        else:
            self.switch_gait("walking")

        # Adjust parameters based on terrain
        if obstacles:
            self.enable_obstacle_aware_navigation(obstacles)
```

## Balance Recovery Behaviors

### Fall Prevention and Recovery

```cpp
class BalanceRecoverySystem
{
public:
    BalanceRecoverySystem() {
        state_ = BALANCED;
        recovery_threshold_ = 0.3;  // ZMP outside support polygon threshold
        fall_threshold_ = 0.5;      // Critical imbalance threshold
    }

    BalanceState assessBalance(const RobotState& state)
    {
        // Calculate ZMP and compare to support polygon
        auto zmp = calculateZMP(state);
        auto support_polygon = calculateSupportPolygon(state);

        if (isOutsideSupportPolygon(zmp, support_polygon)) {
            double distance = distanceToSupportPolygon(zmp, support_polygon);

            if (distance > fall_threshold_) {
                state_ = FALLING;
                return state_;
            } else if (distance > recovery_threshold_) {
                state_ = UNBALANCED;
                initiateRecovery(state);
                return state_;
            }
        }

        state_ = BALANCED;
        return state_;
    }

    void initiateRecovery(const RobotState& state)
    {
        // Choose appropriate recovery action based on situation
        if (fabs(state.com_velocity.z) > 0.5) {
            // Falling - use protective landing
            executeProtectiveLanding(state);
        } else if (state.foot_contacts[LEFT_FOOT] && state.foot_contacts[RIGHT_FOOT]) {
            // Stable stance - use ankle strategy
            executeAnkleStrategy(state);
        } else {
            // Single support - use hip/stepping strategy
            executeHipSteppingStrategy(state);
        }
    }

    void executeAnkleStrategy(const RobotState& state)
    {
        // Ankle strategy: use ankle torques to shift CoM back to support
        double x_error = calculateZMPErrors(state).x;
        double y_error = calculateZMPErrors(state).y;

        // Simple PD control on ankle joints
        double ankle_roll_command = -kp_ankle_roll_ * x_error - kd_ankle_roll_ * state.ankle_velocities[ROLL];
        double ankle_pitch_command = -kp_ankle_pitch_ * y_error - kd_ankle_pitch_ * state.ankle_velocities[PITCH];

        // Apply commands
        setAnkleTorques(ankle_roll_command, ankle_pitch_command);
    }

    void executeHipSteppingStrategy(const RobotState& state)
    {
        // Hip strategy: use hip torques and stepping to recover balance
        auto zmp_error = calculateZMPErrors(state);

        // Decide if to step or use hip torques based on severity
        if (fabs(zmp_error.x) > 0.2 || fabs(zmp_error.y) > 0.2) {
            // Severe imbalance - plan emergency step
            planEmergencyStep(state, zmp_error);
        } else {
            // Moderate imbalance - use hip torques
            executeHipStrategy(state, zmp_error);
        }
    }

private:
    BalanceState state_;
    double recovery_threshold_;
    double fall_threshold_;

    // Control gains
    double kp_ankle_roll_ = 50.0;
    double kd_ankle_roll_ = 10.0;
    double kp_ankle_pitch_ = 50.0;
    double kd_ankle_pitch_ = 10.0;
    double kp_hip_roll_ = 100.0;
    double kd_hip_roll_ = 20.0;
    double kp_hip_pitch_ = 100.0;
    double kd_hip_pitch_ = 20.0;
};
```

## Performance Evaluation

### Metrics for Humanoid Locomotion

```python
class LocomotionEvaluator:
    def __init__(self):
        self.metrics = {
            'stability': [],
            'efficiency': [],
            'balance': [],
            'trajectory_accuracy': [],
            'energy_consumption': []
        }

    def evaluate_locomotion_performance(self, robot_state, reference_trajectory):
        """Evaluate humanoid locomotion performance"""

        # Stability metrics
        zmp_stability = self.calculate_zmp_stability(robot_state)
        self.metrics['stability'].append(zmp_stability)

        # Balance metrics
        com_deviation = self.calculate_com_deviation(robot_state)
        self.metrics['balance'].append(com_deviation)

        # Efficiency metrics
        energy_used = self.calculate_energy_consumption(robot_state)
        self.metrics['efficiency'].append(energy_used)

        # Trajectory tracking accuracy
        tracking_error = self.calculate_trajectory_error(robot_state, reference_trajectory)
        self.metrics['trajectory_accuracy'].append(tracking_error)

        return {
            'stability_score': self.calculate_average('stability'),
            'balance_score': self.calculate_average('balance'),
            'efficiency_score': self.calculate_average('efficiency'),
            'accuracy_score': self.calculate_average('trajectory_accuracy')
        }

    def calculate_zmp_stability(self, state):
        """Calculate ZMP-based stability metric"""
        zmp = state['zmp']
        support_polygon = state['support_polygon']

        # Calculate distance from ZMP to edge of support polygon
        distance_to_edge = self.distance_to_polygon_edge(zmp, support_polygon)

        # Normalize by support polygon area
        support_area = self.calculate_polygon_area(support_polygon)
        stability_metric = distance_to_edge / np.sqrt(support_area)

        return stability_metric

    def calculate_energy_consumption(self, state):
        """Calculate energy consumption based on joint torques and velocities"""
        total_energy = 0.0

        for i, (torque, velocity) in enumerate(zip(state['joint_torques'], state['joint_velocities'])):
            # Energy = integral of torque * velocity over time
            instantaneous_power = abs(torque * velocity)
            total_energy += instantaneous_power * self.dt  # dt = control timestep

        return total_energy

    def calculate_trajectory_error(self, robot_state, reference_trajectory):
        """Calculate error between robot position and reference trajectory"""
        robot_pos = np.array([robot_state['position']['x'], robot_state['position']['y']])

        # Find closest point on reference trajectory
        min_distance = float('inf')
        for ref_point in reference_trajectory:
            dist = np.linalg.norm(robot_pos - np.array([ref_point['x'], ref_point['y']]))
            min_distance = min(min_distance, dist)

        return min_distance
```

## Troubleshooting Common Issues

### 1. Instability and Falls
**Problem**: Robot falls during walking
**Solutions**:
- Check CoM height parameter in controller
- Verify joint limits and physical properties
- Adjust balance controller gains
- Ensure proper initial standing pose

### 2. Joint Limit Violations
**Problem**: Joints reaching limits during walking
**Solutions**:
- Check URDF joint limits
- Adjust gait parameters (step height/length)
- Implement joint limit checking in controller
- Use trajectory optimization to respect limits

### 3. ZMP Outside Support Polygon
**Problem**: Balance errors during locomotion
**Solutions**:
- Reduce walking speed
- Increase step width
- Adjust CoM height
- Improve sensor feedback quality

### 4. Phase Synchronization Issues
**Problem**: Legs getting out of sync during walking
**Solutions**:
- Verify gait phase calculation
- Check timing synchronization
- Ensure proper step sequencing
- Use phase oscillator models

## Best Practices

### 1. Gradual Complexity Increase
- Start with standing balance control
- Progress to simple stepping
- Add forward walking
- Introduce turning and complex maneuvers

### 2. Safety First Approach
- Implement soft limits and safety controllers
- Use simulation extensively before real robot testing
- Have emergency stop mechanisms
- Monitor robot state continuously

### 3. Parameter Tuning
- Start with conservative parameters
- Gradually increase aggressiveness
- Test on various terrains
- Validate with multiple scenarios

### 4. Sensor Fusion
- Combine multiple sensor inputs
- Use IMU for orientation feedback
- Implement sensor validation
- Handle sensor failures gracefully

## Exercise

Create a complete humanoid locomotion system that includes:

1. Implement a ZMP-based balance controller
2. Create a walking pattern generator with adjustable parameters
3. Integrate with Isaac Sim for humanoid robot simulation
4. Implement gait switching capabilities (walking, crawling, stair climbing)
5. Add balance recovery behaviors for disturbance rejection
6. Create a navigation system that uses the locomotion controller
7. Evaluate the system's performance with stability and efficiency metrics

Test your system with various scenarios including:
- Straight line walking
- Turning maneuvers
- Walking on uneven terrain
- Disturbance rejection (external pushes)
- Stair climbing (simulation)
- Obstacle avoidance while walking

Evaluate the system's performance using the metrics discussed in this section.