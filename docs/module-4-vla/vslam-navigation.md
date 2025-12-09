# VSLAM and Navigation

Visual Simultaneous Localization and Mapping (VSLAM) is a critical technology for autonomous robots, enabling them to understand their environment and navigate without prior knowledge. This section covers VSLAM concepts and navigation techniques for humanoid robots using Isaac Sim and Isaac ROS.

## Introduction to VSLAM

VSLAM (Visual Simultaneous Localization and Mapping) combines computer vision and sensor data to:
- **Localize** the robot in its environment
- **Map** the environment in real-time
- **Navigate** safely through the mapped space

### Key Components of VSLAM
- **Feature Detection**: Identify distinctive points in images
- **Feature Matching**: Match features between frames
- **Pose Estimation**: Calculate robot position and orientation
- **Map Building**: Create and update environmental map
- **Loop Closure**: Recognize previously visited locations

## Visual SLAM Approaches

### 1. Feature-Based VSLAM
Relies on detecting and tracking distinctive features in the environment.

### 2. Direct VSLAM
Uses pixel intensities directly rather than features.

### 3. Semi-Direct VSLAM (SVO)
Combines feature-based tracking with direct methods.

## Popular VSLAM Systems

### ORB-SLAM
- **Features**: Real-time operation, loop closure, relocalization
- **Strengths**: Robust, well-tested, handles monocular/stereo/RGB-D
- **Weaknesses**: Requires texture-rich environments

### LSD-SLAM
- **Features**: Dense reconstruction, direct method
- **Strengths**: Works in low-texture environments
- **Weaknesses**: Computationally intensive

### DSO (Direct Sparse Odometry)
- **Features**: Direct optimization, photometric calibration
- **Strengths**: Accurate, handles exposure changes
- **Weaknesses**: Requires good initialization

## Isaac ROS VSLAM Integration

### Isaac ROS Visual SLAM Package

Isaac ROS provides optimized VSLAM capabilities through the Isaac ROS Visual SLAM package:

```yaml
# Example launch configuration
visual_slam_node:
  ros__parameters:
    enable_occupancy_grid: true
    enable_diagnostics: false
    occupancy_grid_resolution: 0.05
    frame_id: "oak-d_frame"
    base_frame: "base_link"
    odom_frame: "odom"
    enable_slam_visualization: true
    enable_landmarks_view: true
    enable_observations_view: true
    calibration_file: "/tmp/calibration.json"
    rescale_threshold: 2.0
```

### Implementation Example

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge
import numpy as np

class IsaacVSLAMNode(Node):
    def __init__(self):
        super().__init__('isaac_vslam_node')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Subscribe to camera and camera info
        self.image_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_rect_color',
            self.image_callback,
            10
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/rgb/camera_info',
            self.camera_info_callback,
            10
        )

        # Publishers
        self.pose_pub = self.create_publisher(PoseStamped, '/visual_slam/pose', 10)
        self.map_pub = self.create_publisher(OccupancyGrid, '/visual_slam/map', 10)

        # Internal state
        self.camera_info = None
        self.latest_image = None

        self.get_logger().info('Isaac VSLAM node initialized')

    def image_callback(self, msg):
        """Process incoming camera images for VSLAM"""
        if self.camera_info is None:
            self.get_logger().warn('Waiting for camera info...')
            return

        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Process with Isaac ROS Visual SLAM (simulated here)
            pose_estimate, map_update = self.process_vslam(cv_image)

            # Publish results
            if pose_estimate is not None:
                pose_msg = PoseStamped()
                pose_msg.header = msg.header
                pose_msg.pose = pose_estimate
                self.pose_pub.publish(pose_msg)

            if map_update is not None:
                map_msg = OccupancyGrid()
                map_msg.header = msg.header
                # Populate map message with map_update data
                self.map_pub.publish(map_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def camera_info_callback(self, msg):
        """Store camera calibration information"""
        self.camera_info = msg

    def process_vslam(self, image):
        """Process image through VSLAM pipeline (simulated)"""
        # In a real implementation, this would interface with Isaac ROS Visual SLAM
        # For simulation, we'll return dummy data

        # Simulate pose estimation
        pose = geometry_msgs.msg.Pose()
        # This would come from actual VSLAM processing
        pose.position.x = 0.0  # Would be actual position
        pose.position.y = 0.0
        pose.position.z = 0.0
        pose.orientation.w = 1.0  # Unit quaternion

        # Simulate map update
        map_data = None  # Would be actual map data

        return pose, map_data
```

## Navigation with VSLAM

### Integration with Nav2

When using VSLAM for localization in navigation, integrate with Nav2:

```yaml
# nav2_params.yaml with VSLAM localization
amcl:
  ros__parameters:
    use_sim_time: True
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_link"
    beam_skip_distance: 0.5
    beam_skip_error_threshold: 0.9
    beam_skip_threshold: 0.3
    do_beamskip: false
    global_frame_id: "map"  # VSLAM provides the map frame
    lambda_short: 0.1
    laser_likelihood_max_dist: 2.0
    laser_max_range: 100.0
    laser_min_range: -1.0
    laser_model_type: "likelihood_field"
    max_beams: 60
    max_particles: 2000
    min_particles: 500
    odom_frame_id: "odom"
    pf_err: 0.05
    pf_z: 0.99
    recovery_alpha_fast: 0.0
    recovery_alpha_slow: 0.0
    resample_interval: 1
    robot_model_type: "nav2_amcl::DifferentialMotionModel"
    save_pose_rate: 0.5
    sigma_hit: 0.2
    tf_broadcast: true
    transform_tolerance: 1.0
    update_min_a: 0.2
    update_min_d: 0.2
    z_hit: 0.5
    z_max: 0.05
    z_rand: 0.5
    z_short: 0.05

# For VSLAM, we might use a different approach than AMCL
slam_toolbox:
  ros__parameters:
    use_sim_time: True
    # SLAM Toolbox parameters for online/offline SLAM
    solver_plugin: "slam_toolbox::OptimizationSolverLevenbergMarquardt"
    ceres_linear_solver: "SPARSE_NORMAL_CHOLESKY"
    ceres_preconditioner: "SCHUR_JACOBI"
    ceres_trust_strategy: "LEVENBERG_MARQUARDT"
    ceres_dogleg_type: "TRADITIONAL_DOGLEG"
    max_iterations: 100
    map_file_name: "map"
    map_start_pose: [0.0, 0.0, 0.0]
    map_update_interval: 5.0
    resolution: 0.05
    max_laser_range: 20.0
    minimum_time_interval: 0.5
    transform_publish_period: 0.02
    tf_buffer_duration: 30.
    stack_size_to_use: 40000000  # 40MB
    enable_interactive_mode: true
    scan_buffer_size: 30
    scan_buffer_maximum_scan_distance: 10.0
    scan_buffer_minimum_scan_distance: 0.1
    scan_topic: "/scan"
    mode: "localization"  # or "mapping" depending on use case
```

### Isaac Sim Navigation Scene

Create a navigation scene in Isaac Sim:

```python
# Isaac Sim navigation scene setup
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.robots import Robot
from omni.isaac.range_sensor import RotatingLidarPhysX
from omni.isaac.sensor import Camera
import numpy as np

class IsaacNavigationScene:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.setup_navigation_environment()

    def setup_navigation_environment(self):
        """Set up a navigation environment in Isaac Sim"""
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            print("Could not find Isaac Sim assets. Please check your Isaac Sim installation.")
            return

        # Add a navigation robot (Carter robot)
        robot_asset_path = assets_root_path + "/Isaac/Robots/Carter/carter_navigate.usd"
        add_reference_to_stage(usd_path=robot_asset_path, prim_path="/World/Carter")

        # Add a simple navigation environment
        room_asset_path = assets_root_path + "/Isaac/Environments/Simple_Room/simple_room.usd"
        add_reference_to_stage(usd_path=room_asset_path, prim_path="/World/Room")

        # Add a LIDAR sensor to the robot
        lidar = RotatingLidarPhysX(
            prim_path="/World/Carter/chassis/lidar",
            translation=np.array([0.0, 0.0, 0.25]),
            config="Carter_2D",
            rotation_frequency=10,
            samples_per_scan=1080
        )

        # Add a camera for visual SLAM
        camera = Camera(
            prim_path="/World/Carter/chassis/camera",
            frequency=30,
            resolution=(640, 480)
        )

        # Initialize the world
        self.world.reset()

    def run_navigation_simulation(self, goal_position):
        """Run navigation simulation with obstacle avoidance"""
        self.world.reset()

        while not self.world.is_stopped():
            self.world.step(render=True)

            # Get robot state
            robot_position = self.get_robot_position()
            robot_orientation = self.get_robot_orientation()
            lidar_data = self.get_lidar_data()

            # Check if reached goal
            if self.is_at_goal(robot_position, goal_position):
                print(f"Reached goal at {goal_position}!")
                break

            # Plan and execute navigation
            cmd_vel = self.plan_navigation_command(
                robot_position, robot_orientation,
                goal_position, lidar_data
            )

            # Apply command to robot (this would interface with ROS control)
            self.execute_command(cmd_vel)

    def get_robot_position(self):
        """Get current robot position from Isaac Sim"""
        # In a real implementation, this would get the robot's position
        # from the simulation
        pass

    def get_robot_orientation(self):
        """Get current robot orientation from Isaac Sim"""
        pass

    def get_lidar_data(self):
        """Get current LIDAR data from Isaac Sim"""
        pass

    def is_at_goal(self, current_pos, goal_pos, tolerance=0.2):
        """Check if robot is at goal position"""
        distance = np.sqrt((current_pos[0] - goal_pos[0])**2 + (current_pos[1] - goal_pos[1])**2)
        return distance < tolerance

    def plan_navigation_command(self, current_pos, current_orient, goal_pos, lidar_data):
        """Plan navigation command based on goal and sensor data"""
        # Calculate direction to goal
        dx = goal_pos[0] - current_pos[0]
        dy = goal_pos[1] - current_pos[1]
        goal_distance = np.sqrt(dx*dx + dy*dy)

        # Calculate goal angle
        goal_angle = np.arctan2(dy, dx)

        # Get robot's current angle
        robot_yaw = self.orientation_to_yaw(current_orient)

        # Calculate angle difference
        angle_diff = self.normalize_angle(goal_angle - robot_yaw)

        # Simple proportional controller
        linear_vel = min(0.5, goal_distance * 0.5)  # Scale with distance
        angular_vel = angle_diff * 1.0  # Proportional control

        # Obstacle avoidance
        min_distance = np.min(lidar_data) if len(lidar_data) > 0 else float('inf')

        if min_distance < 0.5:  # Obstacle detected
            # Slow down and turn away from obstacle
            linear_vel *= 0.3
            angular_vel += self.avoid_obstacle(lidar_data)

        # Ensure velocities are within limits
        linear_vel = np.clip(linear_vel, 0.0, 0.5)
        angular_vel = np.clip(angular_vel, -0.5, 0.5)

        return [linear_vel, 0.0, 0.0], [0.0, 0.0, angular_vel]  # linear, angular velocities

    def orientation_to_yaw(self, orientation):
        """Convert quaternion orientation to yaw angle"""
        # Simplified conversion - in practice, use proper quaternion to euler conversion
        import math
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi] range"""
        while angle > np.pi:
            angle -= 2*np.pi
        while angle < -np.pi:
            angle += 2*np.pi
        return angle

    def avoid_obstacle(self, lidar_data):
        """Calculate avoidance angular velocity based on LIDAR data"""
        if len(lidar_data) == 0:
            return 0.0

        # Find the direction of the closest obstacle
        min_idx = np.argmin(lidar_data)
        angle_resolution = 2 * np.pi / len(lidar_data)
        obstacle_angle = min_idx * angle_resolution - np.pi  # Convert to [-pi, pi]

        # Turn away from the obstacle
        # If obstacle is on the right, turn left (negative angular velocity)
        # If obstacle is on the left, turn right (positive angular velocity)
        if abs(obstacle_angle) < np.pi/2:  # Obstacle is in front
            return -np.sign(obstacle_angle) * 0.3  # Turn away from obstacle
        else:
            return 0.0  # Obstacle is behind, no need to turn

    def execute_command(self, cmd_vel):
        """Execute the navigation command in Isaac Sim"""
        # In a real implementation, this would send the command to the robot
        # controller in Isaac Sim
        pass
```

## Isaac ROS Navigation Components

### Isaac ROS Navigation Stack

Isaac ROS provides navigation components optimized for NVIDIA hardware:

```cpp
// Isaac ROS Navigation Integration Example
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

class IsaacNavigationNode : public rclcpp::Node
{
public:
    IsaacNavigationNode() : Node("isaac_navigation_node")
    {
        // Subscriptions
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "odom", 10,
            std::bind(&IsaacNavigationNode::odomCallback, this, std::placeholders::_1)
        );

        scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "scan", 10,
            std::bind(&IsaacNavigationNode::scanCallback, this, std::placeholders::_1)
        );

        goal_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "goal", 10,
            std::bind(&IsaacNavigationNode::goalCallback, this, std::placeholders::_1)
        );

        // Publisher for velocity commands
        cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 10);

        // Timer for navigation control loop
        control_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),  // 10 Hz
            std::bind(&IsaacNavigationNode::controlLoop, this)
        );

        RCLCPP_INFO(this->get_logger(), "Isaac Navigation Node initialized");
    }

private:
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        current_pose_ = msg->pose.pose;
        current_twist_ = msg->twist.twist;
        has_odom_ = true;
    }

    void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
    {
        latest_scan_ = *msg;
        has_scan_ = true;
    }

    void goalCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        goal_pose_ = msg->pose;
        has_goal_ = true;

        RCLCPP_INFO(
            this->get_logger(),
            "Received new goal: (%.2f, %.2f)",
            goal_pose_.position.x, goal_pose_.position.y
        );
    }

    void controlLoop()
    {
        if (!has_odom_ || !has_scan_ || !has_goal_) {
            return;  // Wait for all required data
        }

        // Calculate control command
        auto cmd_vel = calculateNavigationCommand();

        // Publish command
        cmd_vel_pub_->publish(cmd_vel);

        // Check if goal reached
        if (isGoalReached()) {
            stopRobot();
            has_goal_ = false;
            RCLCPP_INFO(this->get_logger(), "Goal reached!");
        }
    }

    geometry_msgs::msg::Twist calculateNavigationCommand()
    {
        geometry_msgs::msg::Twist cmd;

        // Calculate direction to goal
        double dx = goal_pose_.position.x - current_pose_.position.x;
        double dy = goal_pose_.position.y - current_pose_.position.y;
        double goal_distance = sqrt(dx*dx + dy*dy);
        double goal_angle = atan2(dy, dx);

        // Get robot's current orientation
        tf2::Quaternion q(
            current_pose_.orientation.x,
            current_pose_.orientation.y,
            current_pose_.orientation.z,
            current_pose_.orientation.w
        );
        tf2::Matrix3x3 m(q);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);

        // Calculate angle difference
        double angle_diff = goal_angle - yaw;
        // Normalize angle to [-π, π]
        while (angle_diff > M_PI) angle_diff -= 2*M_PI;
        while (angle_diff < -M_PI) angle_diff += 2*M_PI;

        // Simple proportional controller
        double kp_linear = 0.5;
        double kp_angular = 1.0;

        cmd.linear.x = std::min(0.5, kp_linear * goal_distance);  // Limit max speed
        cmd.angular.z = kp_angular * angle_diff;

        // Obstacle avoidance
        if (has_scan_) {
            double min_distance = *std::min_element(latest_scan_.ranges.begin(), latest_scan_.ranges.end());

            if (min_distance < 0.5) {  // Obstacle detected
                // Reduce forward speed
                cmd.linear.x *= 0.3;

                // Calculate avoidance based on scan data
                cmd.angular.z += calculateObstacleAvoidance();
            }
        }

        // Apply limits
        cmd.linear.x = std::max(0.0, std::min(0.5, cmd.linear.x));  // Positive forward only
        cmd.angular.z = std::max(-0.5, std::min(0.5, cmd.angular.z));

        return cmd;
    }

    double calculateObstacleAvoidance()
    {
        if (latest_scan_.ranges.empty()) return 0.0;

        // Find closest obstacle in front of robot (±60 degrees)
        int start_idx = static_cast<int>((M_PI/3 - latest_scan_.angle_min) / latest_scan_.angle_increment);
        int end_idx = static_cast<int>((M_PI/3 + latest_scan_.angle_min) / latest_scan_.angle_increment);

        start_idx = std::max(0, start_idx);
        end_idx = std::min(static_cast<int>(latest_scan_.ranges.size()), end_idx);

        double min_front_distance = std::numeric_limits<double>::infinity();
        int min_front_idx = -1;

        for (int i = start_idx; i < end_idx; ++i) {
            if (latest_scan_.ranges[i] < min_front_distance &&
                std::isfinite(latest_scan_.ranges[i])) {
                min_front_distance = latest_scan_.ranges[i];
                min_front_idx = i;
            }
        }

        if (min_front_distance < 0.5 && min_front_idx >= 0) {  // Obstacle in front
            // Calculate angle to closest obstacle
            double obstacle_angle = latest_scan_.angle_min + min_front_idx * latest_scan_.angle_increment;

            // Turn away from obstacle
            return -obstacle_angle * 0.5;  // Opposite direction with gain
        }

        return 0.0;  // No obstacle in front
    }

    bool isGoalReached(double distance_threshold = 0.2, double angle_threshold = 0.1)
    {
        double dx = goal_pose_.position.x - current_pose_.position.x;
        double dy = goal_pose_.position.y - current_pose_.position.y;
        double distance = sqrt(dx*dx + dy*dy);

        tf2::Quaternion q(
            current_pose_.orientation.x,
            current_pose_.orientation.y,
            current_pose_.orientation.z,
            current_pose_.orientation.w
        );
        tf2::Matrix3x3 m(q);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);

        double goal_yaw = atan2(dy, dx);
        double angle_diff = std::abs(yaw - goal_yaw);
        // Normalize angle difference
        if (angle_diff > M_PI) angle_diff = 2*M_PI - angle_diff;

        return distance < distance_threshold && angle_diff < angle_threshold;
    }

    void stopRobot()
    {
        geometry_msgs::msg::Twist stop_cmd;
        stop_cmd.linear.x = 0.0;
        stop_cmd.angular.z = 0.0;
        cmd_vel_pub_->publish(stop_cmd);
    }

    // Subscriptions
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;

    // Publisher
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;

    // Timer
    rclcpp::TimerBase::SharedPtr control_timer_;

    // Robot state
    geometry_msgs::msg::Pose current_pose_;
    geometry_msgs::msg::Twist current_twist_;
    geometry_msgs::msg::Pose goal_pose_;
    sensor_msgs::msg::LaserScan latest_scan_;

    // Flags
    bool has_odom_ = false;
    bool has_scan_ = false;
    bool has_goal_ = false;
};
```

## Humanoid Navigation Considerations

### Bipedal Navigation Challenges

Humanoid robots present unique navigation challenges compared to wheeled robots:

1. **Balance Maintenance**: Must maintain balance while moving
2. **Step Planning**: Requires discrete foot placement planning
3. **Dynamic Stability**: Center of mass shifts during locomotion
4. **Terrain Adaptation**: Different gaits for different surfaces

### Humanoid-Specific Navigation Strategies

```python
class HumanoidNavigationPlanner:
    def __init__(self):
        self.step_height = 0.05  # 5cm step height
        self.step_length = 0.3   # 30cm step length
        self.step_width = 0.2    # 20cm step width
        self.step_duration = 0.8 # 800ms per step

    def plan_bipedal_path(self, start_pose, goal_pose, environment_map):
        """Plan a path considering humanoid bipedal constraints"""
        # Use a path planning algorithm that considers humanoid kinematics
        path = self.plan_path_with_constraints(start_pose, goal_pose, environment_map)

        # Convert path to step sequence
        step_sequence = self.convert_path_to_steps(path)

        return step_sequence

    def convert_path_to_steps(self, path):
        """Convert continuous path to discrete step sequence for humanoid"""
        steps = []

        for i in range(1, len(path)):
            # Calculate step direction and distance
            dx = path[i].x - path[i-1].x
            dy = path[i].y - path[i-1].y
            step_distance = math.sqrt(dx*dx + dy*dy)

            # Determine which foot to step with (alternating)
            foot = 'left' if len(steps) % 2 == 0 else 'right'

            # Calculate step parameters
            step = {
                'foot': foot,
                'position': (path[i].x, path[i].y, path[i].z),
                'orientation': self.calculate_step_orientation(dx, dy),
                'height': self.step_height,
                'duration': self.step_duration
            }

            steps.append(step)

        return steps

    def execute_bipedal_navigation(self, step_sequence):
        """Execute navigation using bipedal locomotion"""
        for step in step_sequence:
            # Execute single step with balance control
            self.execute_single_step(step)

            # Wait for step completion
            time.sleep(step['duration'])

            # Verify balance and adjust if needed
            self.verify_balance()

    def execute_single_step(self, step):
        """Execute a single step with balance control"""
        # This would interface with the humanoid's walking controller
        # Implementation would include:
        # 1. Balance preparation
        # 2. Swing leg trajectory planning
        # 3. Step execution with balance feedback
        # 4. Post-step balance recovery
        pass

    def verify_balance(self):
        """Verify robot balance during navigation"""
        # Check center of mass position
        # Verify joint positions and velocities
        # Adjust if balance is compromised
        pass
```

## Isaac Sim Navigation Integration

### Complete Navigation Example

```python
# Complete Isaac Sim navigation example
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.navigation import NavigationGraph
from omni.isaac.range_sensor import RotatingLidarPhysX
from omni.isaac.sensor import Camera
import numpy as np

class IsaacSimNavigationExample:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.setup_complete_navigation_scene()

    def setup_complete_navigation_scene(self):
        """Set up a complete navigation scene with Isaac Sim"""
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            print("Could not find Isaac Sim assets. Please check your Isaac Sim installation.")
            return

        # Add a humanoid robot
        robot_path = assets_root_path + "/Isaac/Robots/Humanoid/humanoid_instanceable.usd"
        add_reference_to_stage(usd_path=robot_path, prim_path="/World/HumanoidRobot")

        # Add a complex navigation environment
        env_path = assets_root_path + "/Isaac/Environments/Office/office.usd"
        add_reference_to_stage(usd_path=env_path, prim_path="/World/OfficeEnvironment")

        # Add sensors for navigation
        self.camera = Camera(
            prim_path="/World/HumanoidRobot/base_link/camera",
            frequency=30,
            resolution=(640, 480)
        )

        self.lidar = RotatingLidarPhysX(
            prim_path="/World/HumanoidRobot/base_link/lidar",
            translation=np.array([0.0, 0.0, 0.5]),  # Higher for humanoid
            config="Carter_2D",  # Using Carter config as base
            rotation_frequency=10,
            samples_per_scan=1080
        )

        # Create navigation graph for path planning
        self.nav_graph = NavigationGraph(
            prim_path="/World/navigation_graph",
            scene_path="/World/OfficeEnvironment"
        )

        # Initialize the world
        self.world.reset()

    def run_complete_navigation(self):
        """Run complete navigation with VSLAM and path planning"""
        self.world.reset()

        # Define navigation goals
        goals = [
            [2.0, 2.0, 0.0],   # Goal 1
            [5.0, -1.0, 0.0],  # Goal 2
            [-1.0, -3.0, 0.0], # Goal 3
            [0.0, 0.0, 0.0]    # Return to start
        ]

        for goal in goals:
            print(f"Navigating to goal: {goal}")

            # Get current robot position
            robot_pos = self.get_robot_position()

            # Plan path using navigation graph
            path = self.nav_graph.plan_path(robot_pos, goal)

            if path:
                # Execute navigation with obstacle avoidance
                self.follow_path_with_obstacle_avoidance(path)
            else:
                print(f"No path found to goal: {goal}")

            # Wait a bit before next goal
            time.sleep(2)

    def follow_path_with_obstacle_avoidance(self, path):
        """Follow a path with dynamic obstacle avoidance"""
        for waypoint in path:
            # Move towards waypoint with local obstacle avoidance
            while not self.reached_waypoint(waypoint):
                # Get sensor data
                lidar_data = self.lidar.get_linear_depth_data()

                # Calculate navigation command
                cmd_vel = self.calculate_obstacle_aware_navigation(waypoint, lidar_data)

                # Execute command
                self.send_velocity_command(cmd_vel)

                # Step simulation
                self.world.step(render=True)

    def calculate_obstacle_aware_navigation(self, target_waypoint, lidar_data):
        """Calculate navigation command with obstacle avoidance"""
        # Get current robot state
        robot_pos = self.get_robot_position()
        robot_orient = self.get_robot_orientation()

        # Calculate direction to target
        dx = target_waypoint[0] - robot_pos[0]
        dy = target_waypoint[1] - robot_pos[1]
        target_angle = np.arctan2(dy, dx)

        # Get robot's current heading
        robot_yaw = self.orientation_to_yaw(robot_orient)

        # Calculate angle difference
        angle_diff = self.normalize_angle(target_angle - robot_yaw)

        # Base navigation command
        linear_vel = min(0.3, np.sqrt(dx*dx + dy*dy) * 0.5)  # Scale with distance
        angular_vel = angle_diff * 1.0

        # Check for obstacles using LIDAR data
        min_distance = np.min(lidar_data) if len(lidar_data) > 0 else float('inf')

        if min_distance < 0.8:  # Potential obstacle
            # Implement Dynamic Window Approach (DWA) for local planning
            cmd_vel = self.dwa_local_planning(
                linear_vel, angular_vel,
                lidar_data, robot_pos, target_waypoint
            )
        else:
            # Pure pursuit to target
            cmd_vel = geometry_msgs.msg.Twist()
            cmd_vel.linear.x = linear_vel
            cmd_vel.angular.z = angular_vel

        # Apply velocity limits
        cmd_vel.linear.x = np.clip(cmd_vel.linear.x, 0.0, 0.5)
        cmd_vel.angular.z = np.clip(cmd_vel.angular.z, -0.5, 0.5)

        return cmd_vel

    def dwa_local_planning(self, desired_linear, desired_angular, lidar_data, robot_pos, goal_pos):
        """Dynamic Window Approach for local path planning with obstacles"""
        # Define velocity search space
        v_min = 0.0
        v_max = 0.5
        w_min = -0.5
        w_max = 0.5

        # Define acceleration limits
        acc_lin = 0.5  # m/s^2
        acc_ang = 1.0  # rad/s^2
        dt = 0.1  # time step

        # Get current velocities (would come from robot state)
        current_v = 0.1  # placeholder
        current_w = 0.0  # placeholder

        # Calculate velocity windows
        v_window = [max(v_min, current_v - acc_lin*dt), min(v_max, current_v + acc_lin*dt)]
        w_window = [max(w_min, current_w - acc_ang*dt), min(w_max, current_w + acc_ang*dt)]

        best_score = -float('inf')
        best_cmd = geometry_msgs.msg.Twist()

        # Sample velocities in the window
        for v_sample in np.linspace(v_window[0], v_window[1], 10):
            for w_sample in np.linspace(w_window[0], w_window[1], 10):
                # Simulate trajectory
                sim_positions = self.simulate_trajectory(robot_pos, current_v, current_w, v_sample, w_sample, dt)

                # Evaluate trajectory
                goal_score = self.evaluate_goal_distance(sim_positions, goal_pos)
                obs_score = self.evaluate_obstacle_clearance(sim_positions, lidar_data)
                speed_score = self.evaluate_speed(v_sample)

                # Weighted score
                total_score = 0.8 * goal_score + 0.1 * obs_score + 0.1 * speed_score

                if total_score > best_score:
                    best_score = total_score
                    best_cmd.linear.x = v_sample
                    best_cmd.angular.z = w_sample

        return best_cmd

    def simulate_trajectory(self, start_pos, v_start, w_start, v_target, w_target, dt, steps=5):
        """Simulate robot trajectory over time"""
        positions = [start_pos]
        pos = [start_pos[0], start_pos[1], start_pos[2]]
        theta = 0  # Would get from robot orientation

        for i in range(steps):
            # Simple kinematic model for differential drive
            pos[0] += v_target * np.cos(theta) * dt
            pos[1] += v_target * np.sin(theta) * dt
            theta += w_target * dt

            positions.append(pos.copy())

        return positions

    def evaluate_goal_distance(self, trajectory, goal_pos):
        """Evaluate how close the trajectory gets to the goal"""
        if not trajectory:
            return -float('inf')

        final_pos = trajectory[-1]
        distance = np.sqrt((final_pos[0] - goal_pos[0])**2 + (final_pos[1] - goal_pos[1])**2)
        return 1.0 / (1.0 + distance)  # Higher score for closer distances

    def evaluate_obstacle_clearance(self, trajectory, lidar_data):
        """Evaluate how well the trajectory avoids obstacles"""
        if not trajectory or len(lidar_data) == 0:
            return -float('inf')

        min_clearance = float('inf')

        for pos in trajectory:
            # Convert position to polar coordinates relative to robot
            # and check against LIDAR data
            pass

        return min_clearance  # Higher score for greater clearance

    def evaluate_speed(self, velocity):
        """Evaluate desirability of a particular speed"""
        # Prefer faster but safe speeds
        return velocity

    def send_velocity_command(self, cmd_vel):
        """Send velocity command to robot in Isaac Sim"""
        # This would interface with the robot's controller
        # In Isaac Sim, this might involve setting joint velocities
        # or using a ROS interface if ROS bridge is enabled
        pass
```

## Performance Optimization

### VSLAM Optimization Techniques

```cpp
// Optimized VSLAM node with performance considerations
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class OptimizedVSLAMNode : public rclcpp::Node
{
public:
    OptimizedVSLAMNode() : Node("optimized_vslam_node")
    {
        // Use intra-process communication where possible
        rclcpp::QoS qos(10);
        qos.best_effort();  // For sensor data

        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "camera/image_rect_color", qos,
            std::bind(&OptimizedVSLAMNode::imageCallback, this, std::placeholders::_1)
        );

        // Throttle processing if needed
        processing_rate_ = this->declare_parameter("processing_rate", 15.0);  // Hz
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(1000.0 / processing_rate_)),
            std::bind(&OptimizedVSLAMNode::processCallback, this)
        );

        pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            "visual_slam/pose", 10
        );

        // Initialize feature detector with optimized parameters
        orb_detector_ = cv::ORB::create(
            1000,        // nfeatures
            1.2f,        // scaleFactor
            4,           // nlevels
            31,          // edgeThreshold
            0,           // firstLevel
            2,           // WTA_K
            cv::ORB::HARRIS_SCORE,  // scoreType
            31,          // patchSize
            20           // fastThreshold
        );

        RCLCPP_INFO(this->get_logger(), "Optimized VSLAM node initialized");
    }

private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // Store image for processing at fixed rate
        latest_image_ = msg;
        image_available_ = true;
    }

    void processCallback()
    {
        if (!image_available_) return;

        try {
            // Convert ROS image to OpenCV
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(latest_image_, "bgr8");

            // Process image for VSLAM
            auto pose_estimate = processVSLAM(cv_ptr->image);

            if (pose_estimate.has_value()) {
                // Publish pose estimate
                auto pose_msg = geometry_msgs::msg::PoseStamped();
                pose_msg.header = latest_image_->header;
                pose_msg.pose = pose_estimate.value();
                pose_pub_->publish(pose_msg);
            }

        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }

        image_available_ = false;
    }

    std::optional<geometry_msgs::msg::Pose> processVSLAM(const cv::Mat& image)
    {
        // Convert to grayscale for feature detection
        cv::Mat gray_image;
        if (image.channels() == 3) {
            cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
        } else {
            gray_image = image;
        }

        // Detect and compute features
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        orb_detector_->detectAndCompute(gray_image, cv::noArray(), keypoints, descriptors);

        if (keypoints.size() < 50) {  // Require minimum features
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                1000,  // 1 second throttle
                "Insufficient features detected: %zu",
                keypoints.size()
            );
            return std::nullopt;
        }

        // In a real implementation, this would include:
        // - Feature matching with previous frame
        // - Pose estimation using PnP or similar
        // - Bundle adjustment
        // - Loop closure detection
        // - Map maintenance

        // For this example, return a dummy pose (would be calculated from features)
        geometry_msgs::msg::Pose pose;
        pose.position.x += 0.01;  // Simulate forward movement
        pose.orientation.w = 1.0;  // Unit quaternion

        return pose;
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    sensor_msgs::msg::Image::SharedPtr latest_image_;
    bool image_available_ = false;
    double processing_rate_;

    cv::Ptr<cv::ORB> orb_detector_;
    std::vector<cv::KeyPoint> previous_keypoints_;
    cv::Mat previous_descriptors_;
};
```

## Troubleshooting Common Issues

### 1. Drift in VSLAM
**Problem**: Accumulated errors causing position drift over time
**Solutions**:
- Implement loop closure detection
- Use sensor fusion with IMU/odometry
- Regular relocalization against known features

### 2. Low-Texture Environments
**Problem**: Insufficient features for tracking
**Solutions**:
- Use direct methods (LSD-SLAM, DSO)
- Add artificial markers or fiducials
- Combine with other sensors (LIDAR, depth)

### 3. Dynamic Objects
**Problem**: Moving objects affecting map/pose estimation
**Solutions**:
- Implement dynamic object detection and filtering
- Use semantic segmentation to identify static objects
- Temporal consistency checks

### 4. Computational Performance
**Problem**: VSLAM consuming too many resources
**Solutions**:
- Reduce feature count in parameters
- Lower processing frequency
- Use GPU acceleration where available
- Optimize feature detection algorithms

## Best Practices

### 1. Robust Initialization
- Ensure good initial pose estimate
- Verify camera calibration
- Check lighting conditions

### 2. Parameter Tuning
- Adjust parameters based on environment
- Monitor performance metrics
- Use adaptive parameters when possible

### 3. Sensor Fusion
- Combine VSLAM with other sensors
- Use IMU for motion prediction
- Integrate with wheel odometry

### 4. Performance Monitoring
- Track processing time per frame
- Monitor memory usage
- Validate trajectory accuracy

## Exercise

Create a complete VSLAM and navigation system that includes:

1. Implement a VSLAM pipeline using Isaac ROS components
2. Integrate the VSLAM system with Nav2 for localization
3. Create a navigation scene in Isaac Sim with obstacles
4. Implement obstacle avoidance in the navigation system
5. Test the complete system in simulation
6. Evaluate the system's performance in terms of localization accuracy and navigation success rate

The system should be able to:
- Build a map of an unknown environment
- Localize the robot within the map
- Navigate to specified goals while avoiding obstacles
- Handle dynamic environments and recover from localization failures

Test your system with various scenarios including:
- Navigation in cluttered environments
- Recovery from localization failures
- Performance under different lighting conditions
- Robustness to sensor noise