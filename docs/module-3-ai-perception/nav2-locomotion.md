# Nav2 for Humanoid Locomotion

Navigation 2 (Nav2) is the ROS 2 navigation stack that enables autonomous navigation for robots. For humanoid robots, Nav2 requires special configuration to handle the unique challenges of bipedal locomotion. This section covers configuring and using Nav2 for humanoid robot navigation.

## Introduction to Nav2

Nav2 is a complete navigation system that includes:
- **Global Planner**: Plans path from start to goal
- **Local Planner**: Executes path while avoiding obstacles
- **Controller**: Sends commands to robot actuators
- **Costmaps**: Represents obstacles and free space
- **Behavior Trees**: Orchestrates navigation behaviors
- **Recovery Behaviors**: Handles navigation failures

## Nav2 Architecture

### Core Components
```
Nav2 System
├── Navigation Server
│   ├── Global Planner (NavFn, A*, etc.)
│   ├── Local Planner (DWA, TEB, etc.)
│   └── Controller (PID, MPC, etc.)
├── Costmap Server
│   ├── Global Costmap
│   └── Local Costmap
├── Lifecycle Manager
├── Behavior Tree Navigator
├── Recovery Server
└── Transform Management (TF2)
```

### Humanoid-Specific Considerations
Humanoid robots require special handling in Nav2:
- **Bipedal dynamics**: Different from wheeled robots
- **Balance constraints**: Must maintain center of mass
- **Step planning**: Requires discrete foot placement
- **Stair navigation**: Special locomotion patterns needed

## Nav2 Configuration for Humanoid Robots

### Basic Parameters Configuration

```yaml
# nav2_params_humanoid.yaml
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
    global_frame_id: "map"
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
    robot_model_type: "nav2_amcl::DifferentialMotionModel"  # This needs humanoid-specific model
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

amcl_map_client:
  ros__parameters:
    use_sim_time: True

amcl_rclcpp_node:
  ros__parameters:
    use_sim_time: True

bt_navigator:
  ros__parameters:
    use_sim_time: True
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom
    bt_loop_duration: 10
    default_server_timeout: 20
    enable_groot_monitoring: True
    groot_zmq_publisher_port: 1666
    groot_zmq_server_port: 1667
    default_nav_through_poses_bt_xml: nav2_bt_navigator/navigate_through_poses_w_replanning_and_recovery.xml
    default_nav_to_pose_bt_xml: nav2_bt_navigator/navigate_to_pose_w_replanning_and_recovery.xml
    plugin_lib_names:
    - nav2_compute_path_to_pose_action_bt_node
    - nav2_compute_path_through_poses_action_bt_node
    - nav2_smooth_path_action_bt_node
    - nav2_follow_path_action_bt_node
    - nav2_spin_action_bt_node
    - nav2_wait_action_bt_node
    - nav2_assisted_teleop_action_bt_node
    - nav2_back_up_action_bt_node
    - nav2_drive_on_heading_bt_node
    - nav2_clear_costmap_service_bt_node
    - nav2_is_stuck_condition_bt_node
    - nav2_goal_reached_condition_bt_node
    - nav2_goal_updated_condition_bt_node
    - nav2_globally_consistent_condition_bt_node
    - nav2_is_path_valid_condition_bt_node
    - nav2_initial_pose_received_condition_bt_node
    - nav2_reinitialize_global_localization_service_bt_node
    - nav2_rate_controller_bt_node
    - nav2_distance_controller_bt_node
    - nav2_speed_controller_bt_node
    - nav2_truncate_path_action_bt_node
    - nav2_truncate_path_local_action_bt_node
    - nav2_goal_updater_node_bt_node
    - nav2_recovery_node_bt_node
    - nav2_pipeline_sequence_bt_node
    - nav2_round_robin_node_bt_node
    - nav2_transform_available_condition_bt_node
    - nav2_time_expired_condition_bt_node
    - nav2_path_expiring_timer_condition
    - nav2_distance_traveled_condition_bt_node
    - nav2_is_battery_low_condition_bt_node
    - nav2_navigate_through_poses_action_bt_node
    - nav2_navigate_to_pose_action_bt_node
    - nav2_remove_passed_goals_action_bt_node
    - nav2_planner_selector_bt_node
    - nav2_controller_selector_bt_node
    - nav2_goal_checker_selector_bt_node
    - nav2_controller_cancel_bt_node
    - nav2_path_longer_on_approach_bt_node
    - nav2_wait_cancel_bt_node
    - nav2_spin_cancel_bt_node
    - nav2_back_up_cancel_bt_node
    - nav2_assisted_teleop_cancel_bt_node
    - nav2_drive_on_heading_cancel_bt_node

bt_navigator_rclcpp_node:
  ros__parameters:
    use_sim_time: True

controller_server:
  ros__parameters:
    use_sim_time: True
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    # Humanoid-specific controller
    FollowPath:
      plugin: "nav2_mppi_controller::MppiController"  # Or custom humanoid controller
      time_steps: 20
      control_frequency: 10.0
      motion_model: "DiffDrive"  # Need humanoid-specific model
      xy_goal_tolerance: 0.25
      yaw_goal_tolerance: 0.25
      state_bounds_planner: 0.3
      control_bounds_planner: 0.5
      control_bounds: 1.0
      state_bounds: 0.5
      dt: 0.05
      noise_coefficient: 0.0
      convergence_integrator_gain: 0.1
      oscillation_threshold: 0.01
      oscillation_window: 10

controller_server_rclcpp_node:
  ros__parameters:
    use_sim_time: True

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: odom
      robot_base_frame: base_link
      use_sim_time: True
      rolling_window: true
      width: 6
      height: 6
      resolution: 0.05
      robot_radius: 0.3  # Adjust for humanoid robot size
      plugins: ["voxel_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: True
        publish_voxel_map: True
        origin_z: 0.0
        z_resolution: 0.2
        z_voxels: 8
        max_obstacle_height: 2.0
        mark_threshold: 0
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
  local_costmap_client:
    ros__parameters:
      use_sim_time: True
  local_costmap_rclcpp_node:
    ros__parameters:
      use_sim_time: True

global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 1.0
      global_frame: map
      robot_base_frame: base_link
      use_sim_time: True
      robot_radius: 0.3  # Adjust for humanoid robot
      resolution: 0.05
      track_unknown_space: true
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: True
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
  global_costmap_client:
    ros__parameters:
      use_sim_time: True
  global_costmap_rclcpp_node:
    ros__parameters:
      use_sim_time: True

map_server:
  ros__parameters:
    use_sim_time: True
    yaml_filename: "turtlebot3_world.yaml"

map_saver:
  ros__parameters:
    use_sim_time: True
    save_map_timeout: 5.0
    free_thresh_default: 0.25
    occupied_thresh_default: 0.65

planner_server:
  ros__parameters:
    use_sim_time: True
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner::NavfnPlanner"
      tolerance: 0.5
      use_astar: false
      allow_unknown: true

planner_server_rclcpp_node:
  ros__parameters:
    use_sim_time: True

recoveries_server:
  ros__parameters:
    use_sim_time: True
    recovery_plugins: ["spin", "backup", "wait"]
    spin:
      plugin: "nav2_recoveries::Spin"
      ideal_linear_velocity: 0.0
      ideal_angular_velocity: 1.0
      tolerance: 1.57
      sampling_frequency: 20.0
      cmd_vel_timeout: 1.0
    backup:
      plugin: "nav2_recoveries::BackUp"
      ideal_linear_velocity: -0.1
      ideal_angular_velocity: 0.0
      tolerance: 0.15
      sampling_frequency: 10.0
      cmd_vel_timeout: 1.0
    wait:
      plugin: "nav2_recoveries::Wait"
      sleep_duration: 2.0
      sampling_frequency: 10.0
      cmd_vel_timeout: 1.0

robot_state_publisher:
  ros__parameters:
    use_sim_time: True
```

## Humanoid-Specific Navigation Challenges

### 1. Bipedal Dynamics
Humanoid robots have different locomotion characteristics:

```cpp
// Humanoid motion model for navigation
class HumanoidMotionModel
{
public:
    HumanoidMotionModel()
    {
        // Humanoid-specific parameters
        step_length_ = 0.3;      // Average step length
        step_duration_ = 0.8;    // Time for one step
        max_step_width_ = 0.2;   // Maximum lateral step
        balance_margin_ = 0.1;   // Safety margin for balance
    }

    geometry_msgs::msg::Pose predictPose(
        const geometry_msgs::msg::Pose& current_pose,
        const geometry_msgs::msg::Twist& cmd_vel,
        double dt)
    {
        // Implement humanoid-specific motion prediction
        // Consider step-by-step locomotion instead of continuous motion
        geometry_msgs::msg::Pose predicted_pose = current_pose;

        // Calculate step-based movement
        double steps = dt / step_duration_;
        double step_dx = cmd_vel.linear.x * step_duration_;
        double step_dy = cmd_vel.linear.y * step_duration_;
        double step_dtheta = cmd_vel.angular.z * step_duration_;

        // Apply step constraints
        step_dx = std::min(step_dx, step_length_);
        step_dy = std::min(step_dy, max_step_width_);

        // Update pose based on steps
        predicted_pose.position.x += step_dx * steps;
        predicted_pose.position.y += step_dy * steps;

        // Update orientation
        tf2::Quaternion q(
            predicted_pose.orientation.x,
            predicted_pose.orientation.y,
            predicted_pose.orientation.z,
            predicted_pose.orientation.w
        );
        tf2::Matrix3x3 m(q);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);
        yaw += step_dtheta * steps;

        // Convert back to quaternion
        q.setRPY(roll, pitch, yaw);
        predicted_pose.orientation.x = q.x();
        predicted_pose.orientation.y = q.y();
        predicted_pose.orientation.z = q.z();
        predicted_pose.orientation.w = q.w();

        return predicted_pose;
    }

private:
    double step_length_;
    double step_duration_;
    double max_step_width_;
    double balance_margin_;
};
```

### 2. Step Planning
Humanoid robots need discrete step planning:

```cpp
// Step planner for humanoid navigation
class StepPlanner
{
public:
    struct Step
    {
        geometry_msgs::msg::Point left_foot;
        geometry_msgs::msg::Point right_foot;
        double time;
    };

    std::vector<Step> planSteps(
        const geometry_msgs::msg::Pose& start,
        const geometry_msgs::msg::Pose& goal,
        const nav_msgs::msg::Path& global_path)
    {
        std::vector<Step> steps;

        // Plan discrete steps along the path
        // Ensure each step maintains balance
        for (size_t i = 0; i < global_path.poses.size(); i += step_spacing_)
        {
            Step step;
            step.left_foot = calculateLeftFootPosition(global_path.poses[i].pose);
            step.right_foot = calculateRightFootPosition(global_path.poses[i].pose);
            step.time = i * step_duration_;

            // Verify step is balanced
            if (isStepBalanced(step))
            {
                steps.push_back(step);
            }
        }

        return steps;
    }

private:
    bool isStepBalanced(const Step& step)
    {
        // Check if the step maintains center of mass within support polygon
        geometry_msgs::msg::Point com = calculateCOM();
        return isInSupportPolygon(com, step);
    }

    geometry_msgs::msg::Point calculateLeftFootPosition(const geometry_msgs::msg::Pose& pose);
    geometry_msgs::msg::Point calculateRightFootPosition(const geometry_msgs::msg::Pose& pose);
    geometry_msgs::msg::Point calculateCOM();
    bool isInSupportPolygon(const geometry_msgs::msg::Point& com, const Step& step);

    size_t step_spacing_ = 2;  // Plan every 2nd point from path
    double step_duration_ = 0.8;
};
```

### 3. Balance Controller
Maintain balance during navigation:

```cpp
// Balance controller for humanoid navigation
class BalanceController
{
public:
    BalanceController()
    {
        // Initialize balance control parameters
        com_height_ = 0.8;  // Height of center of mass
        control_frequency_ = 100.0;  // Balance control frequency
    }

    geometry_msgs::msg::Twist computeBalanceControl(
        const geometry_msgs::msg::Pose& current_pose,
        const geometry_msgs::msg::Twist& desired_twist)
    {
        geometry_msgs::msg::Twist balance_twist = desired_twist;

        // Calculate center of mass position and velocity
        geometry_msgs::msg::Point com = calculateCOM();
        geometry_msgs::msg::Vector3 com_vel = calculateCOMVelocity();

        // Calculate Zero Moment Point (ZMP)
        geometry_msgs::msg::Point zmp = calculateZMP(com, com_vel);

        // Calculate balance error
        double balance_error_x = zmp.x - desired_zmp_.x;
        double balance_error_y = zmp.y - desired_zmp_.y;

        // Apply balance correction using PID control
        balance_twist.linear.x += balance_pid_x_.compute(balance_error_x);
        balance_twist.linear.y += balance_pid_y_.compute(balance_error_y);

        // Limit corrections to maintain stability
        balance_twist.linear.x = std::clamp(balance_twist.linear.x, -0.1, 0.1);
        balance_twist.linear.y = std::clamp(balance_twist.linear.y, -0.05, 0.05);

        return balance_twist;
    }

private:
    geometry_msgs::msg::Point calculateCOM();
    geometry_msgs::msg::Vector3 calculateCOMVelocity();
    geometry_msgs::msg::Point calculateZMP(
        const geometry_msgs::msg::Point& com,
        const geometry_msgs::msg::Vector3& com_vel);

    geometry_msgs::msg::Point desired_zmp_;
    double com_height_;
    double control_frequency_;

    // PID controllers for balance
    PIDController balance_pid_x_;
    PIDController balance_pid_y_;
};
```

## Nav2 Launch Configuration

### Complete Launch File

```python
# launch/humanoid_nav2.launch.py
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    # Get the launch directory
    package_dir = get_package_share_directory('humanoid_navigation')

    # Create the launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    params_file = LaunchConfiguration('params_file')
    bt_xml_filename = LaunchConfiguration('bt_xml_filename')
    map_subscribe_transient_local = LaunchConfiguration('map_subscribe_transient_local')

    # Declare the launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true')

    declare_autostart = DeclareLaunchArgument(
        'autostart',
        default_value='true',
        description='Automatically startup the nav2 stack')

    declare_params_file = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(package_dir, 'config', 'nav2_params_humanoid.yaml'),
        description='Full path to the ROS2 parameters file to use for all launched nodes')

    declare_bt_xml = DeclareLaunchArgument(
        'bt_xml_filename',
        default_value=os.path.join(
            get_package_share_directory('nav2_bt_navigator'),
            'behavior_trees', 'navigate_w_replanning_and_recovery.xml'),
        description='Full path to the behavior tree xml file to use')

    declare_map_subscribe_transient_local = DeclareLaunchArgument(
        'map_subscribe_transient_local',
        default_value='false',
        description='Whether to set the map subscriber QoS to transient local')

    # Make sure we have the right parameters file
    param_substitutions = {
        'use_sim_time': use_sim_time,
        'autostart': autostart,
        'bt_xml_filename': bt_xml_filename,
        'map_subscribe_transient_local': map_subscribe_transient_local}

    configured_params = RewrittenYaml(
        source_file=params_file,
        root_key='',
        param_rewrites=param_substitutions,
        convert_types=True)

    # Specify the actions
    lifecycle_nodes = ['controller_server',
                       'planner_server',
                       'recoveries_server',
                       'bt_navigator',
                       'waypoint_follower']

    # Create the node
    navigation_server = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time},
                    {'autostart': autostart},
                    {'node_names': lifecycle_nodes}])

    # Localization node (AMCL)
    localization = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[configured_params])

    # Controller server
    controller_server = Node(
        package='nav2_controller',
        executable='controller_server',
        output='screen',
        parameters=[configured_params])

    # Planner server
    planner_server = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[configured_params])

    # BT navigator
    bt_navigator = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        output='screen',
        parameters=[configured_params])

    # Recovery server
    recovery_server = Node(
        package='nav2_recoveries',
        executable='recoveries_server',
        name='recoveries_server',
        output='screen',
        parameters=[configured_params])

    # Waypoint follower
    waypoint_follower = Node(
        package='nav2_waypoint_follower',
        executable='waypoint_follower',
        name='waypoint_follower',
        output='screen',
        parameters=[configured_params])

    # Add the actions to the launch description
    ld = LaunchDescription()

    # Declare the launch options
    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_autostart)
    ld.add_action(declare_params_file)
    ld.add_action(declare_bt_xml)
    ld.add_action(declare_map_subscribe_transient_local)

    # Add the nodes to the launch description
    ld.add_action(navigation_server)
    ld.add_action(localization)
    ld.add_action(controller_server)
    ld.add_action(planner_server)
    ld.add_action(bt_navigator)
    ld.add_action(recovery_server)
    ld.add_action(waypoint_follower)

    return ld
```

## Custom Controllers for Humanoid Robots

### Humanoid Path Following Controller

```cpp
#include <nav2_core/controller.hpp>
#include <nav2_util/lifecycle_node.hpp>
#include <nav2_costmap_2d/costmap_2d_ros.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>

class HumanoidController : public nav2_core::Controller
{
public:
    HumanoidController() = default;
    ~HumanoidController() = default;

    void configure(
        const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
        std::string name,
        const std::shared_ptr<tf2_ros::Buffer> & tf,
        const std::shared_ptr<nav2_costmap_2d::Costmap2DROS> & costmap_ros) override
    {
        node_ = parent.lock();
        name_ = name;
        tf_ = tf;
        costmap_ros_ = costmap_ros;
        costmap_ = costmap_ros_->getCostmap();

        // Initialize humanoid-specific parameters
        step_frequency_ = node_->declare_parameter(name_ + ".step_frequency", 1.25);
        max_step_length_ = node_->declare_parameter(name_ + ".max_step_length", 0.3);
        balance_margin_ = node_->declare_parameter(name_ + ".balance_margin", 0.1);

        RCLCPP_INFO(node_->get_logger(), "HumanoidController configured");
    }

    void cleanup() override
    {
        RCLCPP_INFO(node_->get_logger(), "HumanoidController cleaned up");
    }

    void activate() override
    {
        RCLCPP_INFO(node_->get_logger(), "HumanoidController activated");
    }

    void deactivate() override
    {
        RCLCPP_INFO(node_->get_logger(), "HumanoidController deactivated");
    }

    geometry_msgs::msg::Twist computeVelocityCommands(
        const geometry_msgs::msg::PoseStamped & pose,
        const geometry_msgs::msg::Twist & velocity,
        nav2_core::GoalChecker * goal_checker) override
    {
        geometry_msgs::msg::Twist cmd_vel;

        if (current_path_.poses.empty()) {
            RCLCPP_WARN(node_->get_logger(), "No path received, stopping robot");
            return cmd_vel;  // Return zero velocity
        }

        // Find closest point on path
        size_t closest_idx = findClosestPoseIndex(pose, current_path_);

        // Calculate desired velocity based on path
        cmd_vel = calculatePathFollowingVelocity(pose, current_path_, closest_idx);

        // Apply humanoid-specific constraints
        cmd_vel = applyHumanoidConstraints(cmd_vel, pose);

        // Check for obstacles in local costmap
        if (isPathObstructed(pose, cmd_vel)) {
            cmd_vel = handleObstacles(cmd_vel, pose);
        }

        return cmd_vel;
    }

    void setPlan(const nav_msgs::msg::Path & path) override
    {
        current_path_ = path;
        RCLCPP_INFO(node_->get_logger(), "New plan set with %zu waypoints", path.poses.size());
    }

private:
    size_t findClosestPoseIndex(
        const geometry_msgs::msg::PoseStamped & pose,
        const nav_msgs::msg::Path & path)
    {
        double min_dist = std::numeric_limits<double>::max();
        size_t closest_idx = 0;

        for (size_t i = 0; i < path.poses.size(); ++i) {
            double dist = euclideanDistance(pose.pose.position, path.poses[i].pose.position);
            if (dist < min_dist) {
                min_dist = dist;
                closest_idx = i;
            }
        }

        return closest_idx;
    }

    geometry_msgs::msg::Twist calculatePathFollowingVelocity(
        const geometry_msgs::msg::PoseStamped & pose,
        const nav_msgs::msg::Path & path,
        size_t closest_idx)
    {
        geometry_msgs::msg::Twist cmd_vel;

        if (closest_idx >= path.poses.size() - 1) {
            // At end of path, slow down
            cmd_vel.linear.x = 0.0;
            cmd_vel.angular.z = 0.0;
            return cmd_vel;
        }

        // Calculate direction to follow path
        auto target_pose = path.poses[std::min(closest_idx + 1, path.poses.size() - 1)];

        // Calculate error to path
        double dx = target_pose.pose.position.x - pose.pose.position.x;
        double dy = target_pose.pose.position.y - pose.pose.position.y;

        // Convert to robot frame
        double yaw = tf2::getYaw(pose.pose.orientation);
        double local_dx = dx * cos(yaw) + dy * sin(yaw);
        double local_dy = -dx * sin(yaw) + dy * cos(yaw);

        // PID-like control for path following
        cmd_vel.linear.x = std::min(max_linear_speed_,
                                   std::max(-max_linear_speed_, local_dx * linear_gain_));
        cmd_vel.linear.y = std::min(max_linear_speed_,
                                   std::max(-max_linear_speed_, local_dy * linear_gain_));
        cmd_vel.angular.z = std::min(max_angular_speed_,
                                    std::max(-max_angular_speed_, -local_dy * angular_gain_));

        return cmd_vel;
    }

    geometry_msgs::msg::Twist applyHumanoidConstraints(
        const geometry_msgs::msg::Twist & raw_cmd,
        const geometry_msgs::msg::PoseStamped & pose)
    {
        geometry_msgs::msg::Twist constrained_cmd = raw_cmd;

        // Apply humanoid-specific velocity limits
        constrained_cmd.linear.x = std::clamp(constrained_cmd.linear.x,
                                            -max_step_length_ * step_frequency_,
                                            max_step_length_ * step_frequency_);
        constrained_cmd.linear.y = std::clamp(constrained_cmd.linear.y,
                                            -max_step_width_ * step_frequency_,
                                            max_step_width_ * step_frequency_);
        constrained_cmd.angular.z = std::clamp(constrained_cmd.angular.z,
                                             -max_angular_velocity_,
                                             max_angular_velocity_);

        return constrained_cmd;
    }

    bool isPathObstructed(
        const geometry_msgs::msg::PoseStamped & pose,
        const geometry_msgs::msg::Twist & cmd_vel)
    {
        // Check costmap for obstacles in the robot's path
        unsigned int mx, my;
        if (!costmap_->worldToMap(pose.pose.position.x, pose.pose.position.y, mx, my)) {
            return true;  // Robot position not in costmap
        }

        // Check ahead in the direction of movement
        double ahead_x = pose.pose.position.x + cmd_vel.linear.x * 0.5;  // 0.5 seconds ahead
        double ahead_y = pose.pose.position.y + cmd_vel.linear.y * 0.5;

        unsigned int ahead_mx, ahead_my;
        if (costmap_->worldToMap(ahead_x, ahead_y, ahead_mx, ahead_my)) {
            unsigned char cost = costmap_->getCost(ahead_mx, ahead_my);
            return cost > nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE;
        }

        return false;
    }

    geometry_msgs::msg::Twist handleObstacles(
        const geometry_msgs::msg::Twist & cmd_vel,
        const geometry_msgs::msg::PoseStamped & pose)
    {
        // Implement humanoid-specific obstacle avoidance
        geometry_msgs::msg::Twist avoidance_cmd = cmd_vel;

        // For humanoid robots, we might need to step around obstacles
        // rather than just turn
        avoidance_cmd.linear.x = 0.0;  // Stop forward motion
        avoidance_cmd.angular.z = 0.0;  // Stop turning

        // Implement step-based obstacle avoidance
        // This could involve planning discrete steps around obstacles
        // which is more complex than simple velocity modification

        return avoidance_cmd;
    }

    double euclideanDistance(const geometry_msgs::msg::Point & p1,
                           const geometry_msgs::msg::Point & p2)
    {
        return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
    }

    // Member variables
    rclcpp_lifecycle::LifecycleNode::SharedPtr node_;
    std::string name_;
    std::shared_ptr<tf2_ros::Buffer> tf_;
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
    nav2_costmap_2d::Costmap2D * costmap_;
    nav_msgs::msg::Path current_path_;

    // Humanoid-specific parameters
    double step_frequency_;
    double max_step_length_;
    double max_step_width_;
    double balance_margin_;
    double max_linear_speed_ = 0.5;
    double max_angular_speed_ = 0.5;
    double max_angular_velocity_ = 0.5;
    double linear_gain_ = 1.0;
    double angular_gain_ = 2.0;
};
```

## Integration with Isaac Sim

### Isaac Sim Navigation Setup

```python
# Isaac Sim navigation integration
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
import carb

class IsaacSimNavigation:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.setup_scene()

    def setup_scene(self):
        # Add humanoid robot
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets. Please check your Isaac Sim installation.")
            return

        # Add a humanoid robot to the scene
        humanoid_asset_path = assets_root_path + "/Isaac/Robots/Humanoid/humanoid.usd"
        add_reference_to_stage(usd_path=humanoid_asset_path, prim_path="/World/Humanoid")

        # Add a simple environment
        room_asset_path = assets_root_path + "/Isaac/Environments/Simple_Room/simple_room.usd"
        add_reference_to_stage(usd_path=room_asset_path, prim_path="/World/Room")

        # Initialize the world
        self.world.reset()

    def setup_navigation(self):
        # Configure navigation-specific components
        # This would include setting up sensors, costmaps, etc.
        pass

    def run_navigation(self, goal_position):
        # Main navigation loop
        while not self.world.is_stopped():
            self.world.step(render=True)

            # Get robot position
            robot_position = self.get_robot_position()

            # Check if reached goal
            if self.is_at_goal(robot_position, goal_position):
                print("Reached goal!")
                break

            # Continue navigation
            self.navigate_to_goal(goal_position)

    def get_robot_position(self):
        # Get current robot position from Isaac Sim
        pass

    def navigate_to_goal(self, goal_position):
        # Send navigation commands to robot
        pass

    def is_at_goal(self, current_pos, goal_pos):
        # Check if robot is at goal position
        distance = ((current_pos[0] - goal_pos[0])**2 +
                   (current_pos[1] - goal_pos[1])**2)**0.5
        return distance < 0.2  # 20cm tolerance
```

## Performance Evaluation

### Navigation Metrics for Humanoid Robots

```cpp
// Navigation performance evaluation
class NavigationEvaluator
{
public:
    struct NavigationMetrics
    {
        double success_rate;
        double average_time_to_goal;
        double path_efficiency;  // actual_path_length / optimal_path_length
        double collision_count;
        double balance_loss_count;
        double step_success_rate;
    };

    NavigationMetrics evaluateNavigation(
        const std::vector<geometry_msgs::msg::Pose>& trajectory,
        const geometry_msgs::msg::Pose& goal,
        bool navigation_successful)
    {
        NavigationMetrics metrics;

        // Calculate success rate
        metrics.success_rate = navigation_successful ? 1.0 : 0.0;

        // Calculate path efficiency
        double actual_path_length = calculatePathLength(trajectory);
        double optimal_path_length = calculateEuclideanDistance(trajectory.front().position, goal.position);
        metrics.path_efficiency = optimal_path_length > 0 ? actual_path_length / optimal_path_length : 1.0;

        // Count collisions (high cost areas in trajectory)
        metrics.collision_count = countCollisions(trajectory);

        // Count balance losses (if available)
        metrics.balance_loss_count = countBalanceLosses(trajectory);

        return metrics;
    }

private:
    double calculatePathLength(const std::vector<geometry_msgs::msg::Pose>& trajectory)
    {
        double length = 0.0;
        for (size_t i = 1; i < trajectory.size(); ++i) {
            length += calculateEuclideanDistance(
                trajectory[i-1].position,
                trajectory[i].position
            );
        }
        return length;
    }

    double calculateEuclideanDistance(
        const geometry_msgs::msg::Point& p1,
        const geometry_msgs::msg::Point& p2)
    {
        return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
    }

    int countCollisions(const std::vector<geometry_msgs::msg::Pose>& trajectory)
    {
        // Count poses in high-cost areas of costmap
        int collision_count = 0;
        // Implementation would check each pose against costmap
        return collision_count;
    }

    int countBalanceLosses(const std::vector<geometry_msgs::msg::Pose>& trajectory)
    {
        // Count instances where robot lost balance
        // This would require access to balance metrics
        return 0;
    }
};
```

## Troubleshooting Navigation Issues

### Common Problems and Solutions

#### 1. Robot Gets Stuck
**Problem**: Robot stops moving or oscillates in place
**Solutions**:
- Check costmap inflation settings
- Verify sensor data is being received
- Adjust local planner parameters
- Implement proper recovery behaviors

#### 2. Poor Path Quality
**Problem**: Robot takes inefficient or unsafe paths
**Solutions**:
- Tune global planner parameters
- Adjust costmap resolution
- Verify map quality and accuracy
- Check for proper obstacle detection

#### 3. Balance Issues During Navigation
**Problem**: Humanoid robot loses balance while following path
**Solutions**:
- Implement step-by-step planning
- Add balance controller
- Reduce navigation speed
- Improve path smoothing

## Best Practices

### 1. Parameter Tuning
- Start with conservative parameters
- Test in simulation before real robot
- Use systematic parameter tuning methods
- Document parameter sets for different environments

### 2. Safety Considerations
- Implement emergency stops
- Monitor robot state continuously
- Set appropriate velocity limits
- Use proper collision checking

### 3. Performance Optimization
- Use appropriate costmap resolution
- Optimize sensor update rates
- Implement efficient path planning
- Monitor computational resources

## Exercise

Create a complete navigation system for a humanoid robot that includes:

1. Custom Nav2 configuration optimized for humanoid locomotion
2. A step planner that generates discrete foot placements
3. A balance controller to maintain stability during navigation
4. Integration with Isaac Sim for testing
5. Performance evaluation tools to measure navigation success

Test your system in various scenarios including:
- Indoor navigation with furniture
- Narrow passages
- Obstacle avoidance
- Stair navigation (if applicable)
- Dynamic obstacle avoidance

Evaluate the system's performance using the metrics discussed in this section.