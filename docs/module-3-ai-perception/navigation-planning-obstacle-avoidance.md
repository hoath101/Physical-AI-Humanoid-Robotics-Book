# Navigation Planning and Obstacle Avoidance Examples

This section covers navigation planning and obstacle avoidance techniques for humanoid robots using AI-powered perception and navigation systems. We'll explore how to integrate perception data with navigation planning for safe and efficient robot movement.

## Introduction to Navigation Planning

Navigation planning involves determining a safe and efficient path for a robot to reach its goal while avoiding obstacles. For humanoid robots, this includes additional complexities like balance maintenance, step planning, and dynamic stability.

### Key Components of Navigation Planning
- **Global Path Planning**: Long-term path from start to goal
- **Local Path Planning**: Short-term path adjustment based on obstacles
- **Trajectory Generation**: Smooth motion trajectories
- **Obstacle Avoidance**: Real-time obstacle detection and avoidance
- **Recovery Behaviors**: Handling navigation failures

## Global Path Planning

### A* Algorithm Implementation

```cpp
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>

struct Point {
    int x, y;
    double g_cost = 0;  // Cost from start
    double h_cost = 0;  // Heuristic cost to goal
    double f_cost = 0;  // g + h
    Point* parent = nullptr;

    bool operator>(const Point& other) const {
        return f_cost > other.f_cost;
    }
};

class GlobalPlanner {
public:
    GlobalPlanner(const std::vector<std::vector<int>>& grid) : grid_(grid) {}

    std::vector<Point> planPath(const Point& start, const Point& goal) {
        // Initialize open and closed sets
        std::priority_queue<Point, std::vector<Point>, std::greater<Point>> open_set;
        std::vector<std::vector<bool>> closed_set(grid_.size(),
                                                 std::vector<bool>(grid_[0].size(), false));

        // Add start point to open set
        Point start_copy = start;
        start_copy.h_cost = heuristic(start_copy, goal);
        start_copy.f_cost = start_copy.h_cost;
        open_set.push(start_copy);

        while (!open_set.empty()) {
            Point current = open_set.top();
            open_set.pop();

            // Check if we reached the goal
            if (current.x == goal.x && current.y == goal.y) {
                return reconstructPath(current);
            }

            // Mark as visited
            closed_set[current.x][current.y] = true;

            // Check neighbors
            std::vector<Point> neighbors = getNeighbors(current);
            for (auto& neighbor : neighbors) {
                if (neighbor.x < 0 || neighbor.x >= grid_.size() ||
                    neighbor.y < 0 || neighbor.y >= grid_[0].size() ||
                    grid_[neighbor.x][neighbor.y] == 1 ||  // Obstacle
                    closed_set[neighbor.x][neighbor.y]) {
                    continue;
                }

                double tentative_g = current.g_cost + distance(current, neighbor);

                if (tentative_g < neighbor.g_cost || neighbor.parent == nullptr) {
                    neighbor.parent = new Point(current);
                    neighbor.g_cost = tentative_g;
                    neighbor.h_cost = heuristic(neighbor, goal);
                    neighbor.f_cost = neighbor.g_cost + neighbor.h_cost;

                    open_set.push(neighbor);
                }
            }
        }

        // No path found
        return {};
    }

private:
    std::vector<std::vector<int>> grid_;

    double heuristic(const Point& a, const Point& b) {
        // Manhattan distance heuristic
        return std::abs(a.x - b.x) + std::abs(a.y - b.y);
    }

    double distance(const Point& a, const Point& b) {
        // Euclidean distance
        return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2));
    }

    std::vector<Point> getNeighbors(const Point& point) {
        std::vector<Point> neighbors;
        // 8-directional movement
        int dx[] = {-1, -1, -1, 0, 0, 1, 1, 1};
        int dy[] = {-1, 0, 1, -1, 1, -1, 0, 1};

        for (int i = 0; i < 8; i++) {
            Point neighbor;
            neighbor.x = point.x + dx[i];
            neighbor.y = point.y + dy[i];
            neighbors.push_back(neighbor);
        }

        return neighbors;
    }

    std::vector<Point> reconstructPath(Point current) {
        std::vector<Point> path;
        while (current.parent != nullptr) {
            path.push_back(current);
            current = *(current.parent);
        }
        path.push_back(current);  // Add start point
        std::reverse(path.begin(), path.end());
        return path;
    }
};
```

### Nav2 Global Planner Integration

```cpp
#include <nav2_core/global_planner.hpp>
#include <nav2_costmap_2d/costmap_2d_ros.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <pluginlib/class_list_macros.hpp>

class HumanoidGlobalPlanner : public nav2_core::GlobalPlanner
{
public:
    HumanoidGlobalPlanner() = default;
    ~HumanoidGlobalPlanner() override = default;

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

        RCLCPP_INFO(node_->get_logger(), "Configured HumanoidGlobalPlanner");

        // Initialize humanoid-specific parameters
        step_height_threshold_ = node_->declare_parameter(name_ + ".step_height_threshold", 0.2);
        max_slope_angle_ = node_->declare_parameter(name_ + ".max_slope_angle", 15.0);
    }

    void cleanup() override
    {
        RCLCPP_INFO(node_->get_logger(), "Cleaning up HumanoidGlobalPlanner");
    }

    void activate() override
    {
        RCLCPP_INFO(node_->get_logger(), "Activating HumanoidGlobalPlanner");
    }

    void deactivate() override
    {
        RCLCPP_INFO(node_->get_logger(), "Deactivating HumanoidGlobalPlanner");
    }

    nav_msgs::msg::Path createPlan(
        const geometry_msgs::msg::PoseStamped & start,
        const geometry_msgs::msg::PoseStamped & goal) override
    {
        nav_msgs::msg::Path path;

        // Check if start and goal are valid
        if (!isStartValid(start) || !isGoalValid(goal)) {
            RCLCPP_WARN(node_->get_logger(), "Invalid start or goal position");
            return path;
        }

        // Convert poses to grid coordinates
        unsigned int start_x, start_y, goal_x, goal_y;
        if (!costmap_->worldToMap(start.pose.position.x, start.pose.position.y, start_x, start_y) ||
            !costmap_->worldToMap(goal.pose.position.x, goal.pose.position.y, goal_x, goal_y)) {
            RCLCPP_WARN(node_->get_logger(), "Start or goal position not in costmap");
            return path;
        }

        // Plan path using A* with humanoid constraints
        auto path_points = planHumanoidPath(start_x, start_y, goal_x, goal_y);

        // Convert to ROS message
        path = convertToPathMsg(path_points, start.header);

        // Apply path smoothing for humanoid locomotion
        smoothPath(path);

        return path;
    }

private:
    bool isStartValid(const geometry_msgs::msg::PoseStamped & start)
    {
        unsigned int mx, my;
        if (!costmap_->worldToMap(start.pose.position.x, start.pose.position.y, mx, my)) {
            return false;
        }

        // Check if start is in free space
        return costmap_->getCost(mx, my) < nav2_costmap_2d::FREE_SPACE;
    }

    bool isGoalValid(const geometry_msgs::msg::PoseStamped & goal)
    {
        unsigned int mx, my;
        if (!costmap_->worldToMap(goal.pose.position.x, goal.pose.position.y, mx, my)) {
            return false;
        }

        // Check if goal is in free space and not too close to obstacles
        unsigned char cost = costmap_->getCost(mx, my);
        return cost < nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE;
    }

    std::vector<Point> planHumanoidPath(unsigned int start_x, unsigned int start_y,
                                       unsigned int goal_x, unsigned int goal_y)
    {
        // Create grid representation of costmap
        std::vector<std::vector<int>> grid(costmap_->getSizeInCellsY(),
                                          std::vector<int>(costmap_->getSizeInCellsX()));

        // Populate grid with costmap data
        for (unsigned int y = 0; y < costmap_->getSizeInCellsY(); ++y) {
            for (unsigned int x = 0; x < costmap_->getSizeInCellsX(); ++x) {
                unsigned char cost = costmap_->getCost(x, y);
                grid[y][x] = (cost >= nav2_costmap_2d::LETHAL_OBSTACLE) ? 1 : 0;
            }
        }

        // Create start and goal points
        Point start_point, goal_point;
        start_point.x = start_x;
        start_point.y = start_y;
        goal_point.x = goal_x;
        goal_point.y = goal_y;

        // Run A* planning
        GlobalPlanner planner(grid);
        return planner.planPath(start_point, goal_point);
    }

    nav_msgs::msg::Path convertToPathMsg(const std::vector<Point>& points,
                                        const std_msgs::msg::Header& header)
    {
        nav_msgs::msg::Path path;
        path.header = header;

        for (const auto& point : points) {
            geometry_msgs::msg::PoseStamped pose;
            pose.header = header;

            // Convert grid coordinates to world coordinates
            double x, y;
            costmap_->mapToWorld(point.x, point.y, x, y);
            pose.pose.position.x = x;
            pose.pose.position.y = y;
            pose.pose.position.z = 0.0;

            // Set orientation to face next point
            if (&point != &points.back()) {  // Not the last point
                auto next_it = std::next(&point);
                if (next_it != &points.back() + 1) {
                    double dx = points[&point - &points[0] + 1].x - point.x;
                    double dy = points[&point - &points[0] + 1].y - point.y;

                    double yaw = atan2(dy, dx);
                    tf2::Quaternion q;
                    q.setRPY(0, 0, yaw);
                    pose.pose.orientation.x = q.x();
                    pose.pose.orientation.y = q.y();
                    pose.pose.orientation.z = q.z();
                    pose.pose.orientation.w = q.w();
                }
            }

            path.poses.push_back(pose);
        }

        return path;
    }

    void smoothPath(nav_msgs::msg::Path& path)
    {
        // Apply path smoothing for smoother humanoid locomotion
        // This could implement techniques like:
        // - Dubins curves for curvature-constrained paths
        // - B-spline smoothing
        // - Gradient descent-based smoothing

        // Simple smoothing by averaging adjacent points
        if (path.poses.size() < 3) return;

        for (size_t i = 1; i < path.poses.size() - 1; ++i) {
            auto& curr = path.poses[i].pose.position;
            auto prev = path.poses[i-1].pose.position;
            auto next = path.poses[i+1].pose.position;

            // Weighted average: 25% prev, 50% current, 25% next
            curr.x = 0.25 * prev.x + 0.5 * curr.x + 0.25 * next.x;
            curr.y = 0.25 * prev.y + 0.5 * curr.y + 0.25 * next.y;
        }
    }

    rclcpp_lifecycle::LifecycleNode::SharedPtr node_;
    std::string name_;
    std::shared_ptr<tf2_ros::Buffer> tf_;
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
    nav2_costmap_2d::Costmap2D* costmap_;

    // Humanoid-specific parameters
    double step_height_threshold_;
    double max_slope_angle_;
};

PLUGINLIB_EXPORT_CLASS(HumanoidGlobalPlanner, nav2_core::GlobalPlanner)
```

## Local Path Planning and Obstacle Avoidance

### Dynamic Window Approach (DWA) for Humanoid Robots

```cpp
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

class HumanoidLocalPlanner {
public:
    HumanoidLocalPlanner() {
        // Initialize humanoid-specific parameters
        max_vel_x_ = 0.5;      // Max forward velocity (m/s)
        min_vel_x_ = 0.05;     // Min forward velocity (m/s)
        max_vel_th_ = 0.5;     // Max angular velocity (rad/s)
        min_vel_th_ = -0.5;    // Min angular velocity (rad/s)

        max_acc_x_ = 0.5;      // Max acceleration (m/s²)
        max_acc_th_ = 1.0;     // Max angular acceleration (rad/s²)

        vtheta_samp_ = 20;     // Number of angular velocity samples
        vx_samp_ = 10;         // Number of forward velocity samples

        sim_time_ = 2.0;       // Simulation time horizon (seconds)
        sim_granularity_ = 0.05; // Simulation granularity (meters)

        // Humanoid-specific parameters
        step_frequency_ = 1.25;  // Steps per second
        balance_margin_ = 0.1;   // Balance safety margin
    }

    geometry_msgs::msg::Twist calculateVelocityCommands(
        const geometry_msgs::msg::PoseStamped& robot_pose,
        const geometry_msgs::msg::PoseStamped& goal_pose,
        const geometry_msgs::msg::Twist& current_vel,
        const sensor_msgs::msg::LaserScan& scan_data)
    {
        geometry_msgs::msg::Twist cmd_vel;

        // Get possible velocities
        auto vel_samples = getVelocitySamples(current_vel);

        double best_score = -std::numeric_limits<double>::infinity();
        geometry_msgs::msg::Twist best_vel;

        for (const auto& vel : vel_samples) {
            // Simulate trajectory for this velocity
            auto trajectory = simulateTrajectory(robot_pose.pose, vel, current_vel);

            // Evaluate trajectory
            double heading_score = calculateHeadingScore(trajectory, goal_pose);
            double dist_score = calculateDistScore(trajectory);
            double obs_score = calculateObstacleScore(trajectory, scan_data);

            // Weighted combination of scores
            double score = 0.3 * heading_score + 0.2 * dist_score + 0.5 * obs_score;

            if (score > best_score) {
                best_score = score;
                best_vel = vel;
            }
        }

        return best_vel;
    }

private:
    struct Trajectory {
        std::vector<geometry_msgs::msg::Pose> poses;
        geometry_msgs::msg::Twist final_vel;
    };

    std::vector<geometry_msgs::msg::Twist> getVelocitySamples(
        const geometry_msgs::msg::Twist& current_vel)
    {
        std::vector<geometry_msgs::msg::Twist> samples;

        // Calculate velocity windows based on current velocity and accelerations
        double dt = 0.1;  // Time step for sampling
        double max_delta_vx = max_acc_x_ * dt;
        double max_delta_vth = max_acc_th_ * dt;

        double min_vx = std::max(min_vel_x_, current_vel.linear.x - max_delta_vx);
        double max_vx = std::min(max_vel_x_, current_vel.linear.x + max_delta_vx);
        double min_vth = std::max(min_vel_th_, current_vel.angular.z - max_delta_vth);
        double max_vth = std::min(max_vel_th_, current_vel.angular.z + max_delta_vth);

        // Sample velocities
        double dvx = (max_vx - min_vx) / vx_samp_;
        double dvth = (max_vth - min_vth) / vtheta_samp_;

        for (int i = 0; i <= vx_samp_; ++i) {
            for (int j = 0; j <= vtheta_samp_; ++j) {
                geometry_msgs::msg::Twist vel;
                vel.linear.x = min_vx + i * dvx;
                vel.angular.z = min_vth + j * dvth;

                // Humanoid-specific constraints
                if (isValidHumanoidVelocity(vel)) {
                    samples.push_back(vel);
                }
            }
        }

        return samples;
    }

    bool isValidHumanoidVelocity(const geometry_msgs::msg::Twist& vel)
    {
        // Check humanoid-specific constraints
        // For example, ensure velocity is within safe walking parameters
        double speed = sqrt(vel.linear.x * vel.linear.x + vel.linear.y * vel.linear.y);

        // Simple check: speed should be within walking range
        if (speed > max_vel_x_ * 1.5) return false;  // Too fast for stable walking

        return true;
    }

    Trajectory simulateTrajectory(
        const geometry_msgs::msg::Pose& start_pose,
        const geometry_msgs::msg::Twist& target_vel,
        const geometry_msgs::msg::Twist& current_vel)
    {
        Trajectory traj;
        geometry_msgs::msg::Pose current_pose = start_pose;
        geometry_msgs::msg::Twist current_vel_local = current_vel;

        double dt = sim_granularity_ / std::max(std::abs(target_vel.linear.x), 0.1);
        int steps = static_cast<int>(sim_time_ / dt);

        for (int i = 0; i < steps; ++i) {
            // Update pose based on current velocity
            double yaw = tf2::getYaw(current_pose.orientation);

            // Update position
            current_pose.position.x += current_vel_local.linear.x * cos(yaw) * dt;
            current_pose.position.y += current_vel_local.linear.x * sin(yaw) * dt;
            current_pose.position.z += current_vel_local.linear.z * dt;  // For 3D movement

            // Update orientation
            yaw += current_vel_local.angular.z * dt;
            tf2::Quaternion quat;
            quat.setRPY(0, 0, yaw);
            current_pose.orientation = tf2::toMsg(quat);

            // Update velocity towards target (with acceleration constraints)
            double ax = std::min(max_acc_x_,
                                std::abs(target_vel.linear.x - current_vel_local.linear.x) / dt);
            double ath = std::min(max_acc_th_,
                                 std::abs(target_vel.angular.z - current_vel_local.angular.z) / dt);

            if (target_vel.linear.x > current_vel_local.linear.x) {
                current_vel_local.linear.x = std::min(target_vel.linear.x,
                                                    current_vel_local.linear.x + ax * dt);
            } else {
                current_vel_local.linear.x = std::max(target_vel.linear.x,
                                                    current_vel_local.linear.x - ax * dt);
            }

            if (target_vel.angular.z > current_vel_local.angular.z) {
                current_vel_local.angular.z = std::min(target_vel.angular.z,
                                                     current_vel_local.angular.z + ath * dt);
            } else {
                current_vel_local.angular.z = std::max(target_vel.angular.z,
                                                     current_vel_local.angular.z - ath * dt);
            }

            traj.poses.push_back(current_pose);
        }

        traj.final_vel = current_vel_local;
        return traj;
    }

    double calculateHeadingScore(
        const Trajectory& traj,
        const geometry_msgs::msg::PoseStamped& goal_pose)
    {
        if (traj.poses.empty()) return 0.0;

        const auto& final_pose = traj.poses.back();

        // Calculate angle to goal
        double goal_x = goal_pose.pose.position.x;
        double goal_y = goal_pose.pose.position.y;
        double robot_x = final_pose.position.x;
        double robot_y = final_pose.position.y;

        double angle_to_goal = atan2(goal_y - robot_y, goal_x - robot_x);
        double robot_yaw = tf2::getYaw(final_pose.orientation);

        // Normalize angles
        double angle_diff = angle_to_goal - robot_yaw;
        angle_diff = std::atan2(std::sin(angle_diff), std::cos(angle_diff));

        // Score based on how well the robot is oriented toward the goal
        return 1.0 - std::abs(angle_diff) / M_PI;  // Higher score for smaller angle difference
    }

    double calculateDistScore(const Trajectory& traj)
    {
        // Score based on how far the trajectory takes the robot
        if (traj.poses.size() < 2) return 0.0;

        const auto& start = traj.poses.front();
        const auto& end = traj.poses.back();

        double dist = sqrt(pow(end.position.x - start.position.x, 2) +
                          pow(end.position.y - start.position.y, 2));

        // Normalize by simulation time
        return dist / sim_time_;
    }

    double calculateObstacleScore(
        const Trajectory& traj,
        const sensor_msgs::msg::LaserScan& scan_data)
    {
        double score = 0.0;
        double min_dist_to_obstacle = std::numeric_limits<double>::infinity();

        for (const auto& pose : traj.poses) {
            // Check distance to nearest obstacle at this pose
            double dist = getMinDistanceToObstacle(pose, scan_data);
            min_dist_to_obstacle = std::min(min_dist_to_obstacle, dist);

            if (dist < 0.2) {  // Very close to obstacle
                return -std::numeric_limits<double>::infinity();  // Invalid trajectory
            }
        }

        // Score based on minimum distance to obstacles
        // Prefer trajectories that stay farther from obstacles
        return min_dist_to_obstacle;
    }

    double getMinDistanceToObstacle(
        const geometry_msgs::msg::Pose& pose,
        const sensor_msgs::msg::LaserScan& scan_data)
    {
        // Convert laser scan points to robot coordinates and check for obstacles
        double min_dist = std::numeric_limits<double>::infinity();
        double robot_yaw = tf2::getYaw(pose.orientation);

        for (size_t i = 0; i < scan_data.ranges.size(); ++i) {
            if (scan_data.ranges[i] < scan_data.range_min ||
                scan_data.ranges[i] > scan_data.range_max) {
                continue;  // Invalid range
            }

            // Convert polar coordinates to Cartesian in laser frame
            double angle = scan_data.angle_min + i * scan_data.angle_increment;
            double x_laser = scan_data.ranges[i] * cos(angle);
            double y_laser = scan_data.ranges[i] * sin(angle);

            // Transform to robot base frame
            double cos_yaw = cos(robot_yaw);
            double sin_yaw = sin(robot_yaw);

            double x_robot = x_laser * cos_yaw - y_laser * sin_yaw + pose.position.x;
            double y_robot = x_laser * sin_yaw + y_laser * cos_yaw + pose.position.y;

            // Calculate distance from robot center to this point
            double dist = sqrt(pow(x_robot - pose.position.x, 2) +
                              pow(y_robot - pose.position.y, 2));

            min_dist = std::min(min_dist, dist);
        }

        return min_dist;
    }

    // Parameters
    double max_vel_x_, min_vel_x_, max_vel_th_, min_vel_th_;
    double max_acc_x_, max_acc_th_;
    int vtheta_samp_, vx_samp_;
    double sim_time_, sim_granularity_;
    double step_frequency_, balance_margin_;
};
```

## Obstacle Detection and Avoidance Integration

### Perception-Based Obstacle Detection

```cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <isaac_ros_detectnet_interfaces/msg/detection_array.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

class PerceptionObstacleDetector : public rclcpp::Node
{
public:
    PerceptionObstacleDetector() : Node("perception_obstacle_detector")
    {
        // Subscribe to various sensor inputs
        laser_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "scan", 10,
            std::bind(&PerceptionObstacleDetector::laserCallback, this, std::placeholders::_1)
        );

        detection_sub_ = this->create_subscription<isaac_ros_detectnet_interfaces::msg::DetectionArray>(
            "detections", 10,
            std::bind(&PerceptionObstacleDetector::detectionCallback, this, std::placeholders::_1)
        );

        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "points", 10,
            std::bind(&PerceptionObstacleDetector::pointcloudCallback, this, std::placeholders::_1)
        );

        // Publisher for combined obstacle information
        obstacle_pub_ = this->create_publisher<geometry_msgs::msg::PolygonStamped>(
            "obstacle_polygon", 10
        );
    }

private:
    void laserCallback(const sensor_msgs::msg::LaserScan::SharedPtr scan_msg)
    {
        // Process LIDAR data to detect obstacles
        std::vector<geometry_msgs::msg::Point32> laser_obstacles = processLaserScan(*scan_msg);

        // Update obstacle map
        updateObstacleMap(laser_obstacles, "laser");
    }

    void detectionCallback(const isaac_ros_detectnet_interfaces::msg::DetectionArray::SharedPtr detection_msg)
    {
        // Process 2D detections and convert to 3D obstacle positions
        std::vector<geometry_msgs::msg::Point32> vision_obstacles = processDetections(*detection_msg);

        // Update obstacle map
        updateObstacleMap(vision_obstacles, "vision");
    }

    void pointcloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg)
    {
        // Process point cloud to detect obstacles
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*cloud_msg, *cloud);

        std::vector<geometry_msgs::msg::Point32> pointcloud_obstacles = processPointCloud(cloud);

        // Update obstacle map
        updateObstacleMap(pointcloud_obstacles, "pointcloud");
    }

    std::vector<geometry_msgs::msg::Point32> processLaserScan(const sensor_msgs::msg::LaserScan& scan)
    {
        std::vector<geometry_msgs::msg::Point32> obstacles;

        for (size_t i = 0; i < scan.ranges.size(); ++i) {
            if (scan.ranges[i] < scan.range_min || scan.ranges[i] > scan.range_max) {
                continue;  // Invalid range
            }

            if (scan.ranges[i] < obstacle_distance_threshold_) {
                // Convert polar to Cartesian coordinates
                double angle = scan.angle_min + i * scan.angle_increment;
                geometry_msgs::msg::Point32 point;
                point.x = scan.ranges[i] * cos(angle);
                point.y = scan.ranges[i] * sin(angle);
                point.z = 0.0;  // Assume ground level

                obstacles.push_back(point);
            }
        }

        return obstacles;
    }

    std::vector<geometry_msgs::msg::Point32> processDetections(
        const isaac_ros_detectnet_interfaces::msg::DetectionArray& detections)
    {
        std::vector<geometry_msgs::msg::Point32> obstacles;

        for (const auto& detection : detections.detections) {
            if (detection.confidence > detection_confidence_threshold_) {
                // Convert 2D bounding box to 3D position estimate
                // This requires depth information from stereo or depth sensor
                geometry_msgs::msg::Point32 obstacle_pos;

                // For now, assume a fixed depth based on object type and size
                double estimated_depth = estimateDepthFromSize(detection.bbox, detection.label);

                // Convert image coordinates to robot coordinates
                // This requires camera calibration parameters
                obstacle_pos.x = (detection.bbox.center.x - camera_cx_) * estimated_depth / camera_fx_;
                obstacle_pos.y = (detection.bbox.center.y - camera_cy_) * estimated_depth / camera_fy_;
                obstacle_pos.z = estimated_depth;

                obstacles.push_back(obstacle_pos);
            }
        }

        return obstacles;
    }

    std::vector<geometry_msgs::msg::Point32> processPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
    {
        std::vector<geometry_msgs::msg::Point32> obstacles;

        // Use PCL to segment obstacles from ground plane
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

        // Create the segmentation object
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setMaxIterations(100);
        seg.setDistanceThreshold(0.05);  // 5cm tolerance for ground plane

        seg.setInputCloud(cloud);
        seg.segment(*inliers, *coefficients);

        // Extract obstacles (points not belonging to ground plane)
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud);
        extract.setIndices(inliers);
        extract.setNegative(true);  // Extract points NOT on the plane
        extract.filter(obstacles);

        // Convert PCL points to ROS points
        std::vector<geometry_msgs::msg::Point32> obstacle_points;
        for (const auto& point : obstacles.points) {
            geometry_msgs::msg::Point32 ros_point;
            ros_point.x = point.x;
            ros_point.y = point.y;
            ros_point.z = point.z;
            obstacle_points.push_back(ros_point);
        }

        return obstacle_points;
    }

    void updateObstacleMap(
        const std::vector<geometry_msgs::msg::Point32>& new_obstacles,
        const std::string& sensor_type)
    {
        // Fuse obstacles from different sensors
        for (const auto& obstacle : new_obstacles) {
            // Add to combined obstacle map with sensor type information
            auto it = std::find_if(combined_obstacles_.begin(), combined_obstacles_.end(),
                [&obstacle](const FusedObstacle& existing) {
                    double dist = sqrt(pow(existing.point.x - obstacle.x, 2) +
                                      pow(existing.point.y - obstacle.y, 2));
                    return dist < fusion_distance_threshold_;
                });

            if (it != combined_obstacles_.end()) {
                // Update existing obstacle with new information
                it->update(obstacle, sensor_type);
            } else {
                // Add new obstacle
                FusedObstacle new_fused_obstacle(obstacle, sensor_type);
                combined_obstacles_.push_back(new_fused_obstacle);
            }
        }

        // Publish combined obstacle information
        publishCombinedObstacles();
    }

    void publishCombinedObstacles()
    {
        geometry_msgs::msg::PolygonStamped obstacle_polygon;
        obstacle_polygon.header.frame_id = "map";
        obstacle_polygon.header.stamp = this->now();

        for (const auto& obstacle : combined_obstacles_) {
            geometry_msgs::msg::Point32 point;
            point.x = obstacle.point.x;
            point.y = obstacle.point.y;
            point.z = obstacle.point.z;
            obstacle_polygon.polygon.points.push_back(point);
        }

        obstacle_pub_->publish(obstacle_polygon);
    }

    double estimateDepthFromSize(
        const isaac_ros_detectnet_interfaces::msg::BoundingBox& bbox,
        const std::string& label)
    {
        // Estimate depth based on expected object size
        // This is a simplified approach - in practice, you'd use stereo or depth sensor
        double expected_width = getExpectedWidth(label);
        double pixel_width = bbox.size_x;

        // Using thin lens equation: depth = (focal_length * real_width) / pixel_width
        return (camera_fx_ * expected_width) / pixel_width;
    }

    double getExpectedWidth(const std::string& label)
    {
        // Return expected width for common object types (in meters)
        if (label == "person") return 0.5;      // Average person width
        if (label == "car") return 1.8;         // Average car width
        if (label == "chair") return 0.6;       // Average chair width
        if (label == "table") return 1.0;       // Average table width
        return 0.5;  // Default assumption
    }

    struct FusedObstacle {
        geometry_msgs::msg::Point32 point;
        std::map<std::string, int> sensor_votes;  // Count of detections from each sensor
        double confidence;                        // Overall confidence

        FusedObstacle(const geometry_msgs::msg::Point32& p, const std::string& sensor_type) :
            point(p), confidence(0.5) {
            sensor_votes[sensor_type] = 1;
        }

        void update(const geometry_msgs::msg::Point32& new_point, const std::string& sensor_type) {
            // Update position with weighted average
            point.x = (point.x + new_point.x) / 2.0;
            point.y = (point.y + new_point.y) / 2.0;
            point.z = (point.z + new_point.z) / 2.0;

            // Update sensor votes
            sensor_votes[sensor_type]++;

            // Update confidence based on number of confirming sensors
            confidence = std::min(1.0, static_cast<double>(sensor_votes.size()) / 3.0);
        }
    };

    // Subscriptions
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr laser_sub_;
    rclcpp::Subscription<isaac_ros_detectnet_interfaces::msg::DetectionArray>::SharedPtr detection_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;

    // Publisher
    rclcpp::Publisher<geometry_msgs::msg::PolygonStamped>::SharedPtr obstacle_pub_;

    // Obstacle data
    std::vector<FusedObstacle> combined_obstacles_;

    // Parameters
    const double obstacle_distance_threshold_ = 2.0;  // Max distance to consider obstacle
    const double detection_confidence_threshold_ = 0.7;  // Min confidence for detections
    const double fusion_distance_threshold_ = 0.3;    // Distance to fuse detections
    const double camera_fx_ = 616.363;  // Camera focal length x
    const double camera_fy_ = 616.363;  // Camera focal length y
    const double camera_cx_ = 313.071;  // Camera principal point x
    const double camera_cy_ = 245.091;  // Camera principal point y
};
```

## Humanoid-Specific Navigation Behaviors

### Step Planning for Bipedal Locomotion

```cpp
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.h>
#include <visualization_msgs/msg/marker_array.h>

class HumanoidStepPlanner {
public:
    HumanoidStepPlanner() {
        // Initialize humanoid-specific parameters
        step_length_ = 0.3;      // Average step length in meters
        step_width_ = 0.2;       // Side-to-side step distance
        step_height_ = 0.05;     // Step clearance height
        max_step_rotation_ = 0.2; // Max rotation per step (rad)
        step_duration_ = 0.8;    // Time for one step (sec)
    }

    struct Step {
        geometry_msgs::msg::Point left_foot;
        geometry_msgs::msg::Point right_foot;
        double time;
        bool is_support_step;  // Which foot is supporting weight
    };

    std::vector<Step> planSteps(const nav_msgs::msg::Path& path,
                               const geometry_msgs::msg::Pose& start_pose)
    {
        std::vector<Step> steps;

        if (path.poses.empty()) return steps;

        // Start with current stance
        Step initial_step;
        initial_step.left_foot = calculateInitialFootPosition(start_pose, true);
        initial_step.right_foot = calculateInitialFootPosition(start_pose, false);
        initial_step.time = 0.0;
        initial_step.is_support_step = true;  // Right foot starts as support
        steps.push_back(initial_step);

        // Plan steps along the path
        size_t current_path_idx = 0;
        geometry_msgs::msg::Pose current_pose = start_pose;
        double current_time = 0.0;

        while (current_path_idx < path.poses.size()) {
            // Calculate next step based on path direction
            auto next_waypoint = getNextWaypoint(path, current_path_idx, current_pose);
            auto next_step = calculateNextStep(current_pose, next_waypoint, steps.back());

            if (isValidStep(next_step, path)) {
                next_step.time = current_time + step_duration_;
                steps.push_back(next_step);

                // Update current pose based on step
                updatePoseFromStep(current_pose, next_step);
                current_time += step_duration_;

                // Move to next significant waypoint
                current_path_idx = findNextSignificantWaypoint(path, current_path_idx);
            } else {
                // Handle invalid step (obstacle, unstable, etc.)
                auto recovery_step = planRecoveryStep(steps.back());
                if (isValidStep(recovery_step, path)) {
                    recovery_step.time = current_time + step_duration_;
                    steps.push_back(recovery_step);
                    updatePoseFromStep(current_pose, recovery_step);
                    current_time += step_duration_;
                } else {
                    // Cannot proceed, return current plan
                    break;
                }
            }
        }

        return steps;
    }

private:
    geometry_msgs::msg::Point calculateInitialFootPosition(
        const geometry_msgs::msg::Pose& robot_pose, bool is_left_foot)
    {
        geometry_msgs::msg::Point foot_pos;
        double yaw = tf2::getYaw(robot_pose.orientation);

        // Place feet shoulder-width apart initially
        double offset_x = 0.0;
        double offset_y = is_left_foot ? step_width_/2.0 : -step_width_/2.0;

        // Transform offset to robot frame
        foot_pos.x = robot_pose.position.x + offset_x * cos(yaw) - offset_y * sin(yaw);
        foot_pos.y = robot_pose.position.y + offset_x * sin(yaw) + offset_y * cos(yaw);
        foot_pos.z = robot_pose.position.z;  // Ground level

        return foot_pos;
    }

    geometry_msgs::msg::PoseStamped getNextWaypoint(
        const nav_msgs::msg::Path& path, size_t current_idx,
        const geometry_msgs::msg::Pose& current_pose)
    {
        // Find the next waypoint that's ahead of the robot
        for (size_t i = current_idx; i < path.poses.size(); ++i) {
            double dist_sq = pow(path.poses[i].pose.position.x - current_pose.position.x, 2) +
                            pow(path.poses[i].pose.position.y - current_pose.position.y, 2);

            if (dist_sq > pow(step_length_ * 0.8, 2)) {  // Look ahead 80% of step length
                return path.poses[i];
            }
        }

        // If no significant waypoint found, return the last one
        if (!path.poses.empty()) {
            return path.poses.back();
        }

        // Return current pose if no path
        geometry_msgs::msg::PoseStamped dummy;
        dummy.pose = current_pose;
        return dummy;
    }

    Step calculateNextStep(const geometry_msgs::msg::Pose& current_pose,
                          const geometry_msgs::msg::PoseStamped& target_waypoint,
                          const Step& previous_step)
    {
        Step next_step;

        // Calculate direction to target
        double dx = target_waypoint.pose.position.x - current_pose.position.x;
        double dy = target_waypoint.pose.position.y - current_pose.position.y;
        double target_yaw = atan2(dy, dx);
        double current_yaw = tf2::getYaw(current_pose.orientation);

        // Determine which foot to move (opposite of support foot)
        bool move_left_foot = previous_step.is_support_step;  // If right was support, move left

        // Calculate new foot position
        double step_yaw = current_yaw + (move_left_foot ? max_step_rotation_ : -max_step_rotation_);

        geometry_msgs::msg::Point new_foot_pos;
        if (move_left_foot) {
            // Move left foot toward target
            new_foot_pos.x = current_pose.position.x + step_length_ * cos(step_yaw);
            new_foot_pos.y = current_pose.position.y + step_length_ * sin(step_yaw);
            new_foot_pos.z = current_pose.position.z;

            // Keep right foot in place
            next_step.right_foot = previous_step.right_foot;
            next_step.left_foot = new_foot_pos;
        } else {
            // Move right foot toward target
            new_foot_pos.x = current_pose.position.x + step_length_ * cos(step_yaw);
            new_foot_pos.y = current_pose.position.y + step_length_ * sin(step_yaw);
            new_foot_pos.z = current_pose.position.z;

            // Keep left foot in place
            next_step.left_foot = previous_step.left_foot;
            next_step.right_foot = new_foot_pos;
        }

        // Update support foot (alternates with each step)
        next_step.is_support_step = !previous_step.is_support_step;

        return next_step;
    }

    bool isValidStep(const Step& step, const nav_msgs::msg::Path& path)
    {
        // Check if step is stable (center of mass within support polygon)
        geometry_msgs::msg::Point com = calculateCOMPosition(step);

        if (!isWithinSupportPolygon(com, step)) {
            return false;
        }

        // Check for obstacles at step location
        if (isStepLocationBlocked(step)) {
            return false;
        }

        // Check if step deviates too much from planned path
        if (isStepOffPath(step, path)) {
            return false;
        }

        return true;
    }

    bool isWithinSupportPolygon(const geometry_msgs::msg::Point& com, const Step& step)
    {
        // For bipedal locomotion, support polygon is the convex hull of both feet
        // This is a simplified check - in practice, you'd calculate the actual convex hull

        // Check if COM is roughly between the feet
        double min_x = std::min(step.left_foot.x, step.right_foot.x);
        double max_x = std::max(step.left_foot.x, step.right_foot.x);
        double min_y = std::min(step.left_foot.y, step.right_foot.y);
        double max_y = std::max(step.left_foot.y, step.right_foot.y);

        // Add a safety margin
        double margin = balance_margin_;

        return (com.x >= min_x - margin && com.x <= max_x + margin &&
                com.y >= min_y - margin && com.y <= max_y + margin);
    }

    bool isStepLocationBlocked(const Step& step)
    {
        // Check if the step location has obstacles
        // This would interface with the costmap or obstacle detection system
        // For now, return false as a placeholder
        return false;
    }

    bool isStepOffPath(const Step& step, const nav_msgs::msg::Path& path)
    {
        // Check if the step deviates too much from the global path
        // This would require path tracking algorithms
        // For now, return false as a placeholder
        return false;
    }

    geometry_msgs::msg::Point calculateCOMPosition(const Step& step)
    {
        // Simplified COM calculation - in reality, this would consider
        // the full robot kinematics and mass distribution
        geometry_msgs::msg::Point com;
        com.x = (step.left_foot.x + step.right_foot.x) / 2.0;
        com.y = (step.left_foot.y + step.right_foot.y) / 2.0;
        com.z = com_height_;  // Approximate COM height

        return com;
    }

    Step planRecoveryStep(const Step& previous_step)
    {
        // Plan a recovery step when normal stepping is not possible
        // This might involve: stepping in place, taking a smaller step, etc.

        Step recovery_step = previous_step;

        // For now, just return the previous step as a placeholder
        // In practice, this would implement various recovery behaviors
        return recovery_step;
    }

    size_t findNextSignificantWaypoint(const nav_msgs::msg::Path& path, size_t current_idx)
    {
        // Find the next waypoint that represents a significant change in direction
        // This prevents excessive step planning for dense paths
        size_t next_idx = current_idx + 1;

        // Simple approach: skip waypoints that are very close together
        while (next_idx < path.poses.size()) {
            double dist_sq = pow(path.poses[next_idx].pose.position.x -
                                path.poses[current_idx].pose.position.x, 2) +
                            pow(path.poses[next_idx].pose.position.y -
                               path.poses[current_idx].pose.position.y, 2);

            if (dist_sq > pow(step_length_ * 0.5, 2)) {  // Minimum distance between processed waypoints
                return next_idx;
            }
            next_idx++;
        }

        return path.poses.size();  // Return end if no significant waypoint found
    }

    void updatePoseFromStep(geometry_msgs::msg::Pose& pose, const Step& step)
    {
        // Update robot pose based on completed step
        // This would consider the kinematics of the step
        pose.position.x = (step.left_foot.x + step.right_foot.x) / 2.0;
        pose.position.y = (step.left_foot.y + step.right_foot.y) / 2.0;

        // Update orientation based on foot positions
        double dx = step.right_foot.x - step.left_foot.x;
        double dy = step.right_foot.y - step.left_foot.y;
        double yaw = atan2(dy, dx) + M_PI/2;  // Rotate 90 degrees for forward direction

        tf2::Quaternion q;
        q.setRPY(0, 0, yaw);
        pose.orientation = tf2::toMsg(q);
    }

    // Humanoid-specific parameters
    double step_length_;
    double step_width_;
    double step_height_;
    double max_step_rotation_;
    double step_duration_;
    double com_height_ = 0.8;  // Approximate height of center of mass
    double balance_margin_ = 0.1;  // Safety margin for balance
};
```

## Isaac Sim Navigation Integration

### Isaac Sim Navigation Controller

```python
# Isaac Sim navigation integration
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.range_sensor import LidarRtx
import numpy as np
import carb

class IsaacSimNavigationController:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.setup_navigation_environment()

    def setup_navigation_environment(self):
        # Add robot to the scene
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets. Please check your Isaac Sim installation.")
            return

        # Add a humanoid robot (example with Carter robot)
        robot_asset_path = assets_root_path + "/Isaac/Robots/Carter/carter_navigate.usd"
        add_reference_to_stage(usd_path=robot_asset_path, prim_path="/World/Carter")

        # Add a LIDAR sensor to the robot
        self.lidar = LidarRtx(
            prim_path="/World/Carter/chassis/lidar",
            translation=np.array([0.0, 0.0, 0.3]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0]),
            config="Carter",
            rotation_frequency=10,
            samples_per_scan=1080
        )

        # Add a simple environment
        room_asset_path = assets_root_path + "/Isaac/Environments/Simple_Room/simple_room.usd"
        add_reference_to_stage(usd_path=room_asset_path, prim_path="/World/Room")

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
            lidar_data = self.lidar.get_linear_depth_data()

            # Check if reached goal
            if self.is_at_goal(robot_position, goal_position):
                print(f"Reached goal at {goal_position}!")
                break

            # Plan and execute navigation
            cmd_vel = self.plan_navigation_command(
                robot_position, robot_orientation,
                goal_position, lidar_data
            )

            # Apply command to robot
            self.execute_command(cmd_vel)

    def get_robot_position(self):
        """Get current robot position from Isaac Sim"""
        # In a real implementation, this would get the robot's position
        # from the simulation
        pass

    def get_robot_orientation(self):
        """Get current robot orientation from Isaac Sim"""
        pass

    def is_at_goal(self, current_pos, goal_pos, tolerance=0.2):
        """Check if robot is at goal position"""
        distance = np.linalg.norm(np.array(current_pos[:2]) - np.array(goal_pos[:2]))
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
        return np.arctan2(2*(orientation[3]*orientation[2] + orientation[0]*orientation[1]),
                         1 - 2*(orientation[1]**2 + orientation[2]**2))

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
        linear_vel, angular_vel = cmd_vel
        # Apply these velocities to the robot's differential drive controller
        pass
```

## Recovery Behaviors

### Navigation Recovery Behaviors

```cpp
#include <rclcpp/rclcpp.hpp>
#include <nav2_core/recovery.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <pluginlib/class_list_macros.hpp>

class HumanoidSpinRecovery : public nav2_core::Recovery
{
public:
    HumanoidSpinRecovery() = default;
    ~HumanoidSpinRecovery() override = default;

    void configure(
        const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
        const std::string & name,
        const std::shared_ptr<tf2_ros::Buffer> & tf,
        const std::shared_ptr<nav2_costmap_2d::Costmap2DROS> & global_costmap,
        const std::shared_ptr<nav2_costmap_2d::Costmap2DROS> & local_costmap) override
    {
        node_ = parent.lock();
        name_ = name;
        tf_ = tf;
        global_costmap_ = global_costmap;
        local_costmap_ = local_costmap;

        // Declare parameters specific to humanoid recovery
        spin_angular_vel_ = node_->declare_parameter(name_ + ".spin_angular_vel", 0.5);
        min_spin_duration_ = node_->declare_parameter(name_ + ".min_spin_duration", 1.0);
        max_spin_duration_ = node_->declare_parameter(name_ + ".max_spin_duration", 10.0);

        vel_pub_ = node_->create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 1);
    }

    void cleanup() override
    {
        vel_pub_->on_deactivate();
    }

    void activate() override
    {
        vel_pub_->on_activate();
    }

    void deactivate() override
    {
        vel_pub_->on_deactivate();
    }

    nav2_core::RecoveryResult run(
        const std::shared_ptr<const nav2_msgs::action::Recovery::Goal> command) override
    {
        RCLCPP_INFO(node_->get_logger(), "Starting humanoid spin recovery behavior");

        // For humanoid robots, spinning in place may not be feasible
        // Instead, implement a gentle turning motion with steps
        return executeHumanoidSpin();
    }

private:
    nav2_core::RecoveryResult executeHumanoidSpin()
    {
        nav2_core::RecoveryResult result;
        result.outcome = nav2_core::RecoveryResult::SUCCESS;

        auto start_time = node_->now();
        auto current_time = start_time;

        while (rclcpp::ok()) {
            current_time = node_->now();

            // Check if we've spun enough
            if ((current_time - start_time).seconds() > min_spin_duration_) {
                // Check if we've cleared the obstacle
                if (isObstacleClear()) {
                    RCLCPP_INFO(node_->get_logger(), "Obstacle cleared, stopping spin recovery");
                    break;
                }

                // Check if we've spun too long
                if ((current_time - start_time).seconds() > max_spin_duration_) {
                    RCLCPP_WARN(node_->get_logger(), "Spin recovery timed out");
                    result.outcome = nav2_core::RecoveryResult::FAILURE;
                    break;
                }
            }

            // Generate spin command for humanoid
            auto spin_cmd = generateHumanoidSpinCommand();
            vel_pub_->publish(spin_cmd);

            // Sleep briefly to allow other processes
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // Stop the robot
        geometry_msgs::msg::Twist stop_cmd;
        vel_pub_->publish(stop_cmd);

        return result;
    }

    geometry_msgs::msg::Twist generateHumanoidSpinCommand()
    {
        geometry_msgs::msg::Twist cmd;

        // For humanoid, instead of pure rotation, we might want to step-turn
        // This is a simplified approach - real implementation would plan actual steps
        cmd.angular.z = spin_angular_vel_;

        // Small forward motion to maintain momentum
        cmd.linear.x = 0.05;

        return cmd;
    }

    bool isObstacleClear()
    {
        // Check if obstacles are clear in the local costmap
        auto costmap = local_costmap_->getCostmap();
        unsigned int mx, my;

        // Check multiple directions around the robot
        double robot_x = costmap->getOriginX() + costmap->getSizeInMetersX() / 2.0;
        double robot_y = costmap->getOriginY() + costmap->getSizeInMetersY() / 2.0;

        for (double angle = 0; angle < 2*M_PI; angle += M_PI/4) {
            double check_x = robot_x + 0.5 * cos(angle);  // Check 0.5m out
            double check_y = robot_y + 0.5 * sin(angle);

            if (costmap->worldToMap(check_x, check_y, mx, my)) {
                unsigned char cost = costmap->getCost(mx, my);
                if (cost >= nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE) {
                    return false;  // Found an obstacle
                }
            }
        }

        return true;  // No obstacles detected
    }

    rclcpp_lifecycle::LifecycleNode::SharedPtr node_;
    std::string name_;
    std::shared_ptr<tf2_ros::Buffer> tf_;
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> global_costmap_;
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> local_costmap_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr vel_pub_;

    // Parameters
    double spin_angular_vel_;
    double min_spin_duration_;
    double max_spin_duration_;
};

PLUGINLIB_EXPORT_CLASS(HumanoidSpinRecovery, nav2_core::Recovery)
```

## Performance Evaluation

### Navigation Performance Metrics

```cpp
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.h>

class NavigationPerformanceEvaluator
{
public:
    struct NavigationMetrics
    {
        double success_rate = 0.0;
        double average_time_to_goal = 0.0;
        double path_efficiency = 0.0;  // actual_path_length / optimal_path_length
        double average_velocity = 0.0;
        int collision_count = 0;
        int oscillation_count = 0;
        int recovery_count = 0;
        double energy_efficiency = 0.0;
        double obstacle_avoidance_quality = 0.0;
    };

    NavigationPerformanceEvaluator() {}

    void startTrial(const geometry_msgs::msg::Pose& start, const geometry_msgs::msg::Pose& goal)
    {
        trial_start_time_ = std::chrono::high_resolution_clock::now();
        start_pose_ = start;
        goal_pose_ = goal;
        path_length_ = 0.0;
        collision_count_ = 0;
        oscillation_count_ = 0;
        recovery_count_ = 0;
        previous_pose_ = start;
    }

    void update(const geometry_msgs::msg::Pose& current_pose,
               const geometry_msgs::msg::Twist& cmd_vel,
               bool in_collision = false,
               bool in_recovery = false)
    {
        // Update path length
        double delta = std::sqrt(std::pow(current_pose.position.x - previous_pose_.position.x, 2) +
                                std::pow(current_pose.position.y - previous_pose_.position.y, 2));
        path_length_ += delta;
        previous_pose_ = current_pose;

        // Count collisions
        if (in_collision) {
            collision_count_++;
        }

        // Count oscillations (rapid direction changes)
        if (std::abs(cmd_vel.angular.z) > oscillation_threshold_) {
            oscillation_count_++;
        }

        // Count recovery behaviors
        if (in_recovery) {
            recovery_count_++;
        }
    }

    NavigationMetrics completeTrial(bool success)
    {
        auto end_time = std::chrono::high_resolution_clock::now();
        double trial_time = std::chrono::duration<double>(end_time - trial_start_time_).count();

        NavigationMetrics metrics;
        metrics.success_rate = success ? 1.0 : 0.0;
        metrics.average_time_to_goal = success ? trial_time : 0.0;

        // Calculate optimal path length (straight line)
        double optimal_length = std::sqrt(
            std::pow(goal_pose_.position.x - start_pose_.position.x, 2) +
            std::pow(goal_pose_.position.y - start_pose_.position.y, 2)
        );

        metrics.path_efficiency = (optimal_length > 0) ? path_length_ / optimal_length : 1.0;
        metrics.average_velocity = (trial_time > 0) ? path_length_ / trial_time : 0.0;
        metrics.collision_count = collision_count_;
        metrics.oscillation_count = oscillation_count_;
        metrics.recovery_count = recovery_count_;

        // Energy efficiency could be calculated based on actuator commands
        metrics.energy_efficiency = calculateEnergyEfficiency();

        // Obstacle avoidance quality based on minimum distances to obstacles
        metrics.obstacle_avoidance_quality = calculateObstacleAvoidanceQuality();

        return metrics;
    }

    void printMetrics(const NavigationMetrics& metrics)
    {
        RCLCPP_INFO(rclcpp::get_logger("navigation_eval"),
            "Navigation Performance Metrics:");
        RCLCPP_INFO(rclcpp::get_logger("navigation_eval"),
            "  Success Rate: %.2f", metrics.success_rate);
        RCLCPP_INFO(rclcpp::get_logger("navigation_eval"),
            "  Avg Time to Goal: %.2f s", metrics.average_time_to_goal);
        RCLCPP_INFO(rclcpp::get_logger("navigation_eval"),
            "  Path Efficiency: %.2f", metrics.path_efficiency);
        RCLCPP_INFO(rclcpp::get_logger("navigation_eval"),
            "  Avg Velocity: %.2f m/s", metrics.average_velocity);
        RCLCPP_INFO(rclcpp::get_logger("navigation_eval"),
            "  Collisions: %d", metrics.collision_count);
        RCLCPP_INFO(rclcpp::get_logger("navigation_eval"),
            "  Oscillations: %d", metrics.oscillation_count);
        RCLCPP_INFO(rclcpp::get_logger("navigation_eval"),
            "  Recoveries: %d", metrics.recovery_count);
    }

private:
    double calculateEnergyEfficiency()
    {
        // Placeholder for energy efficiency calculation
        // This would consider motor commands, robot dynamics, etc.
        return 1.0;  // Perfect efficiency for now
    }

    double calculateObstacleAvoidanceQuality()
    {
        // Placeholder for obstacle avoidance quality
        // This would consider minimum distances to obstacles during navigation
        return 1.0;  // Perfect avoidance for now
    }

    std::chrono::high_resolution_clock::time_point trial_start_time_;
    geometry_msgs::msg::Pose start_pose_;
    geometry_msgs::msg::Pose goal_pose_;
    geometry_msgs::msg::Pose previous_pose_;
    double path_length_;
    int collision_count_;
    int oscillation_count_;
    int recovery_count_;

    const double oscillation_threshold_ = 0.5;  // rad/s
};
```

## Best Practices

### 1. Multi-Layered Safety System
Implement multiple layers of safety:
- Perception-based obstacle detection
- Costmap-based obstacle representation
- Collision avoidance algorithms
- Emergency stop mechanisms

### 2. Parameter Tuning
- Use systematic parameter tuning methods
- Test in simulation before real robot deployment
- Monitor performance metrics continuously
- Adapt parameters based on environment conditions

### 3. Humanoid-Specific Considerations
- Balance maintenance during navigation
- Step planning for bipedal locomotion
- Fall prevention mechanisms
- Dynamic stability during turning

## Exercise

Create a complete navigation system that includes:

1. Global path planning with A* algorithm adapted for humanoid robots
2. Local path planning with obstacle avoidance using DWA
3. Integration with perception data from Isaac ROS
4. Step planning for bipedal locomotion
5. Recovery behaviors for humanoid robots
6. Performance evaluation metrics
7. Isaac Sim integration for testing

Test your system in various scenarios including:
- Navigation around static obstacles
- Dynamic obstacle avoidance
- Stair climbing (if applicable)
- Tight spaces navigation
- Multi-goal navigation tasks

Evaluate the system's performance using the metrics discussed in this section.