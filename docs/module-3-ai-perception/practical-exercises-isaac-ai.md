# Practical Exercises with Isaac AI Components

This section provides hands-on exercises to reinforce the concepts learned about Isaac AI components for perception and navigation. These exercises will help you gain practical experience with NVIDIA Isaac technologies.

## Exercise 1: Isaac ROS DetectNet Integration

### Objective
Integrate Isaac ROS DetectNet with a humanoid robot to detect and classify objects in real-time.

### Prerequisites
- Isaac Sim installed and running
- Isaac ROS packages installed
- ROS 2 Humble
- Compatible NVIDIA GPU with TensorRT

### Steps

#### 1. Set up the environment
```bash
# Source ROS 2 and Isaac ROS
source /opt/ros/humble/setup.bash
source ~/isaac_ros_ws/install/setup.bash

# Launch Isaac Sim with a simple scene
# (We'll use a script to automate this in the exercise)
```

#### 2. Create a custom DetectNet configuration
```yaml
# config/detectnet_config.yaml
---
log_level: info
model_name: "ssd_mobilenet_v2_coco"
input_topic: "/camera/image_rect_color"
output_topic: "/detectnet/detections"
confidence_threshold: 0.7
max_objects: 10
```

#### 3. Launch the DetectNet pipeline
```bash
# Create launch file: launch/detectnet_humanoid.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('your_robot_perception'),
        'config',
        'detectnet_config.yaml'
    )

    detectnet_node = Node(
        package='isaac_ros_detectnet',
        executable='isaac_ros_detectnet',
        name='detectnet',
        parameters=[config],
        remappings=[
            ('/image_input', '/camera/image_rect_color'),
            ('/detectnet/detections', '/detections')
        ]
    )

    return LaunchDescription([
        detectnet_node
    ])
```

#### 4. Create a perception processing node
```cpp
// src/object_tracker_node.cpp
#include <rclcpp/rclcpp.hpp>
#include <isaac_ros_detectnet_interfaces/msg/detection_array.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

class ObjectTrackerNode : public rclcpp::Node
{
public:
    ObjectTrackerNode() : Node("object_tracker_node"), tf_buffer_(this->get_clock())
    {
        detection_sub_ = this->create_subscription<isaac_ros_detectnet_interfaces::msg::DetectionArray>(
            "detections", 10,
            std::bind(&ObjectTrackerNode::detectionCallback, this, std::placeholders::_1)
        );

        object_pub_ = this->create_publisher<geometry_msgs::msg::PointStamped>(
            "tracked_object_position", 10
        );

        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(tf_buffer_);
    }

private:
    void detectionCallback(const isaac_ros_detectnet_interfaces::msg::DetectionArray::SharedPtr msg)
    {
        if (msg->detections.empty()) {
            return;
        }

        // Process the highest confidence detection
        auto best_detection = std::max_element(
            msg->detections.begin(),
            msg->detections.end(),
            [](const auto& a, const auto& b) {
                return a.confidence < b.confidence;
            }
        );

        if (best_detection->confidence > 0.7) {
            // Calculate 3D position from 2D detection
            geometry_msgs::msg::PointStamped object_pos;
            object_pos.header = msg->header;

            // Convert 2D bounding box center to 3D position
            // This requires depth information which we'll simulate
            object_pos.point.x = (best_detection->bbox.center.x - 320.0) * 0.001; // Simplified
            object_pos.point.y = (best_detection->bbox.center.y - 240.0) * 0.001; // Simplified
            object_pos.point.z = 1.0; // Fixed depth for simulation

            object_pub_->publish(object_pos);

            RCLCPP_INFO(
                this->get_logger(),
                "Detected %s with confidence %.2f at (%.2f, %.2f, %.2f)",
                best_detection->label.c_str(),
                best_detection->confidence,
                object_pos.point.x,
                object_pos.point.y,
                object_pos.point.z
            );
        }
    }

    rclcpp::Subscription<isaac_ros_detectnet_interfaces::msg::DetectionArray>::SharedPtr detection_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr object_pub_;

    tf2_ros::Buffer tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ObjectTrackerNode>());
    rclcpp::shutdown();
    return 0;
}
```

#### 5. Build and run the exercise
```bash
# Build the package
cd ~/isaac_ros_ws
source install/setup.bash
colcon build --packages-select your_robot_perception

# Run the nodes
ros2 launch your_robot_perception detectnet_humanoid.launch.py
```

### Expected Outcome
A running perception pipeline that detects objects and publishes their 3D positions.

## Exercise 2: Isaac ROS Visual SLAM Integration

### Objective
Set up and run Isaac ROS Visual SLAM to create a map of the environment and localize the robot.

### Steps

#### 1. Create SLAM launch file
```python
# launch/visual_slam.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Visual SLAM node
    visual_slam_node = Node(
        package='isaac_ros_visual_slam',
        executable='isaac_ros_visual_slam',
        parameters=[{
            'enable_occupancy_grid': True,
            'enable_diagnostics': False,
            'occupancy_grid_resolution': 0.05,
            'frame_id': 'oak-d_frame',
            'base_frame': 'base_link',
            'odom_frame': 'odom',
            'enable_slam_visualization': True,
            'enable_landmarks_view': True,
            'enable_observations_view': True,
            'calibration_file': '/tmp/calibration.json',
            'rescale_threshold': 2.0
        }],
        remappings=[
            ('/stereo_camera/left/image', '/camera/left/image_rect_color'),
            ('/stereo_camera/right/image', '/camera/right/image_rect_color'),
            ('/stereo_camera/left/camera_info', '/camera/left/camera_info'),
            ('/stereo_camera/right/camera_info', '/camera/right/camera_info'),
            ('/visual_slam/imu', '/imu/data'),
        ]
    )

    return LaunchDescription([
        visual_slam_node
    ])
```

#### 2. Create a SLAM evaluation node
```cpp
// src/slam_evaluator_node.cpp
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

class SlamevaluatorNode : public rclcpp::Node
{
public:
    SlamevaluatorNode() : Node("slam_evaluator_node")
    {
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "visual_slam/odometry", 10,
            std::bind(&SlamevaluatorNode::odometryCallback, this, std::placeholders::_1)
        );

        pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            "estimated_pose", 10
        );

        RCLCPP_INFO(this->get_logger(), "SLAM Evaluator Node initialized");
    }

private:
    void odometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        // Extract position and orientation from odometry
        geometry_msgs::msg::PoseStamped pose_msg;
        pose_msg.header = msg->header;
        pose_msg.pose = msg->pose.pose;

        // Publish the estimated pose
        pose_pub_->publish(pose_msg);

        // Calculate and log trajectory metrics
        if (has_previous_pose_) {
            double distance = calculateDistance(previous_pose_, pose_msg.pose);
            total_distance_ += distance;

            RCLCPP_INFO(
                this->get_logger(),
                "SLAM Position: (%.2f, %.2f, %.2f), Total distance: %.2f",
                pose_msg.pose.position.x,
                pose_msg.pose.position.y,
                pose_msg.pose.position.z,
                total_distance_
            );
        }

        previous_pose_ = pose_msg.pose;
        has_previous_pose_ = true;
    }

    double calculateDistance(const geometry_msgs::msg::Pose& p1, const geometry_msgs::msg::Pose& p2)
    {
        double dx = p1.position.x - p2.position.x;
        double dy = p1.position.y - p2.position.y;
        double dz = p1.position.z - p2.position.z;
        return sqrt(dx*dx + dy*dy + dz*dz);
    }

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;

    geometry_msgs::msg::Pose previous_pose_;
    bool has_previous_pose_ = false;
    double total_distance_ = 0.0;
};
```

#### 3. Run the SLAM exercise
```bash
# Launch SLAM
ros2 launch your_robot_perception visual_slam.launch.py

# Visualize results
ros2 run rviz2 rviz2 -d /path/to/slam_config.rviz
```

### Expected Outcome
A real-time map of the environment with the robot's estimated position and trajectory.

## Exercise 3: Isaac ROS Bi3D Integration

### Objective
Use Isaac ROS Bi3D for 3D object detection and segmentation.

### Steps

#### 1. Create Bi3D launch configuration
```python
# launch/bi3d_segmentation.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    bi3d_node = Node(
        package='isaac_ros_bi3d',
        executable='isaac_ros_bi3d',
        parameters=[{
            'engine_file_path': '/path/to/bi3d_plan_engine.plan',
            'input_tensor_names': ['input_tensor'],
            'output_tensor_names': ['output_tensor'],
            'network_input_height': 512,
            'network_input_width': 512,
            'num_classes': 256,
            'mask_threshold': 0.8
        }],
        remappings=[
            ('/image', '/camera/image_rect_color'),
            ('/segmentation', '/bi3d_segmentation')
        ]
    )

    return LaunchDescription([
        bi3d_node
    ])
```

#### 2. Create a 3D object extraction node
```cpp
// src/bi3d_processor_node.cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class Bi3DProcessorNode : public rclcpp::Node
{
public:
    Bi3DProcessorNode() : Node("bi3d_processor_node")
    {
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "camera/image_rect_color", 10,
            std::bind(&Bi3DProcessorNode::imageCallback, this, std::placeholders::_1)
        );

        segmentation_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "bi3d_segmentation", 10,
            std::bind(&Bi3DProcessorNode::segmentationCallback, this, std::placeholders::_1)
        );

        object_pub_ = this->create_publisher<geometry_msgs::msg::PointStamped>(
            "extracted_3d_objects", 10
        );

        RCLCPP_INFO(this->get_logger(), "Bi3D Processor Node initialized");
    }

private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr image_msg)
    {
        // Store image for processing with segmentation
        latest_image_ = image_msg;
    }

    void segmentationCallback(const sensor_msgs::msg::Image::SharedPtr seg_msg)
    {
        if (!latest_image_) {
            return;  // Wait for image
        }

        try {
            // Convert segmentation image to OpenCV
            cv_bridge::CvImagePtr seg_cv_ptr = cv_bridge::toCvCopy(seg_msg, sensor_msgs::image_encodings::TYPE_32SC1);

            // Process segmentation to extract 3D objects
            std::vector<DetectedObject> objects = extractObjects(seg_cv_ptr->image);

            // Publish each detected object
            for (const auto& obj : objects) {
                geometry_msgs::msg::PointStamped obj_pos;
                obj_pos.header = seg_msg->header;
                obj_pos.point = obj.centroid;

                object_pub_->publish(obj_pos);

                RCLCPP_INFO(
                    this->get_logger(),
                    "3D Object: %s at (%.2f, %.2f, %.2f), pixels: %d",
                    obj.label.c_str(),
                    obj.centroid.x,
                    obj.centroid.y,
                    obj.centroid.z,
                    obj.pixel_count
                );
            }
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }

    struct DetectedObject {
        std::string label;
        geometry_msgs::msg::Point centroid;
        int pixel_count;
        cv::Rect bounding_box;
    };

    std::vector<DetectedObject> extractObjects(const cv::Mat& segmentation_mask)
    {
        std::vector<DetectedObject> objects;

        // Find contours for each class in the segmentation
        for (int class_id = 1; class_id < 256; ++class_id) {  // Skip background (0)
            cv::Mat class_mask = (segmentation_mask == class_id);

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(class_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            for (const auto& contour : contours) {
                if (contour.size() < 50) continue;  // Filter small regions

                DetectedObject obj;
                obj.pixel_count = contour.size();

                // Calculate centroid
                cv::Moments moments = cv::moments(contour);
                if (moments.m00 != 0) {
                    int cx = static_cast<int>(moments.m10 / moments.m00);
                    int cy = static_cast<int>(moments.m01 / moments.m00);

                    // Estimate 3D position (simplified)
                    obj.centroid.x = (cx - 320.0) * 0.002;  // Approximate conversion
                    obj.centroid.y = (cy - 240.0) * 0.002;
                    obj.centroid.z = 1.0;  // Estimated depth

                    obj.bounding_box = cv::boundingRect(contour);

                    // Assign label based on class ID (in real system, use a mapping)
                    obj.label = "object_" + std::to_string(class_id);

                    objects.push_back(obj);
                }
            }
        }

        return objects;
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr segmentation_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr object_pub_;

    sensor_msgs::msg::Image::SharedPtr latest_image_;
};
```

### Expected Outcome
A system that segments the scene into 3D objects and publishes their positions.

## Exercise 4: Isaac Sim Perception Pipeline

### Objective
Create a complete perception pipeline in Isaac Sim that integrates multiple Isaac ROS components.

### Steps

#### 1. Create Isaac Sim perception scene
```python
# scripts/setup_perception_scene.py
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.robots import Robot
from omni.isaac.range_sensor import RotatingLidarPhysX
from omni.isaac.sensor import Camera
import numpy as np

class PerceptionSceneSetup:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.setup_scene()

    def setup_scene(self):
        # Get assets root path
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            print("Could not find Isaac Sim assets")
            return

        # Add a robot with perception sensors
        robot_path = assets_root_path + "/Isaac/Robots/Carter/carter_navigate.usd"
        add_reference_to_stage(usd_path=robot_path, prim_path="/World/Carter")

        # Add a camera sensor
        self.camera = Camera(
            prim_path="/World/Carter/chassis/camera",
            frequency=30,
            resolution=(640, 480)
        )

        # Add a LIDAR sensor
        self.lidar = RotatingLidarPhysX(
            prim_path="/World/Carter/chassis/lidar",
            translation=np.array([0.0, 0.0, 0.25]),
            config="Carter",
            rotation_frequency=10,
            samples_per_scan=1080
        )

        # Add some objects to detect
        cube_path = assets_root_path + "/Isaac/Props/Blocks/block_instanceable.usd"
        add_reference_to_stage(usd_path=cube_path, prim_path="/World/Cube1")
        from pxr import Gf
        from omni.isaac.core.utils.prims import set_targets
        from omni.isaac.core.utils.transformations import quat_from_euler_angles

        # Position the cube in front of the robot
        cube_prim = self.world.scene.add_static_object(
            prim_path="/World/Cube1",
            usd_path=cube_path,
            position=[2.0, 0.0, 0.5],
            orientation=quat_from_euler_angles([0, 0, 0])
        )

        # Initialize the world
        self.world.reset()

    def run_perception_pipeline(self):
        """Run the perception pipeline"""
        while not self.world.is_stopped():
            self.world.step(render=True)

            # Get sensor data
            camera_data = self.camera.get_rgb()
            lidar_data = self.lidar.get_linear_depth_data()

            # Process perception data (placeholder)
            self.process_perception_data(camera_data, lidar_data)

    def process_perception_data(self, camera_data, lidar_data):
        """Process perception data"""
        print(f"Camera data shape: {camera_data.shape if hasattr(camera_data, 'shape') else 'N/A'}")
        print(f"LIDAR data points: {len(lidar_data) if hasattr(lidar_data, '__len__') else 'N/A'}")

# Run the scene setup
if __name__ == "__main__":
    scene_setup = PerceptionSceneSetup()
    scene_setup.run_perception_pipeline()
```

#### 2. Connect Isaac Sim to ROS
```python
# scripts/isaac_sim_ros_bridge.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import numpy as np

class IsaacSimRosBridge(Node):
    def __init__(self):
        super().__init__('isaac_sim_ros_bridge')

        # ROS publishers
        self.image_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)

        # ROS subscriber for robot commands
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10
        )

        self.bridge = CvBridge()

        # Timer to publish sensor data
        self.timer = self.create_timer(0.1, self.publish_sensor_data)

        # Store robot commands
        self.last_cmd_vel = None

    def cmd_vel_callback(self, msg):
        """Store command velocity for Isaac Sim"""
        self.last_cmd_vel = msg
        # In a real implementation, this would send commands to Isaac Sim

    def publish_sensor_data(self):
        """Publish sensor data from Isaac Sim"""
        # This would normally connect to Isaac Sim
        # For this exercise, we'll simulate data

        # Publish a simulated image
        sim_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        image_msg = self.bridge.cv2_to_imgmsg(sim_image, encoding="bgr8")
        image_msg.header.stamp = self.get_clock().now().to_msg()
        image_msg.header.frame_id = "camera_link"
        self.image_pub.publish(image_msg)

        # Publish a simulated scan
        scan_msg = LaserScan()
        scan_msg.header.stamp = self.get_clock().now().to_msg()
        scan_msg.header.frame_id = "lidar_link"
        scan_msg.angle_min = -np.pi / 2
        scan_msg.angle_max = np.pi / 2
        scan_msg.angle_increment = np.pi / 180  # 1 degree
        scan_msg.time_increment = 0.0
        scan_msg.scan_time = 0.1
        scan_msg.range_min = 0.1
        scan_msg.range_max = 10.0
        scan_msg.ranges = [5.0] * 180  # Simulated ranges

        self.scan_pub.publish(scan_msg)

def main(args=None):
    rclpy.init(args=args)
    bridge = IsaacSimRosBridge()

    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Expected Outcome
A complete perception pipeline running in Isaac Sim that connects to ROS and processes data from multiple sensors.

## Exercise 5: Integrated Perception and Navigation

### Objective
Combine perception and navigation systems to create a complete autonomous robot behavior.

### Steps

#### 1. Create integrated launch file
```python
# launch/integrated_perception_navigation.launch.py
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Include perception pipeline
    perception_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                get_package_share_directory('your_robot_perception'),
                'launch',
                'detectnet_humanoid.launch.py'
            ])
        ])
    )

    # Include navigation stack
    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                get_package_share_directory('nav2_bringup'),
                'launch',
                'navigation_launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': 'true'
        }.items()
    )

    # Object avoidance node
    object_avoidance_node = Node(
        package='your_robot_perception',
        executable='object_avoidance_node',
        name='object_avoidance',
        parameters=[
            {'safety_distance': 0.5},
            {'avoidance_strength': 1.0}
        ],
        remappings=[
            ('/detected_objects', '/tracked_object_position'),
            ('/cmd_vel_safe', '/cmd_vel_filtered'),
            ('/cmd_vel_in', '/cmd_vel')
        ]
    )

    return LaunchDescription([
        perception_launch,
        navigation_launch,
        object_avoidance_node
    ])
```

#### 2. Create object avoidance controller
```cpp
// src/object_avoidance_node.cpp
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

class ObjectAvoidanceNode : public rclcpp::Node
{
public:
    ObjectAvoidanceNode() : Node("object_avoidance_node")
    {
        cmd_vel_in_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "cmd_vel_in", 10,
            std::bind(&ObjectAvoidanceNode::cmdVelCallback, this, std::placeholders::_1)
        );

        detected_objects_sub_ = this->create_subscription<geometry_msgs::msg::PointStamped>(
            "detected_objects", 10,
            std::bind(&ObjectAvoidanceNode::objectCallback, this, std::placeholders::_1)
        );

        cmd_vel_out_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(
            "cmd_vel_safe", 10
        );

        // Get parameters
        safety_distance_ = this->declare_parameter("safety_distance", 0.5);
        avoidance_strength_ = this->declare_parameter("avoidance_strength", 1.0);

        RCLCPP_INFO(this->get_logger(), "Object Avoidance Node initialized");
    }

private:
    void cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr cmd_msg)
    {
        latest_cmd_vel_ = *cmd_msg;
        applyObjectAvoidance();
    }

    void objectCallback(const geometry_msgs::msg::PointStamped::SharedPtr obj_msg)
    {
        // Store object position for avoidance calculation
        detected_objects_.push_back(*obj_msg);

        // Keep only recent detections (last 1 second)
        auto current_time = this->now();
        detected_objects_.erase(
            std::remove_if(detected_objects_.begin(), detected_objects_.end(),
                [current_time, this](const geometry_msgs::msg::PointStamped& obj) {
                    return (current_time - obj.header.stamp).seconds() > 1.0;
                }),
            detected_objects_.end()
        );
    }

    void applyObjectAvoidance()
    {
        if (!latest_cmd_vel_) {
            return;
        }

        geometry_msgs::msg::Twist safe_cmd = *latest_cmd_vel_;

        // Check for nearby objects that require avoidance
        for (const auto& obj : detected_objects_) {
            // Calculate distance to object in robot's frame
            double dist_to_obj = sqrt(
                pow(obj.point.x, 2) +
                pow(obj.point.y, 2) +
                pow(obj.point.z, 2)
            );

            if (dist_to_obj < safety_distance_) {
                // Calculate avoidance vector
                double avoidance_x = -obj.point.x * avoidance_strength_ / dist_to_obj;
                double avoidance_y = -obj.point.y * avoidance_strength_ / dist_to_obj;

                // Apply avoidance to commanded velocity
                safe_cmd.linear.x += avoidance_x;
                safe_cmd.linear.y += avoidance_y;

                // Add angular component to turn away from object
                double angle_to_obj = atan2(obj.point.y, obj.point.x);
                safe_cmd.angular.z -= angle_to_obj * avoidance_strength_ * 0.5;
            }
        }

        // Apply velocity limits
        safe_cmd.linear.x = std::clamp(safe_cmd.linear.x, -0.5, 0.5);
        safe_cmd.linear.y = std::clamp(safe_cmd.linear.y, -0.2, 0.2);
        safe_cmd.angular.z = std::clamp(safe_cmd.angular.z, -0.5, 0.5);

        cmd_vel_out_pub_->publish(safe_cmd);
    }

    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_in_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr detected_objects_sub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_out_pub_;

    std::optional<geometry_msgs::msg::Twist> latest_cmd_vel_;
    std::vector<geometry_msgs::msg::PointStamped> detected_objects_;

    double safety_distance_;
    double avoidance_strength_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ObjectAvoidanceNode>());
    rclcpp::shutdown();
    return 0;
}
```

#### 3. Create a mission planner node
```cpp
// src/mission_planner_node.cpp
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav2_msgs/action/navigate_to_pose.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

class MissionPlannerNode : public rclcpp::Node
{
public:
    MissionPlannerNode() : Node("mission_planner_node")
    {
        nav_client_ = rclcpp_action::create_client<nav2_msgs::action::NavigateToPose>(
            this, "navigate_to_pose"
        );

        // Define a simple mission: visit multiple waypoints
        waypoints_ = {
            createPose(1.0, 1.0, 0.0),
            createPose(2.0, 0.0, 1.57),
            createPose(1.0, -1.0, 3.14),
            createPose(0.0, 0.0, 0.0)  // Return to start
        };

        mission_timer_ = this->create_wall_timer(
            std::chrono::seconds(5),
            std::bind(&MissionPlannerNode::executeNextWaypoint, this)
        );

        current_waypoint_ = 0;
        mission_active_ = false;

        RCLCPP_INFO(this->get_logger(), "Mission Planner Node initialized");
    }

private:
    geometry_msgs::msg::PoseStamped createPose(double x, double y, double theta)
    {
        geometry_msgs::msg::PoseStamped pose;
        pose.header.frame_id = "map";
        pose.pose.position.x = x;
        pose.pose.position.y = y;
        pose.pose.position.z = 0.0;

        tf2::Quaternion q;
        q.setRPY(0, 0, theta);
        pose.pose.orientation = tf2::toMsg(q);

        return pose;
    }

    void executeNextWaypoint()
    {
        if (mission_active_ || current_waypoint_ >= waypoints_.size()) {
            return;  // Wait for current navigation to complete
        }

        if (!nav_client_->wait_for_action_server(std::chrono::seconds(5))) {
            RCLCPP_ERROR(this->get_logger(), "Navigation action server not available");
            return;
        }

        auto goal = nav2_msgs::action::NavigateToPose::Goal();
        goal.pose = waypoints_[current_waypoint_];

        auto send_goal_options = rclcpp_action::Client<nav2_msgs::action::NavigateToPose>::SendGoalOptions();
        send_goal_options.result_callback =
            [this](const rclcpp_action::ClientGoalHandle<nav2_msgs::action::NavigateToPose>::WrappedResult& result) {
                if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
                    RCLCPP_INFO(this->get_logger(), "Waypoint %d reached!", current_waypoint_);
                    current_waypoint_++;
                    mission_active_ = false;

                    if (current_waypoint_ >= waypoints_.size()) {
                        RCLCPP_INFO(this->get_logger(), "Mission completed!");
                    }
                } else {
                    RCLCPP_ERROR(this->get_logger(), "Failed to reach waypoint %d", current_waypoint_);
                    mission_active_ = false;
                }
            };

        mission_active_ = true;
        nav_client_->async_send_goal(goal, send_goal_options);
    }

    rclcpp_action::Client<nav2_msgs::action::NavigateToPose>::SharedPtr nav_client_;
    rclcpp::TimerBase::SharedPtr mission_timer_;

    std::vector<geometry_msgs::msg::PoseStamped> waypoints_;
    size_t current_waypoint_;
    bool mission_active_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MissionPlannerNode>());
    rclcpp::shutdown();
    return 0;
}
```

### Expected Outcome
A complete system that performs perception, navigation, and mission planning with object avoidance capabilities.

## Exercise 6: Performance Evaluation and Optimization

### Objective
Evaluate and optimize the performance of your Isaac AI perception system.

### Steps

#### 1. Create performance monitoring node
```cpp
// src/performance_monitor_node.cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>
#include <chrono>

class PerformanceMonitorNode : public rclcpp::Node
{
public:
    PerformanceMonitorNode() : Node("performance_monitor_node")
    {
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "camera/image_raw", 10,
            std::bind(&PerformanceMonitorNode::imageCallback, this, std::placeholders::_1)
        );

        stats_timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&PerformanceMonitorNode::printStats, this)
        );

        RCLCPP_INFO(this->get_logger(), "Performance Monitor Node initialized");
    }

private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        auto current_time = std::chrono::steady_clock::now();

        // Calculate frame rate
        if (last_frame_time_.has_value()) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                current_time - last_frame_time_.value()
            ).count();
            frame_times_.push_back(duration);
        }

        last_frame_time_ = current_time;

        // Calculate bandwidth
        size_t msg_size = msg->data.size();
        total_bytes_processed_ += msg_size;
        messages_processed_++;
    }

    void printStats()
    {
        if (frame_times_.empty()) return;

        // Calculate average frame time
        double avg_frame_time = 0.0;
        for (auto time : frame_times_) {
            avg_frame_time += time;
        }
        avg_frame_time /= frame_times_.size();

        // Calculate frame rate
        double avg_fps = 1e6 / avg_frame_time;  // microseconds to seconds

        // Calculate bandwidth
        double bandwidth = total_bytes_processed_ / 1e6;  // MB
        double avg_msg_size = (messages_processed_ > 0) ?
            static_cast<double>(total_bytes_processed_) / messages_processed_ : 0;

        RCLCPP_INFO(
            this->get_logger(),
            "Performance Stats - FPS: %.2f, Avg Frame Time: %.2f ms, "
            "Avg Msg Size: %.2f KB, Total Processed: %.2f MB",
            avg_fps, avg_frame_time / 1000.0, avg_msg_size / 1024.0, bandwidth
        );

        // Clear for next interval
        frame_times_.clear();
        total_bytes_processed_ = 0;
        messages_processed_ = 0;
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::TimerBase::SharedPtr stats_timer_;

    std::optional<std::chrono::steady_clock::time_point> last_frame_time_;
    std::vector<long> frame_times_;  // in microseconds
    size_t total_bytes_processed_ = 0;
    size_t messages_processed_ = 0;
};
```

#### 2. Create optimization report
```cpp
// scripts/generate_performance_report.py
#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime

class PerformanceAnalyzer:
    def __init__(self):
        self.metrics = []

    def load_metrics(self, metrics_file):
        """Load performance metrics from file"""
        with open(metrics_file, 'r') as f:
            self.metrics = json.load(f)

    def generate_report(self):
        """Generate performance analysis report"""
        if not self.metrics:
            print("No metrics loaded")
            return

        df = pd.DataFrame(self.metrics)

        # Create visualizations
        self.create_visualizations(df)

        # Generate recommendations
        self.generate_recommendations(df)

        # Export detailed report
        self.export_report(df)

    def create_visualizations(self, df):
        """Create performance visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Frame rate over time
        axes[0, 0].plot(df['timestamp'], df['fps'])
        axes[0, 0].set_title('Frame Rate Over Time')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('FPS')

        # Processing time distribution
        axes[0, 1].hist(df['avg_frame_time_ms'], bins=30)
        axes[0, 1].set_title('Frame Processing Time Distribution')
        axes[0, 1].set_xlabel('Processing Time (ms)')
        axes[0, 1].set_ylabel('Frequency')

        # Bandwidth usage
        axes[1, 0].plot(df['timestamp'], df['bandwidth_mb'])
        axes[1, 0].set_title('Bandwidth Usage Over Time')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Bandwidth (MB)')

        # Correlation heatmap
        correlation_data = df[['fps', 'avg_frame_time_ms', 'bandwidth_mb']].corr()
        im = axes[1, 1].imshow(correlation_data, cmap='coolwarm', aspect='auto')
        axes[1, 1].set_xticks(range(len(correlation_data.columns)))
        axes[1, 1].set_yticks(range(len(correlation_data.columns)))
        axes[1, 1].set_xticklabels(correlation_data.columns, rotation=45)
        axes[1, 1].set_yticklabels(correlation_data.columns)
        axes[1, 1].set_title('Performance Metric Correlations')

        # Add colorbar
        plt.colorbar(im, ax=axes[1, 1])

        plt.tight_layout()
        plt.savefig('performance_analysis.png')
        plt.show()

    def generate_recommendations(self, df):
        """Generate optimization recommendations"""
        avg_fps = df['fps'].mean()
        avg_time = df['avg_frame_time_ms'].mean()
        avg_bandwidth = df['bandwidth_mb'].mean()

        print("PERFORMANCE ANALYSIS REPORT")
        print("=" * 50)
        print(f"Average Frame Rate: {avg_fps:.2f} FPS")
        print(f"Average Processing Time: {avg_time:.2f} ms")
        print(f"Average Bandwidth: {avg_bandwidth:.2f} MB")
        print()

        recommendations = []

        if avg_fps < 15:
            recommendations.append("LOW FPS: Consider reducing image resolution or using lighter models")
        if avg_time > 50:
            recommendations.append("HIGH PROCESSING TIME: Optimize algorithms or use TensorRT")
        if avg_bandwidth > 100:
            recommendations.append("HIGH BANDWIDTH: Compress images or reduce frequency")

        if not recommendations:
            recommendations.append("Performance looks good! Consider profiling specific bottlenecks.")

        print("OPTIMIZATION RECOMMENDATIONS:")
        for rec in recommendations:
            print(f"- {rec}")
        print()

    def export_report(self, df):
        """Export detailed report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"performance_report_{timestamp}.txt"

        with open(report_filename, 'w') as f:
            f.write("Isaac AI Performance Report\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            f.write(f"Summary Statistics:\n{df.describe()}\n\n")

        print(f"Detailed report exported to {report_filename}")

if __name__ == "__main__":
    analyzer = PerformanceAnalyzer()
    # analyzer.load_metrics('performance_metrics.json')  # Load your metrics
    # analyzer.generate_report()
```

### Expected Outcome
A complete system for monitoring, analyzing, and optimizing Isaac AI component performance with actionable insights.

## Troubleshooting Common Issues

### 1. TensorRT Optimization Issues
**Problem**: Models not running at expected speeds
**Solutions**:
- Verify TensorRT engine files are properly built
- Check GPU memory availability
- Use appropriate batch sizes
- Profile with NSight Systems

### 2. Memory Issues
**Problem**: GPU memory exhaustion
**Solutions**:
- Reduce model input resolution
- Use model quantization
- Implement memory pooling
- Monitor memory usage during runtime

### 3. Synchronization Issues
**Problem**: Sensor data timing problems
**Solutions**:
- Use message filters for time synchronization
- Implement proper buffer sizes
- Use reliable QoS policies
- Add timestamps to messages

## Best Practices

### 1. Modular Design
- Separate perception, planning, and control components
- Use standard ROS interfaces
- Implement proper error handling
- Design for easy testing and debugging

### 2. Performance Optimization
- Use TensorRT for inference acceleration
- Implement efficient data pipelines
- Use multi-threading where appropriate
- Profile regularly and optimize bottlenecks

### 3. Robustness
- Handle sensor failures gracefully
- Implement fallback behaviors
- Validate inputs and outputs
- Test in diverse conditions

## Exercise Completion Checklist

After completing these exercises, you should be able to:

- [ ] Set up Isaac ROS perception components
- [ ] Integrate multiple Isaac AI modules
- [ ] Process and interpret perception data
- [ ] Implement perception-driven navigation
- [ ] Evaluate and optimize performance
- [ ] Troubleshoot common issues
- [ ] Design robust perception systems

Successfully completing these exercises will provide you with hands-on experience with NVIDIA Isaac AI components and prepare you for implementing perception and navigation systems on real humanoid robots.