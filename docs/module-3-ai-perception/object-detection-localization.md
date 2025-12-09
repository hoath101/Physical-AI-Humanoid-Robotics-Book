# Object Detection and Localization Examples

This section provides practical examples of object detection and localization using NVIDIA Isaac technologies, demonstrating how AI-powered perception systems work in robotics applications.

## Introduction to Object Detection in Robotics

Object detection in robotics involves identifying and localizing objects in the robot's environment. This capability is crucial for:
- Navigation and path planning
- Manipulation and grasping
- Scene understanding
- Human-robot interaction
- Autonomous decision making

## Isaac ROS Perception Pipeline

### Overview of Isaac ROS Perception Stack

````
Isaac ROS Perception
├── Isaac ROS Image Pipeline
│   ├── Image Proc
│   ├── Rectification
│   └── Format Conversion
├── Isaac ROS Visual SLAM
│   ├── Feature Detection
│   ├── Pose Estimation
│   └── Map Building
├── Isaac ROS Object Detection
│   ├── Deep Learning Models
│   ├── TensorRT Optimization
│   └── Post-processing
├── Isaac ROS Pose Estimation
│   ├── 2D-3D Correspondence
│   ├── PnP Solvers
│   └── Refinement
└── Isaac ROS Bi3D
    ├── 3D Segmentation
    ├── Depth Estimation
    └── Instance Segmentation
```

## Isaac ROS Object Detection Examples

### 1. Isaac ROS DetectNet

DetectNet is NVIDIA's specialized network for object detection optimized for robotics applications.

#### Basic DetectNet Node Implementation

```cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <isaac_ros_detectnet_interfaces/msg/detection_array.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class IsaacDetectNetNode : public rclcpp::Node
{
public:
    IsaacDetectNetNode() : Node("isaac_detectnet_node")
    {
        // Create subscribers
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "image_input", 10,
            std::bind(&IsaacDetectNetNode::imageCallback, this, std::placeholders::_1)
        );

        camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "camera_info", 10,
            std::bind(&IsaacDetectNetNode::cameraInfoCallback, this, std::placeholders::_1)
        );

        // Create publisher for detections
        detection_pub_ = this->create_publisher<isaac_ros_detectnet_interfaces::msg::DetectionArray>(
            "detections", 10
        );
    }

private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr image_msg)
    {
        // Convert ROS image to OpenCV
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        // Process image through DetectNet model
        auto detections = runDetectNetInference(cv_ptr->image);

        // Create detection message
        auto detection_msg = createDetectionMessage(detections, image_msg->header);

        // Publish detections
        detection_pub_->publish(detection_msg);
    }

    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr info_msg)
    {
        camera_info_ = *info_msg;
    }

    std::vector<Detection> runDetectNetInference(const cv::Mat& image)
    {
        // This would interface with the actual DetectNet model
        // In practice, this uses TensorRT for optimized inference
        std::vector<Detection> detections;

        // Placeholder for actual inference
        // In real implementation, this would:
        // 1. Preprocess image for the model
        // 2. Run inference using TensorRT
        // 3. Post-process results
        // 4. Apply non-maximum suppression
        // 5. Filter by confidence threshold

        return detections;
    }

    isaac_ros_detectnet_interfaces::msg::DetectionArray createDetectionMessage(
        const std::vector<Detection>& detections,
        const std_msgs::msg::Header& header)
    {
        isaac_ros_detectnet_interfaces::msg::DetectionArray detection_array;
        detection_array.header = header;

        for (const auto& detection : detections) {
            isaac_ros_detectnet_interfaces::msg::Detection det_msg;
            det_msg.label = detection.label;
            det_msg.confidence = detection.confidence;

            // Bounding box coordinates
            det_msg.bbox.center.x = detection.center_x;
            det_msg.bbox.center.y = detection.center_y;
            det_msg.bbox.size_x = detection.width;
            det_msg.bbox.size_y = detection.height;

            detection_array.detections.push_back(det_msg);
        }

        return detection_array;
    }

    struct Detection {
        std::string label;
        float confidence;
        float center_x, center_y;
        float width, height;
    };

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
    rclcpp::Publisher<isaac_ros_detectnet_interfaces::msg::DetectionArray>::SharedPtr detection_pub_;
    sensor_msgs::msg::CameraInfo camera_info_;
};
```

#### DetectNet Launch Configuration

```xml
<!-- detectnet.launch.xml -->
<launch>
  <!-- Image rectification -->
  <node pkg="isaac_ros_image_proc" exec="isaac_ros_image_proc" name="image_proc">
    <param name="input_encoding" value="bgr8"/>
    <param name="output_encoding" value="bgr8"/>
  </node>

  <!-- DetectNet node -->
  <node pkg="isaac_ros_detectnet" exec="isaac_ros_detectnet" name="detectnet">
    <param name="model_name" value="ssd_mobilenet_v2_coco"/>
    <param name="input_topic" value="/image_rect_color"/>
    <param name="output_topic" value="/detections"/>
    <param name="confidence_threshold" value="0.5"/>
    <param name="max_objects" value="10"/>
  </node>

  <!-- Visualization node -->
  <node pkg="isaac_ros_visualization" exec="detection_visualizer" name="detection_visualizer">
    <param name="image_topic" value="/image_rect_color"/>
    <param name="detection_topic" value="/detections"/>
    <param name="output_topic" value="/detection_image"/>
  </node>
</launch>
```

### 2. Isaac ROS Bi3D (3D Object Detection)

Bi3D provides 3D object detection and segmentation capabilities.

#### Bi3D Node Implementation

```cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <stereo_msgs/msg/disparity_image.hpp>
#include <isaac_ros_bi3d_interfaces/msg/bi3_d_inference_array.hpp>

class IsaacBi3DNode : public rclcpp::Node
{
public:
    IsaacBi3DNode() : Node("isaac_bi3d_node")
    {
        // Subscribe to stereo image pair
        left_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "left/image_rect_color", 10,
            std::bind(&IsaacBi3DNode::leftImageCallback, this, std::placeholders::_1)
        );

        right_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "right/image_rect_color", 10,
            std::bind(&IsaacBi3DNode::rightImageCallback, this, std::placeholders::_1)
        );

        // Subscribe to disparity for depth
        disparity_sub_ = this->create_subscription<stereo_msgs::msg::DisparityImage>(
            "disparity", 10,
            std::bind(&IsaacBi3DNode::disparityCallback, this, std::placeholders::_1)
        );

        // Publisher for 3D detections
        bi3d_pub_ = this->create_publisher<isaac_ros_bi3d_interfaces::msg::Bi3DInferenceArray>(
            "bi3d_detections", 10
        );
    }

private:
    void leftImageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        if (has_right_image_ && has_disparity_) {
            processStereoPair(msg, right_image_, disparity_);
        } else {
            left_image_ = msg;
        }
    }

    void rightImageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        has_right_image_ = true;
        right_image_ = msg;

        if (has_left_image_ && has_disparity_) {
            processStereoPair(left_image_, msg, disparity_);
        }
    }

    void disparityCallback(const stereo_msgs::msg::DisparityImage::SharedPtr msg)
    {
        has_disparity_ = true;
        disparity_ = msg;

        if (has_left_image_ && has_right_image_) {
            processStereoPair(left_image_, right_image_, msg);
        }
    }

    void processStereoPair(
        const sensor_msgs::msg::Image::SharedPtr left,
        const sensor_msgs::msg::Image::SharedPtr right,
        const stereo_msgs::msg::DisparityImage::SharedPtr disparity)
    {
        // Run Bi3D inference
        auto bi3d_results = runBi3DInference(left, right);

        // Create 3D detection message
        auto bi3d_msg = createBi3DMessage(bi3d_results, left->header);

        // Publish results
        bi3d_pub_->publish(bi3d_msg);

        // Reset flags
        has_left_image_ = false;
        has_right_image_ = false;
        has_disparity_ = false;
    }

    std::vector<Bi3DResult> runBi3DInference(
        const sensor_msgs::msg::Image::SharedPtr left,
        const sensor_msgs::msg::Image::SharedPtr right)
    {
        // Placeholder for actual Bi3D inference
        // This would:
        // 1. Process stereo images through Bi3D network
        // 2. Generate 3D segmentation masks
        // 3. Extract 3D bounding boxes
        // 4. Estimate 3D poses

        std::vector<Bi3DResult> results;
        // Implementation would go here
        return results;
    }

    isaac_ros_bi3d_interfaces::msg::Bi3DInferenceArray createBi3DMessage(
        const std::vector<Bi3DResult>& results,
        const std_msgs::msg::Header& header)
    {
        isaac_ros_bi3d_interfaces::msg::Bi3DInferenceArray bi3d_array;
        bi3d_array.header = header;

        for (const auto& result : results) {
            isaac_ros_bi3d_interfaces::msg::Bi3DInference bi3d_msg;
            bi3d_msg.class_id = result.class_id;
            bi3d_msg.confidence = result.confidence;

            // 3D bounding box
            bi3d_msg.bounding_box_3d.center.position.x = result.center_x;
            bi3d_msg.bounding_box_3d.center.position.y = result.center_y;
            bi3d_msg.bounding_box_3d.center.position.z = result.center_z;

            // Convert Euler angles to quaternion
            tf2::Quaternion q;
            q.setRPY(result.roll, result.pitch, result.yaw);
            bi3d_msg.bounding_box_3d.center.orientation.x = q.x();
            bi3d_msg.bounding_box_3d.center.orientation.y = q.y();
            bi3d_msg.bounding_box_3d.center.orientation.z = q.z();
            bi3d_msg.bounding_box_3d.center.orientation.w = q.w();

            bi3d_msg.bounding_box_3d.size.x = result.size_x;
            bi3d_msg.bounding_box_3d.size.y = result.size_y;
            bi3d_msg.bounding_box_3d.size.z = result.size_z;

            bi3d_array.inferences.push_back(bi3d_msg);
        }

        return bi3d_array;
    }

    struct Bi3DResult {
        int class_id;
        float confidence;
        float center_x, center_y, center_z;
        float roll, pitch, yaw;
        float size_x, size_y, size_z;
    };

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr left_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr right_sub_;
    rclcpp::Subscription<stereo_msgs::msg::DisparityImage>::SharedPtr disparity_sub_;
    rclcpp::Publisher<isaac_ros_bi3d_interfaces::msg::Bi3DInferenceArray>::SharedPtr bi3d_pub_;

    sensor_msgs::msg::Image::SharedPtr left_image_;
    sensor_msgs::msg::Image::SharedPtr right_image_;
    stereo_msgs::msg::DisparityImage::SharedPtr disparity_;

    bool has_left_image_ = false;
    bool has_right_image_ = false;
    bool has_disparity_ = false;
};
```

## Object Localization Examples

### 1. Camera-Object 3D Localization

```cpp
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <isaac_ros_detectnet_interfaces/msg/detection_array.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

class ObjectLocalizationNode : public rclcpp::Node
{
public:
    ObjectLocalizationNode() : Node("object_localization_node"), tf_buffer_(this->get_clock())
    {
        detection_sub_ = this->create_subscription<isaac_ros_detectnet_interfaces::msg::DetectionArray>(
            "detections", 10,
            std::bind(&ObjectLocalizationNode::detectionCallback, this, std::placeholders::_1)
        );

        camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "camera_info", 10,
            std::bind(&ObjectLocalizationNode::cameraInfoCallback, this, std::placeholders::_1)
        );

        object_pose_pub_ = this->create_publisher<geometry_msgs::msg::PointStamped>(
            "object_3d_position", 10
        );

        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(tf_buffer_);
    }

private:
    void detectionCallback(const isaac_ros_detectnet_interfaces::msg::DetectionArray::SharedPtr detections)
    {
        if (!camera_info_received_ || !has_camera_to_robot_tf_) {
            RCLCPP_WARN(this->get_logger(), "Camera info or TF not available yet");
            return;
        }

        for (const auto& detection : detections->detections) {
            if (detection.confidence < confidence_threshold_) {
                continue;  // Skip low-confidence detections
            }

            // Convert 2D bounding box center to 3D point
            geometry_msgs::msg::PointStamped pixel_point;
            pixel_point.header = detections->header;
            pixel_point.point.x = detection.bbox.center.x;
            pixel_point.point.y = detection.bbox.center.y;
            pixel_point.point.z = 1.0;  // Placeholder depth

            // Convert pixel coordinates to 3D camera frame
            geometry_msgs::msg::PointStamped camera_point;
            camera_point = pixelToCameraFrame(pixel_point);

            // Transform to robot base frame
            geometry_msgs::msg::PointStamped robot_point;
            robot_point = transformToRobotFrame(camera_point);

            // Create and publish object position
            geometry_msgs::msg::PointStamped object_position;
            object_position.header = robot_point.header;
            object_position.point = robot_point.point;

            // Add object label as metadata (in a real system, you might publish this separately)
            RCLCPP_INFO(this->get_logger(),
                "Detected %s at position: (%.2f, %.2f, %.2f) with confidence %.2f",
                detection.label.c_str(),
                object_position.point.x,
                object_position.point.y,
                object_position.point.z,
                detection.confidence
            );

            object_pose_pub_->publish(object_position);
        }
    }

    geometry_msgs::msg::PointStamped pixelToCameraFrame(const geometry_msgs::msg::PointStamped& pixel_point)
    {
        geometry_msgs::msg::PointStamped camera_point;
        camera_point.header = pixel_point.header;  // Keep same timestamp/frame initially

        // Convert pixel coordinates to normalized coordinates
        double x_norm = (pixel_point.point.x - camera_info_.k[2]) / camera_info_.k[0];  // cx, fx
        double y_norm = (pixel_point.point.y - camera_info_.k[5]) / camera_info_.k[4];  // cy, fy

        // For this example, assume depth is known from other sources
        // In practice, you'd get depth from stereo, LIDAR, or depth sensor
        double depth = estimateDepth(pixel_point.point.x, pixel_point.point.y);

        camera_point.point.x = x_norm * depth;
        camera_point.point.y = y_norm * depth;
        camera_point.point.z = depth;

        return camera_point;
    }

    geometry_msgs::msg::PointStamped transformToRobotFrame(const geometry_msgs::msg::PointStamped& camera_point)
    {
        geometry_msgs::msg::PointStamped robot_point;

        try {
            // Transform from camera frame to robot base frame
            tf_buffer_.transform(camera_point, robot_point, "base_link");
        } catch (tf2::TransformException& ex) {
            RCLCPP_ERROR(this->get_logger(), "Transform failed: %s", ex.what());
            return camera_point;  // Return original if transform fails
        }

        return robot_point;
    }

    double estimateDepth(double u, double v)
    {
        // Placeholder depth estimation
        // In a real system, this would come from:
        // 1. Stereo vision
        // 2. Depth sensor (RGB-D camera, LIDAR)
        // 3. Monocular depth estimation
        // 4. Object size-based estimation (if object size is known)

        // For this example, return a fixed depth
        // A more realistic approach would use stereo disparity or other depth sources
        return 1.0;  // 1 meter depth as placeholder
    }

    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        camera_info_ = *msg;
        camera_info_received_ = true;
    }

    rclcpp::Subscription<isaac_ros_detectnet_interfaces::msg::DetectionArray>::SharedPtr detection_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr object_pose_pub_;

    tf2_ros::Buffer tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    sensor_msgs::msg::CameraInfo camera_info_;
    bool camera_info_received_ = false;
    bool has_camera_to_robot_tf_ = false;

    const double confidence_threshold_ = 0.7;
};
```

### 2. Semantic Segmentation for Object Localization

```cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class SemanticSegmentationNode : public rclcpp::Node
{
public:
    SemanticSegmentationNode() : Node("semantic_segmentation_node")
    {
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "image_input", 10,
            std::bind(&SemanticSegmentationNode::imageCallback, this, std::placeholders::_1)
        );

        segmentation_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "segmentation_output", 10
        );
    }

private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr image_msg)
    {
        // Convert ROS image to OpenCV
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        // Run semantic segmentation
        cv::Mat segmentation_mask = runSegmentationInference(cv_ptr->image);

        // Create result image with segmentation overlay
        cv::Mat result_image = createSegmentationOverlay(cv_ptr->image, segmentation_mask);

        // Publish segmentation result
        publishSegmentationResult(result_image, image_msg->header);
    }

    cv::Mat runSegmentationInference(const cv::Mat& image)
    {
        // Placeholder for actual segmentation inference
        // This would typically use a model like DeepLab, SegNet, or similar
        // For Isaac ROS, this might use Isaac ROS Segmentation packages

        cv::Mat segmentation_mask;

        // In a real implementation, this would:
        // 1. Preprocess image for the segmentation model
        // 2. Run inference using TensorRT
        // 3. Post-process to get class labels for each pixel
        // 4. Return a mask where each pixel value represents the class ID

        // For this example, return a dummy mask
        segmentation_mask = cv::Mat::zeros(image.size(), CV_8UC1);

        // Simulate detection of a few classes in specific regions
        cv::rectangle(segmentation_mask, cv::Rect(100, 100, 200, 150), cv::Scalar(1), -1); // Class 1
        cv::rectangle(segmentation_mask, cv::Rect(300, 200, 150, 100), cv::Scalar(2), -1); // Class 2

        return segmentation_mask;
    }

    cv::Mat createSegmentationOverlay(const cv::Mat& original_image, const cv::Mat& segmentation_mask)
    {
        cv::Mat overlay = original_image.clone();

        // Define colors for different classes
        std::vector<cv::Vec3b> class_colors = {
            cv::Vec3b(0, 0, 0),      // Class 0: background (black)
            cv::Vec3b(255, 0, 0),    // Class 1: red
            cv::Vec3b(0, 255, 0),    // Class 2: green
            cv::Vec3b(0, 0, 255),    // Class 3: blue
            cv::Vec3b(255, 255, 0),  // Class 4: cyan
            cv::Vec3b(255, 0, 255),  // Class 5: magenta
        };

        // Create overlay with transparency
        for (int y = 0; y < segmentation_mask.rows; y++) {
            for (int x = 0; x < segmentation_mask.cols; x++) {
                int class_id = segmentation_mask.at<uchar>(y, x);
                if (class_id > 0 && class_id < class_colors.size()) {
                    // Blend original color with class color
                    cv::Vec3b& pixel = overlay.at<cv::Vec3b>(y, x);
                    cv::Vec3b class_color = class_colors[class_id];

                    // Simple blending (50% original, 50% class color)
                    pixel = 0.5 * pixel + 0.5 * class_color;
                }
            }
        }

        return overlay;
    }

    void publishSegmentationResult(const cv::Mat& result_image, const std_msgs::msg::Header& header)
    {
        cv_bridge::CvImage cv_image;
        cv_image.header = header;
        cv_image.encoding = sensor_msgs::image_encodings::BGR8;
        cv_image.image = result_image;

        segmentation_pub_->publish(*cv_image.toImageMsg());
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr segmentation_pub_;
};
```

## Isaac Sim Perception Integration

### Isaac Sim Perception Configuration

```python
# Isaac Sim perception setup
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.sensor import Camera
from omni.isaac.range_sensor import RotatingLidarPhysX
import numpy as np

class IsaacSimPerception:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.setup_perception_sensors()

    def setup_perception_sensors(self):
        # Add a robot to the scene
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            print("Could not find Isaac Sim assets")
            return

        # Add a simple robot with sensors
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

        # Initialize the world
        self.world.reset()

    def get_sensor_data(self):
        # Get camera data
        rgb_data = self.camera.get_rgb()
        depth_data = self.camera.get_depth()
        seg_data = self.camera.get_semantic_segmentation()

        # Get LIDAR data
        lidar_data = self.lidar.get_linear_depth_data()

        return {
            'rgb': rgb_data,
            'depth': depth_data,
            'segmentation': seg_data,
            'lidar': lidar_data
        }

    def run_perception_pipeline(self):
        # Main perception loop
        while not self.world.is_stopped():
            self.world.step(render=True)

            # Get sensor data
            sensor_data = self.get_sensor_data()

            # Process perception data
            objects = self.detect_objects(sensor_data)

            # Localize objects in world coordinates
            object_poses = self.localize_objects(objects, sensor_data)

            # Print results
            self.print_perception_results(object_poses)

    def detect_objects(self, sensor_data):
        # Placeholder for object detection
        # In Isaac Sim, this would interface with Isaac ROS perception packages
        # or use built-in synthetic data generation

        # For this example, return simulated detections
        objects = [
            {'class': 'box', 'confidence': 0.95, 'bbox': [100, 100, 200, 150]},
            {'class': 'cylinder', 'confidence': 0.89, 'bbox': [300, 200, 150, 100]}
        ]

        return objects

    def localize_objects(self, objects, sensor_data):
        # Convert 2D detections to 3D world coordinates
        # This would use depth information and camera parameters

        object_poses = []

        for obj in objects:
            # Convert 2D bbox center to 3D using depth
            center_x = (obj['bbox'][0] + obj['bbox'][2]) // 2
            center_y = (obj['bbox'][1] + obj['bbox'][3]) // 2

            # Get depth at center point
            depth = sensor_data['depth'][center_y, center_x]

            if depth < 10.0:  # Valid depth check
                # Convert pixel coordinates to world coordinates
                # This requires camera intrinsic parameters
                world_pos = self.pixel_to_world(
                    center_x, center_y, depth,
                    self.camera.prim.GetAttribute("xformOp:transform").Get()
                )

                object_poses.append({
                    'class': obj['class'],
                    'position': world_pos,
                    'confidence': obj['confidence']
                })

        return object_poses

    def pixel_to_world(self, u, v, depth, camera_transform):
        # Convert pixel coordinates to world coordinates
        # This is a simplified version - in practice, you'd use camera intrinsics

        # Camera intrinsic parameters (these would come from camera config)
        fx = 616.363  # Focal length x
        fy = 616.363  # Focal length y
        cx = 313.071  # Principal point x
        cy = 245.091  # Principal point y

        # Convert to camera coordinates
        x_cam = (u - cx) * depth / fx
        y_cam = (v - cy) * depth / fy
        z_cam = depth

        # Transform to world coordinates using camera pose
        # (simplified - would need proper transformation matrix)
        x_world = x_cam  # Simplified
        y_world = y_cam
        z_world = z_cam

        return [x_world, y_world, z_world]

    def print_perception_results(self, object_poses):
        print("Perception Results:")
        for obj in object_poses:
            print(f"  {obj['class']}: ({obj['position'][0]:.2f}, {obj['position'][1]:.2f}, {obj['position'][2]:.2f}), conf: {obj['confidence']:.2f}")
```

## Practical Examples

### Example 1: Person Detection and Localization

```cpp
// Complete example for detecting and localizing people
class PersonDetectionNode : public rclcpp::Node
{
public:
    PersonDetectionNode() : Node("person_detection_node")
    {
        // Subscribe to camera image
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "camera/image_raw", 10,
            std::bind(&PersonDetectionNode::imageCallback, this, std::placeholders::_1)
        );

        // Subscribe to camera info
        camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "camera/camera_info", 10,
            std::bind(&PersonDetectionNode::cameraInfoCallback, this, std::placeholders::_1)
        );

        // Publisher for person positions
        person_pub_ = this->create_publisher<geometry_msgs::msg::PointStamped>(
            "person_position", 10
        );

        // Publisher for visualization
        viz_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "person_detection_viz", 10
        );
    }

private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr image_msg)
    {
        if (!camera_info_received_) return;

        // Convert to OpenCV
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        // Run person detection
        std::vector<PersonDetection> persons = detectPersons(cv_ptr->image);

        // Process each detected person
        for (const auto& person : persons) {
            if (person.confidence > 0.8) {  // Confidence threshold
                // Localize person in 3D space
                geometry_msgs::msg::PointStamped person_3d = localizePerson(
                    person, image_msg->header
                );

                // Publish person position
                person_pub_->publish(person_3d);

                // Add to visualization
                cv::rectangle(cv_ptr->image, person.bbox, cv::Scalar(0, 255, 0), 2);
                std::string label = "Person: " + std::to_string(person.confidence);
                cv::putText(cv_ptr->image, label,
                           cv::Point(person.bbox.x, person.bbox.y - 10),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            }
        }

        // Publish visualization image
        viz_pub_->publish(*cv_ptr->toImageMsg());
    }

    std::vector<PersonDetection> detectPersons(const cv::Mat& image)
    {
        std::vector<PersonDetection> detections;

        // In a real implementation, this would run a DNN model
        // such as YOLO, SSD MobileNet, or Isaac ROS DetectNet
        // For this example, we'll use OpenCV's HOG descriptor

        cv::HOGDescriptor hog;
        hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

        std::vector<cv::Rect> found_locations;
        std::vector<double> found_weights;

        hog.detectMultiScale(image, found_locations, found_weights, 0, cv::Size(8,8), cv::Size(32,32), 1.05, 2, false);

        for (size_t i = 0; i < found_locations.size(); ++i) {
            PersonDetection detection;
            detection.bbox = found_locations[i];
            detection.confidence = found_weights[i];
            detection.center_x = detection.bbox.x + detection.bbox.width / 2.0;
            detection.center_y = detection.bbox.y + detection.bbox.height / 2.0;
            detections.push_back(detection);
        }

        return detections;
    }

    geometry_msgs::msg::PointStamped localizePerson(
        const PersonDetection& person,
        const std_msgs::msg::Header& header)
    {
        geometry_msgs::msg::PointStamped person_3d;
        person_3d.header = header;

        // Estimate depth using simple heuristics
        // In practice, you'd use stereo vision or depth sensor
        double depth = estimatePersonDepth(person.bbox.height);

        // Convert pixel to 3D coordinates
        double x_norm = (person.center_x - camera_info_.k[2]) / camera_info_.k[0];
        double y_norm = (person.center_y - camera_info_.k[5]) / camera_info_.k[4];

        person_3d.point.x = x_norm * depth;
        person_3d.point.y = y_norm * depth;
        person_3d.point.z = depth;

        return person_3d;
    }

    double estimatePersonDepth(int bbox_height)
    {
        // Simple depth estimation based on bounding box height
        // Assumes average person height is ~1.7m
        // height_in_pixels = (focal_length * real_height) / depth
        // So depth = (focal_length * real_height) / height_in_pixels

        double focal_length = camera_info_.k[0];  // fx
        double real_person_height = 1.7;  // meters
        double pixel_height = static_cast<double>(bbox_height);

        return (focal_length * real_person_height) / pixel_height;
    }

    struct PersonDetection {
        cv::Rect bbox;
        double confidence;
        double center_x, center_y;
    };

    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        camera_info_ = *msg;
        camera_info_received_ = true;
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr person_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr viz_pub_;

    sensor_msgs::msg::CameraInfo camera_info_;
    bool camera_info_received_ = false;
};
```

### Example 2: Object Detection with Isaac ROS and Isaac Sim

```python
# Isaac Sim + Isaac ROS integration example
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from isaac_ros_detectnet_interfaces.msg import DetectionArray
from geometry_msgs.msg import PointStamped
import cv2
import numpy as np
from cv_bridge import CvBridge

class IsaacPerceptionPipeline(Node):
    def __init__(self):
        super().__init__('isaac_perception_pipeline')

        # ROS 2 interface
        self.bridge = CvBridge()

        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_info', self.camera_info_callback, 10
        )
        self.detection_sub = self.create_subscription(
            DetectionArray, '/detectnet/detections', self.detection_callback, 10
        )
        self.object_pub = self.create_publisher(
            PointStamped, '/detected_object_position', 10
        )
        self.viz_pub = self.create_publisher(
            Image, '/perception_visualization', 10
        )

        # Storage
        self.camera_info = None
        self.latest_image = None

    def image_callback(self, msg):
        self.latest_image = msg

    def camera_info_callback(self, msg):
        self.camera_info = msg

    def detection_callback(self, msg):
        if self.latest_image is None or self.camera_info is None:
            return

        # Convert ROS image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")

        # Process detections
        for detection in msg.detections:
            if detection.confidence > 0.7:  # Confidence threshold
                # Localize object in 3D
                object_3d = self.localize_object_3d(
                    detection.bbox.center.x,
                    detection.bbox.center.y,
                    detection
                )

                # Publish 3D position
                self.object_pub.publish(object_3d)

                # Draw bounding box on image
                pt1 = (int(detection.bbox.center.x - detection.bbox.size_x/2),
                       int(detection.bbox.center.y - detection.bbox.size_y/2))
                pt2 = (int(detection.bbox.center.x + detection.bbox.size_x/2),
                       int(detection.bbox.center.y + detection.bbox.size_y/2))
                cv2.rectangle(cv_image, pt1, pt2, (0, 255, 0), 2)
                cv2.putText(cv_image, f"{detection.label}: {detection.confidence:.2f}",
                           (pt1[0], pt1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Publish visualization
        viz_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        viz_msg.header = self.latest_image.header
        self.viz_pub.publish(viz_msg)

    def localize_object_3d(self, x_2d, y_2d, detection):
        # This is a simplified example
        # In practice, you'd use depth information from stereo or depth sensor
        point_3d = PointStamped()
        point_3d.header = detection.header

        # Estimate depth based on object size or use depth map
        # For this example, assume a fixed depth of 2 meters
        estimated_depth = 2.0  # meters

        # Convert 2D pixel coordinates to 3D using camera intrinsics
        if self.camera_info:
            fx = self.camera_info.k[0]  # Focal length x
            fy = self.camera_info.k[4]  # Focal length y
            cx = self.camera_info.k[2]  # Principal point x
            cy = self.camera_info.k[5]  # Principal point y

            point_3d.point.x = (x_2d - cx) * estimated_depth / fx
            point_3d.point.y = (y_2d - cy) * estimated_depth / fy
            point_3d.point.z = estimated_depth

        return point_3d

def main(args=None):
    rclpy.init(args=args)
    perception_node = IsaacPerceptionPipeline()

    try:
        rclpy.spin(perception_node)
    except KeyboardInterrupt:
        pass
    finally:
        perception_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Performance Optimization

### 1. TensorRT Optimization for Deep Learning Models

```cpp
// Example of TensorRT optimization for perception
#include <NvInfer.h>
#include <cuda_runtime_api.h>

class OptimizedPerceptionNode : public rclcpp::Node
{
public:
    OptimizedPerceptionNode() : Node("optimized_perception_node")
    {
        // Initialize TensorRT engine
        initializeTensorRTEngine();
    }

private:
    void initializeTensorRTEngine()
    {
        // This would load a pre-built TensorRT engine
        // for optimized inference of perception models
        // The engine would be built offline from ONNX models
    }

    std::vector<Detection> runOptimizedInference(const cv::Mat& image)
    {
        // Run inference using TensorRT for maximum performance
        // This would include:
        // 1. Memory management for GPU
        // 2. Batch processing
        // 3. Asynchronous execution
        // 4. Proper input/output binding

        std::vector<Detection> detections;
        // Implementation would go here
        return detections;
    }

    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    cudaStream_t stream_;
    void* buffers_[2];  // Input and output buffers
};
```

### 2. Multi-Threaded Perception Pipeline

```cpp
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

class MultiThreadedPerceptionNode : public rclcpp::Node
{
public:
    MultiThreadedPerceptionNode() : Node("multithreaded_perception_node")
    {
        // Create threads for different perception tasks
        detection_thread_ = std::thread(&MultiThreadedPerceptionNode::detectionLoop, this);
        localization_thread_ = std::thread(&MultiThreadedPerceptionNode::localizationLoop, this);

        // Subscribe to image
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "image_input", 10,
            std::bind(&MultiThreadedPerceptionNode::imageCallback, this, std::placeholders::_1)
        );
    }

    ~MultiThreadedPerceptionNode()
    {
        running_ = false;
        if (detection_thread_.joinable()) detection_thread_.join();
        if (localization_thread_.joinable()) localization_thread_.join();
    }

private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr image_msg)
    {
        std::lock_guard<std::mutex> lock(image_queue_mutex_);
        image_queue_.push(image_msg);

        if (image_queue_.size() > max_queue_size_) {
            image_queue_.pop();  // Drop oldest if queue is full
        }

        image_queue_cond_.notify_one();
    }

    void detectionLoop()
    {
        while (running_) {
            sensor_msgs::msg::Image::SharedPtr image_msg;

            {
                std::unique_lock<std::mutex> lock(image_queue_mutex_);
                image_queue_cond_.wait(lock, [this] { return !image_queue_.empty() || !running_; });

                if (!running_) break;

                image_msg = image_queue_.front();
                image_queue_.pop();
            }

            // Run object detection
            auto detections = runDetection(image_msg);

            // Add to detection queue
            {
                std::lock_guard<std::mutex> lock(detection_queue_mutex_);
                detection_queue_.push(std::make_pair(image_msg->header, detections));
            }

            detection_queue_cond_.notify_one();
        }
    }

    void localizationLoop()
    {
        while (running_) {
            std_msgs::msg::Header header;
            std::vector<Detection> detections;

            {
                std::unique_lock<std::mutex> lock(detection_queue_mutex_);
                detection_queue_cond_.wait(lock, [this] { return !detection_queue_.empty() || !running_; });

                if (!running_) break;

                auto detection_pair = detection_queue_.front();
                header = detection_pair.first;
                detections = detection_pair.second;
                detection_queue_.pop();
            }

            // Run localization
            auto object_positions = runLocalization(detections, header);

            // Publish results
            publishResults(object_positions);
        }
    }

    std::vector<Detection> runDetection(const sensor_msgs::msg::Image::SharedPtr image_msg)
    {
        // Run object detection on the image
        // Implementation would go here
        return std::vector<Detection>();
    }

    std::vector<ObjectPosition> runLocalization(
        const std::vector<Detection>& detections,
        const std_msgs::msg::Header& header)
    {
        // Localize objects in 3D space
        // Implementation would go here
        return std::vector<ObjectPosition>();
    }

    void publishResults(const std::vector<ObjectPosition>& positions)
    {
        // Publish localization results
        // Implementation would go here
    }

    struct Detection {
        std::string label;
        float confidence;
        cv::Rect bbox;
    };

    struct ObjectPosition {
        std::string label;
        geometry_msgs::msg::Point position;
        float confidence;
    };

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;

    // Image processing queue
    std::queue<sensor_msgs::msg::Image::SharedPtr> image_queue_;
    std::mutex image_queue_mutex_;
    std::condition_variable image_queue_cond_;

    // Detection queue
    std::queue<std::pair<std_msgs::msg::Header, std::vector<Detection>>> detection_queue_;
    std::mutex detection_queue_mutex_;
    std::condition_variable detection_queue_cond_;

    std::thread detection_thread_;
    std::thread localization_thread_;
    std::atomic<bool> running_{true};
    const size_t max_queue_size_ = 5;
};
```

## Best Practices

### 1. Confidence Thresholding
Always use confidence thresholds to filter out low-quality detections:
- Set appropriate thresholds based on your application requirements
- Consider using adaptive thresholds based on scene complexity
- Validate detections with geometric consistency checks

### 2. Multi-Sensor Fusion
Combine data from multiple sensors for robust perception:
- Fuse camera, LIDAR, and radar data
- Use Kalman filters or particle filters for tracking
- Implement sensor validation and fault detection

### 3. Performance Monitoring
Monitor perception performance in real-time:
- Track inference time and frame rates
- Monitor memory and GPU usage
- Log detection accuracy and false positive rates

## Exercise

Create a complete perception pipeline that includes:

1. Object detection using Isaac ROS DetectNet
2. 3D localization using stereo vision or depth information
3. Multi-threaded processing for real-time performance
4. Integration with Isaac Sim for testing
5. Visualization of detection results
6. Performance evaluation metrics

Test your pipeline with various objects in different lighting conditions and evaluate its accuracy and performance.