# Multimodal Perception

Multimodal perception refers to the integration of multiple sensory modalities (vision, language, and action) to create a comprehensive understanding of the environment and enable intelligent robot behavior. This section covers the integration of visual, linguistic, and motor information for humanoid robotics.

## Introduction to Multimodal Perception

Multimodal perception is crucial for humanoid robots to interact naturally with humans and environments. It involves:

- **Visual Processing**: Understanding the visual environment
- **Language Understanding**: Interpreting natural language commands
- **Action Generation**: Executing appropriate physical responses
- **Cross-modal Integration**: Combining information from different modalities

### Vision-Language-Action (VLA) Models

VLA models represent the next generation of AI systems that can process visual input, understand language commands, and generate appropriate actions:

```
Vision Input + Language Command → VLA Model → Action Sequence
```

## NVIDIA Isaac Sim for Multimodal Training

### Synthetic Data Generation

Isaac Sim excels at generating synthetic data for multimodal perception:

```python
# Example: Generating synthetic multimodal training data
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.synthetic_utils import SyntheticDataHelper
import numpy as np
import json

class MultimodalTrainingDataGenerator:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.setup_scene()
        self.synthetic_data_helper = SyntheticDataHelper()

    def setup_scene(self):
        """Set up scene with various objects for multimodal training"""
        # Add humanoid robot
        add_reference_to_stage(
            usd_path="path/to/humanoid/robot.usd",
            prim_path="/World/HumanoidRobot"
        )

        # Add various objects with semantic labels
        self.add_object_with_label("RedCube", "/World/RedCube", [0, 2, 0.5], "cube_red")
        self.add_object_with_label("BlueSphere", "/World/BlueSphere", [1, 1, 0.5], "sphere_blue")
        self.add_object_with_label("GreenCylinder", "/World/GreenCylinder", [-1, 1, 0.5], "cylinder_green")

    def add_object_with_label(self, name: str, prim_path: str, position: list, semantic_label: str):
        """Add object with semantic labeling for training data"""
        # Implementation would add object and set semantic labeling
        pass

    def generate_multimodal_data(self, num_samples: int = 1000):
        """Generate synthetic multimodal training data"""
        training_data = []

        for i in range(num_samples):
            # Change scene configuration
            self.randomize_scene()

            # Capture sensor data
            rgb_data = self.get_camera_data()
            depth_data = self.get_depth_data()
            semantic_data = self.get_semantic_segmentation()

            # Generate language annotations
            scene_description = self.generate_scene_description(rgb_data, semantic_data)
            action_sequences = self.generate_possible_actions(scene_description)

            # Create training sample
            sample = {
                "sample_id": f"synthetic_{i:04d}",
                "modalities": {
                    "vision": {
                        "rgb": self.encode_image(rgb_data),
                        "depth": self.encode_depth(depth_data),
                        "semantic": self.encode_semantic(semantic_data)
                    },
                    "language": {
                        "scene_description": scene_description,
                        "possible_commands": [
                            f"Go to the {obj}",
                            f"Pick up the {obj}",
                            f"Move the {obj} to the table"
                            for obj in self.get_visible_objects(semantic_data)
                        ]
                    },
                    "actions": action_sequences
                }
            }

            training_data.append(sample)

        return training_data

    def randomize_scene(self):
        """Randomize object positions and lighting for variety"""
        # Implementation would randomize object positions, lighting, etc.
        pass

    def encode_image(self, image_data):
        """Encode image for storage"""
        # Implementation would compress and encode image
        return "encoded_rgb_data"

    def encode_depth(self, depth_data):
        """Encode depth data for storage"""
        # Implementation would process depth data
        return "encoded_depth_data"

    def encode_semantic(self, semantic_data):
        """Encode semantic segmentation for storage"""
        # Implementation would process semantic data
        return "encoded_semantic_data"

    def generate_scene_description(self, rgb_data, semantic_data):
        """Generate natural language description of scene"""
        # Implementation would analyze scene and generate description
        visible_objects = self.get_visible_objects(semantic_data)
        return f"The scene contains {', '.join(visible_objects)}. The robot is positioned centrally."

    def generate_possible_actions(self, scene_description):
        """Generate possible actions based on scene"""
        # Implementation would generate possible robot actions
        return [
            {"action": "navigate", "target": "object_location", "description": "Move to object"},
            {"action": "grasp", "target": "graspable_object", "description": "Grasp the object"},
            {"action": "manipulate", "target": "manipulable_object", "description": "Manipulate the object"}
        ]

    def get_visible_objects(self, semantic_data):
        """Extract visible objects from semantic segmentation"""
        # Implementation would analyze semantic data
        return ["red_cube", "blue_sphere", "green_cylinder"]
```

## Isaac ROS Perception Integration

### Isaac ROS Perception Pipeline

```cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <isaac_ros_apriltag_interfaces/msg/april_tag_detection_array.hpp>
#include <isaac_ros_detectnet_interfaces/msg/detection_array.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class MultimodalPerceptionNode : public rclcpp::Node
{
public:
    MultimodalPerceptionNode() : Node("multimodal_perception_node")
    {
        // Create subscriptions for different sensor modalities
        rgb_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "camera/rgb/image_rect_color", 10,
            std::bind(&MultimodalPerceptionNode::rgbCallback, this, std::placeholders::_1)
        );

        depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "camera/depth/image_rect_raw", 10,
            std::bind(&MultimodalPerceptionNode::depthCallback, this, std::placeholders::_1)
        );

        detection_sub_ = this->create_subscription<isaac_ros_detectnet_interfaces::msg::DetectionArray>(
            "detectnet/detections", 10,
            std::bind(&MultimodalPerceptionNode::detectionCallback, this, std::placeholders::_1)
        );

        apriltag_sub_ = this->create_subscription<isaac_ros_apriltag_interfaces::msg::AprilTagDetectionArray>(
            "apriltag_detections", 10,
            std::bind(&MultimodalPerceptionNode::apriltagCallback, this, std::placeholders::_1)
        );

        // Publisher for multimodal fusion results
        multimodal_pub_ = this->create_publisher<geometry_msgs::msg::PointStamped>(
            "multimodal_fusion_result", 10
        );
    }

private:
    void rgbCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // Process RGB image for visual understanding
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        // Store RGB data for multimodal fusion
        latest_rgb_image_ = cv_ptr->image;
        rgb_timestamp_ = msg->header.stamp;

        // Trigger multimodal fusion if all modalities are available
        if (has_depth_ && has_detections_ && has_apriltags_) {
            performMultimodalFusion();
        }
    }

    void depthCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // Process depth image for 3D understanding
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        latest_depth_image_ = cv_ptr->image;
        depth_timestamp_ = msg->header.stamp;
        has_depth_ = true;

        // Trigger multimodal fusion if all modalities are available
        if (has_rgb_ && has_detections_ && has_apriltags_) {
            performMultimodalFusion();
        }
    }

    void detectionCallback(const isaac_ros_detectnet_interfaces::msg::DetectionArray::SharedPtr msg)
    {
        latest_detections_ = *msg;
        has_detections_ = true;

        // Trigger multimodal fusion if all modalities are available
        if (has_rgb_ && has_depth_ && has_apriltags_) {
            performMultimodalFusion();
        }
    }

    void apriltagCallback(const isaac_ros_apriltag_interfaces::msg::AprilTagDetectionArray::SharedPtr msg)
    {
        latest_apriltags_ = *msg;
        has_apriltags_ = true;

        // Trigger multimodal fusion if all modalities are available
        if (has_rgb_ && has_depth_ && has_detections_) {
            performMultimodalFusion();
        }
    }

    void performMultimodalFusion()
    {
        // Combine information from all modalities
        auto fused_result = fuseModalities(
            latest_rgb_image_,
            latest_depth_image_,
            latest_detections_,
            latest_apriltags_
        );

        // Publish fused result
        auto result_msg = geometry_msgs::msg::PointStamped();
        result_msg.header.stamp = this->now();
        result_msg.header.frame_id = "multimodal_fused";
        result_msg.point = fused_result.centroid;

        multimodal_pub_->publish(result_msg);

        RCLCPP_INFO(
            this->get_logger(),
            "Multimodal fusion completed: %zu objects detected, %zu fiducials localized",
            latest_detections_.detections.size(),
            latest_apriltags_.detections.size()
        );

        // Reset flags for next cycle
        has_depth_ = false;
        has_detections_ = false;
        has_apriltags_ = false;
    }

    struct FusedPerceptionResult {
        geometry_msgs::msg::Point centroid;
        std::vector<std::string> object_labels;
        std::vector<double> confidences;
        std::vector<geometry_msgs::msg::Point> object_positions;
    };

    FusedPerceptionResult fuseModalities(
        const cv::Mat& rgb_image,
        const cv::Mat& depth_image,
        const isaac_ros_detectnet_interfaces::msg::DetectionArray& detections,
        const isaac_ros_apriltag_interfaces::msg::AprilTagDetectionArray& apriltags)
    {
        FusedPerceptionResult result;

        // Combine visual detections with depth information for 3D localization
        for (const auto& detection : detections.detections) {
            // Get 2D bounding box center
            int center_x = static_cast<int>(detection.bbox.center.x);
            int center_y = static_cast<int>(detection.bbox.center.y);

            // Get depth at this location (average in a small region)
            double avg_depth = getAverageDepthAtPixel(depth_image, center_x, center_y);

            // Convert 2D pixel coordinates to 3D world coordinates
            geometry_msgs::msg::Point world_point = pixelTo3D(
                center_x, center_y, avg_depth, camera_intrinsics_
            );

            result.object_labels.push_back(detection.class_name);
            result.confidences.push_back(detection.confidence);
            result.object_positions.push_back(world_point);
        }

        // Add AprilTag positions (they provide accurate 3D poses)
        for (const auto& apriltag : apriltags.detections) {
            result.object_labels.push_back("fiducial_" + std::to_string(apriltag.id));
            result.confidences.push_back(1.0);  // High confidence for fiducials
            result.object_positions.push_back(apriltag.pose.position);
        }

        // Calculate overall centroid of all detected objects
        if (!result.object_positions.empty()) {
            double sum_x = 0, sum_y = 0, sum_z = 0;
            for (const auto& pos : result.object_positions) {
                sum_x += pos.x;
                sum_y += pos.y;
                sum_z += pos.z;
            }

            result.centroid.x = sum_x / result.object_positions.size();
            result.centroid.y = sum_y / result.object_positions.size();
            result.centroid.z = sum_z / result.object_positions.size();
        }

        return result;
    }

    double getAverageDepthAtPixel(const cv::Mat& depth_image, int x, int y, int radius = 3)
    {
        double sum = 0;
        int count = 0;

        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                int nx = x + dx;
                int ny = y + dy;

                if (nx >= 0 && nx < depth_image.cols && ny >= 0 && ny < depth_image.rows) {
                    float depth_value = depth_image.at<float>(ny, nx);
                    if (depth_value > 0 && depth_value < 10.0) {  // Valid depth range
                        sum += depth_value;
                        count++;
                    }
                }
            }
        }

        return count > 0 ? sum / count : 0.0;
    }

    geometry_msgs::msg::Point pixelTo3D(int x, int y, double depth, const CameraIntrinsics& intrinsics)
    {
        geometry_msgs::msg::Point point;

        // Convert pixel coordinates to camera coordinates
        point.x = (x - intrinsics.cx) * depth / intrinsics.fx;
        point.y = (y - intrinsics.cy) * depth / intrinsics.fy;
        point.z = depth;

        return point;
    }

    // Subscriptions
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
    rclcpp::Subscription<isaac_ros_detectnet_interfaces::msg::DetectionArray>::SharedPtr detection_sub_;
    rclcpp::Subscription<isaac_ros_apriltag_interfaces::msg::AprilTagDetectionArray>::SharedPtr apriltag_sub_;

    // Publisher
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr multimodal_pub_;

    // Data storage
    cv::Mat latest_rgb_image_;
    cv::Mat latest_depth_image_;
    isaac_ros_detectnet_interfaces::msg::DetectionArray latest_detections_;
    isaac_ros_apriltag_interfaces::msg::AprilTagDetectionArray latest_apriltags_;

    // Flags for synchronization
    bool has_depth_ = false;
    bool has_detections_ = false;
    bool has_apriltags_ = false;

    // Timestamps for synchronization
    builtin_interfaces::msg::Time rgb_timestamp_;
    builtin_interfaces::msg::Time depth_timestamp_;

    // Camera intrinsics (would be loaded from camera info)
    struct CameraIntrinsics {
        double fx, fy, cx, cy;
    } camera_intrinsics_ = {616.363, 616.363, 313.071, 245.091};  // Example values
};
```

## Vision-Language Integration

### CLIP for Vision-Language Understanding

```python
import clip
import torch
import cv2
import numpy as np
from PIL import Image
import rospy
from sensor_msgs.msg import Image as ImageMsg
from std_msgs.msg import String
from cv_bridge import CvBridge

class VisionLanguagePerception:
    def __init__(self):
        # Load pre-trained CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        # Initialize ROS components
        self.bridge = CvBridge()

        # Subscribe to camera feed
        self.image_sub = rospy.Subscriber(
            "/camera/rgb/image_raw",
            ImageMsg,
            self.image_callback
        )

        # Publisher for recognized concepts
        self.concept_pub = rospy.Publisher(
            "/vision_language/concepts",
            String,
            queue_size=10
        )

        # Define concepts for classification
        self.concepts = [
            "a photo of a humanoid robot",
            "a photo of a person",
            "a photo of a table",
            "a photo of a chair",
            "a photo of a cup",
            "a photo of a book",
            "a photo of a door",
            "a photo of a window",
            "a scene with furniture",
            "a scene with electronics",
            "a kitchen scene",
            "a living room scene"
        ]

    def image_callback(self, msg):
        """Process incoming image and extract vision-language concepts"""
        try:
            # Convert ROS image to PIL Image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

            # Preprocess image
            image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)

            # Tokenize text descriptions
            text_input = clip.tokenize(self.concepts).to(self.device)

            # Get similarity scores
            with torch.no_grad():
                logits_per_image, logits_per_text = self.model(image_input, text_input)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

            # Get top predictions
            top_indices = np.argsort(probs)[-3:][::-1]  # Top 3 predictions
            top_concepts = [(self.concepts[i], probs[i]) for i in top_indices if probs[i] > 0.1]

            if top_concepts:
                # Publish recognized concepts
                concept_msg = String()
                concept_msg.data = json.dumps({
                    "timestamp": rospy.Time.now().to_sec(),
                    "concepts": [{"label": label.split()[-1], "confidence": float(conf)} for label, conf in top_concepts]
                })
                self.concept_pub.publish(concept_msg)

                rospy.loginfo(f"Recognized concepts: {top_concepts}")

        except Exception as e:
            rospy.logerr(f"Error in vision-language processing: {str(e)}")

    def extract_object_attributes(self, image_path: str, object_name: str):
        """Extract attributes of specific objects using vision-language model"""
        # This would implement more detailed attribute extraction
        # for specific objects mentioned in language commands
        pass
```

## Isaac Sim Perception Pipeline

### Complete Perception System Integration

```python
# Isaac Sim Perception Pipeline
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.range_sensor import LidarRtx
from omni.isaac.sensor import Camera
import numpy as np

class IsaacPerceptionPipeline:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.setup_perception_system()

    def setup_perception_system(self):
        """Set up complete perception system with multiple sensors"""
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            print("Could not find Isaac Sim assets. Please check your Isaac Sim installation.")
            return

        # Add a humanoid robot with sensors
        robot_path = assets_root_path + "/Isaac/Robots/Humanoid/humanoid_instanceable.usd"
        add_reference_to_stage(usd_path=robot_path, prim_path="/World/HumanoidRobot")

        # Add RGB-D camera
        self.camera = Camera(
            prim_path="/World/HumanoidRobot/base_link/camera",
            frequency=30,
            resolution=(640, 480)
        )

        # Add LIDAR for 3D perception
        self.lidar = LidarRtx(
            prim_path="/World/HumanoidRobot/base_link/lidar",
            translation=np.array([0.0, 0.0, 0.5]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0]),
            config="Carter",
            rotation_frequency=10,
            samples_per_scan=1080
        )

        # Add IMU for motion sensing
        self.imu = IMU(
            prim_path="/World/HumanoidRobot/base_link/imu",
            translation=np.array([0.0, 0.0, 0.25])
        )

        # Initialize the world
        self.world.reset()

    def run_perception_cycle(self):
        """Run complete perception cycle with multimodal fusion"""
        while not self.world.is_stopped():
            self.world.step(render=True)

            # Get sensor data
            camera_data = self.camera.get_rgb()
            depth_data = self.camera.get_depth()
            lidar_data = self.lidar.get_linear_depth_data()
            imu_data = self.imu.get_measured()

            # Process perception data
            visual_features = self.extract_visual_features(camera_data)
            3d_features = self.extract_3d_features(lidar_data, depth_data)
            motion_features = self.extract_motion_features(imu_data)

            # Fuse multimodal features
            fused_perception = self.fuse_multimodal_features(
                visual_features,
                3d_features,
                motion_features
            )

            # Generate perception report
            self.report_perception(fused_perception)

    def extract_visual_features(self, rgb_image):
        """Extract visual features from RGB image"""
        # This would interface with Isaac ROS Visual Perception components
        # For now, return basic feature extraction
        features = {
            "edges": self.detect_edges(rgb_image),
            "corners": self.detect_corners(rgb_image),
            "colors": self.extract_color_histogram(rgb_image),
            "objects": self.detect_objects(rgb_image)  # Would use Isaac ROS DetectNet
        }
        return features

    def extract_3d_features(self, lidar_data, depth_data):
        """Extract 3D features from depth and LIDAR data"""
        features = {
            "surfaces": self.detect_planes(lidar_data),
            "obstacles": self.detect_obstacles(lidar_data),
            "free_space": self.compute_free_space(lidar_data),
            "objects_3d": self.extract_3d_objects(depth_data, lidar_data)
        }
        return features

    def extract_motion_features(self, imu_data):
        """Extract motion features from IMU data"""
        features = {
            "acceleration": imu_data.acceleration,
            "angular_velocity": imu_data.angular_velocity,
            "orientation": imu_data.orientation,
            "motion_state": self.classify_motion_state(imu_data)
        }
        return features

    def fuse_multimodal_features(self, visual, features_3d, motion):
        """Fuse features from different modalities"""
        fused_features = {
            "scene_understanding": self.understand_scene(visual, features_3d),
            "robot_state": self.estimate_robot_state(motion),
            "navigation_goals": self.identify_navigation_goals(features_3d),
            "interaction_targets": self.identify_interaction_targets(visual, features_3d)
        }
        return fused_features

    def understand_scene(self, visual_features, features_3d):
        """Create comprehensive scene understanding"""
        # Combine visual and 3D information for scene understanding
        scene_description = {
            "environment_type": self.classify_environment(visual_features, features_3d),
            "object_list": self.merge_object_detections(visual_features, features_3d),
            "spatial_relations": self.compute_spatial_relations(features_3d),
            "affordances": self.identify_affordances(features_3d)
        }
        return scene_description

    def identify_interaction_targets(self, visual_features, features_3d):
        """Identify objects suitable for interaction"""
        # Identify graspable, manipulable, or approachable objects
        targets = []
        for obj in features_3d["objects_3d"]:
            if self.is_interactable(obj, visual_features):
                targets.append({
                    "object_id": obj["id"],
                    "position": obj["position"],
                    "interaction_type": self.determine_interaction_type(obj),
                    "approach_pose": self.calculate_approach_pose(obj)
                })
        return targets

    def report_perception(self, fused_perception):
        """Report perception results (for debugging and monitoring)"""
        print(f"Scene: {fused_perception['scene_understanding']['environment_type']}")
        print(f"Objects detected: {len(fused_perception['scene_understanding']['object_list'])}")
        print(f"Navigation goals: {len(fused_perception['navigation_goals'])}")
        print(f"Interaction targets: {len(fused_perception['interaction_targets'])}")
```

## Multimodal Fusion Algorithms

### Late Fusion vs Early Fusion

```cpp
// Late Fusion Implementation
class LateFusionPerceptor {
public:
    LateFusionPerceptor() {
        // Initialize individual modality processors
        visual_processor_ = std::make_unique<VisualProcessor>();
        language_processor_ = std::make_unique<LanguageProcessor>();
        action_processor_ = std::make_unique<ActionProcessor>();
    }

    MultimodalResult process(const SensorInputs& inputs, const LanguageInput& language) {
        // Process each modality separately
        auto visual_result = visual_processor_->process(inputs.rgb, inputs.depth);
        auto language_result = language_processor_->process(language.text);

        // Fuse at decision level
        return fuseDecisionLevel(visual_result, language_result, inputs.action_space);
    }

private:
    std::unique_ptr<VisualProcessor> visual_processor_;
    std::unique_ptr<LanguageProcessor> language_processor_;
    std::unique_ptr<ActionProcessor> action_processor_;

    MultimodalResult fuseDecisionLevel(
        const VisualResult& visual,
        const LanguageResult& language,
        const ActionSpace& action_space) {

        MultimodalResult result;

        // Combine confidence scores from different modalities
        for (const auto& object : visual.detected_objects) {
            if (language.intent.includes_object(object.label)) {
                // Boost confidence when vision and language agree
                result.confirmed_objects.push_back({
                    object,
                    object.confidence * language.confidence
                });
            }
        }

        // Generate action candidates based on fused understanding
        result.action_candidates = generateActionCandidates(
            result.confirmed_objects,
            language.intent,
            action_space
        );

        return result;
    }
};

// Early Fusion Implementation
class EarlyFusionPerceptor {
public:
    EarlyFusionPerceptor() {
        // Initialize joint embedding network
        embedding_network_ = std::make_unique<JointEmbeddingNetwork>();
    }

    MultimodalResult process(const SensorInputs& inputs, const LanguageInput& language) {
        // Convert all inputs to joint embedding space
        auto visual_embedding = createVisualEmbedding(inputs.rgb, inputs.depth);
        auto language_embedding = createLanguageEmbedding(language.text);

        // Concatenate embeddings
        auto joint_embedding = concatenate(visual_embedding, language_embedding);

        // Process through joint network
        return embedding_network_->infer(joint_embedding);
    }

private:
    std::unique_ptr<JointEmbeddingNetwork> embedding_network_;

    Embedding createVisualEmbedding(const cv::Mat& rgb, const cv::Mat& depth) {
        // Create embedding from visual inputs
        // This would typically use a CNN backbone
        return Embedding(); // Placeholder
    }

    Embedding createLanguageEmbedding(const std::string& text) {
        // Create embedding from text
        // This would typically use a transformer model
        return Embedding(); // Placeholder
    }
};
```

## Cross-Modal Attention Mechanisms

### Attention-Based Fusion

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    """Cross-modal attention for fusing vision and language features"""

    def __init__(self, feature_dim):
        super(CrossModalAttention, self).__init__()
        self.feature_dim = feature_dim

        # Linear projections for query, key, value
        self.vision_proj = nn.Linear(feature_dim, feature_dim)
        self.language_proj = nn.Linear(feature_dim, feature_dim)

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=0.1
        )

        # Output projection
        self.output_proj = nn.Linear(feature_dim * 2, feature_dim)

    def forward(self, vision_features, language_features):
        """
        Args:
            vision_features: [batch_size, seq_len_v, feature_dim]
            language_features: [batch_size, seq_len_l, feature_dim]
        Returns:
            fused_features: [batch_size, seq_len_v, feature_dim]
        """
        # Project features
        vision_q = self.vision_proj(vision_features)
        language_k = self.language_proj(language_features)
        language_v = language_features  # Use original as values

        # Apply cross-attention: vision attends to language
        attended_features, attention_weights = self.attention(
            vision_q.transpose(0, 1),  # Query from vision
            language_k.transpose(0, 1),  # Key from language
            language_v.transpose(0, 1)   # Value from language
        )

        # Transpose back
        attended_features = attended_features.transpose(0, 1)

        # Concatenate original vision features with attended features
        concatenated = torch.cat([vision_features, attended_features], dim=-1)

        # Project to output dimension
        output = self.output_proj(concatenated)

        return output, attention_weights

class MultimodalFusionNetwork(nn.Module):
    """Complete multimodal fusion network"""

    def __init__(self, feature_dim=512):
        super(MultimodalFusionNetwork, self).__init__()

        # Vision encoder (could be pre-trained CNN)
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, feature_dim)
        )

        # Language encoder (could be pre-trained transformer)
        self.language_encoder = nn.Sequential(
            nn.Embedding(10000, feature_dim),  # vocab_size=10000
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=feature_dim,
                    nhead=8,
                    dim_feedforward=2048
                ),
                num_layers=6
            ),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        # Cross-modal attention modules
        self.vision_language_attention = CrossModalAttention(feature_dim)
        self.language_vision_attention = CrossModalAttention(feature_dim)

        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, images, text_tokens):
        # Encode modalities
        vision_features = self.vision_encoder(images)
        language_features = self.language_encoder(text_tokens)

        # Reshape for attention (add sequence dimension if needed)
        if len(vision_features.shape) == 2:
            vision_features = vision_features.unsqueeze(1)  # [batch, 1, feature_dim]
        if len(language_features.shape) == 2:
            language_features = language_features.unsqueeze(1)  # [batch, 1, feature_dim]

        # Apply cross-modal attention
        vl_attended, vl_attention = self.vision_language_attention(
            vision_features, language_features
        )
        lv_attended, lv_attention = self.language_vision_attention(
            language_features, vision_features
        )

        # Combine attended features
        fused_features = torch.cat([
            vl_attended.mean(dim=1),  # Average over sequence
            lv_attended.mean(dim=1)
        ], dim=-1)

        # Final fusion
        output = self.fusion_layer(fused_features)

        return output, (vl_attention, lv_attention)
```

## Practical Implementation

### Isaac ROS Perception Integration

```cpp
// perception_integration_node.cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <isaac_ros_visual_slam/visual_slam_node.hpp>
#include <isaac_ros_detectnet/detectnet_node.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

class PerceptionIntegrationNode : public rclcpp::Node
{
public:
    PerceptionIntegrationNode() : Node("perception_integration_node"), tf_buffer_(this->get_clock())
    {
        // Create synchronized subscribers for multimodal data
        rgb_sub_.subscribe(this, "camera/rgb/image_rect_color", rmw_qos_profile_sensor_data);
        depth_sub_.subscribe(this, "camera/depth/image_rect", rmw_qos_profile_sensor_data);
        camera_info_sub_.subscribe(this, "camera/rgb/camera_info", rmw_qos_profile_sensor_data);

        // Synchronize messages with approximate time policy
        sync_ = std::make_shared<ApproximateTimeSync>(
            SyncPolicy(10),
            rgb_sub_, depth_sub_, camera_info_sub_
        );
        sync_->registerCallback(
            std::bind(&PerceptionIntegrationNode::multimodalCallback, this,
                     std::placeholders::_1, std::placeholders::_2, std::placeholders::_3)
        );

        // Publisher for fused perception results
        fused_perception_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>(
            "fused_perception_results", 10
        );

        // Publisher for visualization
        visualization_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "perception_visualization", 10
        );

        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(tf_buffer_);
    }

private:
    using ApproximateTimeSync = message_filters::Synchronizer<SyncPolicy>;
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Image,
        sensor_msgs::msg::Image,
        sensor_msgs::msg::CameraInfo
    >;

    void multimodalCallback(
        const sensor_msgs::msg::Image::SharedPtr rgb_msg,
        const sensor_msgs::msg::Image::SharedPtr depth_msg,
        const sensor_msgs::msg::CameraInfo::SharedPtr camera_info_msg)
    {
        // Convert ROS images to OpenCV
        cv_bridge::CvImagePtr rgb_cv_ptr, depth_cv_ptr;
        try {
            rgb_cv_ptr = cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::BGR8);
            depth_cv_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        // Process with Isaac ROS perception components
        auto visual_slam_result = processVisualSLAM(rgb_cv_ptr->image, *camera_info_msg);
        auto detection_result = processObjectDetection(rgb_cv_ptr->image);
        auto 3d_localization = process3DLocalization(depth_cv_ptr->image, detection_result);

        // Fuse perception results
        auto fused_result = fusePerceptionResults(
            visual_slam_result,
            detection_result,
            3d_localization
        );

        // Publish fused results
        publishFusedResults(fused_result, rgb_msg->header);

        // Create visualization
        auto visualization_image = createPerceptionVisualization(
            rgb_cv_ptr->image,
            detection_result,
            fused_result
        );

        // Publish visualization
        visualization_pub_->publish(*visualization_image);
    }

    struct FusedPerceptionResult {
        std::vector<geometry_msgs::msg::PoseStamped> object_poses;
        std::vector<std::string> object_labels;
        std::vector<double> confidences;
        geometry_msgs::msg::PoseStamped robot_pose;
        std_msgs::msg::Header header;
    };

    FusedPerceptionResult fusePerceptionResults(
        const VisualSLAMResult& vslam_result,
        const DetectionResult& detection_result,
        const LocalizationResult& localization_result)
    {
        FusedPerceptionResult fused_result;
        fused_result.header = detection_result.header;  // Use detection header as reference

        // Transform detected objects to map frame using VSLAM pose
        for (size_t i = 0; i < detection_result.objects.size(); ++i) {
            if (i < localization_result.positions.size()) {
                geometry_msgs::msg::PointStamped obj_in_camera, obj_in_map;

                // Set up point in camera frame
                obj_in_camera.header = detection_result.header;
                obj_in_camera.point = localization_result.positions[i];

                // Transform to map frame using VSLAM pose
                try {
                    tf2::doTransform(obj_in_camera, obj_in_map, vslam_result.camera_to_map_transform);

                    geometry_msgs::msg::PoseStamped obj_pose;
                    obj_pose.header = fused_result.header;
                    obj_pose.pose.position = obj_in_map.point;
                    obj_pose.pose.orientation.w = 1.0;  // Simple orientation

                    fused_result.object_poses.push_back(obj_pose);
                    fused_result.object_labels.push_back(detection_result.labels[i]);
                    fused_result.confidences.push_back(detection_result.confidences[i]);
                } catch (tf2::TransformException& ex) {
                    RCLCPP_WARN(this->get_logger(), "Could not transform object pose: %s", ex.what());
                }
            }
        }

        // Set robot pose from VSLAM
        fused_result.robot_pose.header = fused_result.header;
        fused_result.robot_pose.pose = vslam_result.robot_pose;

        return fused_result;
    }

    void publishFusedResults(const FusedPerceptionResult& result, const std_msgs::msg::Header& header)
    {
        geometry_msgs::msg::PoseArray pose_array;
        pose_array.header = result.header;

        for (const auto& pose_stamped : result.object_poses) {
            pose_array.poses.push_back(pose_stamped.pose);
        }

        fused_perception_pub_->publish(pose_array);
    }

    sensor_msgs::msg::Image::SharedPtr createPerceptionVisualization(
        const cv::Mat& image,
        const DetectionResult& detections,
        const FusedPerceptionResult& fused_result)
    {
        cv::Mat vis_image = image.clone();

        // Draw bounding boxes for detections
        for (size_t i = 0; i < detections.boxes.size(); ++i) {
            const auto& box = detections.boxes[i];
            cv::rectangle(vis_image,
                         cv::Point(box.xmin, box.ymin),
                         cv::Point(box.xmax, box.ymax),
                         cv::Scalar(0, 255, 0), 2);

            // Add label
            std::string label = detections.labels[i] + ": " +
                               std::to_string(static_cast<int>(detections.confidences[i] * 100)) + "%";
            cv::putText(vis_image, label,
                       cv::Point(box.xmin, box.ymin - 10),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }

        // Convert back to ROS image
        cv_bridge::CvImage cv_vis;
        cv_vis.header = detections.header;
        cv_vis.encoding = sensor_msgs::image_encodings::BGR8;
        cv_vis.image = vis_image;

        return cv_vis.toImageMsg();
    }

    // Subscriptions
    message_filters::Subscriber<sensor_msgs::msg::Image> rgb_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
    message_filters::Subscriber<sensor_msgs::msg::CameraInfo> camera_info_sub_;
    std::shared_ptr<ApproximateTimeSync> sync_;

    // Publishers
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr fused_perception_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr visualization_pub_;

    // TF components
    tf2_ros::Buffer tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
};
```

## Performance Optimization

### Efficient Multimodal Processing

```cpp
// Efficient multimodal processing with threading
class EfficientMultimodalProcessor {
public:
    EfficientMultimodalProcessor() {
        // Create processing threads for each modality
        vision_thread_ = std::thread(&EfficientMultimodalProcessor::visionProcessingLoop, this);
        language_thread_ = std::thread(&EfficientMultimodalProcessor::languageProcessingLoop, this);
        fusion_thread_ = std::thread(&EfficientMultimodalProcessor::fusionProcessingLoop, this);
    }

    ~EfficientMultimodalProcessor() {
        running_ = false;

        if (vision_thread_.joinable()) vision_thread_.join();
        if (language_thread_.joinable()) language_thread_.join();
        if (fusion_thread_.joinable()) fusion_thread_.join();
    }

    void processInput(const SensorInput& sensor_input, const LanguageInput& language_input) {
        // Queue inputs for processing
        {
            std::lock_guard<std::mutex> lock(vision_queue_mutex_);
            vision_queue_.push(sensor_input);
        }

        {
            std::lock_guard<std::mutex> lock(language_queue_mutex_);
            language_queue_.push(language_input);
        }
    }

    std::optional<MultimodalResult> getResult() {
        std::lock_guard<std::mutex> lock(result_mutex_);
        if (!result_queue_.empty()) {
            auto result = result_queue_.front();
            result_queue_.pop();
            return result;
        }
        return std::nullopt;
    }

private:
    void visionProcessingLoop() {
        while (running_) {
            SensorInput input;

            // Get input from queue
            {
                std::lock_guard<std::mutex> lock(vision_queue_mutex_);
                if (!vision_queue_.empty()) {
                    input = vision_queue_.front();
                    vision_queue_.pop();
                }
            }

            if (hasValidInput(input)) {
                // Process vision data
                auto vision_result = processVision(input);

                // Store result
                {
                    std::lock_guard<std::mutex> lock(vision_result_mutex_);
                    latest_vision_result_ = vision_result;
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));  // Yield
        }
    }

    void languageProcessingLoop() {
        while (running_) {
            LanguageInput input;

            // Get input from queue
            {
                std::lock_guard<std::mutex> lock(language_queue_mutex_);
                if (!language_queue_.empty()) {
                    input = language_queue_.front();
                    language_queue_.pop();
                }
            }

            if (hasValidInput(input)) {
                // Process language data
                auto language_result = processLanguage(input);

                // Store result
                {
                    std::lock_guard<std::mutex> lock(language_result_mutex_);
                    latest_language_result_ = language_result;
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));  // Yield
        }
    }

    void fusionProcessingLoop() {
        while (running_) {
            // Check if we have both vision and language results
            std::optional<VisionResult> vision_result;
            std::optional<LanguageResult> language_result;

            {
                std::lock_guard<std::mutex> lock(vision_result_mutex_);
                vision_result = latest_vision_result_;
            }

            {
                std::lock_guard<std::mutex> lock(language_result_mutex_);
                language_result = latest_language_result_;
            }

            if (vision_result && language_result) {
                // Fuse results
                auto fused_result = fuseResults(*vision_result, *language_result);

                // Store final result
                {
                    std::lock_guard<std::mutex> lock(result_mutex_);
                    result_queue_.push(fused_result);
                }

                // Clear processed results
                {
                    std::lock_guard<std::mutex> lock(vision_result_mutex_);
                    latest_vision_result_.reset();
                }
                {
                    std::lock_guard<std::mutex> lock(language_result_mutex_);
                    latest_language_result_.reset();
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));  // Fusion rate
        }
    }

    std::queue<SensorInput> vision_queue_;
    std::queue<LanguageInput> language_queue_;
    std::queue<MultimodalResult> result_queue_;

    std::mutex vision_queue_mutex_;
    std::mutex language_queue_mutex_;
    std::mutex result_mutex_;
    std::mutex vision_result_mutex_;
    std::mutex language_result_mutex_;

    std::optional<VisionResult> latest_vision_result_;
    std::optional<LanguageResult> latest_language_result_;

    std::thread vision_thread_;
    std::thread language_thread_;
    std::thread fusion_thread_;
    std::atomic<bool> running_{true};
};
```

## Quality Assurance

### Perception Validation

```cpp
class PerceptionValidator {
public:
    struct PerceptionMetrics {
        double accuracy;
        double precision;
        double recall;
        double f1_score;
        double processing_time_ms;
        int total_detections;
        int correct_detections;
        int false_positives;
        int false_negatives;
    };

    PerceptionMetrics validatePerception(
        const MultimodalResult& result,
        const GroundTruth& ground_truth)
    {
        PerceptionMetrics metrics;

        // Calculate metrics based on comparison with ground truth
        int true_positives = 0, false_positives = 0, false_negatives = 0;

        for (const auto& detected_obj : result.objects) {
            bool matched = false;
            for (const auto& gt_obj : ground_truth.objects) {
                if (isObjectMatch(detected_obj, gt_obj)) {
                    true_positives++;
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                false_positives++;
            }
        }

        false_negatives = ground_truth.objects.size() - true_positives;

        metrics.accuracy = static_cast<double>(true_positives) /
                          (true_positives + false_positives + false_negatives);
        metrics.precision = static_cast<double>(true_positives) /
                           (true_positives + false_positives);
        metrics.recall = static_cast<double>(true_positives) /
                        (true_positives + false_negatives);
        metrics.f1_score = 2 * (metrics.precision * metrics.recall) /
                          (metrics.precision + metrics.recall);

        metrics.total_detections = result.objects.size();
        metrics.correct_detections = true_positives;
        metrics.false_positives = false_positives;
        metrics.false_negatives = false_negatives;

        return metrics;
    }

private:
    bool isObjectMatch(
        const PerceivedObject& detected,
        const GroundTruthObject& ground_truth,
        double position_threshold = 0.1,  // 10cm tolerance
        double label_threshold = 0.9)     // 90% IoU threshold
    {
        // Check if objects are close enough in 3D space
        double dist = std::sqrt(
            std::pow(detected.position.x - ground_truth.position.x, 2) +
            std::pow(detected.position.y - ground_truth.position.y, 2) +
            std::pow(detected.position.z - ground_truth.position.z, 2)
        );

        if (dist > position_threshold) {
            return false;
        }

        // Check if labels match
        return detected.label == ground_truth.label;
    }
};
```

## Troubleshooting Common Issues

### 1. Sensor Synchronization Issues
**Problem**: Different sensors operating at different frequencies causing temporal misalignment
**Solutions**:
- Use message filters with approximate time synchronization
- Implement interpolation for slower sensors
- Buffer sensor data with timestamps

### 2. Cross-Modal Alignment Issues
**Problem**: Difficulty in relating information across modalities
**Solutions**:
- Ensure proper calibration between sensors
- Use common reference frames (TF)
- Implement spatial verification between modalities

### 3. Computational Bottleneck
**Problem**: Multimodal processing exceeding real-time requirements
**Solutions**:
- Use multi-threading for parallel processing
- Implement processing priorities
- Use lightweight models for real-time applications

### 4. Data Association Issues
**Problem**: Incorrect matching of objects across modalities
**Solutions**:
- Implement robust data association algorithms
- Use temporal consistency checks
- Add geometric verification steps

## Best Practices

### 1. Modular Design
- Keep each modality processing separate initially
- Design clean interfaces between components
- Allow for easy replacement of individual components

### 2. Robust Error Handling
- Handle missing modalities gracefully
- Implement fallback behaviors
- Validate inputs before processing

### 3. Performance Monitoring
- Monitor processing times for each modality
- Track resource utilization
- Implement adaptive processing based on available resources

### 4. Validation and Testing
- Test with various environmental conditions
- Validate in simulation before real-world deployment
- Use ground truth data for performance evaluation

## Exercise

Create a complete multimodal perception system that:

1. Integrates RGB-D camera, IMU, and LIDAR data
2. Uses Isaac ROS components for individual perception tasks
3. Implements a fusion algorithm that combines visual, linguistic, and spatial information
4. Creates a perception pipeline that can handle natural language commands
5. Validates the system performance using synthetic data from Isaac Sim
6. Implements proper error handling and fallback behaviors

Test your system with various scenarios including:
- Object detection and localization
- Natural language command interpretation
- Multi-object tracking and interaction
- Dynamic environment adaptation
- Performance under different lighting conditions

Evaluate the system's performance using the metrics discussed in this section.