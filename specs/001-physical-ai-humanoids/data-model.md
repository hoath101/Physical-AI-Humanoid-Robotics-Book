# Data Model: Physical AI & Humanoid Robotics Book

## Core Entities

### Humanoid Robot
- **Description**: A robotic system designed with human-like form and movement capabilities
- **Attributes**:
  - joints: Array of joint configurations
  - sensors: Array of sensor types and configurations
  - kinematics: Forward and inverse kinematics data
  - urdf_path: Path to URDF description file
- **Relationships**: Used in Digital Twin, controlled by ROS 2 nodes

### ROS 2 Node
- **Description**: An executable process within the Robot Operating System 2 framework that performs a specific task
- **Attributes**:
  - node_name: Unique identifier for the node
  - topics: Array of subscribed/published topics
  - services: Array of available services
  - actions: Array of available actions
- **Relationships**: Controls Humanoid Robot, communicates via Topics/Services/Actions

### Digital Twin
- **Description**: A virtual model of a physical robot, used for simulation and testing
- **Attributes**:
  - simulation_environment: Gazebo or Unity
  - urdf_model: Reference to URDF description
  - physics_properties: Mass, friction, collision properties
  - sensor_configurations: Sensor placements and parameters
- **Relationships**: Based on Humanoid Robot, runs in simulation environment

### NVIDIA Isaac AI Component
- **Description**: AI framework components for perception and navigation
- **Attributes**:
  - component_type: Perception, Navigation, SLAM, etc.
  - configuration: AI model and parameter settings
  - input_sources: Sensor data sources
  - output_targets: Navigation plans, detection results
- **Relationships**: Processes data from Digital Twin sensors, controls navigation

### Large Language Model (LLM)
- **Description**: AI model capable of understanding and generating human-like text, used here for planning
- **Attributes**:
  - model_type: Specific LLM being used
  - prompt_templates: Predefined templates for robotics tasks
  - action_mappings: Mappings from natural language to robot actions
- **Relationships**: Processes Whisper output, generates ROS 2 commands

### Whisper (Speech-to-Text)
- **Description**: A speech-to-text AI model used for transcribing natural language commands
- **Attributes**:
  - transcription_accuracy: Accuracy metrics
  - supported_languages: Languages supported
  - audio_input_config: Audio input parameters
- **Relationships**: Input to LLM, processes voice commands

## Module-Specific Data Structures

### Module 1: ROS 2 Data
- Topic: name, message_type, publishers, subscribers
- Service: name, request_type, response_type
- Action: name, goal_type, result_type, feedback_type

### Module 2: Simulation Data
- URDF: links, joints, materials, collision properties
- Gazebo World: environment, objects, physics parameters
- Unity Scene: GameObjects, components, physics settings

### Module 3: Perception Data
- Sensor Data: lidar, camera, IMU, GPS readings
- Object Detection: bounding_boxes, confidence_scores, class_labels
- Navigation Plan: waypoints, cost_map, path_constraints

### Module 4: VLA Data
- Voice Command: audio_data, transcription, intent
- Action Sequence: ordered list of robot commands
- Multimodal Input: combined sensor and language data