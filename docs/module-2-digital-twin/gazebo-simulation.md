# Gazebo Simulation

Gazebo is the standard simulation environment for ROS and provides a powerful physics engine for realistic robot simulation. This section covers setting up Gazebo, creating simulation worlds, and integrating with ROS 2.

## Introduction to Gazebo

Gazebo provides:
- Realistic physics simulation using ODE, Bullet, or Simbody
- High-quality 3D graphics rendering
- Sensor simulation (cameras, LIDAR, IMU, etc.)
- Multiple robot support
- Plugin architecture for custom functionality

## Installation

If not already installed during ROS 2 setup:

```bash
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control ros-humble-gazebo-dev
```

## Basic Gazebo Concepts

### Worlds
Gazebo worlds define the environment where simulation takes place. They include:
- Physics properties (gravity, air density)
- Models (robots, objects, obstacles)
- Lighting and visual properties
- Plugins for additional functionality

Example world file:
```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="simple_world">
    <!-- Physics properties -->
    <physics type="ode">
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Lighting -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
    </light>

    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Sky -->
    <include>
      <uri>model://sun</uri>
    </include>
  </world>
</sdf>
```

### Models
Models represent objects in the simulation. They include:
- Visual properties (appearance)
- Collision properties (physics interaction)
- Inertial properties (mass, center of mass)
- Joints (connections between parts)

## ROS 2 Integration

### Gazebo ROS Packages
The `gazebo_ros_pkgs` provide ROS 2 interfaces for Gazebo:
- `gazebo_ros`: Core ROS 2 interface
- `gazebo_plugins`: Various simulation plugins
- `gazebo_msgs`: ROS 2 messages for Gazebo

### Launching Gazebo with ROS 2
```bash
# Launch empty world with ROS 2 interface
ros2 launch gazebo_ros empty_world.launch.py

# Launch with a specific world file
ros2 launch gazebo_ros empty_world.launch.py world:=/path/to/world.sdf
```

## Controlling Robots in Gazebo

### Joint State Publisher
Gazebo publishes joint states that can be used with ROS 2:
```bash
# View joint states from simulated robot
ros2 topic echo /joint_states
```

### Robot Control
Use ROS 2 controllers to command robot joints:
```xml
<!-- In your robot's URDF or SDF -->
<gazebo>
  <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
    <robotNamespace>/your_robot</robotNamespace>
  </plugin>
</gazebo>
```

## Creating a Simple Simulation

### 1. Create a World File
Create `simple_room.sdf`:
```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="simple_room">
    <physics type="ode">
      <gravity>0 0 -9.8</gravity>
    </physics>

    <light name="sun" type="directional">
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
    </light>

    <include>
      <uri>model://ground_plane</uri>
    </include>

    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Simple box obstacle -->
    <model name="box">
      <pose>2 0 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.8 0.3 0.3 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.166667</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.166667</iyy>
            <iyz>0</iyz>
            <izz>0.166667</izz>
          </inertia>
        </inertial>
      </link>
    </model>
  </world>
</sdf>
```

### 2. Launch the Simulation
```bash
gz sim simple_room.sdf
```

## Robot Simulation in Gazebo

### URDF Integration
To use your URDF robot in Gazebo, add Gazebo-specific tags:

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Your URDF model -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Gazebo-specific tags -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
    <mu1>0.2</mu1>
    <mu2>0.2</mu2>
  </gazebo>

  <!-- Controller plugin -->
  <gazebo>
    <plugin filename="libgazebo_ros2_control.so" name="gazebo_ros2_control">
      <parameters>$(find your_robot_description)/config/controllers.yaml</parameters>
    </plugin>
  </gazebo>
</robot>
```

### Controllers Configuration
Create `controllers.yaml`:
```yaml
controller_manager:
  ros__parameters:
    update_rate: 100  # Hz

    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

    velocity_controller:
      type: velocity_controllers/JointGroupVelocityController

    position_controller:
      type: position_controllers/JointGroupPositionController
```

## Sensor Simulation

Gazebo can simulate various sensors:

### Camera
```xml
<gazebo reference="camera_link">
  <sensor name="camera" type="camera">
    <camera name="head">
      <horizontal_fov>1.047</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>camera_link</frame_name>
      <topic_name>image_raw</topic_name>
    </plugin>
  </sensor>
</gazebo>
```

### LIDAR
```xml
<gazebo reference="lidar_link">
  <sensor name="lidar" type="ray">
    <ray>
      <scan>
        <horizontal>
          <samples>720</samples>
          <resolution>1</resolution>
          <min_angle>-1.570796</min_angle>
          <max_angle>1.570796</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
      <ros>
        <namespace>laser</namespace>
        <remapping>~/out:=scan</remapping>
      </ros>
      <output_type>sensor_msgs/LaserScan</output_type>
    </plugin>
  </sensor>
</gazebo>
```

## Debugging and Visualization

### Gazebo GUI
Launch with GUI for visualization:
```bash
gz sim -g your_world.sdf
```

### Model States
Monitor model states:
```bash
ros2 topic echo /model_states
```

### TF Tree
Visualize transforms:
```bash
ros2 run tf2_tools view_frames
```

## Best Practices

- Start with simple models and gradually add complexity
- Use appropriate physics properties (mass, friction) for realistic behavior
- Test controllers in simulation before deploying to real robots
- Use collision checking to prevent interpenetration
- Optimize update rates for performance vs. accuracy trade-offs

## Exercise

Create a simple simulation with your humanoid robot navigating around obstacles in Gazebo. Implement a basic controller that allows the robot to move forward, turn, and avoid obstacles using simulated sensors.