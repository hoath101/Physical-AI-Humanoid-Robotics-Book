# URDF Description

URDF (Unified Robot Description Format) is an XML format used to describe robot models in ROS. It defines the physical and visual properties of a robot, including its links, joints, and other components. URDF is essential for simulating robots in Gazebo and visualizing them in RViz.

## URDF Basics

A URDF file describes a robot as a collection of rigid bodies (links) connected by joints. The structure typically includes:

- **Links**: Rigid bodies that make up the robot structure
- **Joints**: Connections between links with specific degrees of freedom
- **Visual**: How the link appears in simulation and visualization
- **Collision**: Collision properties for physics simulation
- **Inertial**: Mass, center of mass, and inertia properties

## Basic URDF Structure

```xml
<?xml version="1.0"?>
<robot name="simple_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Child link connected via joint -->
  <joint name="base_to_top" type="fixed">
    <parent link="base_link"/>
    <child link="top_link"/>
    <origin xyz="0 0 0.3"/>
  </joint>

  <link name="top_link">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
      <material name="red">
        <color rgba="0.8 0 0 1"/>
      </material>
    </visual>
  </link>
</robot>
```

## Links

Links represent rigid bodies in the robot. Each link can have:

- **Visual**: How the link looks in visualization
- **Collision**: How the link interacts in physics simulation
- **Inertial**: Physical properties for dynamics simulation

### Visual Properties

```xml
<link name="link_name">
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <!-- Choose one geometry type -->
      <box size="0.1 0.1 0.1"/>
      <!-- <cylinder radius="0.1" length="0.1"/> -->
      <!-- <sphere radius="0.1"/> -->
      <!-- <mesh filename="package://path/to/mesh.stl"/> -->
    </geometry>
    <material name="material_name">
      <color rgba="0.8 0.2 0.2 1.0"/>
    </material>
  </visual>
</link>
```

### Collision Properties

```xml
<collision>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <geometry>
    <box size="0.1 0.1 0.1"/>
  </geometry>
</collision>
```

### Inertial Properties

```xml
<inertial>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <mass value="0.1"/>
  <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
</inertial>
```

## Joints

Joints connect links and define how they can move relative to each other:

```xml
<joint name="joint_name" type="joint_type">
  <parent link="parent_link_name"/>
  <child link="child_link_name"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>

  <!-- For revolute joints -->
  <axis xyz="0 0 1"/>
  <limit lower="-3.14" upper="3.14" effort="100" velocity="1"/>

  <!-- For continuous joints (like wheels) -->
  <!-- <mimic joint="other_joint" multiplier="1.0" offset="0.0"/> -->
</joint>
```

### Joint Types

- **fixed**: No movement allowed
- **revolute**: Rotational movement around an axis (limited)
- **continuous**: Rotational movement around an axis (unlimited)
- **prismatic**: Linear movement along an axis (limited)
- **floating**: 6 DOF movement (rarely used)
- **planar**: Movement in a plane (rarely used)

## Complete Humanoid Robot Example

Here's a simplified example of a humanoid robot with basic body parts:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Base body -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
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

  <!-- Head -->
  <joint name="neck_joint" type="revolute">
    <parent link="base_link"/>
    <child link="head_link"/>
    <origin xyz="0 0 0.35" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="10" velocity="2"/>
  </joint>

  <link name="head_link">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="skin">
        <color rgba="0.8 0.6 0.4 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Left Arm -->
  <joint name="left_shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.2 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="2"/>
  </joint>

  <link name="left_upper_arm">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
      <material name="gray"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.0005"/>
    </inertial>
  </link>

  <joint name="left_elbow_joint" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_lower_arm"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="2"/>
  </joint>

  <link name="left_lower_arm">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
      <material name="gray"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.3"/>
      <inertia ixx="0.003" ixy="0.0" ixz="0.0" iyy="0.003" iyz="0.0" izz="0.0003"/>
    </inertial>
  </link>
</robot>
```

## Xacro for Complex Models

Xacro is an XML macro language that makes URDF more manageable for complex robots:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_with_xacro">
  <!-- Define properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="body_width" value="0.3" />
  <xacro:property name="body_depth" value="0.2" />
  <xacro:property name="body_height" value="0.5" />

  <!-- Macro for creating limbs -->
  <xacro:macro name="simple_arm" params="name parent *origin">
    <joint name="${name}_shoulder_joint" type="revolute">
      <xacro:insert_block name="origin"/>
      <parent link="${parent}"/>
      <child link="${name}_upper_arm"/>
      <axis xyz="0 1 0"/>
      <limit lower="-1.57" upper="1.57" effort="10" velocity="2"/>
    </joint>

    <link name="${name}_upper_arm">
      <visual>
        <geometry>
          <cylinder length="0.3" radius="0.05"/>
        </geometry>
        <material name="gray">
          <color rgba="0.5 0.5 0.5 1"/>
        </material>
      </visual>
    </link>
  </xacro:macro>

  <!-- Use the macro -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="${body_width} ${body_depth} ${body_height}"/>
      </geometry>
    </visual>
  </link>

  <xacro:simple_arm name="left" parent="base_link">
    <origin xyz="${body_width/2} 0 ${body_height/4}" rpy="0 0 0"/>
  </xacro:simple_arm>
</robot>
```

## Working with URDF in ROS 2

To use URDF in ROS 2 applications:

1. **Robot State Publisher**: Publishes the robot's joint states and transforms
2. **TF2**: Handles coordinate transformations between links
3. **Gazebo Integration**: Use the robot in simulation

### Launching with Robot State Publisher

```python
from launch import LaunchDescription
from launch.substitutions import Command
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('your_robot_description')
    urdf_file = os.path.join(pkg_share, 'urdf', 'your_robot.urdf')

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': Command(['xacro ', urdf_file])}]
    )

    return LaunchDescription([robot_state_publisher_node])
```

## Best Practices

- Use consistent naming conventions for links and joints
- Keep visual and collision geometries as simple as possible for performance
- Use Xacro for complex robots to reduce redundancy
- Include proper inertial properties for accurate simulation
- Validate URDF files using tools like `check_urdf`
- Use appropriate joint limits based on physical constraints

## Exercise

Create a URDF file for a simple humanoid robot with:
1. A body, head, and 4 limbs (arms and legs)
2. Proper joint connections with realistic movement ranges
3. Visual and collision properties for each link
4. Use Xacro macros to avoid redundancy in similar limbs