# URDF Model Creation and Validation

Creating accurate URDF models is fundamental to successful digital twin implementation. This section covers best practices for creating URDF models and validating them for use in simulation environments.

## URDF Model Best Practices

### 1. Proper Kinematic Structure
A well-structured URDF model should have a clear kinematic chain:

```xml
<?xml version="1.0"?>
<robot name="humanoid_with_proper_structure" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Base link (required - no geometry needed for simple base) -->
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0.25" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.3 0.5"/>
      </geometry>
      <material name="light_gray">
        <color rgba="0.7 0.7 0.7 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.25" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.3 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0.25" rpy="0 0 0"/>
      <inertia ixx="0.2" ixy="0.0" ixz="0.0" iyy="0.2" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Head -->
  <joint name="neck_joint" type="revolute">
    <parent link="base_link"/>
    <child link="head_link"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="10" velocity="2"/>
    <dynamics damping="0.5" friction="0.1"/>
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
      <mass value="2.0"/>
      <inertia ixx="0.004" ixy="0.0" ixz="0.0" iyy="0.004" iyz="0.0" izz="0.004"/>
    </inertial>
  </link>

  <!-- Left Arm (using Xacro macros for consistency) -->
  <xacro:macro name="simple_arm" params="side parent xyz rpy joint_limits_lower joint_limits_upper">
    <!-- Shoulder -->
    <joint name="${side}_shoulder_pitch_joint" type="revolute">
      <parent link="${parent}"/>
      <child link="${side}_shoulder_link"/>
      <origin xyz="${xyz}" rpy="${rpy}"/>
      <axis xyz="0 1 0"/>
      <limit lower="${joint_limits_lower}" upper="${joint_limits_upper}" effort="15" velocity="2"/>
      <dynamics damping="0.5" friction="0.1"/>
    </joint>

    <link name="${side}_shoulder_link">
      <visual>
        <geometry>
          <box size="0.08 0.08 0.1"/>
        </geometry>
        <material name="dark_gray">
          <color rgba="0.3 0.3 0.3 1"/>
        </material>
      </visual>
      <collision>
        <geometry>
          <box size="0.08 0.08 0.1"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="1.0"/>
        <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
      </inertial>
    </link>

    <joint name="${side}_shoulder_yaw_joint" type="revolute">
      <parent link="${side}_shoulder_link"/>
      <child link="${side}_upper_arm_link"/>
      <origin xyz="0 0 -0.1" rpy="0 0 0"/>
      <axis xyz="1 0 0"/>
      <limit lower="-1.57" upper="1.57" effort="15" velocity="2"/>
      <dynamics damping="0.5" friction="0.1"/>
    </joint>

    <link name="${side}_upper_arm_link">
      <visual>
        <geometry>
          <cylinder length="0.3" radius="0.05"/>
        </geometry>
        <material name="dark_gray"/>
      </visual>
      <collision>
        <geometry>
          <cylinder length="0.3" radius="0.05"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="1.5"/>
        <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.0005"/>
      </inertial>
    </link>

    <joint name="${side}_elbow_joint" type="revolute">
      <parent link="${side}_upper_arm_link"/>
      <child link="${side}_lower_arm_link"/>
      <origin xyz="0 0 -0.3" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="-0.1" upper="2.0" effort="10" velocity="2"/>
      <dynamics damping="0.3" friction="0.1"/>
    </joint>

    <link name="${side}_lower_arm_link">
      <visual>
        <geometry>
          <cylinder length="0.25" radius="0.04"/>
        </geometry>
        <material name="dark_gray"/>
      </visual>
      <collision>
        <geometry>
          <cylinder length="0.25" radius="0.04"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="1.0"/>
        <inertia ixx="0.003" ixy="0.0" ixz="0.0" iyy="0.003" iyz="0.0" izz="0.0003"/>
      </inertial>
    </link>
  </xacro:macro>

  <!-- Use the macro for both arms -->
  <xacro:simple_arm
    side="left"
    parent="base_link"
    xyz="0.15 0 0.2"
    rpy="0 0 0"
    joint_limits_lower="-1.57"
    joint_limits_upper="1.57"/>

  <xacro:simple_arm
    side="right"
    parent="base_link"
    xyz="-0.15 0 0.2"
    rpy="0 0 0"
    joint_limits_lower="-1.57"
    joint_limits_upper="1.57"/>
</robot>
```

### 2. Proper Inertial Properties
Accurate inertial properties are crucial for realistic physics simulation:

```xml
<!-- For a solid box -->
<inertial>
  <mass value="5.0"/>
  <origin xyz="0 0 0"/>
  <inertia
    ixx="0.0833333"  <!-- m*(h² + d²)/12 -->
    ixy="0.0"
    ixz="0.0"
    iyy="0.0833333"  <!-- m*(w² + d²)/12 -->
    iyz="0.0"
    izz="0.0833333"/> <!-- m*(w² + h²)/12 -->
</inertial>

<!-- For a solid cylinder along Z-axis -->
<inertial>
  <mass value="1.0"/>
  <origin xyz="0 0 0"/>
  <inertia
    ixx="0.0052083"  <!-- m*(3*r² + h²)/12 -->
    ixy="0.0"
    ixz="0.0"
    iyy="0.0052083"
    iyz="0.0"
    izz="0.0025"/>    <!-- m*r²/2 -->
</inertial>

<!-- For a solid sphere -->
<inertial>
  <mass value="2.0"/>
  <origin xyz="0 0 0"/>
  <inertia
    ixx="0.02"       <!-- 2*m*r²/5 -->
    ixy="0.0"
    ixz="0.0"
    iyy="0.02"
    iyz="0.0"
    izz="0.02"/>
</inertial>
```

## Validation Techniques

### 1. URDF Validation Tools

#### Check URDF
Use the `check_urdf` command to validate your URDF:

```bash
# Install the tool if not already installed
sudo apt install ros-humble-urdfdom-py

# Check your URDF file
check_urdf /path/to/your/robot.urdf
```

#### View URDF
Visualize your robot structure:

```bash
# Install visualization tools
sudo apt install ros-humble-joint-state-publisher-gui

# View the robot
urdf_to_graphiz /path/to/your/robot.urdf
# This creates .gv files that can be viewed with Graphviz
```

### 2. Simulation Validation

#### Gazebo Validation
Test your URDF in Gazebo:

```xml
<!-- Add Gazebo-specific tags for simulation -->
<gazebo reference="base_link">
  <material>Gazebo/Blue</material>
  <mu1>0.2</mu1>
  <mu2>0.2</mu2>
  <kp>1000000.0</kp>  <!-- Contact stiffness -->
  <kd>100.0</kd>      <!-- Contact damping -->
</gazebo>

<!-- Add transmission for ROS control -->
<xacro:macro name="transmission_block" params="joint_name">
  <transmission name="${joint_name}_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="${joint_name}">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="${joint_name}_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>
</xacro:macro>
```

### 3. Kinematic Validation

#### Forward Kinematics
Validate that your kinematic chain works correctly:

```python
#!/usr/bin/env python3
import rospy
import tf2_ros
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_kinematics import KDLKinematics

def validate_kinematics():
    # Load robot description
    robot = URDF.from_xml_string(rospy.get_param('robot_description'))

    # Create kinematic chain
    kdl_kin = KDLKinematics(robot, 'base_link', 'left_hand_link')

    # Test joint positions
    q = [0.0, 0.0, 0.0, 0.0, 0.0]  # joint angles
    pose = kdl_kin.forward(q)
    print(f"End effector pose: {pose}")

    # Test inverse kinematics
    q_ik = kdl_kin.inverse(pose)
    print(f"IK solution: {q_ik}")

if __name__ == '__main__':
    rospy.init_node('kinematic_validator')
    validate_kinematics()
```

## Advanced Validation Examples

### 1. Collision Detection Validation
Ensure no self-collision in common poses:

```xml
<!-- Add self-collision checking parameters -->
<robot name="collision_test_robot">
  <!-- Define collision pairs to ignore (if needed) -->
  <gazebo>
    <self_collide>false</self_collide>  <!-- Only for specific testing -->
  </gazebo>

  <!-- Or be more specific with collision filtering -->
  <gazebo reference="link1">
    <collision>
      <surface>
        <contact>
          <collide_without_contact>false</collide_without_contact>
        </contact>
      </surface>
    </collision>
  </gazebo>
</robot>
```

### 2. Center of Mass Validation
Calculate and verify center of mass:

```python
#!/usr/bin/env python3
import rospy
from urdf_parser_py.urdf import URDF
import numpy as np

def calculate_total_com():
    robot = URDF.from_xml_string(rospy.get_param('robot_description'))

    total_mass = 0.0
    weighted_com = np.array([0.0, 0.0, 0.0])

    for link in robot.links:
        if link.inertial and link.inertial.mass:
            mass = link.inertial.mass
            com = np.array([
                link.inertial.origin.xyz[0],
                link.inertial.origin.xyz[1],
                link.inertial.origin.xyz[2]
            ])

            total_mass += mass
            weighted_com += mass * com

    overall_com = weighted_com / total_mass if total_mass > 0 else np.array([0, 0, 0])
    print(f"Overall center of mass: {overall_com}")
    print(f"Total mass: {total_mass}")

if __name__ == '__main__':
    rospy.init_node('com_calculator')
    calculate_total_com()
```

### 3. Dynamic Validation
Test dynamic properties in simulation:

```yaml
# controllers.yaml for testing
controller_manager:
  ros__parameters:
    update_rate: 100

    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

    left_arm_controller:
      type: position_controllers/JointGroupPositionController
      joints:
        - left_shoulder_pitch_joint
        - left_shoulder_yaw_joint
        - left_elbow_joint

    right_arm_controller:
      type: position_controllers/JointGroupPositionController
      joints:
        - right_shoulder_pitch_joint
        - right_shoulder_yaw_joint
        - right_elbow_joint
```

### 4. Range of Motion Validation
Test joint limits and workspace:

```python
#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import JointState
import numpy as np

class JointRangeValidator:
    def __init__(self):
        self.joint_limits = {
            'left_shoulder_pitch_joint': (-1.57, 1.57),
            'left_shoulder_yaw_joint': (-1.57, 1.57),
            'left_elbow_joint': (-0.1, 2.0)
        }

        rospy.Subscriber('/joint_states', JointState, self.joint_callback)

    def joint_callback(self, msg):
        for i, name in enumerate(msg.name):
            if name in self.joint_limits:
                pos = msg.position[i]
                min_pos, max_pos = self.joint_limits[name]

                if pos < min_pos or pos > max_pos:
                    rospy.logwarn(f"Joint {name} out of range: {pos} (limits: {min_pos}, {max_pos})")
                elif abs(pos - min_pos) < 0.1 or abs(pos - max_pos) < 0.1:
                    rospy.loginfo(f"Joint {name} approaching limit: {pos}")

if __name__ == '__main__':
    rospy.init_node('joint_range_validator')
    validator = JointRangeValidator()
    rospy.spin()
```

## Common Validation Issues and Solutions

### 1. Floating Point Errors
- Use appropriate precision in URDF values
- Validate with tolerance checks in code

### 2. Mass Distribution Issues
- Ensure center of mass is reasonable
- Verify moments of inertia are positive and realistic

### 3. Joint Limit Problems
- Test extreme positions in simulation
- Validate with physical constraints

### 4. Collision Issues
- Check for interpenetration at default positions
- Test various poses for self-collision

## Automated Validation Script

Create a comprehensive validation script:

```bash
#!/bin/bash
# validate_robot.sh

URDF_FILE=$1
if [ -z "$URDF_FILE" ]; then
    echo "Usage: $0 <urdf_file>"
    exit 1
fi

echo "Validating $URDF_FILE..."

# Check if file exists
if [ ! -f "$URDF_FILE" ]; then
    echo "Error: File $URDF_FILE does not exist"
    exit 1
fi

# Validate URDF structure
echo "Checking URDF structure..."
check_urdf "$URDF_FILE" 2>&1 | tee urdf_check.log
if [ $? -ne 0 ]; then
    echo "URDF validation failed!"
    exit 1
fi

# Check for common issues
echo "Checking for common issues..."

# Check for duplicate names
DUPLICATE_LINKS=$(grep -oP '(?<=<link name=")[^"]*' "$URDF_FILE" | sort | uniq -d)
if [ ! -z "$DUPLICATE_LINKS" ]; then
    echo "Warning: Duplicate link names found: $DUPLICATE_LINKS"
fi

DUPLICATE_JOINTS=$(grep -oP '(?<=<joint name=")[^"]*' "$URDF_FILE" | sort | uniq -d)
if [ ! -z "$DUPLICATE_JOINTS" ]; then
    echo "Warning: Duplicate joint names found: $DUPLICATE_JOINTS"
fi

# Check for missing inertial properties
MISSING_INERTIAL=$(grep -B 10 'mass' "$URDF_FILE" | grep -E '^<link name=' | wc -l)
TOTAL_LINKS=$(grep -c '<link name=' "$URDF_FILE")
if [ $TOTAL_LINKS -ne $MISSING_INERTIAL ]; then
    echo "Warning: Some links may be missing inertial properties"
fi

echo "Validation complete. Check urdf_check.log for details."
```

## Best Practices Summary

1. **Always validate** URDF files before simulation
2. **Use consistent units** (SI units: meters, kilograms, seconds)
3. **Test kinematic chains** with forward and inverse kinematics
4. **Verify inertial properties** with realistic values
5. **Test collision detection** with various robot poses
6. **Check joint limits** against physical constraints
7. **Validate center of mass** for stability
8. **Use Xacro macros** for complex, repetitive structures

## Exercise

Create a complete humanoid robot URDF model with:
1. Proper kinematic structure (tree topology)
2. Accurate inertial properties for all links
3. Appropriate joint limits based on human anatomy
4. Validation tests to ensure the model is physically realistic
5. Integration with both Gazebo and Unity simulation environments

Validate your model using the tools and techniques described in this section, ensuring it behaves realistically in simulation.