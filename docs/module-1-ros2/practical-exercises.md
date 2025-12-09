# Practical Exercises - ROS 2 Fundamentals

This section provides hands-on exercises to reinforce the concepts learned in the ROS 2 module. Complete these exercises to gain practical experience with ROS 2 nodes, topics, services, and URDF.

## Exercise 1: Basic Publisher and Subscriber

### Objective
Create a simple publisher that sends messages containing the current joint positions of a humanoid robot and a subscriber that logs these positions.

### Steps
1. Create a new ROS 2 package called `humanoid_control`
2. Create a publisher node that publishes joint positions as a `sensor_msgs/JointState` message
3. Create a subscriber node that receives and logs the joint positions
4. Use appropriate message types and topic names
5. Test the communication between nodes

### Code Template
```python
# publisher_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import math

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')
        self.publisher = self.create_publisher(JointState, 'joint_states', 10)
        self.timer = self.create_timer(0.1, self.publish_joint_state)
        self.time = 0.0

    def publish_joint_state(self):
        msg = JointState()
        msg.name = ['hip_joint', 'knee_joint', 'ankle_joint', 'shoulder_joint', 'elbow_joint']
        msg.position = [
            math.sin(self.time),           # hip
            math.cos(self.time) * 0.5,     # knee
            math.sin(self.time * 0.5),     # ankle
            math.cos(self.time * 0.3),     # shoulder
            math.sin(self.time * 0.7)      # elbow
        ]
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        self.publisher.publish(msg)
        self.time += 0.1

def main(args=None):
    rclpy.init(args=args)
    node = JointStatePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Expected Output
The subscriber should log joint positions at 10Hz, showing oscillating values for each joint.

## Exercise 2: Service for Robot Control

### Objective
Create a service that accepts a target position for a humanoid robot and returns whether the position is reachable.

### Steps
1. Define a custom service message for position validation
2. Create a service server that checks if a position is within the robot's reach
3. Create a client that calls the service with different positions
4. Test with both reachable and unreachable positions

### Service Definition (`srv/ValidatePosition.srv`)
```
geometry_msgs/Point target_position
---
bool is_reachable
string reason
```

## Exercise 3: URDF Robot Model

### Objective
Create a complete URDF model of a simple humanoid robot with proper kinematic chains.

### Steps
1. Create a URDF file defining a humanoid with body, head, arms, and legs
2. Include proper joint types and limits
3. Add visual and collision properties
4. Use Xacro to create macros for similar limbs
5. Validate the URDF using `check_urdf`

### Template Structure
```xml
<?xml version="1.0"?>
<robot name="simple_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Define properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />

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
    <!-- Add collision and inertial properties -->
  </link>

  <!-- Use Xacro macros for limbs -->
  <xacro:macro name="arm" params="name parent xyz">
    <!-- Define arm structure with joints -->
  </xacro:macro>

  <!-- Create arms using macros -->
  <xacro:arm name="left" parent="base_link" xyz="0.15 0 0.1"/>
  <xacro:arm name="right" parent="base_link" xyz="0.15 0 0.1"/>
</robot>
```

## Exercise 4: Robot State Publisher Integration

### Objective
Integrate your URDF model with the robot_state_publisher to visualize joint transformations.

### Steps
1. Create a launch file that starts your joint state publisher and robot_state_publisher
2. Use the `--ros-args` to pass the URDF to robot_state_publisher
3. Visualize the robot in RViz2
4. Verify that joint transformations are properly published

### Launch File Template
```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')

    urdf_file = get_package_share_directory('humanoid_control') + '/urdf/humanoid.urdf'

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': Command(['xacro ', urdf_file])
        }]
    )

    joint_state_publisher_node = Node(
        package='humanoid_control',
        executable='joint_state_publisher',
        name='joint_state_publisher'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use sim time if true'),
        robot_state_publisher_node,
        joint_state_publisher_node
    ])
```

## Exercise 5: Complete Control Node

### Objective
Create a complete control node that integrates all concepts: publishers, subscribers, services, and parameters.

### Steps
1. Create a node that subscribes to desired joint positions
2. Publish current joint states
3. Provide a service to validate positions
4. Use parameters to configure control behavior
5. Add proper logging and error handling

### Template
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool  # or your custom service
import numpy as np

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')

        # Declare parameters
        self.declare_parameter('control_frequency', 50)
        self.declare_parameter('max_velocity', 2.0)

        # Create publishers and subscribers
        self.joint_state_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.joint_command_sub = self.create_subscription(
            JointState, 'joint_commands', self.joint_command_callback, 10)

        # Create services
        self.validate_srv = self.create_service(
            ValidatePosition, 'validate_position', self.validate_position_callback)

        # Create timer
        freq = self.get_parameter('control_frequency').value
        self.timer = self.create_timer(1.0/freq, self.control_loop)

        # Initialize joint states
        self.current_joints = JointState()

    def joint_command_callback(self, msg):
        # Process joint commands
        pass

    def validate_position_callback(self, request, response):
        # Validate if position is reachable
        pass

    def control_loop(self):
        # Main control loop
        pass

def main(args=None):
    rclpy.init(args=args)
    controller = HumanoidController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()
```

## Validation and Testing

To verify your implementations:

1. **Check URDF validity**:
   ```bash
   check_urdf path/to/your/robot.urdf
   ```

2. **List ROS 2 topics**:
   ```bash
   ros2 topic list
   ```

3. **Echo topics**:
   ```bash
   ros2 topic echo /joint_states
   ```

4. **Call services**:
   ```bash
   ros2 service call /validate_position your_robot_msgs/srv/ValidatePosition "{target_position: {x: 1.0, y: 0.0, z: 0.0}}"
   ```

## Troubleshooting Tips

- Ensure all packages are properly sourced: `source /opt/ros/humble/setup.bash`
- Check that your package has the correct dependencies in `package.xml`
- Verify that your CMakeLists.txt includes all necessary libraries
- Use `rqt_graph` to visualize the ROS graph and verify connections
- Check that joint names in your messages match those in your URDF

## Extension Challenges

1. Add a simple PID controller to your joint control node
2. Implement a basic trajectory execution interface
3. Create a simple GUI using rqt to control the robot
4. Add collision checking between robot links
5. Implement forward and inverse kinematics for simple movements

Complete these exercises to solidify your understanding of ROS 2 fundamentals before moving to the next module.