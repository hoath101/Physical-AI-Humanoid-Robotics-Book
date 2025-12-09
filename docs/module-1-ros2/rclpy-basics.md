# rclpy Basics

rclpy is the Python client library for ROS 2. It provides the Python API for developing ROS 2 applications, allowing you to create nodes, publish/subscribe to topics, provide/call services, and interact with other ROS 2 concepts.

## Installation and Setup

rclpy is typically installed as part of the ROS 2 Python development packages:

```bash
# If you installed ROS 2 from packages, rclpy should already be available
# Otherwise, install the Python development packages:
sudo apt install python3-ros-dev-tools
```

## Basic Node Structure

Every ROS 2 Python node follows a similar structure:

```python
import rclpy
from rclpy.node import Node

class MyNode(Node):
    def __init__(self):
        super().__init__('node_name')
        # Node initialization code here
        pass

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Creating Publishers and Subscribers

### Publisher

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class PublisherNode(Node):
    def __init__(self):
        super().__init__('publisher_node')

        # Create a publisher
        self.publisher = self.create_publisher(String, 'topic_name', 10)

        # Create a timer to periodically publish messages
        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello from publisher'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: {msg.data}')

def main(args=None):
    rclpy.init(args=args)
    node = PublisherNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Subscriber

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class SubscriberNode(Node):
    def __init__(self):
        super().__init__('subscriber_node')

        # Create a subscriber
        self.subscription = self.create_subscription(
            String,
            'topic_name',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: {msg.data}')

def main(args=None):
    rclpy.init(args=args)
    node = SubscriberNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Creating Services and Clients

### Service Server

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class ServiceServer(Node):
    def __init__(self):
        super().__init__('service_server')

        # Create a service
        self.service = self.create_service(
            AddTwoInts,
            'add_two_ints',
            self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Request received: {request.a} + {request.b} = {response.sum}')
        return response

def main(args=None):
    rclpy.init(args=args)
    node = ServiceServer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Parameters

ROS 2 nodes can use parameters for configuration:

```python
import rclpy
from rclpy.node import Node

class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values
        self.declare_parameter('robot_name', 'humanoid_robot')
        self.declare_parameter('max_velocity', 1.0)

        # Get parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.max_velocity = self.get_parameter('max_velocity').value

        self.get_logger().info(f'Robot: {self.robot_name}, Max Velocity: {self.max_velocity}')

def main(args=None):
    rclpy.init(args=args)
    node = ParameterNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Quality of Service (QoS)

QoS profiles control the behavior of publishers and subscribers:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# Create a QoS profile for reliable communication
qos_profile = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE
)

# Use the QoS profile when creating a publisher
publisher = self.create_publisher(String, 'topic_name', qos_profile)
```

## Working with Custom Messages

To use custom message types, you need to create them in a package and import them:

```python
# Assuming you have a custom message package called 'my_robot_msgs'
from my_robot_msgs.msg import JointState

class CustomMessageNode(Node):
    def __init__(self):
        super().__init__('custom_message_node')
        self.publisher = self.create_publisher(JointState, 'joint_states', 10)

    def publish_joint_state(self, positions):
        msg = JointState()
        msg.position = positions
        self.publisher.publish(msg)
```

## Error Handling and Logging

ROS 2 provides different logging levels:

```python
class LoggingNode(Node):
    def __init__(self):
        super().__init__('logging_node')

        # Different logging levels
        self.get_logger().debug('Debug message')
        self.get_logger().info('Info message')
        self.get_logger().warn('Warning message')
        self.get_logger().error('Error message')
        self.get_logger().fatal('Fatal message')
```

## Exercise

Create a simple rclpy node that:
1. Publishes joint position commands to control a humanoid robot
2. Subscribes to sensor feedback from the robot
3. Uses parameters to configure the control behavior
4. Implements proper error handling and logging