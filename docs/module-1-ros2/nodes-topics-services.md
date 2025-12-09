# Nodes, Topics, Services

In this section, we'll explore the fundamental communication patterns in ROS 2: nodes, topics, and services. These concepts form the backbone of robotic applications and enable different components to work together seamlessly.

## Nodes

A node is an executable process that works as part of a ROS 2 system. Nodes are the basic building blocks of a ROS 2 application and perform specific tasks. In a humanoid robot system, you might have nodes for:

- Sensor processing
- Motor control
- Path planning
- Perception
- Behavior management

### Creating a Node

Here's a basic example of a ROS 2 node in Python:

```python
import rclpy
from rclpy.node import Node

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Topics

Topics enable asynchronous communication between nodes through a publish/subscribe model. Nodes can publish messages to topics and subscribe to topics to receive messages. This pattern is ideal for continuous data streams like sensor readings or motor commands.

### Topic Communication Example

```python
# Publisher node
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Talker(Node):

    def __init__(self):
        super().__init__('talker')
        self.publisher = self.create_publisher(String, 'chatter', 10)
        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World'
        self.publisher.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)

# Subscriber node
class Listener(Node):

    def __init__(self):
        super().__init__('listener')
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)
```

## Services

Services provide synchronous request/response communication between nodes. Unlike topics, services are used for specific requests that require a response, such as requesting a specific action or configuration change.

### Service Example

```python
# Service server
from example_interfaces.srv import AddTwoInts

class AddTwoIntsServer(Node):

    def __init__(self):
        super().__init__('add_two_ints_server')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))
        return response

# Service client
class AddTwoIntsClient(Node):

    def __init__(self):
        super().__init__('add_two_ints_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()
```

## Actions

Actions are used for long-running tasks that may take time to complete and provide feedback during execution. They're particularly useful for tasks like navigation or manipulation that require ongoing communication about progress.

## Best Practices

- Use topics for continuous data streams (sensor data, motor commands)
- Use services for specific requests that need immediate responses
- Use actions for long-running tasks that require feedback
- Keep message types consistent across your system
- Use appropriate Quality of Service (QoS) settings for your use case

## Exercise

Create a simple ROS 2 package with a publisher node that publishes joint positions for a humanoid robot and a subscriber node that logs these positions to the console.