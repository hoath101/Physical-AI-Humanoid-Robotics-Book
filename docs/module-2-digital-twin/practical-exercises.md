# Practical Exercises - Digital Twin Simulation

This section provides hands-on exercises to reinforce the concepts learned in the Digital Twin module. Complete these exercises to gain practical experience with Gazebo and Unity simulation environments.

## Exercise 1: Basic Robot in Gazebo

### Objective
Create a simple robot model and simulate it in Gazebo with basic movement capabilities.

### Steps
1. Create a URDF file for a simple wheeled robot with:
   - A base link
   - Two wheels as child links
   - Revolute joints connecting wheels to base
   - Proper inertial properties for each link

2. Launch the robot in Gazebo:
   ```bash
   # Create a launch file to spawn your robot
   ros2 launch your_robot_description spawn_robot.launch.py
   ```

3. Control the robot using ROS 2 topics:
   ```bash
   # Send velocity commands to move the robot
   ros2 topic pub /cmd_vel geometry_msgs/Twist "{linear: {x: 0.5}, angular: {z: 0.2}}"
   ```

### Expected Outcome
A robot that moves forward and turns in the Gazebo simulation environment when velocity commands are sent.

### Solution Template
```xml
<!-- simple_robot.urdf -->
<?xml version="1.0"?>
<robot name="simple_robot">
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.2" radius="0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.2" radius="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.2"/>
    </inertial>
  </link>

  <joint name="wheel_left_joint" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_left"/>
    <origin xyz="-0.1 0.2 0" rpy="1.5708 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <link name="wheel_left">
    <visual>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <joint name="wheel_right_joint" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_right"/>
    <origin xyz="-0.1 -0.2 0" rpy="1.5708 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <link name="wheel_right">
    <visual>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
      <material name="black"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>
</robot>
```

## Exercise 2: Sensor Integration in Gazebo

### Objective
Add a camera sensor to your robot and visualize the sensor data.

### Steps
1. Add a camera sensor to your robot's URDF:
   ```xml
   <joint name="camera_joint" type="fixed">
     <parent link="base_link"/>
     <child link="camera_link"/>
     <origin xyz="0.2 0 0.1" rpy="0 0 0"/>
   </joint>

   <link name="camera_link">
     <visual>
       <geometry>
         <box size="0.05 0.05 0.05"/>
       </geometry>
     </visual>
     <collision>
       <geometry>
         <box size="0.05 0.05 0.05"/>
       </geometry>
     </collision>
   </link>
   ```

2. Add the camera sensor with Gazebo plugin:
   ```xml
   <gazebo reference="camera_link">
     <sensor name="camera" type="camera">
       <update_rate>30</update_rate>
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

3. Visualize the camera feed:
   ```bash
   # Install image tools if not already installed
   sudo apt install ros-humble-image-view

   # View the camera feed
   ros2 run image_view image_view __ns:=/your_robot_namespace
   ```

### Expected Outcome
A robot with a camera that publishes image data to a ROS topic, viewable in an image viewer.

## Exercise 3: Unity Robot Setup

### Objective
Import a robot model into Unity and set up basic physics properties.

### Steps
1. Install the URDF Importer package in Unity:
   - Window → Package Manager
   - Click the + button → Add package from git URL
   - Enter: `com.unity.robotics.urdf-importer`

2. Import your URDF file:
   - Assets → Import Robot from URDF
   - Select your URDF file
   - Configure import settings (scale, materials, etc.)

3. Add physics properties to imported robot:
   ```csharp
   using UnityEngine;

   public class UnityRobotSetup : MonoBehaviour
   {
       [Header("Physics Configuration")]
       [SerializeField] private float massMultiplier = 1.0f;
       [SerializeField] private PhysicMaterial robotMaterial;

       void Start()
       {
           ConfigureRobotPhysics();
       }

       void ConfigureRobotPhysics()
       {
           // Add Rigidbody to each link
           var links = GetComponentsInChildren<Transform>();
           foreach (Transform link in links)
           {
               if (link.TryGetComponent(out MeshCollider collider))
               {
                   // Add Rigidbody if not present
                   if (!link.GetComponent<Rigidbody>())
                   {
                       Rigidbody rb = link.gameObject.AddComponent<Rigidbody>();
                       rb.mass = CalculateMass(link.name) * massMultiplier;
                       rb.drag = 0.1f;
                       rb.angularDrag = 0.1f;
                       rb.interpolation = RigidbodyInterpolation.Interpolate;
                   }

                   // Apply material
                   if (robotMaterial != null)
                   {
                       collider.material = robotMaterial;
                   }
               }
           }
       }

       float CalculateMass(string linkName)
       {
           // Return approximate mass based on link name
           switch (linkName.ToLower())
           {
               case "base_link":
                   return 10.0f;
               case "wheel_left":
               case "wheel_right":
                   return 1.0f;
               default:
                   return 2.0f;
           }
       }
   }
   ```

4. Test the robot in Unity physics simulation.

### Expected Outcome
A robot imported from URDF with proper physics properties and realistic behavior in Unity's physics engine.

## Exercise 4: Unity Sensor Simulation

### Objective
Create a simulated camera sensor in Unity that publishes data to ROS.

### Steps
1. Create a camera sensor script:
   ```csharp
   using UnityEngine;
   using Unity.Robotics.ROSTCPConnector;
   using RosMessageTypes.Sensor;
   using Unity.Robotics.ROSTCPConnector.MessageGeneration;

   public class UnityCameraSensor : MonoBehaviour
   {
       [Header("Camera Configuration")]
       [SerializeField] private int width = 640;
       [SerializeField] private int height = 480;
       [SerializeField] private float fieldOfView = 60f;
       [SerializeField] private string topicName = "/unity_camera/image_raw";

       private Camera cam;
       private RenderTexture renderTexture;
       private ROSConnection ros;
       private Texture2D texture2D;

       void Start()
       {
           ros = ROSConnection.GetOrCreateInstance();

           // Set up camera
           cam = GetComponent<Camera>();
           if (cam == null)
           {
               cam = gameObject.AddComponent<Camera>();
           }
           cam.fieldOfView = fieldOfView;

           // Create render texture
           renderTexture = new RenderTexture(width, height, 24);
           cam.targetTexture = renderTexture;

           // Create texture for reading pixels
           texture2D = new Texture2D(width, height, TextureFormat.RGB24, false);
       }

       void Update()
       {
           // Capture image and publish to ROS
           PublishCameraImage();
       }

       void PublishCameraImage()
       {
           // Set the active render texture and read pixels
           RenderTexture.active = renderTexture;
           texture2D.ReadPixels(new Rect(0, 0, width, height), 0, 0);
           texture2D.Apply();

           // Convert texture to byte array (simplified - in practice you'd handle image encoding)
           byte[] imageData = texture2D.EncodeToPNG();

           // Create and publish ROS message
           var imageMsg = new Sensor_msgs.ImageMsg();
           imageMsg.header = new Std_msgs.HeaderMsg();
           imageMsg.header.frame_id = transform.name;
           imageMsg.header.stamp = new Builtin_interfaces.TimeMsg();
           imageMsg.height = (uint)height;
           imageMsg.width = (uint)width;
           imageMsg.encoding = "rgb8";
           imageMsg.is_bigendian = 0;
           imageMsg.step = (uint)(width * 3); // 3 bytes per pixel for RGB
           imageMsg.data = imageData;

           ros.Publish(topicName, imageMsg);
       }
   }
   ```

2. Attach the script to a camera in your Unity scene
3. Configure the ROS connection settings
4. Test that the camera publishes images to ROS

### Expected Outcome
A Unity camera that captures images and publishes them to a ROS topic for use in robotics applications.

## Exercise 5: Digital Twin Integration

### Objective
Create a complete digital twin scenario with both Gazebo and Unity representations of the same robot.

### Steps
1. Create a humanoid robot URDF with multiple joints and sensors
2. Set up a Gazebo world with the robot and obstacles
3. Create a Unity scene with the same robot model
4. Implement ROS communication to synchronize state between both simulations
5. Create a control system that works in both environments

### Unity Synchronization Script
```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;

public class RobotStateSynchronizer : MonoBehaviour
{
    [Header("ROS Configuration")]
    [SerializeField] private string jointStateTopic = "/joint_states";
    [SerializeField] private string cmdVelTopic = "/cmd_vel";

    private ROSConnection ros;
    private ArticulationBody[] joints;
    private string[] jointNames;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.Subscribe<JointStateMsg>(jointStateTopic, JointStateCallback);

        // Find all articulation bodies (Unity's joint equivalent)
        joints = GetComponentsInChildren<ArticulationBody>();
        jointNames = new string[joints.Length];

        for (int i = 0; i < joints.Length; i++)
        {
            jointNames[i] = joints[i].name;
        }
    }

    void JointStateCallback(JointStateMsg jointState)
    {
        for (int i = 0; i < jointState.name.Length; i++)
        {
            int jointIndex = System.Array.IndexOf(jointNames, jointState.name[i]);
            if (jointIndex >= 0 && jointIndex < joints.Length)
            {
                // Update joint position in Unity
                ArticulationDrive drive = joints[jointIndex].jointDrive;
                drive.target = jointState.position[i] * Mathf.Rad2Deg; // Convert radians to degrees
                joints[jointIndex].jointDrive = drive;
            }
        }
    }
}
```

### Expected Outcome
A synchronized digital twin where robot movements in Gazebo are reflected in Unity and vice versa, with both simulations representing the same physical system.

## Exercise 6: Performance Optimization Challenge

### Objective
Optimize your simulation for real-time performance while maintaining accuracy.

### Steps
1. Profile your simulation using Unity's Profiler or Gazebo's built-in tools
2. Identify performance bottlenecks (physics, rendering, ROS communication)
3. Implement optimization techniques:
   - Reduce polygon count for visual meshes
   - Use simpler collision geometries
   - Adjust physics solver settings
   - Optimize ROS message frequency
   - Implement level-of-detail (LOD) systems

### Expected Outcome
A simulation that runs at real-time speed (30+ FPS) while maintaining the necessary accuracy for robotics applications.

## Troubleshooting Tips

### Gazebo Issues
- If the robot falls through the ground: Check collision geometry and mass properties
- If joints are unstable: Increase physics solver iterations or adjust joint damping
- If simulation is slow: Simplify collision meshes or reduce physics update rate

### Unity Issues
- If physics are unstable: Check mass ratios and adjust solver iterations
- If ROS connection fails: Verify IP address and port settings
- If sensors don't work: Check layer masks and collision settings

## Extension Challenges

1. **Advanced Navigation**: Implement path planning and obstacle avoidance in both simulators
2. **Multi-Robot Simulation**: Create a scenario with multiple robots coordinating
3. **Dynamic Environments**: Add moving obstacles or changing environmental conditions
4. **Machine Learning Integration**: Use the simulation for training RL agents
5. **Haptic Feedback**: Implement force feedback for human-in-the-loop simulation

Complete these exercises to solidify your understanding of digital twin simulation before moving to the next module.