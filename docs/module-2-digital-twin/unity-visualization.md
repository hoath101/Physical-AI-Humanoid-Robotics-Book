# Unity Visualization

Unity provides a powerful game engine environment for high-fidelity robot simulation and visualization. While Gazebo is the standard for ROS simulation, Unity offers advanced graphics capabilities and a rich ecosystem for creating detailed digital twins.

## Introduction to Unity for Robotics

Unity for Robotics provides:
- High-fidelity visual rendering
- Advanced physics simulation
- Realistic lighting and materials
- VR/AR support for immersive development
- Large asset store with pre-built components
- Cross-platform deployment capabilities

## Unity Robotics Setup

### Installation
1. Download Unity Hub from the Unity website
2. Install Unity 2022.3 LTS (Long Term Support) version
3. Install the Unity Robotics packages through the Package Manager

### Required Packages
- **Unity Robotics Hub**: Central access point for robotics tools
- **Unity Robotics Package**: Core ROS integration
- **Unity Perception Package**: Synthetic data generation
- **Unity Simulation Package**: Cloud-based simulation

## ROS-TCP-Connector

Unity communicates with ROS 2 through the ROS-TCP-Connector:

### Installation
```bash
# Clone the ROS-TCP-Connector repository
git clone https://github.com/Unity-Technologies/ROS-TCP-Connector.git
```

### Setup in Unity
1. Create a new Unity project
2. Import the ROS-TCP-Connector package
3. Add the ROSConnection prefab to your scene
4. Configure the IP address and port to match your ROS 2 setup

## Basic Unity Robot Integration

### Robot Model Setup
```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;

public class RobotController : MonoBehaviour
{
    [SerializeField]
    private string topicName = "/joint_commands";

    private ROSConnection ros;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<JointStateMsg>(topicName);
    }

    void Update()
    {
        // Send joint state messages to ROS
        var jointState = new JointStateMsg();
        jointState.name = new string[] { "joint1", "joint2", "joint3" };
        jointState.position = new double[] { 0.1, 0.2, 0.3 };

        ros.Publish(topicName, jointState);
    }
}
```

### Receiving ROS Messages
```csharp
using Unity.Robotics.ROSTCPConnector.ROSGeometry;
using RosMessageTypes.Std;

public class RobotReceiver : MonoBehaviour
{
    [SerializeField]
    private string topicName = "/joint_states";

    void Start()
    {
        ROSConnection.GetOrCreateInstance().Subscribe<JointStateMsg>(topicName, JointStateCallback);
    }

    void JointStateCallback(JointStateMsg jointState)
    {
        // Update robot joints based on received state
        for (int i = 0; i < jointState.name.Length; i++)
        {
            UpdateJoint(jointState.name[i], jointState.position[i]);
        }
    }

    void UpdateJoint(string jointName, double position)
    {
        // Apply position to corresponding joint in Unity
        Transform joint = transform.Find(jointName);
        if (joint != null)
        {
            joint.localRotation = Quaternion.Euler(0, (float)position * Mathf.Rad2Deg, 0);
        }
    }
}
```

## NVIDIA Isaac Integration

Unity works closely with NVIDIA Isaac for advanced robotics simulation:

### Isaac Unity Plugin
- **Isaac Unity Robotics Package**: Direct integration with Isaac Sim
- **Synthetic Data Generation**: Create training data for AI models
- **Photorealistic Simulation**: High-quality rendering for computer vision

### Setting up Isaac Sim with Unity
1. Install NVIDIA Isaac Sim
2. Configure Unity to work with Isaac Sim
3. Use Omniverse for collaborative simulation

## Creating a Robot in Unity

### 1. Import Robot Model
- Import your URDF as an FBX file
- Or create the robot model directly in Unity
- Set up the kinematic hierarchy with proper joint connections

### 2. Physics Configuration
```csharp
// Configure Rigidbody for each link
public class RobotLink : MonoBehaviour
{
    [Header("Physics Properties")]
    public float mass = 1.0f;
    public Vector3 centerOfMass = Vector3.zero;

    void Start()
    {
        Rigidbody rb = GetComponent<Rigidbody>();
        if (rb != null)
        {
            rb.mass = mass;
            rb.centerOfMass = centerOfMass;
        }
    }
}
```

### 3. Joint Configuration
Unity supports various joint types:
- **Hinge Joint**: Rotational movement around one axis
- **Configurable Joint**: Custom joint with multiple degrees of freedom
- **Fixed Joint**: Rigid connection between bodies

## Sensor Simulation in Unity

### Camera Simulation
```csharp
using Unity.Robotics.Sensors;

public class UnityCamera : MonoBehaviour
{
    [SerializeField] private int width = 640;
    [SerializeField] private int height = 480;
    [SerializeField] private float fov = 60f;

    private UnityCameraSensor cameraSensor;

    void Start()
    {
        cameraSensor = new UnityCameraSensor(
            transform,
            width, height, fov,
            "camera_topic",
            "camera_frame"
        );
    }

    void Update()
    {
        cameraSensor.PublishCameraImage();
    }
}
```

### LIDAR Simulation
Unity can simulate LIDAR using raycasting:

```csharp
using UnityEngine;

public class UnityLidar : MonoBehaviour
{
    [SerializeField] private int numRays = 360;
    [SerializeField] private float maxDistance = 10f;
    [SerializeField] private string topicName = "/scan";

    private float[] ranges;

    void Start()
    {
        ranges = new float[numRays];
    }

    void Update()
    {
        for (int i = 0; i < numRays; i++)
        {
            float angle = (i * 360f / numRays) * Mathf.Deg2Rad;
            Vector3 direction = new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle));

            if (Physics.Raycast(transform.position, direction, out RaycastHit hit, maxDistance))
            {
                ranges[i] = hit.distance;
            }
            else
            {
                ranges[i] = maxDistance;
            }
        }

        // Publish ranges to ROS
        PublishScan(ranges);
    }

    void PublishScan(float[] ranges)
    {
        // Implementation to publish LaserScan message to ROS
    }
}
```

## Unity Package Manager Setup

### Adding Robotics Packages
1. Open Unity Package Manager (Window → Package Manager)
2. Add packages by name or Git URL:
   - `com.unity.robotics.ros-tcp-connector`
   - `com.unity.robotics.urdf-importer`
   - `com.unity.perception`

### URDF Importer
The URDF Importer package allows direct import of ROS URDF files:
```csharp
// Import URDF through the UI or script
using Unity.Robotics.UrdfImporter;

public class UrdfLoader : MonoBehaviour
{
    public string urdfPath;

    void Start()
    {
        // Load URDF file and create Unity robot
        GameObject robot = UrdfRobotExtensions.Create(urdfPath);
        robot.transform.SetParent(transform);
    }
}
```

## Simulation Scenarios

### Indoor Navigation
Create realistic indoor environments:
- Office buildings with furniture
- Warehouses with obstacles
- Multi-floor structures with elevators

### Outdoor Environments
- Urban environments with traffic
- Natural terrains with vegetation
- Weather effects and lighting conditions

### Multi-Robot Simulation
Unity supports multiple robots in the same scene:
- Coordinate multiple robot behaviors
- Implement communication between robots
- Simulate robot swarms or teams

## Performance Optimization

### Rendering Optimization
- Use Level of Detail (LOD) for complex models
- Implement occlusion culling for large environments
- Use occlusion areas to optimize rendering

### Physics Optimization
- Adjust fixed timestep for physics simulation
- Use appropriate collision detection methods
- Optimize joint configurations for performance

## Debugging and Visualization

### Scene View Tools
- Use Gizmos to visualize transforms and bounds
- Enable physics visualization in Scene view
- Use the Profiler to identify performance bottlenecks

### ROS Integration Debugging
- Monitor ROS topics in real-time
- Visualize TF transforms in Unity
- Log messages between Unity and ROS

## Best Practices

- Start with simple scenes and gradually add complexity
- Use prefabs for reusable robot components
- Implement proper error handling for ROS connections
- Optimize assets for real-time performance
- Use version control for Unity projects (Git LFS for large files)

## Exercise

Create a Unity scene with your humanoid robot model imported from URDF. Implement basic joint control through ROS 2 communication and set up a camera sensor to publish images to ROS. Create a simple environment with obstacles and implement a basic navigation task.

## Resources

- [Unity Robotics Hub Documentation](https://github.com/Unity-Technologies/Unity-Robotics-Hub)
- [ROS-TCP-Connector](https://github.com/Unity-Technologies/ROS-TCP-Connector)
- [Unity URDF Importer](https://github.com/Unity-Technologies/URDF-Importer)
- [NVIDIA Isaac Sim Documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)