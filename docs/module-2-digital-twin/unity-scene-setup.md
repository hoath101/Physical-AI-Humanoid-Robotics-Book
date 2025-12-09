# Unity Scene Setup and Physics Configuration

Unity provides a powerful environment for creating high-fidelity digital twins with advanced graphics and physics capabilities. This section covers setting up Unity scenes for humanoid robotics simulation with proper physics configuration.

## Unity Project Setup for Robotics

### Initial Project Configuration
When creating a Unity project for robotics applications:

1. **Project Template**: Start with 3D template
2. **Target Platform**: Consider your deployment needs (PC, VR, mobile)
3. **Scripting Runtime**: Use .NET Standard 2.1 or higher
4. **API Compatibility**: .NET Framework for Windows applications

### Required Packages
Install these packages via the Unity Package Manager (Window → Package Manager):

```csharp
// Key packages for robotics simulation
- com.unity.robotics.ros-tcp-connector: ROS communication
- com.unity.robotics.urdf-importer: URDF model import
- com.unity.perception: Synthetic data generation
- com.unity.physics: Advanced physics simulation
- com.unity.xr: If using VR/AR
```

### Project Settings Configuration
Configure Project Settings for robotics simulation:

```csharp
// Edit → Project Settings → Time
- Fixed Timestep: 0.02 (50 Hz) for physics simulation
- Maximum Allowed Timestep: 0.333 (30 FPS cap)
- Maximum Particle Timestep: 0.03

// Edit → Project Settings → Physics
- Gravity: (0, -9.81, 0) for Earth-like gravity
- Default Material: Create custom material with appropriate friction
- Bounce Threshold: 2 (velocity threshold for bounce)
- Contact Offset: 0.01 (contact distance threshold)
- Solver Iterations: 6 (accuracy vs performance)
- Solver Velocity Iterations: 1 (velocity accuracy)
```

## Scene Architecture

### Basic Scene Structure
Create a well-organized scene hierarchy:

```
Scene Root
├── Environment
│   ├── Ground
│   ├── Walls
│   ├── Obstacles
│   └── Lighting
├── Robots
│   ├── HumanoidRobot
│   │   ├── BaseLink
│   │   ├── Links (joints and bodies)
│   │   └── Sensors
│   └── OtherRobots...
├── Controllers
│   ├── ROSConnection
│   ├── RobotController
│   └── SceneController
└── Managers
    ├── PhysicsManager
    ├── VisualizationManager
    └── CommunicationManager
```

### Environment Setup

#### Ground Plane
Create a realistic ground plane with proper physics properties:

```csharp
using UnityEngine;

public class GroundPlaneSetup : MonoBehaviour
{
    [Header("Ground Properties")]
    [SerializeField] private PhysicMaterial groundMaterial;
    [SerializeField] private float size = 100f;
    [SerializeField] private float friction = 0.5f;
    [SerializeField] private float bounciness = 0.1f;

    void Start()
    {
        // Create ground plane
        GameObject ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
        ground.transform.SetParent(transform);
        ground.transform.localPosition = Vector3.zero;
        ground.transform.localScale = Vector3.one * (size / 10f); // Plane is 10 units by default

        // Configure collider
        var collider = ground.GetComponent<Collider>();
        if (collider != null)
        {
            if (groundMaterial == null)
            {
                groundMaterial = new PhysicMaterial("GroundMaterial");
                groundMaterial.staticFriction = friction;
                groundMaterial.dynamicFriction = friction;
                groundMaterial.bounciness = bounciness;
            }
            collider.material = groundMaterial;
        }

        // Add visual material
        Renderer renderer = ground.GetComponent<Renderer>();
        if (renderer != null)
        {
            renderer.material = CreateGroundMaterial();
        }
    }

    Material CreateGroundMaterial()
    {
        Material groundMat = new Material(Shader.Find("Standard"));
        groundMat.color = Color.gray;
        groundMat.SetFloat("_Metallic", 0.1f);
        groundMat.SetFloat("_Smoothness", 0.2f);
        return groundMat;
    }
}
```

#### Lighting Configuration
Set up realistic lighting for the scene:

```csharp
using UnityEngine;

public class LightingSetup : MonoBehaviour
{
    [Header("Lighting Configuration")]
    [SerializeField] private LightType lightType = LightType.Directional;
    [SerializeField] private Color lightColor = Color.white;
    [SerializeField] private float intensity = 1.0f;
    [SerializeField] private Vector3 lightDirection = new Vector3(-0.5f, -1f, -0.5f);

    void Start()
    {
        // Create main light
        GameObject lightObj = new GameObject("Main Light");
        lightObj.transform.SetParent(transform);
        lightObj.transform.position = new Vector3(0, 10, 0);

        Light light = lightObj.AddComponent<Light>();
        light.type = lightType;
        light.color = lightColor;
        light.intensity = intensity;
        light.transform.rotation = Quaternion.LookRotation(lightDirection);

        // Add ambient lighting
        RenderSettings.ambientLight = new Color(0.2f, 0.2f, 0.2f, 1f);
        RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Trilight;
    }
}
```

## Physics Configuration

### Physics Manager Settings
Configure the Physics Manager for robotics simulation:

```csharp
// This would be configured through Edit → Project Settings → Physics
/*
Physics Settings for Robotics:
- Gravity: (0, -9.81, 0) m/s²
- Default Material: Custom material with realistic friction
- Bounce Threshold: 2 m/s
- Sleep Threshold: 0.005 (keep objects from sleeping too early for control)
- Default Contact Offset: 0.01
- Solver Iteration Count: 6-10 (balance accuracy and performance)
- Solver Velocity Iteration Count: 1
- Queries Hit Backfaces: False (for sensors)
- Queries Hit Triggers: True (for detection zones)
- Enable Adaptive Force: False (for consistent control)
- Layer Collision Matrix: Configure for robot self-collision if needed
*/
```

### Custom Physics Material
Create realistic physics materials for robot components:

```csharp
using UnityEngine;

[CreateAssetMenu(fileName = "RobotPhysicsMaterial", menuName = "Robotics/Physics Material")]
public class RobotPhysicsMaterial : ScriptableObject
{
    [Header("Friction Properties")]
    public float staticFriction = 0.5f;
    public float dynamicFriction = 0.4f;
    [Range(0, 1)] public float frictionCombine = 0; // 0=Average, 1=Minimum, 2=Maximum, 3=Multiply

    [Header("Bounce Properties")]
    [Range(0, 1)] public float bounciness = 0.1f;
    [Range(0, 1)] public float bounceCombine = 0; // Same options as frictionCombine

    [Header("Simulation Properties")]
    public bool enableAdaptiveForce = false;
    public float solverIterations = 6f;
    public float solverVelocityIterations = 1f;

    public PhysicMaterial CreatePhysicMaterial(string name = "RobotMaterial")
    {
        PhysicMaterial material = new PhysicMaterial(name);
        material.staticFriction = staticFriction;
        material.dynamicFriction = dynamicFriction;
        material.frictionCombine = GetCombineMode(frictionCombine);
        material.bounciness = bounciness;
        material.bounceCombine = GetCombineMode(bounceCombine);

        return material;
    }

    PhysicMaterialCombine GetCombineMode(float value)
    {
        if (value < 0.33f) return PhysicMaterialCombine.Average;
        if (value < 0.66f) return PhysicMaterialCombine.Minimum;
        if (value < 1.0f) return PhysicMaterialCombine.Maximum;
        return PhysicMaterialCombine.Multiply;
    }
}
```

## Robot Integration

### Robot Prefab Structure
Create a well-structured robot prefab:

```csharp
using UnityEngine;

public class RobotStructure : MonoBehaviour
{
    [Header("Robot Configuration")]
    public string robotName = "HumanoidRobot";
    public float massMultiplier = 1.0f;

    [Header("Joint Configuration")]
    public float defaultDamping = 0.1f;
    public float defaultSpring = 10000f;
    public float defaultMaxForce = 100f;

    [Header("Sensor Configuration")]
    public Transform[] sensorPoints;

    void Start()
    {
        ConfigureRobotPhysics();
        ConfigureJoints();
    }

    void ConfigureRobotPhysics()
    {
        // Apply consistent physics properties to all rigidbodies
        Rigidbody[] rigidbodies = GetComponentsInChildren<Rigidbody>();
        foreach (Rigidbody rb in rigidbodies)
        {
            rb.mass *= massMultiplier;
            rb.drag = 0.01f;
            rb.angularDrag = 0.05f;
            rb.sleepThreshold = 0.005f;
            rb.interpolation = RigidbodyInterpolation.Interpolate;
        }
    }

    void ConfigureJoints()
    {
        // Configure all joints with realistic properties
        ConfigurableJoint[] joints = GetComponentsInChildren<ConfigurableJoint>();
        foreach (ConfigurableJoint joint in joints)
        {
            ConfigureJoint(joint);
        }
    }

    void ConfigureJoint(ConfigurableJoint joint)
    {
        // Configure linear limits
        SoftJointLimit linearLimit = joint.linearLimit;
        linearLimit.limit = 0.0f; // Fixed position constraint
        joint.linearLimit = linearLimit;

        // Configure angular limits
        SoftJointLimit lowAngularXLimit = joint.lowAngularXLimit;
        lowAngularXLimit.limit = -45f * Mathf.Deg2Rad; // Convert to radians
        joint.lowAngularXLimit = lowAngularXLimit;

        SoftJointLimit highAngularXLimit = joint.highAngularXLimit;
        highAngularXLimit.limit = 45f * Mathf.Deg2Rad;
        joint.highAngularXLimit = highAngularXLimit;

        // Configure drive for control
        JointDrive angularXDrive = joint.angularXDrive;
        angularXDrive.mode = JointDriveMode.PositionAndVelocity;
        angularXDrive.positionSpring = defaultSpring;
        angularXDrive.positionDamper = defaultDamping;
        angularXDrive.maximumForce = defaultMaxForce;
        joint.angularXDrive = angularXDrive;
    }
}
```

### Sensor Integration
Integrate sensors with proper physics configuration:

```csharp
using UnityEngine;

public class RobotSensorSetup : MonoBehaviour
{
    [Header("Camera Sensors")]
    public Camera[] cameras;
    [SerializeField] private string cameraTopicPrefix = "/camera";

    [Header("LIDAR Sensors")]
    public Transform[] lidarPoints;
    [SerializeField] private string lidarTopicPrefix = "/lidar";

    [Header("IMU Sensors")]
    public Transform[] imuPoints;
    [SerializeField] private string imuTopicPrefix = "/imu";

    void Start()
    {
        ConfigureCameras();
        ConfigureLIDAR();
        ConfigureIMUs();
    }

    void ConfigureCameras()
    {
        for (int i = 0; i < cameras.Length; i++)
        {
            Camera cam = cameras[i];
            if (cam != null)
            {
                // Configure camera for realistic vision
                cam.allowMSAA = true;
                cam.allowDynamicResolution = true;

                // Add perception components if using Unity Perception package
                var syntheticCamera = cam.gameObject.AddComponent<Unity.Perception.GroundTruth.SyntheticCameraData>();
                syntheticCamera.sensorId = $"{cameraTopicPrefix}_{i}";
            }
        }
    }

    void ConfigureLIDAR()
    {
        // LIDAR typically uses raycasting, configure collision layers
        foreach (Transform lidarPoint in lidarPoints)
        {
            // Configure the lidar point's collision detection
            // Set up appropriate layers for raycasting
        }
    }

    void ConfigureIMUs()
    {
        // Configure IMU sensors with appropriate physics properties
        foreach (Transform imuPoint in imuPoints)
        {
            // Add IMU-specific components
            var imuSensor = imuPoint.gameObject.AddComponent<IMUMockup>();
            imuSensor.Configure(this, imuPoint.name);
        }
    }
}

// Mock IMU component for demonstration
public class IMUMockup : MonoBehaviour
{
    private RobotSensorSetup robot;
    private string sensorName;

    public void Configure(RobotSensorSetup robotSetup, string name)
    {
        robot = robotSetup;
        sensorName = name;
    }

    void Update()
    {
        // Simulate IMU data
        // This would typically interface with ROS through the TCP connector
    }
}
```

## Scene Management

### Scene Controller
Create a scene controller for managing simulation state:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;

public class SceneController : MonoBehaviour
{
    [Header("Simulation Control")]
    [SerializeField] private float simulationSpeed = 1.0f;
    [SerializeField] private bool useFixedTimeStep = true;
    [SerializeField] private float fixedTimeStep = 0.02f; // 50 Hz

    [Header("ROS Connection")]
    [SerializeField] private string rosIPAddress = "127.0.0.1";
    [SerializeField] private int rosPort = 10000;

    private ROSConnection rosConnection;
    private bool isSimulationRunning = false;

    void Start()
    {
        SetupROSLifecycle();
        ConfigureTimeStep();
    }

    void SetupROSLifecycle()
    {
        // Initialize ROS connection
        rosConnection = ROSConnection.GetOrCreateInstance();
        rosConnection.rosIPAddress = rosIPAddress;
        rosConnection.rosPort = rosPort;

        // Register shutdown callback
        Application.quitting += OnApplicationQuit;
    }

    void ConfigureTimeStep()
    {
        if (useFixedTimeStep)
        {
            Time.fixedDeltaTime = fixedTimeStep;
        }
    }

    void OnApplicationQuit()
    {
        // Clean up ROS connection
        if (rosConnection != null)
        {
            rosConnection.Close();
        }
    }

    public void StartSimulation()
    {
        isSimulationRunning = true;
        // Publish simulation start message to ROS
    }

    public void StopSimulation()
    {
        isSimulationRunning = false;
        // Publish simulation stop message to ROS
    }

    public void ResetSimulation()
    {
        // Reset robot positions, clear data, etc.
        ResetRobots();
    }

    void ResetRobots()
    {
        RobotStructure[] robots = FindObjectsOfType<RobotStructure>();
        foreach (RobotStructure robot in robots)
        {
            // Reset robot to initial configuration
            ResetRobotToInitialPosition(robot);
        }
    }

    void ResetRobotToInitialPosition(RobotStructure robot)
    {
        // Store initial positions and reset to them
        Transform robotTransform = robot.transform;
        robotTransform.position = robotTransform.position; // Or use stored initial position
        robotTransform.rotation = robotTransform.rotation; // Or use stored initial rotation

        // Reset all joint positions
        ConfigurableJoint[] joints = robot.GetComponentsInChildren<ConfigurableJoint>();
        foreach (ConfigurableJoint joint in joints)
        {
            // Reset joint to initial configuration
        }
    }
}
```

## Performance Optimization

### Physics Optimization
Configure physics for optimal performance:

```csharp
using UnityEngine;

public class PhysicsOptimizer : MonoBehaviour
{
    [Header("Performance Settings")]
    [SerializeField] private int solverIterationCount = 6;
    [SerializeField] private int solverVelocityIterationCount = 1;
    [SerializeField] private float contactOffset = 0.01f;
    [SerializeField] private float sleepThreshold = 0.005f;

    [Header("Simulation Quality")]
    [SerializeField] private bool enableAdaptiveForce = false;
    [SerializeField] private bool enableContinuousCollisionDetection = false;
    [SerializeField] private CollisionDetectionMode collisionDetectionMode = CollisionDetectionMode.Discrete;

    void Start()
    {
        OptimizePhysicsSettings();
    }

    void OptimizePhysicsSettings()
    {
        Physics.defaultSolverIterations = solverIterationCount;
        Physics.defaultSolverVelocityIterations = solverVelocityIterationCount;
        Physics.defaultContactOffset = contactOffset;
        Physics.sleepThreshold = sleepThreshold;
        Physics.defaultUseAdaptiveForce = enableAdaptiveForce;
        Physics.defaultSolverContactOffset = contactOffset;

        // Apply settings to all rigidbodies in the scene
        Rigidbody[] allRigidbodies = FindObjectsOfType<Rigidbody>();
        foreach (Rigidbody rb in allRigidbodies)
        {
            rb.collisionDetectionMode = enableContinuousCollisionDetection ?
                CollisionDetectionMode.Continuous : collisionDetectionMode;
        }
    }
}
```

### Level of Detail (LOD) for Complex Robots
Implement LOD for complex robot models:

```csharp
using UnityEngine;

public class RobotLODManager : MonoBehaviour
{
    [System.Serializable]
    public class LODLevel
    {
        public string name;
        public float distance;
        public GameObject[] objects;
    }

    [SerializeField] private LODLevel[] lodLevels;
    [SerializeField] private Transform cameraTransform;

    private int currentLOD = 0;

    void Start()
    {
        if (cameraTransform == null)
        {
            cameraTransform = Camera.main.transform;
        }
    }

    void Update()
    {
        UpdateLOD();
    }

    void UpdateLOD()
    {
        float distance = Vector3.Distance(transform.position, cameraTransform.position);

        int newLOD = 0;
        for (int i = 0; i < lodLevels.Length; i++)
        {
            if (distance > lodLevels[i].distance)
            {
                newLOD = i;
            }
            else
            {
                break;
            }
        }

        if (newLOD != currentLOD)
        {
            SetLOD(newLOD);
            currentLOD = newLOD;
        }
    }

    void SetLOD(int lodIndex)
    {
        for (int i = 0; i < lodLevels.Length; i++)
        {
            bool isActive = i == lodIndex;
            foreach (GameObject obj in lodLevels[i].objects)
            {
                if (obj != null)
                {
                    obj.SetActive(isActive);
                }
            }
        }
    }
}
```

## Debugging and Visualization

### Physics Debugging Tools
Create tools for debugging physics simulation:

```csharp
using UnityEngine;

public class PhysicsDebugger : MonoBehaviour
{
    [Header("Debug Settings")]
    [SerializeField] private bool showColliders = true;
    [SerializeField] private bool showJoints = true;
    [SerializeField] private bool showForces = true;
    [SerializeField] private Color colliderColor = Color.blue;
    [SerializeField] private Color jointColor = Color.green;

    void OnDrawGizmos()
    {
        if (showColliders)
        {
            DrawColliders();
        }

        if (showJoints)
        {
            DrawJoints();
        }

        if (showForces)
        {
            DrawForces();
        }
    }

    void DrawColliders()
    {
        Collider[] colliders = GetComponentsInChildren<Collider>();
        foreach (Collider col in colliders)
        {
            Gizmos.color = colliderColor;
            if (col is BoxCollider)
            {
                BoxCollider box = (BoxCollider)col;
                Gizmos.matrix = Matrix4x4.TRS(box.transform.position, box.transform.rotation, Vector3.one);
                Gizmos.DrawWireCube(box.center, box.size);
            }
            else if (col is SphereCollider)
            {
                SphereCollider sphere = (SphereCollider)col;
                Gizmos.matrix = Matrix4x4.TRS(box.transform.position, box.transform.rotation, Vector3.one);
                Gizmos.DrawWireSphere(sphere.center, sphere.radius);
            }
            // Add other collider types as needed
        }
    }

    void DrawJoints()
    {
        ConfigurableJoint[] joints = GetComponentsInChildren<ConfigurableJoint>();
        foreach (ConfigurableJoint joint in joints)
        {
            Gizmos.color = jointColor;
            Gizmos.DrawWireSphere(joint.transform.position, 0.1f);

            if (joint.connectedBody != null)
            {
                Gizmos.DrawLine(joint.transform.position, joint.connectedBody.transform.position);
            }
        }
    }

    void DrawForces()
    {
        Rigidbody[] rigidbodies = GetComponentsInChildren<Rigidbody>();
        foreach (Rigidbody rb in rigidbodies)
        {
            Vector3 force = rb.velocity * rb.mass; // Approximate force
            Gizmos.color = Color.red;
            Gizmos.DrawRay(rb.position, force.normalized * 0.5f);
        }
    }
}
```

## Best Practices

### 1. Consistent Units
Always use SI units (meters, kilograms, seconds) to match ROS conventions:

```csharp
// Configuration constants using SI units
public static class RobotConstants
{
    public const float Gravity = 9.81f; // m/s²
    public const float MassMultiplier = 1.0f; // kg
    public const float LengthScale = 1.0f; // meters
    public const float VelocityScale = 1.0f; // m/s
    public const float TorqueScale = 1.0f; // N⋅m
}
```

### 2. Modularity
Create modular components that can be reused:

```csharp
// Generic sensor base class
public abstract class RobotSensor : MonoBehaviour
{
    [SerializeField] protected string topicName;
    [SerializeField] protected string frameId;
    [SerializeField] protected float updateRate = 30f; // Hz

    protected float updateInterval;
    protected float lastUpdateTime;

    protected virtual void Start()
    {
        updateInterval = 1.0f / updateRate;
        lastUpdateTime = 0;
    }

    protected virtual void Update()
    {
        if (Time.time - lastUpdateTime >= updateInterval)
        {
            UpdateSensorData();
            lastUpdateTime = Time.time;
        }
    }

    protected abstract void UpdateSensorData();
}
```

### 3. Error Handling
Implement robust error handling for ROS connections:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;

public class RobustROSConnection : MonoBehaviour
{
    [SerializeField] private string rosIPAddress = "127.0.0.1";
    [SerializeField] private int rosPort = 10000;
    [SerializeField] private float connectionRetryDelay = 5f;

    private ROSConnection rosConnection;
    private bool isConnected = false;
    private float lastConnectionAttempt = 0f;

    void Start()
    {
        InitializeROSConnection();
    }

    void InitializeROSConnection()
    {
        try
        {
            rosConnection = ROSConnection.GetOrCreateInstance();
            rosConnection.rosIPAddress = rosIPAddress;
            rosConnection.rosPort = rosPort;
            rosConnection.OnConnected += OnConnected;
            rosConnection.OnDisconnected += OnDisconnected;
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Failed to initialize ROS connection: {e.Message}");
        }
    }

    void OnConnected()
    {
        isConnected = true;
        Debug.Log("Successfully connected to ROS");
        OnROSConnected();
    }

    void OnDisconnected()
    {
        isConnected = false;
        Debug.LogWarning("Disconnected from ROS");
        AttemptReconnection();
    }

    void AttemptReconnection()
    {
        if (Time.time - lastConnectionAttempt > connectionRetryDelay)
        {
            lastConnectionAttempt = Time.time;
            InitializeROSConnection();
        }
    }

    protected virtual void OnROSConnected()
    {
        // Override in derived classes
    }
}
```

## Exercise

Create a complete Unity scene for humanoid robot simulation that includes:
1. A well-structured scene hierarchy with proper organization
2. Physics configuration optimized for humanoid robot simulation
3. A robot prefab with realistic physical properties
4. Proper lighting and environment setup
5. Sensor configurations for cameras, LIDAR, and IMU
6. Performance optimization techniques
7. Debugging visualization tools

Test your scene by ensuring the robot behaves realistically with proper physics interactions and can communicate with ROS systems.