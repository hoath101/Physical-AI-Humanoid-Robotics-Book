# Sensor Simulation

Sensor simulation is a critical component of digital twin technology, enabling robots to perceive their virtual environment just as they would in the real world. This section covers simulating various types of sensors in both Gazebo and Unity environments.

## Overview of Sensor Simulation

Robotic sensors provide the robot with information about its environment and internal state. In simulation, we must accurately model:
- **Physical sensing**: How the sensor would detect real-world phenomena
- **Noise characteristics**: Real sensors have inherent noise and uncertainty
- **Latency**: Processing delays in real sensor systems
- **Field of view**: Physical limitations of sensor coverage
- **Range limitations**: Maximum and minimum detection distances

## Types of Sensors

### Range Sensors
- **LIDAR**: 2D/3D laser range finders
- **Ultrasonic**: Sound-based distance measurement
- **Infrared**: Infrared-based proximity detection

### Vision Sensors
- **Cameras**: RGB, depth, thermal imaging
- **Stereo cameras**: 3D vision capabilities
- **Event cameras**: High-speed dynamic vision

### Inertial Sensors
- **IMU**: Inertial measurement units
- **Accelerometers**: Linear acceleration
- **Gyroscopes**: Angular velocity
- **Magnetometers**: Magnetic field detection

### Force/Torque Sensors
- **Force sensors**: Linear forces
- **Torque sensors**: Rotational forces
- **FT sensors**: Combined force/torque measurement

## Sensor Simulation in Gazebo

### Camera Simulation
Gazebo provides realistic camera simulation through the gazebo_ros_camera plugin:

```xml
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

<gazebo reference="camera_link">
  <sensor name="camera" type="camera">
    <update_rate>30</update_rate>
    <camera name="head">
      <horizontal_fov>1.047</horizontal_fov> <!-- 60 degrees -->
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.007</stddev>
      </noise>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>camera_link</frame_name>
      <topic_name>image_raw</topic_name>
      <camera_info_topic_name>camera_info</camera_info_topic_name>
    </plugin>
  </sensor>
</gazebo>
```

### LIDAR Simulation
Simulate 2D and 3D LIDAR sensors:

```xml
<gazebo reference="lidar_link">
  <sensor name="lidar" type="ray">
    <update_rate>10</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>720</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle> <!-- -π radians -->
          <max_angle>3.14159</max_angle>   <!-- π radians -->
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

### Depth Camera Simulation
```xml
<gazebo reference="depth_camera_link">
  <sensor name="depth_camera" type="depth">
    <update_rate>30</update_rate>
    <camera name="depth_cam">
      <horizontal_fov>1.047</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>10</far>
      </clip>
    </camera>
    <plugin name="depth_camera_controller" filename="libgazebo_ros_openni_kinect.so">
      <baseline>0.2</baseline>
      <alwaysOn>true</alwaysOn>
      <updateRate>10.0</updateRate>
      <cameraName>depth_camera</cameraName>
      <imageTopicName>/rgb/image_raw</imageTopicName>
      <depthImageTopicName>/depth/image_raw</depthImageTopicName>
      <pointCloudTopicName>/depth/points</pointCloudTopicName>
      <cameraInfoTopicName>/rgb/camera_info</cameraInfoTopicName>
      <depthImageCameraInfoTopicName>/depth/camera_info</depthImageCameraInfoTopicName>
      <frameName>depth_camera_optical_frame</frameName>
      <baseline>0.1</baseline>
      <distortion_k1>0.0</distortion_k1>
      <distortion_k2>0.0</distortion_k2>
      <distortion_k3>0.0</distortion_k3>
      <distortion_t1>0.0</distortion_t1>
      <distortion_t2>0.0</distortion_t2>
      <pointCloudCutoff>0.5</pointCloudCutoff>
      <pointCloudCutoffMax>3.0</pointCloudCutoffMax>
    </plugin>
  </sensor>
</gazebo>
```

### IMU Simulation
```xml
<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
            <bias_mean>0.0000075</bias_mean>
            <bias_stddev>0.0000008</bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
            <bias_mean>0.0000075</bias_mean>
            <bias_stddev>0.0000008</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
            <bias_mean>0.0000075</bias_mean>
            <bias_stddev>0.0000008</bias_stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.1</bias_mean>
            <bias_stddev>0.001</bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.1</bias_mean>
            <bias_stddev>0.001</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.1</bias_mean>
            <bias_stddev>0.001</bias_stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
  </sensor>
</gazebo>
```

## Sensor Simulation in Unity

### Camera Simulation with Perception Package
Unity's Perception package provides advanced camera simulation:

```csharp
using UnityEngine;
using Unity.Perception.GroundTruth;
using Unity.Simulation;

public class CameraSensorSetup : MonoBehaviour
{
    [Header("Camera Properties")]
    [SerializeField] private int width = 640;
    [SerializeField] private int height = 480;
    [SerializeField] private float fieldOfView = 60f;
    [SerializeField] private float nearClip = 0.1f;
    [SerializeField] private float farClip = 100f;

    [Header("Sensor Properties")]
    [SerializeField] private string sensorId = "camera_0";
    [SerializeField] private string rosTopic = "/camera/image_raw";

    private Camera cam;
    private SyntheticCameraData syntheticCamera;

    void Start()
    {
        cam = GetComponent<Camera>();
        if (cam == null)
        {
            cam = gameObject.AddComponent<Camera>();
        }

        ConfigureCamera();
        SetupSyntheticCamera();
    }

    void ConfigureCamera()
    {
        cam.fieldOfView = fieldOfView;
        cam.nearClipPlane = nearClip;
        cam.farClipPlane = farClip;
        cam.targetTexture = new RenderTexture(width, height, 24);
    }

    void SetupSyntheticCamera()
    {
        syntheticCamera = GetComponent<SyntheticCameraData>();
        if (syntheticCamera == null)
        {
            syntheticCamera = gameObject.AddComponent<SyntheticCameraData>();
        }

        syntheticCamera.camera = cam;
        syntheticCamera.sensorId = sensorId;
    }
}
```

### LIDAR Simulation in Unity
Create a LIDAR sensor using raycasting:

```csharp
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;

public class UnityLidarSensor : MonoBehaviour
{
    [Header("LIDAR Properties")]
    [SerializeField] private int horizontalRays = 360;
    [SerializeField] private int verticalRays = 1;
    [SerializeField] private float maxDistance = 10f;
    [SerializeField] private float minDistance = 0.1f;
    [SerializeField] private string topicName = "/scan";
    [SerializeField] private float updateRate = 10f; // Hz

    private float[] ranges;
    private ROSConnection ros;
    private float updateInterval;
    private float lastUpdateTime;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        updateInterval = 1.0f / updateRate;
        lastUpdateTime = 0;

        // Initialize ranges array
        ranges = new float[horizontalRays * verticalRays];
    }

    void Update()
    {
        if (Time.time - lastUpdateTime >= updateInterval)
        {
            UpdateLidarScan();
            PublishScan();
            lastUpdateTime = Time.time;
        }
    }

    void UpdateLidarScan()
    {
        for (int v = 0; v < verticalRays; v++)
        {
            float vAngle = (v - (verticalRays - 1) / 2.0f) * 0.1f; // Vertical spread

            for (int h = 0; h < horizontalRays; h++)
            {
                float hAngle = (h * 2 * Mathf.PI) / horizontalRays;

                Vector3 direction = new Vector3(
                    Mathf.Cos(vAngle) * Mathf.Cos(hAngle),
                    Mathf.Sin(vAngle),
                    Mathf.Cos(vAngle) * Mathf.Sin(hAngle)
                );

                if (Physics.Raycast(transform.position, direction, out RaycastHit hit, maxDistance))
                {
                    ranges[v * horizontalRays + h] = hit.distance;
                }
                else
                {
                    ranges[v * horizontalRays + h] = float.PositiveInfinity;
                }
            }
        }
    }

    void PublishScan()
    {
        var laserScan = new LaserScanMsg
        {
            header = new std_msgs.HeaderMsg
            {
                stamp = new builtin_interfaces.TimeMsg
                {
                    sec = (int)Time.time,
                    nanosec = (uint)((Time.time % 1) * 1e9)
                },
                frame_id = transform.name
            },
            angle_min = -Mathf.PI,
            angle_max = Mathf.PI,
            angle_increment = (2 * Mathf.PI) / horizontalRays,
            time_increment = 0,
            scan_time = 1.0f / updateRate,
            range_min = minDistance,
            range_max = maxDistance,
            ranges = ranges
        };

        ros.Publish(topicName, laserScan);
    }
}
```

### IMU Simulation in Unity
```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;

public class UnityImuSensor : MonoBehaviour
{
    [Header("IMU Properties")]
    [SerializeField] private string topicName = "/imu/data";
    [SerializeField] private float updateRate = 100f; // Hz
    [SerializeField] private float noiseLevel = 0.01f;

    private ROSConnection ros;
    private float updateInterval;
    private float lastUpdateTime;
    private Rigidbody attachedRigidbody;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        updateInterval = 1.0f / updateRate;
        lastUpdateTime = 0;

        // Try to find attached rigidbody
        attachedRigidbody = GetComponent<Rigidbody>();
        if (attachedRigidbody == null)
        {
            attachedRigidbody = GetComponentInParent<Rigidbody>();
        }
    }

    void Update()
    {
        if (Time.time - lastUpdateTime >= updateInterval)
        {
            PublishImuData();
            lastUpdateTime = Time.time;
        }
    }

    void PublishImuData()
    {
        var imuMsg = new ImuMsg
        {
            header = new std_msgs.HeaderMsg
            {
                stamp = new builtin_interfaces.TimeMsg
                {
                    sec = (int)Time.time,
                    nanosec = (uint)((Time.time % 1) * 1e9)
                },
                frame_id = transform.name
            }
        };

        // Set orientation (from Unity rotation to quaternion)
        Quaternion unityRotation = transform.rotation;
        // Convert Unity coordinate system to ROS coordinate system
        imuMsg.orientation = new geometry_msgs.QuaternionMsg
        {
            x = unityRotation.x,
            y = unityRotation.y,
            z = unityRotation.z,
            w = unityRotation.w
        };

        // Set angular velocity
        if (attachedRigidbody != null)
        {
            Vector3 angularVel = attachedRigidbody.angularVelocity;
            imuMsg.angular_velocity = new geometry_msgs.Vector3Msg
            {
                x = angularVel.x + Random.Range(-noiseLevel, noiseLevel),
                y = angularVel.y + Random.Range(-noiseLevel, noiseLevel),
                z = angularVel.z + Random.Range(-noiseLevel, noiseLevel)
            };

            // Set linear acceleration
            Vector3 linearAcc = attachedRigidbody.velocity / Time.fixedDeltaTime;
            imuMsg.linear_acceleration = new geometry_msgs.Vector3Msg
            {
                x = linearAcc.x + Random.Range(-noiseLevel, noiseLevel),
                y = linearAcc.y + Random.Range(-noiseLevel, noiseLevel),
                z = linearAcc.z + Random.Range(-noiseLevel, noiseLevel)
            };
        }

        ros.Publish(topicName, imuMsg);
    }
}
```

## Sensor Fusion and Calibration

### Multi-Sensor Integration
Combine data from multiple sensors for enhanced perception:

```csharp
using System.Collections.Generic;
using UnityEngine;

public class SensorFusion : MonoBehaviour
{
    [SerializeField] private List<GameObject> sensors;
    private Dictionary<string, object> sensorData;

    void Start()
    {
        sensorData = new Dictionary<string, object>();
    }

    void Update()
    {
        // Collect data from all sensors
        foreach (var sensor in sensors)
        {
            ISensorDataProvider provider = sensor.GetComponent<ISensorDataProvider>();
            if (provider != null)
            {
                sensorData[provider.GetSensorId()] = provider.GetData();
            }
        }

        // Process fused sensor data
        ProcessFusedData();
    }

    void ProcessFusedData()
    {
        // Implement sensor fusion algorithms (Kalman filters, particle filters, etc.)
        // Combine data from different sensors for better accuracy
    }
}

public interface ISensorDataProvider
{
    string GetSensorId();
    object GetData();
}
```

### Sensor Calibration
Simulate sensor calibration procedures:

```csharp
using UnityEngine;

public class SensorCalibration : MonoBehaviour
{
    [Header("Calibration Parameters")]
    [SerializeField] private float calibrationDuration = 5.0f;
    [SerializeField] private float calibrationInterval = 0.1f;

    private bool isCalibrating = false;
    private float calibrationStartTime = 0f;

    public void StartCalibration()
    {
        isCalibrating = true;
        calibrationStartTime = Time.time;
    }

    void Update()
    {
        if (isCalibrating)
        {
            if (Time.time - calibrationStartTime >= calibrationDuration)
            {
                CompleteCalibration();
            }
            else
            {
                PerformCalibrationStep();
            }
        }
    }

    void PerformCalibrationStep()
    {
        // Perform calibration calculations
        // Adjust sensor parameters based on calibration data
    }

    void CompleteCalibration()
    {
        isCalibrating = false;
        Debug.Log("Calibration completed");
    }
}
```

## Sensor Noise and Uncertainty

### Adding Realistic Noise
Real sensors have inherent noise and uncertainty:

```csharp
using UnityEngine;

public class SensorNoise : MonoBehaviour
{
    [Header("Noise Parameters")]
    [SerializeField] private float gaussianNoiseStdDev = 0.01f;
    [SerializeField] private float bias = 0.0f;
    [SerializeField] private float driftRate = 0.001f;

    private float currentBias = 0f;

    public float ApplyNoise(float rawValue)
    {
        // Add Gaussian noise
        float gaussianNoise = RandomGaussian() * gaussianNoiseStdDev;

        // Add bias
        float biasedValue = rawValue + bias + currentBias;

        // Add noise
        float noisyValue = biasedValue + gaussianNoise;

        // Update drift
        currentBias += Random.Range(-driftRate, driftRate) * Time.deltaTime;

        return noisyValue;
    }

    float RandomGaussian()
    {
        // Box-Muller transform for Gaussian random numbers
        float u1 = Random.value;
        float u2 = Random.value;
        return Mathf.Sqrt(-2.0f * Mathf.Log(u1)) * Mathf.Cos(2.0f * Mathf.PI * u2);
    }
}
```

## Best Practices for Sensor Simulation

### Performance Optimization
- Use appropriate update rates for each sensor type
- Implement level-of-detail for sensor processing
- Use occlusion culling for vision sensors
- Cache expensive calculations when possible

### Realism vs. Performance
- Balance physical accuracy with simulation speed
- Use simplified models for distant objects
- Implement adaptive resolution based on importance

### Validation
- Compare simulation results with real sensor data
- Validate noise characteristics match real sensors
- Test edge cases and failure modes

## Troubleshooting Common Issues

### Sensor Data Quality
- Check coordinate frame alignment between sensors
- Verify proper TF transforms are published
- Ensure sensor mounting positions are accurate

### Performance Issues
- Reduce ray count for LIDAR sensors if performance is poor
- Use lower resolution cameras for faster processing
- Implement sensor frustum culling

## Exercise

Create a complete sensor simulation setup for your humanoid robot that includes:
1. A RGB camera with realistic noise characteristics
2. A 2D LIDAR sensor for navigation
3. An IMU for orientation and motion sensing
4. A fusion algorithm that combines sensor data for improved accuracy

Test the sensor setup in various environments and validate that the simulated data matches expected real-world sensor behavior.