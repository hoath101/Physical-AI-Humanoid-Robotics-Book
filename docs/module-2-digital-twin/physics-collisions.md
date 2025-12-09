# Physics and Collisions

Understanding physics and collision properties is crucial for creating realistic digital twins. This section covers the principles of physics simulation and collision detection in both Gazebo and Unity environments.

## Physics Simulation Fundamentals

Physics simulation in robotics environments involves modeling real-world physical behaviors:
- **Rigid Body Dynamics**: Movement and interaction of solid objects
- **Collision Detection**: Identifying when objects intersect
- **Contact Response**: Calculating forces when objects touch
- **Constraints**: Joints and connections between parts

## Physics in Gazebo

### Physics Engines
Gazebo supports multiple physics engines:
- **ODE (Open Dynamics Engine)**: Default, good balance of speed and accuracy
- **Bullet**: Fast and robust, good for complex interactions
- **Simbody**: Highly accurate, suitable for biomechanics
- **DART**: Advanced constraint handling

### Physics Configuration
In world files, configure physics properties:

```xml
<physics type="ode">
  <max_step_size>0.001</max_step_size>  <!-- Time step for simulation -->
  <real_time_factor>1</real_time_factor> <!-- Simulation speed vs real time -->
  <real_time_update_rate>1000</real_time_update_rate> <!-- Hz -->

  <!-- Gravity -->
  <gravity>0 0 -9.8</gravity>

  <!-- ODE-specific parameters -->
  <ode>
    <solver>
      <type>quick</type>  <!-- or "pgs" -->
      <iters>10</iters>
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>0</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>100</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

### Material Properties
Define material properties for realistic interactions:

```xml
<link name="link_with_materials">
  <collision name="collision">
    <geometry>
      <box><size>0.1 0.1 0.1</size></box>
    </geometry>
    <surface>
      <friction>
        <ode>
          <mu>0.5</mu>      <!-- Static friction coefficient -->
          <mu2>0.5</mu2>    <!-- Secondary friction coefficient -->
          <slip1>0.0</slip1> <!-- Primary slip coefficient -->
          <slip2>0.0</slip2> <!-- Secondary slip coefficient -->
        </ode>
      </friction>
      <bounce>
        <restitution_coefficient>0.1</restitution_coefficient>
        <threshold>100000</threshold>
      </bounce>
      <contact>
        <ode>
          <max_vel>100</max_vel>
          <min_depth>0.001</min_depth>
        </ode>
      </contact>
    </surface>
  </collision>
</link>
```

### Inertial Properties
Proper inertial properties are essential for realistic simulation:

```xml
<inertial>
  <mass>1.0</mass>
  <inertia>
    <ixx>0.0833333</ixx>
    <ixy>0.0</ixy>
    <ixz>0.0</ixz>
    <iyy>0.0833333</iyy>
    <iyz>0.0</iyz>
    <izz>0.0833333</izz>
  </inertia>
</inertial>
```

For a box with mass m and dimensions (x, y, z):
- ixx = m*(y² + z²)/12
- iyy = m*(x² + z²)/12
- izz = m*(x² + y²)/12

## Collision Detection in Gazebo

### Collision Geometries
Gazebo supports various collision geometries:
- **Box**: Rectangular prism
- **Cylinder**: Cylindrical shape
- **Sphere**: Spherical shape
- **Mesh**: Complex shapes from 3D models
- **Plane**: Infinite flat surface

```xml
<collision name="collision_box">
  <geometry>
    <box><size>0.1 0.2 0.3</size></box>
  </geometry>
</collision>

<collision name="collision_cylinder">
  <geometry>
    <cylinder>
      <radius>0.05</radius>
      <length>0.1</length>
    </cylinder>
  </geometry>
</collision>

<collision name="collision_sphere">
  <geometry>
    <sphere><radius>0.05</radius></sphere>
  </geometry>
</collision>
```

### Collision Filtering
Use collision groups and masks to control which objects can collide:

```xml
<collision name="collision_with_filter">
  <surface>
    <contact>
      <collide_without_contact>false</collide_without_contact>
    </contact>
  </surface>
</collision>
```

## Physics in Unity

### Physics Engine
Unity uses NVIDIA PhysX for physics simulation, which provides:
- Advanced collision detection
- Realistic contact response
- Vehicle dynamics
- Cloth simulation
- Soft body dynamics

### Rigidbody Configuration
Each physical object needs a Rigidbody component:

```csharp
using UnityEngine;

public class PhysicsObject : MonoBehaviour
{
    [Header("Physics Properties")]
    public float mass = 1.0f;
    public Vector3 centerOfMass = Vector3.zero;
    public Vector3 inertiaTensor = Vector3.one;
    public float drag = 0.0f;
    public float angularDrag = 0.05f;

    [Header("Collision Properties")]
    public bool useGravity = true;
    public bool isKinematic = false;

    private Rigidbody rb;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
        if (rb != null)
        {
            rb.mass = mass;
            rb.centerOfMass = centerOfMass;
            rb.inertiaTensor = inertiaTensor;
            rb.drag = drag;
            rb.angularDrag = angularDrag;
            rb.useGravity = useGravity;
            rb.isKinematic = isKinematic;
        }
    }
}
```

### Collider Configuration
Unity supports various collider types:

```csharp
// Box Collider
[RequireComponent(typeof(Rigidbody))]
public class BoxColliderSetup : MonoBehaviour
{
    [SerializeField] private Vector3 size = Vector3.one;
    [SerializeField] private Vector3 center = Vector3.zero;
    [SerializeField] private bool isTrigger = false;

    void Start()
    {
        BoxCollider boxCollider = gameObject.AddComponent<BoxCollider>();
        boxCollider.size = size;
        boxCollider.center = center;
        boxCollider.isTrigger = isTrigger;
    }
}

// Mesh Collider
[RequireComponent(typeof(Rigidbody))]
public class MeshColliderSetup : MonoBehaviour
{
    [SerializeField] private bool convex = false;
    [SerializeField] private bool inflateMesh = false;

    void Start()
    {
        MeshCollider meshCollider = gameObject.AddComponent<MeshCollider>();
        meshCollider.convex = convex;
        meshCollider.inflateMesh = inflateMesh;

        // Use the mesh from a MeshFilter component
        MeshFilter meshFilter = GetComponent<MeshFilter>();
        if (meshFilter != null)
        {
            meshCollider.sharedMesh = meshFilter.sharedMesh;
        }
    }
}
```

### Joint Configuration
Unity provides various joint types for connecting rigidbodies:

```csharp
// Hinge Joint Example
public class HingeJointSetup : MonoBehaviour
{
    [SerializeField] private float motorForce = 10f;
    [SerializeField] private float targetVelocity = 0f;
    [SerializeField] private bool useMotor = false;

    void Start()
    {
        HingeJoint hinge = GetComponent<HingeJoint>();

        // Configure motor
        JointMotor motor = hinge.motor;
        motor.force = motorForce;
        motor.targetVelocity = targetVelocity;
        motor.freeSpin = false;
        hinge.motor = motor;
        hinge.useMotor = useMotor;

        // Configure limits
        JointLimits limits = hinge.limits;
        limits.min = -90f;  // Minimum angle in degrees
        limits.max = 90f;   // Maximum angle in degrees
        hinge.limits = limits;
        hinge.useLimits = true;
    }
}
```

### Physics Materials
Create realistic surface interactions:

```csharp
// Create a physics material in code
public class PhysicsMaterialSetup : MonoBehaviour
{
    [SerializeField] private float staticFriction = 0.5f;
    [SerializeField] private float dynamicFriction = 0.4f;
    [SerializeField] private float bounciness = 0.1f;
    [SerializeField] private PhysicMaterialCombine frictionCombine = PhysicMaterialCombine.Average;
    [SerializeField] private PhysicMaterialCombine bounceCombine = PhysicMaterialCombine.Average;

    void Start()
    {
        PhysicMaterial material = new PhysicMaterial("CustomMaterial");
        material.staticFriction = staticFriction;
        material.dynamicFriction = dynamicFriction;
        material.bounciness = bounciness;
        material.frictionCombine = frictionCombine;
        material.bounceCombine = bounceCombine;

        Collider col = GetComponent<Collider>();
        if (col != null)
        {
            col.material = material;
        }
    }
}
```

## Collision Events

### Unity Collision Detection
Handle collision events in Unity:

```csharp
using UnityEngine;

public class CollisionHandler : MonoBehaviour
{
    void OnCollisionEnter(Collision collision)
    {
        Debug.Log($"Collision with {collision.gameObject.name}");

        foreach (ContactPoint contact in collision.contacts)
        {
            Debug.DrawRay(contact.point, contact.normal, Color.white);
            Debug.Log($"Contact point: {contact.point}");
            Debug.Log($"Contact force: {collision.impulse}");
        }
    }

    void OnCollisionStay(Collision collision)
    {
        // Called each frame while colliding
        foreach (ContactPoint contact in collision.contacts)
        {
            Debug.Log($"Contact force: {contact.force}");
        }
    }

    void OnCollisionExit(Collision collision)
    {
        Debug.Log($"Collision ended with {collision.gameObject.name}");
    }

    // Trigger events (for isTrigger colliders)
    void OnTriggerEnter(Collider other)
    {
        Debug.Log($"Trigger entered: {other.name}");
    }
}
```

## Advanced Physics Concepts

### Continuous Collision Detection
For fast-moving objects to prevent tunneling:

```csharp
void Start()
{
    Rigidbody rb = GetComponent<Rigidbody>();
    if (rb != null)
    {
        // Use continuous collision detection for fast objects
        rb.collisionDetectionMode = CollisionDetectionMode.Continuous;
    }
}
```

### Layer-based Collision Matrix
Control which layers can collide with each other in Unity:
- Edit → Project Settings → Physics
- Configure the collision matrix for different layers

### Force Application
Apply forces to simulate realistic interactions:

```csharp
public class ForceApplication : MonoBehaviour
{
    [SerializeField] private float forceMagnitude = 10f;
    [SerializeField] private ForceMode forceMode = ForceMode.Force;

    public void ApplyForce(Vector3 direction)
    {
        Rigidbody rb = GetComponent<Rigidbody>();
        if (rb != null)
        {
            rb.AddForce(direction * forceMagnitude, forceMode);
        }
    }

    public void ApplyTorque(Vector3 torque)
    {
        Rigidbody rb = GetComponent<Rigidbody>();
        if (rb != null)
        {
            rb.AddTorque(torque * forceMagnitude, forceMode);
        }
    }
}
```

## Performance Considerations

### Gazebo Optimization
- Use simpler collision geometries when possible
- Adjust physics update rates for performance vs accuracy
- Use fixed joints instead of complex constraint systems
- Limit the number of active objects in simulation

### Unity Optimization
- Use convex mesh colliders for dynamic objects
- Use compound colliders for complex shapes
- Adjust solver iteration counts for performance
- Use object pooling for frequently created/destroyed objects

## Debugging Physics

### Visualization Tools
- Enable physics visualization in both Gazebo and Unity
- Use debug drawing to visualize collision shapes
- Monitor simulation statistics and performance metrics

### Common Issues
- **Tunneling**: Objects passing through each other (fix with CCD)
- **Jittering**: Unstable contact points (fix with proper mass ratios)
- **Penetration**: Objects sinking into each other (fix with proper solver settings)

## Best Practices

- Use realistic mass properties (don't make everything 1kg)
- Configure friction coefficients based on real materials
- Test with extreme values to ensure stability
- Balance accuracy with performance requirements
- Validate simulation results against real-world data when possible

## Exercise

Create a simulation scene with multiple objects of different materials and masses. Implement collision detection that triggers different behaviors based on collision forces. Test the scene with different physics configurations to observe the effects on stability and realism.