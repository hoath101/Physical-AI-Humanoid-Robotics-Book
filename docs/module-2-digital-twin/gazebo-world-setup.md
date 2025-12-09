# Gazebo World Setup and Configuration

Setting up realistic and functional simulation environments is crucial for effective digital twin development. This section covers creating and configuring Gazebo worlds for humanoid robotics applications.

## World File Structure

A Gazebo world file is an SDF (Simulation Description Format) file that defines the complete simulation environment. The basic structure includes:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="humanoid_test_world">
    <!-- Physics properties -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Scene properties -->
    <scene>
      <ambient>0.4 0.4 0.4 1</ambient>
      <background>0.7 0.7 0.7 1</background>
      <shadows>true</shadows>
    </scene>

    <!-- Lighting -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Models and objects -->
    <!-- Robot, obstacles, furniture, etc. -->

    <!-- Plugins -->
    <!-- Additional functionality -->

  </world>
</sdf>
```

## Physics Configuration

### Physics Engine Selection
Choose the appropriate physics engine based on your simulation requirements:

```xml
<!-- ODE (Open Dynamics Engine) - Default, good balance -->
<physics type="ode">
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1</real_time_factor>
  <real_time_update_rate>1000</real_time_update_rate>
  <gravity>0 0 -9.8</gravity>
  <ode>
    <solver>
      <type>quick</type>  <!-- quick or pgs -->
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

<!-- Bullet - Faster, good for complex interactions -->
<physics type="bullet">
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1</real_time_factor>
  <real_time_update_rate>1000</real_time_update_rate>
  <gravity>0 0 -9.8</gravity>
  <bullet>
    <solver>
      <type>sequential_impulse</type>
      <iteration_count>50</iteration_count>
    </solver>
    <constraints>
      <cfm>0.0</cfm>
      <erp>0.2</erp>
    </constraints>
  </bullet>
</physics>
```

### Performance vs. Accuracy Trade-offs
Adjust physics parameters based on your needs:

```xml
<!-- For real-time simulation (faster, less accurate) -->
<physics type="ode">
  <max_step_size>0.01</max_step_size>  <!-- Larger time steps -->
  <real_time_update_rate>100</real_time_update_rate>  <!-- Lower update rate -->
  <ode>
    <solver>
      <iters>10</iters>  <!-- Fewer iterations -->
    </solver>
  </ode>
</physics>

<!-- For accurate simulation (slower, more accurate) -->
<physics type="ode">
  <max_step_size>0.0001</max_step_size>  <!-- Smaller time steps -->
  <real_time_update_rate>10000</real_time_update_rate>  <!-- Higher update rate -->
  <ode>
    <solver>
      <iters>100</iters>  <!-- More iterations -->
    </solver>
  </ode>
</physics>
```

## Lighting and Visual Environment

### Dynamic Lighting
Create realistic lighting conditions:

```xml
<!-- Directional light (sun) -->
<light name="sun" type="directional">
  <cast_shadows>true</cast_shadows>
  <pose>0 0 10 0 0 0</pose>
  <diffuse>0.8 0.8 0.8 1</diffuse>
  <specular>0.2 0.2 0.2 1</specular>
  <attenuation>
    <range>1000</range>
    <constant>0.9</constant>
    <linear>0.01</linear>
    <quadratic>0.001</quadratic>
  </attenuation>
  <direction>-0.3 -0.3 -1</direction>
</light>

<!-- Point lights for indoor environments -->
<light name="room_light" type="point">
  <pose>0 0 3 0 0 0</pose>
  <diffuse>1 1 1 1</diffuse>
  <specular>0.5 0.5 0.5 1</specular>
  <attenuation>
    <range>10</range>
    <constant>0.2</constant>
    <linear>0.5</linear>
    <quadratic>0.01</quadratic>
  </attenuation>
</light>

<!-- Spot lights for focused illumination -->
<light name="spot_light" type="spot">
  <pose>2 2 3 0 0 0</pose>
  <diffuse>1 0.8 0.5 1</diffuse>
  <specular>1 1 1 1</specular>
  <attenuation>
    <range>5</range>
    <constant>0.2</constant>
    <linear>0.5</linear>
    <quadratic>0.01</quadratic>
  </attenuation>
  <direction>-0.5 -0.5 -1</direction>
  <spot>
    <inner_angle>0.1</inner_angle>
    <outer_angle>0.5</outer_angle>
    <falloff>1</falloff>
  </spot>
</light>
```

### Environment and Sky
Configure the visual environment:

```xml
<scene>
  <ambient>0.3 0.3 0.3 1</ambient>
  <background>0.6 0.7 0.9 1</background>
  <shadows>true</shadows>
  <!-- Enable HDR rendering -->
  <grid>false</grid>
  <origin_visual>false</origin_visual>
  <!-- Sky properties -->
  <sky>
    <time>14:00</time>
    <sun_direction>0.7 -0.7 -0.7</sun_direction>
    <clouds>
      <speed>0.1</speed>
      <direction>0.3 0.7</direction>
      <humidity>0.5</humidity>
      <mean_size>0.5</mean_size>
    </clouds>
  </sky>
</scene>
```

## Model Placement and Configuration

### Including Standard Models
Use Gazebo's built-in models:

```xml
<!-- Ground plane -->
<include>
  <uri>model://ground_plane</uri>
</include>

<!-- Sun light -->
<include>
  <uri>model://sun</uri>
</include>

<!-- Custom models from model database -->
<include>
  <uri>model://cylinder</uri>
  <pose>2 0 1 0 0 0</pose>
</include>

<include>
  <uri>model://box</uri>
  <pose>-2 0 1 0 0 0</pose>
  <name>obstacle_box</name>
</include>
```

### Creating Custom Models
Define custom objects in your world:

```xml
<!-- Simple box obstacle -->
<model name="table">
  <pose>1 1 0.5 0 0 0</pose>
  <link name="table_base">
    <pose>0 0 0.4 0 0 0</pose>
    <collision name="collision">
      <geometry>
        <box>
          <size>1.0 0.8 0.8</size>
        </box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box>
          <size>1.0 0.8 0.8</size>
        </box>
      </geometry>
      <material>
        <ambient>0.8 0.6 0.4 1</ambient>
        <diffuse>0.8 0.6 0.4 1</diffuse>
        <specular>0.2 0.2 0.2 1</specular>
      </material>
    </visual>
    <inertial>
      <mass>50.0</mass>
      <inertia>
        <ixx>6.8667</ixx>
        <ixy>0</ixy>
        <ixz>0</ixz>
        <iyy>8.5333</iyy>
        <iyz>0</iyz>
        <izz>2.6667</izz>
      </inertia>
    </inertial>
  </link>

  <link name="table_top">
    <pose>0 0 0.85 0 0 0</pose>
    <collision name="collision">
      <geometry>
        <box>
          <size>1.2 1.0 0.05</size>
        </box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box>
          <size>1.2 1.0 0.05</size>
        </box>
      </geometry>
      <material>
        <ambient>0.9 0.9 0.9 1</ambient>
        <diffuse>0.9 0.9 0.9 1</diffuse>
        <specular>0.3 0.3 0.3 1</specular>
      </material>
    </visual>
    <inertial>
      <mass>10.0</mass>
      <inertia>
        <ixx>1.0417</ixx>
        <ixy>0</ixy>
        <ixz>0</ixz>
        <iyy>1.45</iyy>
        <iyz>0</iyz>
        <izz>0.4083</izz>
      </inertia>
    </inertial>
  </link>

  <joint name="top_to_base" type="fixed">
    <parent>table_base</parent>
    <child>table_top</child>
  </joint>
</model>
```

## Robot Integration

### Spawning Robots in Worlds
Include your robot model directly in the world file:

```xml
<!-- Include your robot URDF -->
<include>
  <uri>model://humanoid_robot</uri>
  <pose>0 0 1 0 0 0</pose>
  <name>humanoid_1</name>
</include>
```

### Robot-Specific World Configuration
Configure world properties specifically for your robot:

```xml
<world name="humanoid_navigaton_world">
  <!-- Physics tuned for humanoid robot -->
  <physics type="ode">
    <max_step_size>0.001</max_step_size>
    <real_time_factor>1</real_time_factor>
    <real_time_update_rate>1000</real_time_update_rate>
    <gravity>0 0 -9.8</gravity>
    <ode>
      <solver>
        <type>quick</type>
        <iters>20</iters>  <!-- Higher iterations for stability -->
        <sor>1.3</sor>
      </solver>
      <constraints>
        <cfm>1e-5</cfm>  <!-- Lower CFM for better contact stability -->
        <erp>0.2</erp>
        <contact_max_correcting_vel>10</contact_max_correcting_vel>
        <contact_surface_layer>0.001</contact_surface_layer>
      </constraints>
    </ode>
  </physics>

  <!-- Humanoid-specific objects -->
  <include>
    <uri>model://ground_plane</uri>
  </include>

  <!-- Navigation obstacles -->
  <model name="obstacle_1">
    <pose>2 0 0.5 0 0 0</pose>
    <link name="link">
      <collision name="collision">
        <geometry>
          <box><size>0.5 0.5 1.0</size></box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box><size>0.5 0.5 1.0</size></box>
        </geometry>
        <material>
          <ambient>0.8 0.3 0.3 1</ambient>
          <diffuse>0.8 0.3 0.3 1</diffuse>
        </material>
      </visual>
      <inertial>
        <mass>10.0</mass>
        <inertia>
          <ixx>1.0417</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>1.0417</iyy>
          <iyz>0</iyz>
          <izz>0.4167</izz>
        </inertia>
      </inertial>
    </link>
  </model>

  <!-- Add robot -->
  <include>
    <uri>model://humanoid_robot</uri>
    <pose>0 0 1 0 0 0</pose>
    <name>test_humanoid</name>
  </include>
</world>
```

## Plugin Configuration

### ROS Integration Plugins
Add plugins for ROS communication:

```xml
<!-- World plugin for ROS integration -->
<plugin name="world_plugin" filename="libgazebo_ros_init.so">
  <ros>
    <namespace>gazebo</namespace>
  </ros>
  <update_rate>1000</update_rate>
</plugin>

<!-- Physics reset plugin -->
<plugin name="physics_reset" filename="libgazebo_ros_pubslish_odometry.so">
  <ros>
    <namespace>gazebo</namespace>
  </ros>
</plugin>
```

### Custom Plugins
Create custom world plugins for specific functionality:

```xml
<!-- Example: Plugin to spawn objects at random locations -->
<plugin name="random_spawner" filename="librandom_spawner.so">
  <spawn_count>5</spawn_count>
  <spawn_area>
    <min_x>-5</min_x>
    <max_x>5</max_x>
    <min_y>-5</min_y>
    <max_y>5</max_y>
  </spawn_area>
  <object_types>
    <object>box</object>
    <object>sphere</object>
    <object>cylinder</object>
  </object_types>
</plugin>
```

## Advanced World Features

### Heightmaps for Terrain
Create realistic outdoor environments:

```xml
<model name="terrain">
  <static>true</static>
  <link name="link">
    <collision name="collision">
      <geometry>
        <heightmap>
          <uri>model://my_terrain/heightmap.png</uri>
          <size>100 100 20</size>
          <pos>0 0 0</pos>
        </heightmap>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <heightmap>
          <uri>model://my_terrain/heightmap.png</uri>
          <size>100 100 20</size>
          <pos>0 0 0</pos>
        </heightmap>
      </geometry>
      <material>
        <script>
          <uri>file://media/materials/scripts/gazebo.material</uri>
          <name>Gazebo/Ground</name>
        </script>
      </material>
    </visual>
  </link>
</model>
```

### Wind Effects
Simulate environmental forces:

```xml
<world name="windy_world">
  <!-- Add wind plugin -->
  <plugin name="wind" filename="libgazebo_ros_wind.so">
    <always_on>true</always_on>
    <pub_topic>/wind</pub_topic>
    <wind_direction>1 0 0</wind_direction>
    <wind_force>0.5 0 0</wind_force>
    <wind_gust_duration>0</wind_gust_duration>
    <wind_gust_start>0</wind_gust_start>
    <wind_gust_final>0</wind_gust_final>
  </plugin>
</world>
```

## World Launch Configuration

### ROS 2 Launch Files
Create launch files to easily load your worlds:

```python
# launch/humanoid_world.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    world_arg = DeclareLaunchArgument(
        'world',
        default_value='humanoid_test_world.sdf',
        description='Choose one of the world files from `/my_robot_gazebo/worlds`'
    )

    world_path = PathJoinSubstitution([
        get_package_share_directory('my_robot_gazebo'),
        'worlds',
        LaunchConfiguration('world')
    ])

    # Launch Gazebo with world
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('gazebo_ros'),
            '/launch/gazebo.launch.py'
        ]),
        launch_arguments={
            'world': world_path,
            'verbose': 'true'
        }.items()
    )

    return LaunchDescription([
        world_arg,
        gazebo
    ])
```

## Performance Optimization

### Level of Detail (LOD)
Optimize complex models for performance:

```xml
<model name="detailed_object">
  <link name="link">
    <visual name="visual">
      <geometry>
        <mesh>
          <uri>model://object/meshes/complex_model.dae</uri>
        </mesh>
      </geometry>
      <!-- Simplified collision geometry -->
      <collision name="collision">
        <geometry>
          <box><size>1 1 1</size></box>
        </geometry>
      </collision>
    </visual>
  </link>
</model>
```

### Dynamic Loading
Use plugins to load/unload objects dynamically:

```xml
<!-- Plugin to dynamically add/remove objects -->
<plugin name="dynamic_objects" filename="libdynamic_objects.so">
  <max_objects>20</max_objects>
  <object_types>
    <type>
      <name>box</name>
      <uri>model://box</uri>
      <max_count>10</max_count>
    </type>
    <type>
      <name>sphere</name>
      <uri>model://sphere</uri>
      <max_count>10</max_count>
    </type>
  </object_types>
</plugin>
```

## Debugging World Files

### Common Issues and Solutions

#### 1. Robot Falls Through Ground
- Check collision geometry is properly defined
- Verify mass and inertia properties
- Adjust physics parameters (CFM, ERP)

#### 2. Objects Interpenetrate
- Increase constraint solver iterations
- Adjust contact properties
- Verify collision geometry overlaps

#### 3. Performance Issues
- Simplify collision geometry
- Reduce physics update rate
- Use static models where possible

### Validation Commands
```bash
# Check SDF validity
gz sdf -k /path/to/your/world.sdf

# View world in Gazebo
gz sim /path/to/your/world.sdf

# Debug with verbose output
gz sim -v 4 /path/to/your/world.sdf
```

## Best Practices

1. **Start Simple**: Begin with basic worlds and add complexity gradually
2. **Use Standard Models**: Leverage Gazebo's built-in models when possible
3. **Optimize for Performance**: Balance visual quality with simulation speed
4. **Validate Physics**: Test that objects behave realistically
5. **Document Configuration**: Keep comments explaining parameter choices
6. **Test Robot Integration**: Verify your robot works in the world
7. **Version Control**: Track world file changes with your robot code

## Exercise

Create a complete Gazebo world for humanoid robot testing that includes:
1. A physics configuration optimized for humanoid robots
2. Appropriate lighting for vision sensors
3. Various obstacles and navigation challenges
4. A robot spawn point with proper initial configuration
5. ROS integration plugins for communication
6. Performance optimizations for real-time simulation

Test your world by launching it with your humanoid robot model and verifying that the robot can navigate and interact with the environment appropriately.