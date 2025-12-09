# Isaac Sim Installation and Setup

This section provides comprehensive instructions for installing and setting up NVIDIA Isaac Sim for robotics perception and navigation applications.

## System Requirements

### Hardware Requirements
- **GPU**: NVIDIA RTX 3080/4080/4090 or professional GPU (A40, A6000) with 10GB+ VRAM
- **CPU**: 8+ core processor (Intel i7 / AMD Ryzen 7 or better recommended)
- **RAM**: 32GB system memory (64GB+ recommended)
- **Storage**: 50GB+ free space on SSD (100GB+ recommended)
- **OS**: Ubuntu 20.04/22.04 LTS or Windows 10/11 (64-bit)

### Software Requirements
- **NVIDIA GPU Drivers**: Version 520+ (535+ recommended)
- **CUDA Toolkit**: Version 11.8 or 12.x
- **Docker**: Version 20.10+ (for containerized deployment)
- **NVIDIA Container Toolkit**: For GPU-accelerated containers
- **Python**: Version 3.8-3.10 for Isaac ROS integration

## Pre-Installation Checks

### Verify GPU and Drivers
```bash
# Check if NVIDIA GPU is detected
nvidia-smi

# Verify CUDA installation
nvcc --version

# Check for compatible GPU
nvidia-ml-py3 # Python library for GPU management
```

### Install Required Dependencies
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install basic dependencies
sudo apt install -y build-essential cmake git python3-dev python3-pip

# Install graphics drivers (Ubuntu)
sudo apt install -y nvidia-driver-535 nvidia-utils-535
sudo reboot
```

## Installation Methods

### Method 1: Omniverse Launcher (Recommended for Beginners)

#### 1. Download Omniverse Launcher
1. Go to [NVIDIA Developer](https://developer.nvidia.com/omniverse) website
2. Register or sign in to your NVIDIA Developer account
3. Download Omniverse Launcher for your operating system
4. Install the launcher following the on-screen instructions

#### 2. Install Isaac Sim through Launcher
1. Launch Omniverse Launcher
2. Sign in with your NVIDIA Developer account
3. Navigate to "Isaac" section
4. Click "Install" next to Isaac Sim
5. Choose installation location (default is recommended)
6. Wait for download and installation to complete

#### 3. Launch Isaac Sim
1. From the launcher, click "Launch" next to Isaac Sim
2. The application will start and load the initial scene
3. Verify installation by checking Help → About Isaac Sim

### Method 2: Containerized Installation (Recommended for Production)

#### 1. Install Docker and NVIDIA Container Toolkit
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### 2. Pull and Run Isaac Sim Container
```bash
# Pull the Isaac Sim Docker image
docker pull nvcr.io/nvidia/isaac-sim:4.0.0

# Create directories for persistent data
mkdir -p ~/docker/isaac-sim/cache/kit
mkdir -p ~/docker/isaac-sim/assets
mkdir -p ~/docker/isaac-sim/home

# Run Isaac Sim container with GUI support
xhost +local:docker
docker run --gpus all -it --rm \
  --network=host \
  --env "ACCEPT_EULA=Y" \
  --env "NVIDIA_VISIBLE_DEVICES=all" \
  --env "NVIDIA_DRIVER_CAPABILITIES=all" \
  --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --volume $HOME/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
  --volume $HOME/docker/isaac-sim/assets:/isaac-sim/assets:rw \
  --volume $HOME/docker/isaac-sim/home:/isaac-sim/home:rw \
  --volume $HOME/docker/isaac-sim/logs:/isaac-sim/logs:rw \
  --volume $HOME/docker/isaac-sim/config:/isaac-sim/config:rw \
  nvcr.io/nvidia/isaac-sim:4.0.0
```

### Method 3: Native Installation (Advanced Users)

#### 1. Install Isaac Sim Prerequisites
```bash
# Install additional dependencies for native installation
sudo apt install -y \
  libgl1-mesa-glx \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender-dev \
  libgomp1 \
  libssl1.1

# Install Python dependencies
pip3 install --user \
  numpy \
  scipy \
  matplotlib \
  opencv-python \
  transforms3d
```

#### 2. Download and Install Isaac Sim
1. Download Isaac Sim from NVIDIA Developer portal
2. Extract the archive to your preferred location (e.g., `/opt/isaac-sim`)
3. Set up environment variables:

```bash
# Add to ~/.bashrc
echo 'export ISAAC_SIM_PATH=/opt/isaac-sim' >> ~/.bashrc
echo 'export PYTHONPATH=$ISAAC_SIM_PATH/python:$PYTHONPATH' >> ~/.bashrc
echo 'export PATH=$ISAAC_SIM_PATH/python/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

## Isaac ROS Integration Setup

### Install Isaac ROS Dependencies
```bash
# Install ROS 2 Humble (if not already installed)
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install -y ros-humble-desktop
sudo apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
```

### Install Isaac ROS Packages
```bash
# Create workspace
mkdir -p ~/isaac_ros_ws/src
cd ~/isaac_ros_ws

# Source ROS 2
source /opt/ros/humble/setup.bash

# Install vcs tools
sudo apt install python3-vcstool

# Get Isaac ROS packages
wget https://raw.githubusercontent.com/NVIDIA-ISAAC-ROS/.repos/main/isaac_ros.repos
vcs import src < isaac_ros.repos

# Install dependencies
rosdep install --from-paths src --ignore-src -r -y

# Build the workspace
colcon build --symlink-install --packages-select \
  isaac_ros_common \
  isaac_ros_image_pipeline \
  isaac_ros_visual_slam \
  isaac_ros_pose_estimation \
  isaac_ros_bi3d \
  isaac_ros_apriltag \
  isaac_ros_nitros_type_introspection \
  isaac_ros_managed_nh \
  isaac_ros_test \
  isaac_ros_test_utils
```

### Configure Isaac ROS Environment
```bash
# Add to ~/.bashrc
echo 'source ~/isaac_ros_ws/install/setup.bash' >> ~/.bashrc
source ~/.bashrc
```

## Isaac Sim Configuration

### Initial Configuration
After first launch, configure Isaac Sim:

1. **Extensions Setup**:
   - Go to Window → Extensions
   - Enable required extensions:
     - Isaac → All Isaac extensions
     - Robotics → All robotics extensions
     - Perception → All perception extensions
     - Navigation → All navigation extensions
     - ROS/ROS2 Bridge → Enable bridge extensions

2. **Preferences Configuration**:
   - Edit → Preferences → Isaac Sim
   - Set preferred units to meters
   - Configure physics settings (see below)
   - Set rendering quality based on your hardware

### Physics Configuration
```python
# Example physics configuration script
import carb

settings = carb.settings.get_settings()

# Physics settings for robotics simulation
settings.set("/physics/solverType", 0)  # 0=PGS, 1=MLCP
settings.set("/physics/solverPositionIterationCount", 8)
settings.set("/physics/solverVelocityIterationCount", 4)
settings.set("/physics/defaultRestOffset", 0.001)
settings.set("/physics/defaultContactOffset", 0.002)
settings.set("/physics/bounceThreshold", 2.0)
settings.set("/physics/sleepThreshold", 0.005)
```

### Rendering Configuration
```python
# Rendering settings for optimal performance
settings.set("/app/performace/range", 2)  # Performance level
settings.set("/rtx/indirectDiffuseLighting/enabled", True)
settings.set("/rtx/reflections/enabled", True)
settings.set("/rtx/dlss/enable", True)  # If RTX GPU supports DLSS
settings.set("/rtx/dlss/mode", 2)  # Quality mode
```

## Verification and Testing

### Basic Functionality Test
1. Launch Isaac Sim
2. Create a new stage (File → New Stage)
3. Add a simple primitive (Create → Cube)
4. Verify physics by pressing Play and observing gravity effect
5. Add a camera and verify rendering

### ROS Bridge Test
```bash
# Terminal 1: Launch Isaac Sim with ROS bridge
# In Isaac Sim, enable ROS Bridge extension
# Add ROS Bridge node to stage

# Terminal 2: Test ROS communication
source /opt/ros/humble/setup.bash
source ~/isaac_ros_ws/install/setup.bash

# Check available topics
ros2 topic list

# Test camera publishing
ros2 run image_view image_view image:=/rgb_camera

# Test LIDAR publishing
ros2 topic echo /lidar_scan sensor_msgs/msg/LaserScan
```

### Isaac ROS Package Test
```bash
# Test Isaac ROS visual slam
source ~/isaac_ros_ws/install/setup.bash
ros2 launch isaac_ros_visual_slam visual_slam.launch.py input_is_rectified:=False

# Test Isaac ROS apriltag
ros2 launch isaac_ros_apriltag apriltag.launch.py
```

## Troubleshooting Common Issues

### GPU/CUDA Issues
**Problem**: Isaac Sim fails to start or shows rendering errors
**Solutions**:
1. Verify NVIDIA drivers: `nvidia-smi`
2. Check CUDA installation: `nvcc --version`
3. Ensure correct GPU: `lspci | grep -i nvidia`
4. Update drivers if necessary

### ROS Communication Issues
**Problem**: Isaac Sim cannot communicate with ROS
**Solutions**:
1. Check ROS network configuration
2. Verify Isaac Sim ROS Bridge extension is enabled
3. Check firewall settings
4. Ensure correct IP addresses and ports

### Performance Issues
**Problem**: Isaac Sim runs slowly or has low frame rates
**Solutions**:
1. Reduce rendering quality in preferences
2. Lower physics substeps
3. Simplify scene geometry
4. Check GPU memory usage

### Container Issues
**Problem**: Isaac Sim container fails to start
**Solutions**:
1. Verify NVIDIA Container Toolkit installation
2. Check Docker permissions
3. Ensure GPU is accessible in container: `docker run --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi`

## Performance Optimization

### For Development
- Use lower rendering quality during development
- Disable advanced rendering features (ray tracing, global illumination)
- Use simpler collision geometries
- Reduce physics update rate

### For Production
- Optimize asset complexity
- Use level-of-detail (LOD) systems
- Implement occlusion culling
- Use appropriate texture resolutions

## Environment Variables

Set these environment variables for optimal Isaac Sim operation:

```bash
# Add to ~/.bashrc
export ISAAC_SIM_DISABLE_CUDA_DEVICE_GPU_INFO=1  # Disable CUDA device info logging
export ISAAC_SIM_DISABLE_OPEN_GL_GPU_INFO=1      # Disable OpenGL GPU info logging
export ISAAC_SIM_FORCE_GPU=1                     # Force GPU usage
export ISAAC_ROS_BRIDGE_DISABLE_TCP_NODE=1       # Disable TCP node if not needed
```

## Next Steps

After successful installation:

1. Complete the Isaac Sim tutorials to familiarize yourself with the interface
2. Set up your first robot simulation
3. Configure ROS bridges for your specific robot
4. Begin developing perception and navigation pipelines

Your Isaac Sim installation is now ready for developing AI-powered robotics applications!