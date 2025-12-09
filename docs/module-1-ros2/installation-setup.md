# ROS 2 Installation and Setup

This section provides comprehensive instructions for installing and setting up ROS 2 (Humble Hawksbill) for use with humanoid robotics applications. Following these steps will prepare your development environment for the exercises in this book.

## System Requirements

Before installing ROS 2, ensure your system meets the following requirements:

- **Operating System**: Ubuntu 22.04 LTS (Jammy Jellyfish) - Recommended for this book
- **Processor**: Multi-core processor (Intel i7 or equivalent recommended)
- **Memory**: 8 GB RAM minimum, 16 GB recommended
- **Storage**: 20 GB free disk space
- **Internet**: Required for package installation

> **Note**: While ROS 2 can run on other platforms (Windows with WSL2, macOS), this book focuses on Ubuntu 22.04 for consistency and optimal performance.

## Installation Steps

### 1. Set Locale

Ensure your locale is set to UTF-8:

```bash
locale  # Check current locale
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
```

### 2. Add ROS 2 Repository

```bash
sudo apt update && sudo apt install -y locales
sudo locale-gen en_US.UTF-8
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
```

### 3. Install ROS 2 Humble

Install the desktop version which includes Gazebo and other simulation tools:

```bash
sudo apt install ros-humble-desktop
```

### 4. Install Additional Dependencies

```bash
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
```

### 5. Initialize rosdep

```bash
sudo rosdep init
rosdep update
```

### 6. Environment Setup

Add ROS 2 environment setup to your bashrc:

```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## Verification

Test that ROS 2 is installed correctly:

```bash
# Check ROS 2 version
ros2 --version

# Test basic functionality
ros2 topic list

# Run a simple demo
source /opt/ros/humble/setup.bash
ros2 run demo_nodes_cpp talker
```

In another terminal:
```bash
source /opt/ros/humble/setup.bash
ros2 run demo_nodes_py listener
```

## Python Package Dependencies

Install additional Python packages commonly used in robotics:

```bash
pip3 install numpy matplotlib transforms3d pyquaternion
```

## Workspace Setup

Create a workspace for your humanoid robotics projects:

```bash
# Create workspace directory
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Source ROS 2 environment
source /opt/ros/humble/setup.bash

# Build the workspace (initially empty)
colcon build
```

Add workspace to your bashrc:

```bash
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## Common Issues and Solutions

### Issue: Permission Denied Error
If you encounter permission errors, ensure you're sourcing the correct setup file:

```bash
source /opt/ros/humble/setup.bash
```

### Issue: Python Import Errors
If Python packages cannot be imported, ensure your environment is properly set:

```bash
# Check if ROS 2 packages are found
python3 -c "import rclpy; print('rclpy imported successfully')"
```

### Issue: Gazebo Not Found
If Gazebo is not available after installation:

```bash
sudo apt install ros-humble-gazebo-*
```

## Docker Alternative (Optional)

For a containerized development environment, you can use the official ROS 2 Docker image:

```bash
# Pull the ROS 2 Humble image
docker pull osrf/ros:humble-desktop

# Run with GUI support (for visualization tools)
docker run -it \
  --env="DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --network=host \
  osrf/ros:humble-desktop
```

## Next Steps

After completing the installation:

1. Verify your installation by running the talker/listener demo
2. Create your first ROS 2 package following the next section
3. Set up your development environment with your preferred IDE
4. Proceed to the next chapter to learn about ROS 2 concepts

## Troubleshooting

### If ROS 2 Commands Are Not Found

Ensure your environment is properly sourced:

```bash
# Check if environment variables are set
echo $ROS_DISTRO
echo $ROS_VERSION

# If not set, source the setup file manually
source /opt/ros/humble/setup.bash
```

### Network Configuration

For multi-machine ROS 2 communication, you may need to configure RMW (ROS Middleware):

```bash
# Check current RMW implementation
printenv | grep RMW

# Set to Fast DDS (default)
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

## Resources

- [Official ROS 2 Humble Installation Guide](https://docs.ros.org/en/humble/Installation.html)
- [ROS 2 Tutorials](https://docs.ros.org/en/humble/Tutorials.html)
- [ROS 2 Concepts](https://docs.ros.org/en/humble/Concepts.html)

Your ROS 2 environment is now ready for developing humanoid robotics applications. In the next sections, we'll explore ROS 2 concepts and create your first robotic nodes.