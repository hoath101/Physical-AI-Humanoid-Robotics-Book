# Quickstart Guide: Physical AI & Humanoid Robotics Book

## Prerequisites

### System Requirements
- **OS**: Ubuntu 22.04 LTS (recommended) or Windows 10/11 with WSL2
- **CPU**: Intel i7 13th Gen+ or Ryzen 9
- **RAM**: 64 GB (minimum 32 GB)
- **GPU**: RTX 4070 Ti+ with 12-24 GB VRAM
- **Storage**: 100+ GB available space

### Software Requirements
- ROS 2 Humble Hawksbill (or later)
- Gazebo Garden (or compatible version)
- Unity Hub with Unity 2022.3 LTS
- NVIDIA Isaac Sim (latest version)
- Node.js 18+ and npm
- Python 3.10+
- Git

## Setup Instructions

### 1. Install ROS 2 Humble
```bash
# Follow official ROS 2 Humble installation guide
sudo apt update && sudo apt install -y locales
sudo locale-gen en_US.UTF-8
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install ros-humble-desktop
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
```

### 2. Install Docusaurus
```bash
npm init docusaurus@latest docs classic
cd docs
npm install
```

### 3. Set up Simulation Environments
```bash
# For Gazebo
sudo apt install ros-humble-gazebo-*
sudo apt install gazebo

# For Unity (requires Unity Hub)
# Download and install Unity Hub from Unity's website
# Install Unity 2022.3 LTS
```

### 4. Configure NVIDIA Isaac Sim
- Download Isaac Sim from NVIDIA Developer website
- Follow installation instructions for your platform
- Verify installation with provided examples

## Running the Book Locally

### 1. Clone the repository
```bash
git clone [repository-url]
cd [repository-name]
```

### 2. Install dependencies
```bash
cd docs
npm install
```

### 3. Start the development server
```bash
npm start
```

### 4. Access the book
Open your browser to `http://localhost:3000` to view the book.

## Module-Specific Setup

### Module 1: ROS 2 Basics
- Source ROS environment: `source /opt/ros/humble/setup.bash`
- Create a workspace: `mkdir -p ~/ros2_ws/src && cd ~/ros2_ws`
- Build: `colcon build`

### Module 2: Digital Twin Simulation
- Launch Gazebo: `ros2 launch gazebo_ros empty_world.launch.py`
- For Unity: Open the project in Unity Hub and press Play

### Module 3: AI Perception
- Launch Isaac Sim with provided examples
- Ensure CUDA and cuDNN are properly configured

### Module 4: Vision-Language-Action
- Set up Whisper and LLM API access
- Configure ROS 2 bridges for AI integration

## Common Issues and Solutions

### ROS 2 Installation Issues
- Ensure locale is set: `export LANG=en_US.UTF-8`
- Check Python3 availability: `python3 --version`

### GPU Issues
- Verify NVIDIA drivers: `nvidia-smi`
- Ensure CUDA compatibility with Isaac Sim

### Docusaurus Build Issues
- Clear cache: `npm run clear`
- Reinstall dependencies: `rm -rf node_modules && npm install`

## Next Steps
1. Start with Module 1: ROS 2 fundamentals
2. Progress through each module sequentially
3. Complete the capstone project integrating all concepts