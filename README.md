# Physical AI & Humanoid Robotics Educational Book

This repository contains a comprehensive educational book on Physical AI and Humanoid Robotics, designed to teach students how to bridge digital intelligence with physical robotic bodies using NVIDIA Isaac technologies.

## Overview

This book teaches students to build AI-powered humanoid robots that can perceive, navigate, and interact with the physical world using:
- ROS 2 for robotic middleware
- Isaac Sim for high-fidelity simulation
- Isaac ROS for perception and navigation
- NVIDIA AI for vision-language-action tasks

## Book Structure

### Module 1: The Robotic Nervous System (ROS 2)
- ROS 2 nodes, topics, services, and actions
- rclpy for Python-ROS bridging
- URDF for humanoid robot description
- Practical exercises with ROS 2 commands

### Module 2: The Digital Twin (Gazebo & Unity)
- Physics simulation with gravity and collisions
- Sensor simulation (LiDAR, IMU, depth camera)
- Unity-based visualization
- Digital twin validation techniques

### Module 3: The AI-Robot Brain (NVIDIA Isaac)
- Isaac Sim for photorealistic simulation
- Isaac ROS for VSLAM and navigation
- Nav2 for humanoid locomotion
- Perception pipeline implementation

### Module 4: Vision-Language-Action (VLA)
- Whisper for speech commands
- LLM-driven planning
- Multimodal perception
- Voice-to-action pipeline

## Learning Objectives

After completing this book, students will be able to:
- Deploy ROS 2 nodes to control humanoid robots in simulation
- Build and validate digital twins using Gazebo and Unity
- Integrate NVIDIA Isaac AI for perception and navigation
- Execute Vision-Language-Action tasks using LLMs and Whisper
- Complete a capstone project: an autonomous humanoid robot

## Prerequisites

- Basic Python programming knowledge
- Understanding of robotics concepts
- Access to NVIDIA GPU for Isaac Sim (recommended)
- ROS 2 Humble installation
- Isaac Sim installation

## Getting Started

1. Install ROS 2 Humble
2. Install Isaac Sim
3. Clone this repository
4. Navigate to the book directory
5. Start with Module 1 to learn ROS 2 fundamentals

## Technical Requirements

### Hardware
- NVIDIA RTX 4080/4090 or professional GPU (A40, A6000) with 10GB+ VRAM
- 32GB+ system memory (64GB+ recommended)
- 8+ core CPU (Intel i7 / AMD Ryzen 7 or better)

### Software
- Ubuntu 22.04 LTS or Windows 10/11
- ROS 2 Humble Hawksbill
- Isaac Sim 2024.1+
- NVIDIA GPU drivers (520+)
- CUDA 11.8 or 12.x

## Repository Structure

```
├── docs/                   # Docusaurus documentation source
│   ├── module-1-ros2/      # ROS 2 fundamentals
│   ├── module-2-digital-twin/ # Digital twin simulation
│   ├── module-3-ai-perception/ # AI perception and navigation
│   └── module-4-vla/      # Vision-Language-Action
├── specs/                 # Specification files
│   └── 001-physical-ai-humanoids/ # Main specification
├── history/               # Historical records
│   └── prompts/           # Prompt history records
└── .specify/              # Automation tools and templates
```

## Contributing

This book is designed as an educational resource. Contributions are welcome for:
- Technical corrections
- Additional examples
- Updated content for new Isaac versions
- Improved exercises

## License

This educational content is provided for learning purposes in the field of Physical AI and Humanoid Robotics.

## Acknowledgments

This book leverages NVIDIA Isaac technologies and is designed to support the development of Physical AI capabilities in humanoid robotics.

## RAG Chatbot Integration

This repository includes a Retrieval-Augmented Generation (RAG) chatbot that allows users to ask questions about the book content. The chatbot uses:

- OpenAI GPT models for natural language understanding
- Vector embeddings for semantic search through book content
- Python FastAPI backend with advanced RAG capabilities
- Qdrant Cloud for vector storage and retrieval
- Docusaurus integration with floating chat widget for seamless user experience

The RAG chatbot enables students to:
- Ask questions about specific modules and concepts
- Get explanations of complex robotics and AI topics
- Search through the entire book content efficiently
- Receive context-aware responses based on the educational material

To use the chatbot functionality, see the setup instructions in the documentation or use the embedded widget on the website.