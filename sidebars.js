/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    'intro',

    {
      type: 'category',
      label: 'Module 1 – ROS2',
      items: [
        'module-1-ros2/module-1-index',
        'module-1-ros2/installation-setup',
        'module-1-ros2/nodes-topics-services',
        'module-1-ros2/practical-exercises',
        'module-1-ros2/rclpy-basics',
        'module-1-ros2/urdf-description',
      ],
    },

    {
      type: 'category',
      label: 'Module 2 – Digital Twin',
      items: [
        'module-2-digital-twin/module-2-index',
        'module-2-digital-twin/gazebo-simulation',
        'module-2-digital-twin/gazebo-world-setup',
        'module-2-digital-twin/physics-collisions',
        'module-2-digital-twin/practical-exercises',
        'module-2-digital-twin/sensor-simulation',
        'module-2-digital-twin/unity-scene-setup',
        'module-2-digital-twin/unity-visualization',
        'module-2-digital-twin/urdf-validation',
      ],
    },

    {
      type: 'category',
      label: 'Module 3 – AI Perception',
      items: [
        'module-3-ai-perception/module-3-index',
        'module-3-ai-perception/installation-setup',
        'module-3-ai-perception/isaac-sim',
        'module-3-ai-perception/isaac-sim-fundamentals',
        'module-3-ai-perception/object-detection-localization',
        'module-3-ai-perception/navigation-planning-obstacle-avoidance',
        'module-3-ai-perception/nav2-locomotion',
        'module-3-ai-perception/perception-navigation-pipeline-diagrams',
        'module-3-ai-perception/practical-exercises-isaac-ai',
        'module-3-ai-perception/vslam-navigation',
      ],
    },

    {
      type: 'category',
      label: 'Module 4 – VLA',
      items: [
        'module-4-vla/module-4-index',
        'module-4-vla/summary',
        'module-4-vla/humanoid-locomotion-control',
        'module-4-vla/isaac-ros-integration',
        'module-4-vla/isaac-sim-fundamentals',
        'module-4-vla/llm-planning',
        'module-4-vla/multimodal-perception',
        'module-4-vla/vla-architecture-diagrams',
        'module-4-vla/vslam-navigation',
        'module-4-vla/whisper-speech',
      ],
    },
  ],
};

module.exports = sidebars;
