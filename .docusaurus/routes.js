import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/__docusaurus/debug',
    component: ComponentCreator('/__docusaurus/debug', '5ff'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/config',
    component: ComponentCreator('/__docusaurus/debug/config', '5ba'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/content',
    component: ComponentCreator('/__docusaurus/debug/content', 'a2b'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/globalData',
    component: ComponentCreator('/__docusaurus/debug/globalData', 'c3c'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/metadata',
    component: ComponentCreator('/__docusaurus/debug/metadata', '156'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/registry',
    component: ComponentCreator('/__docusaurus/debug/registry', '88c'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/routes',
    component: ComponentCreator('/__docusaurus/debug/routes', '000'),
    exact: true
  },
  {
    path: '/',
    component: ComponentCreator('/', '040'),
    routes: [
      {
        path: '/',
        component: ComponentCreator('/', 'f80'),
        routes: [
          {
            path: '/',
            component: ComponentCreator('/', '7e6'),
            routes: [
              {
                path: '/capstone-project/',
                component: ComponentCreator('/capstone-project/', 'c4c'),
                exact: true
              },
              {
                path: '/capstone-project/autonomous-humanoid',
                component: ComponentCreator('/capstone-project/autonomous-humanoid', '81e'),
                exact: true
              },
              {
                path: '/module-1-ros2/',
                component: ComponentCreator('/module-1-ros2/', 'a0f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/module-1-ros2/installation-setup',
                component: ComponentCreator('/module-1-ros2/installation-setup', '9b8'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/module-1-ros2/nodes-topics-services',
                component: ComponentCreator('/module-1-ros2/nodes-topics-services', '332'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/module-1-ros2/practical-exercises',
                component: ComponentCreator('/module-1-ros2/practical-exercises', 'b25'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/module-1-ros2/rclpy-basics',
                component: ComponentCreator('/module-1-ros2/rclpy-basics', '66f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/module-1-ros2/urdf-description',
                component: ComponentCreator('/module-1-ros2/urdf-description', '24f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/module-2-digital-twin/',
                component: ComponentCreator('/module-2-digital-twin/', 'ad3'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/module-2-digital-twin/gazebo-simulation',
                component: ComponentCreator('/module-2-digital-twin/gazebo-simulation', '9ad'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/module-2-digital-twin/gazebo-world-setup',
                component: ComponentCreator('/module-2-digital-twin/gazebo-world-setup', '904'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/module-2-digital-twin/physics-collisions',
                component: ComponentCreator('/module-2-digital-twin/physics-collisions', '6ec'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/module-2-digital-twin/practical-exercises',
                component: ComponentCreator('/module-2-digital-twin/practical-exercises', '5b4'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/module-2-digital-twin/sensor-simulation',
                component: ComponentCreator('/module-2-digital-twin/sensor-simulation', 'aaa'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/module-2-digital-twin/unity-scene-setup',
                component: ComponentCreator('/module-2-digital-twin/unity-scene-setup', '480'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/module-2-digital-twin/unity-visualization',
                component: ComponentCreator('/module-2-digital-twin/unity-visualization', 'bc9'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/module-2-digital-twin/urdf-validation',
                component: ComponentCreator('/module-2-digital-twin/urdf-validation', '37b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/module-3-ai-perception/',
                component: ComponentCreator('/module-3-ai-perception/', 'b15'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/module-3-ai-perception/installation-setup',
                component: ComponentCreator('/module-3-ai-perception/installation-setup', '0d6'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/module-3-ai-perception/isaac-sim',
                component: ComponentCreator('/module-3-ai-perception/isaac-sim', 'b8e'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/module-3-ai-perception/isaac-sim-fundamentals',
                component: ComponentCreator('/module-3-ai-perception/isaac-sim-fundamentals', '7fa'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/module-3-ai-perception/nav2-locomotion',
                component: ComponentCreator('/module-3-ai-perception/nav2-locomotion', 'c24'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/module-3-ai-perception/navigation-planning-obstacle-avoidance',
                component: ComponentCreator('/module-3-ai-perception/navigation-planning-obstacle-avoidance', 'c6b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/module-3-ai-perception/object-detection-localization',
                component: ComponentCreator('/module-3-ai-perception/object-detection-localization', 'c99'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/module-3-ai-perception/perception-navigation-pipeline-diagrams',
                component: ComponentCreator('/module-3-ai-perception/perception-navigation-pipeline-diagrams', '15f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/module-3-ai-perception/practical-exercises-isaac-ai',
                component: ComponentCreator('/module-3-ai-perception/practical-exercises-isaac-ai', 'b60'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/module-3-ai-perception/vslam-navigation',
                component: ComponentCreator('/module-3-ai-perception/vslam-navigation', 'b3e'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/module-4-vla/',
                component: ComponentCreator('/module-4-vla/', 'f30'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/module-4-vla/humanoid-locomotion-control',
                component: ComponentCreator('/module-4-vla/humanoid-locomotion-control', '645'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/module-4-vla/isaac-ros-integration',
                component: ComponentCreator('/module-4-vla/isaac-ros-integration', 'c38'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/module-4-vla/isaac-sim-fundamentals',
                component: ComponentCreator('/module-4-vla/isaac-sim-fundamentals', '2c8'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/module-4-vla/llm-planning',
                component: ComponentCreator('/module-4-vla/llm-planning', 'd32'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/module-4-vla/multimodal-perception',
                component: ComponentCreator('/module-4-vla/multimodal-perception', '7db'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/module-4-vla/summary',
                component: ComponentCreator('/module-4-vla/summary', '0d0'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/module-4-vla/vla-architecture-diagrams',
                component: ComponentCreator('/module-4-vla/vla-architecture-diagrams', 'fda'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/module-4-vla/vslam-navigation',
                component: ComponentCreator('/module-4-vla/vslam-navigation', '58a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/module-4-vla/whisper-speech',
                component: ComponentCreator('/module-4-vla/whisper-speech', '0db'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/',
                component: ComponentCreator('/', 'fc9'),
                exact: true,
                sidebar: "tutorialSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
