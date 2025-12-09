import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/intro',
    component: ComponentCreator('/intro', 'c7c'),
    routes: [
      {
        path: '/intro',
        component: ComponentCreator('/intro', '235'),
        routes: [
          {
            path: '/intro',
            component: ComponentCreator('/intro', '94c'),
            routes: [
              {
                path: '/intro/capstone-project/',
                component: ComponentCreator('/intro/capstone-project/', '657'),
                exact: true
              },
              {
                path: '/intro/capstone-project/autonomous-humanoid',
                component: ComponentCreator('/intro/capstone-project/autonomous-humanoid', '569'),
                exact: true
              },
              {
                path: '/intro/intro',
                component: ComponentCreator('/intro/intro', '096'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/intro/module-1-ros2/',
                component: ComponentCreator('/intro/module-1-ros2/', 'd61'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/intro/module-1-ros2/installation-setup',
                component: ComponentCreator('/intro/module-1-ros2/installation-setup', '45a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/intro/module-1-ros2/nodes-topics-services',
                component: ComponentCreator('/intro/module-1-ros2/nodes-topics-services', '37b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/intro/module-1-ros2/practical-exercises',
                component: ComponentCreator('/intro/module-1-ros2/practical-exercises', 'e51'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/intro/module-1-ros2/rclpy-basics',
                component: ComponentCreator('/intro/module-1-ros2/rclpy-basics', 'ca5'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/intro/module-1-ros2/urdf-description',
                component: ComponentCreator('/intro/module-1-ros2/urdf-description', '3e1'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/intro/module-2-digital-twin/',
                component: ComponentCreator('/intro/module-2-digital-twin/', 'a8a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/intro/module-2-digital-twin/gazebo-simulation',
                component: ComponentCreator('/intro/module-2-digital-twin/gazebo-simulation', 'a7a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/intro/module-2-digital-twin/gazebo-world-setup',
                component: ComponentCreator('/intro/module-2-digital-twin/gazebo-world-setup', 'd8f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/intro/module-2-digital-twin/physics-collisions',
                component: ComponentCreator('/intro/module-2-digital-twin/physics-collisions', '7f8'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/intro/module-2-digital-twin/practical-exercises',
                component: ComponentCreator('/intro/module-2-digital-twin/practical-exercises', '769'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/intro/module-2-digital-twin/sensor-simulation',
                component: ComponentCreator('/intro/module-2-digital-twin/sensor-simulation', 'bfd'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/intro/module-2-digital-twin/unity-scene-setup',
                component: ComponentCreator('/intro/module-2-digital-twin/unity-scene-setup', 'e96'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/intro/module-2-digital-twin/unity-visualization',
                component: ComponentCreator('/intro/module-2-digital-twin/unity-visualization', '9da'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/intro/module-2-digital-twin/urdf-validation',
                component: ComponentCreator('/intro/module-2-digital-twin/urdf-validation', '944'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/intro/module-3-ai-perception/',
                component: ComponentCreator('/intro/module-3-ai-perception/', '3ad'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/intro/module-3-ai-perception/installation-setup',
                component: ComponentCreator('/intro/module-3-ai-perception/installation-setup', '45b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/intro/module-3-ai-perception/isaac-sim',
                component: ComponentCreator('/intro/module-3-ai-perception/isaac-sim', 'a97'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/intro/module-3-ai-perception/isaac-sim-fundamentals',
                component: ComponentCreator('/intro/module-3-ai-perception/isaac-sim-fundamentals', '1d5'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/intro/module-3-ai-perception/nav2-locomotion',
                component: ComponentCreator('/intro/module-3-ai-perception/nav2-locomotion', '7f0'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/intro/module-3-ai-perception/navigation-planning-obstacle-avoidance',
                component: ComponentCreator('/intro/module-3-ai-perception/navigation-planning-obstacle-avoidance', '847'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/intro/module-3-ai-perception/object-detection-localization',
                component: ComponentCreator('/intro/module-3-ai-perception/object-detection-localization', 'edb'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/intro/module-3-ai-perception/perception-navigation-pipeline-diagrams',
                component: ComponentCreator('/intro/module-3-ai-perception/perception-navigation-pipeline-diagrams', '4d2'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/intro/module-3-ai-perception/practical-exercises-isaac-ai',
                component: ComponentCreator('/intro/module-3-ai-perception/practical-exercises-isaac-ai', '6b3'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/intro/module-3-ai-perception/vslam-navigation',
                component: ComponentCreator('/intro/module-3-ai-perception/vslam-navigation', '6e1'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/intro/module-4-vla/',
                component: ComponentCreator('/intro/module-4-vla/', 'aed'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/intro/module-4-vla/humanoid-locomotion-control',
                component: ComponentCreator('/intro/module-4-vla/humanoid-locomotion-control', '7c6'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/intro/module-4-vla/isaac-ros-integration',
                component: ComponentCreator('/intro/module-4-vla/isaac-ros-integration', '280'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/intro/module-4-vla/isaac-sim-fundamentals',
                component: ComponentCreator('/intro/module-4-vla/isaac-sim-fundamentals', 'a67'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/intro/module-4-vla/llm-planning',
                component: ComponentCreator('/intro/module-4-vla/llm-planning', '1f7'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/intro/module-4-vla/multimodal-perception',
                component: ComponentCreator('/intro/module-4-vla/multimodal-perception', 'a90'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/intro/module-4-vla/summary',
                component: ComponentCreator('/intro/module-4-vla/summary', '455'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/intro/module-4-vla/vla-architecture-diagrams',
                component: ComponentCreator('/intro/module-4-vla/vla-architecture-diagrams', '77c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/intro/module-4-vla/vslam-navigation',
                component: ComponentCreator('/intro/module-4-vla/vslam-navigation', 'b2b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/intro/module-4-vla/whisper-speech',
                component: ComponentCreator('/intro/module-4-vla/whisper-speech', '014'),
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
