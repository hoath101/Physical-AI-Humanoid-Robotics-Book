import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/docs',
    component: ComponentCreator('/docs', 'f7b'),
    routes: [
      {
        path: '/docs',
        component: ComponentCreator('/docs', 'e09'),
        routes: [
          {
            path: '/docs',
            component: ComponentCreator('/docs', '45a'),
            routes: [
              {
                path: '/docs/capstone-project/',
                component: ComponentCreator('/docs/capstone-project/', '9b8'),
                exact: true
              },
              {
                path: '/docs/capstone-project/autonomous-humanoid',
                component: ComponentCreator('/docs/capstone-project/autonomous-humanoid', '006'),
                exact: true
              },
              {
                path: '/docs/intro',
                component: ComponentCreator('/docs/intro', '61d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-1-ros2/',
                component: ComponentCreator('/docs/module-1-ros2/', 'd34'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-1-ros2/installation-setup',
                component: ComponentCreator('/docs/module-1-ros2/installation-setup', '28f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-1-ros2/nodes-topics-services',
                component: ComponentCreator('/docs/module-1-ros2/nodes-topics-services', '51a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-1-ros2/practical-exercises',
                component: ComponentCreator('/docs/module-1-ros2/practical-exercises', '993'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-1-ros2/rclpy-basics',
                component: ComponentCreator('/docs/module-1-ros2/rclpy-basics', 'ece'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-1-ros2/urdf-description',
                component: ComponentCreator('/docs/module-1-ros2/urdf-description', 'f1c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-2-digital-twin/',
                component: ComponentCreator('/docs/module-2-digital-twin/', 'b7a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-2-digital-twin/gazebo-simulation',
                component: ComponentCreator('/docs/module-2-digital-twin/gazebo-simulation', '14b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-2-digital-twin/gazebo-world-setup',
                component: ComponentCreator('/docs/module-2-digital-twin/gazebo-world-setup', 'c6e'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-2-digital-twin/physics-collisions',
                component: ComponentCreator('/docs/module-2-digital-twin/physics-collisions', '410'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-2-digital-twin/practical-exercises',
                component: ComponentCreator('/docs/module-2-digital-twin/practical-exercises', 'b53'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-2-digital-twin/sensor-simulation',
                component: ComponentCreator('/docs/module-2-digital-twin/sensor-simulation', 'b0c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-2-digital-twin/unity-scene-setup',
                component: ComponentCreator('/docs/module-2-digital-twin/unity-scene-setup', '5d3'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-2-digital-twin/unity-visualization',
                component: ComponentCreator('/docs/module-2-digital-twin/unity-visualization', 'dbe'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-2-digital-twin/urdf-validation',
                component: ComponentCreator('/docs/module-2-digital-twin/urdf-validation', '286'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-3-ai-perception/',
                component: ComponentCreator('/docs/module-3-ai-perception/', '49d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-3-ai-perception/installation-setup',
                component: ComponentCreator('/docs/module-3-ai-perception/installation-setup', '67a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-3-ai-perception/isaac-sim',
                component: ComponentCreator('/docs/module-3-ai-perception/isaac-sim', 'e1d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-3-ai-perception/isaac-sim-fundamentals',
                component: ComponentCreator('/docs/module-3-ai-perception/isaac-sim-fundamentals', '3f2'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-3-ai-perception/nav2-locomotion',
                component: ComponentCreator('/docs/module-3-ai-perception/nav2-locomotion', 'fad'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-3-ai-perception/navigation-planning-obstacle-avoidance',
                component: ComponentCreator('/docs/module-3-ai-perception/navigation-planning-obstacle-avoidance', 'b75'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-3-ai-perception/object-detection-localization',
                component: ComponentCreator('/docs/module-3-ai-perception/object-detection-localization', 'ca8'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-3-ai-perception/perception-navigation-pipeline-diagrams',
                component: ComponentCreator('/docs/module-3-ai-perception/perception-navigation-pipeline-diagrams', '457'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-3-ai-perception/practical-exercises-isaac-ai',
                component: ComponentCreator('/docs/module-3-ai-perception/practical-exercises-isaac-ai', 'f71'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-3-ai-perception/vslam-navigation',
                component: ComponentCreator('/docs/module-3-ai-perception/vslam-navigation', 'ac3'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-4-vla/',
                component: ComponentCreator('/docs/module-4-vla/', 'c89'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-4-vla/humanoid-locomotion-control',
                component: ComponentCreator('/docs/module-4-vla/humanoid-locomotion-control', '85e'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-4-vla/isaac-ros-integration',
                component: ComponentCreator('/docs/module-4-vla/isaac-ros-integration', 'ba0'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-4-vla/isaac-sim-fundamentals',
                component: ComponentCreator('/docs/module-4-vla/isaac-sim-fundamentals', '2bc'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-4-vla/llm-planning',
                component: ComponentCreator('/docs/module-4-vla/llm-planning', '764'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-4-vla/multimodal-perception',
                component: ComponentCreator('/docs/module-4-vla/multimodal-perception', '04c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-4-vla/summary',
                component: ComponentCreator('/docs/module-4-vla/summary', 'c1c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-4-vla/vla-architecture-diagrams',
                component: ComponentCreator('/docs/module-4-vla/vla-architecture-diagrams', 'c73'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-4-vla/vslam-navigation',
                component: ComponentCreator('/docs/module-4-vla/vslam-navigation', 'fbd'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-4-vla/whisper-speech',
                component: ComponentCreator('/docs/module-4-vla/whisper-speech', '7f5'),
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
