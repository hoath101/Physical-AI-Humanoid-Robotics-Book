<!--
Sync Impact Report:
Version change: 0.0.0 → 0.1.0 (Minor: Initial population of the constitution)
Modified principles: N/A (all new)
Added sections: Mission, Standards, Constraints, Success Criteria
Removed sections: N/A
Templates requiring updates:
  - .specify/templates/plan-template.md: ⚠ pending
  - .specify/templates/spec-template.md: ⚠ pending
  - .specify/templates/tasks-template.md: ⚠ pending
  - README.md: ⚠ pending
Follow-up TODOs:
  - TODO(RATIFICATION_DATE): Original adoption date of this constitution.
-->
# Physical AI & Humanoid Robotics — Capstone Book Constitution

## Mission

- Produce a comprehensive, academically rigorous book that teaches students how to bridge AI systems with physical humanoid robots using ROS 2, Gazebo, Unity, and NVIDIA Isaac.
- Provide a unified learning pathway from robotic middleware → simulation → perception → VLA (Vision-Language-Action) → autonomous humanoid behaviors.

## Core Principles

### Technical Accuracy
All explanations of robotics, control systems, simulations, and AI methodologies must be factually correct and derived from authoritative sources (ROS 2 docs, NVIDIA Isaac technical papers, robotics textbooks, peer-reviewed research).

### Clarity
Content must be understandable for students with a computer science or AI background, using precise but accessible technical language.

### Reproducibility
All tutorials, code examples, and robotics workflows must be replicable by students using standard ROS 2 and simulation environments.

### Alignment with Physical AI
Every module must emphasize embodied intelligence — connecting the “digital brain” (AI models) with the “physical body” (humanoid robots and simulations).

### Rigor
Preference must be given to robotics engineering literature, official documentation, and peer-reviewed sources whenever available.

## Standards

All robotics concepts (URDF, ROS graph, Nav2, VSLAM, IK/FK, simulations) must include diagrams or structural explanations.
Code examples must follow ROS 2 (rclpy) best practices and be executable.
Simulation workflows must be validated in Gazebo or Unity digital twin environments.
AI/robotics interfaces (Whisper, LLM planning, Isaac ROS, VLA pipelines) must include step-by-step processes.
Minimum 40% of referenced materials must be from academic research or robotics conference proceedings (ICRA, IROS, RSS).
Citation format: APA 7th edition.
Writing level: Flesch-Kincaid Grade 10–12 technical clarity.
Plagiarism tolerance: 0% (must pass plagiarism scan before publishing).

## Constraints

Total book length: 20,000–30,000 words (4 modules + capstone).
Minimum 20 high-quality sources across robotics, AI, simulation, and VLA.
All code snippets must run on ROS 2 Humble or later.
Simulations must work in either Gazebo Classic, Gazebo Fortress, or Isaac Sim.
Final publish format: Docusaurus site deployed to GitHub Pages.
All assets (URDFs, worlds, example code) must be included in the repository.

## Success Criteria

- Students can deploy ROS 2 nodes, control a humanoid robot, and build a functioning digital twin.
- Simulated humanoid robots can navigate, perceive, and execute tasks using VLA pipelines.
- Voice-to-action pipeline (Whisper → LLM → ROS 2) is demonstrated successfully.
- Capstone robot can: receive a natural-language command, plan a path (Nav2), avoid obstacles, detect an object, and perform a manipulation task.
- All technical explanations are validated by robotics experts or documentation.
- Book passes fact-checking, plagiarism scan, and technical reproducibility tests.

## Governance

The constitution supersedes all other practices. Amendments require documentation, approval, and a migration plan. All PRs/reviews must verify compliance. Complexity must be justified.

**Version**: 0.1.0 | **Ratified**: TODO(RATIFICATION_DATE): Original adoption date of this constitution. | **Last Amended**: 2025-12-06
