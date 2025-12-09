# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository is for the "Physical AI & Humanoid Robotics" educational book built with Docusaurus. The project teaches students to bridge AI systems with physical humanoid robots using ROS 2, Gazebo, Unity, and NVIDIA Isaac. The book is structured in 4 modules plus a capstone project, delivered as a Docusaurus-based documentation site.

## Core Principles and Standards

**Core principles:**
-   Technical accuracy in robotics and AI concepts
-   Clarity for computer science and AI students
-   Reproducibility of all examples and tutorials
-   Alignment with Physical AI (embodied intelligence)

**Key standards:**
-   All robotics concepts must be factually correct and from authoritative sources
-   Citation format: APA 7th edition
-   Writing clarity: Flesch-Kincaid grade 10-12 technical level
-   All code examples must be executable on ROS 2 Humble or later
-   Simulation workflows must work in Gazebo/Unity environments

## Common Development Tasks

Since this is a Docusaurus-based documentation site, typical development tasks include:

-   **Local development:** `cd docs && npm start` (starts development server)
-   **Build:** `cd docs && npm run build` (creates static site)
-   **Deploy:** `cd docs && npm run deploy` (deploys to GitHub Pages)
-   **Content creation:** Adding/editing Markdown files in `docs/` directory
-   **Module development:** Creating content for ROS 2, Digital Twin, AI Perception, VLA modules
-   **Version Control:** Standard Git commands (`git status`, `git add`, `git commit`, `git push`)

## High-Level Code Architecture and Structure

The Docusaurus-based book repository follows this structure:

-   `docs/`: Main Docusaurus documentation source
    -   `docs/intro.md`: Book introduction
    -   `docs/module-1-ros2/`: ROS 2 fundamentals module
    -   `docs/module-2-digital-twin/`: Digital twin simulation module
    -   `docs/module-3-ai-perception/`: AI perception and navigation module
    -   `docs/module-4-vla/`: Vision-Language-Action module
    -   `docs/capstone-project/`: Capstone integration project
-   `docusaurus.config.js`: Docusaurus site configuration
-   `sidebars.js`: Navigation sidebar configuration
-   `package.json`: NPM dependencies for Docusaurus
-   `src/`: Custom React components and pages
-   `static/`: Static assets like images
-   `specs/`: Feature specifications and plans
