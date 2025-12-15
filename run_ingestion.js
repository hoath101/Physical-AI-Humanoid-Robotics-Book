const fs = require('fs').promises;
const path = require('path');
const matter = require('gray-matter');

// This script processes the book content from the docs directory
// and prepares it for the RAG system
async function processBookContent() {
  try {
    // Read the docs directory
    const docsPath = path.join(__dirname, 'docs');

    // Check if docs directory exists
    let docsExists = false;
    try {
      await fs.access(docsPath);
      docsExists = true;
    } catch {
      console.log('Docs directory not found, creating sample content...');
    }

    if (docsExists) {
      // Process all markdown files in the docs directory
      const files = await getFiles(docsPath);
      const bookContent = {};

      for (const file of files) {
        if (file.endsWith('.md') || file.endsWith('.mdx')) {
          try {
            const content = await fs.readFile(file, 'utf8');
            const parsed = matter(content);
            const relativePath = path.relative(docsPath, file);
            const sectionKey = relativePath.replace(/\//g, '-').replace(/\.mdx?$/, '');

            bookContent[sectionKey] = {
              title: parsed.data.title || sectionKey,
              content: parsed.content,
              path: relativePath,
              description: parsed.data.description || ''
            };
          } catch (error) {
            console.error(`Error processing file ${file}:`, error);
          }
        }
      }

      // Ensure data directory exists
      await fs.mkdir(path.join(__dirname, 'data'), { recursive: true });

      // Save the processed content
      await fs.writeFile(
        path.join(__dirname, 'data', 'book-content.json'),
        JSON.stringify(bookContent, null, 2)
      );

      console.log(`Processed ${Object.keys(bookContent).length} book sections`);
    } else {
      // Create sample content for demonstration
      const sampleContent = {
        "intro": {
          title: "Introduction to Physical AI",
          content: "Physical AI represents the integration of artificial intelligence with physical robotic systems. This book explores how AI algorithms can be embodied in physical robotic forms, enabling them to interact with the real world through perception, decision-making, and action.",
          path: "intro.md"
        },
        "ros2-fundamentals": {
          title: "ROS 2 Fundamentals",
          content: "ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software. It is a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms. ROS 2 provides a publish-subscribe messaging model, services, and actions for inter-process communication.",
          path: "module-1-ros2/intro.md"
        },
        "digital-twin-simulation": {
          title: "Digital Twin Simulation",
          content: "A digital twin is a virtual representation of a physical object or system that spans its lifecycle. It is updated with real-time data and used to reflect changes to the physical counterpart. In robotics, digital twins enable simulation, testing, and optimization of robot behaviors in virtual environments before deployment in the real world.",
          path: "module-2-digital-twin/intro.md"
        },
        "ai-perception-navigation": {
          title: "AI Perception and Navigation",
          content: "AI perception in robotics involves using sensors and algorithms to understand the environment. This includes computer vision for visual perception, LIDAR for distance measurement, and other sensors for environmental awareness. Navigation systems use this perceptual data to plan and execute movement through space.",
          path: "module-3-ai-perception/intro.md"
        },
        "vla-systems": {
          title: "Vision-Language-Action Systems",
          content: "Vision-Language-Action (VLA) systems integrate visual perception, natural language understanding, and motor control to enable robots to perform complex tasks based on human instructions. These systems combine computer vision, natural language processing, and robotics control to create embodied AI.",
          path: "module-4-vla/intro.md"
        }
      };

      // Ensure data directory exists
      await fs.mkdir(path.join(__dirname, 'data'), { recursive: true });

      await fs.writeFile(
        path.join(__dirname, 'data', 'book-content.json'),
        JSON.stringify(sampleContent, null, 2)
      );

      console.log('Created sample book content for demonstration');
    }
  } catch (error) {
    console.error('Error processing book content:', error);
  }
}

// Helper function to recursively get all files in a directory
async function getFiles(dir) {
  const dirents = await fs.readdir(dir, { withFileTypes: true });
  const files = await Promise.all(
    dirents.map(dirent => {
      const res = path.join(dir, dirent.name);
      return dirent.isDirectory() ? getFiles(res) : res;
    })
  );
  return Array.prototype.flat.call(files);
}

// Run the processing function
processBookContent();