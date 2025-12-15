// API for RAG functionality that connects to your book content
// This is a Node.js/Express API for demonstration purposes

const express = require('express');
const path = require('path');
const fs = require('fs').promises;

const router = express.Router();

// Load book content from the data directory
let bookContent = {};

async function loadBookContent() {
  try {
    const contentPath = path.join(__dirname, '..', 'data', 'book-content.json');
    const contentData = await fs.readFile(contentPath, 'utf8');
    bookContent = JSON.parse(contentData);
    console.log(`Loaded ${Object.keys(bookContent).length} book sections for RAG`);
  } catch (error) {
    console.warn('Could not load book content, using sample content:', error.message);
    // Fallback to sample content if the data file doesn't exist
    bookContent = {
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
  }
}

// Load the book content when the module is loaded
loadBookContent();

// Simple search function to find relevant content based on query
function searchContent(query) {
  const queryLower = query.toLowerCase();
  const results = [];

  for (const [key, section] of Object.entries(bookContent)) {
    // Calculate relevance based on how many times query terms appear in the content
    const titleMatches = (section.title.toLowerCase().match(new RegExp(queryLower, 'g')) || []).length;
    const contentMatches = (section.content.toLowerCase().match(new RegExp(queryLower, 'g')) || []).length;
    const totalMatches = titleMatches * 2 + contentMatches; // Title matches count double

    if (totalMatches > 0) {
      results.push({
        title: section.title,
        content: section.content,
        relevance: totalMatches,
        path: section.path
      });
    }
  }

  // Sort by relevance (descending)
  results.sort((a, b) => b.relevance - a.relevance);

  // If no direct matches, return all content as potential matches
  if (results.length === 0) {
    for (const [key, section] of Object.entries(bookContent)) {
      results.push({
        title: section.title,
        content: section.content,
        relevance: 0.5,
        path: section.path
      });
    }
  }

  return results;
}

// RAG API endpoint
router.post('/rag-query', async (req, res) => {
  try {
    const { query } = req.body;

    if (!query) {
      return res.status(400).json({ error: 'Query is required' });
    }

    // Search for relevant content in the book
    const searchResults = searchContent(query);

    if (searchResults.length === 0) {
      return res.status(404).json({
        response: "I couldn't find any relevant content in the Physical AI & Humanoid Robotics book for your query. Please try rephrasing your question.",
        sources: []
      });
    }

    // Generate a response using the most relevant content
    const topResult = searchResults[0];
    let response = `Based on the "${topResult.title}" section of the Physical AI & Humanoid Robotics book:\n\n${topResult.content.substring(0, 1000)}`; // Limit content length

    // Add more context if there are other relevant results
    if (searchResults.length > 1) {
      response += `\n\nFor additional information, also see the "${searchResults[1].title}" section.`;
    }

    res.json({
      response: response,
      sources: searchResults.slice(0, 3).map(result => ({
        title: result.title,
        path: result.path
      }))
    });
  } catch (error) {
    console.error('Error processing RAG query:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

module.exports = router;