// Main API server file for RAG functionality
const express = require('express');
const cors = require('cors');
const path = require('path');
const ragRoutes = require('./api/rag');

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'data'))); // Serve static data files if needed

// Routes
app.use('/api', ragRoutes);

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'OK', message: 'RAG API is running' });
});

app.listen(PORT, () => {
  console.log(`RAG API server running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`API endpoint: http://localhost:${PORT}/api/rag-query`);
});