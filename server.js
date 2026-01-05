/**
 * Node.js Express Server
 * Handles file uploads and proxies requests to Python API
 */

const express = require('express');
const multer = require('multer');
const axios = require('axios');
const { v4: uuidv4 } = require('uuid');
const path = require('path');
const fs = require('fs');
const archiver = require('archiver');
const cors = require('cors');
const FormData = require('form-data');

const app = express();
const PORT = 3000;
const PYTHON_API_URL = 'http://localhost:5000';

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Configure multer for file uploads
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        const uploadDir = path.join(__dirname, 'uploads', 'temp');
        if (!fs.existsSync(uploadDir)) {
            fs.mkdirSync(uploadDir, { recursive: true });
        }
        cb(null, uploadDir);
    },
    filename: (req, file, cb) => {
        cb(null, file.originalname);
    }
});

const upload = multer({ storage: storage });

// Health check
app.get('/api/health', async (req, res) => {
    try {
        const response = await axios.get(`${PYTHON_API_URL}/health`);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({
            error: 'Python API not available',
            details: error.message
        });
    }
});

// Get logs (Proxy to Python)
app.get('/api/logs', async (req, res) => {
    try {
        const response = await axios.get(`${PYTHON_API_URL}/api/logs`, {
            params: req.query
        });
        res.json(response.data);
    } catch (error) {
        // Don't log 500s for logs to avoid spamming console if python is restarting
        res.status(500).json({ error: 'Failed to fetch logs' });
    }
});

// Extract features from good images
app.post('/api/extract_features', upload.any(), async (req, res) => {
    try {
        const formData = new FormData();

        // Check if files were uploaded
        if (!req.files || req.files.length === 0) {
            return res.status(400).json({ error: 'No files uploaded' });
        }

        // Prepare form data for Python API
        const files = req.files;

        // Check if it's a zip file
        const zipFile = files.find(f => f.originalname.endsWith('.zip'));

        if (zipFile) {
            // Send zip file
            const fileStream = fs.createReadStream(zipFile.path);
            formData.append('zip_file', fileStream, zipFile.originalname);
        } else {
            // Send individual files
            files.forEach(file => {
                const fileStream = fs.createReadStream(file.path);
                formData.append('files', fileStream, file.originalname);
            });
        }

        // Forward to Python API
        const response = await axios.post(
            `${PYTHON_API_URL}/extract_features`,
            formData,
            {
                headers: formData.getHeaders(),
                maxContentLength: Infinity,
                maxBodyLength: Infinity
            }
        );

        // Clean up uploaded files
        files.forEach(file => {
            if (fs.existsSync(file.path)) {
                fs.unlinkSync(file.path);
            }
        });

        res.json(response.data);
    } catch (error) {
        console.error('Error extracting features:', error.message);
        res.status(500).json({
            error: 'Failed to extract features',
            details: error.response?.data || error.message
        });
    }
});

// Save memory bank
app.post('/api/save_memory_bank', async (req, res) => {
    try {
        const response = await axios.post(
            `${PYTHON_API_URL}/save_memory_bank`,
            req.body
        );
        res.json(response.data);
    } catch (error) {
        console.error('Error saving memory bank:', error.message);
        res.status(500).json({
            error: 'Failed to save memory bank',
            details: error.response?.data || error.message
        });
    }
});

// Load memory bank
app.post('/api/load_memory_bank', async (req, res) => {
    try {
        const response = await axios.post(
            `${PYTHON_API_URL}/load_memory_bank`,
            req.body
        );
        res.json(response.data);
    } catch (error) {
        console.error('Error loading memory bank:', error.message);
        res.status(500).json({
            error: 'Failed to load memory bank',
            details: error.response?.data || error.message
        });
    }
});

// List memory banks
app.get('/api/list_memory_banks', async (req, res) => {
    try {
        const response = await axios.get(`${PYTHON_API_URL}/list_memory_banks`);
        res.json(response.data);
    } catch (error) {
        console.error('Error listing memory banks:', error.message);
        res.status(500).json({
            error: 'Failed to list memory banks',
            details: error.response?.data || error.message
        });
    }
});

// Detect anomalies
app.post('/api/detect_anomalies', upload.any(), async (req, res) => {
    try {
        const formData = new FormData();

        if (!req.files || req.files.length === 0) {
            return res.status(400).json({ error: 'No files uploaded' });
        }

        // Generate session ID
        const sessionId = req.body.session_id || uuidv4();
        formData.append('session_id', sessionId);

        const files = req.files;
        const zipFile = files.find(f => f.originalname.endsWith('.zip'));

        // Determine if single image
        const isSingleImage = files.length === 1 && !zipFile;

        if (zipFile) {
            const fileStream = fs.createReadStream(zipFile.path);
            formData.append('zip_file', fileStream, zipFile.originalname);
        } else if (isSingleImage) {
            const fileStream = fs.createReadStream(files[0].path);
            formData.append('single_image', fileStream, files[0].originalname);
        } else {
            files.forEach(file => {
                const fileStream = fs.createReadStream(file.path);
                formData.append('files', fileStream, file.originalname);
            });
        }

        // Forward to Python API
        const response = await axios.post(
            `${PYTHON_API_URL}/detect_anomalies`,
            formData,
            {
                headers: formData.getHeaders(),
                maxContentLength: Infinity,
                maxBodyLength: Infinity
            }
        );

        // Clean up uploaded files
        files.forEach(file => {
            if (fs.existsSync(file.path)) {
                fs.unlinkSync(file.path);
            }
        });

        res.json(response.data);
    } catch (error) {
        console.error('Error detecting anomalies:', error.message);
        res.status(500).json({
            error: 'Failed to detect anomalies',
            details: error.response?.data || error.message
        });
    }
});

// Download results
app.get('/api/download_results/:sessionId', async (req, res) => {
    try {
        const response = await axios.get(
            `${PYTHON_API_URL}/download_results/${req.params.sessionId}`,
            { responseType: 'stream' }
        );

        res.setHeader('Content-Type', 'application/zip');
        res.setHeader(
            'Content-Disposition',
            `attachment; filename="${req.params.sessionId}_results.zip"`
        );

        response.data.pipe(res);
    } catch (error) {
        console.error('Error downloading results:', error.message);
        res.status(500).json({
            error: 'Failed to download results',
            details: error.response?.data || error.message
        });
    }
});

// Get image
app.get('/api/get_image/:sessionId/:filename', async (req, res) => {
    try {
        const response = await axios.get(
            `${PYTHON_API_URL}/get_image/${req.params.sessionId}/${req.params.filename}`,
            { responseType: 'stream' }
        );

        res.setHeader('Content-Type', 'image/png');
        response.data.pipe(res);
    } catch (error) {
        console.error('Error getting image:', error.message);
        res.status(500).json({
            error: 'Failed to get image',
            details: error.response?.data || error.message
        });
    }
});

// Start server
app.listen(PORT, () => {
    console.log('\n' + '='.repeat(60));
    console.log('Anomaly Detection Web Application');
    console.log('='.repeat(60));
    console.log(`\nServer running at http://localhost:${PORT}`);
    console.log(`Python API expected at ${PYTHON_API_URL}`);
    console.log('\nMake sure to start the Python API server first!');
    console.log('  cd python && python api_server.py');
    console.log('\n' + '='.repeat(60) + '\n');
});
