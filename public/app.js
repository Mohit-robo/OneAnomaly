// API Base URL
const API_BASE = '/api';

// Global state
let currentSessionId = null;
let memoryBankLoaded = false;

// DOM Elements
const memoryModeRadios = document.querySelectorAll('input[name="memory-mode"]');
const loadSection = document.getElementById('load-section');
const newSection = document.getElementById('new-section');
const memoryBankSelect = document.getElementById('memory-bank-select');
const loadBtn = document.getElementById('load-btn');
const goodImagesUpload = document.getElementById('good-images-upload');
const extractBtn = document.getElementById('extract-btn');
const saveSection = document.getElementById('save-section');
const memoryBankName = document.getElementById('memory-bank-name');
const saveBtn = document.getElementById('save-btn');
const testImagesUpload = document.getElementById('test-images-upload');
const detectBtn = document.getElementById('detect-btn');
const singleResult = document.getElementById('single-result');
const multipleResults = document.getElementById('multiple-results');
const statusText = document.getElementById('status-text');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadMemoryBankList();
    setupEventListeners();
    checkAPIHealth();
});

// Setup Event Listeners
function setupEventListeners() {
    // Memory mode toggle
    memoryModeRadios.forEach(radio => {
        radio.addEventListener('change', (e) => {
            if (e.target.value === 'load') {
                loadSection.style.display = 'block';
                newSection.style.display = 'none';
            } else {
                loadSection.style.display = 'none';
                newSection.style.display = 'block';
            }
        });
    });

    // Load memory bank
    loadBtn.addEventListener('click', loadMemoryBank);

    // Extract features
    extractBtn.addEventListener('click', extractFeatures);

    // Save memory bank
    saveBtn.addEventListener('click', saveMemoryBank);

    // Detect anomalies
    detectBtn.addEventListener('click', detectAnomalies);

    // Save single result
    document.getElementById('save-single-btn')?.addEventListener('click', saveSingleResult);

    // Download all results
    document.getElementById('download-all-btn')?.addEventListener('click', downloadAllResults);
}

// Check API Health
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();

        if (data.status === 'healthy') {
            updateGlobalStatus('Ready', 'success');
        } else {
            updateGlobalStatus('API Error', 'error');
        }
    } catch (error) {
        updateGlobalStatus('API Offline', 'error');
        showStatus('extract-status', 'Make sure Python API server is running!', 'error');
    }
}

// Load Memory Bank List
async function loadMemoryBankList() {
    try {
        const response = await fetch(`${API_BASE}/list_memory_banks`);
        const data = await response.json();

        memoryBankSelect.innerHTML = '<option value="">-- Select a memory bank --</option>';

        data.memory_banks.forEach(bank => {
            const option = document.createElement('option');
            option.value = bank;
            option.textContent = bank;
            memoryBankSelect.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading memory banks:', error);
    }
}

// Load Memory Bank
async function loadMemoryBank() {
    const filename = memoryBankSelect.value;

    if (!filename) {
        showStatus('load-status', 'Please select a memory bank', 'error');
        return;
    }

    showStatus('load-status', 'Loading memory bank...', 'loading');
    updateGlobalStatus('Loading...', 'loading');

    try {
        const response = await fetch(`${API_BASE}/load_memory_bank`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename })
        });

        const data = await response.json();

        if (data.success) {
            memoryBankLoaded = true;
            showStatus('load-status', 'Loaded successfully!', 'success');
            updateGlobalStatus('Memory Bank Loaded', 'success');
        } else {
            showStatus('load-status', `Error: ${data.error}`, 'error');
            updateGlobalStatus('Error', 'error');
        }
    } catch (error) {
        showStatus('load-status', `Error: ${error.message}`, 'error');
        updateGlobalStatus('Error', 'error');
    }
}

// Extract Features
async function extractFeatures() {
    const files = goodImagesUpload.files;

    if (files.length === 0) {
        showStatus('extract-status', 'Please select files', 'error');
        return;
    }

    showStatus('extract-status', 'Extracting features...', 'loading');
    updateGlobalStatus('Extracting...', 'loading');

    const formData = new FormData();

    // Check if we have a single zip file or multiple images
    let hasZip = false;
    for (let i = 0; i < files.length; i++) {
        if (files[i].name.toLowerCase().endsWith('.zip')) {
            hasZip = true;
            break;
        }
    }

    if (hasZip) {
        // If there's a zip, find it precisely
        let zipFile = null;
        for (let i = 0; i < files.length; i++) {
            if (files[i].name.toLowerCase().endsWith('.zip')) {
                zipFile = files[i];
                break;
            }
        }

        if (zipFile) {
            console.log('Zip details:', zipFile.name, 'Size:', zipFile.size, 'Type:', zipFile.type);

            try {
                // Force Blob creation to ensure clean object
                console.log('Creating fresh Blob from file...');
                const cleanBlob = new Blob([zipFile], { type: zipFile.type || 'application/zip' });
                console.log('Clean Blob created:', cleanBlob);

                formData.append('zip_file', cleanBlob, zipFile.name);
                console.log('Append successful!');
            } catch (e) {
                console.error('Critical error appending file:', e);
                showStatus('extract-status', `Critical Browser Error: ${e.message}`, 'error');
                return;
            }
        } else {
            showStatus('extract-status', 'Zip file detected but not found in list.', 'error');
            return;
        }
    } else {
        // Multiple images
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            console.log(`Appending file ${i}:`, file.name, file.type);
            try {
                formData.append('files', file);
            } catch (e) {
                console.error(`Error appending file ${i}:`, file, e);
                showStatus('extract-status', `Browser error reading file ${file.name}: ${e.message}`, 'error');
                return;
            }
        }
    }

    try {
        const response = await fetch(`${API_BASE}/extract_features`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            showStatus('extract-status', 'Success! Features extracted.', 'success');
            updateGlobalStatus('Features Extracted', 'success');
            saveSection.style.display = 'block';
        } else {
            showStatus('extract-status', `Error: ${data.error}`, 'error');
            updateGlobalStatus('Error', 'error');
        }
    } catch (error) {
        showStatus('extract-status', `Error: ${error.message}`, 'error');
        updateGlobalStatus('Error', 'error');
    }
}

// Save Memory Bank
async function saveMemoryBank() {
    const filename = memoryBankName.value.trim();

    if (!filename) {
        showStatus('save-status', 'Please enter a filename', 'error');
        return;
    }

    showStatus('save-status', 'Saving memory bank...', 'loading');

    try {
        const response = await fetch(`${API_BASE}/save_memory_bank`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename })
        });

        const data = await response.json();

        if (data.success) {
            memoryBankLoaded = true;
            showStatus('save-status', 'Saved successfully!', 'success');
            updateGlobalStatus('Memory Bank Saved', 'success');
            loadMemoryBankList(); // Refresh list
        } else {
            showStatus('save-status', `Error: ${data.error}`, 'error');
        }
    } catch (error) {
        showStatus('save-status', `Error: ${error.message}`, 'error');
    }
}

// Detect Anomalies
async function detectAnomalies() {
    if (!memoryBankLoaded) {
        showStatus('detect-status', 'Please load or create a memory bank first!', 'error');
        return;
    }

    const files = testImagesUpload.files;

    if (files.length === 0) {
        showStatus('detect-status', 'Please select test images', 'error');
        return;
    }

    showStatus('detect-status', 'Detecting anomalies...', 'loading');
    updateGlobalStatus('Detecting...', 'loading');

    // Reset Progress Bar
    const progressContainer = document.getElementById('progress-container');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    if (progressContainer) {
        progressContainer.style.display = 'block';
        progressBar.style.width = '0%';
        progressText.textContent = '0%';
    }

    const formData = new FormData();
    currentSessionId = generateSessionId();
    formData.append('session_id', currentSessionId);

    let isSingleImage = false;
    let hasZip = false;

    // Check file types
    for (let i = 0; i < files.length; i++) {
        if (files[i].name.toLowerCase().endsWith('.zip')) {
            hasZip = true;
            break;
        }
    }

    if (hasZip) {
        let zipFile = null;
        for (let i = 0; i < files.length; i++) {
            if (files[i].name.toLowerCase().endsWith('.zip')) {
                zipFile = files[i];
                break;
            }
        }

        if (files.length > 1) {
            showStatus('detect-status', 'Please upload either one ZIP file or images.', 'error');
            return;
        }

        if (zipFile) formData.append('zip_file', zipFile);

    } else if (files.length === 1) {
        isSingleImage = true;
        formData.append('single_image', files[0]);
    } else {
        // Multiple images
        for (let i = 0; i < files.length; i++) {
            formData.append('files', files[i]);
        }
    }

    try {
        const response = await fetch(`${API_BASE}/detect_anomalies`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            showStatus('detect-status', 'Detection complete!', 'success');
            updateGlobalStatus('Detection Complete', 'success');

            if (data.is_single_image) {
                displaySingleResult(data);
            } else {
                displayMultipleResults(data);
            }
        } else {
            let errorMsg = data.error;
            if (data.details) {
                if (typeof data.details === 'object' && data.details.error) {
                    errorMsg += `: ${data.details.error}`;
                } else {
                    errorMsg += `: ${JSON.stringify(data.details)}`;
                }
            }
            showStatus('detect-status', `Error: ${errorMsg}`, 'error');
            updateGlobalStatus('Error', 'error');
        }
    } catch (error) {
        showStatus('detect-status', `Error: ${error.message}`, 'error');
        updateGlobalStatus('Error', 'error');
    }
}

// Display Single Image Result
function displaySingleResult(data) {
    singleResult.style.display = 'block';
    multipleResults.style.display = 'none';

    const imageName = Object.keys(data.results)[0];
    const result = data.results[imageName];

    const overlayPath = `${API_BASE}/get_image/${data.session_id}/${imageName.replace(/\.[^/.]+$/, '')}_overlay.png`;

    document.getElementById('result-image').src = overlayPath;
    document.getElementById('anomaly-score').textContent = result.anomaly_score.toFixed(4);
}

// Display Multiple Images Results
function displayMultipleResults(data) {
    singleResult.style.display = 'none';
    multipleResults.style.display = 'block';

    const resultsGrid = document.getElementById('results-grid');
    resultsGrid.innerHTML = '';

    for (const [imageName, result] of Object.entries(data.results)) {
        if (!result.processed) continue;

        const resultItem = document.createElement('div');
        resultItem.className = 'result-item';

        const overlayPath = `${API_BASE}/get_image/${data.session_id}/${imageName.replace(/\.[^/.]+$/, '')}_overlay.png`;

        resultItem.innerHTML = `
            <img src="${overlayPath}" alt="${imageName}">
            <div class="filename">${imageName}</div>
            <div class="score">Score: ${result.anomaly_score.toFixed(4)}</div>
        `;

        resultsGrid.appendChild(resultItem);
    }
}

// Save Single Result
function saveSingleResult() {
    const img = document.getElementById('result-image');
    const link = document.createElement('a');
    link.href = img.src;
    link.download = 'anomaly_result.png';
    link.click();
}

// Download All Results
function downloadAllResults() {
    if (!currentSessionId) return;

    window.location.href = `${API_BASE}/download_results/${currentSessionId}`;
}

// Utility Functions
function showStatus(elementId, message, type) {
    const element = document.getElementById(elementId);
    element.textContent = message;
    element.className = `status-message ${type}`;
}

function updateGlobalStatus(text, type) {
    statusText.textContent = text;
    const dot = document.querySelector('.status-dot');
    dot.className = `status-dot ${type}`;
}

function generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

// Log Polling
let logIndex = 0;
let pollInterval = null;

function startLogPolling() {
    if (pollInterval) clearInterval(pollInterval);
    pollInterval = setInterval(pollLogs, 1000); // Poll every second
}

async function pollLogs() {
    try {
        const response = await fetch(`${API_BASE}/logs?after=${logIndex}`);
        if (!response.ok) return;

        const data = await response.json();

        if (data.logs && data.logs.length > 0) {
            const terminal = document.getElementById('terminal-container');
            if (!terminal) return;

            data.logs.forEach(log => {
                const line = document.createElement('div');
                line.className = 'terminal-line';
                // Trim trailing newlines for cleaner display, but keep internal formatting
                const cleanMessage = log.message.replace(/^\n+|\n+$/g, '');
                line.innerHTML = `<span class="terminal-timestamp">${log.time}</span>${cleanMessage}`;
                terminal.appendChild(line);

                // Parse progress from logs
                // Format: "Processing image X/Y: filename"
                const match = cleanMessage.match(/Processing image (\d+)\/(\d+)/);
                if (match) {
                    const current = parseInt(match[1]);
                    const total = parseInt(match[2]);
                    const percent = Math.round((current / total) * 100);

                    const progressBar = document.getElementById('progress-bar');
                    const progressText = document.getElementById('progress-text');
                    const progressContainer = document.getElementById('progress-container');

                    if (progressBar && progressText && progressContainer) {
                        progressContainer.style.display = 'block';
                        progressBar.style.width = `${percent}%`;
                        progressText.textContent = `${percent}%`;
                    }
                }
            });

            // Auto scroll to bottom
            terminal.scrollTop = terminal.scrollHeight;

            logIndex = data.next_index;
        }
    } catch (error) {
        // Silent catch
    }
}

// Start polling
startLogPolling();
