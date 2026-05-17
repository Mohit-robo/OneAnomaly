document.addEventListener('DOMContentLoaded', () => {
    // Initialize stage states
    document.querySelectorAll('.stage').forEach((el, index) => {
        if (index + 1 > 1) el.classList.add('locked');
    });

    // Global State
    let currentStage = 1;
    let loadedBankName = null;
    let analysisSessionId = null;
    let activeAnalysisItem = null;
    let maxUnlockedStage = 1;
    let manualRegions = []; 
    let isDrawing = false;
    let startX, startY;
    let backgroundImage = null;
    let rawPreviewUrl = null;

    const API_BASE = window.location.hostname ? `http://${window.location.hostname}:5000` : 'http://localhost:5000';
    console.log("Using API_BASE:", API_BASE);

    // Jet Colormap for Canvas
    function getJetColor(value) {
        const r = Math.min(255, Math.max(0, 255 * (1.5 - Math.abs(value * 4 - 3))));
        const g = Math.min(255, Math.max(0, 255 * (1.5 - Math.abs(value * 4 - 2))));
        const b = Math.min(255, Math.max(0, 255 * (1.5 - Math.abs(value * 4 - 1))));
        return { r, g, b };
    }

    // UI Elements
    const modeRadios = document.querySelectorAll('input[name="preprocess_mode"]');
    const panelThresholding = document.getElementById('panel-thresholding');
    const panelAverage = document.getElementById('panel-average');
    const saveConfigBtn = document.getElementById('save-config-btn');
    const configStatus = document.getElementById('config-status');

    // Thresholding inline preview elements
    const threshPreviewUpload = document.getElementById('thresh-preview-upload');
    const threshLiveContainer = document.getElementById('thresh-live-preview-container');
    const threshPreviewImg = document.getElementById('thresh-preview-img');

    // Average inline preview elements
    const avgPreviewUpload = document.getElementById('avg-preview-upload');
    const avgLiveContainer = document.getElementById('avg-live-preview-container');
    const avgPreviewImg = document.getElementById('avg-preview-img');
    const overlay = document.getElementById('blocking-overlay');

    // Toggle panels based on mode selection
    modeRadios.forEach(radio => {
        radio.addEventListener('change', (e) => {
            if (e.target.value === 'thresholding') {
                panelThresholding.style.display = 'block';
                panelAverage.style.display = 'none';
            } else {
                panelThresholding.style.display = 'none';
                panelAverage.style.display = 'block';
            }
        });
    });

    // Detector UI slider updates
    const updateSliderVal = (sliderId, valId, fixed=2) => {
        const el = document.getElementById(sliderId);
        const valEl = document.getElementById(valId);
        if(el && valEl) {
            el.addEventListener('input', (e) => {
                valEl.innerText = parseFloat(e.target.value).toFixed(fixed);
                // Trigger dynamic update if we have an active analysis
                if (activeAnalysisItem) {
                    updateDynamicVisualization();
                }
            });
        }
    };
    updateSliderVal('anomaly-threshold', 'anomaly-threshold-val');
    updateSliderVal('heatmap-alpha', 'heatmap-alpha-val');

    // Dynamic Visualization Logic
    function updateDynamicVisualization() {
        if (!activeAnalysisItem) return;

        const threshold = parseFloat(document.getElementById('anomaly-threshold').value);
        const alpha = parseFloat(document.getElementById('heatmap-alpha').value);

        // Update Status Badge
        const statusBadge = document.getElementById('anomaly-status-badge');
        if (!statusBadge) return;
        const isAnomaly = activeAnalysisItem.anomaly_score > threshold;

        if (isAnomaly) {
            statusBadge.className = 'status-badge status-anomaly';
            statusBadge.querySelector('.badge-icon').innerText = '⚠️';
            statusBadge.querySelector('.badge-text').innerText = 'ANOMALY';
        } else {
            statusBadge.className = 'status-badge status-normal';
            statusBadge.querySelector('.badge-icon').innerText = '✓';
            statusBadge.querySelector('.badge-text').innerText = 'NORMAL';
        }

        // Update overlay image opacity via CSS (no canvas, no race conditions)
        const overlayImg = document.getElementById('result-overlay-img');
        if (overlayImg) {
            overlayImg.style.opacity = alpha;
        }
    }

    /**
     * Activates the detailed side-by-side inspection for a specific result image.
     * Simply sets the src of two img tags — _source.png on left, _overlay.png on right.
     */
    function showDetailedAnalysis(res, sessionId) {
        if (!res || !sessionId) return;
        analysisSessionId = sessionId;
        const stem = res.filename.split('.').slice(0, -1).join('.');

        // Store state
        activeAnalysisItem = { ...res, stem };

        // Show container
        const container = document.getElementById('detailed-analysis');
        container.style.display = 'block';
        document.getElementById('analysis-filename').innerText = res.filename;

        // LEFT panel: clean source image
        const sourceImg = document.getElementById('detailed-original');
        sourceImg.src = `${API_BASE}/api/get_image/${sessionId}/${stem}_source.png`;

        // RIGHT panel: pre-baked overlay from backend (_overlay.png)
        const overlayImg = document.getElementById('result-overlay-img');
        const alpha = parseFloat(document.getElementById('heatmap-alpha').value);
        overlayImg.style.opacity = alpha;
        overlayImg.src = `${API_BASE}/api/get_image/${sessionId}/${stem}_overlay.png`;

        // Summary Card
        const cardBox = document.getElementById('analysis-card-container');
        if (cardBox) {
            const threshold = parseFloat(document.getElementById('anomaly-threshold').value);
            const isAnomaly = res.anomaly_score > threshold;
            cardBox.innerHTML = `
                <div class="result-item active" style="background: var(--bg-active); padding: 20px; border-radius: 12px; border: 2px solid ${isAnomaly ? '#ef4444' : '#10b981'};">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <span style="font-weight: 700; font-size: 15px; opacity: 0.9;">IMAGE: ${res.filename}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: baseline;">
                        <div style="display: flex; flex-direction: column;">
                            <span style="font-size: 11px; opacity: 0.6; text-transform: uppercase;">Max Anomaly Score</span>
                            <span style="font-weight: 800; font-size: 24px; color: ${isAnomaly ? '#ef4444' : '#10b981'}">${res.anomaly_score.toFixed(4)}</span>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 14px; font-weight: 700; background: rgba(0,0,0,0.5); padding: 6px 14px; border-radius: 8px; color: ${isAnomaly ? '#ef4444' : '#10b981'}">
                                ${isAnomaly ? '⚠️ ANOMALY' : '✓ NORMAL'}
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }

        // Update badge and scroll
        updateDynamicVisualization();
        container.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }




    // Helper to get current config
    function showToast(title, message, type = 'info', duration = 5000) {
        const container = document.getElementById('toast-container');
        if (!container) return;
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        
        toast.innerHTML = `
            <div class="toast-header">
                <span class="toast-icon">
                    ${type === 'success' ? '✓' : type === 'error' ? '✕' : 'ℹ'}
                </span>
                ${title}
            </div>
            <div class="toast-body">${message}</div>
            <div class="toast-progress"></div>
        `;
        
        container.appendChild(toast);
        
        // Trigger animation
        setTimeout(() => toast.classList.add('show'), 10);
        
        // Progress bar animation
        const progress = toast.querySelector('.toast-progress');
        progress.style.transition = `width ${duration}ms linear`;
        setTimeout(() => progress.style.width = '0%', 10);
        
        // Auto remove
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 400);
        }, duration);
    }

    function getCurrentConfig() {
        const mode = document.querySelector('input[name="preprocess_mode"]:checked').value;
        const morphOpenEl = document.getElementById('morph-open');
        const morphCloseEl = document.getElementById('morph-close');
        
        let config = { mode: mode };

        if (mode === 'thresholding') {
            config.channel = document.getElementById('thresh-channel').value;
            config.thresh_min = parseInt(document.getElementById('thresh-min').value);
            config.thresh_max = parseInt(document.getElementById('thresh-max').value);
            config.morph_open = morphOpenEl ? parseInt(morphOpenEl.value) : 3;
            config.morph_close = morphCloseEl ? parseInt(morphCloseEl.value) : 5;
        } else if (mode === 'average') {
            config.diff_threshold = parseInt(document.getElementById('avg-diff-thresh').value);
            config.fill_holes = document.getElementById('avg-fill-holes').checked;
            config.min_component_ratio = parseFloat(document.getElementById('avg-min-ratio').value);
            config.morph_open = 0; // Disabled for BG subtraction
            config.morph_close = 0; // Disabled for BG subtraction
        }
        return config;
    }

    function updateStageUI(stage) {
        if (stage > maxUnlockedStage) return;
        currentStage = stage;
        console.log(`Switching to Stage ${stage}`);

        // Update Sidebar
        document.querySelectorAll('.stage').forEach((el, index) => {
            const stageNum = index + 1;
            el.classList.remove('active');
            if (stageNum > maxUnlockedStage) {
                el.classList.add('locked');
            } else {
                el.classList.remove('locked');
            }
            if (stageNum === currentStage) el.classList.add('active');
        });

        // Toggle Pages
        document.querySelectorAll('.step-page').forEach(page => page.classList.remove('active'));
        const pageIds = ['page-preprocessing', 'page-spatial', 'page-memory', 'page-detect', 'page-confirm'];
        const activePage = document.getElementById(pageIds[stage - 1]);
        if (activePage) {
            activePage.classList.add('active');
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }
        // Refresh engines when entering spatial stage
        if (stage === 2) {
            if (typeof loadExistingEngines === 'function') loadExistingEngines();
        }
        // Refresh banks when entering detection stage
        if (stage === 4) {
            if (typeof loadExistingBanks === 'function') loadExistingBanks();
        }
        // Populate Phase 5 param summary + batch bank list when entering stage 5
        if (stage === 5) {
            buildPhase5ParamSummary();
            loadBatchBanks();
        }
    }

    // Sidebar navigation
    document.querySelectorAll('.stage').forEach((stageEl, index) => {
        stageEl.addEventListener('click', () => {
            const stageNum = index + 1;
            if (stageNum <= maxUnlockedStage) {
                updateStageUI(stageNum);
            }
        });
    });

    function updateInspector() {
        const inspectorContent = document.getElementById('parameters-summary');
        if (!inspectorContent) return;

        const config = getCurrentConfig();
        const spatialRadios = document.querySelector('input[name="spatial_mode"]:checked');
        const spatialMode = spatialRadios ? spatialRadios.value : 'N/A';
        
        let html = `
            <div class="parameter-group">
                <h3>Stage 1: Pre-Processing</h3>
                <ul>
                    <li>Mode: <span class="val">${config.mode.charAt(0).toUpperCase() + config.mode.slice(1)}</span></li>
                    ${config.mode === 'thresholding' ? `
                        <li>Channel: <span class="val">${config.channel}</span></li>
                        <li>Min: <span class="val">${config.thresh_min}</span></li>
                        <li>Max: <span class="val">${config.thresh_max}</span></li>
                        <li>Morph Open: <span class="val">${config.morph_open}</span></li>
                        <li>Morph Close: <span class="val">${config.morph_close}</span></li>
                    ` : `
                        <li>Diff Thresh: <span class="val">${config.diff_threshold}</span></li>
                        <li>Fill Holes: <span class="val">${config.fill_holes}</span></li>
                        <li>Min Ratio: <span class="val">${config.min_component_ratio}</span></li>
                    `}
                </ul>
            </div>
        `;

        if (maxUnlockedStage >= 2 || currentStage >= 2) {
            html += `
                <div class="parameter-group">
                    <h3>Stage 2: Spatial Partitioning</h3>
                    <ul>
                        <li>Mode: <span class="val">${spatialMode}</span></li>
                        ${spatialMode === 'grid' ? `
                            <li>Split X: <span class="val">${document.getElementById('grid-x-ratio').value}%</span></li>
                            <li>Split Y: <span class="val">${document.getElementById('grid-y-ratio').value}%</span></li>
                        ` : ''}
                        ${spatialMode === 'manual' ? `
                            <li>Regions: <span class="val">${manualRegions.length} boxes</span></li>
                        ` : ''}
                    </ul>
                </div>
            `;
        }
        
        inspectorContent.innerHTML = html;
    }

    // Attach inspector updates to all inputs
    document.querySelectorAll('input, select').forEach(input => {
        input.addEventListener('input', updateInspector);
        input.addEventListener('change', updateInspector);
    });

    function showStatus(element, message, isError = false) {
        element.innerText = message;
        element.className = 'status-msg ' + (isError ? 'status-error' : 'status-success');
        setTimeout(() => { element.innerText = ''; }, 3000);
    }

    // Save Configuration
    saveConfigBtn.addEventListener('click', async () => {
        const config = getCurrentConfig();
        const formData = new FormData();
        formData.append('config', JSON.stringify(config));

        // If average mode, upload the reference files
        if (config.mode === 'average') {
            const files = document.getElementById('bg-ref-upload').files;
            if (files.length === 0) {
                showStatus(configStatus, 'Error: Please upload empty background images/ZIP.', true);
                return;
            }
            for (let i = 0; i < files.length; i++) {
                formData.append('ref_files', files[i]);
            }
        }

        saveConfigBtn.disabled = true;
        saveConfigBtn.innerText = 'Saving...';
        
        try {
            const response = await fetch(`${API_BASE}/api/configure_preprocess`, {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            
            if (response.ok) {
                showStatus(configStatus, 'Configuration saved successfully!');
                maxUnlockedStage = 2;
                updateStageUI(2);
                initCanvas();
            } else {
                showStatus(configStatus, 'Error: ' + result.error, true);
            }
        } catch (error) {
            console.error('Save config error:', error);
            showStatus(configStatus, 'Network error while saving.', true);
        } finally {
            saveConfigBtn.disabled = false;
            saveConfigBtn.innerText = 'Save Configuration & Continue';
        }
    });

    // Debounce helper
    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    let threshAbortController = null;
    let avgAbortController = null;

    // Interactive Threshold Preview Update
    const updateThreshPreview = debounce(async () => {
        if (threshPreviewUpload.files.length === 0) return;
        
        if (threshAbortController) threshAbortController.abort();
        threshAbortController = new AbortController();
        
        const formData = new FormData();
        formData.append('test_image', threshPreviewUpload.files[0]);
        formData.append('config', JSON.stringify(getCurrentConfig()));

        threshLiveContainer.style.display = 'block';
        threshPreviewImg.style.opacity = '0.5';

        try {
                const response = await fetch(`${API_BASE}/api/preview_mask`, {
                method: 'POST',
                body: formData,
                signal: threshAbortController.signal
            });
            if (response.ok) {
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                threshPreviewImg.src = url;
            }
        } catch (error) {
            if (error.name !== 'AbortError') console.error('Threshold Preview error:', error);
        } finally {
            threshPreviewImg.style.opacity = '1';
        }
    }, 150);

    // Bind threshold sliders to trigger live update
    const threshInputs = ['thresh-channel', 'thresh-min', 'thresh-max', 'morph-open', 'morph-close'];
    threshInputs.forEach(id => {
        const el = document.getElementById(id);
        if(el) {
            el.addEventListener('input', updateThreshPreview);
        }
    });

    // Also trigger update when a file is selected
    threshPreviewUpload.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            rawPreviewUrl = URL.createObjectURL(file);
            updateThreshPreview();
        }
    });

    // Interactive Average Preview Update
    const updateAvgPreview = debounce(async () => {
        if (avgPreviewUpload.files.length === 0) return;
        
        if (avgAbortController) avgAbortController.abort();
        avgAbortController = new AbortController();
        
        const config = getCurrentConfig();
        const formData = new FormData();
        formData.append('test_image', avgPreviewUpload.files[0]);
        formData.append('config', JSON.stringify(config));

        const bgRefUpload = document.getElementById('bg-ref-upload');
        if (bgRefUpload && bgRefUpload.files.length > 0) {
            for (let i = 0; i < bgRefUpload.files.length; i++) {
                formData.append('ref_files', bgRefUpload.files[i]);
            }
        }

        avgLiveContainer.style.display = 'block';
        avgPreviewImg.style.opacity = '0.5';

        try {
            const response = await fetch('http://localhost:5000/api/preview_mask', {
                method: 'POST',
                body: formData,
                signal: avgAbortController.signal
            });
            if (response.ok) {
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                avgPreviewImg.src = url;
            }
        } catch (error) {
            if (error.name !== 'AbortError') console.error('Average Preview error:', error);
        } finally {
            avgPreviewImg.style.opacity = '1';
        }
    }, 150);

    // Bind average sliders to trigger live update
    const avgInputs = ['avg-diff-thresh', 'avg-min-ratio', 'avg-fill-holes'];
    avgInputs.forEach(id => {
        const el = document.getElementById(id);
        if(el) {
            el.addEventListener('input', updateAvgPreview);
        }
    });

    avgPreviewUpload.addEventListener('change', updateAvgPreview);

    // --- Stage 2: Spatial Regions ---
    const spatialRadios = document.querySelectorAll('input[name="spatial_mode"]');
    const regionCanvas = document.getElementById('region-canvas');
    const ctx = regionCanvas.getContext('2d');
    const clearRegionsBtn = document.getElementById('clear-regions-btn');
    const saveSpatialBtn = document.getElementById('save-spatial-btn');
    const exportTrtBtn = document.getElementById('export-trt-btn');
    const engineSelect = document.getElementById('existing-engine-select');

    spatialRadios.forEach(r => {
        r.addEventListener('change', (e) => {
            const val = e.target.value;
            document.getElementById('panel-grid').style.display = val === 'grid' ? 'block' : 'none';
            document.getElementById('panel-manual').style.display = val === 'manual' ? 'block' : 'none';
            document.getElementById('panel-canvas').style.display = (val === 'manual' || val === 'grid') ? 'block' : 'none';
            
            // Highlight sub-tasks in sidebar
            document.querySelectorAll('.sub-task').forEach(st => st.classList.remove('active'));
            const taskId = 'task-' + val;
            const taskEl = document.getElementById(taskId);
            if (taskEl) taskEl.classList.add('active');

            if (val === 'manual' || val === 'grid') {
                initCanvas();
            }
        });
    });

    // Canvas Logic
    async function initCanvas() {
        showStatus(configStatus, 'Updating spatial background...');
        
        // Fetch the Masked Result from Stage 1 to use as background
        const config = getCurrentConfig();
        const threshFile = document.getElementById('thresh-preview-upload').files[0];
        const avgFile = document.getElementById('avg-preview-upload').files[0];
        const file = threshFile || avgFile;
        
        if (file) {
            const formData = new FormData();
            formData.append('test_image', file);
            formData.append('config', JSON.stringify(config));
            formData.append('return_result', 'true');

            // Include reference files for average mode if they exist
            const bgRefUpload = document.getElementById('bg-ref-upload');
            if (config.mode === 'average' && bgRefUpload && bgRefUpload.files.length > 0) {
                for (let i = 0; i < bgRefUpload.files.length; i++) {
                    formData.append('ref_files', bgRefUpload.files[i]);
                }
            }
            
            try {
                const response = await fetch('http://localhost:5000/api/preview_mask', {
                    method: 'POST',
                    body: formData
                });
                if (response.ok) {
                    const blob = await response.blob();
                    rawPreviewUrl = URL.createObjectURL(blob);
                }
            } catch (e) {
                console.error('Error fetching masked background:', e);
            }
        }

        // Use the masked preview as the background
        if (rawPreviewUrl) {
            backgroundImage = new Image();
            backgroundImage.src = rawPreviewUrl;
            backgroundImage.onload = () => {
                regionCanvas.width = backgroundImage.width;
                regionCanvas.height = backgroundImage.height;
                drawRegions();
            };
        } else {
            // Default size if no preview image yet
            regionCanvas.width = 640;
            regionCanvas.height = 480;
            drawRegions();
        }
    }

    regionCanvas.addEventListener('mousedown', e => {
        if (document.querySelector('input[name="spatial_mode"]:checked').value !== 'manual') return;
        if (manualRegions.length >= 5) return;

        const rect = regionCanvas.getBoundingClientRect();
        const scaleX = regionCanvas.width / rect.width;
        const scaleY = regionCanvas.height / rect.height;
        
        isDrawing = true;
        startX = (e.clientX - rect.left) * scaleX;
        startY = (e.clientY - rect.top) * scaleY;
    });

    regionCanvas.addEventListener('mousemove', e => {
        if (!isDrawing) return;
        const rect = regionCanvas.getBoundingClientRect();
        const scaleX = regionCanvas.width / rect.width;
        const scaleY = regionCanvas.height / rect.height;
        
        const currX = (e.clientX - rect.left) * scaleX;
        const currY = (e.clientY - rect.top) * scaleY;
        
        drawRegions(currX, currY);
    });

    window.addEventListener('mouseup', e => {
        if (!isDrawing) return;
        isDrawing = false;
        
        const rect = regionCanvas.getBoundingClientRect();
        const scaleX = regionCanvas.width / rect.width;
        const scaleY = regionCanvas.height / rect.height;
        
        const endX = (e.clientX - rect.left) * scaleX;
        const endY = (e.clientY - rect.top) * scaleY;

        const w = endX - startX;
        const h = endY - startY;

        if (Math.abs(w) > 10 && Math.abs(h) > 10) {
            manualRegions.push({
                x: Math.min(startX, endX),
                y: Math.min(startY, endY),
                w: Math.abs(w),
                h: Math.abs(h)
            });
        }
        drawRegions();
    });

    function drawRegions(currX = null, currY = null) {
        ctx.clearRect(0, 0, regionCanvas.width, regionCanvas.height);
        
        if (backgroundImage) {
            ctx.drawImage(backgroundImage, 0, 0);
        } else {
            ctx.fillStyle = '#18181b';
            ctx.fillRect(0, 0, regionCanvas.width, regionCanvas.height);
        }

        const mode = document.querySelector('input[name="spatial_mode"]:checked').value;

        if (mode === 'grid') {
            const xr = parseInt(document.getElementById('grid-x-ratio').value) / 100;
            const yr = parseInt(document.getElementById('grid-y-ratio').value) / 100;
            
            ctx.strokeStyle = '#3b82f6';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(regionCanvas.width * xr, 0);
            ctx.lineTo(regionCanvas.width * xr, regionCanvas.height);
            ctx.moveTo(0, regionCanvas.height * yr);
            ctx.lineTo(regionCanvas.width, regionCanvas.height * yr);
            ctx.stroke();
            
            ctx.fillStyle = '#3b82f6';
            ctx.font = 'bold 16px Inter, system-ui, sans-serif';
            ctx.fillText('Q1', 10, 25);
            ctx.fillText('Q2', regionCanvas.width * xr + 10, 25);
            ctx.fillText('Q3', 10, regionCanvas.height * yr + 25);
            ctx.fillText('Q4', regionCanvas.width * xr + 10, regionCanvas.height * yr + 25);

            // Subtle quadrant tints
            ctx.fillStyle = 'rgba(59, 130, 246, 0.05)';
            ctx.fillRect(0, 0, regionCanvas.width * xr, regionCanvas.height * yr);
            ctx.fillRect(regionCanvas.width * xr, regionCanvas.height * yr, regionCanvas.width * (1-xr), regionCanvas.height * (1-yr));
        } else if (mode === 'manual') {
            manualRegions.forEach((r, idx) => {
                ctx.strokeStyle = '#10b981';
                ctx.lineWidth = 3;
                ctx.strokeRect(r.x, r.y, r.w, r.h);
                ctx.fillStyle = 'rgba(16, 185, 129, 0.1)';
                ctx.fillRect(r.x, r.y, r.w, r.h);
                ctx.fillStyle = '#10b981';
                ctx.fillText(`Region ${idx + 1}`, r.x + 5, r.y + 15);
            });

            if (isDrawing && currX !== null) {
                ctx.strokeStyle = '#3b82f6';
                ctx.setLineDash([5, 5]);
                ctx.strokeRect(startX, startY, currX - startX, currY - startY);
                ctx.setLineDash([]);
            }
        }
    }

    clearRegionsBtn.addEventListener('click', () => {
        manualRegions = [];
        drawRegions();
    });

    // Global progress handler for streaming tasks
    function runStreamingTask(title, apiPath, bodyData, onComplete) {
        const overlayTitle = document.getElementById('overlay-title');
        const overlayMsg = document.getElementById('overlay-msg');
        const overlayProgress = document.getElementById('overlay-progress');
        const overlayLog = document.getElementById('overlay-log');

        overlay.style.display = 'flex';
        overlayTitle.innerText = title;
        overlayMsg.innerText = 'Initializing...';
        overlayProgress.style.width = '0%';
        overlayLog.style.display = 'block';
        overlayLog.innerText = '';
        document.getElementById('overlay-spinner').style.display = 'block';
        document.getElementById('overlay-close-btn').style.display = 'none';

        const updateUI = (data) => {
            if (data.title) overlayTitle.innerText = data.title;
            if (data.message) {
                overlayLog.innerText += data.message + '\n';
                overlayLog.scrollTop = overlayLog.scrollHeight;
            }
            if (data.progress !== undefined) {
                overlayProgress.style.width = data.progress + '%';
            }

            if (data.success) {
                document.getElementById('overlay-spinner').style.display = 'none';
                document.getElementById('overlay-close-btn').style.display = 'inline-block';
                if (onComplete) onComplete(data);
                
                setTimeout(() => {
                    overlay.style.display = 'none';
                }, 3000); // 3 seconds instead of 1.5
            }

            if (data.error) {
                overlayLog.innerText += `ERROR: ${data.error}\n`;
                document.getElementById('overlay-spinner').style.display = 'none';
                document.getElementById('overlay-close-btn').style.display = 'inline-block';
                showToast('Task Failed', data.error, 'error');
            }
        };

        (async () => {
            try {
                const response = await fetch(`${API_BASE}${apiPath}`, {
                    method: 'POST',
                    headers: bodyData instanceof FormData ? {} : { 'Content-Type': 'application/json' },
                    body: bodyData instanceof FormData ? bodyData : JSON.stringify(bodyData)
                });

                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.error || 'Server error');
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder();

                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;
                    const text = decoder.decode(value);
                    const lines = text.split('\n');
                    lines.forEach(line => {
                        if (!line.trim()) return;
                        try {
                            const data = JSON.parse(line);
                            updateUI(data);
                        } catch (e) {
                            console.error('Parse error individual line', line);
                        }
                    });
                }
            } catch (error) {
                updateUI({ error: error.message });
            }
        })();
    }

    document.getElementById('export-trt-btn').addEventListener('click', async () => {
        const bs = parseInt(document.getElementById('batch-size-trt').value);
        const spatialRadios = document.querySelector('input[name="spatial_mode"]:checked');
        const spatialMode = spatialRadios ? spatialRadios.value : 'whole';
        
        const preProcMode = document.querySelector('input[name="preprocess_mode"]:checked').value;
        let preProcInfo = "";
        if (preProcMode === 'thresholding') {
            const min = document.getElementById('thresh-min').value;
            const max = document.getElementById('thresh-max').value;
            preProcInfo = `int${min}-${max}`;
        } else {
            const diff = document.getElementById('avg-diff-thresh').value;
            preProcInfo = `avgD${diff}`;
        }

        if (spatialMode === 'manual' && manualRegions.length === 0) {
            showToast('Region Missing', 'Please draw at least one region or select another mode.', 'error');
            return;
        }

        runStreamingTask('Model Export', '/api/export_trt', {
            batch_size: bs,
            spatial_mode: spatialMode,
            pre_proc: preProcInfo
        }, () => {
            if (typeof loadExistingEngines === 'function') loadExistingEngines();
        });
    });

    document.getElementById('overlay-close-btn').addEventListener('click', () => {
        overlay.style.display = 'none';
    });

    // Expose update function for HTML oninput events
    window.updateSpatialPreview = function() {
        drawRegions();
    };

    async function loadExistingEngines() {
        console.log("Loading engines from...", `${API_BASE}/api/list_engines`);
        try {
            const response = await fetch(`${API_BASE}/api/list_engines`);
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const data = await response.json();
            console.log("Received engines:", data.engines);
            
            if (!engineSelect) {
                console.error("engineSelect element not found!");
                return;
            }
            
            engineSelect.innerHTML = '<option value="">-- No engine selected --</option>';
            if (data.engines && data.engines.length > 0) {
                data.engines.forEach(eng => {
                    const opt = document.createElement('option');
                    opt.value = eng;
                    opt.innerText = eng;
                    engineSelect.appendChild(opt);
                });
            } else {
                console.warn("No engines found on server.");
            }
        } catch (e) {
            console.error("Failed to load engines:", e);
        }
    }

    async function loadExistingBanks() {
        try {
            const response = await fetch(`${API_BASE}/api/list_memory_banks`);
            const data = await response.json();
            const select = document.getElementById('existing-bank-select');
            if (!select) return;
            
            select.innerHTML = '<option value="">-- No bank selected (use current) --</option>';
            if (data.memory_banks) {
                data.memory_banks.forEach(bank => {
                    const opt = document.createElement('option');
                    opt.value = bank;
                    opt.innerText = bank;
                    select.appendChild(opt);
                });
            }
        } catch (e) {
            console.error("Failed to load memory banks:", e);
        }
    }

    loadExistingEngines();
    loadExistingBanks();

    saveSpatialBtn.addEventListener('click', async () => {
        const mode = document.querySelector('input[name="spatial_mode"]:checked').value;
        const engineSelectEl = document.getElementById('existing-engine-select');
        const engine = engineSelectEl ? engineSelectEl.value : "";
        
        if (!engine || engine === "") {
            showToast('Engine Missing', 'Please select or generate a TensorRT engine before continuing.', 'error');
            if (engineSelectEl) {
                engineSelectEl.style.borderColor = '#ef4444';
                setTimeout(() => engineSelectEl.style.borderColor = '', 2000);
            }
            return;
        }

        const config = {
            spatial_mode: mode,
            engine: engine,
            regions: (mode === 'manual') ? manualRegions : (mode === 'grid' ? [] : []), // coords will be calculated by backend for grid
            grid_ratios: {
                x: parseInt(document.getElementById('grid-x-ratio').value),
                y: parseInt(document.getElementById('grid-y-ratio').value)
            }
        };

        const response = await fetch(`${API_BASE}/api/configure_spatial`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });

        if (response.ok) {
            showStatus(document.getElementById('spatial-status'), 'Stage 2 Configured Successfully!');
            maxUnlockedStage = 3;
            updateStageUI(3); // Advance to recap
        }
    });

    // -------------------------------------------------------------------------
    // STAGE 3: MEMORY BANK CREATION
    // -------------------------------------------------------------------------
    document.getElementById('extract-features-btn').addEventListener('click', async () => {
        const files = document.getElementById('memory-upload').files;
        const numShots = document.getElementById('num-shots').value;
        const statusSpan = document.getElementById('memory-status');

        if (files.length === 0) {
            showStatus(statusSpan, 'Please select images or a ZIP file first.', true);
            return;
        }

        const formData = new FormData();
        Array.from(files).forEach(file => {
            if (file.name.endsWith('.zip')) {
                formData.append('zip_file', file);
            } else {
                formData.append('files', file);
            }
        });
        
        if (numShots && numShots > 0) {
            formData.append('num_shots', numShots);
        }

        runStreamingTask('Feature Extraction', '/api/extract_features', formData, (data) => {
            const totalItems = data.num_features_after_coreset || data.num_features;
            showStatus(statusSpan, `Success: ${data.num_images} images processed across ${data.num_regions} regions. Total ${totalItems} items in memory.`);
            showToast('Memory Bank Built Successfully', `Processed ${data.num_images} images across ${data.num_regions} regions.\nTotal features: ${totalItems}`, 'success');
            
            // Enable save button
            document.getElementById('save-memory-btn').disabled = false;
        });
    });

    document.getElementById('save-memory-btn').addEventListener('click', async () => {
        const bankName = document.getElementById('memory-bank-name').value;
        const statusSpan = document.getElementById('memory-status');

        if (!bankName) {
            showStatus(statusSpan, 'Please provide a name for the Memory Bank.', true);
            return;
        }

        const btn = document.getElementById('save-memory-btn');
        btn.innerText = 'Saving...';
        btn.disabled = true;

        try {
            const response = await fetch(`${API_BASE}/api/save_memory_bank`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename: bankName })
            });

            const data = await response.json();
            if (data.error) throw new Error(data.error);

            showStatus(statusSpan, `Memory Bank saved successfully as ${bankName}`);
            showToast('Memory Bank Saved', `Bank: ${bankName}`, 'success');
            
            // Unlock Stage 4
            if (maxUnlockedStage < 4) maxUnlockedStage = 4;
            updateStageUI(4);
        } catch (error) {
            showStatus(statusSpan, error.message, true);
        } finally {
            btn.innerText = 'Save & Continue to Detection';
            btn.disabled = false;
        }
    });

    document.getElementById('skip-memory-btn').addEventListener('click', () => {
        if (maxUnlockedStage < 4) maxUnlockedStage = 4;
        updateStageUI(4);
    });

    // -------------------------------------------------------------------------
    // STAGE 4: ANOMALY DETECTION
    // -------------------------------------------------------------------------
    document.getElementById('detect-btn').addEventListener('click', async () => {
        const files = document.getElementById('detect-upload').files;
        const bankName = document.getElementById('existing-bank-select').value;
        const statusSpan = document.getElementById('detect-status');
        const resultsGrid = document.getElementById('results-grid');
        const resultsArea = document.getElementById('detect-results');

        if (files.length === 0) {
            showStatus(statusSpan, 'Please select test images first.', true);
            return;
        }

        const btn = document.getElementById('detect-btn');
        btn.innerText = 'Detecting...';
        btn.disabled = true;
        statusSpan.innerText = '';

        try {
            // Only load the bank if selected and NOT already the active one
            if (bankName && bankName !== loadedBankName) {
                statusSpan.innerText = 'Loading Memory Bank...';
                const loadResp = await fetch(`${API_BASE}/api/load_memory_bank`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ filename: bankName })
                });
                const loadData = await loadResp.json();
                if (loadData.error) throw new Error("Load Bank Failed: " + loadData.error);
                loadedBankName = bankName;
            }

            const sessionId = 'detect_' + new Date().getTime();
            const formData = new FormData();
            formData.append('session_id', sessionId);
            
            Array.from(files).forEach(file => {
                if (file.name.endsWith('.zip')) {
                    formData.append('zip_file', file);
                } else {
                    formData.append('files', file);
                }
            });

            // Get dynamic analysis values
            const anomalyThreshold = document.getElementById('anomaly-threshold').value;
            const heatmapAlpha = document.getElementById('heatmap-alpha').value;
            formData.append('threshold', anomalyThreshold);
            formData.append('alpha', heatmapAlpha);

            const response = await fetch(`${API_BASE}/api/detect_anomalies`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            if (data.error) throw new Error(data.error);

            showStatus(statusSpan, `Inference complete! Processed ${data.num_images} images. Overall max score: ${data.max_score.toFixed(4)}`);
            
            // Expert View Optimization: Redundant result tiles are suppressed. 
            // The system proceeds directly to the high-fidelity Detailed Analysis for the first result.
            resultsGrid.innerHTML = '';
            resultsArea.style.display = 'none';
            
            if (data.results && data.results.length > 0) {
                // Auto-activate the first (primary) result in the detailed pane
                showDetailedAnalysis(data.results[0], data.session_id);
                // Unlock Stage 5 after first detection
                if (maxUnlockedStage < 5) maxUnlockedStage = 5;
                updateStageUI(currentStage); // re-render sidebar without navigating
            }

        } catch (error) {
            showStatus(statusSpan, error.message, true);
        } finally {
            btn.innerText = 'Run Detection';
            btn.disabled = false;
        }
    });

    // =========================================================================
    // STAGE 5: CONFIRMATION & FINAL INFERENCE
    // =========================================================================

    // Phase 5 state
    let batchResults = [];
    let batchSessionId = null;
    let batchCurrentPage = 1;
    const batchPageSize = 10;
    let batchFilteredResults = [];
    let currentBatchFilter = 'all';

    // Populate the param summary card when entering Stage 5
    function buildPhase5ParamSummary() {
        const el = document.getElementById('phase5-param-summary');
        if (!el) return;

        const config = getCurrentConfig();
        const spatialMode = (document.querySelector('input[name="spatial_mode"]:checked') || {}).value || 'N/A';
        const engineEl = document.getElementById('existing-engine-select');
        const engine = engineEl ? (engineEl.options[engineEl.selectedIndex] || {}).text : 'None';
        const bankName = loadedBankName || document.getElementById('memory-bank-name')?.value || 'None';
        const threshold = document.getElementById('anomaly-threshold')?.value || '0.50';
        const alpha = document.getElementById('heatmap-alpha')?.value || '0.50';

        const row = (label, value, color='') => `
            <div style="display:flex; justify-content:space-between; align-items:center; padding: 8px 0; border-bottom: 1px solid var(--border-subtle);">
                <span style="color: var(--text-tertiary); font-size: 13px;">${label}</span>
                <span style="font-weight: 600; font-size: 13px; ${color ? 'color:'+color : ''}">${value}</span>
            </div>`;

        const section = (title, rows) => `
            <div style="background: var(--bg-surface-elevated); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm); padding: 16px; margin-bottom: 12px;">
                <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--accent-primary); margin-bottom: 12px;">${title}</div>
                ${rows}
            </div>`;

        let s1rows = row('Mode', config.mode === 'thresholding' ? 'Intensity Thresholding' : 'Avg. Background Subtraction');
        if (config.mode === 'thresholding') {
            s1rows += row('Channel', config.channel || 'gray');
            s1rows += row('Intensity Range', `${config.thresh_min} – ${config.thresh_max}`);
            s1rows += row('Morph Open / Close', `${config.morph_open} / ${config.morph_close}`);
        } else {
            s1rows += row('Diff Threshold', config.diff_threshold);
            s1rows += row('Fill Holes', config.fill_holes ? 'Yes' : 'No');
            s1rows += row('Min Component Ratio', config.min_component_ratio);
        }

        const spatialLabel = {grid: 'Quadrant Grid (2×2)', manual: 'Manual Regions', whole: 'Whole Image'};
        const s2rows = row('Spatial Mode', spatialLabel[spatialMode] || spatialMode)
            + row('TRT Engine', engine)
            + (spatialMode === 'manual' ? row('Region Count', manualRegions.length + ' boxes') : '');

        const s3rows = row('Memory Bank', bankName || 'Not loaded', bankName ? '#10b981' : '#ef4444');
        const s4rows = row('Anomaly Threshold (τ)', parseFloat(threshold).toFixed(2))
            + row('Heatmap Opacity (α)', parseFloat(alpha).toFixed(2));

        el.innerHTML =
            '<div style="display:grid; grid-template-columns: 1fr 1fr; gap:12px;">' +
            '<div>' + section('Stage 1 — Preprocessing', s1rows) + section('Stage 3 — Memory Bank', s3rows) + '</div>' +
            '<div>' + section('Stage 2 — Spatial Regions', s2rows) + section('Stage 4 — Detection Params', s4rows) + '</div>' +
            '</div>';
    }

    // Populate batch bank dropdown for Stage 5
    async function loadBatchBanks() {
        try {
            const resp = await fetch(`${API_BASE}/api/list_memory_banks`);
            const data = await resp.json();
            const sel = document.getElementById('batch-bank-select');
            if (!sel) return;
            sel.innerHTML = '<option value="">-- Use currently loaded bank --</option>';
            (data.memory_banks || []).forEach(b => {
                const opt = document.createElement('option');
                opt.value = b;
                opt.text = b;
                sel.appendChild(opt);
            });
        } catch(e) { /* silent */ }
    }

    // Lightbox open/close
    window.openBatchLightbox = function(src, filename, score, badge, color) {
        const lb = document.getElementById('batch-lightbox');
        if (!lb) return;
        document.getElementById('lb-img').src = src;
        document.getElementById('lb-filename').innerText = filename;
        document.getElementById('lb-score').innerText = 'Score: ' + score;
        document.getElementById('lb-badge').innerText = badge;
        document.getElementById('lb-badge').style.color = color;
        lb.style.display = 'flex';
    };

    window.closeBatchLightbox = function() {
        const lb = document.getElementById('batch-lightbox');
        if (lb) lb.style.display = 'none';
    };

    document.addEventListener('keydown', e => {
        if (e.key === 'Escape') window.closeBatchLightbox();
    });

    // Render a page of the batch results grid
    function renderBatchGrid() {
        const grid = document.getElementById('batch-results-grid');
        const pageIndicator = document.getElementById('batch-page-indicator');
        const prevBtn = document.getElementById('batch-prev-btn');
        const nextBtn = document.getElementById('batch-next-btn');
        if (!grid) return;

        const totalPages = Math.max(1, Math.ceil(batchFilteredResults.length / batchPageSize));
        batchCurrentPage = Math.min(batchCurrentPage, totalPages);
        const start = (batchCurrentPage - 1) * batchPageSize;
        const page = batchFilteredResults.slice(start, start + batchPageSize);

        const threshold = parseFloat(document.getElementById('batch-threshold').value);

        grid.innerHTML = page.map(res => {
            const stem = res.filename.split('.').slice(0, -1).join('.');
            const sourceSrc  = `${API_BASE}/api/get_image/${batchSessionId}/${stem}_source.png`;
            const overlaySrc = `${API_BASE}/api/get_image/${batchSessionId}/${stem}_overlay.png`;
            const torchSrc   = `${API_BASE}/api/get_image/${batchSessionId}/torch_res_${stem}.png`;
            const isAnomaly  = res.anomaly_score > threshold;
            const badgeColor = isAnomaly ? '#ef4444' : '#10b981';
            const badgeText  = isAnomaly ? '⚠ ANOMALY' : '✓ NORMAL';
            const scoreStr   = res.anomaly_score.toFixed(4);
            const esc        = res.filename.replace(/'/g, "\\'");
            return `
                <div onclick="openBatchLightbox('${torchSrc}', '${esc}', '${scoreStr}', '${badgeText}', '${badgeColor}')"
                     style="background: var(--bg-surface-elevated); border: 1px solid ${isAnomaly ? '#ef444440' : '#10b98130'};
                            border-radius: var(--radius-sm); overflow:hidden; display:flex; flex-direction:column;
                            cursor:pointer; transition: transform 0.15s ease, box-shadow 0.15s ease;"
                     onmouseenter="this.style.transform='translateY(-3px)'; this.style.boxShadow='0 8px 20px rgba(0,0,0,0.5)';"
                     onmouseleave="this.style.transform=''; this.style.boxShadow='';">
                    <!-- Side-by-side image strip -->
                    <div style="display:flex; gap:1px; background:#111; overflow:hidden; position:relative;">
                        <div style="flex:1; overflow:hidden;">
                            <img src="${sourceSrc}" alt="Original"
                                 style="width:100%; aspect-ratio:1; object-fit:cover; display:block;"
                                 onerror="this.style.background='#1a1a1a'">
                        </div>
                        <div style="flex:1; overflow:hidden; position:relative;">
                            <img src="${overlaySrc}" alt="Anomaly map"
                                 style="width:100%; aspect-ratio:1; object-fit:cover; display:block;"
                                 onerror="this.style.background='#1a1a1a'">
                            <div style="position:absolute; top:5px; right:5px; background:${badgeColor}22;
                                        border:1px solid ${badgeColor}; color:${badgeColor};
                                        font-size:9px; font-weight:700; padding:1px 7px; border-radius:20px;">
                                ${badgeText}
                            </div>
                        </div>
                    </div>
                    <!-- Column labels -->
                    <div style="display:flex; border-top:1px solid var(--border-subtle);">
                        <div style="flex:1; font-size:9px; color:var(--text-tertiary); text-align:center; padding:3px 0; border-right:1px solid var(--border-subtle);">Original</div>
                        <div style="flex:1; font-size:9px; color:var(--text-tertiary); text-align:center; padding:3px 0;">Anomaly Map</div>
                    </div>
                    <!-- Info row -->
                    <div style="padding: 9px 12px; display:flex; justify-content:space-between; align-items:center;">
                        <div style="font-size:12px; font-weight:600; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; flex:1;" title="${res.filename}">${res.filename}</div>
                        <div style="font-size:11px; color:${badgeColor}; font-weight:700; margin-left:8px; flex-shrink:0;">${scoreStr}</div>
                    </div>
                </div>`;
        }).join('');

        pageIndicator.innerText = `Page ${batchCurrentPage} of ${totalPages}  (${batchFilteredResults.length} images)`;
        prevBtn.disabled = batchCurrentPage <= 1;
        nextBtn.disabled = batchCurrentPage >= totalPages;
    }

    window.changeBatchPage = function(delta) {
        batchCurrentPage += delta;
        renderBatchGrid();
    };

    window.applyBatchFilter = function(filter, btn) {
        currentBatchFilter = filter;
        document.querySelectorAll('.batch-filter-btn').forEach(b => b.classList.remove('active'));
        if (btn) btn.classList.add('active');
        const threshold = parseFloat(document.getElementById('batch-threshold').value);
        if (filter === 'all') {
            batchFilteredResults = [...batchResults];
        } else if (filter === 'anomaly') {
            batchFilteredResults = batchResults.filter(r => r.anomaly_score > threshold);
        } else {
            batchFilteredResults = batchResults.filter(r => r.anomaly_score <= threshold);
        }
        batchCurrentPage = 1;
        renderBatchGrid();
        // Update summary chips
        const chips = document.getElementById('batch-summary-chips');
        if (chips) chips.innerHTML = buildSummaryChips(threshold);
    };

    function buildSummaryChips(threshold) {
        const total = batchResults.length;
        const anomCount = batchResults.filter(r => r.anomaly_score > threshold).length;
        const normCount = total - anomCount;
        return `
            <span style="font-size:12px; background:rgba(255,255,255,0.05); border:1px solid var(--border-subtle); padding:3px 10px; border-radius:20px;">${total} Total</span>
            <span style="font-size:12px; background:#ef444415; border:1px solid #ef444440; color:#ef4444; padding:3px 10px; border-radius:20px;">⚠ ${anomCount} Anomaly</span>
            <span style="font-size:12px; background:#10b98115; border:1px solid #10b98140; color:#10b981; padding:3px 10px; border-radius:20px;">✓ ${normCount} Normal</span>`;
    }

    // Batch Run button
    document.getElementById('batch-run-btn').addEventListener('click', async () => {
        const files = document.getElementById('batch-upload').files;
        const bankOverride = document.getElementById('batch-bank-select').value;
        const statusSpan = document.getElementById('batch-status');
        const btn = document.getElementById('batch-run-btn');

        if (files.length === 0) {
            showStatus(statusSpan, 'Please upload test images or a ZIP file.', true);
            return;
        }

        btn.innerText = 'Running...';
        btn.disabled = true;
        statusSpan.innerText = '';

        try {
            // Load a different bank if selected
            if (bankOverride) {
                statusSpan.innerText = 'Loading memory bank...';
                const loadResp = await fetch(`${API_BASE}/api/load_memory_bank`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ filename: bankOverride })
                });
                const ld = await loadResp.json();
                if (ld.error) throw new Error('Bank load failed: ' + ld.error);
                loadedBankName = bankOverride;
            }

            const sessionId = 'batch_' + new Date().getTime();
            batchSessionId = sessionId;

            const formData = new FormData();
            formData.append('session_id', sessionId);
            Array.from(files).forEach(file => {
                if (file.name.endsWith('.zip')) {
                    formData.append('zip_file', file);
                } else {
                    formData.append('files', file);
                }
            });
            formData.append('threshold', document.getElementById('batch-threshold').value);
            formData.append('alpha', '0.6');

            statusSpan.innerText = 'Running inference...';
            const response = await fetch(`${API_BASE}/api/detect_anomalies`, {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            if (data.error) throw new Error(data.error);

            batchResults = data.results || [];
            batchSessionId = data.session_id;

            const threshold = parseFloat(document.getElementById('batch-threshold').value);
            const anomCount = batchResults.filter(r => r.anomaly_score > threshold).length;

            showStatus(statusSpan, `Done: ${batchResults.length} images processed. ${anomCount} anomalies detected.`);
            showToast('Batch Inference Complete',
                `${batchResults.length} images | ${anomCount} anomalies | ${batchResults.length - anomCount} normal`,
                anomCount > 0 ? 'error' : 'success');

            // Show results + export sections
            document.getElementById('batch-results-section').style.display = 'block';
            document.getElementById('batch-export-section').style.display = 'block';

            // Populate chips and grid
            const chips = document.getElementById('batch-summary-chips');
            if (chips) chips.innerHTML = buildSummaryChips(threshold);

            batchFilteredResults = [...batchResults];
            batchCurrentPage = 1;
            currentBatchFilter = 'all';
            document.querySelectorAll('.batch-filter-btn').forEach(b => b.classList.remove('active'));
            const allBtn = document.getElementById('filter-all-btn');
            if (allBtn) allBtn.classList.add('active');
            renderBatchGrid();

            document.getElementById('batch-results-section').scrollIntoView({ behavior: 'smooth', block: 'start' });

        } catch (err) {
            showStatus(statusSpan, err.message, true);
        } finally {
            btn.innerText = '▶  Run Final Inference';
            btn.disabled = false;
        }
    });

    // Filter button event listeners
    document.getElementById('filter-all-btn').addEventListener('click', function() { window.applyBatchFilter('all', this); });
    document.getElementById('filter-anomaly-btn').addEventListener('click', function() { window.applyBatchFilter('anomaly', this); });
    document.getElementById('filter-normal-btn').addEventListener('click', function() { window.applyBatchFilter('normal', this); });

    // Pagination button event listeners
    document.getElementById('batch-prev-btn').addEventListener('click', () => window.changeBatchPage(-1));
    document.getElementById('batch-next-btn').addEventListener('click', () => window.changeBatchPage(1));

    // Unified Download — ZIP (overlay images only) + CSV + Session JSON
    document.getElementById('download-batch-btn').addEventListener('click', async () => {
        if (!batchSessionId) {
            showToast('No Results', 'Run batch inference first.', 'error');
            return;
        }
        const btn = document.getElementById('download-batch-btn');
        btn.innerText = 'Preparing...';
        btn.disabled = true;
        try {
            const threshold = parseFloat(document.getElementById('batch-threshold').value);

            // 1. ZIP of overlay images from server
            window.open(`${API_BASE}/api/download_results/${batchSessionId}`, '_blank');

            // Small delay between downloads to avoid browser blocking
            await new Promise(r => setTimeout(r, 600));

            // 2. Results CSV
            const rows = [['filename', 'anomaly_score', 'status']];
            batchResults.forEach(r => {
                rows.push([r.filename, r.anomaly_score.toFixed(6), r.anomaly_score > threshold ? 'ANOMALY' : 'NORMAL']);
            });
            const csvBlob = new Blob([rows.map(r => r.join(',')).join('\n')], { type: 'text/csv' });
            const csvUrl = URL.createObjectURL(csvBlob);
            const csvLink = document.createElement('a');
            csvLink.href = csvUrl;
            csvLink.download = `${batchSessionId}_results.csv`;
            document.body.appendChild(csvLink);
            csvLink.click();
            document.body.removeChild(csvLink);
            URL.revokeObjectURL(csvUrl);

            await new Promise(r => setTimeout(r, 400));

            // 3. Session config JSON
            const config = getCurrentConfig();
            const spatialMode = (document.querySelector('input[name="spatial_mode"]:checked') || {}).value || '';
            const engineEl = document.getElementById('existing-engine-select');
            const sessionPayload = {
                timestamp: new Date().toISOString(),
                session_id: batchSessionId,
                preprocess_config: config,
                spatial_config: {
                    spatial_mode: spatialMode,
                    engine: engineEl ? engineEl.value : '',
                    regions: manualRegions
                },
                memory_bank: loadedBankName || null,
                detection_params: {
                    threshold: threshold,
                    alpha: parseFloat(document.getElementById('heatmap-alpha')?.value || '0.5')
                },
                results_summary: {
                    total: batchResults.length,
                    anomaly: batchResults.filter(r => r.anomaly_score > threshold).length,
                    normal: batchResults.filter(r => r.anomaly_score <= threshold).length
                }
            };
            const jsonBlob = new Blob([JSON.stringify(sessionPayload, null, 2)], { type: 'application/json' });
            const jsonUrl = URL.createObjectURL(jsonBlob);
            const jsonLink = document.createElement('a');
            jsonLink.href = jsonUrl;
            jsonLink.download = `${batchSessionId}_session_config.json`;
            document.body.appendChild(jsonLink);
            jsonLink.click();
            document.body.removeChild(jsonLink);
            URL.revokeObjectURL(jsonUrl);

            showToast('Export Complete', '3 files: ZIP (overlays) + CSV + Session JSON', 'success');
        } catch(err) {
            showToast('Export Failed', err.message, 'error');
        } finally {
            btn.innerText = 'Download Results';
            btn.disabled = false;
        }
    });

    // =========================================================================
    // END STAGE 5
    // =========================================================================

        // Trigger UI show for Stage 1 on init
    updateStageUI(1);
    loadExistingEngines();
    loadExistingBanks();
    
    // Initialize correct spatial panel visibility
    const activeSpatialMode = document.querySelector('input[name="spatial_mode"]:checked').value;
    document.getElementById('panel-grid').style.display = activeSpatialMode === 'grid' ? 'block' : 'none';
    document.getElementById('panel-manual').style.display = activeSpatialMode === 'manual' ? 'block' : 'none';
    document.getElementById('panel-canvas').style.display = (activeSpatialMode === 'manual' || activeSpatialMode === 'grid') ? 'block' : 'none';
});
