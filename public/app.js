document.addEventListener('DOMContentLoaded', () => {
    // Initialize stage states
    document.querySelectorAll('.stage').forEach((el, index) => {
        if (index + 1 > 1) el.classList.add('locked');
    });
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

    let currentStage = 1;
    let maxUnlockedStage = 1;
    let manualRegions = []; // Stores objects {x, y, w, h} in normalized 0-1 coords
    let isDrawing = false;
    let startX, startY;
    let backgroundImage = null; // Image object for canvas background
    let rawPreviewUrl = null; // URL of the raw image uploaded in Stage 1


    // Helper to get current config
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
        const pageIds = ['page-preprocessing', 'page-spatial', 'page-recap', 'page-detect'];
        const activePage = document.getElementById(pageIds[stage - 1]);
        if (activePage) {
            activePage.classList.add('active');
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }
        updateInspector();
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
            const response = await fetch('http://localhost:5000/api/configure_preprocess', {
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
            const response = await fetch('http://localhost:5000/api/preview_mask', {
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

    // Model Export & Status Overlay
    const overlay = document.getElementById('blocking-overlay');
    const overlayTitle = document.getElementById('overlay-title');
    const overlayMsg = document.getElementById('overlay-msg');
    const overlayProgress = document.getElementById('overlay-progress');

    exportTrtBtn.addEventListener('click', async () => {
        const spatialMode = document.querySelector('input[name="spatial_mode"]:checked').value;
        const preprocessMode = document.querySelector('input[name="preprocess_mode"]:checked').value;
        const bs = (spatialMode === 'grid') ? 4 : (spatialMode === 'manual' ? manualRegions.length : 1);
        
        let preProcInfo = "";
        if (preprocessMode === 'thresholding') {
            const min = document.getElementById('thresh-min').value;
            const max = document.getElementById('thresh-max').value;
            preProcInfo = `int${min}-${max}`;
        } else {
            const diff = document.getElementById('avg-diff-thresh').value;
            preProcInfo = `avgD${diff}`;
        }

        if (spatialMode === 'manual' && bs === 0) {
            alert('Please draw at least one region or select another mode.');
            return;
        }

        const overlay = document.getElementById('blocking-overlay');
        const overlayTitle = document.getElementById('overlay-title');
        const overlayMsg = document.getElementById('overlay-msg');
        const overlayProgress = document.getElementById('overlay-progress');
        const overlayLog = document.getElementById('overlay-log');
        
        overlay.style.display = 'flex';
        overlayProgress.style.width = '0%';
        overlayLog.style.display = 'block';
        overlayLog.innerText = '';
        document.getElementById('overlay-spinner').style.display = 'block';
        document.getElementById('overlay-close-btn').style.display = 'none';
        
        const updateStatus = (title, msg, progress) => {
            overlayTitle.innerText = title;
            overlayMsg.innerText = 'See logs below...';
            overlayProgress.style.width = progress + '%';
            
            // Append to log
            overlayLog.innerText += msg + '\n';
            overlayLog.scrollTop = overlayLog.scrollHeight;

            if (title.includes('Failed') || title.includes('Error')) {
                document.getElementById('overlay-spinner').style.display = 'none';
                document.getElementById('overlay-close-btn').style.display = 'inline-block';
            }

            // SUCCESS AUTO-CLOSE
            if (title.includes('Success')) {
                document.getElementById('overlay-spinner').style.display = 'none';
                
                // Update Phase 2 status area
                const spatialStatus = document.getElementById('spatial-status');
                if (spatialStatus) {
                    spatialStatus.innerText = '✓ ' + msg;
                    spatialStatus.style.color = 'var(--status-ready)';
                }

                setTimeout(() => {
                    overlay.style.display = 'none';
                    if (typeof loadExistingEngines === 'function') loadExistingEngines();
                }, 1500);
            }
        };

        try {
            updateStatus('Export Initiated', 'Connecting to backend...', 5);
            
            const response = await fetch('http://localhost:5000/api/export_trt', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    batch_size: bs,
                    spatial_mode: spatialMode,
                    pre_proc: preProcInfo
                })
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.error || 'Server error during export');
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
                        updateStatus(data.title, data.message, data.progress);
                        if (data.title === 'Export Failed') throw new Error(data.message);
                    } catch (e) {
                        if (line.includes('Failed')) throw new Error(line);
                    }
                });
            }


        } catch (error) {
            updateStatus('Export Failed', error.message, 0);
        }
    });

    document.getElementById('overlay-close-btn').addEventListener('click', () => {
        overlay.style.display = 'none';
    });

    // Expose update function for HTML oninput events
    window.updateSpatialPreview = function() {
        drawRegions();
    };

    async function loadExistingEngines() {
        try {
            const response = await fetch('http://localhost:5000/api/list_engines');
            const data = await response.json();
            engineSelect.innerHTML = '<option value="">-- No engine selected --</option>';
            data.engines.forEach(eng => {
                const opt = document.createElement('option');
                opt.value = eng;
                opt.innerText = eng;
                engineSelect.appendChild(opt);
            });
        } catch (e) {}
    }

    loadExistingEngines();

    saveSpatialBtn.addEventListener('click', async () => {
        const mode = document.querySelector('input[name="spatial_mode"]:checked').value;
        const engine = engineSelect.value;
        
        if (!engine) {
            alert('Please select or generate a TensorRT engine first.');
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

        const response = await fetch('http://localhost:5000/api/configure_spatial', {
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

    // Trigger UI show for Stage 1 on init
    updateStageUI(1);
    
    // Initialize correct spatial panel visibility
    const activeSpatialMode = document.querySelector('input[name="spatial_mode"]:checked').value;
    document.getElementById('panel-grid').style.display = activeSpatialMode === 'grid' ? 'block' : 'none';
    document.getElementById('panel-manual').style.display = activeSpatialMode === 'manual' ? 'block' : 'none';
    document.getElementById('panel-canvas').style.display = (activeSpatialMode === 'manual' || activeSpatialMode === 'grid') ? 'block' : 'none';
});
