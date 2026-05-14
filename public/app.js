document.addEventListener('DOMContentLoaded', () => {
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

    // Interactive Thresholding Preview Update
    const updateThreshPreview = debounce(async () => {
        if (threshPreviewUpload.files.length === 0) return;
        
        const formData = new FormData();
        formData.append('test_image', threshPreviewUpload.files[0]);
        // Force evaluation via current settings overlaying state
        formData.append('config', JSON.stringify(getCurrentConfig()));

        threshLiveContainer.style.display = 'block';
        threshPreviewImg.style.opacity = '0.5';

        try {
            const response = await fetch('http://localhost:5000/api/preview_mask', {
                method: 'POST',
                body: formData
            });
            if (response.ok) {
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                threshPreviewImg.src = url;
            }
        } catch (error) {
            console.error('Threshold Preview error:', error);
        } finally {
            threshPreviewImg.style.opacity = '1';
        }
    }, 400);

    // Bind threshold sliders to trigger live update
    const threshInputs = ['thresh-channel', 'thresh-min', 'thresh-max', 'morph-open', 'morph-close'];
    threshInputs.forEach(id => {
        const el = document.getElementById(id);
        if(el) {
            el.addEventListener('input', updateThreshPreview);
        }
    });

    // Also trigger update when a file is selected
    threshPreviewUpload.addEventListener('change', updateThreshPreview);

    // Interactive Average Preview Update
    const updateAvgPreview = debounce(async () => {
        if (avgPreviewUpload.files.length === 0) return;
        
        const config = getCurrentConfig();
        const formData = new FormData();
        formData.append('test_image', avgPreviewUpload.files[0]);
        formData.append('config', JSON.stringify(config));

        // For average mode, we rely on the backend having the reference already saved from a previous "Save Configuration" call.
        // We do NOT send the ZIP files on every slider drag.

        avgLiveContainer.style.display = 'block';
        avgPreviewImg.style.opacity = '0.5';

        try {
            const response = await fetch('http://localhost:5000/api/preview_mask', {
                method: 'POST',
                body: formData
            });
            if (response.ok) {
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                avgPreviewImg.src = url;
            }
        } catch (error) {
            console.error('Average Preview error:', error);
        } finally {
            avgPreviewImg.style.opacity = '1';
        }
    }, 400);

    // Bind average sliders to trigger live update
    const avgInputs = ['avg-diff-thresh', 'avg-min-ratio', 'avg-fill-holes'];
    avgInputs.forEach(id => {
        const el = document.getElementById(id);
        if(el) {
            el.addEventListener('input', updateAvgPreview);
            el.addEventListener('change', updateAvgPreview);
        }
    });

    avgPreviewUpload.addEventListener('change', updateAvgPreview);
});
