// Application State & Interactive Configuration Editor Logic

let selectedFile = null;
let currentResultData = null;
let allConfigsData = {};

window.addEventListener('DOMContentLoaded', () => {
    fetchConfigsFromAPI();
});

async function fetchConfigsFromAPI() {
    try {
        const res = await fetch('/configs');
        if (res.ok) {
            allConfigsData = await res.json();
            populateConfigSelect();
            populateFieldSelect();
            renderConfiguredFieldsList();
        }
    } catch (e) {
        console.warn('Using default preset configurations:', e);
    }
}

// Tab Switching
function switchTab(tabId) {
    document.querySelectorAll('.tab-section').forEach(sec => sec.style.display = 'none');
    document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));

    document.getElementById(`tab-${tabId}`).style.display = 'block';
    document.getElementById(`tab-${tabId}-btn`).classList.add('active');

    if (tabId === 'calibrate') {
        populateConfigSelect();
        populateFieldSelect();
        renderConfiguredFieldsList();
        if (selectedFile) {
            initCanvasWithFile(selectedFile);
        }
    }
}

function handleFileSelect(event) {
    const files = event.target.files;
    if (files && files[0]) {
        setFile(files[0]);
    }
}

const dropzone = document.getElementById('image-dropzone');
dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.style.borderColor = 'var(--primary)';
});

dropzone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dropzone.style.borderColor = 'rgba(59, 130, 246, 0.4)';
});

dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.style.borderColor = 'rgba(59, 130, 246, 0.4)';
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        setFile(e.dataTransfer.files[0]);
    }
});

function setFile(file) {
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        const preview = document.getElementById('image-preview');
        preview.src = e.target.result;
        preview.style.display = 'block';
        document.getElementById('dropzone-prompt').style.display = 'none';
        document.getElementById('process-btn').disabled = false;
        if (document.getElementById('tab-calibrate').style.display !== 'none') {
            initCanvasWithFile(file);
        }
    };
    reader.readAsDataURL(file);
}

// Document Processing API Call
async function processDocument() {
    if (!selectedFile) return;

    const btn = document.getElementById('process-btn');
    btn.disabled = true;
    btn.innerHTML = '⏳ Preprocessing & OCR Extraction...';

    const docType = document.getElementById('doc-type-select').value;
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('document_type', docType);

    try {
        const ocrRes = await fetch('/ocr', {
            method: 'POST',
            body: formData
        });

        if (!ocrRes.ok) {
            throw new Error(`Server returned status: ${ocrRes.status}`);
        }

        const data = await ocrRes.json();
        currentResultData = data;

        const debugFormData = new FormData();
        debugFormData.append('file', selectedFile);
        debugFormData.append('document_type', docType);

        const debugRes = await fetch('/ocr/debug', {
            method: 'POST',
            body: debugFormData
        });

        if (debugRes.ok) {
            const blob = await debugRes.blob();
            document.getElementById('debug-overlay-img').src = URL.createObjectURL(blob);
        }

        renderResults(data);

    } catch (err) {
        alert(`Error processing document: ${err.message}`);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '🚀 Run Preprocessing & OCR';
    }
}

// Render Results View
function renderResults(data) {
    document.getElementById('empty-state').style.display = 'none';
    document.getElementById('status-bar').style.display = 'flex';
    document.getElementById('results-card').style.display = 'block';

    // Review state badge
    const reviewBadge = document.getElementById('review-state-badge');
    const stateStr = data.review_state || 'AUTO_ACCEPT';
    reviewBadge.innerText = stateStr;
    reviewBadge.className = `review-state-badge ${stateStr}`;

    if (data.quality_report) {
        document.getElementById('metric-blur').innerText = data.quality_report.blur_score ? data.quality_report.blur_score.toFixed(1) : '-';
        document.getElementById('metric-overall').innerText = data.quality_report.overall_score ? (data.quality_report.overall_score * 100).toFixed(0) + '%' : '-';
        document.getElementById('status-text').innerText = data.quality_report.warnings && data.quality_report.warnings.length ? 'NEEDS ATTENTION' : 'USABLE QUALITY';
    }

    // Preprocessing audit badges
    const auditBadges = document.getElementById('audit-badges');
    auditBadges.innerHTML = '';
    
    if (data.preprocessing_metadata) {
        const meta = data.preprocessing_metadata;
        const badges = [];

        if (meta.exif_orientation_corrected) badges.push('✓ EXIF Orientation Corrected');
        if (meta.document_detection && meta.document_detection.applied) {
            const conf = (meta.document_detection.confidence * 100).toFixed(0);
            badges.push(`✓ Document Boundary Detected (${conf}%)`);
        }
        if (meta.stages) {
            meta.stages.forEach(s => {
                if (s !== 'original' && s !== 'orientation_corrected') {
                    badges.push(`✓ Stage: ${s.replace(/_/g, ' ')}`);
                }
            });
        }

        if (badges.length === 0) {
            badges.push('✓ Base Image Normalization Only');
        }

        badges.forEach(bText => {
            const span = document.createElement('span');
            span.className = 'audit-badge';
            span.innerText = bText;
            auditBadges.appendChild(span);
        });
    }

    // Bilingual name display
    const bilingualBox = document.getElementById('bilingual-container');
    if (data.name && (data.name.en || data.name.ur)) {
        bilingualBox.style.display = 'grid';
        document.getElementById('name-en-val').innerText = data.name.en || '-';
        document.getElementById('name-ur-val').innerText = data.name.ur || '-';
        document.getElementById('father-en-val').innerText = (data.father_name && data.father_name.en) ? data.father_name.en : '-';
        document.getElementById('father-ur-val').innerText = (data.father_name && data.father_name.ur) ? data.father_name.ur : '-';
    } else {
        bilingualBox.style.display = 'none';
    }

    // Fields grid
    const fieldsGrid = document.getElementById('fields-grid');
    fieldsGrid.innerHTML = '';

    const fields = data.fields || {};
    for (const [key, field] of Object.entries(fields)) {
        const card = document.createElement('div');
        card.className = 'field-card';

        const valClass = field.validated ? 'valid' : 'invalid';
        const valSymbol = field.validated ? '✓ VALIDATED' : '? UNVALIDATED';

        const bboxNormStr = field.bbox_norm ? `norm: [${field.bbox_norm.join(', ')}]` : '';

        card.innerHTML = `
            <div class="field-name">${key.replace(/_/g, ' ')}</div>
            <div class="field-val">${field.value || '<span style="color:var(--text-muted)">None</span>'}</div>
            <div class="field-coords">${bboxNormStr}</div>
            <span class="val-badge ${valClass}">${valSymbol}</span>
        `;
        fieldsGrid.appendChild(card);
    }

    document.getElementById('json-code').innerText = JSON.stringify(data, null, 2);
}

function toggleResultView(paneName) {
    document.querySelectorAll('.result-pane').forEach(p => p.style.display = 'none');
    document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));

    document.getElementById(`view-${paneName}`).style.display = 'block';
    document.getElementById(`view-${paneName}-btn`).classList.add('active');
}

// Canvas & Calibrator Logic
const canvas = document.getElementById('calibrator-canvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;
let startX = 0, startY = 0;
let calibImage = new Image();

function populateConfigSelect() {
    const select = document.getElementById('calib-config-select');
    const currVal = select.value;
    select.innerHTML = '';

    const keys = Object.keys(allConfigsData);
    if (keys.length === 0) {
        ['passport.yaml', 'cnic_front.yaml', 'cnic_back.yaml'].forEach(k => {
            const opt = document.createElement('option');
            opt.value = k;
            opt.textContent = k;
            select.appendChild(opt);
        });
    } else {
        keys.forEach(k => {
            const opt = document.createElement('option');
            opt.value = k;
            opt.textContent = k;
            select.appendChild(opt);
        });
    }

    if (currVal && select.querySelector(`option[value="${currVal}"]`)) {
        select.value = currVal;
    }
}

function populateFieldSelect() {
    const configName = document.getElementById('calib-config-select').value;
    const select = document.getElementById('calib-field-select');
    select.innerHTML = '<option value="">-- Add New Field or Select Existing --</option>';

    const cfg = allConfigsData[configName] || {};
    const fields = cfg.fields || {};

    for (const [key, fieldObj] of Object.entries(fields)) {
        const opt = document.createElement('option');
        opt.value = key;
        opt.textContent = `${fieldObj.label || key} (${key})`;
        select.appendChild(opt);
    }
}

function renderConfiguredFieldsList() {
    const configName = document.getElementById('calib-config-select').value;
    document.getElementById('current-config-label').innerText = configName;

    const listDiv = document.getElementById('configured-fields-list');
    listDiv.innerHTML = '';

    const cfg = allConfigsData[configName] || {};
    const fields = cfg.fields || {};

    if (Object.keys(fields).length === 0) {
        listDiv.innerHTML = '<span style="color:var(--text-muted); font-size:0.8rem;">No fields configured in this file yet.</span>';
        return;
    }

    for (const [key, fieldObj] of Object.entries(fields)) {
        const chip = document.createElement('div');
        chip.className = 'configured-field-chip';
        chip.innerHTML = `
            <span>${fieldObj.label || key}</span>
            <span class="delete-btn" title="Delete Field" onclick="deleteField('${configName}', '${key}')">×</span>
        `;
        chip.onclick = (e) => {
            if (e.target.className !== 'delete-btn') {
                document.getElementById('calib-field-select').value = key;
                onFieldSelectChange();
            }
        };
        listDiv.appendChild(chip);
    }
}

function onConfigSelectChange() {
    populateFieldSelect();
    renderConfiguredFieldsList();
    drawCanvasFromInputs();
}

function onFieldSelectChange() {
    const configName = document.getElementById('calib-config-select').value;
    const fieldKey = document.getElementById('calib-field-select').value;
    if (!fieldKey) return;

    const cfg = allConfigsData[configName] || {};
    const item = (cfg.fields || {})[fieldKey];

    if (item) {
        document.getElementById('calib-field-key').value = fieldKey;
        document.getElementById('calib-field-label').value = item.label || '';
        document.getElementById('calib-strategy').value = item.strategy || 'region';
        document.getElementById('calib-language').value = item.language || 'en';

        if (item.region) {
            document.getElementById('calib-x1').value = item.region.x1.toFixed(3);
            document.getElementById('calib-y1').value = item.region.y1.toFixed(3);
            document.getElementById('calib-x2').value = item.region.x2.toFixed(3);
            document.getElementById('calib-y2').value = item.region.y2.toFixed(3);
        }

        if (item.anchor) {
            document.getElementById('calib-anchor-kw').value = item.anchor.keyword || '';
            document.getElementById('calib-anchor-dir').value = item.anchor.direction || 'right';
        }

        document.getElementById('calib-normalization').value = item.normalization || 'none';
        document.getElementById('calib-validator').value = item.validator || 'none';

        toggleAnchorInputs();
        drawCanvasFromInputs();
    }
}

function toggleAnchorInputs() {
    const strategy = document.getElementById('calib-strategy').value;
    document.getElementById('anchor-inputs-row').style.display = (strategy === 'anchor') ? 'flex' : 'none';
}

function showNewConfigModal() {
    const configName = prompt("Enter new config file name (e.g. driving_license.yaml):");
    if (!configName) return;
    const docType = prompt("Enter document type ID (e.g. driving_license):", configName.replace('.yaml', ''));
    if (!docType) return;

    fetch('/config/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config_name: configName, document_type: docType, language: 'en' })
    }).then(res => res.json()).then(data => {
        alert(data.message);
        fetchConfigsFromAPI();
    });
}

function deleteField(configName, fieldKey) {
    if (confirm(`Are you sure you want to delete field '${fieldKey}' from ${configName}?`)) {
        fetch(`/config/field/${configName}/${fieldKey}`, { method: 'DELETE' })
            .then(res => res.json())
            .then(data => {
                alert(data.message);
                fetchConfigsFromAPI();
            });
    }
}

function initCanvasWithFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        calibImage.onload = () => {
            canvas.width = calibImage.width;
            canvas.height = calibImage.height;
            drawCanvasFromInputs();
        };
        calibImage.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

function drawCanvasFromInputs() {
    const x1 = parseFloat(document.getElementById('calib-x1').value) || 0;
    const y1 = parseFloat(document.getElementById('calib-y1').value) || 0;
    const x2 = parseFloat(document.getElementById('calib-x2').value) || 0;
    const y2 = parseFloat(document.getElementById('calib-y2').value) || 0;

    const rect = {
        x: x1 * canvas.width,
        y: y1 * canvas.height,
        w: (x2 - x1) * canvas.width,
        h: (y2 - y1) * canvas.height
    };

    drawCanvas(rect);
}

function onCoordInputChange() {
    drawCanvasFromInputs();
}

function drawCanvas(activeRect) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (calibImage.src) {
        ctx.drawImage(calibImage, 0, 0);
    }

    const configName = document.getElementById('calib-config-select').value;
    const cfg = allConfigsData[configName] || {};
    const fields = cfg.fields || {};

    for (const [key, fieldObj] of Object.entries(fields)) {
        if (fieldObj.region) {
            const px1 = fieldObj.region.x1 * canvas.width;
            const py1 = fieldObj.region.y1 * canvas.height;
            const pw = (fieldObj.region.x2 - fieldObj.region.x1) * canvas.width;
            const ph = (fieldObj.region.y2 - fieldObj.region.y1) * canvas.height;

            ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 4]);
            ctx.strokeRect(px1, py1, pw, ph);
            ctx.setLineDash([]);
        }
    }

    if (activeRect && activeRect.w > 0 && activeRect.h > 0) {
        ctx.strokeStyle = '#3b82f6';
        ctx.lineWidth = 3;
        ctx.strokeRect(activeRect.x, activeRect.y, activeRect.w, activeRect.h);
        ctx.fillStyle = 'rgba(59, 130, 246, 0.25)';
        ctx.fillRect(activeRect.x, activeRect.y, activeRect.w, activeRect.h);

        const name = document.getElementById('calib-field-key').value || 'Selected Region';
        ctx.fillStyle = '#2563eb';
        ctx.fillRect(activeRect.x, Math.max(0, activeRect.y - 24), ctx.measureText(name).width + 16, 24);
        ctx.fillStyle = '#ffffff';
        ctx.font = '14px Inter, sans-serif';
        ctx.fillText(name, activeRect.x + 8, Math.max(16, activeRect.y - 7));
    }
}

canvas.addEventListener('mousedown', (e) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    startX = (e.clientX - rect.left) * scaleX;
    startY = (e.clientY - rect.top) * scaleY;
    isDrawing = true;
});

canvas.addEventListener('mousemove', (e) => {
    if (!isDrawing) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const currX = (e.clientX - rect.left) * scaleX;
    const currY = (e.clientY - rect.top) * scaleY;

    const w = currX - startX;
    const h = currY - startY;

    drawCanvas({ x: startX, y: startY, w: w, h: h });
});

canvas.addEventListener('mouseup', (e) => {
    if (!isDrawing) return;
    isDrawing = false;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const endX = (e.clientX - rect.left) * scaleX;
    const endY = (e.clientY - rect.top) * scaleY;

    const x1 = Math.min(startX, endX) / canvas.width;
    const y1 = Math.min(startY, endY) / canvas.height;
    const x2 = Math.max(startX, endX) / canvas.width;
    const y2 = Math.max(startY, endY) / canvas.height;

    document.getElementById('calib-x1').value = x1.toFixed(3);
    document.getElementById('calib-y1').value = y1.toFixed(3);
    document.getElementById('calib-x2').value = x2.toFixed(3);
    document.getElementById('calib-y2').value = y2.toFixed(3);

    drawCanvasFromInputs();
});

// Save Calibrated Region API Call
async function saveCalibratedRegion() {
    const configName = document.getElementById('calib-config-select').value;
    const fieldKey = document.getElementById('calib-field-key').value.trim();
    const label = document.getElementById('calib-field-label').value.trim() || fieldKey;

    if (!fieldKey) {
        alert('Please enter a field key name.');
        return;
    }

    const payload = {
        config_name: configName,
        field_key: fieldKey,
        label: label,
        language: document.getElementById('calib-language').value,
        strategy: document.getElementById('calib-strategy').value,
        x1: parseFloat(document.getElementById('calib-x1').value),
        y1: parseFloat(document.getElementById('calib-y1').value),
        x2: parseFloat(document.getElementById('calib-x2').value),
        y2: parseFloat(document.getElementById('calib-y2').value),
        anchor_keyword: document.getElementById('calib-anchor-kw').value,
        anchor_direction: document.getElementById('calib-anchor-dir').value,
        normalization: document.getElementById('calib-normalization').value,
        validator: document.getElementById('calib-validator').value
    };

    try {
        const res = await fetch('/config/field', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (res.ok) {
            const fb = document.getElementById('calib-feedback');
            fb.innerText = `✓ Successfully saved field '${fieldKey}' into ${configName}!`;
            fb.style.display = 'block';
            setTimeout(() => fb.style.display = 'none', 4000);
            fetchConfigsFromAPI();
        } else {
            alert('Failed to save field config.');
        }
    } catch (e) {
        alert(`Error saving field: ${e.message}`);
    }
}
