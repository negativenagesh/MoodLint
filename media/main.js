(function() {
    const cameraFeed = document.getElementById('camera-feed');
    const debugCanvas = document.getElementById('debug-canvas');
    const cameraPlaceholder = document.getElementById('camera-placeholder');
    const enableCameraBtn = document.getElementById('enable-camera-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    const statusIndicator = document.querySelector('.status-indicator');
    const statusMessage = document.querySelector('.status-message');
    const currentMoodDisplay = document.getElementById('current-mood-display');
    const confidenceBar = document.getElementById('confidence-bar');
    const moodConfidenceSection = document.querySelector('.mood-confidence');
    const resultsList = document.getElementById('results-list');
    const resultsSection = document.querySelector('.results-section');

    let cameraEnabled = false;
    let cameraStream = null;
    let frameCaptureInterval = null;
    let currentMood = null;
    let moodConfidence = 0;

    const vscode = acquireVsCodeApi();

    function setupEventListeners() {
        enableCameraBtn.addEventListener('click', toggleCamera);
        analyzeBtn.addEventListener('click', startAnalysis);

        window.addEventListener('message', event => {
            const message = event.data;
            console.log(`[Webview] Received: ${JSON.stringify(message)}`);
            switch (message.command) {
                case 'startCamera':
                    console.log('[Webview] Triggering camera start');
                    enableCamera();
                    break;
                case 'moodDetected':
                    handleMoodDetection(message.mood, message.confidence);
                    break;
                case 'analysisComplete':
                    displayResults(message.results);
                    break;
            }
        });
    }

    function toggleCamera() {
        if (cameraEnabled) {
            disableCamera();
        } else {
            enableCamera();
        }
    }

    function enableCamera() {
        if (cameraEnabled) {
            console.log('[Webview] Camera already enabled');
            return;
        }

        console.log('[Webview] Requesting camera...');
        updateStatus('busy', 'Requesting camera access...');
        enableCameraBtn.disabled = true;
        enableCameraBtn.textContent = 'Requesting...';

        if (!cameraFeed || !debugCanvas) {
            console.error('[Webview] Missing camera elements');
            handleCameraError('Camera setup failed');
            return;
        }

        cameraFeed.width = 320;
        cameraFeed.height = 240;
        debugCanvas.width = 320;
        debugCanvas.height = 240;

        navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 320 }, height: { ideal: 240 }, facingMode: 'user' }
        })
        .then(stream => {
            console.log('[Webview] Camera access granted');
            cameraStream = stream;
            cameraFeed.srcObject = stream;

            cameraFeed.onloadedmetadata = () => {
                console.log('[Webview] Video loaded');
                cameraFeed.style.display = 'block';
                cameraPlaceholder.style.display = 'none';
                cameraFeed.play()
                    .then(() => {
                        console.log('[Webview] Camera active');
                        enableCameraBtn.disabled = false;
                        enableCameraBtn.textContent = 'Disable Camera';
                        updateStatus('ready', 'Camera on!');
                        cameraEnabled = true;
                        startFrameCapture();
                        vscode.postMessage({ command: 'cameraEnabled' });
                    })
                    .catch(err => {
                        console.error('[Webview] Play error:', err);
                        handleCameraError('Cannot play video: ' + err.message);
                    });
            };
        })
        .catch(err => {
            console.error('[Webview] Camera error:', err);
            let errorMessage = 'Unknown camera error';
            if (err.name === 'NotAllowedError') {
                errorMessage = 'Camera permission denied. Please allow access in your browser/OS settings and retry.';
            } else if (err.name === 'NotFoundError') {
                errorMessage = 'No camera found. Connect a camera and retry.';
            } else if (err.name === 'NotReadableError') {
                errorMessage = 'Camera in use by another app.';
            } else {
                errorMessage = 'Camera error: ' + err.message;
            }
            handleCameraError(errorMessage);
        });
    }

    function handleCameraError(errorMessage) {
        console.log('[Webview] Handling error:', errorMessage);
        updateStatus('', errorMessage);
        enableCameraBtn.disabled = false;
        enableCameraBtn.textContent = 'Go with mood debug';
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = null;
        }
        cameraEnabled = false;
    }

    function disableCamera() {
        console.log('[Webview] Disabling camera');
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = null;
        }
        cameraFeed.srcObject = null;
        cameraFeed.style.display = 'none';
        cameraPlaceholder.style.display = 'flex';
        if (frameCaptureInterval) {
            clearInterval(frameCaptureInterval);
        }
        moodConfidenceSection.style.display = 'none';
        enableCameraBtn.textContent = 'Go with mood debug';
        analyzeBtn.disabled = true;
        updateStatus('', 'Camera off');
        vscode.postMessage({ command: 'cameraDisabled' });
        cameraEnabled = false;
    }

    function startFrameCapture() {
        if (!cameraEnabled || !cameraStream) return;
        console.log('[Webview] Starting frame capture');
        const context = debugCanvas.getContext('2d');
        frameCaptureInterval = setInterval(() => {
            context.drawImage(cameraFeed, 0, 0, 320, 240);
            const imageData = debugCanvas.toDataURL('image/jpeg', 0.7);
            vscode.postMessage({ command: 'processMood', imageData });
        }, 1000);
    }

    function handleMoodDetection(mood, confidence) {
        console.log(`[Webview] Mood: ${mood} (${confidence})`);
        currentMood = mood;
        moodConfidence = confidence;
        currentMoodDisplay.textContent = mood;
        moodConfidenceSection.style.display = 'block';
        confidenceBar.style.width = `${Math.round(confidence * 100)}%`;
        if (confidence >= 0.6) analyzeBtn.disabled = false;
    }

    function startAnalysis() {
        if (!currentMood) {
            updateStatus('', 'No mood detected');
            return;
        }
        updateStatus('busy', 'Analyzing...');
        analyzeBtn.disabled = true;
        resultsSection.style.display = 'none';

        const options = {
            syntax: document.getElementById('check-syntax').checked,
            style: document.getElementById('check-style').checked,
            performance: document.getElementById('check-performance').checked,
            bestPractices: document.getElementById('check-best-practices').checked
        };

        vscode.postMessage({ command: 'analyze', mood: currentMood, confidence: moodConfidence, options });
    }

    function displayResults(results) {
        console.log('[Webview] Results:', results);
        analyzeBtn.disabled = false;
        updateStatus('ready', 'Analysis done!');
        resultsList.innerHTML = '';

        let errors = 0, warnings = 0, suggestions = 0;
        if (results.issues && results.issues.length) {
            results.issues.forEach(issue => {
                if (issue.severity === 'error') errors++;
                else if (issue.severity === 'warning') warnings++;
                else suggestions++;
                const item = document.createElement('div');
                item.className = `result-item ${issue.severity}`;
                item.innerHTML = `<div class="result-location">Line ${issue.line}</div><div class="result-message">${issue.message}</div>`;
                resultsList.appendChild(item);
            });
        } else {
            resultsList.textContent = 'No issues found!';
        }

        document.getElementById('errors-count').textContent = errors;
        document.getElementById('warnings-count').textContent = warnings;
        document.getElementById('suggestions-count').textContent = suggestions;
        resultsSection.style.display = 'block';
    }

    function updateStatus(statusType, message) {
        statusMessage.textContent = message;
        statusIndicator.className = 'status-indicator' + (statusType ? ` ${statusType}` : '');
    }

    function initialize() {
        console.log('[Webview] Initializing');
        setupEventListeners();
        updateStatus('', 'Initializing...');
        console.log('[Webview] Sending webviewReady');
        vscode.postMessage({ command: 'webviewReady' });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initialize);
    } else {
        initialize();
    }
})();