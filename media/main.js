(function() {
    // DOM elements
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

    // State
    let cameraEnabled = false;
    let cameraStream = null;
    let frameCaptureInterval = null;
    let currentMood = null;
    let moodConfidence = 0;

    // VS Code API
    const vscode = acquireVsCodeApi();

    // Set up event listeners
    function setupEventListeners() {
        enableCameraBtn.addEventListener('click', toggleCamera);
        analyzeBtn.addEventListener('click', startAnalysis);

        window.addEventListener('message', event => {
            const message = event.data;
            console.log(`[Webview] Received message: ${JSON.stringify(message)}`);
            switch (message.command) {
                case 'startCamera':
                    console.log('[Webview] Starting camera...');
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

        console.log('[Webview] Event listeners set up');
    }

    // Toggle camera
    function toggleCamera() {
        if (cameraEnabled) {
            disableCamera();
        } else {
            enableCamera();
        }
    }

    // Enable camera
    function enableCamera() {
        if (cameraEnabled) {
            console.log('[Webview] Camera already enabled');
            return;
        }

        console.log('[Webview] Requesting camera access...');
        updateStatus('busy', 'Requesting camera permission...');
        enableCameraBtn.disabled = true;
        enableCameraBtn.textContent = 'Requesting permission...';

        if (!cameraFeed || !debugCanvas) {
            console.error('[Webview] Camera feed or debug canvas not found');
            handleCameraError('Camera elements not found');
            return;
        }

        cameraFeed.width = 320;
        cameraFeed.height = 240;
        debugCanvas.width = 320;
        debugCanvas.height = 240;

        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 320 }, height: { ideal: 240 }, facingMode: 'user' }
            })
            .then(stream => {
                console.log('[Webview] Camera permission granted');
                cameraStream = stream;
                cameraFeed.srcObject = stream;

                cameraFeed.onloadedmetadata = () => {
                    console.log('[Webview] Video metadata loaded');
                    cameraFeed.style.display = 'block';
                    cameraPlaceholder.style.display = 'none';

                    cameraFeed.play()
                        .then(() => {
                            console.log('[Webview] Camera playback started');
                            enableCameraBtn.disabled = false;
                            enableCameraBtn.textContent = 'Disable Camera';
                            updateStatus('ready', 'Camera activated!');
                            cameraEnabled = true;
                            startFrameCapture();
                            vscode.postMessage({ command: 'cameraEnabled' });
                        })
                        .catch(err => {
                            console.error('[Webview] Playback error:', err);
                            handleCameraError('Failed to play video: ' + err.message);
                        });
                };
            })
            .catch(err => {
                console.error('[Webview] Camera access error:', err);
                if (err.name === 'NotAllowedError') {
                    handleCameraError('Camera permission denied');
                } else if (err.name === 'NotFoundError') {
                    handleCameraError('No camera found');
                } else {
                    handleCameraError('Camera error: ' + err.message);
                }
            });
        } else {
            console.error('[Webview] getUserMedia not supported');
            handleCameraError('Camera not supported');
        }
    }

    // Handle camera errors
    function handleCameraError(errorMessage) {
        updateStatus('', errorMessage);
        enableCameraBtn.disabled = false;
        enableCameraBtn.textContent = 'Go with mood debug';
        cameraEnabled = false;
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = null;
        }
    }

    // Disable camera
    function disableCamera() {
        console.log('[Webview] Disabling camera');
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = null;
        }
        if (cameraFeed) {
            cameraFeed.srcObject = null;
            cameraFeed.style.display = 'none';
        }
        cameraPlaceholder.style.display = 'flex';
        if (frameCaptureInterval) {
            clearInterval(frameCaptureInterval);
            frameCaptureInterval = null;
        }
        moodConfidenceSection.style.display = 'none';
        enableCameraBtn.textContent = 'Go with mood debug';
        analyzeBtn.disabled = true;
        updateStatus('', 'Camera disabled');
        vscode.postMessage({ command: 'cameraDisabled' });
        cameraEnabled = false;
    }

    // Start frame capture for mood detection
    function startFrameCapture() {
        if (!debugCanvas || !cameraFeed || !cameraEnabled) {
            console.error('[Webview] Cannot capture frames: missing elements or camera off');
            return;
        }

        console.log('[Webview] Starting frame capture');
        const context = debugCanvas.getContext('2d');
        if (!context) return;

        frameCaptureInterval = setInterval(() => {
            if (!cameraEnabled || !cameraStream) {
                clearInterval(frameCaptureInterval);
                frameCaptureInterval = null;
                return;
            }
            context.drawImage(cameraFeed, 0, 0, debugCanvas.width, debugCanvas.height);
            const imageData = debugCanvas.toDataURL('image/jpeg', 0.7);
            vscode.postMessage({ command: 'processMood', imageData });
        }, 1000);
    }

    // Handle mood detection
    function handleMoodDetection(mood, confidence) {
        console.log(`[Webview] Mood detected: ${mood} (${confidence})`);
        currentMood = mood;
        moodConfidence = confidence;
        currentMoodDisplay.textContent = mood;
        moodConfidenceSection.style.display = 'block';
        confidenceBar.style.width = `${Math.round(confidence * 100)}%`;
        if (confidence >= 0.6) analyzeBtn.disabled = false;
    }

    // Start analysis
    function startAnalysis() {
        if (!currentMood) {
            updateStatus('', 'No mood detected');
            return;
        }
        updateStatus('busy', 'Analyzing code...');
        analyzeBtn.disabled = true;
        resultsSection.style.display = 'none';

        const options = {
            syntax: document.getElementById('check-syntax')?.checked || false,
            style: document.getElementById('check-style')?.checked || false,
            performance: document.getElementById('check-performance')?.checked || false,
            bestPractices: document.getElementById('check-best-practices')?.checked || false
        };

        vscode.postMessage({ command: 'analyze', mood: currentMood, confidence: moodConfidence, options });
    }

    // Display results
    function displayResults(results) {
        console.log('[Webview] Displaying results:', results);
        analyzeBtn.disabled = false;
        updateStatus('ready', 'Analysis complete!');
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

    // Update status
    function updateStatus(statusType, message) {
        console.log(`[Webview] Status: ${statusType} - ${message}`);
        statusMessage.textContent = message;
        statusIndicator.className = 'status-indicator';
        if (statusType) statusIndicator.classList.add(statusType);
    }

    // Initialize
    function initialize() {
        console.log('[Webview] Initializing MoodLint');
        if (!cameraFeed || !enableCameraBtn) {
            console.error('[Webview] Essential DOM elements missing');
            return;
        }
        setupEventListeners();
        updateStatus('', 'Initializing...');
        // Send webviewReady after listeners are set up
        setTimeout(() => {
            console.log('[Webview] Sending webviewReady');
            vscode.postMessage({ command: 'webviewReady' });
        }, 100); // Small delay to ensure listeners are ready
    }

    // Start when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initialize);
    } else {
        initialize();
    }
})();