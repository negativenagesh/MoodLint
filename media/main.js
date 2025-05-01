(function() {
    const cameraFeed = document.getElementById('camera-feed');
    const debugCanvas = document.getElementById('debug-canvas');
    const enableCameraBtn = document.getElementById('enable-camera-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    const statusIndicator = document.querySelector('.status-indicator');
    const statusMessage = document.querySelector('.status-message');
    const currentMoodDisplay = document.getElementById('current-mood-display');
    const confidenceBar = document.getElementById('confidence-bar');
    const moodConfidenceSection = document.querySelector('.mood-confidence');
    const resultsList = document.getElementById('results-list');
    const resultsSection = document.querySelector('.results-section');

    let externalCameraActive = false;
    let currentMood = null;
    let moodConfidence = 0;

    const vscode = acquireVsCodeApi();

    function setupEventListeners() {
        enableCameraBtn.addEventListener('click', toggleExternalCamera);
        analyzeBtn.addEventListener('click', startAnalysis);
        document.getElementById('predict-mood-btn').addEventListener('click', predictFutureMood);
    
        window.addEventListener('message', event => {
            const message = event.data;
            console.log(`[Webview] Received: ${JSON.stringify(message)}`);
            switch (message.command) {
                case 'externalCameraStarted':
                    handleExternalCameraStarted();
                    break;
                case 'externalCameraFailed':
                    handleExternalCameraFailed(message.error);
                    break;
                case 'cameraOff':
                    handleCameraOff();
                    break;
                case 'moodDetected':
                    handleMoodDetection(message.mood, message.confidence);
                    break;
                case 'analysisComplete':
                    displayResults(message.results);
                    break;
                case 'futureMoodPredicted':
                    handleFutureMoodPrediction(message);
                    break;
            }
        });
    }

    function toggleExternalCamera() {
        if (externalCameraActive) {
            disableExternalCamera();
        } else {
            enableExternalCamera();
        }
    }

    function enableExternalCamera() {
        console.log('[Webview] Requesting external camera...');
        updateStatus('busy', 'Launching external camera...');
        enableCameraBtn.disabled = true;
        enableCameraBtn.textContent = 'Launching...';

        // Send the command to start the external camera app
        vscode.postMessage({ command: 'startExternalCamera' });
    }

    function handleExternalCameraStarted() {
        console.log('[Webview] External camera started');
        enableCameraBtn.disabled = false;
        enableCameraBtn.textContent = 'Stop Camera';
        updateStatus('ready', 'External camera active');
        externalCameraActive = true;
        // Enable the analyze button since we expect mood data from the external app
        analyzeBtn.disabled = false;
    }

    function handleExternalCameraFailed(error) {
        console.error('[Webview] External camera failed:', error);
        updateStatus('', `Camera error: ${error}`);
        enableCameraBtn.disabled = false;
        enableCameraBtn.textContent = 'Go with mood debug';
        externalCameraActive = false;
        analyzeBtn.disabled = true;
    }

    function disableExternalCamera() {
        console.log('[Webview] Disabling external camera');
        vscode.postMessage({ command: 'stopExternalCamera' });
        handleCameraOff();
    }

    function handleCameraOff() {
        enableCameraBtn.textContent = 'Go with mood debug';
        analyzeBtn.disabled = true;
        updateStatus('', 'Camera off');
        externalCameraActive = false;
        moodConfidenceSection.style.display = 'none';
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

    function predictFutureMood() {
        console.log('[Webview] Predicting future mood');
        updateStatus('busy', 'Predicting your future mood...');
        
        vscode.postMessage({ 
            command: 'predictFutureMood',
            currentMood: currentMood || 'neutral'
        });
    }

    function handleFutureMoodPrediction(message) {
        if (message.error) {
            updateStatus('', `Error: ${message.error}`);
            return;
        }
        
        updateStatus('ready', message.message || `Your future mood: ${message.mood}`);
        
        // Display the prediction in a visual way
        const moodDisplay = document.getElementById('current-mood-display');
        if (moodDisplay) {
            moodDisplay.innerHTML = `${currentMood || 'Unknown'} → <strong>${message.mood}</strong> (predicted)`;
        }
        
        // Show the confidence bar
        if (message.confidence && moodConfidenceSection) {
            moodConfidenceSection.style.display = 'block';
            confidenceBar.style.width = `${Math.round(message.confidence * 100)}%`;
        }
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
        updateStatus('', 'Click "Go with mood debug" to enable external camera');
        vscode.postMessage({ command: 'webviewReady' });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initialize);
    } else {
        initialize();
    }
})();