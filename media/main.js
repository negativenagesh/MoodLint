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
        document.getElementById('generate-future-mood-btn').addEventListener('click', generateFutureMood);
    
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
                    handleMoodDetection(message);
                    break;
                case 'analysisComplete':
                    displayResults(message.results);
                    break;
                case 'futureMoodPredicted':
                    handleFutureMoodPrediction(message);
                    break;
                case 'futureMoodGenerated':
                    handleFutureMoodGeneration(message);
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

    function handleMoodDetection(message) {
        console.log(`[Webview] Mood detected: ${message.mood} (${message.confidence})`);
        
        currentMood = message.mood;
        moodConfidence = message.confidence;
        
        currentMoodDisplay.textContent = message.mood;
        moodConfidenceSection.style.display = 'block';
        confidenceBar.style.width = `${Math.round(message.confidence * 100)}%`;
        
        // Enable analyze button if confidence is high enough
        if (message.confidence >= 0.6) {
            analyzeBtn.disabled = false;
        }
        
        // Enable the Generate Future Mood button if we have an image
        if (message.hasImage) {
            const generateFutureMoodBtn = document.getElementById('generate-future-mood-btn');
            if (generateFutureMoodBtn) {
                generateFutureMoodBtn.disabled = false;
                console.log('[Webview] Enabling Generate Future Mood button');
            } else {
                console.error('[Webview] Could not find generate-future-mood-btn element');
            }
        } else {
            console.log('[Webview] No image detected, future mood button remains disabled');
        }
        
        updateStatus('ready', `Mood detected: ${message.mood}`);
    }

    function predictFutureMood() {
        console.log('[Webview] Predicting future mood');
        updateStatus('busy', 'Predicting your future mood...');
        
        vscode.postMessage({ 
            command: 'predictFutureMood',
            currentMood: currentMood || 'neutral'
        });
    }

    function generateFutureMood() {
        if (!currentMood) {
            console.log('[Webview] No mood detected, cannot generate future mood');
            updateStatus('error', 'Please detect your mood first');
            return;
        }
        
        const generateFutureMoodBtn = document.getElementById('generate-future-mood-btn');
        if (generateFutureMoodBtn && generateFutureMoodBtn.disabled) {
            console.log('[Webview] Generate future mood button is disabled, ignoring click');
            updateStatus('error', 'Please upload your image with Go with mood debug first');
            return;
        }
        
        console.log(`[Webview] Generating future mood for: ${currentMood}`);
        updateStatus('busy', `Launching future mood generation for ${currentMood}...`);
        
        vscode.postMessage({
            command: 'generateFutureMood',
            currentMood: currentMood
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

    function handleFutureMoodGeneration(message) {
        if (message.error) {
            updateStatus('', `Error: ${message.error}`);
            return;
        }
        
        updateStatus('ready', 'Future mood visualization generated!');
        
        // If we have an image URL to display
        if (message.imageUrl) {
            // Create or get a container for the generated image
            let container = document.getElementById('generated-mood-container');
            if (!container) {
                container = document.createElement('div');
                container.id = 'generated-mood-container';
                container.style.textAlign = 'center';
                container.style.marginTop = '20px';
                container.style.padding = '10px';
                container.style.backgroundColor = 'rgba(0,0,0,0.05)';
                container.style.borderRadius = '8px';
                document.querySelector('.mood-status').after(container);
            }
            
            // Create title
            if (!document.getElementById('generated-mood-title')) {
                const title = document.createElement('h3');
                title.id = 'generated-mood-title';
                title.textContent = 'Your Future Mood Visualization';
                container.appendChild(title);
            }
            
            // Create or update image
            let img = document.getElementById('generated-mood-image');
            if (!img) {
                img = document.createElement('img');
                img.id = 'generated-mood-image';
                img.style.maxWidth = '100%';
                img.style.maxHeight = '300px';
                img.style.borderRadius = '4px';
                img.style.boxShadow = '0 4px 8px rgba(0,0,0,0.1)';
                container.appendChild(img);
            }
            
            // Set the image source
            img.src = message.imageUrl;
            img.alt = `Generated visualization for ${message.mood || 'future'} mood`;
            
            // Add caption if provided
            if (message.caption) {
                let caption = document.getElementById('generated-mood-caption');
                if (!caption) {
                    caption = document.createElement('p');
                    caption.id = 'generated-mood-caption';
                    caption.style.marginTop = '10px';
                    caption.style.fontStyle = 'italic';
                    container.appendChild(caption);
                }
                caption.textContent = message.caption;
            }
            
            // Make sure the container is visible
            container.style.display = 'block';
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
        // ...existing code...
        
        // Generate Future Mood button
        const generateFutureMoodBtn = document.getElementById('generate-future-mood-btn');
        if (generateFutureMoodBtn) {
            generateFutureMoodBtn.addEventListener('click', generateFutureMood);
        }

        // Set up all event listeners
        setupEventListeners();
    }

    // Check if the document is already loaded
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initialize);
    } else {
        initialize();
    }
})();