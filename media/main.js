// MoodLint Main Script

(function() {
    // Initialize state
    const state = {
        selectedMood: null,
        analyzing: false,
        results: []
    };
    
    // Get elements
    const moodOptions = document.querySelectorAll('.mood-option');
    const analyzeBtn = document.getElementById('analyze-btn');
    const statusSection = document.querySelector('.status-section');
    const statusIndicator = document.querySelector('.status-indicator');
    const statusMessage = document.querySelector('.status-message');
    const resultsSection = document.querySelector('.results-section');
    const resultsList = document.getElementById('results-list');
    const errorsCount = document.getElementById('errors-count');
    const warningsCount = document.getElementById('warnings-count');
    const suggestionsCount = document.getElementById('suggestions-count');
    
    // Initialize VS Code API connection
    const vscode = acquireVsCodeApi();
    
    // Setup event listeners
    function initializeEventListeners() {
        // Mood selection
        moodOptions.forEach(option => {
            option.addEventListener('click', () => {
                selectMood(option);
            });
        });
        
        // Analyze button
        analyzeBtn.addEventListener('click', () => {
            startAnalysis();
        });
        
        // Listen for messages from the extension
        window.addEventListener('message', event => {
            const message = event.data;
            handleExtensionMessage(message);
        });
    }
    
    // Handle mood selection
    function selectMood(option) {
        // Remove selected class from all options
        moodOptions.forEach(opt => opt.classList.remove('selected'));
        
        // Add selected class to clicked option
        option.classList.add('selected');
        
        // Update selected mood
        state.selectedMood = option.getAttribute('data-mood');
        
        // Update status
        updateStatus('ready', `Ready to analyze with mood: ${state.selectedMood}`);
    }
    
    // Update the status section
    function updateStatus(statusType, message) {
        statusMessage.textContent = message;
        
        // Update indicator
        statusIndicator.className = 'status-indicator';
        if (statusType) {
            statusIndicator.classList.add(statusType);
        }
    }
    
    // Start the analysis
    function startAnalysis() {
        if (!state.selectedMood) {
            updateStatus('', 'Please select a mood first.');
            return;
        }
        
        if (state.analyzing) {
            return; // Prevent multiple clicks
        }
        
        // Update state
        state.analyzing = true;
        
        // Get checkbox values
        const checkSyntax = document.getElementById('check-syntax').checked;
        const checkStyle = document.getElementById('check-style').checked;
        const checkPerformance = document.getElementById('check-performance').checked;
        const checkBestPractices = document.getElementById('check-best-practices').checked;
        
        // Update UI
        updateStatus('busy', `Analyzing code with mood: ${state.selectedMood}...`);
        analyzeBtn.disabled = true;
        resultsSection.style.display = 'none';
        
        // Send message to extension
        vscode.postMessage({
            command: 'analyze',
            mood: state.selectedMood,
            options: {
                syntax: checkSyntax,
                style: checkStyle,
                performance: checkPerformance,
                bestPractices: checkBestPractices
            }
        });
    }
    
    // Handle messages from the extension
    function handleExtensionMessage(message) {
        switch (message.command) {
            case 'analysisComplete':
                renderResults(message.results);
                break;
        }
    }
    
    // Render analysis results
    function renderResults(results) {
        // Reset state
        state.analyzing = false;
        analyzeBtn.disabled = false;
        
        // Update status
        updateStatus('ready', `Analysis complete for mood: ${results.mood}`);
        
        // Clear previous results
        resultsList.innerHTML = '';
        
        // Count types
        let errors = 0;
        let warnings = 0;
        let suggestions = 0;
        
        // Process and render issues
        if (results.issues && results.issues.length > 0) {
            results.issues.forEach(issue => {
                // Count by severity
                if (issue.severity === 'error') errors++;
                else if (issue.severity === 'warning') warnings++;
                else suggestions++;
                
                // Create result item
                const resultItem = document.createElement('div');
                resultItem.className = `result-item ${issue.severity}`;
                
                // Create location element
                const locationElement = document.createElement('div');
                locationElement.className = 'result-location';
                locationElement.textContent = `Line ${issue.line}`;
                
                // Create message element
                const messageElement = document.createElement('div');
                messageElement.className = 'result-message';
                messageElement.textContent = issue.message;
                
                // Assemble item
                resultItem.appendChild(locationElement);
                resultItem.appendChild(messageElement);
                
                // Add to list
                resultsList.appendChild(resultItem);
            });
        } else {
            const noIssuesElement = document.createElement('div');
            noIssuesElement.textContent = 'No issues found. Your code looks good!';
            resultsList.appendChild(noIssuesElement);
        }
        
        // Update count displays
        errorsCount.textContent = errors;
        warningsCount.textContent = warnings;
        suggestionsCount.textContent = suggestions;
        
        // Show results section
        resultsSection.style.display = 'block';
    }
    
    // Initialize the application
    function initialize() {
        initializeEventListeners();
        
        // Set initial status
        updateStatus('', 'Ready to analyze your code. Select your mood and press "Start Analysis".');
    }
    
    // Start the application
    initialize();
})();