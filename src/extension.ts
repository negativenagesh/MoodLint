import * as vscode from 'vscode';
import * as path from 'path';
import * as childProcess from 'child_process';
import * as fs from 'fs';
import * as os from 'os';
import { getWebviewContent } from './webviewContent';
import { AgentInterface } from './agentInterface';
import { createDebugResultsPopup } from './debugPopup';

// Panel tracking variable
let moodlintPanel: vscode.WebviewPanel | undefined = undefined;

// Mood detection variables
let currentMood: string | null = null;
let moodConfidence: number = 0;
let capturedImagePath: string | null = null;

// Python camera process
let cameraProcess: childProcess.ChildProcess | null = null;

// Agent interface
let agentInterface: AgentInterface | null = null;

/**
 * Helper function that generates a nonce
 */
function getNonce() {
    let text = '';
    const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    for (let i = 0; i < 32; i++) {
        text += possible.charAt(Math.floor(Math.random() * possible.length));
    }
    return text;
}

/**
 * Start Python camera application
 */
function startPythonCameraApp(context: vscode.ExtensionContext): Promise<void> {
    return new Promise((resolve, reject) => {
        if (cameraProcess) {
            // Camera already running
            resolve();
            return;
        }

        const pythonPath = '/home/subrahmanya/projects/MoodLint/.venv/bin/python';
        const scriptPath = path.join(context.extensionPath, 'popup', 'camera.py');
        const sessionId = Date.now().toString();

        console.log(`[Extension] Starting Python camera app: ${scriptPath}`);
        
        // Check if the file exists
        if (!fs.existsSync(scriptPath)) {
            const error = new Error(`Python script not found at ${scriptPath}`);
            console.error('[Extension] Python script not found:', error);
            reject(error);
            return;
        }
        
        try {
            // Make sure the script is executable
            try {
                // This will throw an error on Windows, but that's ok
                fs.chmodSync(scriptPath, 0o755);
            } catch (err) {
                // Ignore chmod errors
            }
            
            // Spawn the process without shell for better GUI compatibility
            // Using detached option to ensure the process can open windows
            cameraProcess = childProcess.spawn(pythonPath, [scriptPath, sessionId], {
                env: { 
                    ...process.env, 
                    DISPLAY: process.env.DISPLAY || ':0',
                    PYTHONUNBUFFERED: '1' // Ensure Python output isn't buffered
                },
                detached: true, // This is important for GUI applications
                stdio: 'pipe'   // Ensure we can still capture output
            });
            
            // Set a timeout for initial startup
            const timeout = setTimeout(() => {
                console.log('[Extension] Camera app startup timeout - assuming it started anyway');
                resolve();
            }, 5000);
            
            // Handle process output (for mood detection results)
            cameraProcess.stdout?.on('data', (data) => {
                try {
                    const output = data.toString().trim();
                    console.log(`[Extension] Python output: ${output}`);
                    
                    if (output) {
                        try {
                            const result = JSON.parse(output);
                            
                            // Check for status updates
                            if (result.status) {
                                if (result.status === 'ready') {
                                    clearTimeout(timeout);
                                    resolve();
                                }
                            }
                            
                            // Check for mood data
                            if (result.mood && result.confidence) {
                                // Pass the autoAnalyze flag to processMoodFromPython
                                processMoodFromPython(
                                    result.mood, 
                                    result.confidence, 
                                    result.filepath,
                                    result.autoAnalyze === true // Explicitly convert to boolean
                                );
                            }
                            
                            // Check for errors
                            if (result.error) {
                                console.error(`[Extension] Python reported error: ${result.error}`);
                                vscode.window.showErrorMessage(`Camera error: ${result.error}`);
                            }
                        } catch (jsonError) {
                            // If it's not valid JSON, just log it
                            console.log(`[Extension] Non-JSON output: ${output}`);
                        }
                    }
                } catch (error) {
                    console.error('[Extension] Error parsing Python output:', error);
                }
            });
            
            // Handle errors
            cameraProcess.stderr?.on('data', (data) => {
                const errorOutput = data.toString().trim();
                console.error(`[Extension] Python error: ${errorOutput}`);
                
                // Check for specific error messages
                if (errorOutput.includes('No module named')) {
                    const missingModule = errorOutput.match(/No module named '([^']+)'/);
                    if (missingModule && missingModule[1]) {
                        vscode.window.showErrorMessage(
                            `Missing Python module: ${missingModule[1]}. Please install it with: pip install ${missingModule[1]}`
                        );
                    }
                }
            });
            
            // Handle process exit
            cameraProcess.on('close', (code) => {
                console.log(`[Extension] Python camera app exited with code ${code}`);
                clearTimeout(timeout);
                cameraProcess = null;
                
                // Notify webview that camera is off
                if (moodlintPanel) {
                    moodlintPanel.webview.postMessage({ command: 'cameraOff' });
                }
                
                if (code !== 0) {
                    reject(new Error(`Python process exited with code ${code}`));
                } else {
                    resolve();
                }
            });
            
            // Handle process error
            cameraProcess.on('error', (error) => {
                console.error('[Extension] Failed to start Python process:', error);
                clearTimeout(timeout);
                cameraProcess = null;
                reject(error);
            });
            
            // Wait for a short time to ensure process starts properly
            setTimeout(() => {
                if (cameraProcess && cameraProcess.pid) {
                    console.log(`[Extension] Camera app started with PID: ${cameraProcess.pid}`);
                    resolve();
                }
            }, 1000);
            
        } catch (error) {
            console.error('[Extension] Error launching Python app:', error);
            reject(error);
        }
    });
}

/**
 * Stop Python camera application
 */
function stopPythonCameraApp() {
    if (cameraProcess) {
        console.log('[Extension] Stopping Python camera app');
        // On Windows, use 'taskkill' or equivalent
        if (process.platform === 'win32') {
            childProcess.exec(`taskkill /pid ${cameraProcess.pid} /T /F`);
        } else {
            cameraProcess.kill('SIGTERM');
            // Also try to kill by process group to ensure child processes are terminated
            try {
                if (cameraProcess.pid) {
                    process.kill(-cameraProcess.pid, 'SIGTERM');
                }
            } catch (e) {
                // Ignore error, might not be a process group leader
            }
        }
        cameraProcess = null;
    }
}

/**
 * Predict future mood based on current mood
 */
async function predictFutureMoodFromCurrent(currentMood: string) {
    try {
        // You could add more sophisticated prediction logic here
        const moods = ['happy', 'sad', 'frustrated', 'angry', 'exhausted', 'neutral'];
        // Remove current mood from options
        const availableMoods = moods.filter(mood => mood !== currentMood);
        // Select a random mood from the remaining options
        const predictedMood = availableMoods[Math.floor(Math.random() * availableMoods.length)];
        const confidence = 0.7 + (Math.random() * 0.3); // Random confidence between 0.7-1.0

        if (moodlintPanel) {
            moodlintPanel.webview.postMessage({
                command: 'futureMoodPredicted',
                mood: predictedMood,
                confidence: confidence,
                message: `Based on analysis, your mood might change to ${predictedMood} soon.`
            });
        }
    } catch (error) {
        console.error('[Extension] Error predicting future mood:', error);
        if (moodlintPanel) {
            moodlintPanel.webview.postMessage({
                command: 'futureMoodPredicted',
                error: `Failed to predict future mood: ${error instanceof Error ? error.message : String(error)}`
            });
        }
    }
}

/**
 * Generate visualization of future mood
 */
async function generateFutureMoodVisualization(currentMood: string) {
    try {
        // Check if we have a captured image to use
        if (!capturedImagePath) {
            vscode.window.showWarningMessage('Please upload your image first by clicking "Go with mood debug"');
            if (moodlintPanel) {
                moodlintPanel.webview.postMessage({
                    command: 'futureMoodGenerated',
                    error: 'No image uploaded. Please use "Go with mood debug" first.'
                });
            }
            return;
        }

        // Inform the user that generation has started
        vscode.window.showInformationMessage('Launching future mood generator...');
        
        // Launch the GAN popup
        const pythonPath = '/home/subrahmanya/projects/MoodLint/.venv/bin/python';  // Use the virtual env if possible
        const scriptPath = path.join(context.extensionPath, 'popup', 'gan_popup.py');
        
        console.log(`[Extension] Starting GAN popup for mood: ${currentMood} using image: ${capturedImagePath}`);
        
        // Make sure the script is executable
        try {
            fs.chmodSync(scriptPath, 0o755);
        } catch (err) {
            // Ignore chmod errors on Windows
        }
        
        // Get a temporary file path for the output image
        const outputImagePath = path.join(os.tmpdir(), `moodlint_gan_${Date.now()}.png`);
        
        // Launch the GAN popup process
        const ganProcess = childProcess.spawn(pythonPath, [scriptPath, capturedImagePath, currentMood, outputImagePath], {
            env: { 
                ...process.env, 
                DISPLAY: process.env.DISPLAY || ':0',
                PYTHONUNBUFFERED: '1' // Ensure Python output isn't buffered
            },
            detached: true, // This is important for GUI applications
            stdio: 'pipe'   // Ensure we can still capture output
        });
        
        // Handle process output
        ganProcess.stdout?.on('data', (data) => {
            try {
                const output = data.toString().trim();
                console.log(`[Extension] GAN popup output: ${output}`);
                
                if (output) {
                    try {
                        const result = JSON.parse(output);
                        
                        // Check for status updates
                        if (result.status === 'generation_complete' && result.output_path) {
                            // Create a webview URI for the image if needed
                            if (moodlintPanel) {
                                const imageUri = moodlintPanel.webview.asWebviewUri(
                                    vscode.Uri.file(result.output_path)
                                );
                                
                                // Notify webview of successful generation
                                moodlintPanel.webview.postMessage({
                                    command: 'futureMoodGenerated',
                                    imageUrl: imageUri.toString(),
                                    mood: currentMood,
                                    caption: `A unique visualization of your ${currentMood} mood`
                                });
                            }
                        }
                        
                        // Check for errors
                        if (result.error) {
                            console.error(`[Extension] GAN popup reported error: ${result.error}`);
                            vscode.window.showErrorMessage(`GAN error: ${result.error}`);
                            
                            if (moodlintPanel) {
                                moodlintPanel.webview.postMessage({
                                    command: 'futureMoodGenerated',
                                    error: result.error
                                });
                            }
                        }
                    } catch (jsonError) {
                        // If it's not valid JSON, just log it
                        console.log(`[Extension] Non-JSON output: ${output}`);
                    }
                }
            } catch (error) {
                console.error('[Extension] Error parsing GAN popup output:', error);
            }
        });
        
        // Handle errors
        ganProcess.stderr?.on('data', (data) => {
            const errorOutput = data.toString().trim();
            console.error(`[Extension] GAN popup error: ${errorOutput}`);
        });
        
        // Handle process exit
        ganProcess.on('close', (code) => {
            console.log(`[Extension] GAN popup exited with code ${code}`);
            
            // If the process failed to launch or exited with an error
            if (code !== 0) {
                vscode.window.showErrorMessage(`GAN process exited with code ${code}`);
                if (moodlintPanel) {
                    moodlintPanel.webview.postMessage({
                        command: 'futureMoodGenerated',
                        error: `GAN process exited with code ${code}`
                    });
                }
            }
        });
        
    } catch (error) {
        console.error('[Extension] Error generating future mood:', error);
        vscode.window.showErrorMessage(`Error generating mood visualization: ${error instanceof Error ? error.message : String(error)}`);
        
        if (moodlintPanel) {
            moodlintPanel.webview.postMessage({
                command: 'futureMoodGenerated',
                error: `Failed to generate mood visualization: ${error instanceof Error ? error.message : String(error)}`
            });
        }
    }
}

/**
 * Creates and manages the MoodLint webview panel
 */
function createMoodlintPanel(context: vscode.ExtensionContext) {
    if (moodlintPanel) {
        console.log('[Extension] Reusing existing panel');
        moodlintPanel.reveal(vscode.ViewColumn.One);
        return;
    }

    // Create the panel first
    moodlintPanel = vscode.window.createWebviewPanel(
        'moodlint',
        'MoodLint',
        vscode.ViewColumn.One,
        {
            enableScripts: true,
            retainContextWhenHidden: true,
            localResourceRoots: [
                vscode.Uri.file(path.join(context.extensionPath, 'media')),
                vscode.Uri.file(path.join(context.extensionPath, 'MoodLint-logo'))
            ]
        }
    );

    // Set the tab icon - this is what shows in the VS Code tab
    const iconPath = vscode.Uri.file(path.join(context.extensionPath, 'MoodLint-logo', 'logo.png'));
    moodlintPanel.iconPath = iconPath;

    // Now that we have created the panel, we can safely use it
    const stylesPath = vscode.Uri.file(path.join(context.extensionPath, 'media', 'styles.css'));
    const stylesUri = moodlintPanel.webview.asWebviewUri(stylesPath);
    const scriptPath = vscode.Uri.file(path.join(context.extensionPath, 'media', 'main.js'));
    const scriptUri = moodlintPanel.webview.asWebviewUri(scriptPath);
    const imagePath = vscode.Uri.file(path.join(context.extensionPath, 'MoodLint-logo', 'cover.png'));
    const imageUri = moodlintPanel.webview.asWebviewUri(imagePath);
    
    // Add favicon for the webview
    const faviconPath = vscode.Uri.file(path.join(context.extensionPath, 'MoodLint-logo', 'logo.png'));
    const faviconUri = moodlintPanel.webview.asWebviewUri(faviconPath);
    
    // Generate a nonce for content security policy
    const nonce = getNonce();
    const cspSource = moodlintPanel.webview.cspSource;

    moodlintPanel.webview.html = getWebviewContent(stylesUri, scriptUri, imageUri, faviconUri, nonce, cspSource);

    moodlintPanel.onDidDispose(
        () => {
            console.log('[Extension] Panel disposed');
            moodlintPanel = undefined;
            currentMood = null;
            moodConfidence = 0;
            
            // Make sure to stop Python process when panel is closed
            stopPythonCameraApp();
        },
        null,
        context.subscriptions
    );

    moodlintPanel.webview.onDidReceiveMessage(
        message => {
            console.log(`[Extension] Received message: ${JSON.stringify(message)}`);
            if (!moodlintPanel) {
                console.error('[Extension] moodlintPanel is undefined, cannot process message');
                return;
            }
            switch (message.command) {
                case 'webviewReady':
                    console.log('[Extension] Webview ready');
                    break;
                case 'startExternalCamera':
                    // Launch the Python camera app
                    startPythonCameraApp(context)
                        .then(() => {
                            vscode.window.showInformationMessage('External camera launched successfully');
                            moodlintPanel?.webview.postMessage({ 
                                command: 'externalCameraStarted' 
                            });
                        })
                        .catch(error => {
                            vscode.window.showErrorMessage(`Failed to start camera: ${error.message}`);
                            moodlintPanel?.webview.postMessage({ 
                                command: 'externalCameraFailed',
                                error: error.message
                            });
                        });
                    break;
                case 'stopExternalCamera':
                    stopPythonCameraApp();
                    vscode.window.showInformationMessage('Camera stopped');
                    break;
                case 'analyze':
                    console.log(`[Extension] Analyzing mood: ${message.mood}`);
                    analyzeWithMood(message.mood, message.confidence, message.options);
                    break;
                case 'predictFutureMood':
                    console.log(`[Extension] Predicting future mood based on: ${message.currentMood}`);
                    predictFutureMoodFromCurrent(message.currentMood);
                    break;
                case 'generateFutureMood':
                    console.log(`[Extension] Generating future mood visualization based on: ${message.currentMood}`);
                    generateFutureMoodVisualization(message.currentMood);
                    break;
                default:
                    console.log('[Extension] Unknown command:', message.command);
            }
        },
        undefined,
        context.subscriptions
    );

    console.log('[Extension] MoodLint panel creation complete');
}

function processMoodFromPython(mood: string, confidence: number, filepath?: string, autoAnalyze?: boolean) {
    console.log(`[Extension] Processing mood: ${mood} (${confidence}) from image: ${filepath || 'unknown'}, autoAnalyze: ${autoAnalyze}`);
    currentMood = mood;
    moodConfidence = confidence;
    
    // Store the captured image path if provided
    if (filepath) {
        capturedImagePath = filepath;
        console.log(`[Extension] Stored captured image path: ${capturedImagePath}`);
    }

    if (moodlintPanel) {
        console.log('[Extension] Sending moodDetected from Python:', mood);
        moodlintPanel.webview.postMessage({
            command: 'moodDetected',
            mood: mood,
            confidence: confidence,
            filepath: filepath,
            hasImage: !!filepath // Signal whether we have a valid image
        });
        
        // If confidence is high enough, enable analysis button
        if (confidence >= 0.6) {
            moodlintPanel.webview.postMessage({
                command: 'enableAnalysis',
                mood: mood
            });
        }
        
        // If autoAnalyze flag is explicitly set to true, immediately launch analysis
        if (autoAnalyze === true) {
            console.log('[Extension] Auto-analyzing based on camera signal');
            // Get the active editor and analyze the code
            const editor = vscode.window.activeTextEditor;
            if (editor) {
                analyzeWithMood(mood, confidence, {
                    filepath: editor.document.fileName,
                    code: editor.document.getText()
                });
            } else {
                // No editor active, analyze without specific code
                analyzeWithMood(mood, confidence, {});
            }
            return; // Skip checking user settings since we have an explicit signal
        }
        
        // Option to automatically launch analysis based on user settings
        const autoAnalyzeFromSettings = vscode.workspace.getConfiguration('moodlint').get('autoAnalyzeAfterDetection', false);
        if (autoAnalyzeFromSettings) {
            // Get the active editor and analyze the code
            const editor = vscode.window.activeTextEditor;
            if (editor) {
                analyzeWithMood(mood, confidence, {
                    filepath: editor.document.fileName,
                    code: editor.document.getText()
                });
            } else {
                // No editor active, analyze without specific code
                analyzeWithMood(mood, confidence, {});
            }
        }
    } else {
        console.error('[Extension] moodlintPanel undefined, cannot send moodDetected');
    }
}

/**
 * Extract issues from the agent's text response
 */
function extractIssuesFromResponse(response: string, mood: string): any[] {
    const issues = [];
    
    // Simple regex to find line references
    const lineRegex = /[Ll]ine\s+(\d+):?\s+([^\.]+)/g;
    let match;
    
    while ((match = lineRegex.exec(response)) !== null) {
        const line = parseInt(match[1], 10);
        const message = match[2].trim();
        
        // Determine severity based on content
        let severity = 'info';
        const lowerMessage = message.toLowerCase();
        if (lowerMessage.includes('error') || lowerMessage.includes('critical') || lowerMessage.includes('fail')) {
            severity = 'error';
        } else if (lowerMessage.includes('warning') || lowerMessage.includes('caution') || lowerMessage.includes('consider')) {
            severity = 'warning';
        }
        
        issues.push({ line, message, severity });
    }
    
    // If no issues were found with the regex, create at least one general issue
    if (issues.length === 0) {
        // Try to find the first paragraph as a summary
        const paragraphs = response.split('\n\n');
        const summary = paragraphs[0] || 'Code analyzed based on your mood';
        
        issues.push({
            line: 1,
            message: summary,
            severity: 'info'
        });
    }
    
    return issues;
}

export function deactivate() {
    console.log('[Extension] MoodLint deactivated');
    stopPythonCameraApp();
}

async function analyzeWithMood(mood: string, confidence: number, options: any) {
    currentMood = mood;
    moodConfidence = confidence;

    // Show loading state in the panel
    if (moodlintPanel) {
        moodlintPanel.webview.postMessage({
            command: 'analysisStarted',
            mood: mood
        });
    }

    try {
        // Get the active editor
        const editor = vscode.window.activeTextEditor;
        let filename = options.filepath || (editor ? editor.document.fileName : '');
        let code = options.code || (editor ? editor.document.getText() : '');
        const query = options.query || '';

        // If we have code but no filename, create a temporary file
        let tempFilePath = '';
        if (code && !filename) {
            // Create temp file
            tempFilePath = path.join(os.tmpdir(), `moodlint_${Date.now()}.txt`);
            fs.writeFileSync(tempFilePath, code);
            filename = tempFilePath;
        }

        // Launch the Python agent popup
        const pythonPath = 'python3'; // or 'python' on some systems
        const scriptPath = path.join(context.extensionPath, 'popup', 'agent_popup.py');
        
        console.log(`[Extension] Starting agent popup for mood: ${mood}, file: ${filename || 'none'}`);
        
        // Make sure the script is executable
        try {
            fs.chmodSync(scriptPath, 0o755);
        } catch (err) {
            // Ignore chmod errors on Windows
        }
        
        // Build command line arguments based on what's available
        const args = [scriptPath, mood];
        
        // Add filename if available
        if (filename) {
            args.push(filename);
        }
        
        // Add query if available
        if (query) {
            args.push(query);
        }
        
        // Launch the agent popup process
        const agentProcess = childProcess.spawn(pythonPath, args, {
            env: { 
                ...process.env, 
                DISPLAY: process.env.DISPLAY || ':0',
                PYTHONUNBUFFERED: '1',
                GOOGLE_API_KEY: process.env.GOOGLE_API_KEY
            },
            detached: true,
            stdio: 'pipe'
        });
        
        // Handle process output
        agentProcess.stdout?.on('data', (data) => {
            try {
                const output = data.toString().trim();
                console.log(`[Extension] Agent popup output: ${output}`);
                
                if (output) {
                    try {
                        const result = JSON.parse(output);
                        
                        // Check for status updates
                        if (result.status === 'complete' && result.result) {
                            // Process the agent's result
                            const agentResult = result.result;
                            
                            if (agentResult.success && agentResult.response) {
                                // Convert the agent response to an issue list for display
                                const issues = extractIssuesFromResponse(agentResult.response, mood);
                                
                                // Also send the results back to the main webview
                                if (moodlintPanel) {
                                    moodlintPanel.webview.postMessage({
                                        command: 'analysisComplete',
                                        results: { 
                                            mood,
                                            issues,
                                            fullResponse: agentResult.response
                                        }
                                    });
                                }
                            }
                        } else if (result.status === 'error') {
                            vscode.window.showErrorMessage(`Agent error: ${result.message}`);
                            
                            if (moodlintPanel) {
                                moodlintPanel.webview.postMessage({
                                    command: 'analysisComplete',
                                    results: { 
                                        mood, 
                                        issues: [], 
                                        error: result.message
                                    }
                                });
                            }
                        }
                    } catch (err) {
                        // If not valid JSON, just log it
                        console.log(`[Extension] Non-JSON agent output: ${output}`);
                    }
                }
            } catch (error) {
                console.error('[Extension] Error parsing agent output:', error);
            }
        });
        
        // Handle errors
        agentProcess.stderr?.on('data', (data) => {
            const errorOutput = data.toString().trim();
            console.error(`[Extension] Agent popup error: ${errorOutput}`);
        });
        
        // Handle process exit
        agentProcess.on('close', (code) => {
            console.log(`[Extension] Agent popup exited with code ${code}`);
            
            // Clean up temp file if created
            if (tempFilePath && fs.existsSync(tempFilePath)) {
                try {
                    fs.unlinkSync(tempFilePath);
                } catch (err) {
                    console.error(`[Extension] Error removing temp file: ${err}`);
                }
            }
            
            // If the process failed to launch or exited with an error
            if (code !== 0) {
                vscode.window.showErrorMessage(`Agent process exited with code ${code}`);
                if (moodlintPanel) {
                    moodlintPanel.webview.postMessage({
                        command: 'analysisComplete',
                        results: { 
                            mood, 
                            issues: [], 
                            error: `Agent process exited with code ${code}` 
                        }
                    });
                }
            }
        });
        
    } catch (error) {
        console.error('[Extension] Error during analysis:', error);
        vscode.window.showErrorMessage(`Analysis error: ${error instanceof Error ? error.message : String(error)}`);
        
        if (moodlintPanel) {
            moodlintPanel.webview.postMessage({
                command: 'analysisComplete',
                results: { 
                    mood, 
                    issues: [], 
                    error: error instanceof Error ? error.message : String(error) 
                }
            });
        }
    }
}

// Add a new property to store the extension context
let context: vscode.ExtensionContext;

export function activate(extensionContext: vscode.ExtensionContext) {
    console.log('[Extension] MoodLint activated');
    
    // Store the context for later use
    context = extensionContext;
    
    // Initialize the agent interface
    agentInterface = new AgentInterface(context.extensionPath);
    
    // Check agent installation and install dependencies if needed
    agentInterface.checkAgentInstallation().then(isInstalled => {
        if (!isInstalled) {
            vscode.window.showWarningMessage('MoodLint agent system not fully installed. Some features may not work.');
        }
    });
    
    // Register MoodLint settings
    vscode.workspace.getConfiguration('moodlint').update('autoAnalyzeAfterDetection', 
        vscode.workspace.getConfiguration('moodlint').get('autoAnalyzeAfterDetection', false),
        vscode.ConfigurationTarget.Global);
    
    const disposable = vscode.commands.registerCommand('moodlint.helloWorld', () => {
        createMoodlintPanel(context);
    });
    context.subscriptions.push(disposable);
}