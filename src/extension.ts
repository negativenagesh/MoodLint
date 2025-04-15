import * as vscode from 'vscode';
import * as path from 'path';
import * as childProcess from 'child_process';
import * as fs from 'fs';
import { getWebviewContent } from './webviewContent';
import { AgentInterface } from './agentInterface';
import { createDebugResultsPopup } from './debugPopup';


// Panel tracking variable
let moodlintPanel: vscode.WebviewPanel | undefined = undefined;

// Mood detection variables
let currentMood: string | null = null;
let moodConfidence: number = 0;

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

        const pythonPath = 'python3'; // or 'python' on some systems
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
                                processMoodFromPython(result.mood, result.confidence);
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
            localResourceRoots: [vscode.Uri.file(path.join(context.extensionPath, 'media'))]
        }
    );

    // Now that we have created the panel, we can safely use it
    const stylesPath = vscode.Uri.file(path.join(context.extensionPath, 'media', 'styles.css'));
    const stylesUri = moodlintPanel.webview.asWebviewUri(stylesPath);
    const scriptPath = vscode.Uri.file(path.join(context.extensionPath, 'media', 'main.js'));
    const scriptUri = moodlintPanel.webview.asWebviewUri(scriptPath);
    
    // Generate a nonce for content security policy
    const nonce = getNonce();
    const cspSource = moodlintPanel.webview.cspSource;

    moodlintPanel.webview.html = getWebviewContent(stylesUri, scriptUri, nonce, cspSource);

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
                default:
                    console.log('[Extension] Unknown command:', message.command);
            }
        },
        undefined,
        context.subscriptions
    );

    console.log('[Extension] MoodLint panel creation complete');
}

/**
 * Process mood data from Python
 */
function processMoodFromPython(mood: string, confidence: number) {
    currentMood = mood;
    moodConfidence = confidence;

    if (moodlintPanel) {
        console.log('[Extension] Sending moodDetected from Python:', mood);
        moodlintPanel.webview.postMessage({
            command: 'moodDetected',
            mood: mood,
            confidence: confidence
        });
    } else {
        console.error('[Extension] moodlintPanel undefined, cannot send moodDetected');
    }
}

/**
 * Analyze code based on mood using the agent system
 */

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

    // Get the active editor
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showWarningMessage('No active editor found');
        if (moodlintPanel) {
            moodlintPanel.webview.postMessage({
                command: 'analysisComplete',
                results: { mood, issues: [] }
            });
        }
        return;
    }

    // Show loading state in the panel
    if (moodlintPanel) {
        moodlintPanel.webview.postMessage({
            command: 'analysisStarted',
            mood: mood
        });
    }

    try {
        // Get the document text and filename
        const text = editor.document.getText();
        const filename = path.basename(editor.document.uri.fsPath);
        
        // Create a temporary file to hold the code
        const tempDir = path.join(context.extensionPath, 'temp');
        if (!fs.existsSync(tempDir)) {
            fs.mkdirSync(tempDir, { recursive: true });
        }
        
        const tempCodeFile = path.join(tempDir, `debug_code_${Date.now()}.txt`);
        fs.writeFileSync(tempCodeFile, text);
        
        // Launch the Python agent popup
        const pythonPath = 'python3'; // or 'python' on some systems
        const scriptPath = path.join(context.extensionPath, 'popup', 'agent_popup.py');
        
        console.log(`[Extension] Starting agent popup: ${scriptPath}`);
        
        // Make sure the script is executable
        try {
            fs.chmodSync(scriptPath, 0o755);
        } catch (err) {
            // Ignore chmod errors on Windows
        }
        
        // Launch the agent popup process
        const agentProcess = childProcess.spawn(pythonPath, [
            scriptPath, 
            tempCodeFile, 
            filename, 
            mood,
            options.query || ''
        ], {
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
                                
                                // Also create the VS Code webview popup
                                createDebugResultsPopup(
                                    context,
                                    mood,
                                    agentResult.response
                                );
                                
                                // Send the results back to the main webview
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
            
            // Clean up the temporary file
            try {
                fs.unlinkSync(tempCodeFile);
            } catch (err) {
                console.error('[Extension] Error cleaning up temp file:', err);
            }
            
            // If the process failed to launch or exited with an error
            if (code !== 0) {
                // Fall back to the regular agent interface
                console.log('[Extension] Falling back to regular agent interface');
                
                // Check if agent interface is initialized
                if (!agentInterface) {
                    vscode.window.showErrorMessage('Agent system not initialized properly');
                    return;
                }
                
                // Use agent workflow to debug
                agentInterface.debugCode(text, filename, mood)
                    .then(result => {
                        if (result.success && result.response) {
                            // Convert the agent response to an issue list for display
                            const issues = extractIssuesFromResponse(result.response, mood);
                            
                            // Create a popup with the debugging results
                            createDebugResultsPopup(
                                context,
                                mood,
                                result.response
                            );
                            
                            // Also send the results back to the webview
                            if (moodlintPanel) {
                                moodlintPanel.webview.postMessage({
                                    command: 'analysisComplete',
                                    results: { 
                                        mood,
                                        issues,
                                        fullResponse: result.response
                                    }
                                });
                            }
                        } else {
                            // Handle error case
                            vscode.window.showErrorMessage(`Debug failed: ${result.error || 'Unknown error'}`);
                            if (moodlintPanel) {
                                moodlintPanel.webview.postMessage({
                                    command: 'analysisComplete',
                                    results: { 
                                        mood, 
                                        issues: [], 
                                        error: result.error 
                                    }
                                });
                            }
                        }
                    })
                    .catch(error => {
                        console.error('[Extension] Error during fallback analysis:', error);
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
                    });
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
    
    const disposable = vscode.commands.registerCommand('moodlint.helloWorld', () => {
        createMoodlintPanel(context);
    });
    context.subscriptions.push(disposable);
}