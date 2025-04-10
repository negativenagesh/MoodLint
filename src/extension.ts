import * as vscode from 'vscode';
import * as path from 'path';
import { getWebviewContent } from './webviewContent';

// Panel tracking variables
let moodlintPanel: vscode.WebviewPanel | undefined = undefined;

// Mood detection variables
let currentMood: string | null = null;
let moodConfidence: number = 0;

/**
 * Creates and manages the MoodLint webview panel
 */
function createMoodlintPanel(context: vscode.ExtensionContext) {
    // If we already have a panel, show it
    if (moodlintPanel) {
        moodlintPanel.reveal(vscode.ViewColumn.One);
        return;
    }

    // Create a new panel
    moodlintPanel = vscode.window.createWebviewPanel(
        'moodlintPanel',
        'MoodLint',
        vscode.ViewColumn.One,
        {
            enableScripts: true,
            retainContextWhenHidden: true,
            localResourceRoots: [
                vscode.Uri.file(path.join(context.extensionPath, 'media'))
            ]
        }
    );

    // Get paths to media files
    const stylesPath = vscode.Uri.file(
        path.join(context.extensionPath, 'media', 'styles.css')
    );
    const stylesUri = moodlintPanel.webview.asWebviewUri(stylesPath);
    
    const scriptPath = vscode.Uri.file(
        path.join(context.extensionPath, 'media', 'main.js')
    );
    const scriptUri = moodlintPanel.webview.asWebviewUri(scriptPath);

    // Set panel HTML content
    moodlintPanel.webview.html = getWebviewContent(stylesUri, scriptUri);

    // Reset panel when closed
    moodlintPanel.onDidDispose(
        () => {
            moodlintPanel = undefined;
            currentMood = null;
            moodConfidence = 0;
        },
        null,
        context.subscriptions
    );

    // Handle messages from the webview
    moodlintPanel.webview.onDidReceiveMessage(
        message => {
            console.log(`[Extension] Received message: ${JSON.stringify(message)}`);
            switch (message.command) {
                case 'webviewReady':
                    console.log('[Extension] Webview is ready, sending startCamera command');
                    if (moodlintPanel) {
                        moodlintPanel.webview.postMessage({ command: 'startCamera' });
                    }
                    break;
                case 'cameraEnabled':
                    console.log('[Extension] Camera enabled in webview');
                    vscode.window.showInformationMessage('Camera activated successfully');
                    break;
                case 'cameraDisabled':
                    console.log('[Extension] Camera disabled in webview');
                    vscode.window.showInformationMessage('Camera deactivated');
                    break;
                case 'processMood':
                    processMoodFromImage(message.imageData);
                    break;
                case 'analyze':
                    console.log(`[Extension] Analyzing with mood: ${message.mood}`);
                    vscode.window.showInformationMessage(`Analyzing code with mood: ${message.mood}`);
                    analyzeWithMood(message.mood, message.confidence, message.options);
                    break;
            }
        },
        undefined,
        context.subscriptions
    );
    
    console.log('[Extension] MoodLint panel created');
}

/**
 * Process image data from webcam to detect mood (placeholder)
 */
function processMoodFromImage(imageData: string) {
    setTimeout(() => {
        const moods = ['happy', 'focused', 'tired', 'creative', 'stressed'];
        const randomIndex = Math.floor(Math.random() * moods.length);
        const detectedMood = moods[randomIndex];
        const confidence = 0.6 + Math.random() * 0.35;

        currentMood = detectedMood;
        moodConfidence = confidence;

        if (moodlintPanel) {
            moodlintPanel.webview.postMessage({
                command: 'moodDetected',
                mood: detectedMood,
                confidence: confidence
            });
        }
    }, 500);
}

/**
 * Perform code analysis based on mood (placeholder)
 */
function analyzeWithMood(mood: string, confidence: number, options: any) {
    currentMood = mood;
    moodConfidence = confidence;

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

    const text = editor.document.getText();
    const issues = analyzeCodeBasedOnMood(text, mood, options);

    if (moodlintPanel) {
        setTimeout(() => {
            if (moodlintPanel) {
                moodlintPanel.webview.postMessage({
                    command: 'analysisComplete',
                    results: { mood, issues }
                });
            }
        }, 2000);
    }
}

/**
 * Placeholder for code analysis
 */
function analyzeCodeBasedOnMood(code: string, mood: string, options: any): any[] {
    const issues = [];
    switch (mood) {
        case 'happy':
            issues.push({ line: 10, message: 'Great job! Consider adding comments.', severity: 'info' });
            break;
        case 'focused':
            issues.push({ line: 10, message: 'Optimize this function.', severity: 'warning' });
            break;
        case 'tired':
            issues.push({ line: 15, message: 'Potential memory leak.', severity: 'error' });
            break;
        default:
            issues.push({ line: 10, message: 'General suggestion.', severity: 'info' });
    }
    return issues;
}

// Extension activation
export function activate(context: vscode.ExtensionContext) {
    console.log('[Extension] MoodLint extension activated');
    const disposable = vscode.commands.registerCommand('moodlint.helloWorld', () => {
        createMoodlintPanel(context);
    });
    context.subscriptions.push(disposable);
}

// Extension deactivation
export function deactivate() {
    console.log('[Extension] MoodLint extension deactivated');
}