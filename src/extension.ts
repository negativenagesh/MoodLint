import * as vscode from 'vscode';
import * as path from 'path';
import { getWebviewContent } from './webviewContent';

// Panel tracking variable
let moodlintPanel: vscode.WebviewPanel | undefined = undefined;

// Mood detection variables
let currentMood: string | null = null;
let moodConfidence: number = 0;

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
                    console.log('[Extension] Webview ready, sending startCamera');
                    moodlintPanel.webview.postMessage({ command: 'startCamera' });
                    break;
                case 'cameraEnabled':
                    console.log('[Extension] Camera enabled');
                    vscode.window.showInformationMessage('Camera activated successfully');
                    break;
                case 'cameraDisabled':
                    console.log('[Extension] Camera disabled');
                    vscode.window.showInformationMessage('Camera deactivated');
                    break;
                case 'processMood':
                    processMoodFromImage(message.imageData);
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
 * Process image data for mood detection (placeholder)
 */
function processMoodFromImage(imageData: string) {
    setTimeout(() => {
        const moods = ['happy', 'focused', 'tired', 'creative', 'stressed'];
        const detectedMood = moods[Math.floor(Math.random() * moods.length)];
        const confidence = 0.6 + Math.random() * 0.35;

        currentMood = detectedMood;
        moodConfidence = confidence;

        if (moodlintPanel) {
            console.log('[Extension] Sending moodDetected:', detectedMood);
            moodlintPanel.webview.postMessage({
                command: 'moodDetected',
                mood: detectedMood,
                confidence: confidence
            });
        } else {
            console.error('[Extension] moodlintPanel undefined, cannot send moodDetected');
        }
    }, 500);
}

/**
 * Analyze code based on mood (placeholder)
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
            moodlintPanel?.webview.postMessage({    
                command: 'analysisComplete',
                results: { mood, issues }
            });
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
            issues.push({ line: 10, message: 'Great job! Add comments?', severity: 'info' });
            break;
        case 'focused':
            issues.push({ line: 10, message: 'Optimize this.', severity: 'warning' });
            break;
        case 'tired':
            issues.push({ line: 15, message: 'Potential leak.', severity: 'error' });
            break;
        default:
            issues.push({ line: 10, message: 'General tip.', severity: 'info' });
    }
    return issues;
}

export function activate(context: vscode.ExtensionContext) {
    console.log('[Extension] MoodLint activated');
    const disposable = vscode.commands.registerCommand('moodlint.helloWorld', () => {
        createMoodlintPanel(context);
    });
    context.subscriptions.push(disposable);
}

export function deactivate() {
    console.log('[Extension] MoodLint deactivated');
}