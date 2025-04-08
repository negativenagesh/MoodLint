import * as vscode from 'vscode';
import * as path from 'path';
import { getWebviewContent } from './webviewContent';

// Panel tracking variables
let moodlintPanel: vscode.WebviewPanel | undefined = undefined;

/**
 * Manages MoodLint webview panels
 */
function createMoodlintPanel(context: vscode.ExtensionContext) {
    // If we already have a panel, show it
    if (moodlintPanel) {
        moodlintPanel.reveal(vscode.ViewColumn.One);
        return;
    }

    // Otherwise, create a new panel
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

    // Reset panel variable when panel is closed
    moodlintPanel.onDidDispose(
        () => {
            moodlintPanel = undefined;
        },
        null,
        context.subscriptions
    );

    // Handle messages from the webview
    moodlintPanel.webview.onDidReceiveMessage(
        message => {
            switch (message.command) {
                case 'analyze':
                    vscode.window.showInformationMessage(`Analyzing with mood: ${message.mood}`);
                    // Here you would call your actual analysis logic
                    analyzeWithMood(message.mood, message.options);
                    return;
            }
        },
        undefined,
        context.subscriptions
    );
}

/**
 * Perform the analysis based on the selected mood and options
 */
function analyzeWithMood(mood: string, options: any) {
    // Placeholder for actual analysis logic
    console.log(`Analyzing with mood: ${mood}`);
    console.log('Options:', options);
    
    // In a real implementation, you would:
    // 1. Get the active editor content
    // 2. Run analysis based on mood
    // 3. Return results to the webview
    
    // Example of sending message back to webview (if implemented)
    if (moodlintPanel) {
        setTimeout(() => {
            moodlintPanel?.webview.postMessage({
                command: 'analysisComplete',
                results: {
                    mood: mood,
                    issues: [
                        { line: 10, message: 'This code looks sad', severity: 'info' },
                        { line: 25, message: 'Consider refactoring', severity: 'warning' }
                    ]
                }
            });
        }, 2000); // Simulate processing time
    }
}

// This method is called when your extension is activated
export function activate(context: vscode.ExtensionContext) {
    console.log('Congratulations, your extension "moodlint" is now active!');

    // Register the command to open the MoodLint panel
    const disposable = vscode.commands.registerCommand('moodlint.helloWorld', () => {
        createMoodlintPanel(context);
    });

    context.subscriptions.push(disposable);
}

// This method is called when your extension is deactivated
export function deactivate() {}