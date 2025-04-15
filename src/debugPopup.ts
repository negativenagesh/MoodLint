import * as vscode from 'vscode';
import * as path from 'path';

/**
 * Creates and manages the popup that displays agent debugging results
 */
export function createDebugResultsPopup(
    context: vscode.ExtensionContext,
    mood: string,
    response: string
): vscode.WebviewPanel {
    // Create the webview panel
    const panel = vscode.window.createWebviewPanel(
        'moodlintResults',
        `MoodLint Debug: ${mood.charAt(0).toUpperCase() + mood.slice(1)}`,
        vscode.ViewColumn.Two,
        {
            enableScripts: true,
            retainContextWhenHidden: true,
            localResourceRoots: [vscode.Uri.file(path.join(context.extensionPath, 'media'))]
        }
    );

    // Get path to style and script resources
    const stylesPath = vscode.Uri.file(path.join(context.extensionPath, 'media', 'styles.css'));
    const stylesUri = panel.webview.asWebviewUri(stylesPath);
    const nonce = getNonce();
    const cspSource = panel.webview.cspSource;

    // Process response to highlight code blocks
    const processedResponse = processMarkdown(response);

    // Set the HTML content of the webview
    panel.webview.html = `<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src ${cspSource} blob: data:; style-src ${cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}'; media-src mediastream:;">
        <title>MoodLint Debug Results</title>
        <link rel="stylesheet" href="${stylesUri}">
        <style nonce="${nonce}">
            .agent-container {
                padding: 20px;
                max-width: 800px;
                margin: 0 auto;
            }
            .agent-header {
                display: flex;
                align-items: center;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 1px solid var(--border-color);
            }
            .agent-icon {
                font-size: 24px;
                margin-right: 10px;
            }
            .agent-title {
                margin: 0;
            }
            .agent-mood {
                margin-left: auto;
                font-weight: bold;
                color: var(--primary-color);
            }
            .agent-response {
                background-color: var(--section-bg-color);
                padding: 15px;
                border-radius: 8px;
                overflow-x: auto;
                line-height: 1.5;
            }
            .code-block {
                background-color: rgba(0,0,0,0.1);
                border-radius: 4px;
                padding: 10px;
                margin: 10px 0;
                white-space: pre-wrap;
                font-family: monospace;
            }
            .mood-badge {
                display: inline-block;
                padding: 5px 10px;
                border-radius: 15px;
                font-size: 14px;
                margin-left: 10px;
            }
            .happy { background-color: #4CAF50; color: white; }
            .frustrated { background-color: #FF9800; color: white; }
            .exhausted { background-color: #9C27B0; color: white; }
            .sad { background-color: #2196F3; color: white; }
            .angry { background-color: #F44336; color: white; }
        </style>
    </head>
    <body>
        <div class="agent-container">
            <div class="agent-header">
                <div class="agent-icon">🤖</div>
                <h2 class="agent-title">MoodLint Agent Response</h2>
                <div class="agent-mood">
                    Current Mood: <span class="mood-badge ${mood}">${mood}</span>
                </div>
            </div>
            <div class="agent-response">${processedResponse}</div>
        </div>
        <script nonce="${nonce}">
            // Add any needed JavaScript here
            document.addEventListener('DOMContentLoaded', () => {
                console.log('Debug results popup loaded');
            });
        </script>
    </body>
    </html>`;

    return panel;
}

/**
 * Generate a random nonce string
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
 * Process markdown text to highlight code blocks
 */
function processMarkdown(text: string): string {
    // Simple markdown processor to handle code blocks
    let processed = text;
    
    // Handle code blocks (```code```)
    processed = processed.replace(/```([\s\S]*?)```/g, '<div class="code-block">$1</div>');
    
    // Handle inline code (`code`)
    processed = processed.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Handle line breaks
    processed = processed.replace(/\n/g, '<br>');
    
    return processed;
}