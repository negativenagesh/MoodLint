import * as vscode from 'vscode';

export function getWebviewContent(stylesUri: vscode.Uri, scriptUri: vscode.Uri): string {
    return `<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>MoodLint</title>
        <link rel="stylesheet" href="${stylesUri}">
    </head>
    <body>
        <div class="app-container">
            <header class="app-header">
                <h1>MoodLint</h1>
                <div class="header-subtitle">Emotion-aware code debugging assistant</div>
            </header>
            <main class="app-content">
                <section class="mood-detector-section">
                    <h2>Mood Detection</h2>
                    <p>MoodLint analyzes your facial expressions to detect your emotional state.</p>
                    <div class="camera-container">
                        <video id="camera-feed" autoplay playsinline style="display: none;"></video>
                        <canvas id="debug-canvas" style="display: none;"></canvas>
                        <div id="camera-placeholder" class="camera-placeholder">
                            <div class="camera-icon">📷</div>
                            <p>Camera will appear here</p>
                        </div>
                    </div>
                    <div class="mood-status">
                        <div class="detected-mood">
                            <span>Detected Mood: </span>
                            <span id="current-mood-display">Not detected yet</span>
                        </div>
                        <div class="mood-confidence" style="display: none;">
                            <div class="confidence">
                                <div class="confidence-label">Confidence:</div>
                                <div class="confidence-bar-container">
                                    <div id="confidence-bar" class="confidence-bar" style="width: 0%"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="camera-actions">
                        <button id="enable-camera-btn" class="primary-button">Go with mood debug</button>
                    </div>
                </section>
                <section class="settings-section">
                    <h2>Analysis Settings</h2>
                    <div class="settings-container">
                        <div class="settings-column">
                            <input type="checkbox" id="check-syntax" checked><label for="check-syntax">Syntax issues</label>
                            <input type="checkbox" id="check-style" checked><label for="check-style">Style suggestions</label>
                        </div>
                        <div class="settings-column">
                            <input type="checkbox" id="check-performance"><label for="check-performance">Performance</label>
                            <input type="checkbox" id="check-best-practices"><label for="check-best-practices">Best practices</label>
                        </div>
                    </div>
                </section>
                <section class="actions-section">
                    <button id="analyze-btn" class="primary-button" disabled>Start Debugging</button>
                </section>
                <section class="status-section">
                    <div class="status-indicator"></div>
                    <p class="status-message">Click "Go with mood debug" to enable emotion detection.</p>
                </section>
                <section class="results-section" style="display: none;">
                    <h2>Debugging Results</h2>
                    <div class="results-container">
                        <div class="results-summary">
                            <div class="summary-item"><span class="summary-count" id="errors-count">0</span><span>Errors</span></div>
                            <div class="summary-item"><span class="summary-count" id="warnings-count">0</span><span>Warnings</span></div>
                            <div class="summary-item"><span class="summary-count" id="suggestions-count">0</span><span>Suggestions</span></div>
                        </div>
                        <div class="results-list" id="results-list"></div>
                    </div>
                </section>
            </main>
            <footer class="app-footer">
                <p>MoodLint - Debugging aligned with your emotional state</p>
            </footer>
        </div>
        <script src="${scriptUri}"></script>
    </body>
    </html>`;
}