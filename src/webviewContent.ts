import * as vscode from 'vscode';

/**
 * Generates the HTML content for the MoodLint webview panel
 */
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
                <div class="header-subtitle">Code analysis aligned with your mood</div>
            </header>
            
            <main class="app-content">
                <section class="mood-selector-section">
                    <h2>Select Your Mood</h2>
                    <p>Choose your current mood to customize how MoodLint will analyze your code</p>
                    <div class="mood-options">
                        <div class="mood-option" data-mood="happy">
                            <div class="mood-icon">😊</div>
                            <div class="mood-label">Happy</div>
                            <div class="mood-description">Focus on positive improvements</div>
                        </div>
                        <div class="mood-option" data-mood="focused">
                            <div class="mood-icon">🧠</div>
                            <div class="mood-label">Focused</div>
                            <div class="mood-description">Strict and thorough analysis</div>
                        </div>
                        <div class="mood-option" data-mood="tired">
                            <div class="mood-icon">😴</div>
                            <div class="mood-label">Tired</div>
                            <div class="mood-description">Focus on critical issues only</div>
                        </div>
                        <div class="mood-option" data-mood="creative">
                            <div class="mood-icon">🎨</div>
                            <div class="mood-label">Creative</div>
                            <div class="mood-description">Suggest alternative approaches</div>
                        </div>
                        <div class="mood-option" data-mood="stressed">
                            <div class="mood-icon">😰</div>
                            <div class="mood-label">Stressed</div>
                            <div class="mood-description">Simplified, less overwhelming feedback</div>
                        </div>
                    </div>
                </section>
                
                <section class="settings-section">
                    <h2>Analysis Settings</h2>
                    <div class="settings-container">
                        <div class="settings-column">
                            <div class="settings-item">
                                <input type="checkbox" id="check-syntax" checked>
                                <label for="check-syntax">Syntax issues</label>
                            </div>
                            <div class="settings-item">
                                <input type="checkbox" id="check-style" checked>
                                <label for="check-style">Style suggestions</label>
                            </div>
                        </div>
                        <div class="settings-column">
                            <div class="settings-item">
                                <input type="checkbox" id="check-performance">
                                <label for="check-performance">Performance recommendations</label>
                            </div>
                            <div class="settings-item">
                                <input type="checkbox" id="check-best-practices">
                                <label for="check-best-practices">Best practices</label>
                            </div>
                        </div>
                    </div>
                </section>
                
                <section class="actions-section">
                    <button id="analyze-btn" class="primary-button">Start Analysis</button>
                </section>
                
                <section class="status-section">
                    <div class="status-indicator"></div>
                    <p class="status-message">Ready to analyze your code. Select your mood and press "Start Analysis".</p>
                </section>
                
                <section class="results-section" style="display: none;">
                    <h2>Analysis Results</h2>
                    <div class="results-container">
                        <div class="results-summary">
                            <div class="summary-item">
                                <span class="summary-count" id="errors-count">0</span>
                                <span class="summary-label">Errors</span>
                            </div>
                            <div class="summary-item">
                                <span class="summary-count" id="warnings-count">0</span>
                                <span class="summary-label">Warnings</span>
                            </div>
                            <div class="summary-item">
                                <span class="summary-count" id="suggestions-count">0</span>
                                <span class="summary-label">Suggestions</span>
                            </div>
                        </div>
                        <div class="results-list" id="results-list">
                            <!-- Results will be inserted here by JavaScript -->
                        </div>
                    </div>
                </section>
            </main>
            
            <footer class="app-footer">
                <div class="footer-content">
                    <p>MoodLint - Let your mood guide your code quality</p>
                </div>
            </footer>
        </div>
        
        <script src="${scriptUri}"></script>
    </body>
    </html>`;
}