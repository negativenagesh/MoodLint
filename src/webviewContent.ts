import * as vscode from 'vscode';

export function getWebviewContent(
    stylesUri: vscode.Uri, 
    scriptUri: vscode.Uri,
    imageUri: vscode.Uri,
    faviconUri: vscode.Uri,
    nonce: string,
    cspSource: string
): string {
    return `<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src ${cspSource} blob: data:; style-src ${cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}'; media-src mediastream:;">
        <title>MoodLint</title>
        <link rel="shortcut icon" type="image/png" href="${faviconUri}" />
        <link rel="stylesheet" href="${stylesUri}">
        <style>
            /* Ensure proper scrolling */
            html, body {
                height: 100%;
                overflow-y: auto !important;
            }
            
            .app-container {
                min-height: 100%;
                overflow-y: visible !important;
                overflow: visible !important;
            }
            
            .app-content {
                overflow-y: visible !important;
            }
            
            /* Fix logo scrolling */
            .logo {
                max-width: 100%;
                position: relative !important;
                width: auto !important;
                height: auto !important;
                max-height: 300px;
            }
            
            /* Center buttons */
            .button-container {
                display: flex;
                justify-content: center;
                gap: 10px;
                margin-top: var(--spacing-unit);
            }
            
            /* GitHub corner link */
            .github-corner {
                position: fixed;
                top: 12px;
                left: 12px;
                z-index: 1000;
            }
            
            .github-corner svg {
                width: 32px;
                height: 32px;
                fill: #333;
                transition: fill 0.3s ease;
            }
            
            .github-corner:hover svg {
                fill: #0366d6;
            }
            
            /* Adjust for dark theme */
            body.vscode-dark .github-corner svg {
                fill: #fff;
            }
            
            body.vscode-dark .github-corner:hover svg {
                fill: #58a6ff;
            }
        </style>
    </head>
    <body>
        <div class="app-container">
            <!-- GitHub Corner Link -->
            <a href="https://github.com/negativenagesh/MoodLint" target="_blank" class="github-corner" aria-label="View source on GitHub">
                <svg viewBox="0 0 16 16" width="32" height="32" aria-hidden="true">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            </a>
            
            <header class="app-header">
                <img src="${imageUri}" alt="MoodLint Logo" class="logo">
                <div class="header-subtitle">You may be in any mood MoodLint will make you productive</div>
            </header>
            <main class="app-content">
                <section class="mood-detector-section">
                    <h2>MoodLint</h2>
                    <p>Emotion-Based Debugging assistant to make you productive in any mood</p>
                    
                    <div class="about-moodlint">
                        <p>MoodLint is a Visual Studio Code extension designed to enhance the debugging experience by integrating real-time emotion analysis with tailored debugging suggestions and artistic code visualizations powered by a Generative Adversarial Network (GAN). But why is such a tool necessary? Research, such as the study "Do Moods Affect Programmers' Debug Performance?" by Khan, Brinkman, and Hierons (2011), provides compelling evidence that programmers' emotional states significantly influence their debugging performance. Below, we explore this connection and explain why a mood-based debugging tool like MoodLint addresses a critical need for developers.</p>
                        
                        <p>Debugging is a critical part of software development, requiring intense focus, problem-solving skills, and patience to identify and fix errors in code. A programmer's mood—whether they are stressed, frustrated, anxious, or calm—can directly affect how well they perform this task. Here's why:</p>
                        
                        <p>Negative Moods and Cognitive Impact: When a programmer is in a bad mood, such as feeling frustrated after hours of chasing a bug, their ability to concentrate may diminish. Stress or anxiety can lead to tunnel vision, where they fixate on a single approach (even if it's ineffective) or overlook obvious solutions. For example, a stressed programmer might repeatedly tweak the same section of code without stepping back to consider the broader system, prolonging the debugging process. Positive Moods and Creativity: Conversely, a positive or calm mood can enhance creativity and flexibility. A programmer who feels relaxed or confident might approach a bug with fresh perspectives, experimenting with alternative solutions or spotting patterns that a frustrated programmer might miss. This suggests that mood doesn't just affect speed—it can influence the quality of the debugging outcome. Emotional Fatigue: Debugging often involves dealing with complex, elusive problems that can wear down a programmer over time. Emotional fatigue from prolonged debugging sessions can reduce attention to detail, increasing the likelihood of errors or incomplete fixes. Recognizing mood could help mitigate this by prompting breaks or adjustments in approach.</p>
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
                    <div class="button-container">
                        <button id="enable-camera-btn" class="primary-button">Go with mood debug</button>
                        <button id="analyze-btn" class="primary-button" disabled>Start Debugging</button>
                        <button id="predict-mood-btn" class="primary-button">Predict Future Mood</button>                        
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
        <script nonce="${nonce}" src="${scriptUri}"></script>
    </body>
    </html>`;
}