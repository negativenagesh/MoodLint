import * as vscode from 'vscode';
import * as path from 'path';
import * as childProcess from 'child_process';
import * as fs from 'fs';

export interface DebugRequest {
    code: string;
    filename: string;
    mood: string;
    query?: string;
}

export interface DebugResponse {
    success: boolean;
    mood: string;
    response?: string;
    error?: string;
}

/**
 * Interface with the Python agent system for mood-based debugging
 */
export class AgentInterface {
    private pythonPath: string;
    private agentPath: string;
    private extensionPath: string;

    constructor(extensionPath: string) {
        this.extensionPath = extensionPath;
        this.agentPath = path.join(extensionPath, 'agents', 'workflow.py');
        this.pythonPath = 'python3'; // Adjust based on OS if needed
    }

    /**
     * Execute the agent workflow to debug code
     */
    public async debugCode(
        code: string,
        filename: string,
        mood: string = 'focused',
        query: string = ''
    ): Promise<DebugResponse> {
        try {
            // Create a temporary file to hold the code and request
            const tempDir = path.join(this.extensionPath, 'temp');
            if (!fs.existsSync(tempDir)) {
                fs.mkdirSync(tempDir, { recursive: true });
            }

            const tempCodeFile = path.join(tempDir, `debug_code_${Date.now()}.txt`);
            const tempRequestFile = path.join(tempDir, `debug_request_${Date.now()}.json`);

            // Write the code to a temp file
            fs.writeFileSync(tempCodeFile, code);

            // Create the request object
            const request: DebugRequest = {
                code,
                filename,
                mood,
                query
            };

            // Write request to temp file
            fs.writeFileSync(tempRequestFile, JSON.stringify(request, null, 2));

            // Execute the Python agent workflow
            return new Promise<DebugResponse>((resolve, reject) => {
                const process = childProcess.spawn(this.pythonPath, [
                    this.agentPath,
                    '--file', tempCodeFile,
                    '--mood', mood,
                    '--query', query || ''
                ]);

                let stdout = '';
                let stderr = '';

                process.stdout.on('data', (data) => {
                    stdout += data.toString();
                });

                process.stderr.on('data', (data) => {
                    stderr += data.toString();
                });

                process.on('close', (code) => {
                    // Clean up temp files
                    try {
                        fs.unlinkSync(tempCodeFile);
                        fs.unlinkSync(tempRequestFile);
                    } catch (err) {
                        console.error('Error cleaning up temp files:', err);
                    }

                    if (code === 0) {
                        try {
                            // Parse the JSON response
                            const response: DebugResponse = JSON.parse(stdout);
                            resolve(response);
                        } catch (err) {
                            reject({
                                success: false,
                                mood,
                                error: `Failed to parse agent response: ${err}`
                            });
                        }
                    } else {
                        reject({
                            success: false,
                            mood,
                            error: `Agent process exited with code ${code}: ${stderr}`
                        });
                    }
                });

                process.on('error', (err) => {
                    reject({
                        success: false,
                        mood,
                        error: `Failed to start agent process: ${err.message}`
                    });
                });
            });
        } catch (err) {
            return {
                success: false,
                mood,
                error: `Agent interface error: ${err instanceof Error ? err.message : String(err)}`
            };
        }
    }

    /**
     * Check if the agent system is properly installed
     */
    public async checkAgentInstallation(): Promise<boolean> {
        try {
            // Check if the agent path exists
            if (!fs.existsSync(this.agentPath)) {
                vscode.window.showErrorMessage(`Agent workflow script not found at: ${this.agentPath}`);
                return false;
            }

            // Try running the Python interpreter
            return new Promise<boolean>((resolve) => {
                const process = childProcess.spawn(this.pythonPath, ['--version']);
                
                process.on('close', (code) => {
                    if (code === 0) {
                        // Python is available, now check for required packages
                        this.checkDependencies().then(resolve).catch(() => resolve(false));
                    } else {
                        vscode.window.showErrorMessage(`Python interpreter not found. Please install Python 3.8+.`);
                        resolve(false);
                    }
                });

                process.on('error', () => {
                    vscode.window.showErrorMessage(`Python interpreter not found. Please install Python 3.8+.`);
                    resolve(false);
                });
            });
        } catch (err) {
            vscode.window.showErrorMessage(`Failed to check agent installation: ${err instanceof Error ? err.message : String(err)}`);
            return false;
        }
    }

    /**
     * Install required Python dependencies
     */
    public async installDependencies(): Promise<boolean> {
        try {
            const requirementsPath = path.join(this.extensionPath, 'agents', 'requirements.txt');
            
            if (!fs.existsSync(requirementsPath)) {
                vscode.window.showErrorMessage(`Requirements file not found at: ${requirementsPath}`);
                return false;
            }

            // Show installation progress
            const terminal = vscode.window.createTerminal('MoodLint Dependency Installation');
            terminal.show();
            terminal.sendText(`${this.pythonPath} -m pip install -r "${requirementsPath}"`);
            
            // We can't easily detect when the terminal command completes,
            // so we'll just return true and let the user see the terminal output
            return true;
        } catch (err) {
            vscode.window.showErrorMessage(`Failed to install dependencies: ${err instanceof Error ? err.message : String(err)}`);
            return false;
        }
    }

    /**
     * Check if required dependencies are installed
     */
    private async checkDependencies(): Promise<boolean> {
        return new Promise<boolean>((resolve, reject) => {
            // Check for a key package like langchain
            const process = childProcess.spawn(this.pythonPath, [
                '-c', 'import langchain, google.generativeai, langraph'
            ]);
            
            process.on('close', (code) => {
                if (code === 0) {
                    resolve(true);
                } else {
                    vscode.window.showWarningMessage(
                        'MoodLint dependencies not found. Would you like to install them?',
                        'Install', 'Cancel'
                    ).then(selection => {
                        if (selection === 'Install') {
                            this.installDependencies().then(resolve).catch(reject);
                        } else {
                            reject(new Error('Dependencies not installed'));
                        }
                    });
                }
            });

            process.on('error', reject);
        });
    }
}