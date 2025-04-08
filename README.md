<div>

<div align="center">
    <img src="MoodLint-logo/cover.png" alt="UnFake Logo" style="width: 700px; height: 300px;">
<p>Emotion-Based Debugging Extension for Visual Studio Code</p>

</div>    
</div>
</div>

<div align="center">
  
![GitHub stars](https://img.shields.io/github/stars/negativenagesh/MoodLint?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/negativenagesh/MoodLint?style=social)
![GitHub forks](https://img.shields.io/github/forks/negativenagesh/MoodLint?style=social)
![GitHub license](https://img.shields.io/github/license/negativenagesh/MoodLint)
</div>

Welcome to the **MoodLint**! MoodLint is a groundbreaking Visual Studio Code extension that revolutionizes debugging by integrating real-time emotion analysis with advanced visualization capabilities. Designed for developers navigating the emotional highs and lows of coding, MoodLint uses machine learning to detect your emotional state—whether you're frustrated, focused, or fatigued—and provides tailored debugging suggestions. Additionally, it leverages a Generative Adversarial Network (GAN) to generate artistic visualizations of your code structure, enhancing understanding with fun, animated graphs. The project architecture, illustrated below, ensures a robust, scalable foundation for this innovative tool.

The attached project architecture image reveals a well-organized structure within the `MOODLINT` root folder:
- **`.vscode`**: Contains configuration files like `launch.json` and `tasks.json` for debugging and task automation.
- **`media`**: Houses subfolders `images` (for assets like screenshots) and `styles.css` (for webview styling).
- **`node_modules`**: Stores installed dependencies for the extension.
- **`out`**: Holds compiled output files generated during the build process.
- **`src`**: Contains source files such as `extension.js`, `extension.js.map`, `webviewContent.js`, and `webviewContent.js.map`.
- **`test`**: Includes test files like `extension.test.ts`, `extension.ts`, and `webviewContent.ts`.
- **Configuration Files**: Includes `.gitignore`, `.vscodeignore`, `CHANGELOG.md`, `.eslintrc.json`, `LICENSE`, `package-lock.json`, `package.json`, `README.md`, `tsconfig.json`, and `vscode-extension-quickstart.md`.

## Features

MoodLint redefines debugging and code comprehension by aligning technical assistance with your emotional state and visual learning preferences. Below are its core features, each accompanied by detailed descriptions and placeholders for illustrative screenshots:

- **Emotion Detection**
  - **Description**: MoodLint employs machine learning models running on a local Python server to analyze your webcam feed or typing patterns, detecting emotions such as frustration, focus, or relaxation in real time. This feature forms the backbone of MoodLint's ability to adapt debugging support to your current mood.
  - *Screenshot*: `![Emotion Detection](images/emotion-detection.png)`
  - *Details*: The screenshot highlights the emotion detection interface, featuring a real-time mood indicator that updates as you code.

- **Tailored Debugging Suggestions**
  - **Description**: Based on your detected emotional state, MoodLint offers personalized debugging tips. For example, if frustration is sensed, it might suggest simplifying a complex function or stepping away briefly, while a focused state could prompt deeper code optimization recommendations.
  - *Screenshot*: `![Debugging Suggestions](images/debugging-suggestions.png)`
  - *Details*: This image shows a pop-up notification with context-aware debugging advice, seamlessly integrated into the VSCode editor.

- **Mood Dashboard**
  - **Description**: A visual tool displaying your emotional trends over time, helping you identify patterns (e.g., frequent frustration during late-night coding) and adjust your habits for better productivity and mental health.
  - *Screenshot*: `![Mood Dashboard](images/mood-dashboard.png)`
  - *Details*: The screenshot illustrates the dashboard, with graphs and summaries of your emotional state during coding sessions.

- **Integration with VSCode Debugger**
  - **Description**: MoodLint enhances VSCode’s native debugging tools by overlaying emotion-driven insights, such as highlighting error-prone areas when stress is detected, making debugging more intuitive and effective.
  - *Screenshot*: `![Debugger Integration](images/debugger-integration.png)`
  - *Details*: This image demonstrates how MoodLint annotates breakpoints and error messages with mood-based suggestions.

- **Graph/Visualization Generation Using GAN**
  - **Description**: MoodLint incorporates a GAN trained from scratch to generate artistic, animated visualizations of your code structure (e.g., flowcharts or abstract syntax trees). Users can trigger this feature with a button, receiving two unique, stylized graphs that enhance code comprehension. The GAN learns from artistic datasets, applying styles like impressionism or abstract patterns to static graphs parsed from your code.
  - *Screenshot*: `![GAN Visualization](images/gan-visualization.png)`
  - *Details*: This screenshot showcases two animated graphs generated from a sample code snippet, styled with vibrant GAN-produced textures.

> **Tip**: To fully showcase MoodLint’s capabilities, include animations in the `media/images/animations` folder (e.g., `emotion-tracking.gif` or `gan-animation.gif`). Short, focused animations can vividly demonstrate real-time emotion detection, suggestion delivery, and GAN-generated visualizations. Ensure all image paths are relative to this README (e.g., `images/feature-x.png`) and stored in the `media/images` folder as per the project architecture.

## Requirements

To harness MoodLint’s full potential, ensure your environment meets the following prerequisites. This section provides detailed installation and configuration steps:

- **Visual Studio Code**
  - **Version**: 1.60.0 or higher for compatibility with MoodLint’s features.
  - **Installation**: Download from [code.visualstudio.com](https://code.visualstudio.com/) and verify with `code --version`.

- **Node.js**
  - **Version**: 14.x or later, required for extension development and runtime.
  - **Installation**: Install from [nodejs.org](https://nodejs.org/), then confirm with `node -v`.

- **Python**
  - **Version**: 3.8 or later, essential for the local Python server powering emotion detection and GAN generation.
  - **Installation**: Get it from [python.org](https://www.python.org/), ensuring `pip` is included. Verify with `python --version` or `python3 --version`.

- **Dependencies**
  - **Setup**: Navigate to the `MOODLINT` root folder and run `npm install` to install Node.js dependencies, populating the `node_modules` folder. For Python dependencies, install the following:
    ```bash
    pip install fastapi uvicorn opencv-python deepface tensorflow numpy pillow graphviz

## Requirements

If you have any requirements or dependencies, add a section describing those and how to install and configure them.

## Extension Settings

Include if your extension adds any VS Code settings through the `contributes.configuration` extension point.

For example:

This extension contributes the following settings:

* `myExtension.enable`: Enable/disable this extension.
* `myExtension.thing`: Set to `blah` to do something.

## Known Issues

Calling out known issues can help limit users opening duplicate issues against your extension.

## Release Notes

Users appreciate release notes as you update your extension.

### 1.0.0

Initial release of ...

### 1.0.1

Fixed issue #.

### 1.1.0

Added features X, Y, and Z.

---

## Following extension guidelines

Ensure that you've read through the extensions guidelines and follow the best practices for creating your extension.

* [Extension Guidelines](https://code.visualstudio.com/api/references/extension-guidelines)

## Working with Markdown

You can author your README using Visual Studio Code. Here are some useful editor keyboard shortcuts:

* Split the editor (`Cmd+\` on macOS or `Ctrl+\` on Windows and Linux).
* Toggle preview (`Shift+Cmd+V` on macOS or `Shift+Ctrl+V` on Windows and Linux).
* Press `Ctrl+Space` (Windows, Linux, macOS) to see a list of Markdown snippets.

## For more information

* [Visual Studio Code's Markdown Support](http://code.visualstudio.com/docs/languages/markdown)
* [Markdown Syntax Reference](https://help.github.com/articles/markdown-basics/)

**Enjoy!**
