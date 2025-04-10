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

Welcome to the **MoodLint**! MoodLint is a Visual Studio Code extension that helps in debugging by integrating real-time emotion analysis with advanced visualization capabilities. Designed for developers navigating the emotional highs and lows of coding, MoodLint uses deep learning to detect your emotional state—whether you're frustrated, focused, or fatigued—and provides tailored debugging suggestions. Additionally, it leverages a Generative Adversarial Network (GAN) to generate artistic visualizations of your code structure, enhancing understanding with fun, animated graphs.

Why MoodLint exist?

https://link.springer.com/article/10.1007/s10111-010-0164-1

## Features

MoodLint redefines debugging and code comprehension by aligning technical assistance with your emotional state and visual learning preferences. Below are its core features, each accompanied by detailed descriptions and placeholders for illustrative screenshots:

- **Emotion Detection**
  - **Description**: MoodLint employs Deep learning model to analyze your webcam feed or typing patterns, detecting emotions such as frustration, focus, or relaxation in real time. This feature forms the backbone of MoodLint's ability to adapt debugging support to your current mood.

- **Tailored Debugging Suggestions**
  - **Description**: Based on your detected emotional state, MoodLint offers personalized debugging tips. For example, if frustration is sensed, it might suggest simplifying a complex function or stepping away briefly, while a focused state could prompt deeper code optimization recommendations.

- **Mood Dashboard**
  - **Description**: A visual tool displaying your emotional trends over time, helping you identify patterns (e.g., frequent frustration during late-night coding) and adjust your habits for better productivity and mental health.

- **Integration with VSCode Debugger**
  - **Description**: MoodLint enhances VSCode’s native debugging tools by overlaying emotion-driven insights, such as highlighting error-prone areas when stress is detected, making debugging more intuitive and effective.

- **Graph/Visualization Generation Using GAN**
  - **Description**: MoodLint incorporates a GAN trained from scratch to generate artistic, animated visualizations of your code structure (e.g., flowcharts or abstract syntax trees). Users can trigger this feature with a button, receiving two unique, stylized graphs that enhance code comprehension. The GAN learns from artistic datasets, applying styles like impressionism or abstract patterns to static graphs parsed from your code.

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

For example:

This extension contributes the following settings:

* `myExtension.enable`: Enable/disable this extension.
* `myExtension.thing`: Set to `blah` to do something.

## Release Notes

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
