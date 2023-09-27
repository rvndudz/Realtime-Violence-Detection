# Realtime-Violence-Detection
A deep learning-based violence detection system powered by CLIP. Detect and analyze violent content in videos with ease.

# Real-Time Violence Detection with CLIP

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Overview

This Python project utilizes the CLIP (Contrastive Language–Image Pretraining) model to perform real-time violence detection in video content. It allows you to investigate video files or analyze live webcam input to identify violent scenes.

![Demo](demo.gif)

**Key Features:**

- Real-time violence detection using CLIP.
- Support for video file analysis and live webcam input.
- Customizable violence probability threshold.
- GPU acceleration for faster inference (if available).
- Annotates frames with violence probability when the threshold is met.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/rvndudz/clip-violence-detection.git
   ```

2. Change to the project directory:
   ```
   cd clip-violence-detection
   ```

4. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

6. Download the CLIP model weights:
   ```
   python download_clip_model.py
   ```

## Usage
- Video File Analysis
- Live Webcam Analysis

## Contributing
### Contributions are welcome! If you would like to contribute to the project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with descriptive messages.
4. Push your branch to your fork.
5. Submit a pull request to the main branch of this repository.

## License
This project is licensed under the MIT License - see the [<ins>LICENSE</ins>](https://opensource.org/license/mit/) file for details.
