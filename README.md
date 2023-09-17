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
- [Options](#options)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
   git clone https://github.com/yourusername/clip-violence-detection.git

Change to the project directory:
  cd clip-violence-detection

Install the required Python packages:
  pip install -r requirements.txt

Download the CLIP model weights:
  python download_clip_model.py
