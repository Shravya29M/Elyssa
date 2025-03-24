---
title: Elyssa - detects conflicting emotions
emoji: 😊
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.20.1
app_file: app.py
pinned: false
license: mit
models:
- seasalt29/imageModelBig
- seasalt29/model3
---

# Elyssa - Emotion-Aware Mental Health Assistant

Elyssa captures facial expressions in real-time before each interaction to incorporate non-verbal emotional indicators. The system combines facial emotion detection with text-based analysis to provide more personalized and effective mental health support.

## Features

- Real-time facial emotion detection using Llama 3.2 11B Vision Instruct model
- Mental health counseling using a fine-tuned Llama-3.2-3B model
- Integration of detected emotions with user's text input
- User-friendly interface with webcam support

## Technical Details

### Models

- **Emotion Detection**: Fine-tuned Llama 3.2 11B Vision Instruct model trained on the Tukey Human Emotion Dataset
- **Mental Health Counseling**: Fine-tuned Llama-3.2-3B model trained on the MentalChat16k dataset

### Hardware Requirements

This application requires a GPU for optimal performance.

## Usage

1. Allow webcam access when prompted
2. Capture your facial expression
3. Type your message or question
4. Click "Submit" to receive a personalized response

## Development

This project uses Unsloth for efficient model loading and inference, combined with Gradio for the user interface.

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference