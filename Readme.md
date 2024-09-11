# LipBuddy: Lip-Reading Application

LipBuddy is a deep learning-based application for lip reading. Built on top of TensorFlow, it processes video input, extracts lip movements, and predicts text from the video using a trained model. The application features a clean and interactive interface powered by Streamlit.

## Features

- Lip movement to text prediction using a trained deep learning model
- Interactive video processing and model output display
- Conversion of video formats (MPG to MP4)
- Displays video frames and corresponding prediction tokens
- Modular design with easy-to-extend components

## How It Works

LipBuddy uses a combination of **Conv3D**, **LSTM**, and **CTC decoding** to process video frames and predict sequences of lip movements. It uses pre-trained weights to decode lip movements into meaningful text.