# Real-Time Emotion Detector

This project is a Python-based real-time emotion detection system that uses advanced AI and computer vision to recognize human emotions through facial expressions captured from a webcam feed.

## Features
- **Emotion Recognition**: Detects emotions such as happiness, sadness, anger, and more.
- **Real-Time Analysis**: Processes video frames live to provide immediate feedback.
- **Accurate Detection**: Combines MTCNN for face detection and DeepFace for emotion analysis.
- **Smooth Performance**: Optimized buffering ensures stable and efficient emotion recognition.
- **Mirrored Camera Feed**: Provides a natural, user-friendly interaction.

## Technologies Used
- **Python**: Core language for development.
- **OpenCV**: Handles video processing and display.
- **MTCNN**: Face detection framework.
- **DeepFace**: For emotion analysis and recognition.
- **TensorFlow**: Backend for deep learning computations.

---

## Setup Guide

Follow these steps to set up the project:

### Prerequisites
1. Ensure you have **Python 3.7+** installed.
2. Install **pip** for managing Python packages.

### Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/real-time-emotion-detector.git
   cd real-time-emotion-detection
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv emvenv
   source emvenv/bin/activate   # On Windows: emvenv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Required Models**:
   Ensure the required MTCNN and DeepFace models are available. The system will automatically download them during the first run if not present.

---

## Usage

1. **Run the Script**:
   ```bash
   python main.py
   ```

2. **Interact with the Webcam Feed**:
   - The system will start capturing video from your default webcam.
   - Detected emotions will be displayed above the face in real time.

---

## Project Structure
```
real-time-emotion-detection/
├── models/                # Pre-trained models
├── main.py                # Main script for running emotion detection
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── emvenv/                # Virtual environment (excluded from Git)
```

---

## Troubleshooting
- If you encounter issues with TensorFlow or DeepFace, ensure that your environment meets the prerequisites and dependencies.
- Use `pip freeze` to verify the installed package versions.

---

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments
Special thanks to the creators of MTCNN, DeepFace, and OpenCV for their excellent libraries that make this project possible.
