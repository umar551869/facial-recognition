# 👤 Facial Recognition System

A secure facial recognition system built with Python, OpenCV, and Streamlit that provides user registration, authentication, and continuous monitoring with gaze detection.

## 🚀 Features

- **User Registration**: Capture and store face samples for new users
- **Secure Login**: Face-based authentication system
- **Real-time Monitoring**: Continuous face recognition and monitoring
- **Gaze Detection**: Monitors user attention and gaze direction
- **Security Features**: 
  - Auto-lock on unauthorized access
  - Screen turn-off after multiple gaze violations
  - Session timeout management
- **Cross-platform**: Works on Windows, macOS, and Linux

## 🛠️ Technologies Used

- **Python 3.7+**
- **OpenCV (opencv-contrib-python)**: Computer vision and face detection
- **Streamlit**: Web-based user interface
- **NumPy**: Numerical operations
- **Haar Cascades**: Face and eye detection
- **LBPH Face Recognizer**: Face recognition algorithm

## 📋 Requirements

- Python 3.7 or higher
- Webcam access
- Windows/macOS/Linux operating system

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/facial_recognition.git
   cd facial_recognition
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - **Windows:**
     ```bash
     .\venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies:**
   ```bash
   pip install opencv-contrib-python numpy streamlit
   ```

## 🎯 Usage

1. **Run the application:**
   ```bash
   streamlit run facial_recognition.py
   ```

2. **Open your browser** and navigate to the provided URL (usually `http://localhost:8501`)

3. **Register a new user:**
   - Select "Register" mode
   - Enter a username
   - Click "Start Registration"
   - Keep your face in frame while the system captures 60 samples
   - Wait for model training to complete

4. **Login:**
   - Select "Login" mode
   - Enter your registered username
   - Look at the camera for face verification
   - Once verified, you're logged in

5. **Monitoring:**
   - The system continuously monitors your presence
   - Look away 3 times and the screen will turn off
   - Unauthorized users trigger immediate computer lock

## ⚙️ Configuration

Key parameters can be modified in `facial_recognition.py`:

- `FACE_SAMPLES_PER_USER`: Number of face samples per user (default: 60)
- `MIN_FACES_TO_TRAIN`: Minimum faces required for training (default: 20)
- `REQUIRED_CONSEC_MATCHES`: Consecutive matches needed for login (default: 8)
- `RECOGNITION_CONFIDENCE_MAX`: Maximum confidence threshold (default: 80)
- `WARNING_LIMIT`: Gaze warnings before screen turn-off (default: 3)
- `SESSION_TIMEOUT_MIN`: Session timeout in minutes (default: 10)

## 🔒 Security Features

- **Face Recognition**: Uses LBPH algorithm for reliable identification
- **Gaze Monitoring**: Tracks user attention and direction
- **Auto-lock**: Locks computer on unauthorized access
- **Screen Control**: Turns off screen after multiple violations
- **Session Management**: Automatic timeout and logout

## 📁 Project Structure

```
facial_recognition/
├── facial_recognition.py    # Main application file
├── README.md               # Project documentation
├── .gitignore             # Git ignore rules
├── venv/                  # Virtual environment (not in repo)
├── face_data/             # Face samples (auto-generated)
└── models/                # Trained models (auto-generated)
```

## 🌟 How It Works

1. **Face Detection**: Uses Haar cascades to detect faces in real-time
2. **Feature Extraction**: Captures multiple face samples during registration
3. **Model Training**: Trains LBPH face recognizer on collected samples
4. **Recognition**: Compares live face with trained model for authentication
5. **Gaze Detection**: Analyzes eye positions to determine gaze direction
6. **Security Actions**: Executes security measures based on violations

## 🐛 Troubleshooting

- **Webcam not working**: Ensure camera permissions are granted
- **Installation errors**: Make sure you're using Python 3.7+ and have pip updated
- **Face detection issues**: Ensure good lighting and face visibility
- **Performance issues**: Reduce `FACE_SAMPLES_PER_USER` for faster training

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This system is designed for educational and personal use. Users are responsible for ensuring compliance with local privacy and security regulations.

## 📞 Support

If you encounter any issues or have questions, please open an issue on GitHub.

---

**Made with ❤️ using Python, OpenCV, and Streamlit** 