# AVAQ

AVAQ is a Python-based AI assistant project that combines voice interaction, webcam input, computer vision, and AI-powered processing.

## 🚀 Features

* 🎙️ Voice interaction
* 👁️ Webcam and live video processing
* 🤖 Object detection using YOLO
* 🧠 Vision-based AI processing
* 🔊 Voice testing and wake-word functionality
* 📷 Real-time frame processing

## 📁 Project Structure

```text
AVAQ/
│
├── avaq.py              # Main AVAQ application
├── avaq_v2.py           # Second version / development version
├── avaq_v3.py           # Third version / development version
├── webcam.py            # Webcam functionality
├── qwen_vision.py       # Vision AI functionality
├── test_voice.py        # Voice testing
├── wake_test.py         # Wake-word testing
├── yolo26n.pt           # YOLO model
├── .gitignore           # Git ignored files
└── README.md            # Project documentation
```

## 🛠️ Requirements

* Python 3.x
* Git
* Webcam (for vision features)
* Microphone (for voice features)

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/prathameshkarve11/AVAQ.git
cd AVAQ
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
```

### 3. Activate the virtual environment

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

If a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

Otherwise, install the required Python packages used by the individual scripts.

## ▶️ Running AVAQ

Run the main application:

```bash
python avaq.py
```

For webcam functionality:

```bash
python webcam.py
```

For vision functionality:

```bash
python qwen_vision.py
```

## 🔐 Environment Variables

If the project uses API keys or other secrets, store them in a `.env` file.

**Never commit API keys, passwords, or other secrets to GitHub.**

Example:

```env
API_KEY=your_api_key_here
```

The `.env` file is ignored by Git.

## 📌 Project Status

🚧 **Active Development**

AVAQ is currently under development, and features may change as the project evolves.

## 🔮 Future Plans

* Improve natural voice interaction
* Add more AI capabilities
* Improve real-time computer vision
* Add better wake-word detection
* Create a more complete personal AI assistant
* Improve performance and reliability

## 👨‍💻 Author

**Prathamesh**

GitHub: [@prathameshkarve11](https://github.com/prathameshkarve11)

## 📄 License

This project currently does not specify a license.
