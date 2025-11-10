# Voice Agent - Educational Q&A System

A Flask-based voice-enabled chatbot that helps students with educational queries and collects student information for MyClassBoard integration.

## Features

- **Voice Interaction**: Speech-to-text and text-to-speech capabilities
- **Q&A System**: Cluster-based question answering using CSV data
- **Student Information Collection**: Guided inquiry flow for collecting student details
- **AI Integration**: Powered by Google Gemini for intelligent responses
- **API Integration**: Connects with MyClassBoard for student enrollment

## Technology Stack

- **Backend**: Flask (Python)
- **AI/ML**: Google Gemini API
- **Speech**: Google Text-to-Speech (gTTS)
- **Data Processing**: Pandas, RapidFuzz
- **Frontend**: HTML, CSS, JavaScript

## Setup Instructions

### Prerequisites

- Python 3.9+
- Google Gemini API key
- MyClassBoard API credentials

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd DSE_DRAFT-1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:
```env
GEMINI_API_KEY=your_gemini_api_key
ENQUIRY_API_URL=https://api.myclassboard.com/api/EnquiryService/Save_EnquiryDetails
MYCLASSBOARD_API_KEY=your_api_key
MYCLASSBOARD_AUTH=your_auth_token
ORGANISATION_ID=45
BRANCH_ID=79
ACADEMIC_YEAR_ID=17
CLASS_ID=477
```

4. Prepare your Q&A data in `qa.csv` with columns: Cluster, Question, Answer, Resource

### Running the Application

```bash
python app.py
```

The application will start on `http://localhost:5001`

## Project Structure

```
DSE_DRAFT-1/
├── app.py                 # Main Flask application
├── voice_agent/
│   └── qa.py             # Core voice agent logic
├── templates/
│   └── index.html        # Frontend interface
├── static/
│   ├── css/
│   └── js/
├── qa.csv                # Q&A knowledge base
├── requirements.txt      # Python dependencies
└── README.md
```

## API Endpoints

- `GET /` - Main interface
- `GET /api/start` - Initialize conversation
- `POST /api/ask` - Process voice input
- `POST /api/choice` - Handle text/voice choices
- `POST /api/finalize_review` - Submit student information
- `GET /api/tts/<filename>` - Serve TTS audio files

## Usage

1. **Voice Interaction**: Click the microphone button and speak your question
2. **Cluster Navigation**: Say a cluster name to explore specific topics
3. **Student Information**: Say "HELP" to start the information collection process
4. **Review & Submit**: Review collected information before final submission

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Author

**Sai Manvish**
- Email: venkatasaimanvish@gmail.com