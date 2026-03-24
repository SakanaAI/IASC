# IASC Phonotactics Generator - Web Application

An interactive web interface for generating phonological systems for constructed languages using the IASC system.

## Features

- **User-friendly interface** for generating phonotactics without command-line knowledge
- **Multiple language models** supported (Claude, GPT, Gemini)
- **Target language selection** from 5 pre-configured languages (French, Hawaiian, Japanese, Spanish, Welsh)
- **Iterative refinement** with configurable iteration count
- **Visual phoneme display** showing consonants and vowels with frequencies
- **Sample word generation** from the created phonotactic system
- **Generation process transparency** with iteration-by-iteration LLM reasoning
- **Secure API key handling** (optional, can use environment variables)
- **Download capability** for generated Python phonotactics files

## Installation

### Prerequisites

- Python 3.8+
- Parent IASC repository dependencies installed (see main `requirements.txt`)

### Setup

1. Install web app dependencies:
```bash
cd WebApp
pip install -r requirements.txt
```

2. Set up API credentials (choose one method):

   **Method A: Environment Variables (Recommended)**
   ```bash
   # For Claude (via AWS Bedrock)
   export AWS_ACCESS_KEY_ID=<your_key>
   export AWS_SECRET_ACCESS_KEY=<your_secret>

   # For OpenAI
   export OPENAI_API_KEY=<your_key>

   # For Gemini
   export GEMINI_API_KEY=<your_key>
   ```

   **Method B: Web Interface**
   - Enter credentials directly in the web UI (they're only used for the current session)

## Running the Application

```bash
cd WebApp
python app.py
```

The web app will start on `http://localhost:5000`

## Usage

1. **Select Target Language**: Choose a language whose phonological system you want to model
2. **Set Iterations**: Choose how many refinement iterations (1-50, default: 10)
3. **Select Language Model**: Choose your preferred LLM
4. **Configure API Keys** (if not using environment variables): Click "Configure API Keys" and enter your credentials
5. **Generate**: Click "Generate Phonotactics" and wait for the process to complete
6. **View Results**:
   - **Phoneme Inventory**: See the generated consonants and vowels with their frequencies
   - **Sample Words**: View example words generated using the phonotactic rules
   - **Generation Process**: Explore the LLM's reasoning through each iteration
7. **Download**: Download the final Python generator file for use in your projects

## How It Works

The web app integrates with the existing IASC phonology generation system:

1. User submits configuration through the web interface
2. Backend calls `agentic_phonology/run_phonology_main.py` with the specified parameters
3. The system iteratively generates and refines phonotactics over multiple iterations
4. Each iteration produces:
   - A `.txt` file with the LLM's reasoning and changes
   - A `.py` file with the phonotactics generator code
5. Results are parsed and displayed in an accessible format
6. Users can download the final generator for their own use

## Architecture

```
WebApp/
├── app.py                 # Flask backend
├── templates/
│   └── index.html        # Main web interface
├── static/
│   ├── css/
│   │   └── style.css     # Styling
│   └── js/
│       └── app.js        # Frontend JavaScript
├── requirements.txt       # Web app dependencies
└── README.md             # This file
```

## API Endpoints

### `POST /api/generate`
Generate phonotactics based on configuration.

**Request Body:**
```json
{
  "language": "Japanese",
  "iterations": 10,
  "model": "claude",
  "aws_access_key_id": "optional",
  "aws_secret_access_key": "optional",
  "openai_api_key": "optional",
  "gemini_api_key": "optional"
}
```

**Response:**
```json
{
  "session_id": "abc123...",
  "status": "success",
  "results": {
    "final_phonemes": {
      "consonants": {"k": 10, "s": 9, ...},
      "vowels": {"a": 10, "o": 8, ...}
    },
    "sample_words": ["k a t a", "s u k i", ...],
    "reasoning": [...]
  }
}
```

### `GET /api/download/<session_id>/<filename>`
Download a generated file.

### `POST /api/cleanup/<session_id>`
Clean up temporary session files.

## Security Notes

- API keys entered in the web interface are only used for the current session
- Keys are stored in memory and cleared when the session ends
- For production use, consider implementing:
  - User authentication
  - Rate limiting
  - Persistent storage (Redis/database) instead of in-memory sessions
  - HTTPS/TLS encryption
  - Input sanitization and validation

## Troubleshooting

### "Module not found" errors
Ensure you've installed the main IASC dependencies:
```bash
cd ..  # Go to main IASC directory
pip install -r requirements.txt
```

### "Can't invoke Claude" errors
- Check that your AWS credentials are set correctly
- Verify you have access to AWS Bedrock with Claude models
- Ensure your AWS credentials have the necessary permissions

### Generation takes too long
- Reduce the number of iterations
- Use a faster model (e.g., Gemini Flash instead of Pro)
- Check your internet connection

### Sample words not generating
- This may occur if the phonotactics Python file has syntax errors
- Check the iteration tabs to see if there were errors in earlier iterations
- Try regenerating with fewer iterations

## Future Enhancements

- [ ] Real-time progress updates (WebSocket support)
- [ ] Interactive editing of phoneme inventories
- [ ] Phonological rule generation interface
- [ ] Export to multiple formats (JSON, CSV, documentation)
- [ ] User accounts and saved projects
- [ ] Comparison between different generated systems
- [ ] Integration with morphosyntax generation stages

## Contributing

This web interface is part of the IASC research project. For questions or contributions, see the main repository README.

## Citation

If you use this tool in your research, please cite:

```
Chihiro Taguchi and Richard Sproat. 2025. "IASC: Interactive Agentic System for ConLangs".
https://arxiv.org/abs/2510.07591.
```
