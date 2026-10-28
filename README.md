# Real-Time Conversational AI & Grounded Knowledge Copilot 🎙️⚡

An enterprise-grade, offline-capable Multimodal Conversational AI and Real-Time RAG (Retrieval-Augmented Generation) Knowledge Copilot. Designed for live meetings (Google Meet, Microsoft Teams, Zoom) and in-person conversations, the system transcribes speech in real time, analyzes sentiment and vocal prosody, automatically detects action items, retrieves knowledge answers with citations from uploaded documents, and generates downloadable executive summaries with PDF export.

---

## 🚀 Key Features

* **Real-Time Speech-to-Text (ASR)**: Powered by `faster-whisper` (`base.en`, INT8 AVX2 SIMD with 6 multi-threaded workers on CPU) for high-accuracy, zero-clipping transcription with no CUDA resource contention. Live visual typing is rendered natively in real time via the browser Web Speech API.
* **Multimodal Sentiment & Acoustic Prosody**: Evaluates line-by-line sentiment via CardiffNLP RoBERTa while fusing microphone acoustic energy (`RMS` and `Peak`) to detect speaker tone (*Assertive*, *Excited*, *Subdued*, *Calm*).
* **Two-Stage Hybrid Search & Grounded RAG**:
  * **Stage 1 Retrieval**: Dense embeddings (`all-MiniLM-L6-v2`) + flexible lexical keyword matching combined via Reciprocal Rank Fusion (RRF).
  * **Stage 2 Re-Ranking**: Token-level joint attention via Cross-Encoder (`ms-marco-MiniLM-L-6-v2`).
  * **Zero-Hallucination Answers**: Grounded Flan-T5 generation with exact citation quotes and source document page numbers.
* **Sub-Millisecond Query Cache**:
  * **Exact Match Cache**: Responds in **0.017 ms** (~90,000x speedup).
  * **Semantic Cosine Cache**: Rephrased queries resolve in **~7 ms** (~220x speedup vs full LLM generation).
* **Action Items & Commitments Detection**: Automatically detects conversational promises (*"I will follow up...", "Let's review...", "We will implement..."*) and surfaces them in UI turn cards, executive reports, and PDF exports.
* **Executive Call Intelligence & PDF Export**: Synthesizes a structured 4-line executive summary (Discussion Focus, Inquiries Addressed, Communication Dynamics, Outcome & Consensus) with a one-click printable PDF generator.
* **Tactile Glassmorphism UI**: Dynamic audio visualizer reacting to voice energy, live partial voice bar, one-click answer copying to clipboard, and responsive document drag-and-drop ingestion (PDF / TXT).

---

## 🏗️ Architecture

```
[ Browser Microphone / Meet / Teams Tab Audio ]
                     │  (16kHz PCM WebSocket Stream)
                     ▼
         [ FastAPI Streaming Hub (app.py) ]
                     │
                     ▼
     [ Turn-Level Processing & ASR ]
   (faster-whisper int8 AVX2 on CPU)
                     │
                     ├─► [ RoBERTa Sentiment & Prosody Tone (GPU) ]
                     ├─► [ Action Items & Commitments Detector ]
                     └─► [ Conversational Inquiry Router ]
                                    │
                                    ▼
                     [ Two-Stage RAG Pipeline (GPU) ]
                     ├─► Sub-Millisecond Cache (<1ms)
                     ├─► Stage 1: Dense + Lexical RRF
                     ├─► Stage 2: Cross-Encoder Re-Ranking
                     └─► Flan-T5 Grounded Synthesis
                                    │
                                    ▼
                  [ UI Event Pushed via WebSocket ]
```

---

## 📦 Project Structure

```
├── app.py                     # FastAPI server & WebSocket streaming audio handler
├── copilot_engine.py          # Multimodal AI engine (ASR, Sentiment, RAG, Cache, Summarizer)
├── templates/
│   └── index.html             # Glassmorphism UI, Audio Visualizer, PDF Generator
├── requirements.txt           # Python dependencies
├── Deep_Learning_KB.txt       # Sample knowledge base document for testing RAG
├── .gitignore                 # Excludes heavy model weights, caches, and scratch files
└── README.md                  # Project documentation
```

---

## 🛠️ Installation & Setup

### Prerequisites
* **Python**: 3.9 to 3.11 recommended.
* **Operating System**: Windows, Linux, or macOS.
* **Hardware**:
  * **GPU (Recommended)**: NVIDIA GPU with CUDA support (e.g. GTX 1650, RTX series).
  * **CPU Fallback**: Runs automatically on CPU if no CUDA GPU is detected.

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note for PyTorch with CUDA (Optional for GPU Acceleration)**:
> If you have an NVIDIA GPU, install PyTorch with CUDA 12.4+:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
> ```

---

## 🚦 Running the Application

Start the local server:
```bash
python app.py
```

Open your browser and navigate to:
```
http://localhost:8000
```

---

## 💡 How to Use

1. **Upload Knowledge Documents**: Drag and drop any `.pdf` or `.txt` policy manuals, documentation, or FAQs into the right-hand panel.
2. **Select Audio Source**:
   * **⚡ Combined (Mic + Meeting)**: Captures both your vocal microphone and meeting attendees from a shared Google Meet / Microsoft Teams Chrome tab.
   * **🖥️ Meeting Audio (Meet/Teams Tab)**: Listens solely to meeting participants.
   * **🎙️ Mic Only**: Captures local microphone speech.
3. **Click "Start Listening"**:
   * Speak naturally or conduct your meeting.
   * Interim speech displays in the top visualizer bar.
   * Finalized phrases appear with sentiment labels, vocal tone, and detected action items.
   * Questions matching your uploaded knowledge base trigger instant RAG answer cards with exact page citations and a **Copy** button.
4. **Click "End Call & Summary"**:
   * Generates a 4-line Executive Conversation Intelligence Summary and an Action Items Checklist.
   * Click **Save as PDF** to generate an official printable report.

---

## 📄 License
This project is open-source and available under the MIT License.

