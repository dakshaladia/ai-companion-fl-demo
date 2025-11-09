# FL Mental Health Portal 🧠💚

A privacy-preserving mental health chatbot that uses **Federated Learning** to adapt responses based on user mood.

Waitlist: https://mindfree.app

## 🎯 What It Does

1. **Chat**: Talk to an empathetic AI assistant (Llama 3.2)
2. **Label**: Optionally mark your mood (positive/negative)
3. **Train**: Federated learning trains a mood classifier on your labeled data
4. **Adapt**: The AI automatically detects your mood and tailors responses

**Key Feature:** All processing happens **on your device** - your conversations never leave your computer!

---

## ⚡ Quick Start

### Prerequisites
- Python 3.11 or 3.12
- macOS (for MLX), Linux, or Windows
- ~4GB RAM, ~2GB disk space

### 3-Step Setup

```bash
# 1. Setup environment
./setup_env.sh
source .venv311/bin/activate

# 2. Start chatting & label 10-20 messages
./run_portal.sh
# → Visit http://localhost:5001

# 3. Train FL model on your data
USE_CHAT_DATA=1 ./run.sh
# → Wait 2-3 minutes

# 4. Restart portal - now with mood detection!
./run_portal.sh
```

**That's it!** Your chat responses now adapt based on your mood 😊😔

---

## 📊 Documentation

| Document | Purpose | For Who |
|----------|---------|---------|
| **[FLOWCHARTS.md](FLOWCHARTS.md)** | **Visual diagrams & how to run** | Everyone (START HERE) |
| [QUICK_START.md](QUICK_START.md) | 5-minute getting started guide | New users |
| [FL_INTEGRATION_README.md](FL_INTEGRATION_README.md) | Technical deep-dive | Developers |
| [INTEGRATION_DIAGRAM.txt](INTEGRATION_DIAGRAM.txt) | ASCII architecture diagram | Visual learners |
| [CHANGES.md](CHANGES.md) | What was changed/added | Contributors |

**👉 Start with [FLOWCHARTS.md](FLOWCHARTS.md) for visual step-by-step instructions!**

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    YOUR DEVICE                            │
│  (All processing happens here - nothing leaves)           │
└──────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
  ┌──────────┐      ┌──────────┐     ┌──────────┐
  │  Portal  │      │ FL Train │     │ Storage  │
  │  (Chat)  │      │ (Model)  │     │ (Local)  │
  └──────────┘      └──────────┘     └──────────┘
  │                 │                │
  │ Llama 3.2      │ DistilBERT    │ chat.jsonl
  │ MLX            │ + LoRA        │ checkpoints/
  │ Mood detect    │ + Opacus DP   │
  └────────────────┴───────────────┴──────────────┘
```

### Models Used

1. **Llama 3.2 1B** (4-bit quantized) - Main chat model
2. **DistilBERT + LoRA** - Mood classifier (trained via FL)
3. **Whisper tiny.en** - Voice input (optional)

### How It Works

```
User labels mood → FL training → Model checkpoint
                                      ↓
Portal loads checkpoint → Predicts mood → Adapts response
```

**See [FLOWCHARTS.md](FLOWCHARTS.md) for detailed visual flows!**

---

## 🎓 Example

### Before FL Training
```
User: I'm feeling lonely
Assistant: I hear you. What's on your mind?
```

### After FL Training
```
User: I'm feeling lonely
[😔 negative (87%)]
Assistant: It sounds like you're going through a really 
          tough time right now. I'm here for you, and 
          you're not alone in this. Would you like to 
          talk about what's making you feel this way?
```

The response is more empathetic and validating because the AI detected negative mood!

---

## 🔒 Privacy & Security

✅ **What stays on your device:**
- All chat conversations
- Your mood labels
- Raw text data
- Personal information

✅ **What gets shared (during FL training):**
- Only model weight updates (numbers)
- Aggregated and noised with differential privacy
- No raw text, ever

✅ **Additional protection:**
- Differential privacy (Opacus)
- Federated averaging (no single client dominates)
- Local-only processing
- No external API calls

---

## 🛠️ Project Structure

```
fl-mentalhealth/
├── 📚 Documentation
│   ├── README.md              ← You are here
│   ├── FLOWCHARTS.md          ← Visual diagrams (START HERE)
│   ├── QUICK_START.md         ← 5-min getting started
│   ├── FL_INTEGRATION_README.md ← Technical details
│   └── INTEGRATION_DIAGRAM.txt  ← ASCII architecture
│
├── 🚀 Running Scripts
│   ├── run_portal.sh          ← Start chat portal
│   ├── run.sh                 ← Run FL training
│   └── test_integration.sh    ← Test everything
│
├── 💻 Source Code
│   ├── portal/app.py          ← Flask app + mood detection
│   ├── server/server.py       ← FL server (Flower)
│   ├── client/client.py       ← FL client (trains model)
│   ├── models/lora_model.py   ← Model architecture
│   └── utils/dataset.py       ← Data loading
│
├── 💾 Data & Models
│   ├── data/chat.jsonl        ← Your conversations
│   └── checkpoints/           ← Trained FL models
│
└── ⚙️ Configuration
    ├── requirements.txt       ← Python dependencies
    └── .gitignore            ← Excludes data/checkpoints
```

---

## 🧪 Testing

### Automated Test
```bash
./test_integration.sh
```

### Manual API Test
```bash
# Start portal
./run_portal.sh

# Test mood prediction (in another terminal)
curl -X POST http://localhost:5001/predict_mood \
  -H "Content-Type: application/json" \
  -d '{"text": "I am feeling sad"}'

# Response: {"label": 0, "confidence": 0.87, "mood": "negative"}
```

---

## 📊 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.11 | 3.11 or 3.12 |
| **RAM** | 2GB | 4GB |
| **Disk** | 1.5GB | 2GB |
| **OS** | macOS/Linux/Windows | macOS (for MLX) |

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| No mood predictions | Run FL training first: `USE_CHAT_DATA=1 ./run.sh` |
| "Dataset is empty" | Label 10+ messages in the portal first |
| Port 5001 in use | Change port: `PORTAL_PORT=5002 ./run_portal.sh` |
| Python version error | Use Python 3.11 or 3.12 (not 3.13) |

**See [FLOWCHARTS.md](FLOWCHARTS.md#troubleshooting-flowchart) for troubleshooting flowchart**

---

## 🔬 Technical Stack

- **Federated Learning**: Flower 1.9.0
- **LLM**: Llama 3.2 1B (MLX, 4-bit quantized)
- **Classifier**: DistilBERT-base-uncased
- **Adapter**: LoRA (r=8, alpha=16)
- **Privacy**: Opacus (differential privacy)
- **Backend**: Flask 3.0.3
- **Frontend**: Vanilla HTML/CSS/JS

---

## 📈 Performance

- **Mood prediction**: ~50-100ms (GPU/MPS), ~200ms (CPU)
- **Chat response**: ~1-2 seconds
- **FL training**: 2-3 minutes (3 rounds, 2 clients)
- **Model size**: ~2MB (LoRA adapters only)
- **Memory**: ~200MB additional for FL model

---

## 🎯 Use Cases

- Personal mental health journaling with mood tracking
- Research on privacy-preserving mental health ML
- Learning federated learning and differential privacy
- Building empathetic conversational AI
- Mood-aware chatbot development

---

## 🤝 Contributing

This is a research/educational project. Feel free to:
- Open issues for bugs or questions
- Submit PRs for improvements
- Fork for your own projects
- Share feedback on the FL integration

---

## 📝 License

This project is for educational and research purposes. Check individual model licenses:
- Llama 3.2: Meta license
- DistilBERT: Apache 2.0
- Flower: Apache 2.0

---

## 🙏 Acknowledgments

- **Flower**: Federated learning framework
- **MLX**: Apple Silicon ML acceleration
- **Hugging Face**: Model hosting and transformers
- **Meta**: Llama 3.2 model
- **Opacus**: Differential privacy library

---

## 🔗 Quick Links

| Link | Description |
|------|-------------|
| [FLOWCHARTS.md](FLOWCHARTS.md) | **Visual diagrams (START HERE)** |
| [QUICK_START.md](QUICK_START.md) | 3-step quickstart guide |
| [FL_INTEGRATION_README.md](FL_INTEGRATION_README.md) | Technical documentation |
| [INTEGRATION_DIAGRAM.txt](INTEGRATION_DIAGRAM.txt) | ASCII architecture |

---

## ⚡ TL;DR

```bash
# 1. Setup
./setup_env.sh && source .venv311/bin/activate

# 2. Chat & label moods
./run_portal.sh  # Visit http://localhost:5001

# 3. Train model
USE_CHAT_DATA=1 ./run.sh

# 4. Use mood-aware chat
./run_portal.sh

# Done! 🎉
```

**For visual step-by-step instructions, see [FLOWCHARTS.md](FLOWCHARTS.md)** 📊

---

Made with 💚 for privacy-preserving mental health support
