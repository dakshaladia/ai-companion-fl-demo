# Quick Start: FL-Powered Mental Health Portal

## 🚀 One-Line Summary
Train a federated mood classifier, then let it automatically improve chat responses based on detected user mood.

## 📊 Visual Flowcharts
**For detailed visual flowcharts and diagrams, see:** [`FLOWCHARTS.md`](FLOWCHARTS.md)
- Data collection flow
- FL training flow  
- Mood-aware response flow
- Complete end-to-end flow
- How to run (step-by-step)

## 📦 Setup

```bash
# Setup environment (Python 3.11 or 3.12)
./setup_env.sh

# Or manually:
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements.txt
```

## 🎯 Quick Test (3 steps)

### 1. Label some chats
```bash
./run_portal.sh
# Visit http://localhost:5001
# Chat and label your mood (dropdown: 0=down, 1=okay)
# Label at least 10 messages (5 of each)
```

### 2. Train FL model
```bash
# In a new terminal:
USE_CHAT_DATA=1 ./run.sh
# Wait for 3 rounds (~2-3 minutes)
# Model saved to: checkpoints/fl_mood_classifier.pt
```

### 3. Use mood-aware portal
```bash
./run_portal.sh
# Chat again - now with mood detection! 😊😔
# Responses adapt based on your mood
```

## 🧪 Test Integration

```bash
./test_integration.sh
# This will:
# - Create test data
# - Test mood prediction API
# - Verify mood-aware responses
```

## 📊 What You'll See

Before FL training:
```
User: I'm feeling lonely
Assistant: I hear you. What's on your mind?
```

After FL training:
```
User: I'm feeling lonely
[😔 negative (87%)]
Assistant: It sounds like you're going through a really tough time...
          [tailored for negative mood]
```

## 🔧 Key Files

- `server/server.py` - Saves FL checkpoints after training
- `portal/app.py` - Loads checkpoint & uses for mood prediction
- `checkpoints/` - FL model checkpoints (created during training)
- `data/chat.jsonl` - Chat logs with labels

## 💡 How It Works

```
User labels mood → FL training → Model checkpoint
                                      ↓
Portal loads checkpoint → Predicts mood → Adapts response
```

## 🎓 Learn More

- Full documentation: `FL_INTEGRATION_README.md`
- FL basics: Check Flower docs at https://flower.dev
- Model architecture: DistilBERT + LoRA + Opacus (differential privacy)

## ⚠️ Troubleshooting

**"FL model checkpoint not found"**
→ Run FL training first: `USE_CHAT_DATA=1 ./run.sh`

**"ChatLogDataset is empty"**
→ Label some messages in the portal first

**"Port 5001 in use"**
→ Change port: `PORTAL_PORT=5002 ./run_portal.sh`

**Python version issues**
→ MLX requires Python 3.11 or 3.12 (not 3.13)

