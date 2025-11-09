# Federated Learning Integration with Portal

This document explains how federated learning is integrated into the mental health portal.

## 📊 Visual Flowcharts
**For interactive flowcharts and step-by-step diagrams, see:** [`FLOWCHARTS.md`](FLOWCHARTS.md)

## Overview

The system now has a complete feedback loop where:
1. Users chat with the portal and optionally label their mood
2. FL clients train a mood classifier on labeled chat data
3. The trained model is saved as a checkpoint
4. The portal loads the checkpoint and uses it to predict user mood
5. Mood predictions improve response quality through mood-aware prompts

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    COMPLETE FLOW                             │
└─────────────────────────────────────────────────────────────┘

1. User chats → Portal saves to data/chat.jsonl
                ↓
2. FL Training (run.sh)
   ├─ Server aggregates model updates
   ├─ Clients train on labeled data
   └─ Saves checkpoint: checkpoints/fl_mood_classifier.pt
                ↓
3. Portal loads checkpoint
   ├─ Predicts mood for each message
   ├─ Adapts prompts based on mood
   └─ Shows mood with emoji (😊 positive / 😔 negative)
```

## Components

### 1. Server (`server/server.py`)
- **SaveModelStrategy**: Custom Flower strategy that saves model after each round
- **save_model_checkpoint()**: Saves aggregated weights to PyTorch checkpoint
- Saves to: `checkpoints/fl_mood_classifier.pt` and `checkpoints/fl_mood_classifier_roundN.pt`

### 2. Portal (`portal/app.py`)
- **_load_fl_model()**: Loads trained FL model from checkpoint
- **predict_mood(text)**: Predicts mood (0=negative, 1=positive) with confidence
- **generate_response()**: Uses mood prediction to adapt prompts:
  - Negative mood → Extra empathetic and validating
  - Positive mood → Encouraging and building on positivity
- **/predict_mood** endpoint: Standalone API for mood prediction
- UI shows mood predictions with emoji icons

### 3. Client (`client/client.py`)
- Trains DistilBERT with LoRA adapters
- Uses Opacus for differential privacy
- Can train on labeled chat data with `USE_CHAT_DATA=1`

## Usage

### Step 1: Generate labeled training data
```bash
# Start the portal
./run_portal.sh

# Chat and label your mood (use the label dropdown)
# Visit http://localhost:5001
# Label some messages as:
#   0 = Feeling down
#   1 = Feeling okay
```

### Step 2: Run federated learning
```bash
# Train FL model on labeled data
USE_CHAT_DATA=1 ./run.sh

# This will:
# - Start FL server
# - Train 2 clients for 3 rounds
# - Save checkpoint to checkpoints/fl_mood_classifier.pt
```

### Step 3: Use mood-aware portal
```bash
# Restart portal to load FL model
./run_portal.sh

# Now your chats will:
# - Show mood predictions with emoji
# - Receive mood-adapted responses
# - Be logged with predicted_mood in chat.jsonl
```

## API Endpoints

### POST /send
Send a message and get a response with mood prediction.

**Request:**
```json
{
  "text": "I'm feeling lonely today",
  "label": null  // optional: 0 or 1
}
```

**Response:**
```json
{
  "reply": "I hear you...",
  "mood": {
    "label": 0,
    "confidence": 0.87,
    "mood": "negative"
  }
}
```

### POST /predict_mood
Get mood prediction without generating a response.

**Request:**
```json
{
  "text": "I'm feeling great!"
}
```

**Response:**
```json
{
  "label": 1,
  "confidence": 0.92,
  "mood": "positive"
}
```

## Configuration

- `CHECKPOINT_DIR`: Directory for FL checkpoints (default: `checkpoints/`)
- `FL_CHECKPOINT_PATH`: Path to FL model (default: `checkpoints/fl_mood_classifier.pt`)
- `USE_CHAT_DATA=1`: Train FL clients on real chat data instead of synthetic data

## Model Details

- **Architecture**: DistilBERT-base-uncased with LoRA adapters
- **Task**: Binary sequence classification (negative/positive mood)
- **LoRA config**:
  - r=8, alpha=16
  - Target modules: q_lin, v_lin (attention layers)
  - Dropout: 0.05
- **Privacy**: Differential privacy via Opacus
  - Noise multiplier: 1.0 (configurable)
  - Max grad norm: 1.0 (configurable)

## Benefits

1. **Privacy-preserving**: Only model updates shared, never raw text
2. **Personalized**: Adapts to your mood patterns over time
3. **Transparent**: Shows mood predictions with confidence scores
4. **Lightweight**: LoRA adapters are small (~1-2MB) compared to full model
5. **Continuous improvement**: Model gets better as you label more data

## Limitations

- Requires labeled data for training (at least 10-20 labeled messages)
- Binary classification only (negative/positive)
- Mood prediction confidence threshold: 60% (lower confidence uses standard prompt)
- FL training requires at least 2 clients

## Future Enhancements

- [ ] Multi-class mood classification (sad, anxious, happy, neutral, etc.)
- [ ] Automatic mood labeling using sentiment analysis
- [ ] Personalized models per user (federated personalization)
- [ ] Active learning: suggest which messages to label
- [ ] Mood tracking dashboard

