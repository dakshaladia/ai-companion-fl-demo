# FL Mental Health Portal - Flowcharts

## Table of Contents
1. [System Overview](#system-overview)
2. [Data Collection Flow](#data-collection-flow)
3. [Federated Learning Training Flow](#federated-learning-training-flow)
4. [Mood-Aware Response Flow](#mood-aware-response-flow)
5. [Complete End-to-End Flow](#complete-end-to-end-flow)
6. [How to Run](#how-to-run)

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    FL MENTAL HEALTH PORTAL                       │
│                      System Architecture                         │
└─────────────────────────────────────────────────────────────────┘

             ┌──────────────────────────────────┐
             │         USER DEVICE              │
             │  (All processing happens here)   │
             └──────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Portal    │    │ FL Training │    │   Storage   │
│  (Flask)    │    │  (Flower)   │    │   (Local)   │
└─────────────┘    └─────────────┘    └─────────────┘
│                  │                  │
│ • Llama 3.2     │ • DistilBERT    │ • chat.jsonl
│ • MLX           │ • LoRA          │ • checkpoints/
│ • Mood detect   │ • Opacus DP     │
│                 │ • FedAvg        │
└─────────────────┴─────────────────┴─────────────────┘

         Privacy: All data stays on-device
         Communication: Model updates only (no raw text)
```

---

## Data Collection Flow

```
┌─────────────────────────────────────────────────────────────────┐
│              PHASE 1: DATA COLLECTION FLOW                       │
│              (How chat data gets labeled)                        │
└─────────────────────────────────────────────────────────────────┘

START: User opens portal (http://localhost:5001)
  │
  ├─→ 1. User types message
  │    ↓
  ├─→ 2. User selects mood label (optional)
  │    │
  │    ├─ [Feeling down (0)]  → Negative
  │    ├─ [Feeling okay (1)]  → Positive
  │    └─ [No label]          → Unlabeled
  │    ↓
  ├─→ 3. User clicks "Send"
  │    ↓
  ├─→ 4. Portal receives message
  │    ↓
  ├─→ 5. Llama 3.2 generates response
  │    ↓
  ├─→ 6. Save to data/chat.jsonl
  │    │
  │    ├─ User message + label
  │    └─ Assistant response
  │    ↓
  └─→ 7. Display response to user
       ↓
  LOOP: User continues chatting

  ┌──────────────────────────────────┐
  │ data/chat.jsonl (Example)        │
  ├──────────────────────────────────┤
  │ {"role": "user",                 │
  │  "text": "I'm feeling lonely",   │
  │  "label": 0,                     │
  │  "timestamp": "..."}             │
  │                                  │
  │ {"role": "assistant",            │
  │  "text": "I hear you...",        │
  │  "timestamp": "..."}             │
  └──────────────────────────────────┘

RESULT: Chat history with labeled mood data
        Ready for FL training!
```

---

## Federated Learning Training Flow

```
┌─────────────────────────────────────────────────────────────────┐
│         PHASE 2: FEDERATED LEARNING TRAINING FLOW                │
│         (How the mood classifier is trained)                     │
└─────────────────────────────────────────────────────────────────┘

START: Run `USE_CHAT_DATA=1 ./run.sh`
  │
  ├─→ 1. FL Server starts (port 8080)
  │    │
  │    ├─ Initialize FedAvg strategy
  │    ├─ Wait for 2 clients
  │    └─ Prepare for 3 rounds
  │    ↓
  ├─→ 2. FL Client 1 & 2 start
  │    │
  │    ├─ Load DistilBERT + LoRA
  │    ├─ Attach Opacus (DP)
  │    ├─ Read data/chat.jsonl
  │    │   │
  │    │   ├─ Filter: Only labeled messages
  │    │   ├─ Filter: Only user messages
  │    │   └─ Tokenize with DistilBERT
  │    │
  │    └─ Connect to server
  │    ↓
  ├─→ 3. Training Round 1 begins
  │    │
  │    ┌─────────────────────────┐
  │    │   FOR EACH ROUND:       │
  │    └─────────────────────────┘
  │    │
  │    ├─→ a) Server sends global model to clients
  │    │    ↓
  │    ├─→ b) Each client trains locally
  │    │    │
  │    │    ├─ Forward pass (predict mood)
  │    │    ├─ Calculate loss
  │    │    ├─ Backward pass (gradients)
  │    │    ├─ Add DP noise (Opacus)
  │    │    └─ Update local model
  │    │    ↓
  │    ├─→ c) Clients send updates to server
  │    │    ↓
  │    ├─→ d) Server aggregates (FedAvg)
  │    │    │
  │    │    ├─ Average client weights
  │    │    └─ Create new global model
  │    │    ↓
  │    ├─→ e) Server saves checkpoint
  │    │    │
  │    │    ├─ checkpoints/fl_mood_classifier_round1.pt
  │    │    └─ checkpoints/fl_mood_classifier.pt (latest)
  │    │    ↓
  │    └─→ f) Repeat for rounds 2 & 3
  │         ↓
  └─→ 4. Training complete
       │
       ├─ Final model saved
       ├─ Clients disconnect
       └─ Server stops

  ┌──────────────────────────────────────┐
  │ Model Weights Aggregation (FedAvg)   │
  ├──────────────────────────────────────┤
  │                                      │
  │  Client 1 Weights  ┐                │
  │                    ├─→ Average ─→    │
  │  Client 2 Weights  ┘                │
  │                                      │
  │  = New Global Weights                │
  │  (with DP noise added)               │
  └──────────────────────────────────────┘

RESULT: Trained mood classifier saved to:
        checkpoints/fl_mood_classifier.pt
```

---

## Mood-Aware Response Flow

```
┌─────────────────────────────────────────────────────────────────┐
│        PHASE 3: MOOD-AWARE RESPONSE GENERATION FLOW              │
│        (How FL model improves chat responses)                    │
└─────────────────────────────────────────────────────────────────┘

START: User sends message in portal
  │
  ├─→ 1. Portal receives message
  │    ↓
  ├─→ 2. Load FL model (if available)
  │    │
  │    ├─ Check: checkpoints/fl_mood_classifier.pt exists?
  │    │
  │    ├─ YES: Load DistilBERT + LoRA checkpoint
  │    │   ↓
  │    │   Go to step 3
  │    │
  │    └─ NO: Skip mood prediction
  │        ↓
  │        Go to step 6 (standard prompt)
  │    ↓
  ├─→ 3. Predict mood
  │    │
  │    ├─→ a) Tokenize message (DistilBERT)
  │    │    ↓
  │    ├─→ b) Run inference (forward pass)
  │    │    ↓
  │    ├─→ c) Get prediction
  │    │    │
  │    │    ├─ Label: 0 or 1
  │    │    ├─ Confidence: 0.0 - 1.0
  │    │    └─ Mood: "negative" or "positive"
  │    │    ↓
  │    └─→ d) Check confidence
  │         │
  │         ├─ IF confidence > 60%
  │         │   Go to step 4
  │         │
  │         └─ ELSE
  │             Go to step 6 (standard prompt)
  │    ↓
  ├─→ 4. Build mood-aware prompt
  │    │
  │    ├─ IF mood == "negative":
  │    │   Prompt: "User is struggling. Be extra empathetic
  │    │            and validating. Offer gentle support..."
  │    │
  │    └─ IF mood == "positive":
  │        Prompt: "User is in better mood. Be encouraging
  │                 and help build on positive feelings..."
  │    ↓
  ├─→ 5. Generate response (Llama 3.2)
  │    │
  │    ├─ Use mood-aware prompt
  │    ├─ MLX inference
  │    └─ Generate empathetic response
  │    ↓
  │    Go to step 7
  │
  ├─→ 6. Standard response (fallback)
  │    │
  │    ├─ Use generic prompt
  │    ├─ MLX inference
  │    └─ Generate standard response
  │    ↓
  ├─→ 7. Return response to user
  │    │
  │    ├─ Response text
  │    ├─ Mood prediction (if available)
  │    │   │
  │    │   ├─ Show emoji: 😔 or 😊
  │    │   └─ Show confidence: "87%"
  │    │
  │    └─ Log to chat.jsonl
  │    ↓
  └─→ END

  ┌──────────────────────────────────────────┐
  │ Example Output:                          │
  ├──────────────────────────────────────────┤
  │                                          │
  │ User: "I'm feeling really lonely"        │
  │ [😔 negative (87%)]                      │
  │                                          │
  │ Assistant: "I hear you. It sounds like   │
  │ you're going through a really tough      │
  │ time right now. I'm here for you, and    │
  │ you're not alone in this..."             │
  │ [Tailored for negative mood]             │
  │                                          │
  └──────────────────────────────────────────┘

RESULT: User receives mood-aware, empathetic response
        that adapts to their emotional state
```

---

## Complete End-to-End Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                  COMPLETE END-TO-END FLOW                        │
│              (All phases working together)                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ DAY 1: COLLECT DATA                                              │
└─────────────────────────────────────────────────────────────────┘

  User → Portal → Label mood → Save to chat.jsonl
                                     ↓
                            [10-20 labeled messages]

┌─────────────────────────────────────────────────────────────────┐
│ DAY 2: TRAIN MODEL                                               │
└─────────────────────────────────────────────────────────────────┘

  Run: USE_CHAT_DATA=1 ./run.sh
                ↓
       FL Server + 2 Clients
                ↓
       3 Training Rounds (2-3 mins)
                ↓
  Save: checkpoints/fl_mood_classifier.pt

┌─────────────────────────────────────────────────────────────────┐
│ DAY 3+: USE MOOD-AWARE PORTAL                                    │
└─────────────────────────────────────────────────────────────────┘

  User → Portal → Predict mood → Adapt response → Display
                       ↓                              ↓
              [Uses FL model]              [Better responses!]
                       ↓                              ↓
               [Save new data]              [Continue training]
                       
                       
┌─────────────────────────────────────────────────────────────────┐
│                    CONTINUOUS IMPROVEMENT                         │
└─────────────────────────────────────────────────────────────────┘

    More labels → Better model → Better responses
         ↑                                ↓
         └────────────────────────────────┘
              (Feedback loop)
```

---

## How to Run

### 🚀 Quick Start (3 Steps)

#### **Step 1: Collect Labeled Data** (5-10 minutes)

```bash
# Start the portal
./run_portal.sh

# Visit http://localhost:5001 in your browser

# Chat and label your mood:
# - Type: "I'm feeling overwhelmed today"
# - Select: "Feeling down (label 0)"
# - Click: "Send"
# 
# Repeat for 10-20 messages (mix of both moods)
```

**What happens:**
- Portal saves to `data/chat.jsonl`
- Each message stored with your mood label
- Ready for training!

---

#### **Step 2: Train FL Model** (2-3 minutes)

```bash
# Stop the portal (Ctrl+C)

# Train the model on your labeled data
USE_CHAT_DATA=1 ./run.sh

# Wait for output:
# "🚀 Starting Flower server..."
# "🤝 Starting Flower client..."
# "📊 Round 1 complete - model saved"
# "📊 Round 2 complete - model saved"
# "📊 Round 3 complete - model saved"
# "✅ Server finished training rounds"
# "💾 Final model saved to checkpoints/fl_mood_classifier.pt"
```

**What happens:**
- FL server starts on port 8080
- 2 clients connect and train
- 3 rounds of federated averaging
- Model saved to `checkpoints/`

---

#### **Step 3: Use Mood-Aware Portal** (Ongoing)

```bash
# Restart the portal
./run_portal.sh

# Visit http://localhost:5001

# Chat as normal - now with mood detection!
# - Your messages will show mood emoji (😊 or 😔)
# - Responses adapt based on your mood
# - More empathetic for negative mood
# - More encouraging for positive mood
```

**What happens:**
- Portal loads FL model on startup
- Predicts your mood for each message
- Adapts responses based on mood
- Shows mood with confidence %

---

### 🧪 Test Integration

```bash
# Automated test
./test_integration.sh

# This will:
# 1. Create sample labeled data
# 2. Check if FL model exists
# 3. Test mood prediction API
# 4. Verify mood-aware responses
```

---

### 📊 Manual Testing

#### Test Mood Prediction API

```bash
# Start portal
./run_portal.sh

# In another terminal:

# Test negative mood
curl -X POST http://localhost:5001/predict_mood \
  -H "Content-Type: application/json" \
  -d '{"text": "I am feeling so sad and lonely"}'

# Expected output:
# {
#   "label": 0,
#   "confidence": 0.87,
#   "mood": "negative"
# }

# Test positive mood
curl -X POST http://localhost:5001/predict_mood \
  -H "Content-Type: application/json" \
  -d '{"text": "I am feeling great and happy"}'

# Expected output:
# {
#   "label": 1,
#   "confidence": 0.92,
#   "mood": "positive"
# }
```

#### Test Chat with Mood

```bash
curl -X POST http://localhost:5001/send \
  -H "Content-Type: application/json" \
  -d '{"text": "I am feeling overwhelmed"}'

# Expected output:
# {
#   "reply": "I hear you... [empathetic response]",
#   "mood": {
#     "label": 0,
#     "confidence": 0.85,
#     "mood": "negative"
#   }
# }
```

---

### 🔧 Troubleshooting Flowchart

```
Problem: No mood predictions showing?
  │
  ├─→ Check: Does checkpoints/fl_mood_classifier.pt exist?
  │    │
  │    ├─ NO: Run FL training first
  │    │      → USE_CHAT_DATA=1 ./run.sh
  │    │
  │    └─ YES: Check portal logs for errors
  │           → Look for "✅ FL mood classifier loaded"
  │
Problem: "ChatLogDataset is empty"
  │
  ├─→ Check: Do you have labeled messages?
  │    │
  │    ├─ Run: cat data/chat.jsonl | grep '"label":'
  │    │
  │    ├─ If empty: Label messages in portal first
  │    │
  │    └─ Need at least 5-10 labeled messages
  │
Problem: Low confidence predictions?
  │
  ├─→ More training data needed
  │    │
  │    ├─ Label 20+ messages (10+ per mood)
  │    ├─ Retrain: USE_CHAT_DATA=1 ./run.sh
  │    └─ Test again
  │
Problem: Port already in use?
  │
  └─→ Change port: PORTAL_PORT=5002 ./run_portal.sh
```

---

### 📁 Directory Structure

```
fl-mentalhealth/
├── portal/
│   └── app.py              ← Portal + FL integration
├── server/
│   └── server.py           ← FL server (saves checkpoints)
├── client/
│   └── client.py           ← FL client (trains model)
├── models/
│   └── lora_model.py       ← DistilBERT + LoRA definition
├── utils/
│   └── dataset.py          ← ChatLogDataset + ToyDataset
├── data/
│   └── chat.jsonl          ← Labeled chat history
├── checkpoints/            ← FL model checkpoints (created)
│   ├── fl_mood_classifier.pt
│   ├── fl_mood_classifier_round1.pt
│   ├── fl_mood_classifier_round2.pt
│   └── fl_mood_classifier_round3.pt
├── run_portal.sh           ← Start portal
├── run.sh                  ← Run FL training
└── test_integration.sh     ← Test everything
```

---

### ⏱️ Timeline

| Task | Time | Command |
|------|------|---------|
| Label data | 5-10 min | `./run_portal.sh` (chat & label) |
| Train model | 2-3 min | `USE_CHAT_DATA=1 ./run.sh` |
| Use portal | Ongoing | `./run_portal.sh` |
| Total setup | **~10 min** | One-time |

---

## Summary

**The flowcharts show:**
1. How data collection works (label moods in portal)
2. How FL training works (distributed model training)
3. How mood-aware responses work (better chat experience)
4. How to run everything (step-by-step commands)

**Key insight:** The three phases work together in a continuous loop:
- More labels → Better model → Better responses → More engagement → More labels

**Privacy:** All processing happens on your device. Only model weights (not data) are aggregated during FL training.

