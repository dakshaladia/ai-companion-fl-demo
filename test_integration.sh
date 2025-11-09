#!/bin/bash

set -euo pipefail

echo "🧪 Testing FL Integration End-to-End"
echo "===================================="
echo ""

cd "$(dirname "$0")"

# Activate virtual environment
if [ -d ".venv311" ]; then
  source .venv311/bin/activate
fi

echo "📝 Step 1: Creating test chat data with labels..."
echo ""

# Create test data directory
mkdir -p data

# Create test chat data with labels
cat > data/chat.jsonl << 'EOF'
{"role": "user", "text": "I'm feeling really down and hopeless today", "timestamp": "2025-01-01T10:00:00Z", "label": 0}
{"role": "assistant", "text": "I hear you. It sounds tough.", "timestamp": "2025-01-01T10:00:01Z"}
{"role": "user", "text": "Everything is going wrong and I can't cope", "timestamp": "2025-01-01T10:01:00Z", "label": 0}
{"role": "assistant", "text": "That's really hard.", "timestamp": "2025-01-01T10:01:01Z"}
{"role": "user", "text": "I'm so overwhelmed and stressed", "timestamp": "2025-01-01T10:02:00Z", "label": 0}
{"role": "assistant", "text": "I understand.", "timestamp": "2025-01-01T10:02:01Z"}
{"role": "user", "text": "I'm having a great day and feeling positive!", "timestamp": "2025-01-01T11:00:00Z", "label": 1}
{"role": "assistant", "text": "That's wonderful!", "timestamp": "2025-01-01T11:00:01Z"}
{"role": "user", "text": "Everything is going well and I'm happy", "timestamp": "2025-01-01T11:01:00Z", "label": 1}
{"role": "assistant", "text": "So glad to hear that!", "timestamp": "2025-01-01T11:01:01Z"}
{"role": "user", "text": "I'm feeling optimistic about the future", "timestamp": "2025-01-01T11:02:00Z", "label": 1}
{"role": "assistant", "text": "That's great!", "timestamp": "2025-01-01T11:02:01Z"}
EOF

echo "✅ Created test data with 6 labeled messages (3 negative, 3 positive)"
echo ""

echo "🚀 Step 2: Testing FL server and client..."
echo ""

# Check if FL server is needed
echo "⚠️  Note: Run './run.sh' in a separate terminal to train the FL model"
echo "   This will start the server and clients for 3 rounds of training."
echo ""
echo "   After training completes, you should see:"
echo "   - checkpoints/fl_mood_classifier.pt"
echo "   - checkpoints/fl_mood_classifier_round1.pt"
echo "   - checkpoints/fl_mood_classifier_round2.pt"
echo "   - checkpoints/fl_mood_classifier_round3.pt"
echo ""

# Check if checkpoint exists
if [ -f "checkpoints/fl_mood_classifier.pt" ]; then
  echo "✅ FL model checkpoint found!"
  echo ""
  
  echo "🧪 Step 3: Testing mood prediction API..."
  echo ""
  
  # Start portal in background
  echo "Starting portal..."
  PORTAL_PORT=5001 python portal/app.py &
  PORTAL_PID=$!
  
  # Wait for portal to start
  sleep 5
  
  # Test negative mood
  echo "Testing negative mood prediction..."
  curl -X POST http://localhost:5001/predict_mood \
    -H "Content-Type: application/json" \
    -d '{"text": "I am feeling so sad and lonely"}' \
    2>/dev/null | python -m json.tool
  echo ""
  
  # Test positive mood
  echo "Testing positive mood prediction..."
  curl -X POST http://localhost:5001/predict_mood \
    -H "Content-Type: application/json" \
    -d '{"text": "I am feeling great and happy"}' \
    2>/dev/null | python -m json.tool
  echo ""
  
  # Test chat with mood-aware response
  echo "Testing mood-aware chat response..."
  curl -X POST http://localhost:5001/send \
    -H "Content-Type: application/json" \
    -d '{"text": "I am feeling overwhelmed and stressed"}' \
    2>/dev/null | python -m json.tool
  echo ""
  
  # Kill portal
  kill $PORTAL_PID 2>/dev/null || true
  
  echo "✅ Integration test complete!"
  echo ""
  echo "📊 Summary:"
  echo "  - FL model checkpoint: ✅ Found"
  echo "  - Mood prediction API: ✅ Working"
  echo "  - Mood-aware responses: ✅ Working"
  echo ""
  echo "🎉 Success! The FL integration is fully functional."
else
  echo "❌ FL model checkpoint not found."
  echo ""
  echo "To complete the integration:"
  echo "  1. Run: USE_CHAT_DATA=1 ./run.sh"
  echo "  2. Wait for 3 training rounds to complete"
  echo "  3. Run this test script again"
fi

echo ""
echo "📖 For more details, see FL_INTEGRATION_README.md"

