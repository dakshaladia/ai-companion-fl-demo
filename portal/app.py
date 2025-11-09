import os
import sys
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
from flask import Flask, request, jsonify, Response
from mlx_lm import load as mlx_load, generate as mlx_generate
from mlx_lm.sample_utils import make_sampler, make_repetition_penalty
import torch
try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None  # Optional: STT disabled if not installed
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.lora_model import get_lora_model


APP_PORT = int(os.environ.get("PORTAL_PORT", 5000))
APP_HOST = os.environ.get("PORTAL_HOST", "127.0.0.1")  # bind to localhost only
MLX_MODEL_NAME = os.environ.get("PORTAL_MLX_MODEL", "mlx-community/Llama-3.2-1B-Instruct-4bit")
STT_MODEL_NAME = os.environ.get("PORTAL_STT_MODEL", "tiny.en")
COMPUTE_DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
PORTAL_MAX_TOKENS = int(os.environ.get("PORTAL_MAX_TOKENS", "160"))
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CHAT_LOG_PATH = os.path.join(DATA_DIR, "chat.jsonl")
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")
FL_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "fl_mood_classifier.pt")


def ensure_data_dir() -> None:
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)


def append_chat_event(event: Dict[str, Any]) -> None:
    ensure_data_dir()
    with open(CHAT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def read_recent_chats(limit: int = 50) -> List[Dict[str, Any]]:
    if not os.path.exists(CHAT_LOG_PATH):
        return []
    events: List[Dict[str, Any]] = []
    with open(CHAT_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events[-limit:]


_mlx_model = None
_mlx_tokenizer = None
_fl_model = None
_fl_tokenizer = None


def _load_mlx_model() -> None:
    global _mlx_model, _mlx_tokenizer
    if _mlx_model is not None and _mlx_tokenizer is not None:
        return
    # MLX API: load(model_name) -> (model, tokenizer)
    # Prefer slow tokenizers to avoid Rust tokenizers build on Python 3.13
    _mlx_model, _mlx_tokenizer = mlx_load(MLX_MODEL_NAME, tokenizer_config={"use_fast": True})


def _load_fl_model() -> None:
    """Load the federated learning mood classifier if available."""
    global _fl_model, _fl_tokenizer
    if _fl_model is not None and _fl_tokenizer is not None:
        return
    
    if not os.path.exists(FL_CHECKPOINT_PATH):
        logging.warning(f"FL model checkpoint not found at {FL_CHECKPOINT_PATH}")
        return
    
    try:
        # Load the model architecture
        _fl_model = get_lora_model()
        _fl_model.to(COMPUTE_DEVICE)
        
        # Load the checkpoint
        checkpoint = torch.load(FL_CHECKPOINT_PATH, map_location=COMPUTE_DEVICE)
        _fl_model.load_state_dict(checkpoint['model_state_dict'])
        _fl_model.eval()
        
        # Load tokenizer
        _fl_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        logging.info(f"✅ FL mood classifier loaded from {FL_CHECKPOINT_PATH}")
    except Exception as e:
        logging.exception(f"Failed to load FL model: {e}")
        _fl_model = None
        _fl_tokenizer = None


def predict_mood(text: str) -> Optional[Dict[str, Any]]:
    """
    Predict mood using the federated learning model.
    
    Returns:
        dict with 'label' (0=negative, 1=positive) and 'confidence' (0-1)
        or None if model is not available
    """
    _load_fl_model()
    
    if _fl_model is None or _fl_tokenizer is None:
        return None
    
    try:
        # Tokenize input
        inputs = _fl_tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        inputs = {k: v.to(COMPUTE_DEVICE) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = _fl_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predicted_label = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, predicted_label].item()
        
        return {
            'label': predicted_label,
            'confidence': confidence,
            'mood': 'positive' if predicted_label == 1 else 'negative'
        }
    except Exception as e:
        logging.exception(f"Mood prediction failed: {e}")
        return None


_stt_model: Optional[WhisperModel] = None


def _load_stt_model() -> None:
    global _stt_model
    if WhisperModel is None:
        return
    if _stt_model is not None:
        return
    # Compute type int8 for CPU efficiency
    compute_type = "int8" if COMPUTE_DEVICE == "cpu" else "float16"
    _stt_model = WhisperModel(STT_MODEL_NAME, device=COMPUTE_DEVICE, compute_type=compute_type)


def _load_tf_model() -> None:
    global _tf_model, _tf_tokenizer
    if _tf_model is not None and _tf_tokenizer is not None:
        return
    model_name = os.environ.get("PORTAL_TF_MODEL", "distilgpt2")
    _tf_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    _tf_model = AutoModelForCausalLM.from_pretrained(model_name)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    _tf_model.to(device)
    _tf_model.eval()


MAX_CONTEXT_TURNS = int(os.environ.get("PORTAL_MAX_CONTEXT_TURNS", "2"))


def _build_mlx_prompt(latest_user_text: str, max_turns: int = MAX_CONTEXT_TURNS) -> str:
    """Build a prompt with conversation context for the mental health assistant."""
    
    # Get recent conversation history
    recent_chats = read_recent_chats(limit=3)
    
    # Build conversation history (last N turns, excluding current message)
    conversation_history = []
    for event in recent_chats[-(max_turns * 2):]:  # Get last N turns (user + assistant pairs)
        role = event.get("role", "")
        text = event.get("text", "").strip()
        if text:
            if role == "user":
                conversation_history.append(f"User: {text}")
            elif role == "assistant":
                conversation_history.append(f"Assistant: {text}")
    
    # Build the full prompt with context
    if conversation_history:
        context = "\n".join(conversation_history)
        prompt = f"""You are a compassionate mental health support assistant. Continue this conversation naturally. Respond with 1-2 short, empathetic sentences that directly address what the user just said. Be warm, understanding, and offer gentle suggestions when appropriate. Avoid medical advice.

Previous conversation:
{context}

Current message (respond to THIS):
User: {latest_user_text.strip()}
Assistant:"""
    else:
        # First message, no history
        prompt = f"""You are a compassionate mental health support assistant. Respond with 1-2 short, empathetic sentences to help the user feel heard and supported. Be warm, understanding, and offer gentle suggestions when appropriate. Avoid medical advice.

User: {latest_user_text.strip()}
Assistant:"""
    
    return prompt


def generate_response(user_text: str) -> str:
    text = user_text.strip()
    lower = text.lower()
    if any(k in lower for k in ["suicide", "kill myself", "end my life", "hurt myself"]):
        return (
            "I'm really sorry you're feeling this way. If you're in immediate danger, "
            "please contact your local emergency number now. You can also reach out "
            "to your regional crisis hotline. In the U.S., call or text 988 for the Suicide & "
            "Crisis Lifeline. I'm here to listen."
        )

    _load_mlx_model()
    
    # Predict mood using federated learning model
    mood_prediction = predict_mood(text)
    
    # Build prompt with conversation context
    prompt = _build_mlx_prompt(text)
    
    # If we have mood prediction with high confidence, inject mood context
    if mood_prediction and mood_prediction['confidence'] > 0.6:
        mood = mood_prediction['mood']
        confidence = mood_prediction['confidence']
        
        if mood == 'negative':
            mood_hint = "\n[Note: The user seems to be struggling. Be extra empathetic and validating.]"
        else:
            mood_hint = "\n[Note: The user seems to be in a better mood. Be encouraging and positive.]"
        
        # Insert mood hint before the final "Assistant:" line
        prompt = prompt.replace("Assistant:", f"{mood_hint}\nAssistant:")
        logging.info(f"Mood-aware response: {mood} (confidence: {confidence:.2f})")
    
    # Create sampler with the new API
    sampler = make_sampler(
        temp=0.25,
        top_p=0.75
    )
    
    # Create logits processors for repetition penalty
    logits_processors = [make_repetition_penalty(1.15)]
    
    # Generate response using the new MLX API
    reply = mlx_generate(
        _mlx_model,
        _mlx_tokenizer,
        prompt=prompt,
        max_tokens=PORTAL_MAX_TOKENS,
        sampler=sampler,
        logits_processors=logits_processors,
        verbose=False,
    ).strip()
    
    # Handle stop sequences manually since they're not in the new API
    if "\nUser:" in reply:
        reply = reply.split("\nUser:")[0].strip()
    
    # Remove quotes from the response - more comprehensive approach
    # Remove quotes at the beginning and end
    while reply.startswith('"') or reply.startswith("'"):
        reply = reply[1:].strip()
    while reply.endswith('"') or reply.endswith("'"):
        reply = reply[:-1].strip()
    
    # Also remove any escaped quotes that might be in the middle
    reply = reply.replace('\\"', '"').replace("\\'", "'")

    # Prefer returning a complete sentence: trim to last sentence-ending punctuation if needed
    if not reply.endswith((".", "!", "?")):
        last_period = max(reply.rfind("."), reply.rfind("!"), reply.rfind("?"))
        if last_period != -1:
            reply = reply[: last_period + 1].strip()

    if not reply:
        # More varied fallback responses
        fallback_responses = [
            "Thanks for sharing that with me. How are you feeling right now?",
            "I hear you. What's one small thing that might help you feel a bit better?",
            "That sounds really tough. What would be most helpful for you right now?",
            "I'm here to listen. What's on your mind?",
            "Thanks for opening up. How can I best support you today?"
        ]
        import random
        reply = random.choice(fallback_responses)
    return reply


app = Flask(__name__)


@app.get("/")
def index() -> Response:
    recent = read_recent_chats(50)
    
    # Build chat history HTML
    chat_history = ""
    for e in recent:
        role = e.get("role", "user")
        text = e.get("text", "")
        timestamp = e.get("timestamp", "")
        label = e.get("label")
        label_text = f" · label={label}" if label is not None else ""
        chat_history += f'<div class="msg {role}"><div>{text}</div><div class="meta">{timestamp}{label_text}</div></div>'
    
    # Build the HTML template properly to avoid f-string conflicts
    html_template = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>On-Device Mental Health Portal</title>
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }
      .chat { max-width: 720px; margin: 0 auto; }
      .msg { padding: 8px 12px; border-radius: 12px; margin: 6px 0; width: fit-content; max-width: 80%; }
      .user { background: #e6f0ff; align-self: flex-end; }
      .assistant { background: #f4f4f4; }
      .row { display: flex; flex-direction: column; }
      .meta { color: #666; font-size: 12px; margin-top: 2px; }
      .disclaimer { background: #fff9e6; border: 1px solid #ffe08a; padding: 12px; border-radius: 8px; margin-bottom: 16px; }
      .input-row { display: grid; grid-template-columns: 1fr 160px 100px 120px; gap: 8px; margin-top: 12px; }
      textarea { width: 100%; height: 80px; padding: 10px; }
      select, button { padding: 10px; }
    </style>
  </head>
  <body>
    <div class="chat">
      <div class="disclaimer">
        <strong>Privacy:</strong> Conversations stay on this device. Only anonymized model updates may be shared during federated learning rounds (never raw text). This is not a crisis service. If you're in immediate danger, call your local emergency number or, in the U.S., dial <strong>988</strong>.<br>
        <strong>Label Box:</strong> Optionally label your mood to help improve the AI's understanding of different emotional states (for federated learning).<br>
        <strong>Mood Detection:</strong> If available, a trained FL model will automatically detect your mood and tailor responses accordingly (shown with emoji icons).
      </div>
      <div id="log" class="row">
        CHAT_HISTORY_PLACEHOLDER
      </div>
      <div class="input-row">
        <textarea id="text" placeholder="Share what's on your mind..."></textarea>
        <select id="label" title="Optional: Label your mood to help improve the AI's understanding">
          <option value="">No label</option>
          <option value="0">Feeling down (label 0)</option>
          <option value="1">Feeling okay (label 1)</option>
        </select>
        <button onclick="send()">Send</button>
        # <button id="recordBtn" onclick="toggleRecord()">🎤 Start</button>
      </div>
    </div>
    <script>
      async function send() {
        const text = document.getElementById('text').value.trim();
        const label = document.getElementById('label').value;
        if (!text) return;
        const btn = event?.target || document.querySelector('button[onclick="send()"]');
        if (btn) btn.disabled = true;
        const res = await fetch('/send', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text, label: label === '' ? null : parseInt(label) })
        });
        const data = await res.json();
        const log = document.getElementById('log');
        const user = document.createElement('div');
        user.className = 'msg user';
        let moodText = '';
        if (data.mood) {
          const moodEmoji = data.mood.mood === 'positive' ? '😊' : '😔';
          moodText = ' · ' + moodEmoji + ' ' + data.mood.mood + ' (' + Math.round(data.mood.confidence * 100) + '%)';
        }
        user.innerHTML = '<div>' + text + '</div><div class="meta">just now' + (label!==''? ' · label='+label: '') + moodText + '</div>';
        log.appendChild(user);
        const asst = document.createElement('div');
        asst.className = 'msg assistant';
        asst.innerHTML = '<div>' + data.reply + '</div><div class="meta">just now</div>';
        log.appendChild(asst);
        document.getElementById('text').value = '';
        // Speak only if the last input came from voice
        if (window._lastInputWasVoice) {
          speak(data.reply);
          window._lastInputWasVoice = false;
        }
        if (btn) btn.disabled = false;
      }

      // Text-to-speech via Web Speech API
      function speak(text) {
        if (!('speechSynthesis' in window)) return;
        const u = new SpeechSynthesisUtterance(text);
        u.lang = 'en-US';
        u.rate = 1.0;
        speechSynthesis.cancel();
        speechSynthesis.speak(u);
      }

      // Voice record and STT
      let recorder;
      let chunks = [];
      let isRecording = false;

      async function toggleRecord() {
        const btn = document.getElementById('recordBtn');
        if (!isRecording) {
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
          recorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
          recorder.onstop = async () => {
            const blob = new Blob(chunks, { type: 'audio/webm' });
            chunks = [];
            const form = new FormData();
            form.append('audio', blob, 'audio.webm');
            const res = await fetch('/stt', { method: 'POST', body: form });
            const data = await res.json();
            if (data.text) {
              document.getElementById('text').value = data.text;
              window._lastInputWasVoice = true;
            }
          };
          recorder.start();
          isRecording = true; btn.textContent = '⏹ Stop';
        } else {
          recorder.stop();
          isRecording = false; btn.textContent = '🎤 Start';
        }
      }
    </script>
  </body>
 </html>
    """
    
    # Replace placeholder with actual chat history
    html = html_template.replace("CHAT_HISTORY_PLACEHOLDER", chat_history)
    return Response(html, mimetype="text/html")


@app.post("/send")
def send() -> Response:
    payload = request.get_json(force=True, silent=True) or {}
    text = (payload.get("text") or "").strip()
    label = payload.get("label")
    if not text:
        return jsonify({"error": "empty"}), 400

    ts = datetime.utcnow().isoformat() + "Z"
    
    # Predict mood if FL model is available
    mood_prediction = predict_mood(text)
    
    user_event = {"role": "user", "text": text, "timestamp": ts, "label": label}
    if mood_prediction:
        user_event["predicted_mood"] = mood_prediction
    append_chat_event(user_event)

    try:
        reply = generate_response(text)
    except Exception as e:
        logging.exception("MLX generation failed")
        return jsonify({
            "error": "mlx_generation_failed",
            "message": "MLX model failed to load or generate. Ensure MLX is installed and PORTAL_MLX_MODEL points to a valid model.",
            "details": str(e),
        }), 500
    asst_event = {"role": "assistant", "text": reply, "timestamp": datetime.utcnow().isoformat() + "Z"}
    append_chat_event(asst_event)

    response_data = {"reply": reply}
    if mood_prediction:
        response_data["mood"] = mood_prediction
    
    return jsonify(response_data)


@app.post("/predict_mood")
def predict_mood_endpoint() -> Response:
    """Endpoint to predict mood from text using the FL model."""
    payload = request.get_json(force=True, silent=True) or {}
    text = (payload.get("text") or "").strip()
    if not text:
        return jsonify({"error": "empty"}), 400
    
    mood_prediction = predict_mood(text)
    if mood_prediction is None:
        return jsonify({
            "error": "model_not_available",
            "message": "FL mood classifier not available. Run federated training first."
        }), 503
    
    return jsonify(mood_prediction)


@app.post("/stt")
def stt() -> Response:
    if WhisperModel is None:
        return jsonify({"error": "stt_unavailable"}), 400
    if "audio" not in request.files:
        return jsonify({"error": "missing_audio"}), 400
    audio_file = request.files["audio"]
    ensure_data_dir()
    tmp_path = os.path.join(DATA_DIR, f"stt_{datetime.utcnow().timestamp()}.webm")
    audio_file.save(tmp_path)
    try:
        _load_stt_model()
        assert _stt_model is not None
        segments, info = _stt_model.transcribe(tmp_path, beam_size=1, vad_filter=True, language="en")
        text_parts: List[str] = [seg.text for seg in segments]
        transcript = " ".join(text_parts).strip()
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    return jsonify({"text": transcript})


if __name__ == "__main__":
    print(f"🌐 Starting local portal on http://{APP_HOST}:{APP_PORT} (local only)")
    app.run(host=APP_HOST, port=APP_PORT, debug=False)


