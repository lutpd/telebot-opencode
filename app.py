import os
import requests
import uuid
import json
from flask import Flask, request
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Distance, VectorParams, Filter, FieldCondition, MatchValue
from datetime import datetime

app = Flask(__name__)

# --- Configuration (Make sure these are set in Render Environment Variables) ---
BOT_TOKEN = os.environ.get("TELEGRAM_TOKEN")
API_KEY = os.environ.get("LLM_API_KEY")
BASE_URL = os.environ.get("LLM_BASE_URL")    # e.g., https://api.groq.com/openai/v1
MODEL_NAME = os.environ.get("LLM_MODEL_NAME") # e.g., llama3-8b-8192

# Qdrant Configuration
QDRANT_URL = os.environ.get("QDRANT_URL")      # e.g., https://your-cluster.qdrant.io
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION_NAME = "telegram_chat_memory"

# Initialize the OpenAI Client
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

# Initialize Qdrant Client
qdrant_client = None
if QDRANT_URL and QDRANT_API_KEY:
    try:
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        # Create collection if it doesn't exist
        collections = qdrant_client.get_collections().collections
        if not any(c.name == COLLECTION_NAME for c in collections):
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
            # Create payload index for chat_id filtering
            qdrant_client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="chat_id",
                field_schema="keyword"
            )
        print("✅ Qdrant client initialized successfully")
    except Exception as e:
        print(f"⚠️ Qdrant initialization error: {e}")
        qdrant_client = None
else:
    print("⚠️ Qdrant not configured. Running without memory.")

# In-memory fallback for chat sessions (if Qdrant fails)
memory_fallback = {}

def get_chat_history(chat_id, limit=10):
    """Retrieve chat history from Qdrant or fallback memory."""
    if qdrant_client:
        try:
            # Search for messages from this chat
            scroll_filter = Filter(
                must=[
                    FieldCondition(
                        key="chat_id",
                        match=MatchValue(value=str(chat_id))
                    )
                ]
            )
            results = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=scroll_filter,
                limit=limit * 2,  # Get more to account for user+assistant pairs
                with_payload=True,
                with_vectors=False
            )[0]
            
            # Sort by timestamp
            messages = []
            for point in results:
                payload = point.payload
                messages.append({
                    "role": payload.get("role"),
                    "content": payload.get("content"),
                    "timestamp": payload.get("timestamp")
                })
            
            messages.sort(key=lambda x: x.get("timestamp", ""))
            return [{"role": m["role"], "content": m["content"]} for m in messages[-limit:]]
        except Exception as e:
            print(f"Qdrant error, using fallback: {e}")
    
    # Fallback to memory
    return memory_fallback.get(str(chat_id), [])

def store_message(chat_id, role, content):
    """Store a message in Qdrant or fallback memory."""
    timestamp = datetime.now().isoformat()
    
    if qdrant_client:
        try:
            # Create a simple vector (using timestamp-based placeholder since we may not have embeddings)
            # In production, you might want to use actual text embeddings
            vector = [0.0] * 768  # Placeholder vector
            point_id = str(uuid.uuid4())
            
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "chat_id": str(chat_id),
                            "role": role,
                            "content": content,
                            "timestamp": timestamp
                        }
                    )
                ]
            )
            return True
        except Exception as e:
            print(f"Qdrant store error: {e}")
    
    # Fallback to memory
    chat_key = str(chat_id)
    if chat_key not in memory_fallback:
        memory_fallback[chat_key] = []
    memory_fallback[chat_key].append({"role": role, "content": content})
    # Keep only last 20 messages
    memory_fallback[chat_key] = memory_fallback[chat_key][-20:]
    return True

def clear_chat_memory(chat_id):
    """Clear chat memory for a specific chat."""
    if qdrant_client:
        try:
            # Delete all points for this chat_id
            qdrant_client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="chat_id",
                            match=MatchValue(value=str(chat_id))
                        )
                    ]
                )
            )
        except Exception as e:
            print(f"Qdrant clear error: {e}")
    
    # Clear fallback memory
    if str(chat_id) in memory_fallback:
        memory_fallback[str(chat_id)] = []
    return True

def format_bold_text(text):
    """Ensure bold text formatting is properly applied for Telegram Markdown."""
    return text

def get_status_message():
    """Check and return Qdrant status."""
    if not QDRANT_URL or not QDRANT_API_KEY:
        return "⚠️ **Qdrant not configured**\n\nRunning in fallback mode (memory only).\n\nSet QDRANT_URL and QDRANT_API_KEY env vars."
    
    if qdrant_client is None:
        return "❌ **Qdrant connection failed**\n\nCheck your QDRANT_URL and QDRANT_API_KEY.\nUsing fallback memory."
    
    try:
        # Test connection
        collections = qdrant_client.get_collections().collections
        collection_exists = any(c.name == COLLECTION_NAME for c in collections)
        
        # Count messages in collection
        count = 0
        if collection_exists:
            count_info = qdrant_client.count(collection_name=COLLECTION_NAME)
            count = count_info.count
        
        status = "✅ **Qdrant is working!**\n\n"
        status += f"📁 Collection: `{COLLECTION_NAME}`\n"
        status += f"📝 Total messages stored: `{count}`\n"
        status += f"🔗 URL: `{QDRANT_URL[:30]}...`\n\n"
        status += "Memory is persistent across restarts!"
        return status
    except Exception as e:
        return f"❌ **Qdrant error:** `{str(e)[:100]}`\n\nUsing fallback memory."

# 1. Homepage Route (To avoid "Not Found" error)
@app.route("/", methods=["GET"])
def index():
    return "<h1>Bot is Online</h1><p>Telegram Webhook is active.</p>", 200

# 2. Keep-Alive Route (For UptimeRobot)
@app.route("/ping", methods=["GET"])
def ping():
    return "Bot is awake!", 200

# 3. Telegram Webhook Route
@app.route("/webhook", methods=["POST"])
def telegram_webhook():
    data = request.json
    
    if "message" in data and "text" in data["message"]:
        chat_id = data["message"]["chat"]["id"]
        user_text = data["message"]["text"]
        user_id = data["message"]["from"]["id"]
        
        # Handle /start command
        if user_text.startswith("/start"):
            welcome_msg = """🤖 **Welcome to Your AI Assistant!**

I'm here to help you with anything you need. I can remember our conversations too!

**Commands:**
• **/start** - Show this welcome message
• **/bbb** - Start a fresh chat (clear memory)
• **/status** - Check Qdrant memory status

Just send me a message and I'll respond!"""
            send_message(chat_id, welcome_msg)
            return "ok", 200
        
        # Handle /bbb command (new chat section)
        if user_text.startswith("/bbb"):
            clear_chat_memory(chat_id)
            send_message(chat_id, "🆕 **New chat started!**\n\nMemory cleared. Let's begin fresh! ✨")
            return "ok", 200
        
        # Handle /status command (check Qdrant status)
        if user_text.startswith("/status"):
            status_msg = get_status_message()
            send_message(chat_id, status_msg)
            return "ok", 200
        
        # Regular message - get AI response with memory
        try:
            # Get chat history
            history = get_chat_history(chat_id, limit=10)
            
            # Store user message
            store_message(chat_id, "user", user_text)
            
            # Get AI response
            ai_response = get_ai_response(user_text, history)
            
            # Store assistant response
            store_message(chat_id, "assistant", ai_response)
            
            # Format and send response
            formatted_response = format_bold_text(ai_response)
            send_message(chat_id, formatted_response)
            
        except Exception as e:
            print(f"Error: {e}")
            send_message(chat_id, "⚠️ Sorry, I encountered an error. Please try again.")

    return "ok", 200

def get_ai_response(prompt, history=None):
    """Calls the OpenAI-compatible API with conversation history."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use markdown formatting with **bold** for important words and titles. Be concise but informative."}
    ]
    
    # Add conversation history
    if history:
        messages.extend(history)
    
    # Add current prompt
    messages.append({"role": "user", "content": prompt})
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages
    )
    return response.choices[0].message.content

def send_message(chat_id, text):
    """Sends a message to Telegram with Markdown enabled."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id, 
        "text": text,
        "parse_mode": "Markdown"  # This converts **text** into actual BOLD
    }
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Failed to send message: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
