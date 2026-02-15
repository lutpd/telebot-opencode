import os
import requests
from flask import Flask, request
from openai import OpenAI

app = Flask(__name__)

# --- Configuration (Make sure these are set in Render Environment Variables) ---
BOT_TOKEN = os.environ.get("TELEGRAM_TOKEN")
API_KEY = os.environ.get("LLM_API_KEY")
BASE_URL = os.environ.get("LLM_BASE_URL")    # e.g., https://api.groq.com/openai/v1
MODEL_NAME = os.environ.get("LLM_MODEL_NAME") # e.g., llama3-8b-8192

# Initialize the OpenAI Client
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

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
        
        if user_text.startswith("/start"):
            send_message(chat_id, "Hello! I am your AI assistant. Ask me anything!")
            return "ok", 200

        try:
            # Get response from AI
            ai_response = get_ai_response(user_text)
            # Send response to Telegram
            send_message(chat_id, ai_response)
        except Exception as e:
            print(f"Error: {e}")
            send_message(chat_id, "⚠️ Sorry, I encountered an error. Please try again.")

    return "ok", 200

def get_ai_response(prompt):
    """Calls the OpenAI-compatible API."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Use markdown for formatting."},
            {"role": "user", "content": prompt}
        ]
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
