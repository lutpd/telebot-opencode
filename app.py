import os
import requests
from flask import Flask, request
from openai import OpenAI

app = Flask(__name__)

# --- Configuration (Set these in Render Environment Variables) ---
BOT_TOKEN = os.environ.get("TELEGRAM_TOKEN")
API_KEY = os.environ.get("LLM_API_KEY")
BASE_URL = os.environ.get("LLM_BASE_URL")    # e.g., https://api.groq.com/openai/v1
MODEL_NAME = os.environ.get("LLM_MODEL_NAME") # e.g., llama3-8b-8192

# Initialize the OpenAI Client (Works with Groq, DeepSeek, OpenRouter, etc.)
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

# 1. Keep-Alive Route (For UptimeRobot)
@app.route("/ping", methods=["GET"])
def ping():
    """Simple route to keep the Render instance awake."""
    return "Bot is awake!", 200

# 2. Telegram Webhook Route
@app.route("/webhook", methods=["POST"])
def telegram_webhook():
    data = request.json
    
    # Check if the update contains a message and text
    if "message" in data and "text" in data["message"]:
        chat_id = data["message"]["chat"]["id"]
        user_text = data["message"]["text"]
        
        # Ignore commands like /start if you just want to chat, 
        # or handle them specifically here.
        if user_text.startswith("/start"):
            send_message(chat_id, "Hello! I am your AI assistant. How can I help you today?")
            return "ok", 200

        try:
            # Get response from the OpenAI-compatible AI
            ai_response = get_ai_response(user_text)
            # Send the AI response back to Telegram
            send_message(chat_id, ai_response)
        except Exception as e:
            print(f"Error processing AI response: {e}")
            send_message(chat_id, "⚠️ Sorry, I'm having trouble thinking right now.")

    return "ok", 200

def get_ai_response(prompt):
    """Calls the OpenAI-compatible API."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful and concise Telegram bot."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def send_message(chat_id, text):
    """Sends a text message to the Telegram chat."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id, 
        "text": text
    }
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Error sending message to Telegram: {e}")

if __name__ == "__main__":
    # Get port from environment (Render provides this) or default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
