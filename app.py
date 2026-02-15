from flask import Flask, request
from openai import OpenAI
import os
import requests

app = Flask(__name__)

# Environment Variables
BOT_TOKEN = os.environ.get("TELEGRAM_TOKEN")
API_KEY = os.environ.get("LLM_API_KEY")
BASE_URL = os.environ.get("LLM_BASE_URL") # e.g., https://api.groq.com/openai/v1
MODEL_NAME = os.environ.get("LLM_MODEL_NAME") # e.g., llama3-8b-8192

# Initialize OpenAI Client (works with any compatible provider)
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

@app.route("/webhook", methods=["POST"])
def telegram_webhook():
    data = request.json
    
    # Simple check to ensure we have a message and it contains text
    if "message" in data and "text" in data["message"]:
        chat_id = data["message"]["chat"]["id"]
        text = data["message"]["text"]
        
        try:
            # Get response from AI
            ai_response = get_ai_response(text)
            send_message(chat_id, ai_response)
        except Exception as e:
            print(f"Error: {e}")
            send_message(chat_id, "Sorry, I encountered an error processing your request.")

    return "ok", 200

def get_ai_response(prompt):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful Telegram bot."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def send_message(chat_id, text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    requests.post(url, json=payload)

if __name__ == "__main__":
    # This is for local testing only. Render uses Gunicorn.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
