"""Quick debug to see actual replies"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"
API_KEY = "honeypot_key_2026_eval"

def test_reply():
    payload = {
        "sessionId": f"debug_{int(time.time())}",
        "message": {
            "sender": "scammer",
            "text": "URGENT: Your SBI account has been compromised. Share OTP immediately!",
            "timestamp": int(time.time() * 1000)
        },
        "conversationHistory": [],
        "metadata": {"channel": "SMS", "language": "English", "locale": "IN"}
    }
    
    response = requests.post(
        f"{BASE_URL}/analyze",
        json=payload,
        headers={"Content-Type": "application/json", "x-api-key": API_KEY},
        timeout=30
    )
    
    if response.status_code == 200:
        data = response.json()
        reply = data.get("reply", "")
        print(f"Reply: {reply}")
        print(f"Has question mark: {'?' in reply}")
        print(f"Question count: {reply.count('?')}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_reply()
