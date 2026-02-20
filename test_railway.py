"""Quick API test against Railway deployment"""
import requests
import json

# Railway URL
BASE_URL = "https://honeypot-api-production-176c.up.railway.app"
API_ENDPOINT = f"{BASE_URL}/analyze"
API_KEY = "honeypot_key_2026_eval"

def test_api():
    """Test API with scam message"""
    payload = {
        "sessionId": "test_session_123",
        "message": {
            "sender": "scammer",
            "text": "URGENT: Your SBI account blocked. Call 9876543210 immediately. Account: 1234567890123456 UPI: test@paytm",
            "timestamp": 1707753600000
        },
        "conversationHistory": [],
        "metadata": {"channel": "SMS", "language": "English", "locale": "IN"}
    }
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY
    }
    
    print("=" * 60)
    print("TESTING HONEYPOT API")
    print("=" * 60)
    print(f"URL: {API_ENDPOINT}")
    print(f"API Key: {API_KEY}")
    print()
    
    try:
        response = requests.post(API_ENDPOINT, json=payload, headers=headers, timeout=30)
        print(f"Status Code: {response.status_code}")
        print()
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API CALL SUCCESSFUL")
            print()
            print("Response:")
            print(json.dumps(data, indent=2))
            print()
            
            # Check reply quality
            reply = data.get("reply", "")
            if reply and "didn't catch" not in reply.lower():
                print("✅ HONEYPOT RESPONSE: AI-powered reply generated")
                print(f"Reply: {reply[:100]}...")
            else:
                print("⚠️ FALLBACK RESPONSE: Gemini API may not be working")
                print(f"Reply: {reply}")
            
            return True
        else:
            print(f"❌ API ERROR: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"❌ CONNECTION ERROR: {e}")
        return False

if __name__ == "__main__":
    success = test_api()
    exit(0 if success else 1)
