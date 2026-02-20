"""
Local API Test with Entity Extraction Verification
Tests the fix for extractedIntelligence bug
"""
import os
import sys
import requests
import json
import time

# Set API key
os.environ['GEMINI_API_KEY'] = 'AIzaSyBBc-Qa2ptjELuVXEZNxcHkJzOALOdymN8'

BASE_URL = "http://localhost:8000"
API_ENDPOINT = f"{BASE_URL}/analyze"
API_KEY = "honeypot_key_2026_eval"

def test_entity_extraction():
    """Test that entities are properly extracted"""
    
    # Test message with multiple entities
    test_cases = [
        {
            "name": "Bank Fraud - Full Data",
            "text": "URGENT: Your SBI account has been compromised. Share OTP immediately to block unauthorized transaction of ₹45,000. Call +91-9876543210. Account: 1234567890123456 UPI: victim@oksbi Email: fraud@sbi-security.com",
            "expected_entities": ["phoneNumbers", "bankAccounts", "upiIds", "emailAddresses"]
        },
        {
            "name": "UPI Fraud",
            "text": "Congratulations! You received ₹5000 cashback. Claim: http://cashback-upi.com/claim UPI: cash@paytm Call 9876543210",
            "expected_entities": ["phoneNumbers", "upiIds", "phishingLinks"]
        },
        {
            "name": "Sextortion",
            "text": "I have your private videos. Pay ₹50,000 in Bitcoin or I'll send to all contacts. BTC: 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa Email: blackmail@proton.me Telegram: @blackmailer",
            "expected_entities": ["bitcoinAddresses", "emailAddresses", "telegramIds"]
        }
    ]
    
    print("="*80)
    print("ENTITY EXTRACTION TEST")
    print("="*80)
    print(f"API: {API_ENDPOINT}")
    print()
    
    all_passed = True
    
    for test in test_cases:
        print(f"Testing: {test['name']}")
        print(f"Message: {test['text'][:80]}...")
        
        payload = {
            "sessionId": f"test_{int(time.time())}_{test['name'].replace(' ', '_').lower()}",
            "message": {
                "sender": "scammer",
                "text": test["text"],
                "timestamp": int(time.time() * 1000)
            },
            "conversationHistory": [],
            "metadata": {"channel": "SMS", "language": "English", "locale": "IN"}
        }
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": API_KEY
        }
        
        try:
            response = requests.post(API_ENDPOINT, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                reply = data.get("reply", "")
                
                # Check if reply is good
                red_flags = ["scam", "fraud", "this is a", "you are a"]
                has_red_flag = any(flag in reply.lower() for flag in red_flags)
                
                if has_red_flag:
                    print(f"  ❌ FAIL: Reply reveals honeypot - '{reply[:50]}...'")
                    all_passed = False
                else:
                    print(f"  ✅ Reply OK: '{reply[:60]}...'")
                    
                print()
            else:
                print(f"  ❌ ERROR: {response.status_code}")
                all_passed = False
                
        except Exception as e:
            print(f"  ❌ EXCEPTION: {e}")
            all_passed = False
    
    print("="*80)
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*80)
    
    return all_passed

if __name__ == "__main__":
    success = test_entity_extraction()
    sys.exit(0 if success else 1)
