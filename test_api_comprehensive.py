"""
Comprehensive API Test - Check all 12 scenarios
"""
import requests
import json

BASE_URL = "https://honeypot-api-production-176c.up.railway.app"
API_ENDPOINT = f"{BASE_URL}/analyze"
API_KEY = "honeypot_key_2026_eval"

SCENARIOS = [
    {
        "name": "Bank Fraud",
        "text": "URGENT: Your SBI account has been compromised. Share OTP immediately to block unauthorized transaction of ₹45,000. Call +91-9876543210. Account: 1234567890123456",
        "expected_type": "Bank Fraud"
    },
    {
        "name": "UPI Fraud", 
        "text": "Congratulations! You received ₹5000 cashback. Claim: http://cashback-upi.com/claim Call +91-8765432109",
        "expected_type": "UPI Fraud"
    },
    {
        "name": "Phishing",
        "text": "Amazon Great Sale! You won iPhone 15. Claim: http://amazon-deals.fake-site.com/claim?id=WIN12345",
        "expected_type": "Phishing"
    },
    {
        "name": "Sextortion",
        "text": "I have your private videos. Pay ₹50,000 in Bitcoin or I'll send to all contacts. Contact: blackmail@proton.me BTC: 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
        "expected_type": "Sextortion"
    },
    {
        "name": "Job Scam",
        "text": "URGENT HIRING! Work from home. Earn ₹50,000/month. Register: http://quick-job-portal.com/apply",
        "expected_type": "Job Scam"
    }
]

def test_scenario(scenario):
    """Test a single scenario"""
    payload = {
        "sessionId": f"test_{scenario['name'].replace(' ', '_').lower()}",
        "message": {
            "sender": "scammer",
            "text": scenario["text"],
            "timestamp": 1707753600000
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
            
            # Check if reply reveals honeypot
            red_flags = ["scam", "fraud", "this is a", "you are", "attempt"]
            has_red_flag = any(flag in reply.lower() for flag in red_flags)
            
            return {
                "status": "SUCCESS",
                "reply": reply[:100],
                "reveals_honeypot": has_red_flag,
                "issue": "REVEALS HONEYPOT" if has_red_flag else "OK"
            }
        else:
            return {"status": "ERROR", "code": response.status_code, "error": response.text[:100]}
            
    except Exception as e:
        return {"status": "FAILED", "error": str(e)}

print("="*80)
print("COMPREHENSIVE API TEST - 5 SCENARIOS")
print("="*80)
print(f"URL: {API_ENDPOINT}")
print()

passed = 0
failed = 0
reveals = 0

for scenario in SCENARIOS:
    print(f"Testing: {scenario['name']}")
    result = test_scenario(scenario)
    
    if result["status"] == "SUCCESS":
        if result["reveals_honeypot"]:
            print(f"  ❌ FAIL - {result['issue']}")
            print(f"     Reply: {result['reply']}")
            reveals += 1
        else:
            print(f"  ✅ PASS")
            print(f"     Reply: {result['reply']}")
            passed += 1
    else:
        print(f"  ❌ ERROR: {result.get('error', result.get('code'))}")
        failed += 1
    print()

print("="*80)
print("SUMMARY")
print("="*80)
print(f"Passed: {passed}/{len(SCENARIOS)}")
print(f"Reveals Honeypot: {reveals}/{len(SCENARIOS)}")
print(f"Failed/Error: {failed}/{len(SCENARIOS)}")
print()

if reveals > 0:
    print("⚠️  ISSUE: API is revealing itself as a honeypot!")
    print("   The persona prompts need to be fixed on Railway.")
    print()
    print("SOLUTION:")
    print("1. Redeploy to Railway: railway up")
    print("2. Or verify GEMINI_API_KEY is set in Railway variables")
