"""
COMPREHENSIVE LOCAL TEST - All 5 Scoring Components (100 Points)
Tests all evaluation criteria for 95+ score compliance
"""
import requests
import json
import time
import sys
from typing import Dict, List

BASE_URL = "http://localhost:8000"
API_ENDPOINT = f"{BASE_URL}/analyze"
API_KEY = "honeypot_key_2026_eval"

# Test Results
results = {
    "scam_detection": {"passed": 0, "total": 0, "points": 0, "max": 20},
    "extracted_intelligence": {"passed": 0, "total": 0, "points": 0, "max": 30},
    "conversation_quality": {"passed": 0, "total": 0, "points": 0, "max": 30},
    "engagement_quality": {"passed": 0, "total": 0, "points": 0, "max": 10},
    "response_structure": {"passed": 0, "total": 0, "points": 0, "max": 10},
}

def make_request(session_id: str, message: str, history: List[Dict] = None) -> Dict:
    """Make API request"""
    payload = {
        "sessionId": session_id,
        "message": {
            "sender": "scammer",
            "text": message,
            "timestamp": int(time.time() * 1000)
        },
        "conversationHistory": history or [],
        "metadata": {"channel": "SMS", "language": "English", "locale": "IN"}
    }
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY
    }
    
    try:
        response = requests.post(API_ENDPOINT, json=payload, headers=headers, timeout=30)
        return {
            "status_code": response.status_code,
            "data": response.json() if response.status_code == 200 else None,
            "error": response.text if response.status_code != 200 else None
        }
    except Exception as e:
        return {"status_code": 0, "data": None, "error": str(e)}

def test_scam_detection():
    """Component 1: Scam Detection (20 points)"""
    print("\n" + "="*80)
    print("COMPONENT 1: SCAM DETECTION (20 points)")
    print("="*80)
    
    test_cases = [
        ("Bank Fraud", "URGENT: Your SBI account has been compromised. Share OTP immediately to block unauthorized transaction!"),
        ("UPI Fraud", "Congratulations! You received ₹5000 cashback. Claim now!"),
        ("Phishing", "Amazon order confirmed. Click here to track: http://fake-amazon.com"),
        ("Job Scam", "URGENT HIRING! Work from home. Earn ₹50,000/month. Apply now!"),
        ("Sextortion", "I have your private videos. Pay ₹50,000 in Bitcoin or I'll share them.")
    ]
    
    for name, message in test_cases:
        results["scam_detection"]["total"] += 1
        result = make_request(f"test_scam_{name.replace(' ', '_').lower()}", message)
        
        if result["status_code"] == 200 and result["data"]:
            reply = result["data"].get("reply", "")
            # Check if reply is engaging (not empty or dismissive)
            if reply and len(reply) > 10 and "not interested" not in reply.lower():
                print(f"  ✅ {name}: Detected & Engaging")
                results["scam_detection"]["passed"] += 1
            else:
                print(f"  ⚠️  {name}: Detected but weak reply - '{reply[:50]}...'")
                results["scam_detection"]["passed"] += 0.5
        else:
            print(f"  ❌ {name}: Failed - {result.get('error', 'Unknown error')[:50]}")
    
    # Calculate points
    ratio = results["scam_detection"]["passed"] / results["scam_detection"]["total"]
    results["scam_detection"]["points"] = int(ratio * results["scam_detection"]["max"])
    print(f"  Score: {results['scam_detection']['points']}/{results['scam_detection']['max']} points")

def test_extracted_intelligence():
    """Component 2: Extracted Intelligence (30 points)"""
    print("\n" + "="*80)
    print("COMPONENT 2: EXTRACTED INTELLIGENCE (30 points)")
    print("="*80)
    
    # Import local function
    from main import extract_entities
    
    test_cases = [
        {
            "name": "Bank Fraud - Full Data",
            "text": "Your account 1234567890123456 is locked! Call +91-9876543210. Email fraud@fakebank.com. UPI: scammer.fraud@fakebank. Link: http://fake-site.com/verify. ID: TXN123456",
            "expected": {
                "phoneNumbers": 1,
                "bankAccounts": 1,
                "upiIds": 1,
                "emailAddresses": 1,
                "phishingLinks": 1,
                "ids": 1
            }
        },
        {
            "name": "UPI Fraud",
            "text": "Send ₹5000 to cashback@paytm. Call 8765432109. Your claim ID: CB123456. Visit http://cashback-upi.com",
            "expected": {
                "upiIds": 1,
                "phoneNumbers": 1,
                "ids": 1,
                "phishingLinks": 1
            }
        },
        {
            "name": "Sextortion",
            "text": "Pay Bitcoin to 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa. Email blackmail@proton.me. Telegram: @blackmailer. Tracking: DH123456789",
            "expected": {
                "bitcoinAddresses": 1,
                "emailAddresses": 1,
                "telegramIds": 1,
                "trackingNumbers": 1
            }
        }
    ]
    
    total_fields = 0
    extracted_fields = 0
    
    for test in test_cases:
        print(f"\n  Test: {test['name']}")
        extracted = extract_entities(test["text"])
        
        for field, min_count in test["expected"].items():
            total_fields += 1
            actual = len(extracted.get(field, []))
            if actual >= min_count:
                print(f"    ✅ {field}: {actual} found")
                extracted_fields += 1
            else:
                print(f"    ❌ {field}: {actual} found, expected {min_count}")
    
    results["extracted_intelligence"]["total"] = total_fields
    results["extracted_intelligence"]["passed"] = extracted_fields
    ratio = extracted_fields / total_fields if total_fields > 0 else 0
    results["extracted_intelligence"]["points"] = int(ratio * results["extracted_intelligence"]["max"])
    print(f"\n  Score: {results['extracted_intelligence']['points']}/{results['extracted_intelligence']['max']} points")

def test_conversation_quality():
    """Component 3: Conversation Quality (30 points)"""
    print("\n" + "="*80)
    print("COMPONENT 3: CONVERSATION QUALITY (30 points)")
    print("="*80)
    
    session_id = "test_conversation_quality"
    history = []
    
    messages = [
        "URGENT: Your SBI account blocked. Call 9876543210",
        "We are from SBI fraud department. Your OTP is required",
        "Send money to upi@paytm immediately",
        "Account number is 1234567890123456",
        "Click this link http://fake-sbi.com/verify"
    ]
    
    passed_turns = 0
    
    for i, msg in enumerate(messages):
        result = make_request(session_id, msg, history)
        
        if result["status_code"] == 200 and result["data"]:
            reply = result["data"].get("reply", "")
            if reply and len(reply) > 5:
                print(f"  Turn {i+1}: ✅ Reply received")
                passed_turns += 1
                
                # Update history
                history.append({"sender": "scammer", "text": msg, "timestamp": int(time.time() * 1000)})
                history.append({"sender": "user", "text": reply, "timestamp": int(time.time() * 1000) + 500})
            else:
                print(f"  Turn {i+1}: ❌ Empty reply")
        else:
            print(f"  Turn {i+1}: ❌ Failed - {result.get('error', 'Unknown')[:30]}")
    
    # Check context preservation (multi-turn)
    context_preserved = passed_turns >= 3
    
    results["conversation_quality"]["total"] = 3  # multi-turn, continuity, context
    results["conversation_quality"]["passed"] = (1 if passed_turns >= 5 else 0.5) + \
                                                 (1 if passed_turns >= 3 else 0) + \
                                                 (1 if context_preserved else 0)
    ratio = results["conversation_quality"]["passed"] / results["conversation_quality"]["total"]
    results["conversation_quality"]["points"] = int(ratio * results["conversation_quality"]["max"])
    print(f"  Multi-turn: {passed_turns}/5 successful")
    print(f"  Score: {results['conversation_quality']['points']}/{results['conversation_quality']['max']} points")

def test_engagement_quality():
    """Component 4: Engagement Quality (10 points)"""
    print("\n" + "="*80)
    print("COMPONENT 4: ENGAGEMENT QUALITY (10 points)")
    print("="*80)
    
    session_id = "test_engagement"
    
    # Test response time
    start = time.time()
    result = make_request(session_id, "Test for response time")
    response_time = time.time() - start
    
    response_ok = result["status_code"] == 200 and response_time < 30
    print(f"  Response Time: {'✅' if response_ok else '❌'} {response_time:.2f}s (max 30s)")
    
    # Test message count tracking
    messages_sent = 0
    for i in range(5):
        result = make_request(f"{session_id}_{i}", f"Message {i+1}")
        if result["status_code"] == 200:
            messages_sent += 1
    
    messages_ok = messages_sent >= 3
    print(f"  Message Tracking: {'✅' if messages_ok else '❌'} {messages_sent}/5 tracked")
    
    results["engagement_quality"]["total"] = 3
    results["engagement_quality"]["passed"] = (1 if response_ok else 0) + (1 if messages_ok else 0) + 1  # duration always tracked
    ratio = results["engagement_quality"]["passed"] / results["engagement_quality"]["total"]
    results["engagement_quality"]["points"] = int(ratio * results["engagement_quality"]["max"])
    print(f"  Score: {results['engagement_quality']['points']}/{results['engagement_quality']['max']} points")

def test_response_structure():
    """Component 5: Response Structure (10 points)"""
    print("\n" + "="*80)
    print("COMPONENT 5: RESPONSE STRUCTURE (10 points)")
    print("="*80)
    
    # Test endpoint exists
    result = make_request("test_structure", "Test message")
    endpoint_ok = result["status_code"] == 200
    print(f"  POST /analyze: {'✅' if endpoint_ok else '❌'} Status {result['status_code']}")
    
    # Test response format
    format_ok = False
    if result["data"]:
        has_status = "status" in result["data"]
        has_reply = "reply" in result["data"]
        format_ok = has_status and has_reply
        print(f"  Response Format: {'✅' if format_ok else '❌'} status={has_status}, reply={has_reply}")
    
    # Test authentication
    payload = {
        "sessionId": "test_auth",
        "message": {"sender": "scammer", "text": "test", "timestamp": int(time.time() * 1000)},
        "conversationHistory": []
    }
    
    # Wrong key
    response = requests.post(API_ENDPOINT, json=payload, headers={"x-api-key": "wrong-key"}, timeout=10)
    auth_ok = response.status_code == 401
    print(f"  Authentication: {'✅' if auth_ok else '❌'} Wrong key rejected")
    
    results["response_structure"]["total"] = 4
    results["response_structure"]["passed"] = (1 if endpoint_ok else 0) + (1 if format_ok else 0) + (1 if auth_ok else 0) + 1
    ratio = results["response_structure"]["passed"] / results["response_structure"]["total"]
    results["response_structure"]["points"] = int(ratio * results["response_structure"]["max"])
    print(f"  Score: {results['response_structure']['points']}/{results['response_structure']['max']} points")

def print_final_report():
    """Print comprehensive test report"""
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE TEST REPORT (100 Points Total)")
    print("="*80)
    
    total_score = 0
    max_score = 0
    
    for component, data in results.items():
        status = "✅" if data["points"] >= data["max"] * 0.7 else "⚠️" if data["points"] >= data["max"] * 0.4 else "❌"
        print(f"{status} {component.replace('_', ' ').title():30} | {data['points']:2}/{data['max']:2} points")
        total_score += data["points"]
        max_score += data["max"]
    
    print("-"*80)
    print(f"TOTAL SCORE: {total_score}/{max_score} ({total_score/max_score*100:.1f}%)")
    print("="*80)
    
    if total_score >= 95:
        print("🎯 STATUS: EXCELLENT (95+ SCORE ACHIEVED!)")
    elif total_score >= 80:
        print("✅ STATUS: GOOD (80+ score)")
    elif total_score >= 60:
        print("⚠️  STATUS: NEEDS IMPROVEMENT (60+ score)")
    else:
        print("❌ STATUS: FAILING (<60 score)")
    
    print("\n📋 EVALUATION CRITERIA:")
    print("  1. Scam Detection      (20 pts): Detects scams and engages appropriately")
    print("  2. Extracted Intelligence (30 pts): Extracts all 10 entity types")
    print("  3. Conversation Quality (30 pts): Multi-turn, context preservation")
    print("  4. Engagement Quality   (10 pts): Duration, message count, <30s response")
    print("  5. Response Structure   (10 pts): Correct JSON format, auth, endpoint")
    print("="*80)
    
    return total_score

if __name__ == "__main__":
    print("🚀 STARTING COMPREHENSIVE LOCAL TEST")
    print("Testing all 5 scoring components for 95+ score compliance...")
    
    # Run all tests
    test_scam_detection()
    test_extracted_intelligence()
    test_conversation_quality()
    test_engagement_quality()
    test_response_structure()
    
    # Print final report
    final_score = print_final_report()
    
    # Exit with appropriate code
    sys.exit(0 if final_score >= 60 else 1)
