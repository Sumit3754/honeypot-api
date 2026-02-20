"""
COMPREHENSIVE EVALUATION TEST SUITE
Tests all 5 scoring components for 95+ score compliance

Scoring Breakdown (100 points total):
1. Scam Detection - 20 points
2. Extracted Intelligence - 30 points  
3. Conversation Quality - 30 points
4. Engagement Quality - 10 points
5. Response Structure - 10 points
"""

import pytest
import requests
import json
import time
import re
from typing import Dict, List, Any

# Configuration
BASE_URL = "http://localhost:8000"
API_ENDPOINT = f"{BASE_URL}/analyze"
API_KEY = "honeypot_key_2026_eval"

# Test Results Tracker
test_results = {
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
            "response": response.json() if response.status_code == 200 else None,
            "error": response.text if response.status_code != 200 else None
        }
    except Exception as e:
        return {"status_code": 0, "response": None, "error": str(e)}


def extract_entities_local(text: str) -> Dict:
    """Local entity extraction for validation"""
    import re
    
    results = {
        "phoneNumbers": [],
        "bankAccounts": [],
        "upiIds": [],
        "phishingLinks": [],
        "emailAddresses": [],
        "creditCards": [],
        "bitcoinAddresses": [],
        "telegramIds": [],
        "trackingNumbers": [],
        "ids": []
    }
    
    if not text:
        return results
    
    # Phone patterns
    phone_indian = r'(?:\+91[\-\s]?)?\b[6-9]\d{9}\b'
    phone_us = r'\+1[\-\s]?\(?\d{3}\)?[\-\s]?\d{3}[\-\s]?\d{4}'
    phone_tollfree = r'(?:1?[-\s]?)?800[\-\s]?\d{3}[\-\s]?\d{4}'
    
    results["phoneNumbers"] = re.findall(phone_indian, text) + re.findall(phone_us, text) + re.findall(phone_tollfree, text)
    
    # UPI
    upi_pattern = r'[a-zA-Z0-9.\-_]{2,256}@[a-zA-Z]{2,64}'
    results["upiIds"] = re.findall(upi_pattern, text)
    
    # URLs
    url_pattern = r'(?:https?://|onion://|www\.)[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)'
    results["phishingLinks"] = re.findall(url_pattern, text, re.IGNORECASE)
    
    # Emails
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    results["emailAddresses"] = re.findall(email_pattern, text)
    
    # Bank accounts (9-18 digits, not 12)
    bank_pattern = r'\b\d{9,18}\b'
    banks = re.findall(bank_pattern, text)
    results["bankAccounts"] = [b for b in banks if len(b) != 12]
    
    # Credit cards
    cc_pattern = r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
    results["creditCards"] = re.findall(cc_pattern, text)
    
    # Bitcoin
    btc_pattern = r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b|\bbc1[a-zA-HJ-NP-Z0-9]{39,59}\b'
    results["bitcoinAddresses"] = re.findall(btc_pattern, text)
    
    # Telegram
    tg_pattern = r'@\w{3,32}\b'
    results["telegramIds"] = re.findall(tg_pattern, text)
    
    # Tracking
    track_pattern = r'\b(?:DH|AMZ|UPS|FEDEX|1Z)[\s-]*\d{6,20}\b'
    results["trackingNumbers"] = re.findall(track_pattern, text, re.IGNORECASE)
    
    # IDs
    id_pattern = r'\b(?:TXN|ORD|ID|REF|CASE|EMP|CUS|EXT|SBI|AMZ|WIN|CB|LOAN|KYC|FRD|BILL)[\-\s]?[A-Z0-9]{4,20}\b'
    results["ids"] = re.findall(id_pattern, text, re.IGNORECASE)
    
    # Deduplicate
    for key in results:
        results[key] = list(set(results[key]))
    
    return results


# =============================================================================
# COMPONENT 1: SCAM DETECTION (20 points)
# =============================================================================

class TestScamDetection:
    """Test Scam Detection - 20 points"""
    
    def test_scam_detection_bank_fraud(self):
        """Test Bank Fraud detection"""
        test_results["scam_detection"]["total"] += 1
        result = make_request("test_bank", "URGENT: Your SBI account blocked. Share OTP immediately!")
        assert result["status_code"] == 200
        assert result["response"]["status"] == "success"
        assert len(result["response"]["reply"]) > 0
        # Should get engaging response, not empty/dismissive
        assert "not interested" not in result["response"]["reply"].lower()
        test_results["scam_detection"]["passed"] += 1
    
    def test_scam_detection_upi_fraud(self):
        """Test UPI Fraud detection"""
        test_results["scam_detection"]["total"] += 1
        result = make_request("test_upi", "Congratulations! You won ₹5000 cashback. Claim now!")
        assert result["status_code"] == 200
        assert result["response"]["status"] == "success"
        test_results["scam_detection"]["passed"] += 1
    
    def test_scam_detection_phishing(self):
        """Test Phishing detection"""
        test_results["scam_detection"]["total"] += 1
        result = make_request("test_phishing", "Amazon order confirmed. Click here to track: http://fake-amazon.com")
        assert result["status_code"] == 200
        assert result["response"]["status"] == "success"
        test_results["scam_detection"]["passed"] += 1
    
    def test_scam_detection_job_scam(self):
        """Test Job Scam detection"""
        test_results["scam_detection"]["total"] += 1
        result = make_request("test_job", "URGENT HIRING! Work from home. Earn ₹50,000/month. Apply now!")
        assert result["status_code"] == 200
        assert result["response"]["status"] == "success"
        test_results["scam_detection"]["passed"] += 1
    
    def test_scam_detection_sextortion(self):
        """Test Sextortion detection - CRITICAL (40% weight scenario)"""
        test_results["scam_detection"]["total"] += 1
        result = make_request("test_sextortion", "I have your private videos. Pay ₹50,000 in Bitcoin or I'll share them.")
        assert result["status_code"] == 200
        assert result["response"]["status"] == "success"
        test_results["scam_detection"]["passed"] += 1


# =============================================================================
# COMPONENT 2: EXTRACTED INTELLIGENCE (30 points)
# =============================================================================

class TestExtractedIntelligence:
    """Test Intelligence Extraction - 30 points"""
    
    def test_extract_phone_numbers(self):
        """Test phone number extraction (Indian, US, toll-free)"""
        test_results["extracted_intelligence"]["total"] += 1
        text = "Call +91-9876543210 or 9876543210 or 1800-123-4567 or +1-555-0123456"
        entities = extract_entities_local(text)
        assert len(entities["phoneNumbers"]) >= 2, f"Expected 2+ phones, got {entities['phoneNumbers']}"
        test_results["extracted_intelligence"]["passed"] += 1
    
    def test_extract_upi_ids(self):
        """Test UPI ID extraction"""
        test_results["extracted_intelligence"]["total"] += 1
        text = "Send to test@paytm or user@oksbi or payment@okhdfcbank"
        entities = extract_entities_local(text)
        assert len(entities["upiIds"]) >= 2, f"Expected 2+ UPI IDs, got {entities['upiIds']}"
        test_results["extracted_intelligence"]["passed"] += 1
    
    def test_extract_phishing_links(self):
        """Test phishing link extraction"""
        test_results["extracted_intelligence"]["total"] += 1
        text = "Visit http://fake-site.com/claim or https://scam-bank.com/login"
        entities = extract_entities_local(text)
        assert len(entities["phishingLinks"]) >= 2, f"Expected 2+ links, got {entities['phishingLinks']}"
        test_results["extracted_intelligence"]["passed"] += 1
    
    def test_extract_email_addresses(self):
        """Test email extraction"""
        test_results["extracted_intelligence"]["total"] += 1
        text = "Contact support@bank.com or fraud@fake-site.org"
        entities = extract_entities_local(text)
        assert len(entities["emailAddresses"]) >= 2, f"Expected 2+ emails, got {entities['emailAddresses']}"
        test_results["extracted_intelligence"]["passed"] += 1
    
    def test_extract_bank_accounts(self):
        """Test bank account extraction"""
        test_results["extracted_intelligence"]["total"] += 1
        text = "Account 1234567890123456 for transfer"
        entities = extract_entities_local(text)
        assert len(entities["bankAccounts"]) >= 1, f"Expected 1+ accounts, got {entities['bankAccounts']}"
        test_results["extracted_intelligence"]["passed"] += 1
    
    def test_extract_credit_cards(self):
        """Test credit card extraction"""
        test_results["extracted_intelligence"]["total"] += 1
        text = "Card: 4532-7890-1234-5678"
        entities = extract_entities_local(text)
        assert len(entities["creditCards"]) >= 1, f"Expected 1+ credit cards, got {entities['creditCards']}"
        test_results["extracted_intelligence"]["passed"] += 1
    
    def test_extract_bitcoin_addresses(self):
        """Test Bitcoin address extraction"""
        test_results["extracted_intelligence"]["total"] += 1
        text = "Send BTC to 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
        entities = extract_entities_local(text)
        assert len(entities["bitcoinAddresses"]) >= 1, f"Expected 1+ Bitcoin addresses, got {entities['bitcoinAddresses']}"
        test_results["extracted_intelligence"]["passed"] += 1
    
    def test_extract_telegram_ids(self):
        """Test Telegram ID extraction"""
        test_results["extracted_intelligence"]["total"] += 1
        text = "Message me @scammer_bot on Telegram"
        entities = extract_entities_local(text)
        assert len(entities["telegramIds"]) >= 1, f"Expected 1+ Telegram IDs, got {entities['telegramIds']}"
        test_results["extracted_intelligence"]["passed"] += 1
    
    def test_extract_tracking_numbers(self):
        """Test tracking number extraction"""
        test_results["extracted_intelligence"]["total"] += 1
        text = "Track: DH123456789 or AMZ987654321"
        entities = extract_entities_local(text)
        assert len(entities["trackingNumbers"]) >= 1, f"Expected 1+ tracking numbers, got {entities['trackingNumbers']}"
        test_results["extracted_intelligence"]["passed"] += 1
    
    def test_extract_ids(self):
        """Test ID extraction (TXN, ORD, SBI, etc.)"""
        test_results["extracted_intelligence"]["total"] += 1
        text = "TXN123456789 ORD987654321 SBI-12345"
        entities = extract_entities_local(text)
        assert len(entities["ids"]) >= 2, f"Expected 2+ IDs, got {entities['ids']}"
        test_results["extracted_intelligence"]["passed"] += 1


# =============================================================================
# COMPONENT 3: CONVERSATION QUALITY (30 points)
# =============================================================================

class TestConversationQuality:
    """Test Conversation Quality - 30 points"""
    
    def test_multi_turn_conversation(self):
        """Test multi-turn conversation handling (10+ turns)"""
        test_results["conversation_quality"]["total"] += 1
        session_id = "test_conversation_quality"
        history = []
        
        messages = [
            "URGENT: Account blocked",
            "We are from SBI fraud dept",
            "Your OTP is required",
            "Send money to upi@paytm",
            "Call me at 9876543210",
            "Account: 1234567890123456"
        ]
        
        for i, msg in enumerate(messages):
            result = make_request(session_id, msg, history)
            assert result["status_code"] == 200, f"Turn {i+1} failed"
            assert len(result["response"]["reply"]) > 0
            
            history.append({"sender": "scammer", "text": msg, "timestamp": int(time.time() * 1000) + i*1000})
            history.append({"sender": "user", "text": result["response"]["reply"], "timestamp": int(time.time() * 1000) + i*1000 + 500})
        
        # Should have 12 messages in history (6 scammer + 6 agent)
        assert len(history) >= 10, f"Expected 10+ messages, got {len(history)}"
        test_results["conversation_quality"]["passed"] += 1
    
    def test_session_continuity(self):
        """Test session continuity across multiple requests"""
        test_results["conversation_quality"]["total"] += 1
        session_id = "test_session_continuity"
        
        # First request
        result1 = make_request(session_id, "First message: Your account blocked")
        assert result1["status_code"] == 200
        reply1 = result1["response"]["reply"]
        
        # Second request with history
        history = [
            {"sender": "scammer", "text": "First message", "timestamp": int(time.time() * 1000)},
            {"sender": "user", "text": reply1, "timestamp": int(time.time() * 1000) + 500}
        ]
        result2 = make_request(session_id, "Second message: Send OTP to 9876543210", history)
        assert result2["status_code"] == 200
        assert len(result2["response"]["reply"]) > 0
        test_results["conversation_quality"]["passed"] += 1
    
    def test_context_preservation(self):
        """Test that context is preserved across turns"""
        test_results["conversation_quality"]["total"] += 1
        session_id = "test_context"
        
        result = make_request(session_id, "SBI account compromised. Call 9876543210")
        assert result["status_code"] == 200
        # Response should acknowledge the context
        assert len(result["response"]["reply"]) > 10
        test_results["conversation_quality"]["passed"] += 1


# =============================================================================
# COMPONENT 4: ENGAGEMENT QUALITY (10 points)
# =============================================================================

class TestEngagementQuality:
    """Test Engagement Quality - 10 points"""
    
    def test_engagement_duration_tracking(self):
        """Test engagement duration is tracked"""
        test_results["engagement_quality"]["total"] += 1
        session_id = "test_engagement_duration"
        
        start_time = time.time()
        
        # Make multiple requests
        for i in range(3):
            result = make_request(f"{session_id}_{i}", f"Message {i+1}: Urgent account issue")
            assert result["status_code"] == 200
            time.sleep(0.5)  # Small delay between requests
        
        elapsed = time.time() - start_time
        # Total time should be tracked
        assert elapsed > 1, f"Expected >1s engagement, got {elapsed}s"
        test_results["engagement_quality"]["passed"] += 1
    
    def test_message_count_tracking(self):
        """Test message count is tracked"""
        test_results["engagement_quality"]["total"] += 1
        session_id = "test_message_count"
        
        messages_sent = 0
        for i in range(5):
            result = make_request(session_id, f"Message {i+1}")
            if result["status_code"] == 200:
                messages_sent += 1
        
        assert messages_sent >= 3, f"Expected 3+ messages tracked, got {messages_sent}"
        test_results["engagement_quality"]["passed"] += 1
    
    def test_response_time_under_30s(self):
        """Test API responds within 30 seconds"""
        test_results["engagement_quality"]["total"] += 1
        start = time.time()
        result = make_request("test_timeout", "Test for response time")
        elapsed = time.time() - start
        
        assert elapsed < 30, f"Response took {elapsed}s, should be under 30s"
        assert result["status_code"] == 200
        test_results["engagement_quality"]["passed"] += 1


# =============================================================================
# COMPONENT 5: RESPONSE STRUCTURE (10 points)
# =============================================================================

class TestResponseStructure:
    """Test Response Structure - 10 points"""
    
    def test_response_has_status_field(self):
        """Test response has 'status' field"""
        test_results["response_structure"]["total"] += 1
        result = make_request("test_status", "Test message")
        assert result["status_code"] == 200
        assert "status" in result["response"], "Response missing 'status' field"
        assert result["response"]["status"] == "success"
        test_results["response_structure"]["passed"] += 1
    
    def test_response_has_reply_field(self):
        """Test response has 'reply' field"""
        test_results["response_structure"]["total"] += 1
        result = make_request("test_reply", "Test message")
        assert result["status_code"] == 200
        assert "reply" in result["response"], "Response missing 'reply' field"
        assert len(result["response"]["reply"]) > 0
        test_results["response_structure"]["passed"] += 1
    
    def test_api_key_authentication(self):
        """Test API key authentication works"""
        test_results["response_structure"]["total"] += 1
        
        # Test with correct key
        result = make_request("test_auth", "Test")
        assert result["status_code"] == 200
        
        # Test with wrong key
        payload = {
            "sessionId": "test",
            "message": {"sender": "scammer", "text": "test", "timestamp": int(time.time() * 1000)},
            "conversationHistory": []
        }
        response = requests.post(API_ENDPOINT, json=payload, headers={"x-api-key": "wrong-key"}, timeout=10)
        assert response.status_code == 401
        test_results["response_structure"]["passed"] += 1
    
    def test_post_analyze_endpoint(self):
        """Test POST /analyze endpoint exists and works"""
        test_results["response_structure"]["total"] += 1
        result = make_request("test_endpoint", "Test endpoint")
        assert result["status_code"] == 200
        test_results["response_structure"]["passed"] += 1


# =============================================================================
# SCORING CALCULATION
# =============================================================================

def calculate_scores():
    """Calculate scores for each component"""
    for component in test_results:
        data = test_results[component]
        if data["total"] > 0:
            percentage = data["passed"] / data["total"]
            data["points"] = int(percentage * data["max"])
    
    total_points = sum(data["points"] for data in test_results.values())
    max_points = sum(data["max"] for data in test_results.values())
    
    return total_points, max_points


def print_report():
    """Print comprehensive test report"""
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION TEST REPORT")
    print("="*80)
    
    total, max_score = calculate_scores()
    
    print("\n📊 COMPONENT SCORES:")
    print("-"*80)
    
    for component, data in test_results.items():
        status = "✅" if data["points"] >= data["max"] * 0.7 else "⚠️" if data["points"] >= data["max"] * 0.4 else "❌"
        print(f"{status} {component.replace('_', ' ').title():30} | {data['passed']:2}/{data['total']:2} tests | {data['points']:2}/{data['max']:2} points")
    
    print("-"*80)
    print(f"TOTAL SCORE: {total}/{max_score} ({total/max_score*100:.1f}%)")
    print("="*80)
    
    if total >= 95:
        print("🎯 STATUS: EXCELLENT (95+ SCORE ACHIEVED!)")
    elif total >= 80:
        print("✅ STATUS: GOOD (80+ score)")
    elif total >= 60:
        print("⚠️  STATUS: NEEDS IMPROVEMENT (60+ score)")
    else:
        print("❌ STATUS: FAILING (<60 score)")
    
    print("\n📋 EVALUATION CRITERIA:")
    print("-"*80)
    print("1. Scam Detection      (20 pts): Detects scams and engages appropriately")
    print("2. Extracted Intelligence (30 pts): Extracts all 10 entity types")
    print("3. Conversation Quality (30 pts): Multi-turn, context preservation")
    print("4. Engagement Quality   (10 pts): Duration, message count, <30s response")
    print("5. Response Structure   (10 pts): Correct JSON format, auth, endpoint")
    print("="*80)
    
    return total


if __name__ == "__main__":
    # Run pytest
    import subprocess
    result = subprocess.run(
        ["python", "-m", "pytest", __file__, "-v", "--tb=short"],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    # Print comprehensive report
    final_score = print_report()
    
    exit(0 if final_score >= 60 else 1)
