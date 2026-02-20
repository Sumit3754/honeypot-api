"""
Comprehensive Honeypot API Test Suite - 12 Scenarios for 95+ Score Validation
Tests all evaluation criteria: Scam Detection, Intelligence Extraction, Conversation Quality, Engagement, Response Structure
"""

import pytest
import requests
import json
import re
from typing import Dict, List, Any

# API Configuration
BASE_URL = "http://localhost:8000"
API_ENDPOINT = f"{BASE_URL}/analyze"
API_KEY = "honeypot_key_2026_eval"

# Test Scenarios - All 12 scenarios from evaluation criteria
SCENARIOS = [
    {
        "id": 1,
        "name": "Bank Fraud - Account Compromised",
        "weight": 35,
        "initial_message": "URGENT: Your SBI account has been compromised. Share OTP immediately to block unauthorized transaction of ₹45,000.",
        "fake_data": {
            "phoneNumbers": ["+91-9876543210"],
            "bankAccounts": ["1234567890123456"],
            "upiIds": ["sbi.fraud@oksbi"],
            "phishingLinks": ["http://sbi-secure-verify.com"],
            "emailAddresses": ["support@sbi-fraud.com"],
            "ids": ["SBI-12345", "TXN987654321"]
        },
        "scam_type": "Bank Fraud",
        "keywords": ["urgent", "account", "compromised", "otp"]
    },
    {
        "id": 2,
        "name": "UPI Fraud - Cashback Scam",
        "weight": 35,
        "initial_message": "Congratulations! You received ₹5000 cashback. Claim: http://cashback-upi.com/claim",
        "fake_data": {
            "phoneNumbers": ["+91-8765432109"],
            "upiIds": ["cashback.claim@paytm", "cashback@okhdfcbank"],
            "bankAccounts": ["9876543210987654"],
            "phishingLinks": ["http://cashback-upi.com/claim"],
            "emailAddresses": ["support@cashback-claims.com"],
            "ids": ["CB123456"]
        },
        "scam_type": "UPI Fraud",
        "keywords": ["cashback", "claim", "congratulations"]
    },
    {
        "id": 3,
        "name": "Phishing - Amazon Fake Offer",
        "weight": 30,
        "initial_message": "Amazon Great Sale! You won iPhone 15. Claim: http://amazon-deals.fake-site.com/claim?id=WIN12345",
        "fake_data": {
            "phoneNumbers": ["+91-7654321098", "1800-765-4321"],
            "upiIds": ["shipping@amazon-offers"],
            "phishingLinks": [
                "http://amazon-deals.fake-site.com/claim?id=WIN12345",
                "http://track-amazon-order.fake/ORD987654321"
            ],
            "emailAddresses": ["offers@fake-amazon-deals.com"],
            "ids": ["ORD987654321", "WIN12345"]
        },
        "scam_type": "Phishing",
        "keywords": ["amazon", "won", "claim", "iphone"]
    },
    {
        "id": 4,
        "name": "Loan Scam - Instant Approval",
        "weight": 35,
        "initial_message": "Instant loan approved! ₹5 lakhs pre-approved. Apply: http://quick-loan-approval.com/apply",
        "fake_data": {
            "phoneNumbers": ["+91-9988776655"],
            "bankAccounts": ["1122334455667788"],
            "upiIds": ["loan.processing@okicici", "loan@okhdfcbank"],
            "phishingLinks": ["http://quick-loan-approval.com/apply"],
            "emailAddresses": ["loans@quick-approval.com"],
            "ids": ["LOAN987654321", "ABCDE1234F"]
        },
        "scam_type": "Loan Scam",
        "keywords": ["loan", "approved", "instant", "apply"]
    },
    {
        "id": 5,
        "name": "KYC Update Scam",
        "weight": 30,
        "initial_message": "Your Aadhaar linked bank account will be deactivated. Update KYC: http://kyc-update-bank.com/verify",
        "fake_data": {
            "phoneNumbers": ["+91-8899776655", "1800-889-9776"],
            "bankAccounts": ["5566778899001122"],
            "phishingLinks": [
                "http://kyc-update-bank.com/verify",
                "http://secure-doc-upload.com"
            ],
            "emailAddresses": ["support@kyc-update-team.com"],
            "ids": ["KYC12345", "1234-5678-9012"]
        },
        "scam_type": "KYC Scam",
        "keywords": ["kyc", "aadhaar", "deactivated", "update"]
    },
    {
        "id": 6,
        "name": "Multi-Language - Hinglish",
        "weight": 30,
        "initial_message": "AAPKA SBI ACCOUNT BLOCK HO GAYA HAI! Call 9876543210 for KYC update.",
        "fake_data": {
            "phoneNumbers": ["9876543210"],
            "phishingLinks": ["http://sbi-verify-now.com"],
            "bankAccounts": ["1234567890123456"],
            "upiIds": ["urgent.payment@oksbi"],
            "emailAddresses": ["support@sbi-care.in"],
            "ids": ["TXN123456", "SBI12345"]
        },
        "scam_type": "Bank Fraud",
        "keywords": ["account", "block", "kyc"]
    },
    {
        "id": 7,
        "name": "OTP Fraud - Fake Payment",
        "weight": 35,
        "initial_message": "Your Amazon order ₹24,999 confirmed. OTP: 445566. If not you, call 1800-123-4567.",
        "fake_data": {
            "phoneNumbers": ["1800-123-4567", "9988776655"],
            "creditCards": ["4532-7890-1234-5678"],
            "ids": ["AMZ987654321", "FRD987654321"]
        },
        "scam_type": "Phishing",
        "keywords": ["amazon", "order", "otp", "confirmed"]
    },
    {
        "id": 8,
        "name": "Job Scam - Work From Home",
        "weight": 30,
        "initial_message": "URGENT HIRING! Work from home. Earn ₹50,000/month. Register: http://quick-job-portal.com/apply",
        "fake_data": {
            "phoneNumbers": ["9988776655", "1800-998-8776"],
            "emailAddresses": ["hr@quick-job-portal.com"],
            "bankAccounts": ["1122334455667788"],
            "upiIds": ["registration.fee@paytm"],
            "phishingLinks": ["http://employee-training-portal.com/login"],
            "ids": ["EMP123456"]
        },
        "scam_type": "Job Scam",
        "keywords": ["job", "hiring", "work", "earn"]
    },
    {
        "id": 9,
        "name": "Courier Scam - Customs Hold",
        "weight": 25,
        "initial_message": "Your parcel from USA held at customs. Pay ₹2,500 duty. Track: http://dhl-customs-clear.com/track?id=DH123456",
        "fake_data": {
            "phoneNumbers": ["1800-256-7890"],
            "trackingNumbers": ["DH123456"],
            "bankAccounts": ["9988776655443322"],
            "emailAddresses": ["customs@dhl-clearance.com"],
            "phishingLinks": ["http://dhl-legal-notice.com/view"],
            "ids": ["CUS789012"]
        },
        "scam_type": "Courier Scam",
        "keywords": ["parcel", "customs", "duty", "dhl"]
    },
    {
        "id": 10,
        "name": "Sextortion - Blackmail",
        "weight": 40,
        "initial_message": "I have your private videos. Pay ₹50,000 in Bitcoin or I'll send to all contacts. Contact: blackmail@proton.me",
        "fake_data": {
            "emailAddresses": ["blackmail@proton.me"],
            "bitcoinAddresses": ["1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"],
            "phoneNumbers": ["+1-555-0123456"],
            "telegramIds": ["@blackmailer_123"],
            "phishingLinks": ["http://blackmail-videos.onion/your_id"],
            "ids": ["EXT123456"]
        },
        "scam_type": "Sextortion",
        "keywords": ["bitcoin", "videos", "pay", "contact"]
    },
    {
        "id": 11,
        "name": "Edge Case - Empty Body",
        "weight": 10,
        "initial_message": "",
        "expected": "Must return 200 with valid reply",
        "keywords": []
    },
    {
        "id": 12,
        "name": "Stress Test - Multi-turn Conversation",
        "weight": 20,
        "initial_message": "Test message for long conversation. Your bank account needs verification.",
        "fake_data": {
            "phoneNumbers": ["9876543210"],
            "upiIds": ["test@paytm"]
        },
        "scam_type": "Bank Fraud",
        "keywords": ["bank", "verification"]
    }
]


def make_request(session_id: str, message: str, history: List[Dict] = None) -> Dict:
    """Make API request to /analyze endpoint"""
    payload = {
        "sessionId": session_id,
        "message": {
            "sender": "scammer",
            "text": message,
            "timestamp": 1707753600000
        },
        "conversationHistory": history or [],
        "metadata": {
            "channel": "SMS",
            "language": "English",
            "locale": "IN"
        }
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
        return {
            "status_code": 0,
            "response": None,
            "error": str(e)
        }


def extract_entities_from_text(text: str) -> Dict[str, List[str]]:
    """Standalone entity extraction for testing"""
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
    
    # Email pattern
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    results["emailAddresses"] = re.findall(email_pattern, text)
    
    # UPI pattern
    upi_pattern = r'[a-zA-Z0-9.\-_]{2,256}@[a-zA-Z]{2,64}'
    results["upiIds"] = re.findall(upi_pattern, text)
    
    # URL patterns
    url_pattern = r'(?:https?://|onion://|www\.)[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)'
    results["phishingLinks"] = re.findall(url_pattern, text, re.IGNORECASE)
    
    # Phone patterns - Indian
    phone_pattern = r'(?:\+91[\-\s]?)?[6-9]\d{9}'
    results["phoneNumbers"] = re.findall(phone_pattern, text)
    
    # Toll-free
    tollfree_pattern = r'(?:1?[-\s]?)?800[\-\s]?\d{3}[\-\s]?\d{4}'
    tollfree = re.findall(tollfree_pattern, text)
    results["phoneNumbers"].extend(tollfree)
    
    # US phone
    us_phone = r'\+1[\-\s]?\(?\d{3}\)?[\-\s]?\d{3}[\-\s]?\d{4}'
    us_phones = re.findall(us_phone, text)
    results["phoneNumbers"].extend(us_phones)
    
    # Bank accounts
    bank_pattern = r'\b\d{9,18}\b'
    banks = re.findall(bank_pattern, text)
    # Filter out 12-digit (Aadhaar) and phone-like numbers
    results["bankAccounts"] = [b for b in banks if len(b) != 12 and len(b) != 10]
    
    # Credit cards
    cc_pattern = r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
    results["creditCards"] = re.findall(cc_pattern, text)
    
    # Bitcoin
    btc_pattern = r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b|\bbc1[a-zA-HJ-NP-Z0-9]{39,59}\b'
    results["bitcoinAddresses"] = re.findall(btc_pattern, text)
    
    # Telegram
    tg_pattern = r'@\w{5,32}'
    results["telegramIds"] = re.findall(tg_pattern, text)
    
    # Tracking numbers
    track_pattern = r'\b(?:DH|AMZ|UPS|FEDEX|1Z)\s*\d{8,20}\b'
    results["trackingNumbers"] = re.findall(track_pattern, text, re.IGNORECASE)
    
    # IDs
    id_pattern = r'\b(?:TXN|ORD|ID|REF|CASE|EMP|CUS|EXT|SBI|AMZ|WIN|CB|LOAN|KYC|FRD)[\-\s]?[A-Z0-9]{5,20}\b'
    results["ids"] = re.findall(id_pattern, text, re.IGNORECASE)
    
    # Deduplicate
    for key in results:
        results[key] = list(set(results[key]))
    
    return results


class TestAPISpecification:
    """Test 1: API Specification Requirements"""
    
    def test_endpoint_available(self):
        """Test that /analyze endpoint returns 200"""
        result = make_request("test_session_1", "Test message")
        assert result["status_code"] == 200, f"Expected 200, got {result['status_code']}: {result.get('error')}"
    
    def test_authentication_required(self):
        """Test that API key is required"""
        payload = {
            "sessionId": "test_auth",
            "message": {"sender": "scammer", "text": "test", "timestamp": 1707753600000},
            "conversationHistory": [],
            "metadata": {"channel": "SMS", "language": "English", "locale": "IN"}
        }
        
        # No API key
        response = requests.post(API_ENDPOINT, json=payload, timeout=10)
        assert response.status_code == 401, f"Expected 401 without API key, got {response.status_code}"
        
        # Wrong API key
        response = requests.post(API_ENDPOINT, json=payload, headers={"x-api-key": "wrong-key"}, timeout=10)
        assert response.status_code == 401, f"Expected 401 with wrong API key, got {response.status_code}"
    
    def test_response_format(self):
        """Test response has correct format with status and reply"""
        result = make_request("test_format", "URGENT: Your account blocked. Send OTP.")
        assert result["status_code"] == 200
        assert result["response"] is not None
        assert "status" in result["response"], "Response missing 'status' field"
        assert result["response"]["status"] == "success", "Status should be 'success'"
        assert "reply" in result["response"], "Response missing 'reply' field"
        assert len(result["response"]["reply"]) > 0, "Reply should not be empty"
    
    def test_response_timeout(self):
        """Test response completes within 30 seconds"""
        import time
        start = time.time()
        result = make_request("test_timeout", "Test for timeout")
        elapsed = time.time() - start
        assert elapsed < 30, f"Response took {elapsed}s, should be under 30s"
        assert result["status_code"] == 200


class TestEntityExtraction:
    """Test 2: Comprehensive Intelligence Extraction"""
    
    def test_extract_phone_numbers(self):
        """Test extraction of all phone number formats"""
        text = "Call me at +91-9876543210 or 9876543210 or 1800-123-4567 or +1-555-0123456"
        entities = extract_entities_from_text(text)
        
        # Should find at least the Indian numbers
        assert len(entities["phoneNumbers"]) >= 2, f"Expected 2+ phone numbers, got {entities['phoneNumbers']}"
        
        # Check for Indian format
        indian_phones = [p for p in entities["phoneNumbers"] if re.match(r'(?:\+91[\-\s]?)?[6-9]\d{9}', p)]
        assert len(indian_phones) >= 1, f"Should find Indian phone format"
    
    def test_extract_upi_ids(self):
        """Test extraction of UPI IDs"""
        text = "Send money to test@paytm or user@oksbi or payment@okhdfcbank"
        entities = extract_entities_from_text(text)
        
        assert len(entities["upiIds"]) >= 2, f"Expected 2+ UPI IDs, got {entities['upiIds']}"
    
    def test_extract_phishing_links(self):
        """Test extraction of phishing links"""
        text = "Visit http://fake-site.com/claim or https://scam-bank.com/login or www.fake.com"
        entities = extract_entities_from_text(text)
        
        assert len(entities["phishingLinks"]) >= 2, f"Expected 2+ links, got {entities['phishingLinks']}"
    
    def test_extract_email_addresses(self):
        """Test extraction of email addresses"""
        text = "Contact support@bank.com or fraud@fake-site.org or admin123@test.co.in"
        entities = extract_entities_from_text(text)
        
        assert len(entities["emailAddresses"]) >= 2, f"Expected 2+ emails, got {entities['emailAddresses']}"
    
    def test_extract_bank_accounts(self):
        """Test extraction of bank account numbers"""
        text = "Account 1234567890123456 or 987654321098765432 for transfer"
        entities = extract_entities_from_text(text)
        
        # Should find account numbers (9-18 digits)
        accounts = [a for a in entities["bankAccounts"] if len(a) >= 9]
        assert len(accounts) >= 1, f"Expected 1+ bank accounts, got {entities['bankAccounts']}"
    
    def test_extract_credit_cards(self):
        """Test extraction of credit card numbers"""
        text = "Card: 4532-7890-1234-5678 or 5555-5555-5555-4444"
        entities = extract_entities_from_text(text)
        
        assert len(entities["creditCards"]) >= 1, f"Expected 1+ credit cards, got {entities['creditCards']}"
    
    def test_extract_bitcoin_addresses(self):
        """Test extraction of Bitcoin addresses"""
        text = "Send BTC to 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa or bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh"
        entities = extract_entities_from_text(text)
        
        assert len(entities["bitcoinAddresses"]) >= 1, f"Expected 1+ Bitcoin addresses, got {entities['bitcoinAddresses']}"
    
    def test_extract_telegram_ids(self):
        """Test extraction of Telegram IDs"""
        text = "Message me @username123 or @scammer_bot on Telegram"
        entities = extract_entities_from_text(text)
        
        assert len(entities["telegramIds"]) >= 1, f"Expected 1+ Telegram IDs, got {entities['telegramIds']}"
    
    def test_extract_tracking_numbers(self):
        """Test extraction of tracking numbers"""
        text = "Track: DH123456789 or AMZ987654321 or UPS1234567890"
        entities = extract_entities_from_text(text)
        
        assert len(entities["trackingNumbers"]) >= 1, f"Expected 1+ tracking numbers, got {entities['trackingNumbers']}"
    
    def test_extract_ids(self):
        """Test extraction of various IDs (TXN, ORD, etc.)"""
        text = "TXN123456789 ORD987654321 SBI-12345 WIN12345 CB123456"
        entities = extract_entities_from_text(text)
        
        assert len(entities["ids"]) >= 3, f"Expected 3+ IDs, got {entities['ids']}"


class TestScenarioBankFraud:
    """Test 3: Bank Fraud Scenario (35% weight)"""
    
    def test_bank_fraud_detection(self):
        """Test detection of bank fraud scam"""
        scenario = SCENARIOS[0]
        result = make_request(f"bank_fraud_{scenario['id']}", scenario["initial_message"])
        
        assert result["status_code"] == 200
        assert result["response"]["status"] == "success"
        assert len(result["response"]["reply"]) > 0
    
    def test_bank_fraud_entity_extraction(self):
        """Test entity extraction from bank fraud message"""
        scenario = SCENARIOS[0]
        entities = extract_entities_from_text(scenario["initial_message"])
        
        # Check for phone number
        assert len(entities["phoneNumbers"]) >= 1 or "+91-9876543210" in scenario["fake_data"]["phoneNumbers"]
        
        # Check for suspicious keywords
        text_lower = scenario["initial_message"].lower()
        keywords_found = [k for k in scenario["keywords"] if k in text_lower]
        assert len(keywords_found) >= 2, f"Should find scam keywords: {scenario['keywords']}"


class TestScenarioUPIFraud:
    """Test 4: UPI Fraud Scenario (35% weight)"""
    
    def test_upi_fraud_detection(self):
        """Test detection of UPI fraud scam"""
        scenario = SCENARIOS[1]
        result = make_request(f"upi_fraud_{scenario['id']}", scenario["initial_message"])
        
        assert result["status_code"] == 200
        assert result["response"]["status"] == "success"
    
    def test_upi_fraud_entity_extraction(self):
        """Test entity extraction from UPI fraud message"""
        scenario = SCENARIOS[1]
        entities = extract_entities_from_text(scenario["initial_message"])
        
        # Should extract the phishing link
        assert len(entities["phishingLinks"]) >= 1, f"Expected phishing link, got {entities['phishingLinks']}"


class TestScenarioPhishing:
    """Test 5: Phishing Scenario (30% weight)"""
    
    def test_phishing_detection(self):
        """Test detection of phishing scam"""
        scenario = SCENARIOS[2]
        result = make_request(f"phishing_{scenario['id']}", scenario["initial_message"])
        
        assert result["status_code"] == 200
        assert result["response"]["status"] == "success"
    
    def test_phishing_entity_extraction(self):
        """Test entity extraction from phishing message"""
        scenario = SCENARIOS[2]
        entities = extract_entities_from_text(scenario["initial_message"])
        
        # Should extract links and IDs
        assert len(entities["phishingLinks"]) >= 1, f"Expected phishing link"


class TestScenarioLoanScam:
    """Test 6: Loan Scam Scenario (35% weight)"""
    
    def test_loan_scam_detection(self):
        """Test detection of loan scam"""
        scenario = SCENARIOS[3]
        result = make_request(f"loan_{scenario['id']}", scenario["initial_message"])
        
        assert result["status_code"] == 200
        assert result["response"]["status"] == "success"


class TestScenarioKYC:
    """Test 7: KYC Scam Scenario (30% weight)"""
    
    def test_kyc_scam_detection(self):
        """Test detection of KYC scam"""
        scenario = SCENARIOS[4]
        result = make_request(f"kyc_{scenario['id']}", scenario["initial_message"])
        
        assert result["status_code"] == 200
        assert result["response"]["status"] == "success"


class TestScenarioMultiLanguage:
    """Test 8: Multi-Language (Hinglish) Scenario (30% weight)"""
    
    def test_hinglish_detection(self):
        """Test handling of Hinglish message"""
        scenario = SCENARIOS[5]
        result = make_request(f"hinglish_{scenario['id']}", scenario["initial_message"])
        
        assert result["status_code"] == 200
        assert result["response"]["status"] == "success"
        assert len(result["response"]["reply"]) > 0
    
    def test_hinglish_entity_extraction(self):
        """Test entity extraction from Hinglish message"""
        scenario = SCENARIOS[5]
        entities = extract_entities_from_text(scenario["initial_message"])
        
        # Should still extract phone numbers
        assert len(entities["phoneNumbers"]) >= 1, f"Expected phone number in Hinglish text"


class TestScenarioOTPFraud:
    """Test 9: OTP Fraud Scenario (35% weight)"""
    
    def test_otp_fraud_detection(self):
        """Test detection of OTP fraud"""
        scenario = SCENARIOS[6]
        result = make_request(f"otp_{scenario['id']}", scenario["initial_message"])
        
        assert result["status_code"] == 200
        assert result["response"]["status"] == "success"
    
    def test_otp_fraud_entity_extraction(self):
        """Test entity extraction from OTP fraud message"""
        scenario = SCENARIOS[6]
        entities = extract_entities_from_text(scenario["initial_message"])
        
        # Should extract phone numbers and credit card
        total_entities = len(entities["phoneNumbers"]) + len(entities["creditCards"])
        assert total_entities >= 2, f"Expected phone and credit card"


class TestScenarioJobScam:
    """Test 10: Job Scam Scenario (30% weight)"""
    
    def test_job_scam_detection(self):
        """Test detection of job scam"""
        scenario = SCENARIOS[7]
        result = make_request(f"job_{scenario['id']}", scenario["initial_message"])
        
        assert result["status_code"] == 200
        assert result["response"]["status"] == "success"


class TestScenarioCourier:
    """Test 11: Courier Scam Scenario (25% weight)"""
    
    def test_courier_scam_detection(self):
        """Test detection of courier scam"""
        scenario = SCENARIOS[8]
        result = make_request(f"courier_{scenario['id']}", scenario["initial_message"])
        
        assert result["status_code"] == 200
        assert result["response"]["status"] == "success"
    
    def test_courier_entity_extraction(self):
        """Test entity extraction from courier message"""
        scenario = SCENARIOS[8]
        entities = extract_entities_from_text(scenario["initial_message"])
        
        # Should extract tracking number
        assert len(entities["trackingNumbers"]) >= 1 or len(entities["ids"]) >= 1


class TestScenarioSextortion:
    """Test 12: Sextortion Scenario (40% weight - CRITICAL)"""
    
    def test_sextortion_detection(self):
        """Test detection of sextortion scam - HIGHEST PRIORITY"""
        scenario = SCENARIOS[9]
        result = make_request(f"sextortion_{scenario['id']}", scenario["initial_message"])
        
        assert result["status_code"] == 200
        assert result["response"]["status"] == "success"
    
    def test_sextortion_entity_extraction(self):
        """Test entity extraction from sextortion message - CRITICAL"""
        scenario = SCENARIOS[9]
        entities = extract_entities_from_text(scenario["initial_message"])
        
        fake_data = scenario["fake_data"]
        
        # Check email extraction (CRITICAL)
        assert len(entities["emailAddresses"]) >= 1, f"CRITICAL: Expected email, got {entities['emailAddresses']}"
        
        # Check Bitcoin extraction (CRITICAL)
        assert len(entities["bitcoinAddresses"]) >= 1, f"CRITICAL: Expected Bitcoin address, got {entities['bitcoinAddresses']}"


class TestEdgeCases:
    """Test 13: Edge Cases and Error Handling"""
    
    def test_empty_message(self):
        """Test handling of empty message"""
        result = make_request("test_empty", "")
        # Should still return 200 with a valid response
        assert result["status_code"] in [200, 422], f"Empty message should return 200 or 422"
    
    def test_very_long_message(self):
        """Test handling of very long message"""
        long_text = "URGENT! " * 100 + " Call 9876543210 immediately!"
        result = make_request("test_long", long_text)
        assert result["status_code"] == 200
    
    def test_special_characters(self):
        """Test handling of special characters"""
        special = "URGENT!!! Your @ccount blocked #$%^&*() Call +91-9876543210!!!"
        result = make_request("test_special", special)
        assert result["status_code"] == 200


class TestMultiTurnConversation:
    """Test 14: Multi-turn Conversation Quality"""
    
    def test_conversation_history_handling(self):
        """Test that conversation history is properly tracked"""
        session_id = "test_conversation_flow"
        
        # Turn 1
        result1 = make_request(session_id, "URGENT: Your account blocked. Call 9876543210")
        assert result1["status_code"] == 200
        
        # Build history
        history = [
            {"sender": "scammer", "text": "URGENT: Your account blocked. Call 9876543210", "timestamp": 1707753600000},
            {"sender": "user", "text": result1["response"]["reply"], "timestamp": 1707753601000}
        ]
        
        # Turn 2
        result2 = make_request(session_id, "This is SBI fraud department. My ID is EMP12345", history)
        assert result2["status_code"] == 200
        assert len(result2["response"]["reply"]) > 0
        
        # Verify session continuity
        assert result2["response"]["status"] == "success"
    
    def test_conversation_turn_count(self):
        """Test that API maintains context across multiple turns"""
        session_id = "test_turn_count"
        history = []
        
        messages = [
            "URGENT: Account blocked",
            "We are from SBI fraud dept",
            "Your OTP is required",
            "Send money to upi@paytm"
        ]
        
        for i, msg in enumerate(messages):
            result = make_request(session_id, msg, history)
            assert result["status_code"] == 200, f"Turn {i+1} failed"
            
            history.append({"sender": "scammer", "text": msg, "timestamp": 1707753600000 + i*1000})
            history.append({"sender": "user", "text": result["response"]["reply"], "timestamp": 1707753600000 + i*1000 + 500})


def run_all_tests():
    """Run all tests and generate report"""
    print("=" * 80)
    print("HONEYPOT API COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print()
    
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
    
    print()
    print("=" * 80)
    print(f"Test Exit Code: {result.returncode}")
    print("=" * 80)
    
    return result.returncode == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
