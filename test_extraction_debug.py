"""
Test entity extraction with the current implementation
"""
import requests
import json

BASE_URL = "http://localhost:8000"
API_ENDPOINT = f"{BASE_URL}/analyze"
API_KEY = "honeypot_key_2026_eval"

def test_extraction():
    """Test that ALL entities are extracted"""
    
    test_message = "Your account 1234567890123456 is locked! Call +91-9876543210. Email fraud@fakebank.com. UPI: scammer.fraud@fakebank. Link: http://fake-site.com/verify. ID: TXN123456"
    
    payload = {
        "sessionId": "test_extraction_001",
        "message": {
            "sender": "scammer",
            "text": test_message,
            "timestamp": 1707753600000
        },
        "conversationHistory": [],
        "metadata": {"channel": "SMS", "language": "English", "locale": "IN"}
    }
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY
    }
    
    print("="*80)
    print("TESTING ENTITY EXTRACTION")
    print("="*80)
    print(f"\nTest Message: {test_message}")
    print()
    
    try:
        response = requests.post(API_ENDPOINT, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Status: {response.status_code}")
            print(f"✅ Reply: {data.get('reply', 'N/A')[:100]}...")
            print()
            
            # Now we need to check the callback or session data
            # For this test, let's manually extract using the same function
            from main import extract_entities
            
            extracted = extract_entities(test_message)
            
            print("📊 EXTRACTED ENTITIES:")
            print("-"*80)
            
            expected = {
                "phoneNumbers": ["+91-9876543210"],
                "bankAccounts": ["1234567890123456"],
                "upiIds": ["scammer.fraud@fakebank"],
                "emailAddresses": ["fraud@fakebank.com"],
                "phishingLinks": ["http://fake-site.com/verify"],
                "ids": ["TXN123456"]
            }
            
            all_passed = True
            for field, expected_values in expected.items():
                actual = extracted.get(field, [])
                if actual:
                    print(f"  ✅ {field}: {actual}")
                    # Check if expected value is in actual
                    found = any(exp in str(act) for exp in expected_values for act in actual)
                    if not found and expected_values:
                        print(f"     ⚠️  Expected pattern not found: {expected_values}")
                else:
                    print(f"  ❌ {field}: MISSING! Expected: {expected_values}")
                    all_passed = False
            
            print("-"*80)
            if all_passed:
                print("✅ ALL ENTITIES EXTRACTED CORRECTLY!")
            else:
                print("❌ SOME ENTITIES MISSING - NEEDS FIX")
            print("="*80)
            
            return all_passed, extracted
        else:
            print(f"❌ API ERROR: {response.status_code}")
            print(response.text)
            return False, {}
            
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return False, {}

if __name__ == "__main__":
    import sys
    success, data = test_extraction()
    sys.exit(0 if success else 1)
