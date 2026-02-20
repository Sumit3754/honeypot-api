# Honeypot API - 95+ Score Ready

A hackathon-ready API that analyzes scam messages, extracts intelligence, and provides agentic responses to waste scammers' time. **Optimized for 95+ scoring compliance**.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key (required)
export HONEYPOT_API_KEY="honeypot_key_2026_eval"


# Run server
uvicorn main:app --reload
```

## Features

- **Comprehensive Entity Extraction**: Phone numbers, bank accounts, UPI IDs, emails, credit cards, Bitcoin addresses, Telegram IDs, tracking numbers, and IDs
- **Multi-Language Support**: English and Hinglish (Roman Hindi)
- **Dynamic Personas**: Grandma, Student, Skeptic, Parent - automatically selected based on scam type
- **Scam Type Detection**: Automatically categorizes Bank Fraud, UPI Fraud, Phishing, Loan Scam, KYC Scam, Job Scam, Courier Scam, Sextortion, Digital Arrest, Utility Scam, Lottery Scam
- **Conversation Quality Tracking**: Turn count, questions asked, red flags, elicitation attempts
- **Final Output Compliance**: All required fields for 95+ scoring

## API Specification

### POST /analyze

**Headers:**
- `Content-Type`: `application/json`
- `x-api-key`: `honeypot_key_2026_eval`

**Request:**
```json
{
  "sessionId": "uuid-v4-string",
  "message": {
    "sender": "scammer",
    "text": "URGENT: Your account has been compromised...",
    "timestamp": 1707753600000
  },
  "conversationHistory": [],
  "metadata": {"channel": "SMS", "language": "English", "locale": "IN"}
}
```

**Response:**
```json
{
  "status": "success",
  "reply": "Your honeypot's response to the scammer"
}
```

## Final Output Format

```json
{
  "sessionId": "abc123",
  "scamDetected": true,
  "totalMessagesExchanged": 18,
  "engagementDurationSeconds": 240,
  "extractedIntelligence": {
    "phoneNumbers": ["+91-9876543210"],
    "bankAccounts": ["1234567890123456"],
    "upiIds": ["scammer.fraud@oksbi"],
    "phishingLinks": ["http://malicious-site.com"],
    "emailAddresses": ["scammer@fake.com"],
    "creditCards": ["4532-7890-1234-5678"],
    "bitcoinAddresses": ["1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"],
    "telegramIds": ["@scammer_bot"],
    "trackingNumbers": ["DH123456789"],
    "ids": ["TXN123456", "SBI-12345"]
  },
  "agentNotes": "SCAM DETECTED: Bank Fraud. Persona 'grandma' used...",
  "scamType": "Bank Fraud",
  "confidenceLevel": 0.95
}
```

## Entity Extraction Coverage

| Entity Type | Pattern Examples |
|-------------|------------------|
| Phone Numbers (Indian) | `+91-9876543210`, `9876543210` |
| Phone Numbers (US) | `+1-555-0123456` |
| Toll-free | `1800-123-4567` |
| Bank Accounts | 9-18 digit numbers |
| UPI IDs | `user@paytm`, `user@oksbi` |
| Credit Cards | `4532-7890-1234-5678` |
| Email Addresses | `user@example.com` |
| Bitcoin Addresses | `1A1z...`, `bc1...` |
| Telegram IDs | `@username` |
| Tracking Numbers | `DH...`, `AMZ...`, `UPS...` |
| IDs | `TXN123`, `SBI-12345`, `ORD987` |
| Phishing Links | `http://`, `https://`, `www.` |

## Testing

```bash
# Entity extraction validation
python validate_entities.py

# Full test suite
python -m pytest test_comprehensive_95plus.py -v

# System check
python full_system_check.py
```

## Deployment

```bash
# Railway (recommended)
railway login
railway up

# Or use Docker
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Submission

```json
{
  "deployed_url": "https://your-api.com/analyze",
  "api_key": "honeypot_key_2026_eval",
  "github_url": "https://github.com/username/honeypot-api"
}
```

## Status: Ready for 95+ Score Evaluation

See `VALIDATION_SUMMARY.md` for complete validation details.
