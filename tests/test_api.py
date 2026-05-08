#!/usr/bin/env python3
"""Quick test of the API endpoint"""

import requests
import time
import json

time.sleep(2)  # Wait for server

test_data = {
    'heart_rate': 72,
    'hrv': 150,
    'respiration': 16,
    'skin_temp': 35.8,
    'bp_systolic': 120,
    'bp_diastolic': 80,
    'cognitive_state': 5,
    'emotional_state': 5
}

print("Testing API endpoint...")
print(f"Sending data: {json.dumps(test_data, indent=2)}\n")

try:
    response = requests.post('http://127.0.0.1:5001/api/predict', 
                            json=test_data, 
                            timeout=5)
    
    print(f"Status Code: {response.status_code}\n")
    
    if response.status_code == 200:
        result = response.json()
        print("✅ SUCCESS! API is working!\n")
        print(f"Stress Level: {result.get('stress_level')}")
        print(f"Confidence: {result.get('confidence')}%")
        if 'recommendation' in result:
            rec = result['recommendation']
            print(f"Recommendation: {rec.get('title', 'N/A')}")
    else:
        print(f"❌ Error {response.status_code}\n")
        print(f"Response: {response.text[:300]}")
        
except Exception as e:
    print(f"❌ Connection Error: {e}")
