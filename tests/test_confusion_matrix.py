import requests
import time

print("[Testing Confusion Matrix Integration...]")

# Wait a bit for server to be ready
time.sleep(2)

try:
    # Test about page
    resp = requests.get('http://127.0.0.1:5000/about', timeout=10)
    
    if resp.status_code == 200:
        print("[+] About page loaded (HTTP 200)")
        
        if 'Confusion Matrix' in resp.text:
            print("[OK] Confusion Matrix section found")
        else:
            print("[FAIL] Confusion Matrix section NOT found")
        
        if 'confusion_matrix.png' in resp.text:
            print("[OK] Image reference found in HTML")
        else:
            print("[FAIL] Image reference NOT found")
        
        if 'Per-Class Performance' not in resp.text:
            print("[OK] Per-Class Performance section successfully removed")
        else:
            print("[WARNING] Per-Class Performance still present in HTML")
        
        if 'High-Stress Detection' in resp.text:
            print("[OK] Key Insights section found")
        else:
            print("[WARNING] Key Insights section missing")
            
    else:
        print(f"[FAIL] HTTP {resp.status_code}")
        
    # Test image file is accessible
    img_resp = requests.get('http://127.0.0.1:5000/static/confusion_matrix.png', timeout=5)
    if img_resp.status_code == 200:
        print("[OK] Confusion matrix image is accessible and served")
    else:
        print(f"[FAIL] Image returned HTTP {img_resp.status_code}")
        
except Exception as e:
    print(f"[ERROR] {e}")
