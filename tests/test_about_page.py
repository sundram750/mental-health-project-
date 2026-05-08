import requests

print("[Testing About Page...]")

try:
    # Test home page with navbar
    response = requests.get('http://127.0.0.1:5000/', timeout=5)
    if response.status_code == 200:
        if 'navbar' in response.text and 'About the System' in response.text:
            print("✓ Home page loads with navbar")
            print("✓ 'About the System' link found in navbar")
        else:
            print("✗ Navbar missing")
    else:
        print(f"✗ Home page returned {response.status_code}")

    # Test about page
    response = requests.get('http://127.0.0.1:5000/about', timeout=5)
    if response.status_code == 200:
        print("\n✓ About page loaded (200 OK)")
        
        # Check for key sections
        sections = [
            ('System Overview', 'A. System Overview'),
            ('Model Metrics', 'B. Model Performance Metrics'),
            ('Per-Class Performance', 'C. Per-Class Performance'),
            ('Algorithm Comparison', 'D. Algorithm Comparison Table'),
            ('Model Selection', 'E. Model Selection Justification'),
            ('System Architecture', 'F. System Architecture')
        ]
        
        for check_name, pattern in sections:
            if pattern in response.text:
                print(f"  ✓ {check_name} section found")
            else:
                print(f"  ✗ {check_name} section missing")
        
        # Check for metrics
        metrics_check = [
            ('90.28%', 'Model accuracy'),
            ('Voting Ensemble', 'Ensemble model type'),
            ('Extra Trees', 'Algorithms table'),
            ('Logistic Regression', 'Algorithms'),
            ('90.28%', 'Accuracy metric'),
        ]
        
        print("\nMetrics verification:")
        for metric, desc in metrics_check:
            if metric in response.text:
                print(f"  ✓ {desc}: '{metric}' found")
            else:
                print(f"  ✗ {desc}: '{metric}' NOT found")
    else:
        print(f"✗ About page returned {response.status_code}")
        
except Exception as e:
    print(f"✗ Error: {e}")
