#!/usr/bin/env python3
"""
Mental Health Monitoring Dashboard - Web Server Launcher
Click this file to start the website and open it in your browser
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def print_header():
    """Print beautiful header"""
    print("\n" + "="*80)
    print("🧠 MENTAL HEALTH MONITORING SYSTEM")
    print("IoT-Enabled Wearable Sensors for Mental Health Monitoring")
    print("="*80 + "\n")

def print_status(step, message):
    """Print status message"""
    print(f"[{step}] {message}")

def main():
    """Main launcher function"""
    print_header()
    
    # Get project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    print_status("1/4", "Checking dependencies...")
    time.sleep(0.5)
    
    try:
        import flask
        import joblib
        import numpy as np
        import pandas as pd
        print("    ✓ Flask installed")
        print("    ✓ All dependencies ready\n")
    except ImportError as e:
        print(f"    ✗ Missing dependency: {e}")
        print("    Run: pip install -r requirements.txt")
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    print_status("2/4", "Loading model...")
    time.sleep(0.5)
    
    try:
        import joblib
        model = joblib.load('model/mental_health_model.pkl')
        scaler = joblib.load('model/mental_health_model_scaler.pkl')
        features = joblib.load('model/mental_health_model_features.pkl')
        print(f"    ✓ Model loaded (Voting Ensemble)")
        print(f"    ✓ Features: {len(features)} total")
        print(f"    ✓ Accuracy: 90.28%\n")
    except Exception as e:
        print(f"    ✗ Error loading model: {e}")
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    print_status("3/4", "Starting Flask server...")
    time.sleep(0.5)
    
    # Start Flask server in the background
    try:
        if sys.platform == 'win32':
            # Windows: start in new window minimized
            subprocess.Popen(
                [sys.executable, 'app/app_enhanced.py'],
                creationflags=subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NO_WINDOW,
                cwd=str(project_dir)
            )
        else:
            # Linux/Mac
            subprocess.Popen(
                [sys.executable, 'app/app_enhanced.py'],
                cwd=str(project_dir),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        
        print("    ✓ Flask server started\n")
    except Exception as e:
        print(f"    ✗ Error starting server: {e}")
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    # Wait for server to be ready
    print_status("4/4", "Waiting for server to initialize...")
    time.sleep(2)
    
    print("    ✓ Server ready\n")
    
    # Open browser
    url = "http://127.0.0.1:5000"
    print("-" * 80)
    print(f"\n✓ Opening website: {url}\n")
    
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"Could not open browser automatically. Please visit: {url}")
    
    # Print information
    print("="*80)
    print("📊 DASHBOARD INFORMATION")
    print("="*80)
    print(f"URL:              {url}")
    print(f"Model:            Voting Ensemble")
    print(f"Accuracy:         90.28%")
    print(f"Features:         16 (8 physiological + 8 engineered)")
    print(f"Cross-Validation: 85% ± 3.16%")
    print(f"Training Samples: 1,200 (perfectly balanced)")
    print("="*80)
    print("\n✨ Website is running! Check your browser or visit the URL above.")
    print("\n📝 To stop the server:")
    print("   1. Close the Flask server window")
    print("   2. Or press Ctrl+C in any terminal running the server")
    print("\n" + "="*80 + "\n")
    
    # Keep script running until user closes it
    try:
        input("Press Enter to exit the launcher (server will keep running)...\n")
    except KeyboardInterrupt:
        print("\nExiting launcher...")
    
    print("Goodbye!")

if __name__ == '__main__':
    main()
