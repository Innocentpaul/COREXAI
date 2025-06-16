"""
Test the health endpoint of the running Flask app
"""

import requests
import json

def test_health():
    """Test the health endpoint"""
    try:
        response = requests.get("http://localhost:5000/health")
        if response.status_code == 200:
            data = response.json()
            print("Health Check Response:")
            print(json.dumps(data, indent=2))
        else:
            print(f"Health check failed with status: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the Flask app.")
        print("Make sure the app is running with: python app3.py")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_health()