#!/usr/bin/env python3
"""
Test script to verify MyClassBoard API call works
"""
import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_api_call():
    # Get credentials
    api_url = os.getenv("ENQUIRY_API_URL", "https://api.myclassboard.com/api/EnquiryService/Save_EnquiryDetails")
    api_key = os.getenv("MYCLASSBOARD_API_KEY", "")
    auth = os.getenv("MYCLASSBOARD_AUTH", "")
    
    # Test payload
    test_payload = {
        "OrganisationID": "45",
        "BranchID": "79", 
        "AcademicYearID": "17",
        "ClassID": 477,
        "StudentName": "Test Student",
        "Gender": 1,
        "FatherName": "Test Father",
        "FatherEmailID": "test@example.com",
        "StudentEmailID": "student@example.com",
        "mobileNo": "9876543210",
        "dob": "2010-01-01",
        "enquirySource": "Website Chatbot Test",
        "remarks": "Test submission",
        "genderText": "male",
        "schoolName": "Test School",
        "admissionClass": "5th"
    }
    
    # Headers
    headers = {
        "Content-Type": "application/json",
        "api_Key": api_key,
        "Authorization": auth
    }
    
    print("=== Testing MyClassBoard API Call ===")
    print(f"URL: {api_url}")
    print(f"API Key present: {bool(api_key)}")
    print(f"Auth present: {bool(auth)}")
    print(f"Payload: {json.dumps(test_payload, indent=2)}")
    print()
    
    try:
        print("Making API call...")
        response = requests.post(
            api_url,
            headers=headers,
            json=test_payload,
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        try:
            response_json = response.json()
            print(f"Response JSON: {json.dumps(response_json, indent=2)}")
        except:
            print(f"Response Text: {response.text}")
            
        if response.status_code in (200, 201):
            print("✅ SUCCESS: Data submitted successfully!")
            return True
        else:
            print(f"❌ FAILED: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    test_api_call()