#!/usr/bin/env python3
"""
Simple script to check if API keys and authentication are loaded correctly
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_auth():
    print("=== Checking API Keys and Authentication ===\n")
    
    # Check Gemini API Key
    gemini_key = os.getenv("GEMINI_API_KEY")
    print(f"GEMINI_API_KEY: {'✓ Found' if gemini_key else '✗ Missing'}")
    if gemini_key:
        print(f"  Length: {len(gemini_key)} characters")
        print(f"  Preview: {gemini_key[:10]}...{gemini_key[-4:] if len(gemini_key) > 14 else ''}")
    
    print()
    
    # Check MyClassBoard API credentials
    mcb_api_key = os.getenv("MYCLASSBOARD_API_KEY")
    mcb_auth = os.getenv("MYCLASSBOARD_AUTH")
    
    print(f"MYCLASSBOARD_API_KEY: {'✓ Found' if mcb_api_key else '✗ Missing'}")
    if mcb_api_key:
        print(f"  Length: {len(mcb_api_key)} characters")
        print(f"  Preview: {mcb_api_key[:8]}...{mcb_api_key[-4:] if len(mcb_api_key) > 12 else ''}")
    
    print(f"MYCLASSBOARD_AUTH: {'✓ Found' if mcb_auth else '✗ Missing'}")
    if mcb_auth:
        print(f"  Length: {len(mcb_auth)} characters")
        print(f"  Preview: {mcb_auth[:8]}...{mcb_auth[-4:] if len(mcb_auth) > 12 else ''}")
    
    print()
    
    # Check API URL
    api_url = os.getenv("ENQUIRY_API_URL", "https://api.myclassboard.com/api/EnquiryService/Save_EnquiryDetails")
    print(f"ENQUIRY_API_URL: {api_url}")
    
    print()
    
    # Check other config values
    org_id = os.getenv("ORGANISATION_ID", "45")
    branch_id = os.getenv("BRANCH_ID", "79")
    academic_year_id = os.getenv("ACADEMIC_YEAR_ID", "17")
    class_id = os.getenv("CLASS_ID", "477")
    
    print("=== Configuration Values ===")
    print(f"ORGANISATION_ID: {org_id}")
    print(f"BRANCH_ID: {branch_id}")
    print(f"ACADEMIC_YEAR_ID: {academic_year_id}")
    print(f"CLASS_ID: {class_id}")
    
    print()
    
    # Summary
    all_keys_present = all([gemini_key, mcb_api_key, mcb_auth])
    print("=== Summary ===")
    print(f"All required keys present: {'✓ Yes' if all_keys_present else '✗ No'}")
    
    if not all_keys_present:
        print("\nMissing keys:")
        if not gemini_key:
            print("  - GEMINI_API_KEY")
        if not mcb_api_key:
            print("  - MYCLASSBOARD_API_KEY")
        if not mcb_auth:
            print("  - MYCLASSBOARD_AUTH")
    
    return all_keys_present

if __name__ == "__main__":
    check_auth()