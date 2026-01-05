#!/usr/bin/env python3
"""
Test script to verify fallback service authentication.
This uses the same authentication method as the main service.
"""
import sys
import google.auth
import google.auth.transport.requests
import google.oauth2.id_token
import requests

FALLBACK_SERVICE_URL = "https://bpm-fallback-service-340051416180.europe-west3.run.app"

def test_authentication():
    print("Testing Fallback Service Authentication")
    print("=" * 50)
    print(f"Fallback Service URL: {FALLBACK_SERVICE_URL}")
    print()
    
    # Test 1: Generate ID token (same method as main service)
    print("1. Generating ID token using google.auth...")
    try:
        credentials, project = google.auth.default()
        print(f"   Credentials found: {type(credentials).__name__}")
        print(f"   Project: {project}")
        
        if not credentials:
            print("   ❌ Error: No credentials found")
            return False
        
        # Refresh credentials if needed
        if not credentials.valid:
            print("   Refreshing credentials...")
            auth_request = google.auth.transport.requests.Request()
            credentials.refresh(auth_request)
        
        # Create a request object for fetching the ID token
        auth_request = google.auth.transport.requests.Request()
        
        # Fetch the ID token with the audience (fallback service URL)
        print(f"   Fetching ID token for audience: {FALLBACK_SERVICE_URL}")
        token = google.oauth2.id_token.fetch_id_token(auth_request, FALLBACK_SERVICE_URL)
        
        if not token:
            print("   ❌ Error: Empty token generated")
            return False
        
        print(f"   ✅ Token generated (length: {len(token)} characters)")
        print()
        
    except Exception as e:
        print(f"   ❌ Error generating token: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Call health endpoint
    print("2. Testing authentication with fallback service...")
    print(f"   Calling: {FALLBACK_SERVICE_URL}/health")
    print()
    
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{FALLBACK_SERVICE_URL}/health", headers=headers, timeout=10)
        
        print(f"   HTTP Status Code: {response.status_code}")
        print(f"   Response: {response.text}")
        print()
        
        if response.status_code == 200:
            print("   ✅ Authentication successful! The token works.")
            return True
        elif response.status_code == 403:
            print("   ❌ Authentication failed (403 Forbidden)")
            print()
            print("   Troubleshooting:")
            print("   1. Verify the service account has permission:")
            print(f"      gcloud run services get-iam-policy bpm-fallback-service --region europe-west3")
            print()
            print("   2. Check if the primary service's service account is in the policy")
            print("   3. Grant permission if missing")
            return False
        else:
            print(f"   ⚠️  Unexpected status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error calling service: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_authentication()
    sys.exit(0 if success else 1)

