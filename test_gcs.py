#!/usr/bin/env python3
"""
Test GCS Connection

This script helps you test your GCS credentials before deploying.

Usage:
    # Method 1: Use JSON key file
    export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
    python test_gcs.py
    
    # Method 2: Use JSON key string
    export GCS_SERVICE_ACCOUNT_JSON='{"type":"service_account",...}'
    python test_gcs.py
    
    # Method 3: Use individual env vars
    export GCP_PROJECT_ID="your-project"
    export GCP_CLIENT_EMAIL="sa@project.iam.gserviceaccount.com"
    export GCP_PRIVATE_KEY_B64="LS0tLS1..."
    python test_gcs.py
"""

import os
import sys
import json
import base64
from pathlib import Path

def test_gcs_connection():
    """Test GCS connection with current credentials."""
    
    print("=" * 70)
    print("GCS Connection Test")
    print("=" * 70)
    
    # Check which method is being used
    print("\n🔍 Checking authentication methods...\n")
    
    has_json_string = bool(os.environ.get("GCS_SERVICE_ACCOUNT_JSON"))
    has_env_vars = bool(os.environ.get("GCP_CLIENT_EMAIL") and os.environ.get("GCP_PRIVATE_KEY_B64"))
    has_json_file = bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
    
    if has_json_string:
        print("✅ Method 1: GCS_SERVICE_ACCOUNT_JSON found")
        try:
            creds = json.loads(os.environ.get("GCS_SERVICE_ACCOUNT_JSON"))
            print(f"   Project ID: {creds.get('project_id')}")
            print(f"   Client Email: {creds.get('client_email')}")
        except json.JSONDecodeError as e:
            print(f"   ❌ Invalid JSON: {e}")
            return False
    else:
        print("⚠️  Method 1: GCS_SERVICE_ACCOUNT_JSON not set")
    
    if has_env_vars:
        print("✅ Method 2: Individual env vars found")
        print(f"   GCP_PROJECT_ID: {os.environ.get('GCP_PROJECT_ID')}")
        print(f"   GCP_CLIENT_EMAIL: {os.environ.get('GCP_CLIENT_EMAIL')}")
        print(f"   GCP_PRIVATE_KEY_B64: {'set' if os.environ.get('GCP_PRIVATE_KEY_B64') else 'missing'}")
    else:
        print("⚠️  Method 2: Individual env vars not set")
    
    if has_json_file:
        path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if os.path.exists(path):
            print(f"✅ Method 3: GOOGLE_APPLICATION_CREDENTIALS found")
            print(f"   Path: {path}")
            try:
                with open(path) as f:
                    creds = json.load(f)
                print(f"   Project ID: {creds.get('project_id')}")
                print(f"   Client Email: {creds.get('client_email')}")
            except Exception as e:
                print(f"   ❌ Error reading file: {e}")
        else:
            print(f"⚠️  Method 3: GOOGLE_APPLICATION_CREDENTIALS set but file not found: {path}")
    else:
        print("⚠️  Method 3: GOOGLE_APPLICATION_CREDENTIALS not set")
    
    if not (has_json_string or has_env_vars or has_json_file):
        print("\n❌ No credentials found! Set one of:")
        print("   - GCS_SERVICE_ACCOUNT_JSON")
        print("   - GCP_CLIENT_EMAIL + GCP_PRIVATE_KEY_B64")
        print("   - GOOGLE_APPLICATION_CREDENTIALS")
        return False
    
    # Try to initialize GCS client
    print("\n🔄 Initializing GCS client...\n")
    
    try:
        from google.cloud import storage
        from google.oauth2 import service_account
        
        bucket_name = os.environ.get("GCS_BUCKET_NAME", "serverless_media_outputs")
        project_id = os.environ.get("GCP_PROJECT_ID")
        
        # Try different methods in order
        client = None
        
        if has_json_string:
            print("   Trying GCS_SERVICE_ACCOUNT_JSON...")
            creds_info = json.loads(os.environ.get("GCS_SERVICE_ACCOUNT_JSON"))
            credentials = service_account.Credentials.from_service_account_info(creds_info)
            client = storage.Client(project=project_id, credentials=credentials)
            print("   ✅ Connected via GCS_SERVICE_ACCOUNT_JSON")
        
        elif has_env_vars:
            print("   Trying individual env vars...")
            private_key_b64 = os.environ.get("GCP_PRIVATE_KEY_B64")
            private_key = base64.b64decode(private_key_b64).decode('utf-8')
            private_key = private_key.replace('\\n', '\n')
            
            creds_info = {
                "type": "service_account",
                "project_id": os.environ.get("GCP_PROJECT_ID", ""),
                "private_key_id": os.environ.get("GCP_PRIVATE_KEY_ID", ""),
                "private_key": private_key,
                "client_email": os.environ.get("GCP_CLIENT_EMAIL"),
                "client_id": os.environ.get("GCP_CLIENT_ID", ""),
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
            credentials = service_account.Credentials.from_service_account_info(creds_info)
            client = storage.Client(project=project_id, credentials=credentials)
            print("   ✅ Connected via env vars")
        
        elif has_json_file:
            print("   Trying GOOGLE_APPLICATION_CREDENTIALS...")
            client = storage.Client(project=project_id)
            print("   ✅ Connected via GOOGLE_APPLICATION_CREDENTIALS")
        
        # Test bucket access
        print(f"\n🪣 Testing bucket access: {bucket_name}\n")
        bucket = client.bucket(bucket_name)
        
        # Check if bucket exists
        try:
            exists = bucket.exists()
        except Exception as e:
            print(f"   ❌ Cannot check if bucket exists: {e}")
            print(f"\n📋 Possible issues:")
            print(f"   1. Bucket '{bucket_name}' doesn't exist")
            print(f"   2. Service account lacks permissions")
            print(f"   3. Bucket is in a different project")
            print(f"\n🔧 To fix:")
            print(f"\n   Option A: Grant permissions to existing bucket")
            print(f"   1. Go to: https://console.cloud.google.com/storage/browser/{bucket_name}")
            print(f"   2. Click 'Permissions' tab")
            print(f"   3. Click '+ Grant Access'")
            print(f"   4. Add principal: {os.environ.get('GCP_CLIENT_EMAIL', 'your-service-account')}")
            print(f"   5. Role: 'Storage Object Admin' or 'Storage Admin'")
            print(f"\n   Option B: Create new bucket with correct permissions")
            print(f"   Run: gsutil mb -p {project_id} gs://{bucket_name}")
            print(f"   Then: gsutil iam ch serviceAccount:{os.environ.get('GCP_CLIENT_EMAIL', 'SA')}:objectAdmin gs://{bucket_name}")
            print(f"\n   Option C: Use a different bucket you already have access to")
            print(f"   Set: export GCS_BUCKET_NAME=your-existing-bucket")
            return False
        
        if exists:
            print(f"   ✅ Bucket '{bucket_name}' exists and is accessible")
            
            # Try to list some files
            try:
                blobs = list(bucket.list_blobs(max_results=5))
                print(f"   📁 Found {len(blobs)} files (showing max 5):")
                for blob in blobs:
                    print(f"      - {blob.name}")
                
                # Test signed URL generation
                print("\n🔗 Testing signed URL generation...")
                if blobs:
                    test_blob = blobs[0]
                    from datetime import timedelta
                    url = test_blob.generate_signed_url(
                        version="v4",
                        expiration=timedelta(minutes=15),
                        method="GET",
                    )
                    print(f"   ✅ Signed URL generated successfully")
                    print(f"   URL (15 min expiry): {url[:80]}...")
                else:
                    print("   ⚠️  No files in bucket to test signed URL generation")
                    print("   This is OK - bucket is empty but accessible")
            except Exception as e:
                print(f"   ⚠️  Can access bucket but cannot list objects: {e}")
                print(f"   You may need 'Storage Object Viewer' role")
        else:
            print(f"   ❌ Bucket '{bucket_name}' does not exist")
            print(f"   Create it at: https://console.cloud.google.com/storage/browser?project={project_id}")
            return False
        
        print("\n" + "=" * 70)
        print("✅ GCS CONNECTION TEST PASSED!")
        print("=" * 70)
        print("\nYour credentials are valid and ready to use.")
        print(f"Bucket: {bucket_name}")
        print(f"Project: {project_id}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("   Install with: pip install google-cloud-storage")
        return False
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gcs_connection()
    sys.exit(0 if success else 1)
