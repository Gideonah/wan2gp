# GCS Upload Configuration Guide

This guide explains how to configure Google Cloud Storage (GCS) uploads for Wan2GP.

## Quick Test

Test your GCS credentials before deploying:

```bash
python test_gcs.py
```

## Authentication Methods

The system tries authentication methods in this order:

1. **GCS_SERVICE_ACCOUNT_JSON** (easiest for testing)
2. **GCP env vars** (GCP_CLIENT_EMAIL + GCP_PRIVATE_KEY_B64) (secure for production)
3. **GOOGLE_APPLICATION_CREDENTIALS** (JSON file path)
4. **Application Default Credentials** (gcloud auth)

---

## Method 1: JSON Key String (Easiest for Testing)

**Best for:** Local testing, quick setup

### Setup

1. Download your service account JSON key from GCP Console
2. Set it as an environment variable:

```bash
# Copy the entire JSON content into the variable
export GCS_SERVICE_ACCOUNT_JSON='{"type":"service_account","project_id":"your-project",...}'

# Or load from file
export GCS_SERVICE_ACCOUNT_JSON=$(cat path/to/service-account.json)
```

### Full Example

```bash
export GCS_SERVICE_ACCOUNT_JSON='{"type":"service_account","project_id":"your-project","private_key_id":"abc123","private_key":"-----BEGIN PRIVATE KEY-----\n...","client_email":"your-sa@project.iam.gserviceaccount.com",...}'
export GCS_BUCKET_NAME="your-bucket-name"
export GCS_ENABLED="true"

# Test it
python test_gcs.py

# Run server
python api_server.py
```

---

## Method 2: Individual Env Vars (Most Secure)

**Best for:** Production deployments (Vast.ai, RunPod, Modal)

### Why This Method?

- No JSON file written to disk
- Private key is base64-encoded
- Safer for serverless environments

### Setup

1. Get your service account JSON key from GCP Console

2. Extract and encode the private key:

```bash
# Extract the private key from JSON (includes -----BEGIN PRIVATE KEY----- markers)
cat service-account.json | jq -r '.private_key' | base64

# Or with Python
python3 << 'EOF'
import json, base64
with open('service-account.json') as f:
    key = json.load(f)
    encoded = base64.b64encode(key['private_key'].encode()).decode()
    print(encoded)
EOF
```

3. Set environment variables:

```bash
export GCP_PROJECT_ID="your-project-id"
export GCP_CLIENT_EMAIL="your-service-account@your-project.iam.gserviceaccount.com"
export GCP_PRIVATE_KEY_B64="LS0tLS1CRUdJTiBQUklWQVRFIEtFWS0tLS0t..."
export GCS_BUCKET_NAME="your-bucket-name"
export GCS_ENABLED="true"

# Optional (can leave empty)
export GCP_PRIVATE_KEY_ID="abc123..."
export GCP_CLIENT_ID="123456789..."
```

### Vast.ai Example

Add to your instance environment:

```bash
GCP_PROJECT_ID=your-project-id
GCP_CLIENT_EMAIL=sa@project.iam.gserviceaccount.com
GCP_PRIVATE_KEY_B64=LS0tLS1CRUdJTiBQUklWQVRFIEtFWS0tLS0t...
GCS_BUCKET_NAME=your-bucket
GCS_ENABLED=true
```

---

## Method 3: JSON File Path

**Best for:** Local development, traditional server deployments

### Setup

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
export GCS_BUCKET_NAME="your-bucket-name"
export GCS_ENABLED="true"

python api_server.py
```

---

## Method 4: Application Default Credentials

**Best for:** Running on GCP (Cloud Run, GCE, etc.)

If your code is running on GCP, it can use the instance's service account automatically:

```bash
export GCS_BUCKET_NAME="your-bucket-name"
export GCS_ENABLED="true"

# No credentials needed - uses instance service account
python api_server.py
```

---

## Creating a GCP Service Account

1. Go to [GCP Console → IAM & Admin → Service Accounts](https://console.cloud.google.com/iam-admin/serviceaccounts)

2. Click **"Create Service Account"**

3. Fill in details:
   - Name: `wan2gp-storage`
   - Description: "Service account for Wan2GP GCS uploads"

4. Grant permissions:
   - **Storage Object Admin** (full control of objects)
   - Or **Storage Object Creator** (write-only, more secure)

5. Click **"Create Key"** → **JSON** → Download

6. Save the JSON file securely (you'll need it for authentication)

---

## Creating a GCS Bucket

```bash
# Using gcloud CLI
gsutil mb -p YOUR_PROJECT_ID -c STANDARD -l us-central1 gs://your-bucket-name

# Or via web console
# https://console.cloud.google.com/storage/browser
```

### Bucket Settings

- **Name:** Choose a globally unique name (e.g., `wan2gp-outputs-prod`)
- **Location:** Choose closest to your servers (e.g., `us-central1`)
- **Storage class:** Standard
- **Access control:** Uniform (recommended)

### Make Objects Publicly Readable (Optional)

If you want anyone with the signed URL to access files:

```bash
gsutil iam ch allUsers:objectViewer gs://your-bucket-name
```

---

## Environment Variables Reference

### Required

```bash
GCS_ENABLED=true                    # Enable GCS uploads
GCS_BUCKET_NAME=your-bucket-name    # Your GCS bucket
```

### Authentication (pick one method)

**Method 1: JSON String**
```bash
GCS_SERVICE_ACCOUNT_JSON='{"type":"service_account",...}'
```

**Method 2: Individual Env Vars**
```bash
GCP_PROJECT_ID=your-project-id
GCP_CLIENT_EMAIL=sa@project.iam.gserviceaccount.com
GCP_PRIVATE_KEY_B64=LS0tLS1...
```

**Method 3: JSON File**
```bash
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

### Optional

```bash
GCS_URL_EXPIRATION_DAYS=7          # Signed URL expiration (default: 7)
GCP_PRIVATE_KEY_ID=abc123...       # Key ID (optional)
GCP_CLIENT_ID=123456789...         # Client ID (optional)
```

---

## Testing Your Setup

### 1. Test credentials

```bash
python test_gcs.py
```

This will:
- ✅ Check which auth method is being used
- ✅ Verify credentials are valid
- ✅ Test bucket access
- ✅ Test signed URL generation

### 2. Test with API server

```bash
# Start the server
python api_server.py

# In another terminal, generate a video
curl -X POST http://localhost:8000/generate/ltx2/i2v \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A person smiling",
    "image_url": "https://example.com/image.jpg",
    "duration": 5.0
  }'

# Check the response - output_url should be a GCS signed URL
# https://storage.googleapis.com/your-bucket/videos/...
```

---

## Troubleshooting

### "GCS upload disabled"

Check: `GCS_ENABLED=true` is set

### "GCS client not available"

Check: Credentials are set (run `python test_gcs.py`)

### "Bucket does not exist"

Create the bucket or check `GCS_BUCKET_NAME` matches your bucket name

### "Failed to parse GCS_SERVICE_ACCOUNT_JSON"

Make sure JSON is properly quoted:
```bash
export GCS_SERVICE_ACCOUNT_JSON='{"type":"service_account",...}'  # Single quotes!
```

### "Permission denied"

Check your service account has **Storage Object Admin** or **Storage Object Creator** role

---

## Security Best Practices

1. **Never commit credentials to git**
   - Add `service-account.json` to `.gitignore`
   - Use env vars for production

2. **Use minimal permissions**
   - `Storage Object Creator` is sufficient (write-only)
   - `Storage Object Admin` if you need delete access

3. **Rotate keys regularly**
   - Create new service account keys every 90 days
   - Delete old keys from GCP Console

4. **Use separate service accounts**
   - Dev vs Prod environments
   - Different projects

5. **Prefer Method 2 for production**
   - Base64-encoded private key
   - No file on disk
   - Harder to accidentally expose

---

## Cost Estimation

GCS costs for video storage:

- **Storage:** ~$0.02/GB/month (Standard class, us-central1)
- **Operations:** $0.05 per 10,000 writes
- **Network:** ~$0.12/GB egress (first 1TB/month)

Example monthly cost for 1000 videos:
- 1000 videos × 50MB = 50GB storage = **$1/month**
- 1000 uploads = **$0.005**
- 5TB downloads = **$600** (use CDN for high traffic!)

**Tip:** Set a lifecycle policy to auto-delete files after 30 days:
```bash
gsutil lifecycle set lifecycle.json gs://your-bucket-name
```

lifecycle.json:
```json
{
  "lifecycle": {
    "rule": [{
      "action": {"type": "Delete"},
      "condition": {"age": 30}
    }]
  }
}
```

---

## Support

- GCP Storage Docs: https://cloud.google.com/storage/docs
- Service Account Setup: https://cloud.google.com/iam/docs/service-accounts
- Python Client Library: https://googleapis.dev/python/storage/latest/
