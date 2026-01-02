# BPM Finder API

A Google Cloud Run microservice that computes BPM (beats per minute) from 30-second audio preview URLs. The service is deployed as a **private** Cloud Run service, requiring Google Cloud IAM authentication.

## Features

- 🎵 BPM computation from audio preview URLs (Apple, Spotify, Deezer)
- 🔒 Private Cloud Run service with IAM authentication
- 🛡️ SSRF protection with host whitelisting
- ⚡ Fast processing with Essentia and ffmpeg
- 📊 Returns BPM, raw BPM, confidence, and source host

## Architecture

- **Runtime**: Python 3 + FastAPI + Uvicorn
- **Audio Processing**: Essentia (RhythmExtractor2013) + ffmpeg
- **Container**: MTG Essentia base image (`ghcr.io/mtg/essentia`)
- **Deployment**: Google Cloud Run (Frankfurt region: `europe-west3`)
- **Authentication**: Cloud Run IAM (Identity Tokens)

## Prerequisites

- Google Cloud SDK (`gcloud`) installed and authenticated
- GCP project: `bpm-api-microservice`
- Billing enabled on the GCP project
- **Owner or Editor role** on the GCP project, OR the following IAM roles:
  - `roles/cloudbuild.builds.editor` (Cloud Build Editor)
  - `roles/artifactregistry.writer` (Artifact Registry Writer)
  - `roles/run.admin` (Cloud Run Admin)
  - `roles/iam.serviceAccountUser` (Service Account User)
  - `roles/iam.serviceAccountAdmin` (Service Account Admin) - for creating service accounts

## One-Time Setup

### 0. Grant Required IAM Permissions (if needed)

If you're not a project Owner/Editor, grant yourself the required roles:

```bash
# Get your email
YOUR_EMAIL=$(gcloud config get-value account)

# Grant Cloud Build Editor
gcloud projects add-iam-policy-binding bpm-api-microservice \
    --member="user:${YOUR_EMAIL}" \
    --role="roles/cloudbuild.builds.editor"

# Grant Artifact Registry Writer
gcloud projects add-iam-policy-binding bpm-api-microservice \
    --member="user:${YOUR_EMAIL}" \
    --role="roles/artifactregistry.writer"

# Grant Cloud Run Admin
gcloud projects add-iam-policy-binding bpm-api-microservice \
    --member="user:${YOUR_EMAIL}" \
    --role="roles/run.admin"

# Grant Service Account User
gcloud projects add-iam-policy-binding bpm-api-microservice \
    --member="user:${YOUR_EMAIL}" \
    --role="roles/iam.serviceAccountUser"

# Grant Service Account Admin (to create service accounts)
gcloud projects add-iam-policy-binding bpm-api-microservice \
    --member="user:${YOUR_EMAIL}" \
    --role="roles/iam.serviceAccountAdmin"
```

**Note**: If you don't have permission to grant yourself these roles, ask a project Owner/Editor to grant them.

### 1. Enable Required APIs

```bash
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    --project=bpm-api-microservice
```

### 2. Create Artifact Registry Repository

```bash
gcloud artifacts repositories create bpm-repo \
    --repository-format=docker \
    --location=europe-west3 \
    --description="BPM service Docker images" \
    --project=bpm-api-microservice
```

### 3. Create Service Account for Vercel

```bash
# Create service account
gcloud iam service-accounts create vercel-bpm-invoker \
    --display-name="Vercel BPM Invoker" \
    --project=bpm-api-microservice

# Grant Cloud Run Invoker role (will be bound to specific service after deployment)
# This is done automatically by deploy.sh, but you can also do it manually:
# gcloud run services add-iam-policy-binding bpm-service \
#     --region=europe-west3 \
#     --member="serviceAccount:vercel-bpm-invoker@bpm-api-microservice.iam.gserviceaccount.com" \
#     --role="roles/run.invoker" \
#     --project=bpm-api-microservice
```

### 4. Download Service Account Key (for Vercel)

```bash
# Create and download service account key
gcloud iam service-accounts keys create vercel-bpm-invoker-key.json \
    --iam-account=vercel-bpm-invoker@bpm-api-microservice.iam.gserviceaccount.com \
    --project=bpm-api-microservice
```

**⚠️ Security Note**: Store this JSON key securely in Vercel environment variables (see Vercel Integration section below).

## Deployment

### Deploy to Cloud Run

```bash
# Default region (europe-west3)
./deploy.sh

# Or override region
REGION=us-central1 ./deploy.sh
```

The `deploy.sh` script will:
1. Build the Docker image using Cloud Build
2. Push to Artifact Registry
3. Deploy to Cloud Run **without public access** (`--no-allow-unauthenticated`)
4. Create/verify the `vercel-bpm-invoker` service account
5. Grant `roles/run.invoker` permission to the service account

### Get Service URL

After deployment, get the service URL:

```bash
gcloud run services describe bpm-service \
    --region=europe-west3 \
    --format="value(status.url)" \
    --project=bpm-api-microservice
```

## Testing

### Test Health Endpoint

```bash
# Get identity token
TOKEN=$(gcloud auth print-identity-token)

# Replace with your actual service URL
SERVICE_URL="https://bpm-service-xxxxx-ew.a.run.app"

# Test health endpoint
curl -H "Authorization: Bearer $TOKEN" "${SERVICE_URL}/health"
```

Expected response:
```json
{"ok": true}
```

### Test BPM Endpoint

```bash
# Get identity token
TOKEN=$(gcloud auth print-identity-token)

# Replace with your actual service URL and a valid preview URL
SERVICE_URL="https://bpm-service-xxxxx-ew.a.run.app"
PREVIEW_URL="https://audio-ssl.itunes.apple.com/..."

curl -X POST \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"url\": \"${PREVIEW_URL}\"}" \
    "${SERVICE_URL}/bpm"
```

Expected response:
```json
{
  "bpm": 128,
  "bpm_raw": 64.0,
  "confidence": 0.73,
  "source_url_host": "audio-ssl.itunes.apple.com"
}
```

### Test with Service Account (Simulating Vercel)

```bash
# Authenticate as service account
gcloud auth activate-service-account \
    vercel-bpm-invoker@bpm-api-microservice.iam.gserviceaccount.com \
    --key-file=vercel-bpm-invoker-key.json

# Get identity token (audience is the service URL)
SERVICE_URL="https://bpm-service-xxxxx-ew.a.run.app"
TOKEN=$(gcloud auth print-identity-token --audience="${SERVICE_URL}")

# Test endpoint
curl -H "Authorization: Bearer $TOKEN" "${SERVICE_URL}/health"
```

## API Endpoints

### `GET /health`

Health check endpoint.

**Response:**
```json
{"ok": true}
```

### `POST /bpm`

Compute BPM from audio preview URL.

**Request Body:**
```json
{
  "url": "https://audio-ssl.itunes.apple.com/..."
}
```

**Response:**
```json
{
  "bpm": 128,
  "bpm_raw": 64.0,
  "confidence": 0.73,
  "source_url_host": "audio-ssl.itunes.apple.com"
}
```

**Field Descriptions:**
- `bpm`: Normalized BPM (70-200 range, doubled/halved as needed)
- `bpm_raw`: Raw BPM value from Essentia
- `confidence`: Confidence score (0.0-1.0)
- `source_url_host`: Hostname of the source URL

**Error Responses:**
- `400`: Invalid URL, non-allowed host, file too large, or redirect to non-allowed host
- `500`: Processing error (download, conversion, or BPM computation failed)

## SSRF Protection

The service implements strict SSRF protection:

- ✅ Only `https://` URLs allowed
- ✅ Host whitelist:
  - `.mzstatic.com` (Apple previews)
  - `.scdn.co` (Spotify previews)
  - `.deezer.com`, `.dzcdn.net` (Deezer previews)
- ✅ Redirect validation (rejects redirects to non-allowed hosts)
- ✅ Download limits:
  - Connect timeout: 5 seconds
  - Total timeout: 20 seconds
  - Max file size: 10MB

## Vercel Integration

To call this private Cloud Run service from Vercel, you need to:

1. **Store the service account key** in Vercel environment variables (as JSON string)
2. **Mint an Identity Token** with the Cloud Run service URL as the audience
3. **Send the token** in the `Authorization: Bearer` header

### Node.js Example (Vercel Serverless Function)

```javascript
const { GoogleAuth } = require('google-auth-library');

export default async function handler(req, res) {
  // Get service account credentials from Vercel env var
  const serviceAccountKey = JSON.parse(
    process.env.GCP_SERVICE_ACCOUNT_KEY
  );
  
  // Cloud Run service URL
  const cloudRunUrl = process.env.BPM_SERVICE_URL; // e.g., https://bpm-service-xxx-ew.a.run.app
  
  // Create auth client
  const auth = new GoogleAuth({
    credentials: serviceAccountKey,
    scopes: ['https://www.googleapis.com/auth/cloud-platform'],
  });
  
  // Get identity token with Cloud Run URL as audience
  const client = await auth.getIdTokenClient(cloudRunUrl);
  const idToken = await client.idTokenProvider.fetchIdToken(cloudRunUrl);
  
  // Call Cloud Run service
  const response = await fetch(`${cloudRunUrl}/bpm`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${idToken}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      url: req.body.url, // Preview URL from your frontend
    }),
  });
  
  if (!response.ok) {
    const error = await response.text();
    return res.status(response.status).json({ error });
  }
  
  const data = await response.json();
  return res.status(200).json(data);
}
```

### Vercel Environment Variables

Set these in your Vercel project settings:

- `GCP_SERVICE_ACCOUNT_KEY`: Full JSON content of `vercel-bpm-invoker-key.json`
- `BPM_SERVICE_URL`: Your Cloud Run service URL (e.g., `https://bpm-service-xxxxx-ew.a.run.app`)

### Python Example (Alternative)

```python
from google.auth.transport.requests import Request
from google.oauth2 import service_account
import json
import os
import requests

# Load service account key from env var
service_account_info = json.loads(os.environ['GCP_SERVICE_ACCOUNT_KEY'])
cloud_run_url = os.environ['BPM_SERVICE_URL']

# Create credentials
credentials = service_account.Credentials.from_service_account_info(
    service_account_info,
    scopes=['https://www.googleapis.com/auth/cloud-platform']
)

# Refresh to get access token (for ID token, use google-auth library)
credentials.refresh(Request())

# For ID token, use:
from google.auth.transport.requests import Request
from google.oauth2 import id_token

# Get ID token with audience
id_token_obj = id_token.fetch_id_token(Request(), cloud_run_url)

# Call Cloud Run
response = requests.post(
    f"{cloud_run_url}/bpm",
    headers={
        "Authorization": f"Bearer {id_token_obj}",
        "Content-Type": "application/json",
    },
    json={"url": preview_url}
)
```

## Local Development

### Prerequisites

- Python 3.9+
- Docker (for testing container)
- Essentia and ffmpeg (if running locally without Docker)

### Run Locally (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Note: You'll need Essentia Python bindings installed
# This is complex, so Docker is recommended for local testing

# Run server
uvicorn main:app --host 0.0.0.0 --port 8080
```

### Run with Docker

```bash
# Build image
docker build -t bpm-service .

# Run container
docker run -p 8080:8080 bpm-service

# Test
curl http://localhost:8080/health
```

## Processing Pipeline

1. **Download**: Fetch audio from preview URL (with SSRF protection)
2. **Convert**: Use ffmpeg to convert to mono 44100Hz 16-bit PCM WAV
3. **Analyze**: Use Essentia `RhythmExtractor2013` to compute BPM and confidence
4. **Normalize**: Adjust BPM to 70-200 range:
   - If BPM < 70: multiply by 2 (repeat until >= 70)
   - If BPM > 200: divide by 2 (repeat until <= 200)
5. **Cleanup**: Delete temporary files
6. **Return**: JSON response with BPM data

## Security Notes

- ✅ Cloud Run IAM authentication (no public access)
- ✅ SSRF protection with host whitelisting
- ✅ Download size and timeout limits
- ✅ Redirect validation
- ✅ No audio persistence (temp files deleted immediately)
- ✅ Minimal logging (URLs with tokens are not logged)

## Troubleshooting

### "Permission denied" errors

Ensure the service account has `roles/run.invoker` permission:

```bash
gcloud run services add-iam-policy-binding bpm-service \
    --region=europe-west3 \
    --member="serviceAccount:vercel-bpm-invoker@bpm-api-microservice.iam.gserviceaccount.com" \
    --role="roles/run.invoker" \
    --project=bpm-api-microservice
```

### "Host not allowed" errors

Check that the preview URL host ends with one of the allowed suffixes:
- `.mzstatic.com`
- `.scdn.co`
- `.deezer.com`
- `.dzcdn.net`

### ffmpeg conversion errors

Ensure the audio file is a valid format. The service supports common audio formats (MP3, M4A, etc.) that ffmpeg can handle.

### High memory usage

The service is configured with 2GB memory. For very large files or high concurrency, consider increasing:

```bash
gcloud run services update bpm-service \
    --region=europe-west3 \
    --memory 4Gi \
    --project=bpm-api-microservice
```

## License

GNU General Public License v3.0

