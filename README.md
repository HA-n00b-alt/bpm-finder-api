# BPM Finder API

A Google Cloud Run microservice that computes BPM (beats per minute) from 30-second audio preview URLs. The service is deployed as a **private** Cloud Run service, requiring Google Cloud IAM authentication.

## Features

- BPM computation from audio preview URLs (Apple, Spotify, Deezer)
- Private Cloud Run service with IAM authentication
- SSRF protection with host whitelisting
- Fast processing with Essentia and ffmpeg
- Returns BPM, raw BPM, confidence, and source host

## Architecture

- **Runtime**: Python 3 + FastAPI + Uvicorn
- **Audio Processing**: Essentia (RhythmExtractor2013) + ffmpeg
- **Container**: MTG Essentia base image (`ghcr.io/mtg/essentia`)
- **Deployment**: Google Cloud Run
- **Authentication**: Cloud Run IAM (Identity Tokens)

## Prerequisites

- Google Cloud SDK (`gcloud`) installed and authenticated
- A GCP project with billing enabled
- **Owner or Editor role** on the GCP project, OR the following IAM roles:
  - `roles/cloudbuild.builds.editor` (Cloud Build Editor)
  - `roles/artifactregistry.writer` (Artifact Registry Writer)
  - `roles/run.admin` (Cloud Run Admin)
  - `roles/iam.serviceAccountUser` (Service Account User)
  - `roles/iam.serviceAccountAdmin` (Service Account Admin) - for creating service accounts

## Configuration

Before deployment, configure the following variables in `deploy.sh` or set them as environment variables:

- `PROJECT_ID`: Your GCP project ID (default: `bpm-api-microservice`)
- `REGION`: Cloud Run region (default: `europe-west3`)
- `SERVICE_NAME`: Cloud Run service name (default: `bpm-service`)
- `ARTIFACT_REPO`: Artifact Registry repository name (default: `bpm-repo`)
- `SERVICE_ACCOUNT`: Service account name for external callers (default: `vercel-bpm-invoker`)

## One-Time Setup

### 0. Grant Required IAM Permissions (if needed)

If you're not a project Owner/Editor, grant yourself the required roles:

```bash
# Set your project ID
PROJECT_ID="your-project-id"

# Get your email
YOUR_EMAIL=$(gcloud config get-value account)

# Grant Cloud Build Editor
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="user:${YOUR_EMAIL}" \
    --role="roles/cloudbuild.builds.editor"

# Grant Artifact Registry Writer
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="user:${YOUR_EMAIL}" \
    --role="roles/artifactregistry.writer"

# Grant Cloud Run Admin
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="user:${YOUR_EMAIL}" \
    --role="roles/run.admin"

# Grant Service Account User
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="user:${YOUR_EMAIL}" \
    --role="roles/iam.serviceAccountUser"

# Grant Service Account Admin (to create service accounts)
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="user:${YOUR_EMAIL}" \
    --role="roles/iam.serviceAccountAdmin"
```

**Note**: If you don't have permission to grant yourself these roles, ask a project Owner/Editor to grant them.

### 1. Enable Required APIs

```bash
PROJECT_ID="your-project-id"
REGION="your-region"  # e.g., europe-west3, us-central1

gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    --project=${PROJECT_ID}
```

### 2. Create Artifact Registry Repository

```bash
PROJECT_ID="your-project-id"
REGION="your-region"
ARTIFACT_REPO="bpm-repo"

gcloud artifacts repositories create ${ARTIFACT_REPO} \
    --repository-format=docker \
    --location=${REGION} \
    --description="BPM service Docker images" \
    --project=${PROJECT_ID}
```

### 3. Create Service Account for External Callers

```bash
PROJECT_ID="your-project-id"
SERVICE_ACCOUNT="bpm-invoker"  # Change to your preferred name

# Create service account
gcloud iam service-accounts create ${SERVICE_ACCOUNT} \
    --display-name="BPM Service Invoker" \
    --project=${PROJECT_ID}
```

The `deploy.sh` script will automatically grant the `roles/run.invoker` permission to this service account after deployment.

### 4. Download Service Account Key

```bash
PROJECT_ID="your-project-id"
SERVICE_ACCOUNT="bpm-invoker"

# Create and download service account key
gcloud iam service-accounts keys create ${SERVICE_ACCOUNT}-key.json \
    --iam-account=${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com \
    --project=${PROJECT_ID}
```

**Security Note**: Store this JSON key securely. You'll need it to authenticate external applications calling the service.

## Deployment

### Deploy to Cloud Run

```bash
# Set your project ID
export PROJECT_ID="your-project-id"

# Deploy with default region (europe-west3)
./deploy.sh

# Or override region
REGION=us-central1 ./deploy.sh
```

The `deploy.sh` script will:
1. Build the Docker image using Cloud Build
2. Push to Artifact Registry
3. Deploy to Cloud Run **without public access** (`--no-allow-unauthenticated`)
4. Create/verify the service account
5. Grant `roles/run.invoker` permission to the service account

### Get Service URL

After deployment, get the service URL:

```bash
PROJECT_ID="your-project-id"
REGION="your-region"
SERVICE_NAME="bpm-service"

gcloud run services describe ${SERVICE_NAME} \
    --region=${REGION} \
    --format="value(status.url)" \
    --project=${PROJECT_ID}
```

## Testing

### Test Health Endpoint

```bash
# Get identity token
TOKEN=$(gcloud auth print-identity-token)

# Replace with your actual service URL
SERVICE_URL="https://your-service-url.run.app"

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
SERVICE_URL="https://your-service-url.run.app"
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

### Test with Service Account

To test authentication from an external application:

```bash
# Authenticate as service account
gcloud auth activate-service-account \
    your-service-account@your-project.iam.gserviceaccount.com \
    --key-file=your-service-account-key.json

# Get identity token (audience is the service URL)
SERVICE_URL="https://your-service-url.run.app"
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

- Only `https://` URLs allowed
- Host whitelist:
  - `.mzstatic.com` (Apple previews)
  - `.scdn.co` (Spotify previews)
  - `.deezer.com`, `.dzcdn.net` (Deezer previews)
- Redirect validation (rejects redirects to non-allowed hosts)
- Download limits:
  - Connect timeout: 5 seconds
  - Total timeout: 20 seconds
  - Max file size: 10MB

## Using the API from External Applications

Since the Cloud Run service is private (requires authentication), external applications must authenticate using Google Cloud Identity Tokens. The general process is:

1. **Obtain a service account key** from your GCP project
2. **Mint an Identity Token** with the Cloud Run service URL as the audience
3. **Send the token** in the `Authorization: Bearer` header

### Example: Vercel Serverless Function (Node.js)

This example shows how to call the BPM service from a Vercel serverless function:

```javascript
const { GoogleAuth } = require('google-auth-library');

export default async function handler(req, res) {
  // Get service account credentials from Vercel env var
  const serviceAccountKey = JSON.parse(
    process.env.GCP_SERVICE_ACCOUNT_KEY
  );
  
  // Cloud Run service URL
  const cloudRunUrl = process.env.BPM_SERVICE_URL;
  
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

**Vercel Environment Variables:**
- `GCP_SERVICE_ACCOUNT_KEY`: Full JSON content of your service account key file
- `BPM_SERVICE_URL`: Your Cloud Run service URL

### Example: Python Application

```python
from google.auth.transport.requests import Request
from google.oauth2 import id_token
import json
import os
import requests

# Load service account key from env var
service_account_info = json.loads(os.environ['GCP_SERVICE_ACCOUNT_KEY'])
cloud_run_url = os.environ['BPM_SERVICE_URL']

# Create credentials
from google.oauth2 import service_account
credentials = service_account.Credentials.from_service_account_info(
    service_account_info,
    scopes=['https://www.googleapis.com/auth/cloud-platform']
)

# Get ID token with audience
request = Request()
id_token_obj = id_token.fetch_id_token(request, cloud_run_url)

# Call Cloud Run service
response = requests.post(
    f"{cloud_run_url}/bpm",
    headers={
        "Authorization": f"Bearer {id_token_obj}",
        "Content-Type": "application/json",
    },
    json={"url": preview_url}
)

data = response.json()
```

### Example: Other Platforms

The same pattern applies to any platform:

1. Store the service account JSON key securely (environment variables, secrets manager, etc.)
2. Use the Google Auth library for your language to mint an Identity Token
3. Set the audience to your Cloud Run service URL
4. Include the token in the `Authorization: Bearer` header

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

- Cloud Run IAM authentication (no public access)
- SSRF protection with host whitelisting
- Download size and timeout limits
- Redirect validation
- No audio persistence (temp files deleted immediately)
- Minimal logging (URLs with tokens are not logged)

## Troubleshooting

### "Permission denied" errors

Ensure the service account has `roles/run.invoker` permission:

```bash
PROJECT_ID="your-project-id"
REGION="your-region"
SERVICE_NAME="bpm-service"
SERVICE_ACCOUNT="your-service-account"

gcloud run services add-iam-policy-binding ${SERVICE_NAME} \
    --region=${REGION} \
    --member="serviceAccount:${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/run.invoker" \
    --project=${PROJECT_ID}
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
PROJECT_ID="your-project-id"
REGION="your-region"
SERVICE_NAME="bpm-service"

gcloud run services update ${SERVICE_NAME} \
    --region=${REGION} \
    --memory 4Gi \
    --project=${PROJECT_ID}
```

## License

GNU General Public License v3.0
