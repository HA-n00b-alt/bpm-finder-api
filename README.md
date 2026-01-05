# BPM Finder API

A Google Cloud Run microservice that computes BPM (beats per minute) and musical key from 30-second audio preview URLs. The service is deployed as a **private** Cloud Run service, requiring Google Cloud IAM authentication.

## Features

- **Single-Method BPM Extraction**: Uses Essentia's `RhythmExtractor2013(method="multifeature")` for BPM detection
- **Selective Fallback Architecture**: Automatically uses a high-accuracy fallback service (librosa-based) only when needed:
  - BPM fallback: Triggered when BPM confidence < `max_confidence` threshold
  - Key fallback: Triggered when key strength < `max_confidence` threshold
  - Can call fallback for BPM only, key only, both, or neither
- **Configurable Confidence Threshold**: `max_confidence` parameter (default 0.65) controls when fallback is triggered
- **Musical Key Detection**: Multiple Essentia key profile types with automatic selection of best result
- **Normalized Confidence Scores**: All confidence values normalized to 0-1 range for consistent interpretation
- **Comprehensive Debug Information**: Detailed debug info including method comparisons, confidence analysis, and error reporting
- **Private Cloud Run Service**: IAM authentication required for access
- **SSRF Protection**: HTTPS-only requirement with redirect validation
- **Fast Processing**: Essentia and ffmpeg for efficient audio processing
- **Complete Response Data**: Returns BPM, raw BPM, BPM confidence, BPM method, debug info, key, scale, and key confidence

## Architecture

### Primary Service (`bpm-service`)

- **Runtime**: Python 3 + FastAPI + Uvicorn
- **Audio Processing**: Essentia (RhythmExtractor2013 for BPM, KeyExtractor for key detection) + ffmpeg
- **BPM Method**: Single method:
  - `multifeature`: RhythmExtractor2013 with multifeature method (confidence range: 0-5.32)
- **Container**: MTG Essentia base image (`ghcr.io/mtg/essentia:latest`)
- **Deployment**: Google Cloud Run
- **Authentication**: Cloud Run IAM (Identity Tokens)
- **Fallback Integration**: Selectively calls fallback service based on `max_confidence` threshold (default 0.65)

### Fallback Service (`bpm-fallback-service`)

- **Runtime**: Python 3 + FastAPI + Uvicorn
- **Audio Processing**: Librosa (HPSS, beat tracking, chroma features)
- **BPM Method**: Harmonic-Percussive Source Separation (HPSS) with percussive component beat tracking
- **Key Method**: Krumhansl-Schmuckler algorithm on harmonic component chroma features
- **Container**: Python 3.11-slim with librosa, numpy, scipy
- **Deployment**: Google Cloud Run (higher resources: 4GB RAM, 2 CPU, CPU boost)
- **Authentication**: Cloud Run IAM (service-to-service authentication)
- **Use Case**: High-accuracy, high-cost fallback for low-confidence primary results

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

### Primary Service Configuration

Before deployment, configure the following variables in `deploy.sh` or set them as environment variables:

- `PROJECT_ID`: Your GCP project ID (default: `bpm-api-microservice`)
- `REGION`: Cloud Run region (default: `europe-west3`)
- `SERVICE_NAME`: Cloud Run service name (default: `bpm-service`)
- `ARTIFACT_REPO`: Artifact Registry repository name (default: `bpm-repo`)
- `SERVICE_ACCOUNT`: Service account name for external callers (default: `vercel-bpm-invoker`)

### Fallback Service Configuration

The fallback service configuration is in `deploy_fallback.sh`:

- `PROJECT_ID`: Your GCP project ID (same as primary service)
- `REGION`: Cloud Run region (default: `europe-west3`)
- `SERVICE_NAME`: Fallback service name (default: `bpm-fallback-service`)
- `ARTIFACT_REPO`: Artifact Registry repository name (default: `bpm-repo`)

### Primary Service Fallback Settings

The primary service fallback configuration is in `main.py`:

- `FALLBACK_SERVICE_URL`: URL of the fallback service (must be updated after fallback deployment)
- `FALLBACK_SERVICE_AUDIENCE`: OIDC audience for service-to-service authentication (same as `FALLBACK_SERVICE_URL`)

**Note**: The confidence threshold is now configurable per-request via the `max_confidence` parameter (default: 0.65). This allows clients to control when fallback is triggered.

**Important**: After deploying the fallback service, update `FALLBACK_SERVICE_URL` and `FALLBACK_SERVICE_AUDIENCE` in `main.py` with the actual fallback service URL, then redeploy the primary service.

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

### Deploy Primary Service

To deploy the primary BPM service to Google Cloud Run:

```bash
# Set your project ID
export PROJECT_ID="your-project-id"

# Deploy with default region (europe-west3)
./deploy.sh

# Or override region
REGION=us-central1 ./deploy.sh
```

**Note**: Make sure you've completed the [One-Time Setup](#one-time-setup) steps before deploying.

The `deploy.sh` script will:
1. Build the Docker image using Cloud Build
2. Push to Artifact Registry
3. Deploy to Cloud Run **without public access** (`--no-allow-unauthenticated`)
4. Create/verify the service account
5. Grant `roles/run.invoker` permission to the service account

### Deploy Fallback Service

To deploy the high-accuracy fallback service:

```bash
# Set your project ID (same as primary service)
export PROJECT_ID="your-project-id"

# Deploy fallback service
./deploy_fallback.sh

# Or override region
REGION=us-central1 ./deploy_fallback.sh
```

The `deploy_fallback.sh` script will:
1. Build the Docker image using Cloud Build
2. Push to Artifact Registry
3. Deploy to Cloud Run **without public access** with higher resources (4GB RAM, 2 CPU, CPU boost)
4. Configure service-to-service authentication for primary service calls

**Important**: The primary service must be able to authenticate to the fallback service. The primary service uses its default Cloud Run service account to generate OIDC tokens for calling the fallback service.

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

**Option 1: Using the test script**

A convenient test script is provided:

```bash
# Test with default URL
./test_api.sh

# Test with custom URL
./test_api.sh "https://audio-ssl.itunes.apple.com/..."
```

**Option 2: Manual curl command**

```bash
# Get identity token
TOKEN=$(gcloud auth print-identity-token)

# Replace with your actual service URL and a valid preview URL
SERVICE_URL="https://your-service-url.run.app"
PREVIEW_URL="https://audio-ssl.itunes.apple.com/..."

curl -X POST \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"url\": \"${PREVIEW_URL}\", \"max_confidence\": 0.65}" \
    "${SERVICE_URL}/bpm" | python3 -m json.tool
```

Expected response:
```json
{
  "bpm": 128,
  "bpm_raw": 128.0,
  "bpm_confidence": 0.77,
  "bpm_method": "multifeature",
  "debug_info": "Max confidence threshold: 0.65\nURL fetch: SUCCESS (https://audio-ssl.itunes.apple.com/...)\nAudio conversion: SUCCESS\n=== BPM Analysis (Essentia) ===\nBPM=128.0 (normalized=128.0)\nConfidence: raw=4.10 (range: 0-5.32), normalized=0.77 (0-1), quality=excellent\nBPM confidence (0.770) >= threshold (0.65) - using Essentia result\n=== Key Analysis (Essentia) ===\nkey_profile=temperley: key=C major, strength=0.859\nWinner: temperley profile (strength=0.859, normalized=0.859)\nKey strength (0.859) >= threshold (0.65) - using Essentia result",
  "key": "C",
  "scale": "major",
  "key_confidence": 0.86
}
```

**Example with fallback triggered:**
```json
{
  "bpm": 130,
  "bpm_raw": 130.0,
  "bpm_confidence": 0.82,
  "bpm_method": "librosa_hpss_fallback",
  "debug_info": "Max confidence threshold: 0.65\nURL fetch: SUCCESS (...)\nAudio conversion: SUCCESS\n=== BPM Analysis (Essentia) ===\nBPM=130.0 (normalized=130.0)\nConfidence: raw=2.50 (range: 0-5.32), normalized=0.47 (0-1), quality=moderate\nBPM confidence (0.470) < threshold (0.65) - fallback needed\n=== Key Analysis (Essentia) ===\nkey_profile=temperley: key=C major, strength=0.859\nWinner: temperley profile (strength=0.859, normalized=0.859)\nKey strength (0.859) >= threshold (0.65) - using Essentia result\n=== Fallback Service Call (BPM) ===\nFallback service: SUCCESS\nFallback BPM: 130.0 (confidence=0.82)",
  "key": "C",
  "scale": "major",
  "key_confidence": 0.86
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

Compute BPM and key from audio preview URL.

**Request Body:**
```json
{
  "url": "https://audio-ssl.itunes.apple.com/...",
  "max_confidence": 0.65
}
```

**Request Parameters:**
- `url` (required): HTTPS URL to the audio preview file
- `max_confidence` (optional, default: 0.65): Confidence threshold (0.0-1.0) below which the fallback service is called. If Essentia's confidence is above this threshold, the primary service result is used.

**Response:**
```json
{
  "bpm": 128,
  "bpm_raw": 64.0,
  "bpm_confidence": 0.73,
  "bpm_method": "multifeature",
  "debug_info": "URL fetch: SUCCESS (https://audio-ssl.itunes.apple.com/...)\nAudio conversion: SUCCESS\n=== BPM Analysis ===\nmultifeature: BPM=128.0 (norm=128.0), confidence=2.319 (norm=0.77)\ndegara: BPM=128.0 (norm=128.0), confidence=0.000 (norm=0.00)\nonset: BPM=128.0 (norm=128.0), confidence=1.500 (norm=0.50)\nEnsemble: avg_confidence=0.42, range=[0.00, 0.77]\nWinner: multifeature (confidence=0.77)\n=== Key Analysis ===\nkey_profile=temperley: key=C major, strength=0.859\nWinner: temperley profile (strength=0.859)",
  "key": "C",
  "scale": "major",
  "key_confidence": 0.86
}
```

**Field Descriptions:**
- `bpm`: Normalized BPM (integer, rounded from normalized value). Only extreme outliers are corrected:
  - If raw BPM < 40: multiplied by 2
  - If raw BPM > 220: divided by 2
  - Otherwise: returned unchanged
- `bpm_raw`: Raw BPM value from Essentia or fallback service (before normalization, rounded to 2 decimal places)
- `bpm_confidence`: BPM confidence score (0.0-1.0, rounded to 2 decimal places). Normalized from Essentia's raw confidence values (0-5.32 range). Higher value indicates more reliable BPM detection.
- `bpm_method`: The BPM extraction method used:
  - `"multifeature"`: Essentia RhythmExtractor2013 with multifeature method (primary service)
  - `"librosa_hpss_fallback"`: Fallback service was used (confidence was below `max_confidence` threshold)
- `debug_info`: Comprehensive debug information string including:
  - URL fetch status
  - Audio conversion status
  - BPM analysis details (all three methods with raw and normalized confidence)
  - Key analysis details (all profile types tested)
  - Fallback service status (if triggered)
  - Error messages (if any)
- `key`: Detected musical key (e.g., "C", "D", "E", "F", "G", "A", "B")
- `scale`: Detected scale ("major" or "minor")
- `key_confidence`: Key detection confidence score (0.0-1.0, rounded to 2 decimal places). Higher value indicates more reliable key detection.

**Error Responses:**
- `400`: Invalid URL, non-HTTPS URL, file too large, or redirect to non-HTTPS URL
- `500`: Processing error (download, conversion, BPM computation, or key detection failed)

## SSRF Protection

The service implements SSRF protection through:

- Only `https://` URLs allowed
- Redirect validation (ensures redirects stay on HTTPS)
- Download limits:
  - Connect timeout: 5 seconds
  - Total timeout: 20 seconds
  - Max file size: 10MB

**Note**: Host/domain restrictions are handled through authentication. Only authenticated callers can access the service.

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

### Primary Service Flow

1. **Download**: Fetch audio from preview URL (with SSRF protection)
2. **Convert**: Use ffmpeg to convert to mono 44100Hz 16-bit PCM WAV
3. **BPM Analysis**:
   - Use `RhythmExtractor2013(method="multifeature")` to extract BPM and confidence
   - Confidence range: 0-5.32 (raw), normalized to 0-1
   - Quality levels:
     - [0, 1): very low confidence
     - [1, 2): low confidence
     - [2, 3): moderate confidence
     - [3, 3.5): high confidence
     - (3.5, 5.32]: excellent confidence
   - Check if normalized confidence >= `max_confidence` threshold
4. **Key Analysis**: Use Essentia `KeyExtractor` with multiple profile types (temperley, krumhansl, edma, edmm) and select best result
   - Check if normalized strength >= `max_confidence` threshold
5. **Selective Fallback Decision**:
   - If BPM confidence < `max_confidence`: Note need for BPM fallback
   - If key strength < `max_confidence`: Note need for key fallback
   - Call fallback service for BPM only, key only, both, or neither
6. **Fallback Service Call** (if needed):
   - Upload WAV file to fallback service
   - Receive high-accuracy results from librosa-based processing
   - Overwrite only the results that needed fallback (BPM, key, or both)
7. **Normalize BPM**: Adjust for extreme outliers only:
   - If BPM < 40: multiply by 2
   - If BPM > 220: divide by 2
   - Otherwise: return unchanged
8. **Cleanup**: Delete temporary files
9. **Return**: JSON response with BPM and key data

### Fallback Service Flow (when triggered)

1. **Receive**: WAV file upload from primary service
2. **Load**: Load audio with librosa (44100Hz, mono)
3. **HPSS**: Apply Harmonic-Percussive Source Separation:
   - **Percussive component**: Used for BPM detection
   - **Harmonic component**: Used for key detection
4. **BPM Extraction**: Use `librosa.beat.beat_track()` on percussive component
5. **Key Extraction**: Use `librosa.feature.chroma_stft()` on harmonic component + Krumhansl-Schmuckler algorithm
6. **Confidence Calculation**: Calculate BPM confidence from beat consistency (capped at 0.85)
7. **Return**: High-accuracy results to primary service

## Algorithms and Confidence Ranges

### Primary Service (Essentia)

**BPM Extraction:**
- **RhythmExtractor2013(method="multifeature")**: Confidence range 0-5.32, normalized by dividing by 5.32 (values > 5.32 clamped to 1.0)
  - Quality guidelines:
    - [0, 1): very low confidence
    - [1, 2): low confidence
    - [2, 3): moderate confidence
    - [3, 3.5): high confidence
    - (3.5, 5.32]: excellent confidence

**Key Detection:**
- **KeyExtractor**: Strength values (range varies by profile type). The service tries multiple profiles (temperley, krumhansl, edma, edmm) and selects the best result.

### Fallback Service (Librosa)

**BPM Extraction:**
- **librosa.beat.beat_track()**: No built-in confidence. Custom confidence calculated from beat consistency (coefficient of variation), capped at 0.85.

**Key Detection:**
- **librosa.feature.chroma_stft() + Krumhansl-Schmuckler**: Correlation values (-1 to 1) normalized to 0-1 using `(corr + 1) / 2`.

## Confidence Normalization

The service normalizes confidence values from different algorithms to a consistent 0-1 range:

- **Essentia RhythmExtractor2013(method="multifeature")**: Confidence range 0-5.32, normalized by dividing by 5.32 (values > 5.32 clamped to 1.0)
  - Quality levels are determined from raw confidence before normalization
- **Essentia KeyExtractor**: Strength values (typically 0-1 range), used as-is if already 0-1, otherwise clamped to [0, 1]
- **Librosa beat_track**: Custom confidence calculation from beat consistency (already 0-1, capped at 0.85)
- **Krumhansl-Schmuckler**: Correlation values (-1 to 1) normalized to 0-1 using `(corr + 1) / 2`

## Security Notes

- Cloud Run IAM authentication (no public access)
- Service-to-service authentication for fallback calls (OIDC tokens)
- SSRF protection through HTTPS-only requirement and redirect validation
- Download size and timeout limits
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

### "Invalid URL" errors

Ensure the URL:
- Starts with `https://`
- Is a valid, accessible URL

### ffmpeg conversion errors

Ensure the audio file is a valid format. The service supports common audio formats (MP3, M4A, etc.) that ffmpeg can handle.

### High memory usage

The primary service is configured with 2GB memory. For very large files or high concurrency, consider increasing:

```bash
PROJECT_ID="your-project-id"
REGION="your-region"
SERVICE_NAME="bpm-service"

gcloud run services update ${SERVICE_NAME} \
    --region=${REGION} \
    --memory 4Gi \
    --project=${PROJECT_ID}
```

The fallback service is already configured with 4GB memory and 2 CPU cores for high-accuracy processing.

### Fallback service not being called

If the fallback service is not being triggered when expected:

1. **Check confidence threshold**: Ensure the primary service's `FALLBACK_THRESHOLD` in `main.py` is set correctly (default: 0.70)
2. **Verify fallback URL**: Ensure `FALLBACK_SERVICE_URL` in `main.py` matches the deployed fallback service URL
3. **Check authentication**: The primary service needs permission to call the fallback service. Ensure the primary service's default Cloud Run service account has `roles/run.invoker` permission on the fallback service
4. **Check debug_info**: The `debug_info` field in the response will indicate if fallback was triggered and any errors encountered

### Fallback service authentication errors

If you see authentication errors when the fallback is triggered:

```bash
PROJECT_ID="your-project-id"
REGION="your-region"
PRIMARY_SERVICE="bpm-service"
FALLBACK_SERVICE="bpm-fallback-service"

# Get the primary service's service account
PRIMARY_SA=$(gcloud run services describe ${PRIMARY_SERVICE} \
    --region=${REGION} \
    --format="value(spec.template.spec.serviceAccountName)" \
    --project=${PROJECT_ID})

# Grant invoker permission
gcloud run services add-iam-policy-binding ${FALLBACK_SERVICE} \
    --region=${REGION} \
    --member="serviceAccount:${PRIMARY_SA}" \
    --role="roles/run.invoker" \
    --project=${PROJECT_ID}
```

## License

GNU General Public License v3.0
