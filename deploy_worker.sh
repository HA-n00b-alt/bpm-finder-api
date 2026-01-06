#!/bin/bash
set -euo pipefail

# Configuration
PROJECT_ID="${PROJECT_ID:-bpm-api-microservice}"
REGION="${REGION:-europe-west3}"
SERVICE_NAME="bpm-worker"
ARTIFACT_REPO="bpm-repo"
PUBSUB_TOPIC="bpm-analysis-tasks"
PUBSUB_SUBSCRIPTION="bpm-analysis-worker-sub"

# Image tag
IMAGE_TAG="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPO}/${SERVICE_NAME}:latest"

echo "🚀 Deploying BPM Worker Service to Cloud Run"
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Service: ${SERVICE_NAME}"
echo ""

# Set GCP project
gcloud config set project "${PROJECT_ID}"

# Check if user has necessary permissions
echo "🔍 Checking permissions..."
CURRENT_USER=$(gcloud config get-value account)
if [ -z "${CURRENT_USER}" ]; then
    echo "❌ Error: No active gcloud account. Run 'gcloud auth login' first."
    exit 1
fi
echo "Authenticated as: ${CURRENT_USER}"
echo ""

# Enable required APIs
echo "📡 Enabling required APIs..."
gcloud services enable \
    pubsub.googleapis.com \
    firestore.googleapis.com \
    --project="${PROJECT_ID}" || true
echo ""

# Create Pub/Sub topic if it doesn't exist
echo "📨 Checking Pub/Sub topic..."
if ! gcloud pubsub topics describe "${PUBSUB_TOPIC}" \
    --project="${PROJECT_ID}" &>/dev/null; then
    echo "Creating Pub/Sub topic: ${PUBSUB_TOPIC}"
    gcloud pubsub topics create "${PUBSUB_TOPIC}" \
        --project="${PROJECT_ID}"
    echo "✅ Topic created successfully"
else
    echo "Topic already exists: ${PUBSUB_TOPIC}"
fi
echo ""

# Check if Artifact Registry repository exists, create if not
echo "📦 Checking Artifact Registry repository..."
if ! gcloud artifacts repositories describe "${ARTIFACT_REPO}" \
    --location="${REGION}" \
    --project="${PROJECT_ID}" &>/dev/null; then
    echo "Creating Artifact Registry repository: ${ARTIFACT_REPO}"
    if ! gcloud artifacts repositories create "${ARTIFACT_REPO}" \
        --repository-format=docker \
        --location="${REGION}" \
        --description="BPM service Docker images" \
        --project="${PROJECT_ID}"; then
        echo ""
        echo "❌ Error: Failed to create Artifact Registry repository."
        echo "You may need to enable the Artifact Registry API:"
        echo "  gcloud services enable artifactregistry.googleapis.com --project=${PROJECT_ID}"
        exit 1
    fi
    echo "✅ Repository created successfully"
else
    echo "Repository already exists: ${ARTIFACT_REPO}"
fi
echo ""

# Build and push image using Cloud Build
# Use temporary build context to avoid mutating the repo
echo "📦 Building and pushing Docker image..."
TEMP_DIR=$(mktemp -d)
cp worker.py "${TEMP_DIR}/"
cp shared_processing.py "${TEMP_DIR}/"
cp requirements.txt "${TEMP_DIR}/requirements.txt"
cp Dockerfile.worker "${TEMP_DIR}/Dockerfile"

if ! gcloud builds submit \
    --tag "${IMAGE_TAG}" \
    --region "${REGION}" \
    "${TEMP_DIR}"; then
    rm -rf "${TEMP_DIR}"
    echo ""
    echo "❌ Error: Cloud Build failed. You may need additional permissions."
    echo ""
    echo "Required IAM roles:"
    echo "  - roles/cloudbuild.builds.editor (Cloud Build Editor)"
    echo "  - roles/artifactregistry.writer (Artifact Registry Writer)"
    echo "  - roles/iam.serviceAccountUser (Service Account User)"
    echo ""
    echo "To grant yourself these roles, run:"
    echo "  gcloud projects add-iam-policy-binding ${PROJECT_ID} \\"
    echo "    --member=\"user:${CURRENT_USER}\" \\"
    echo "    --role=\"roles/cloudbuild.builds.editor\""
    echo ""
    echo "  gcloud projects add-iam-policy-binding ${PROJECT_ID} \\"
    echo "    --member=\"user:${CURRENT_USER}\" \\"
    echo "    --role=\"roles/artifactregistry.writer\""
    echo ""
    echo "  gcloud projects add-iam-policy-binding ${PROJECT_ID} \\"
    echo "    --member=\"user:${CURRENT_USER}\" \\"
    echo "    --role=\"roles/iam.serviceAccountUser\""
    echo ""
    exit 1
fi

rm -rf "${TEMP_DIR}"
echo "✅ Image built and pushed successfully"
echo ""

# Deploy to Cloud Run (allow unauthenticated for Pub/Sub push)
# High resources for audio processing, high concurrency for parallel tasks
echo "🚢 Deploying to Cloud Run..."
if ! gcloud run deploy "${SERVICE_NAME}" \
    --image "${IMAGE_TAG}" \
    --region "${REGION}" \
    --platform managed \
    --no-allow-unauthenticated \
    --port 8080 \
    --memory 4Gi \
    --cpu 4 \
    --timeout 600s \
    --max-instances 20 \
    --concurrency 10 \
    --cpu-boost \
    --set-env-vars "GOOGLE_CLOUD_PROJECT=${PROJECT_ID}" \
    --project "${PROJECT_ID}"; then
    echo ""
    echo "❌ Error: Cloud Run deployment failed. You may need additional permissions."
    echo ""
    echo "Required IAM role:"
    echo "  - roles/run.admin (Cloud Run Admin)"
    echo ""
    echo "To grant yourself this role, run:"
    echo "  gcloud projects add-iam-policy-binding ${PROJECT_ID} \\"
    echo "    --member=\"user:${CURRENT_USER}\" \\"
    echo "    --role=\"roles/run.admin\""
    echo ""
    exit 1
fi

# Get service URL
WORKER_URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --region "${REGION}" \
    --format "value(status.url)" \
    --project "${PROJECT_ID}")

echo ""
echo "✅ Deployment complete!"
echo "Worker URL: ${WORKER_URL}"
echo ""

# Create Pub/Sub push subscription if it doesn't exist
echo "📨 Creating Pub/Sub push subscription..."
if ! gcloud pubsub subscriptions describe "${PUBSUB_SUBSCRIPTION}" \
    --project="${PROJECT_ID}" &>/dev/null; then
    echo "Creating push subscription: ${PUBSUB_SUBSCRIPTION}"
    gcloud pubsub subscriptions create "${PUBSUB_SUBSCRIPTION}" \
        --topic="${PUBSUB_TOPIC}" \
        --push-endpoint="${WORKER_URL}/pubsub/process" \
        --ack-deadline=600 \
        --project="${PROJECT_ID}"
    echo "✅ Subscription created successfully"
else
    echo "Subscription already exists: ${PUBSUB_SUBSCRIPTION}"
    echo "Updating push endpoint..."
    gcloud pubsub subscriptions update "${PUBSUB_SUBSCRIPTION}" \
        --push-endpoint="${WORKER_URL}/pubsub/process" \
        --ack-deadline=600 \
        --project="${PROJECT_ID}"
    echo "✅ Subscription updated"
fi
echo ""

# Grant Pub/Sub permission to invoke the worker service
echo "🔐 Granting Pub/Sub permission to invoke worker service..."
# Get the Pub/Sub service account
PUBSUB_SA="service-$(gcloud projects describe ${PROJECT_ID} --format='value(projectNumber)')@gcp-sa-pubsub.iam.gserviceaccount.com"

# Grant invoke permission
gcloud run services add-iam-policy-binding "${SERVICE_NAME}" \
    --region "${REGION}" \
    --member "serviceAccount:${PUBSUB_SA}" \
    --role "roles/run.invoker" \
    --project "${PROJECT_ID}"

echo ""
echo "✅ Worker service configured!"
echo ""
echo "To test locally, run:"
echo "  curl ${WORKER_URL}/health"
echo ""
echo "Pub/Sub Topic: ${PUBSUB_TOPIC}"
echo "Pub/Sub Subscription: ${PUBSUB_SUBSCRIPTION}"
echo "Worker URL: ${WORKER_URL}"

