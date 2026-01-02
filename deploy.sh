#!/bin/bash
set -euo pipefail

# Configuration
PROJECT_ID="${PROJECT_ID:-bpm-api-microservice}"
REGION="${REGION:-europe-west3}"
SERVICE_NAME="bpm-service"
ARTIFACT_REPO="bpm-repo"
SERVICE_ACCOUNT="vercel-bpm-invoker"

# Image tag
IMAGE_TAG="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPO}/${SERVICE_NAME}:latest"

echo "🚀 Deploying BPM Service to Cloud Run"
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

# Build and push image using Cloud Build
echo "📦 Building and pushing Docker image..."
if ! gcloud builds submit \
    --tag "${IMAGE_TAG}" \
    --region "${REGION}"; then
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

# Deploy to Cloud Run (without public access)
echo "🚢 Deploying to Cloud Run..."
if ! gcloud run deploy "${SERVICE_NAME}" \
    --image "${IMAGE_TAG}" \
    --region "${REGION}" \
    --platform managed \
    --no-allow-unauthenticated \
    --port 8080 \
    --memory 2Gi \
    --cpu 2 \
    --timeout 60s \
    --max-instances 10 \
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
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --region "${REGION}" \
    --format "value(status.url)" \
    --project "${PROJECT_ID}")

echo ""
echo "✅ Deployment complete!"
echo "Service URL: ${SERVICE_URL}"
echo ""

# Create service account if it doesn't exist
echo "🔐 Setting up service account..."
if ! gcloud iam service-accounts describe "${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com" \
    --project "${PROJECT_ID}" &>/dev/null; then
    echo "Creating service account: ${SERVICE_ACCOUNT}"
    gcloud iam service-accounts create "${SERVICE_ACCOUNT}" \
        --display-name "Vercel BPM Invoker" \
        --project "${PROJECT_ID}"
else
    echo "Service account already exists: ${SERVICE_ACCOUNT}"
fi

# Grant invoke permission
echo "Granting invoke permission to service account..."
gcloud run services add-iam-policy-binding "${SERVICE_NAME}" \
    --region "${REGION}" \
    --member "serviceAccount:${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role "roles/run.invoker" \
    --project "${PROJECT_ID}"

echo ""
echo "✅ Service account configured!"
echo ""
echo "To test locally, run:"
echo "  TOKEN=\$(gcloud auth print-identity-token)"
echo "  curl -H \"Authorization: Bearer \$TOKEN\" ${SERVICE_URL}/health"
echo ""
echo "Service account email: ${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com"

