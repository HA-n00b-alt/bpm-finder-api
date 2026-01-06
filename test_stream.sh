#!/bin/bash

# Test script for streaming endpoint only
# Usage: ./test_stream.sh <batch_id>

SERVICE_URL="https://bpm-service-340051416180.europe-west3.run.app"
BATCH_ID="${1}"

if [ -z "$BATCH_ID" ]; then
    echo "Usage: ./test_stream.sh <batch_id>"
    echo "Example: ./test_stream.sh 123e4567-e89b-12d3-a456-426614174000"
    exit 1
fi

echo "Streaming results for batch: $BATCH_ID"
echo "Endpoint: ${SERVICE_URL}/stream/${BATCH_ID}"
echo ""

# Get authentication token
TOKEN=$(gcloud auth print-identity-token 2>/dev/null)

if [ -z "$TOKEN" ]; then
    echo "Error: Failed to get authentication token"
    exit 1
fi

# Stream results
curl -s -N -X GET "${SERVICE_URL}/stream/${BATCH_ID}" \
    -H "Authorization: Bearer $TOKEN" | while IFS= read -r line; do
    if [ -z "$line" ]; then
        continue
    fi
    
    # Pretty print JSON lines
    echo "$line" | python3 -m json.tool 2>/dev/null || echo "$line"
done

