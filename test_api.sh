#!/bin/bash

# Test script for BPM Finder API (Batch Processing)
# Usage: ./test_api.sh [max_confidence]

SERVICE_URL="https://bpm-service-340051416180.europe-west3.run.app"
MAX_CONFIDENCE="${1:-0.65}"

# Test URLs
TEST_URLS=(
    "https://audio-ssl.itunes.apple.com/itunes-assets/AudioPreview126/v4/a4/a6/07/a4a60792-ae3e-6776-d0a9-535498919cf4/mzaf_100636618272118686.plus.aac.p.m4a"
    "https://audio-ssl.itunes.apple.com/itunes-assets/AudioPreview211/v4/b6/1b/ad/b61bada7-821a-b2da-3ca2-9d9c5438e647/mzaf_4131481278294988624.plus.aac.p.m4a"
    "https://audio-ssl.itunes.apple.com/itunes-assets/AudioPreview126/v4/ad/d4/f9/add4f9be-9e99-592e-200e-29f86c325dbc/mzaf_12598569366036027534.plus.aac.p.m4a"
)

echo "Testing BPM Finder API (Batch Processing)..."
echo "Service: $SERVICE_URL"
echo "Max Confidence Threshold: $MAX_CONFIDENCE"
echo "Number of URLs: ${#TEST_URLS[@]}"
echo ""

# Get authentication token
echo "Getting authentication token..."
TOKEN=$(gcloud auth print-identity-token 2>/dev/null)

if [ -z "$TOKEN" ]; then
    echo "Error: Failed to get authentication token"
    exit 1
fi

# Build JSON payload with URLs array
JSON_PAYLOAD="{\"urls\": ["
for i in "${!TEST_URLS[@]}"; do
    if [ $i -gt 0 ]; then
        JSON_PAYLOAD+=", "
    fi
    JSON_PAYLOAD+="\"${TEST_URLS[$i]}\""
done
JSON_PAYLOAD+="], \"max_confidence\": $MAX_CONFIDENCE}"

# Make the API request
echo "Making batch API request..."
echo ""

curl -s -X POST "${SERVICE_URL}/analyze/batch" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    -d "$JSON_PAYLOAD" \
    | python3 -m json.tool

echo ""
echo "Done!"

