#!/bin/bash

# Test script for BPM Finder API
# Usage: ./test_api.sh [URL]

SERVICE_URL="https://bpm-service-340051416180.europe-west3.run.app"
TEST_URL="${1:-https://audio-ssl.itunes.apple.com/itunes-assets/AudioPreview126/v4/a4/a6/07/a4a60792-ae3e-6776-d0a9-535498919cf4/mzaf_100636618272118686.plus.aac.p.m4a}"

echo "Testing BPM Finder API..."
echo "Service: $SERVICE_URL"
echo "Audio URL: $TEST_URL"
echo ""

# Get authentication token
echo "Getting authentication token..."
TOKEN=$(gcloud auth print-identity-token 2>/dev/null)

if [ -z "$TOKEN" ]; then
    echo "Error: Failed to get authentication token"
    exit 1
fi

# Make the API request
echo "Making API request..."
echo ""

curl -s -X POST "${SERVICE_URL}/bpm" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    -d "{\"url\": \"${TEST_URL}\"}" \
    | python3 -m json.tool

echo ""
echo "Done!"

