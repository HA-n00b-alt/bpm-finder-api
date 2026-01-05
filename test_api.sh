#!/bin/bash

# Test script for BPM Finder API (Batch Processing)
# 
# Usage: ./test_api.sh [max_confidence] [debug_level]
#   max_confidence: Confidence threshold (0.0-1.0), default: 0.65
#   debug_level: minimal, normal (default), detailed
#
# The script tests the /analyze/batch endpoint with multiple audio URLs.
# Expected response: Array of BPMResponse objects with separate Essentia and Librosa fields.
# Librosa fields (bpm_librosa, key_librosa, etc.) will be null if fallback was not used.

SERVICE_URL="https://bpm-service-340051416180.europe-west3.run.app"
MAX_CONFIDENCE="${1:-0.65}"
DEBUG_LEVEL="${2:-normal}"

# Test URLs
TEST_URLS=(
    "https://audio-ssl.itunes.apple.com/itunes-assets/AudioPreview126/v4/a4/a6/07/a4a60792-ae3e-6776-d0a9-535498919cf4/mzaf_100636618272118686.plus.aac.p.m4a"
    "https://audio-ssl.itunes.apple.com/itunes-assets/AudioPreview211/v4/b6/1b/ad/b61bada7-821a-b2da-3ca2-9d9c5438e647/mzaf_4131481278294988624.plus.aac.p.m4a"
    "https://audio-ssl.itunes.apple.com/itunes-assets/AudioPreview126/v4/ad/d4/f9/add4f9be-9e99-592e-200e-29f86c325dbc/mzaf_12598569366036027534.plus.aac.p.m4a"
)

echo "Testing BPM Finder API (Batch Processing)..."
echo "Service: $SERVICE_URL"
echo "Max Confidence Threshold: $MAX_CONFIDENCE"
echo "Debug Level: $DEBUG_LEVEL"
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
JSON_PAYLOAD+="], \"max_confidence\": $MAX_CONFIDENCE, \"debug_level\": \"$DEBUG_LEVEL\"}"

# Make the API request
echo "Making batch API request..."
echo ""

# Save response to a temp file to check status and content
TEMP_RESPONSE=$(mktemp)
HTTP_CODE=$(curl -s -w "\n%{http_code}" -X POST "${SERVICE_URL}/analyze/batch" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    -d "$JSON_PAYLOAD" \
    -o "$TEMP_RESPONSE")

# Extract HTTP status code (last line)
STATUS_CODE=$(echo "$HTTP_CODE" | tail -n1)
RESPONSE_BODY=$(head -n -1 "$TEMP_RESPONSE" 2>/dev/null || cat "$TEMP_RESPONSE")

echo "HTTP Status Code: $STATUS_CODE"
echo ""

if [ "$STATUS_CODE" -eq 200 ]; then
    # Try to format as JSON
    echo "✅ Success! Response (formatted JSON):"
    echo ""
    echo "$RESPONSE_BODY" | python3 -m json.tool 2>/dev/null || {
        echo "Response (raw):"
        echo "$RESPONSE_BODY"
    }
    echo ""
    echo "Note: Librosa fields (bpm_librosa, key_librosa, etc.) will be null if fallback was not used."
else
    echo "❌ Error Response (HTTP $STATUS_CODE):"
    echo "$RESPONSE_BODY"
fi

# Cleanup
rm -f "$TEMP_RESPONSE"

echo ""
echo "Done!"

