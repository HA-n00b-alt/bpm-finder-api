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
    "https://audio-ssl.itunes.apple.com/itunes-assets/AudioPreview211/v4/37/d8/c9/37d8c9f7-838e-a222-0fe7-141c39e9f51c/mzaf_9144538674023163935.plus.aac.p.m4a"
    "https://audio-ssl.itunes.apple.com/itunes-assets/AudioPreview125/v4/6c/45/59/6c4559aa-e474-1366-66e7-9cd5279acd05/mzaf_11210908680488989277.plus.aac.p.m4a"
    "https://audio-ssl.itunes.apple.com/itunes-assets/AudioPreview221/v4/26/7b/cf/267bcf9d-abeb-2703-1b63-05528298a273/mzaf_12137933217638002816.plus.aac.p.m4a"
    "https://audio-ssl.itunes.apple.com/itunes-assets/AudioPreview122/v4/98/51/0b/98510b2f-0e16-e295-5527-b014e43ae3a5/mzaf_918609115701790643.plus.aac.p.m4a"
    "https://audio-ssl.itunes.apple.com/itunes-assets/AudioPreview115/v4/60/04/f7/6004f766-8053-0ef6-36a5-0e19f2533d13/mzaf_10236136789037090616.plus.aac.p.m4a"
    "https://audio-ssl.itunes.apple.com/itunes-assets/AudioPreview125/v4/7f/9f/49/7f9f4990-13ce-6493-554b-4823a872adc0/mzaf_8275565668510064829.plus.aac.p.m4a"
    "https://audio-ssl.itunes.apple.com/itunes-assets/AudioPreview221/v4/ee/71/69/ee716969-5306-652f-e34b-c3c84881814d/mzaf_8883394412938746971.plus.aac.p.m4a"
    "https://audio-ssl.itunes.apple.com/itunes-assets/AudioPreview125/v4/d0/22/9b/d0229bfa-b891-5e1d-fa88-6232dc67a22d/mzaf_11766518334733672028.plus.aac.p.m4a"
    "https://audio-ssl.itunes.apple.com/itunes-assets/AudioPreview122/v4/f5/80/1e/f5801e26-8222-bd45-c1c5-73d78cd6df2b/mzaf_14290826928673879999.plus.aac.p.m4a"
    "https://audio-ssl.itunes.apple.com/itunes-assets/AudioPreview221/v4/72/15/9d/72159de4-6c2b-4557-ce7a-cc9a7c0b6015/mzaf_15624793323477937695.plus.aac.p.m4a"
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

