#!/bin/bash

# Test script for BPM Finder API (Async Streaming Architecture)
# 
# Usage: ./test_api.sh [max_confidence] [debug_level]
#   max_confidence: Confidence threshold (0.0-1.0), default: 0.65
#   debug_level: minimal, normal (default), detailed
#
# The script tests the new async streaming architecture:
# 1. Submits batch via POST /analyze/batch (returns batch_id immediately)
# 2. Streams results via GET /stream/{batch_id} (NDJSON format)
# 3. Optionally checks final status via GET /batch/{batch_id}

SERVICE_URL="https://bpm-service-pgkjwjbhqq-ey.a.run.app"
MAX_CONFIDENCE="${1:-0.65}"
DEBUG_LEVEL="${2:-normal}"

# Test URLs
TEST_URLS=(
    "https://audio-ssl.itunes.apple.com/itunes-assets/AudioPreview211/v4/37/d8/c9/37d8c9f7-838e-a222-0fe7-141c39e9f51c/mzaf_9144538674023163935.plus.aac.p.m4a"
    "https://audio-ssl.itunes.apple.com/itunes-assets/AudioPreview125/v4/6c/45/59/6c4559aa-e474-1366-66e7-9cd5279acd05/mzaf_11210908680488989277.plus.aac.p.m4a"
    "https://audio-ssl.itunes.apple.com/itunes-assets/AudioPreview221/v4/26/7b/cf/267bcf9d-abeb-2703-1b63-05528298a273/mzaf_12137933217638002816.plus.aac.p.m4a"
    "https://audio-ssl.itunes.apple.com/itunes-assets/AudioPreview122/v4/98/51/0b/98510b2f-0e16-e295-5527-b014e43ae3a5/mzaf_918609115701790643.plus.aac.p.m4a"
    "https://audio-ssl.itunes.apple.com/itunes-assets/AudioPreview115/v4/60/04/f7/6004f766-8053-0ef6-36a5-0e19f2533d13/mzaf_10236136789037090616.plus.aac.p.m4a"
)

echo "Testing BPM Finder API (Async Streaming Architecture)..."
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

# Step 1: Submit batch and get batch_id
echo "Step 1: Submitting batch..."
echo ""

START_TIME=$(date +%s)
TEMP_RESPONSE=$(mktemp)
HTTP_CODE=$(curl -s -w "\n%{http_code}" -X POST "${SERVICE_URL}/analyze/batch" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    -d "$JSON_PAYLOAD" \
    -o "$TEMP_RESPONSE")

SUBMIT_TIME=$(date +%s)
SUBMIT_ELAPSED=$((SUBMIT_TIME - START_TIME))

STATUS_CODE=$(echo "$HTTP_CODE" | tail -n1)
RESPONSE_BODY=$(head -n -1 "$TEMP_RESPONSE" 2>/dev/null || cat "$TEMP_RESPONSE")

echo "HTTP Status Code: $STATUS_CODE"
echo ""

if [ "$STATUS_CODE" -ne 200 ]; then
    echo "❌ Error submitting batch (HTTP $STATUS_CODE):"
    echo "$RESPONSE_BODY"
    rm -f "$TEMP_RESPONSE"
    exit 1
fi

# Extract batch_id from response
BATCH_ID=$(echo "$RESPONSE_BODY" | python3 -c "import sys, json; print(json.load(sys.stdin)['batch_id'])" 2>/dev/null)

if [ -z "$BATCH_ID" ]; then
    echo "❌ Error: Could not extract batch_id from response"
    echo "Response: $RESPONSE_BODY"
    rm -f "$TEMP_RESPONSE"
    exit 1
fi

echo "✅ Batch submitted successfully! (took ${SUBMIT_ELAPSED}s)"
echo "Batch ID: $BATCH_ID"
echo "Response:"
echo "$RESPONSE_BODY" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE_BODY"
echo ""

rm -f "$TEMP_RESPONSE"

# Step 2: Stream results
echo "Step 2: Streaming results (NDJSON format)..."
echo "Connecting to: ${SERVICE_URL}/stream/${BATCH_ID}"
echo "Waiting for results (will timeout after 5 minutes)..."
echo ""

# Initialize timing variables
STREAM_START_TIME=$(date +%s)
LAST_WARNING_TIME=0

# Use a temporary file to track result count across subshells
RESULT_FILE=$(mktemp)
echo "0" > "$RESULT_FILE"

curl -s -N --max-time 300 -X GET "${SERVICE_URL}/stream/${BATCH_ID}" \
    -H "Authorization: Bearer $TOKEN" 2>&1 | while IFS= read -r line; do
    if [ -z "$line" ]; then
        continue
    fi
    
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - STREAM_START_TIME))
    
    # Parse NDJSON line
    TYPE=$(echo "$line" | python3 -c "import sys, json; print(json.load(sys.stdin).get('type', 'unknown'))" 2>/dev/null)
    
    case "$TYPE" in
        "status")
            STATUS=$(echo "$line" | python3 -c "import sys, json; d=json.load(sys.stdin); print(f\"Status: {d.get('status')} - Processed: {d.get('processed')}/{d.get('total')}\")" 2>/dev/null)
            echo "[${ELAPSED}s] 📊 $STATUS"
            ;;
        "result")
            CURRENT_COUNT=$(cat "$RESULT_FILE" 2>/dev/null || echo "0")
            NEW_COUNT=$((CURRENT_COUNT + 1))
            echo "$NEW_COUNT" > "$RESULT_FILE"
            INDEX=$(echo "$line" | python3 -c "import sys, json; print(json.load(sys.stdin).get('index', '?'))" 2>/dev/null)
            BPM=$(echo "$line" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('bpm_essentia') or d.get('bpm_librosa') or 'N/A')" 2>/dev/null)
            KEY=$(echo "$line" | python3 -c "import sys, json; d=json.load(sys.stdin); k=d.get('key_essentia') or d.get('key_librosa') or 'N/A'; s=d.get('scale_essentia') or d.get('scale_librosa') or ''; print(f\"{k} {s}\".strip())" 2>/dev/null)
            echo "[${ELAPSED}s] ✅ Result #${INDEX}: BPM=${BPM}, Key=${KEY}"
            ;;
        "progress")
            PROGRESS=$(echo "$line" | python3 -c "import sys, json; d=json.load(sys.stdin); print(f\"Progress: {d.get('processed')}/{d.get('total')}\")" 2>/dev/null)
            echo "[${ELAPSED}s] 📈 $PROGRESS"
            ;;
        "complete")
            TOTAL=$(echo "$line" | python3 -c "import sys, json; print(json.load(sys.stdin).get('total', '?'))" 2>/dev/null)
            TOTAL_ELAPSED=$((CURRENT_TIME - START_TIME))
            echo "[${ELAPSED}s] 🎉 Batch complete! Total results: $TOTAL (Total time: ${TOTAL_ELAPSED}s)"
            rm -f "$RESULT_FILE"
            exit 0
            ;;
        "error")
            ERROR=$(echo "$line" | python3 -c "import sys, json; print(json.load(sys.stdin).get('message', 'Unknown error'))" 2>/dev/null)
            echo "[${ELAPSED}s] ❌ Error: $ERROR"
            ;;
        *)
            echo "[${ELAPSED}s] 📄 $line"
            ;;
    esac
    
    # Show warning if no results after 30 seconds, then every 30 seconds
    CURRENT_COUNT=$(cat "$RESULT_FILE" 2>/dev/null || echo "0")
    if [ $ELAPSED -ge 30 ] && [ "$CURRENT_COUNT" -eq 0 ]; then
        if [ $((ELAPSED - LAST_WARNING_TIME)) -ge 30 ]; then
            echo "[${ELAPSED}s] ⚠️  Warning: No results yet after ${ELAPSED} seconds..."
            LAST_WARNING_TIME=$ELAPSED
        fi
    fi
done

# Cleanup and show final status
CURRENT_TIME=$(date +%s)
STREAM_ELAPSED=$((CURRENT_TIME - STREAM_START_TIME))
rm -f "$RESULT_FILE"

if [ $STREAM_ELAPSED -ge 300 ]; then
    echo ""
    echo "[${STREAM_ELAPSED}s] ⚠️  Stream timed out after 5 minutes"
fi

echo ""

# Step 3: Get final batch status
echo "Step 3: Checking final batch status..."
echo ""

FINAL_STATUS=$(curl -s -X GET "${SERVICE_URL}/batch/${BATCH_ID}" \
    -H "Authorization: Bearer $TOKEN")

END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))

echo "Final Status:"
echo "$FINAL_STATUS" | python3 -m json.tool 2>/dev/null || echo "$FINAL_STATUS"
echo ""

echo "✅ Test complete!"
echo "Batch ID: $BATCH_ID"
echo "Total elapsed time: ${TOTAL_ELAPSED}s"
echo "  - Submission: ${SUBMIT_ELAPSED}s"
echo "  - Streaming: $((END_TIME - STREAM_START_TIME))s"
echo ""
echo "You can reconnect to the stream anytime: ${SERVICE_URL}/stream/${BATCH_ID}"
