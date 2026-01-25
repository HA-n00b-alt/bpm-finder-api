#!/bin/bash
set -euo pipefail

# Concurrency stress test - tests worker's ability to handle many URLs in parallel
# This validates the optimized concurrency settings (worker: 16, essentia: 8)

SERVICE_URL="${1:-https://bpm-service-pgkjwjbhqq-ey.a.run.app}"
MAX_CONFIDENCE="${2:-0.65}"
DEBUG_LEVEL="${3:-minimal}"

# Large batch of test URLs (20 URLs to stress test concurrency)
TEST_URLS=(
    "https://cdnt-preview.dzcdn.net/api/1/1/0/1/3/0/01362a91c97d085494ad64b63b9d88f4.mp3?hdnea=exp=1767892549~acl=/api/1/1/0/1/3/0/01362a91c97d085494ad64b63b9d88f4.mp3*~data=user_id=0,application_id=42~hmac=0597d1d65a6463081580e0109d2e3218cdefd4c2cf3bb212e1fd66c51d24c80e"
    "https://p.scdn.co/mp3-preview/1c57741674dbcee27add7e06b6086d71da0cff5e"
    "https://audio-ssl.itunes.apple.com/itunes-assets/AudioPreview211/v4/37/d8/c9/37d8c9f7-838e-a222-0fe7-141c39e9f51c/mzaf_9144538674023163935.plus.aac.p.m4a"
    "https://p.scdn.co/mp3-preview/0b2426cba10cea8ffdd9d69b78f3f58073da6ab1?cid=18c612430a234d2da59d742217981da8"
    "https://audio-ssl.itunes.apple.com/itunes-assets/AudioPreview125/v4/6c/45/59/6c4559aa-e474-1366-66e7-9cd5279acd05/mzaf_11210908680488989277.plus.aac.p.m4a"
    "https://cdnt-preview.dzcdn.net/api/1/1/e/3/a/0/e3a23182a2b65997a9f4f40321865f6f.mp3?hdnea=exp=1769264366~acl=/api/1/1/e/3/a/0/e3a23182a2b65997a9f4f40321865f6f.mp3*~data=user_id=0,application_id=42~hmac=12caca3810169841acce9f630d58532ec86030b1d8e98573db1e698278bdc96f"
    "https://cdnt-preview.dzcdn.net/api/1/1/0/1/3/0/01362a91c97d085494ad64b63b9d88f4.mp3?hdnea=exp=1767892549~acl=/api/1/1/0/1/3/0/01362a91c97d085494ad64b63b9d88f4.mp3*~data=user_id=0,application_id=42~hmac=0597d1d65a6463081580e0109d2e3218cdefd4c2cf3bb212e1fd66c51d24c80e"
    "https://p.scdn.co/mp3-preview/1c57741674dbcee27add7e06b6086d71da0cff5e"
    "https://audio-ssl.itunes.apple.com/itunes-assets/AudioPreview211/v4/37/d8/c9/37d8c9f7-838e-a222-0fe7-141c39e9f51c/mzaf_9144538674023163935.plus.aac.p.m4a"
    "https://p.scdn.co/mp3-preview/0b2426cba10cea8ffdd9d69b78f3f58073da6ab1?cid=18c612430a234d2da59d742217981da8"
    "https://audio-ssl.itunes.apple.com/itunes-assets/AudioPreview125/v4/6c/45/59/6c4559aa-e474-1366-66e7-9cd5279acd05/mzaf_11210908680488989277.plus.aac.p.m4a"
    "https://cdnt-preview.dzcdn.net/api/1/1/e/3/a/0/e3a23182a2b65997a9f4f40321865f6f.mp3?hdnea=exp=1769264366~acl=/api/1/1/e/3/a/0/e3a23182a2b65997a9f4f40321865f6f.mp3*~data=user_id=0,application_id=42~hmac=12caca3810169841acce9f630d58532ec86030b1d8e98573db1e698278bdc96f"
    "https://cdnt-preview.dzcdn.net/api/1/1/0/1/3/0/01362a91c97d085494ad64b63b9d88f4.mp3?hdnea=exp=1767892549~acl=/api/1/1/0/1/3/0/01362a91c97d085494ad64b63b9d88f4.mp3*~data=user_id=0,application_id=42~hmac=0597d1d65a6463081580e0109d2e3218cdefd4c2cf3bb212e1fd66c51d24c80e"
    "https://p.scdn.co/mp3-preview/1c57741674dbcee27add7e06b6086d71da0cff5e"
    "https://audio-ssl.itunes.apple.com/itunes-assets/AudioPreview211/v4/37/d8/c9/37d8c9f7-838e-a222-0fe7-141c39e9f51c/mzaf_9144538674023163935.plus.aac.p.m4a"
    "https://p.scdn.co/mp3-preview/0b2426cba10cea8ffdd9d69b78f3f58073da6ab1?cid=18c612430a234d2da59d742217981da8"
    "https://audio-ssl.itunes.apple.com/itunes-assets/AudioPreview125/v4/6c/45/59/6c4559aa-e474-1366-66e7-9cd5279acd05/mzaf_11210908680488989277.plus.aac.p.m4a"
    "https://cdnt-preview.dzcdn.net/api/1/1/e/3/a/0/e3a23182a2b65997a9f4f40321865f6f.mp3?hdnea=exp=1769264366~acl=/api/1/1/e/3/a/0/e3a23182a2b65997a9f4f40321865f6f.mp3*~data=user_id=0,application_id=42~hmac=12caca3810169841acce9f630d58532ec86030b1d8e98573db1e698278bdc96f"
    "https://cdnt-preview.dzcdn.net/api/1/1/0/1/3/0/01362a91c97d085494ad64b63b9d88f4.mp3?hdnea=exp=1767892549~acl=/api/1/1/0/1/3/0/01362a91c97d085494ad64b63b9d88f4.mp3*~data=user_id=0,application_id=42~hmac=0597d1d65a6463081580e0109d2e3218cdefd4c2cf3bb212e1fd66c51d24c80e"
    "https://p.scdn.co/mp3-preview/1c57741674dbcee27add7e06b6086d71da0cff5e"
)

echo "=========================================="
echo "Concurrency Stress Test"
echo "=========================================="
echo "Service URL: $SERVICE_URL"
echo "Number of URLs: ${#TEST_URLS[@]}"
echo "Max Confidence: $MAX_CONFIDENCE"
echo "Debug Level: $DEBUG_LEVEL"
echo ""
echo "This test validates:"
echo "  - Worker concurrency: 16 requests"
echo "  - Essentia threads: 8 concurrent analyses"
echo "  - Fallback service: 2 concurrent + 3 process workers"
echo ""

TOKEN=$(gcloud auth print-identity-token 2>/dev/null)
if [ -z "$TOKEN" ]; then
    echo "Error: Failed to get authentication token"
    exit 1
fi

# Build JSON payload
JSON_PAYLOAD="{\"urls\": ["
for i in "${!TEST_URLS[@]}"; do
    if [ $i -gt 0 ]; then
        JSON_PAYLOAD+=", "
    fi
    JSON_PAYLOAD+="\"${TEST_URLS[$i]}\""
done
JSON_PAYLOAD+="], \"max_confidence\": $MAX_CONFIDENCE, \"debug_level\": \"$DEBUG_LEVEL\"}"

# Submit batch
echo "Submitting batch of ${#TEST_URLS[@]} URLs..."
START_TIME=$(date +%s)

RESPONSE=$(curl -s -X POST "${SERVICE_URL}/analyze/batch" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    -d "$JSON_PAYLOAD")

BATCH_ID=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['batch_id'])" 2>/dev/null)

if [ -z "$BATCH_ID" ]; then
    echo "❌ Error: Could not submit batch"
    echo "Response: $RESPONSE"
    exit 1
fi

echo "✅ Batch submitted: $BATCH_ID"
echo ""
echo "Streaming results..."
echo ""

# Track timing metrics
FIRST_PARTIAL_TIME=""
FIRST_FINAL_TIME=""
RESULTS_COUNT=0
PARTIALS_COUNT=0

curl -s -N --max-time 600 -X GET "${SERVICE_URL}/stream/${BATCH_ID}" \
    -H "Authorization: Bearer $TOKEN" | while IFS= read -r line; do
    if [ -z "$line" ]; then
        continue
    fi

    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))

    TYPE=$(echo "$line" | python3 -c "import sys, json; print(json.load(sys.stdin).get('type', 'unknown'))" 2>/dev/null)

    case "$TYPE" in
        "result")
            STATUS_FIELD=$(echo "$line" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'final'))" 2>/dev/null)
            INDEX=$(echo "$line" | python3 -c "import sys, json; print(json.load(sys.stdin).get('index', '?'))" 2>/dev/null)

            if [ "$STATUS_FIELD" = "partial" ]; then
                PARTIALS_COUNT=$((PARTIALS_COUNT + 1))
                if [ -z "$FIRST_PARTIAL_TIME" ]; then
                    FIRST_PARTIAL_TIME=$ELAPSED
                    echo "[${ELAPSED}s] ⚡ First partial result! (Time to first result: ${ELAPSED}s)"
                fi
                echo "[${ELAPSED}s] ⏳ Partial #${INDEX} (total partials: ${PARTIALS_COUNT})"
            else
                RESULTS_COUNT=$((RESULTS_COUNT + 1))
                if [ -z "$FIRST_FINAL_TIME" ]; then
                    FIRST_FINAL_TIME=$ELAPSED
                    echo "[${ELAPSED}s] 🎯 First final result! (Time to first final: ${ELAPSED}s)"
                fi
                echo "[${ELAPSED}s] ✅ Final #${INDEX} (completed: ${RESULTS_COUNT}/${#TEST_URLS[@]})"
            fi
            ;;
        "complete")
            TOTAL_ELAPSED=$((CURRENT_TIME - START_TIME))
            echo ""
            echo "=========================================="
            echo "✅ Test Complete!"
            echo "=========================================="
            echo "Total URLs: ${#TEST_URLS[@]}"
            echo "Time to first partial: ${FIRST_PARTIAL_TIME}s"
            echo "Time to first final: ${FIRST_FINAL_TIME}s"
            echo "Total processing time: ${TOTAL_ELAPSED}s"
            echo "Average per URL: $((TOTAL_ELAPSED / ${#TEST_URLS[@]}))s"
            echo "Throughput: $(echo "scale=2; ${#TEST_URLS[@]} / $TOTAL_ELAPSED" | bc) URLs/second"
            echo ""

            if [ "$TOTAL_ELAPSED" -lt 120 ]; then
                echo "🚀 Excellent! Completed in under 2 minutes"
            elif [ "$TOTAL_ELAPSED" -lt 180 ]; then
                echo "✅ Good! Completed in under 3 minutes"
            else
                echo "⚠️  Slower than expected (>3 minutes for 20 URLs)"
            fi
            exit 0
            ;;
        "error")
            ERROR=$(echo "$line" | python3 -c "import sys, json; print(json.load(sys.stdin).get('message', 'Unknown error'))" 2>/dev/null)
            echo "[${ELAPSED}s] ❌ Error: $ERROR"
            ;;
    esac
done

echo ""
echo "Test finished or timed out"
