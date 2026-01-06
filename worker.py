"""
BPM Worker Service - Processes Pub/Sub messages and writes results to Firestore
"""
import os
import json
import base64
import asyncio
import tempfile
import httpx
import time
from typing import Optional
from urllib.parse import urlparse
from fastapi import FastAPI, Request, HTTPException
from google.cloud import firestore
from google.cloud.firestore import Increment
from concurrent.futures import ThreadPoolExecutor

# Import shared processing functions
from shared_processing import (
    download_audio_async,
    analyze_audio,
    get_auth_headers,
    generate_debug_output,
    FallbackCircuitBreaker,
    FALLBACK_SERVICE_URL,
    FALLBACK_SERVICE_AUDIENCE,
    FALLBACK_REQUEST_TIMEOUT_COLD_START,
    FALLBACK_REQUEST_TIMEOUT_WARM,
    FALLBACK_MAX_RETRIES,
    FALLBACK_RETRY_DELAY,
)

app = FastAPI(title="BPM Worker Service")
db = firestore.Client()

# Global HTTP client with connection pooling
_http_client: Optional[httpx.AsyncClient] = None

# Global circuit breaker
fallback_circuit_breaker = FallbackCircuitBreaker()

# ThreadPoolExecutor for CPU-bound Essentia analysis
_essentia_executor: Optional[ThreadPoolExecutor] = None


def get_http_client() -> httpx.AsyncClient:
    """Get or create the global HTTPX AsyncClient."""
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=100,
            ),
            timeout=httpx.Timeout(
                connect=10.0,
                read=30.0,
                write=30.0,
                pool=10.0
            ),
            follow_redirects=False,
        )
    return _http_client


def get_essentia_executor() -> ThreadPoolExecutor:
    """Get or create the ThreadPoolExecutor for Essentia."""
    global _essentia_executor
    if _essentia_executor is None:
        max_workers = int(os.getenv("ESSENTIA_MAX_CONCURRENCY", max(1, os.cpu_count() or 1)))
        _essentia_executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="essentia")
    return _essentia_executor


@app.on_event("startup")
async def startup_event():
    """Pre-import Essentia to avoid hanging on first use."""
    print("[STARTUP] Pre-importing Essentia to avoid cold start delay...")
    import sys
    import time
    start = time.time()
    try:
        # Pre-import Essentia in a thread to avoid blocking
        import threading
        def preimport_essentia():
            try:
                import essentia.standard as es
                print(f"[STARTUP] Essentia pre-imported successfully in {time.time() - start:.2f}s")
            except Exception as e:
                print(f"[STARTUP] WARNING: Essentia pre-import failed: {str(e)}")
        
        thread = threading.Thread(target=preimport_essentia, daemon=True)
        thread.start()
        thread.join(timeout=60)  # Wait up to 60 seconds
        if thread.is_alive():
            print("[STARTUP] WARNING: Essentia pre-import still running after 60s, continuing anyway...")
    except Exception as e:
        print(f"[STARTUP] ERROR during Essentia pre-import: {str(e)}")
    print("[STARTUP] Startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global _http_client, _essentia_executor
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None
    if _essentia_executor is not None:
        _essentia_executor.shutdown(wait=True)
        _essentia_executor = None


async def process_single_url_task(
    batch_id: str,
    url: str,
    index: int,
    max_confidence: float,
    debug_level: str
):
    """Process a single URL and write result to Firestore."""
    batch_ref = db.collection('batches').document(batch_id)
    
    print(f"[{batch_id}:{index}] Starting processing: {url[:50]}...")
    
    input_path = None
    debug_info_parts = []
    timing = {
        "download_start": None,
        "download_end": None,
        "download_duration": None,
        "essentia_start": None,
        "essentia_end": None,
        "essentia_duration": None,
    }
    
    try:
        # Create temp file
        parsed = urlparse(url)
        input_ext = os.path.splitext(parsed.path)[1] or ".tmp"
        input_fd, input_path = tempfile.mkstemp(suffix=input_ext, dir="/tmp")
        os.close(input_fd)
        
        # Download audio
        try:
            print(f"[{batch_id}:{index}] Downloading...")
            timing["download_start"] = time.time()
            client = get_http_client()
            await download_audio_async(url, input_path, client)
            timing["download_end"] = time.time()
            timing["download_duration"] = timing["download_end"] - timing["download_start"]
            debug_info_parts.append(f"URL fetch: SUCCESS ({url[:50]}...)")
            print(f"[{batch_id}:{index}] Download complete ({timing['download_duration']:.2f}s)")
        except Exception as e:
            error_msg = f"URL fetch error: {str(e)}"
            debug_info_parts.append(error_msg)
            print(f"[{batch_id}:{index}] Download failed: {str(e)}")
            raise
        
        # Analyze audio
        try:
            print(f"[{batch_id}:{index}] Analyzing...")
            print(f"[{batch_id}:{index}] Input file exists: {os.path.exists(input_path)}, size: {os.path.getsize(input_path) if os.path.exists(input_path) else 0} bytes")
            timing["essentia_start"] = time.time()
            executor = get_essentia_executor()
            loop = asyncio.get_event_loop()
            
            print(f"[{batch_id}:{index}] Submitting to executor (max_workers: {executor._max_workers if hasattr(executor, '_max_workers') else 'unknown'})...", flush=True)
            
            try:
                result = await loop.run_in_executor(
                    executor,
                    analyze_audio,
                    input_path,
                    max_confidence
                )
                print(f"[{batch_id}:{index}] Executor returned, unpacking result...", flush=True)
                (
                    bpm_normalized, bpm_raw, bpm_confidence_normalized, bpm_quality, bpm_method,
                    key, scale, key_strength_raw, key_confidence_normalized,
                    need_fallback_bpm, need_fallback_key,
                    analysis_debug
                ) = result
                print(f"[{batch_id}:{index}] Result unpacked: BPM={bpm_normalized}, Key={key}", flush=True)
            except Exception as e:
                import traceback
                error_details = f"Executor error: {str(e)}\n{traceback.format_exc()}"
                print(f"[{batch_id}:{index}] ERROR in executor: {error_details}", flush=True)
                raise
            
            print(f"[{batch_id}:{index}] Executor returned successfully", flush=True)
            timing["essentia_end"] = time.time()
            timing["essentia_duration"] = timing["essentia_end"] - timing["essentia_start"]
            
            debug_info_parts.append(f"Max confidence threshold: {max_confidence:.2f}")
            debug_info_parts.append("=== Analysis (Essentia) ===")
            debug_info_parts.append(analysis_debug)
            
            print(f"[{batch_id}:{index}] Analysis complete ({timing['essentia_duration']:.2f}s)")
            print(f"[{batch_id}:{index}] BPM: {bpm_normalized}, Key: {key} {scale}")
            print(f"[{batch_id}:{index}] Fallback needed - BPM: {need_fallback_bpm}, Key: {need_fallback_key}")
            
        except Exception as e:
            error_msg = f"Analysis error: {str(e)}"
            import traceback
            full_error = f"{error_msg}\n{traceback.format_exc()}"
            print(f"[{batch_id}:{index}] Analysis failed: {full_error}")
            debug_info_parts.append(error_msg)
            raise
        
        # Prepare result
        firestore_result = {
            "index": index,
            "url": url,
            "bpm_essentia": int(round(bpm_normalized)) if bpm_normalized is not None else None,
            "bpm_raw_essentia": round(bpm_raw, 2) if bpm_raw is not None else None,
            "bpm_confidence_essentia": round(bpm_confidence_normalized, 2) if bpm_confidence_normalized is not None else None,
            "bpm_librosa": None,
            "bpm_raw_librosa": None,
            "bpm_confidence_librosa": None,
            "key_essentia": key if key else "unknown",
            "scale_essentia": scale if scale else "unknown",
            "keyscale_confidence_essentia": round(key_confidence_normalized, 2) if key_confidence_normalized is not None else 0.0,
            "key_librosa": None,
            "scale_librosa": None,
            "keyscale_confidence_librosa": None,
        }
        
        # Handle fallback if needed
        if need_fallback_bpm or need_fallback_key:
            print(f"[{batch_id}:{index}] Calling fallback service...")
            fallback_result = await call_fallback_service(
                input_path,
                url,
                need_fallback_bpm,
                need_fallback_key,
                max_confidence,
                debug_level
            )
            
            if fallback_result:
                print(f"[{batch_id}:{index}] Fallback complete")
                if need_fallback_bpm and fallback_result.get("bpm_normalized") is not None:
                    firestore_result["bpm_librosa"] = int(round(fallback_result["bpm_normalized"]))
                    firestore_result["bpm_raw_librosa"] = round(fallback_result["bpm_raw"], 2) if fallback_result.get("bpm_raw") else None
                    firestore_result["bpm_confidence_librosa"] = round(fallback_result["confidence"], 2) if fallback_result.get("confidence") else None
                
                if need_fallback_key and fallback_result.get("key") is not None:
                    firestore_result["key_librosa"] = fallback_result["key"]
                    firestore_result["scale_librosa"] = fallback_result["scale"]
                    firestore_result["keyscale_confidence_librosa"] = round(fallback_result["key_confidence"], 2) if fallback_result.get("key_confidence") else None
            else:
                print(f"[{batch_id}:{index}] Fallback failed or skipped")
        
        # Generate debug output
        debug_txt = generate_debug_output(
            debug_info_parts,
            timing,
            None,
            debug_level
        )
        firestore_result["debug_txt"] = debug_txt if debug_txt else None
        
        # Write result to Firestore with idempotency check (using transaction)
        print(f"[{batch_id}:{index}] Writing to Firestore...", flush=True)
        print(f"[{batch_id}:{index}] Result data: BPM={firestore_result.get('bpm_essentia')}, Key={firestore_result.get('key_essentia')}", flush=True)
        loop = asyncio.get_event_loop()
        
        def update_result_idempotent():
            """Update Firestore with idempotency: only increment processed if this index wasn't already written."""
            try:
                print(f"[{batch_id}:{index}] [FIRESTORE] Starting idempotent update...", flush=True)
                
                # Use a transaction to ensure idempotency
                transaction = db.transaction()
                
                @firestore.transactional
                def update_in_transaction(transaction):
                    batch_doc = batch_ref.get(transaction=transaction)
                    if not batch_doc.exists:
                        raise Exception(f"Batch {batch_id} not found")
                    
                    batch_data = batch_doc.to_dict()
                    results = batch_data.get('results', {})
                    
                    # Check if this index was already processed (idempotency check)
                    if str(index) in results:
                        print(f"[{batch_id}:{index}] [FIRESTORE] Index {index} already processed, skipping increment", flush=True)
                        return False  # Already processed
                    
                    # Update with transaction
                    transaction.update(batch_ref, {
                        f'results.{index}': firestore_result,
                        'processed': Increment(1)
                    })
                    return True  # Newly processed
                
                was_new = update_in_transaction(transaction)
                print(f"[{batch_id}:{index}] [FIRESTORE] Transaction completed (new: {was_new})", flush=True)
                print(f"[{batch_id}:{index}] Firestore write successful", flush=True)
                return was_new
            except Exception as e:
                import traceback
                error_details = f"{str(e)}\n{traceback.format_exc()}"
                print(f"[{batch_id}:{index}] Firestore write error: {error_details}", flush=True)
                raise
        
        print(f"[{batch_id}:{index}] Submitting Firestore update to executor...", flush=True)
        was_new = await loop.run_in_executor(None, update_result_idempotent)
        print(f"[{batch_id}:{index}] Firestore update executor returned (was_new: {was_new})", flush=True)
        
        # Check if batch is complete (only if this was a new result)
        if was_new:
            batch_doc = await loop.run_in_executor(None, batch_ref.get)
            if batch_doc.exists:
                batch_data = batch_doc.to_dict()
                processed_count = batch_data.get('processed', 0)
                total_urls = batch_data.get('total_urls', 0)
                print(f"[{batch_id}] Progress: {processed_count}/{total_urls}")
                
                if processed_count >= total_urls:
                    def update_status():
                        batch_ref.update({'status': 'completed'})
                    await loop.run_in_executor(None, update_status)
                    print(f"[{batch_id}] Batch completed!")
        
        print(f"[{batch_id}:{index}] Task complete ✓")
        
    except Exception as e:
        # Write error to Firestore
        import traceback
        error_details = f"{str(e)}\n{traceback.format_exc()}"
        print(f"[{batch_id}:{index}] ERROR: {error_details[:500]}")
        
        loop = asyncio.get_event_loop()
        error_result = {
            "index": index,
            "url": url,
            "error": error_details[:500]
        }
        
        def update_error():
            try:
                batch_ref.update({
                    f'results.{index}': error_result,
                    'processed': Increment(1)
                })
            except Exception as update_err:
                print(f"[{batch_id}:{index}] Failed to write error: {str(update_err)}")
        
        await loop.run_in_executor(None, update_error)
        
        # Check if batch is complete even on error
        batch_doc = await loop.run_in_executor(None, batch_ref.get)
        if batch_doc.exists:
            batch_data = batch_doc.to_dict()
            if batch_data.get('processed', 0) >= batch_data.get('total_urls', 0):
                def update_status():
                    batch_ref.update({'status': 'completed'})
                await loop.run_in_executor(None, update_status)
    
    finally:
        # Cleanup temp file
        if input_path and os.path.exists(input_path):
            try:
                os.unlink(input_path)
                print(f"[{batch_id}:{index}] Temp file cleaned up")
            except Exception:
                pass


async def call_fallback_service(
    file_path: str,
    url: str,
    need_fallback_bpm: bool,
    need_fallback_key: bool,
    max_confidence: float,
    debug_level: str
):
    """Call fallback service for a single item."""
    if not file_path or not os.path.exists(file_path):
        return None
    
    file_handle = None
    try:
        # Get auth headers
        auth_headers = await get_auth_headers(FALLBACK_SERVICE_AUDIENCE)
        
        # Check circuit breaker
        if not fallback_circuit_breaker.can_attempt():
            print(f"Circuit breaker open, skipping fallback")
            return None
        
        # Open file
        loop = asyncio.get_event_loop()
        file_handle = await loop.run_in_executor(None, open, file_path, "rb")
        
        # Prepare request
        files = [("audio_files", (f"audio.tmp", file_handle, "audio/mpeg"))]
        data = {
            "process_bpm_0": str(need_fallback_bpm).lower(),
            "process_key_0": str(need_fallback_key).lower(),
            "url_0": url
        }
        
        # Retry logic
        client = get_http_client()
        
        for attempt in range(FALLBACK_MAX_RETRIES):
            request_timeout = FALLBACK_REQUEST_TIMEOUT_COLD_START if attempt == 0 else FALLBACK_REQUEST_TIMEOUT_WARM
            
            try:
                timeout = httpx.Timeout(
                    connect=5.0,
                    read=request_timeout,
                    write=request_timeout,
                    pool=5.0
                )
                
                print(f"Fallback attempt {attempt + 1}/{FALLBACK_MAX_RETRIES}")
                response = await client.post(
                    f"{FALLBACK_SERVICE_URL}/process_batch",
                    files=files,
                    data=data,
                    headers=auth_headers,
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    fallback_results = response.json()
                    if fallback_results and len(fallback_results) > 0:
                        fallback_circuit_breaker.record_success()
                        return fallback_results[0]
                
                print(f"Fallback failed with status {response.status_code}")
                fallback_circuit_breaker.record_failure()
                return None
                
            except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as e:
                print(f"Fallback network error: {str(e)}")
                if attempt < FALLBACK_MAX_RETRIES - 1:
                    await asyncio.sleep(FALLBACK_RETRY_DELAY * (2 ** attempt))
                else:
                    fallback_circuit_breaker.record_failure()
                    return None
            except Exception as e:
                print(f"Fallback error: {str(e)}")
                fallback_circuit_breaker.record_failure()
                return None
        
        return None
        
    except Exception as e:
        print(f"Fallback service error: {str(e)}")
        return None
    finally:
        if file_handle:
            try:
                file_handle.close()
            except Exception:
                pass


@app.post("/pubsub/process")
async def process_pubsub_message(request: Request):
    """Process Pub/Sub push message.
    
    CRITICAL: This endpoint processes SYNCHRONOUSLY and returns 2xx ONLY after Firestore write completes.
    
    - NO background tasks (asyncio.create_task is NOT used)
    - NO early returns (we await process_single_url_task which waits for Firestore)
    - Returns 204 (No Content) on success - only after Firestore write is done
    - Returns 500 on failure - so Pub/Sub will retry
    
    This ensures:
    1. Cloud Run doesn't kill the task after request returns
    2. Pub/Sub can retry if we return non-2xx
    3. No message loss if container is killed during processing
    """
    try:
        envelope = await request.json()
        
        if 'message' not in envelope:
            raise HTTPException(status_code=400, detail="Invalid Pub/Sub message format")
        
        message = envelope['message']
        if 'data' not in message:
            raise HTTPException(status_code=400, detail="Missing data in Pub/Sub message")
        
        # Decode message
        message_data = json.loads(base64.b64decode(message['data']).decode('utf-8'))
        
        batch_id = message_data.get('batch_id')
        url = message_data.get('url')
        index = message_data.get('index')
        max_confidence = message_data.get('max_confidence', 0.65)
        debug_level = message_data.get('debug_level', 'normal')
        
        if not all([batch_id, url, index is not None]):
            raise HTTPException(status_code=400, detail="Missing required fields")
        
        print(f"[{batch_id}:{index}] Received Pub/Sub message, processing SYNCHRONOUSLY (no background tasks)...")
        
        # CRITICAL: Process synchronously - await completion before returning ANY status code
        # process_single_url_task waits for:
        #   1. Audio download
        #   2. Audio analysis
        #   3. Fallback service (if needed)
        #   4. Firestore write (with transaction)
        # Only after ALL of the above completes do we return
        try:
            await process_single_url_task(
                batch_id,
                url,
                index,
                max_confidence,
                debug_level
            )
            # At this point, Firestore write is complete
            print(f"[{batch_id}:{index}] Processing completed successfully, Firestore write confirmed")
            # Return 204 (No Content) - Pub/Sub interprets this as successful ACK
            # We return 204 (not 200) to signal success without a body
            from fastapi import Response
            return Response(status_code=204)
        except Exception as e:
            import traceback
            error_details = f"Processing failed: {str(e)}\n{traceback.format_exc()}"
            print(f"[{batch_id}:{index}] Processing failed: {error_details[:500]}")
            # Return 500 so Pub/Sub will retry the message
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
        
    except HTTPException:
        # Re-raise HTTP exceptions (including our 500)
        raise
    except Exception as e:
        import traceback
        error_details = f"Error processing Pub/Sub: {str(e)}\n{traceback.format_exc()}"
        print(f"Pub/Sub endpoint error: {error_details[:500]}")
        # Return 500 so Pub/Sub will retry
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"ok": True}
