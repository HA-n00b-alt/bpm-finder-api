"""
BPM Finder API - Google Cloud Run microservice
Batch processing: Computes BPM and key from multiple audio preview URLs.
"""
import os
import tempfile
import asyncio
import io
from typing import Optional, Tuple, List
from urllib.parse import urlparse
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl, field_validator
import essentia.standard as es

# Google Cloud authentication for service-to-service calls
try:
    import google.auth
    import google.auth.transport.requests
    import google.oauth2.id_token
    GCP_AUTH_AVAILABLE = True
except ImportError:
    GCP_AUTH_AVAILABLE = False

app = FastAPI(title="BPM Finder API")

# Download limits
CONNECT_TIMEOUT = 5.0  # seconds
TOTAL_TIMEOUT = 20.0  # seconds
MAX_SIZE = 10 * 1024 * 1024  # 10MB
MAX_AUDIO_DURATION = 35.0  # seconds - cap analysis to first 35s

# Fallback Service Configuration
FALLBACK_SERVICE_URL = "https://bpm-fallback-service-340051416180.europe-west3.run.app"
FALLBACK_SERVICE_AUDIENCE = FALLBACK_SERVICE_URL


class BatchBPMRequest(BaseModel):
    urls: List[HttpUrl]
    max_confidence: Optional[float] = 0.65

    @field_validator("urls")
    @classmethod
    def validate_urls(cls, v):
        """Validate URL schemes and non-empty list."""
        if not v:
            raise ValueError("urls list cannot be empty")
        for url in v:
            url_str = str(url)
            if not url_str.startswith("https://"):
                raise ValueError("Only HTTPS URLs are allowed")
        return v
    
    @field_validator("max_confidence")
    @classmethod
    def validate_max_confidence(cls, v):
        """Validate max_confidence is in valid range."""
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError("max_confidence must be between 0.0 and 1.0")
        return v


class BPMResponse(BaseModel):
    bpm: int
    bpm_raw: float
    bpm_confidence: float
    bpm_method: str
    debug_info: Optional[str] = None
    key: str
    scale: str
    key_confidence: float


def validate_redirect_url(url: str) -> bool:
    """Validate that a redirect URL uses HTTPS."""
    return url.startswith("https://")


async def download_audio_async(url: str, output_path: str) -> None:
    """Download audio file with SSRF protection and size limits using async streaming."""
    timeout = httpx.Timeout(
        connect=CONNECT_TIMEOUT,
        read=TOTAL_TIMEOUT,
        write=TOTAL_TIMEOUT,
        pool=CONNECT_TIMEOUT
    )
    
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=False,
        max_redirects=10,
    ) as client:
        current_url = url
        redirect_count = 0
        max_redirects = 10
        
        # Follow redirects manually, validating each one
        while redirect_count < max_redirects:
            response = await client.get(current_url)
            
            # If redirect, validate and follow
            if response.status_code in (301, 302, 303, 307, 308):
                redirect_url = response.headers.get("location")
                if not redirect_url:
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid redirect: missing Location header"
                    )
                
                # Resolve relative URLs
                if redirect_url.startswith("/"):
                    parsed = urlparse(current_url)
                    redirect_url = f"{parsed.scheme}://{parsed.netloc}{redirect_url}"
                elif not redirect_url.startswith("http"):
                    parsed = urlparse(current_url)
                    redirect_url = f"{parsed.scheme}://{parsed.netloc}/{redirect_url.lstrip('/')}"
                
                # Validate redirect URL
                if not validate_redirect_url(redirect_url):
                    raise HTTPException(
                        status_code=400,
                        detail="Redirect to non-HTTPS URL not allowed"
                    )
                
                current_url = redirect_url
                redirect_count += 1
                continue
            
            # If not a redirect, we have the final response
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Download failed: HTTP {response.status_code}"
                )
            
            # Check final URL uses HTTPS
            final_url = str(response.url)
            if not validate_redirect_url(final_url):
                raise HTTPException(
                    status_code=400,
                    detail="Final URL must use HTTPS"
                )
            
            # Check content length
            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > MAX_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"File too large (max {MAX_SIZE / 1024 / 1024:.1f}MB)"
                )
            
            # Stream download directly to file with size limit (non-blocking I/O)
            total_size = 0
            # Open file in binary write mode
            f = open(output_path, "wb")
            try:
                async for chunk in response.aiter_bytes():
                    total_size += len(chunk)
                    if total_size > MAX_SIZE:
                        f.close()
                        raise HTTPException(
                            status_code=400,
                            detail=f"File too large (max {MAX_SIZE / 1024 / 1024:.1f}MB)"
                        )
                    # Use asyncio.to_thread to make file write non-blocking
                    # This allows the event loop to switch to other pending requests
                    await asyncio.to_thread(f.write, chunk)
            finally:
                # Ensure file is closed even if an error occurs
                f.close()
            
            return
        
        # Too many redirects
        raise HTTPException(
            status_code=400,
            detail="Too many redirects"
        )


def normalize_confidence(confidence: float) -> Tuple[float, str]:
    """Normalize confidence value to 0-1 range and determine quality level.
    
    RhythmExtractor2013(method="multifeature") returns confidence in range 0-5.32.
    This function normalizes to 0-1 and categorizes quality.
    
    Quality guidelines:
    - [0, 1): very low confidence
    - [1, 2): low confidence
    - [2, 3): moderate confidence
    - [3, 3.5): high confidence
    - (3.5, 5.32]: excellent confidence
    
    Args:
        confidence: Raw confidence value from Essentia (0-5.32)
    
    Returns:
        Tuple of (normalized_confidence, quality_level) where:
        - normalized_confidence: Value in [0, 1] range
        - quality_level: Quality category string
    """
    # Clamp negative values to 0
    if confidence < 0:
        return 0.0, "very low"
    
    # Determine quality level based on raw confidence
    MAX_CONFIDENCE = 5.32
    if confidence < 1.0:
        quality = "very low"
    elif confidence < 2.0:
        quality = "low"
    elif confidence < 3.0:
        quality = "moderate"
    elif confidence < 3.5:
        quality = "high"
    else:
        quality = "excellent"
    
    # Normalize to 0-1 range
    normalized = min(1.0, confidence / MAX_CONFIDENCE)
    
    return float(normalized), quality


def normalize_bpm(bpm: float) -> float:
    """Normalize BPM by applying corrections only for extreme outliers.
    
    Args:
        bpm: The BPM value to normalize
    
    Returns:
        Normalized BPM value (float, rounded to 1 decimal place)
    """
    normalized = bpm
    
    # Apply corrections only for extreme outliers
    if normalized < 40:
        normalized *= 2
    elif normalized > 220:
        normalized /= 2
    
    # Round to 1 decimal place
    return round(normalized, 1)


def analyze_audio(audio_path: str, max_confidence: float) -> Tuple[
    Optional[float], Optional[float], Optional[float], Optional[str], Optional[str],
    Optional[str], Optional[str], Optional[float], Optional[float],
    bool, bool, str
]:
    """Analyze audio file: compute BPM and key from same loaded audio array.
    
    Args:
        audio_path: Path to the audio file (MP3/AAC - Essentia handles decoding)
        max_confidence: Confidence threshold for fallback decision
    
    Returns:
        Tuple of:
        - bpm_normalized, bpm_raw, bpm_confidence_normalized, bpm_quality, bpm_method
        - key, scale, key_strength_raw, key_confidence_normalized
        - need_fallback_bpm, need_fallback_key
        - debug_info
    """
    debug_lines = []
    
    try:
        # Load audio once - Essentia handles MP3/AAC decoding directly
        # Cap duration to MAX_AUDIO_DURATION seconds
        loader = es.MonoLoader(
            filename=audio_path,
            sampleRate=44100,
            endTime=MAX_AUDIO_DURATION
        )
        audio = loader()
        debug_lines.append(f"Audio loaded: {len(audio)/44100:.1f}s (capped at {MAX_AUDIO_DURATION}s)")
    except Exception as e:
        error_msg = f"Essentia audio loading error: {str(e)}"
        debug_lines.append(error_msg)
        return None, None, None, None, "error", "unknown", "unknown", 0.0, 0.0, True, True, "\n".join(debug_lines)
    
    # BPM extraction using multifeature method
    bpm_normalized = None
    bpm_raw = None
    bpm_confidence_normalized = None
    bpm_quality = None
    bpm_method = "multifeature"
    need_fallback_bpm = False
    
    try:
        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        bpm_raw, beats, confidence_raw, _, beats_intervals = rhythm_extractor(audio)
        
        # Normalize confidence and get quality level
        bpm_confidence_normalized, bpm_quality = normalize_confidence(float(confidence_raw))
        bpm_normalized = normalize_bpm(float(bpm_raw))
        
        debug_lines.append(f"BPM={bpm_raw:.2f} (normalized={bpm_normalized:.1f})")
        debug_lines.append(f"Confidence: raw={confidence_raw:.3f} (range: 0-5.32), normalized={bpm_confidence_normalized:.3f} (0-1), quality={bpm_quality}")
        
        # Check if BPM confidence meets threshold
        if bpm_confidence_normalized >= max_confidence:
            debug_lines.append(f"BPM confidence ({bpm_confidence_normalized:.3f}) >= threshold ({max_confidence:.2f}) - using Essentia result")
        else:
            need_fallback_bpm = True
            debug_lines.append(f"BPM confidence ({bpm_confidence_normalized:.3f}) < threshold ({max_confidence:.2f}) - fallback needed")
    except Exception as e:
        error_msg = f"BPM extraction error: {str(e)}"
        debug_lines.append(error_msg)
        need_fallback_bpm = True
    
    # Key extraction using same audio array
    key = "unknown"
    scale = "unknown"
    key_strength_raw = 0.0
    key_confidence_normalized = 0.0
    need_fallback_key = False
    
    try:
        # Try multiple key profile types
        key_profiles = ['temperley', 'krumhansl', 'edma', 'edmm']
        results = []
        
        for profile in key_profiles:
            try:
                key_extractor = es.KeyExtractor(
                    profileType=profile,
                    pcpSize=36,
                    numHarmonics=4
                )
                key_result, scale_result, strength = key_extractor(audio)
                results.append((str(key_result), str(scale_result), float(strength), profile))
                debug_lines.append(f"key_profile={profile}: key={key_result} {scale_result}, strength={strength:.3f}")
            except Exception as e:
                # Profile not available in this Essentia version, skip
                debug_lines.append(f"key_profile={profile}: error={str(e)}")
                continue
        
        # If we have results, return the one with highest strength
        if results:
            # Sort by strength (index 2) in descending order
            results.sort(key=lambda x: x[2], reverse=True)
            key, scale, key_strength_raw, winning_profile = results[0]
            
            # Normalize strength (assume KeyExtractor returns 0-1, but clamp if > 1)
            key_confidence_normalized = min(1.0, max(0.0, key_strength_raw))
            
            debug_lines.append(f"Winner: {winning_profile} profile (strength={key_strength_raw:.3f}, normalized={key_confidence_normalized:.3f})")
            if len(results) > 1:
                strength_range = results[0][2] - results[-1][2]
                debug_lines.append(f"Key ensemble: {len(results)} profiles, strength_range={strength_range:.3f}")
        else:
            # Fall back to default KeyExtractor if no profiles worked
            try:
                key_extractor = es.KeyExtractor()
                key, scale, key_strength_raw = key_extractor(audio)
                key_confidence_normalized = min(1.0, max(0.0, key_strength_raw))
                debug_lines.append(f"Fallback: default KeyExtractor, key={key} {scale}, strength={key_strength_raw:.3f} (normalized={key_confidence_normalized:.3f})")
            except Exception as e:
                error_msg = f"KeyExtractor fallback error: {str(e)}"
                debug_lines.append(error_msg)
                need_fallback_key = True
        
        # Check if key strength meets threshold
        if key_confidence_normalized >= max_confidence:
            debug_lines.append(f"Key strength ({key_confidence_normalized:.3f}) >= threshold ({max_confidence:.2f}) - using Essentia result")
        else:
            need_fallback_key = True
            debug_lines.append(f"Key strength ({key_confidence_normalized:.3f}) < threshold ({max_confidence:.2f}) - fallback needed")
    except Exception as e:
        error_msg = f"Key computation error: {str(e)}"
        debug_lines.append(error_msg)
        need_fallback_key = True
    
    return (
        bpm_normalized, bpm_raw, bpm_confidence_normalized, bpm_quality, bpm_method,
        key, scale, key_strength_raw, key_confidence_normalized,
        need_fallback_bpm, need_fallback_key,
        "\n".join(debug_lines)
    )


async def process_single_url(
    url: HttpUrl,
    max_confidence: float,
    index: int
) -> Tuple[int, BPMResponse]:
    """Process a single URL: download, analyze, return result.
    
    Returns:
        Tuple of (index, BPMResponse)
    """
    url_str = str(url)
    parsed = urlparse(url_str)
    input_ext = os.path.splitext(parsed.path)[1] or ".tmp"
    input_path = None
    debug_info_parts = []
    
    try:
        # Create temp file
        input_fd, input_path = tempfile.mkstemp(suffix=input_ext, dir="/tmp")
        os.close(input_fd)  # Close file descriptor, we'll write via path
        
        # Download audio (async streaming)
        try:
            await download_audio_async(url_str, input_path)
            debug_info_parts.append(f"URL fetch: SUCCESS ({url_str[:50]}...)")
        except Exception as e:
            error_msg = f"URL fetch error: {str(e)}"
            debug_info_parts.append(error_msg)
            return index, BPMResponse(
                bpm=0,
                bpm_raw=0.0,
                bpm_confidence=0.0,
                bpm_method="error",
                debug_info="\n".join(debug_info_parts),
                key="unknown",
                scale="unknown",
                key_confidence=0.0,
            )
        
        # Analyze audio (BPM + key from same loaded array)
        (
            bpm_normalized, bpm_raw, bpm_confidence_normalized, bpm_quality, bpm_method,
            key, scale, key_strength_raw, key_confidence_normalized,
            need_fallback_bpm, need_fallback_key,
            analysis_debug
        ) = analyze_audio(input_path, max_confidence)
        
        debug_info_parts.append(f"Max confidence threshold: {max_confidence:.2f}")
        debug_info_parts.append("=== Analysis (Essentia) ===")
        debug_info_parts.append(analysis_debug)
        
        # Store for potential fallback batch call
        # Return result with fallback flags
        return index, {
            "index": index,
            "url": url_str,
            "file_path": input_path,  # Keep file for batch fallback if needed
            "bpm_normalized": bpm_normalized,
            "bpm_raw": bpm_raw,
            "bpm_confidence_normalized": bpm_confidence_normalized,
            "bpm_method": bpm_method,
            "key": key,
            "scale": scale,
            "key_confidence_normalized": key_confidence_normalized,
            "need_fallback_bpm": need_fallback_bpm,
            "need_fallback_key": need_fallback_key,
            "debug_info_parts": debug_info_parts,
        }
    
    except Exception as e:
        error_msg = f"Processing error: {str(e)}"
        debug_info_parts.append(error_msg)
        return index, {
            "index": index,
            "url": url_str,
            "file_path": None,
            "bpm_normalized": 0.0,
            "bpm_raw": 0.0,
            "bpm_confidence_normalized": 0.0,
            "bpm_method": "error",
            "key": "unknown",
            "scale": "unknown",
            "key_confidence_normalized": 0.0,
            "need_fallback_bpm": True,
            "need_fallback_key": True,
            "debug_info_parts": debug_info_parts,
        }
    
    finally:
        # Note: Don't delete input_path here - we may need it for fallback batch
        # Cleanup will happen after fallback processing
        pass


async def get_auth_headers(audience: str) -> dict:
    """Generates an OIDC token for the target audience for internal Cloud Run calls."""
    if not GCP_AUTH_AVAILABLE:
        return {}
    try:
        credentials, _ = google.auth.default()
        auth_request = google.auth.transport.requests.Request()
        token = google.oauth2.id_token.fetch_id_token(auth_request, audience)
        return {"Authorization": f"Bearer {token}"}
    except Exception as e:
        # If running locally or without proper service account, log error
        print(f"Error generating auth token: {e}")
        return {}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"ok": True}


@app.post("/analyze/batch", response_model=List[BPMResponse])
async def analyze_batch(request: BatchBPMRequest):
    """Batch process multiple audio URLs: compute BPM and key for each.
    
    Processes URLs concurrently, then sends a single batch request to fallback
    service for items that need it.
    """
    max_confidence = request.max_confidence if request.max_confidence is not None else 0.65
    
    # Process all URLs concurrently
    tasks = [
        process_single_url(url, max_confidence, i)
        for i, url in enumerate(request.urls)
    ]
    results = await asyncio.gather(*tasks)
    
    # Sort by index to maintain order
    results.sort(key=lambda x: x[0])
    processed_items = [item for _, item in results]
    
    # Collect items that need fallback
    fallback_items = [
        item for item in processed_items
        if item["need_fallback_bpm"] or item["need_fallback_key"]
    ]
    
    # Single batch request to fallback service if needed
    if fallback_items:
        try:
            auth_headers = await get_auth_headers(FALLBACK_SERVICE_AUDIENCE)
            
            # Prepare multipart files for fallback (streaming, not reading into RAM)
            files = []
            data = {}
            file_handles = []  # Keep track for cleanup
            
            for i, item in enumerate(fallback_items):
                if item["file_path"] and os.path.exists(item["file_path"]):
                    # Open file for streaming upload
                    file_handle = open(item["file_path"], "rb")
                    file_handles.append(file_handle)
                    files.append(
                        ("audio_files", (f"audio_{i}.tmp", file_handle, "audio/mpeg"))
                    )
                    data[f"process_bpm_{i}"] = str(item["need_fallback_bpm"]).lower()
                    data[f"process_key_{i}"] = str(item["need_fallback_key"]).lower()
                    data[f"url_{i}"] = item["url"]
            
            # Make batch request to fallback service
            if files:  # Only make request if we have files to send
                try:
                    async with httpx.AsyncClient(timeout=120.0) as client:
                        response = await client.post(
                            f"{FALLBACK_SERVICE_URL}/process_batch",
                            files=files,
                            data=data,
                            headers=auth_headers
                        )
                        
                        # Close file handles
                        for fh in file_handles:
                            try:
                                fh.close()
                            except Exception:
                                pass
                        
                        if response.status_code == 200:
                            fallback_results = response.json()
                            
                            # Update processed items with fallback results
                            for i, fallback_result in enumerate(fallback_results):
                                if i < len(fallback_items):
                                    item = fallback_items[i]
                                    item_index = item["index"]
                                    
                                    # Update BPM if fallback was needed and returned
                                    if item["need_fallback_bpm"] and fallback_result.get("bpm_normalized") is not None:
                                        processed_items[item_index]["bpm_normalized"] = fallback_result["bpm_normalized"]
                                        processed_items[item_index]["bpm_raw"] = fallback_result["bpm_raw"]
                                        processed_items[item_index]["bpm_confidence_normalized"] = fallback_result["confidence"]
                                        processed_items[item_index]["bpm_method"] = "librosa_hpss_fallback"
                                        processed_items[item_index]["debug_info_parts"].append(
                                            f"Fallback BPM: {fallback_result['bpm_normalized']:.1f} (confidence={fallback_result['confidence']:.3f})"
                                        )
                                    
                                    # Update key if fallback was needed and returned
                                    if item["need_fallback_key"] and fallback_result.get("key") is not None:
                                        processed_items[item_index]["key"] = fallback_result["key"]
                                        processed_items[item_index]["scale"] = fallback_result["scale"]
                                        processed_items[item_index]["key_confidence_normalized"] = fallback_result["key_confidence"]
                                        processed_items[item_index]["debug_info_parts"].append(
                                            f"Fallback key: {fallback_result['key']} {fallback_result['scale']} (confidence={fallback_result['key_confidence']:.3f})"
                                        )
                except Exception as e:
                    # Log error but continue with Essentia results
                    for item in fallback_items:
                        processed_items[item["index"]]["debug_info_parts"].append(
                            f"Fallback service error: {str(e)[:200]}"
                        )
                finally:
                    # Ensure file handles are closed even if there's an error
                    for fh in file_handles:
                        try:
                            fh.close()
                        except Exception:
                            pass
        except Exception as e:
            # Log error but continue with Essentia results
            for item in fallback_items:
                processed_items[item["index"]]["debug_info_parts"].append(
                    f"Fallback preparation error: {str(e)[:200]}"
                )
    
    # Build final responses and cleanup
    final_responses = []
    for item in processed_items:
        # Ensure valid values
        bpm_normalized = item["bpm_normalized"] if item["bpm_normalized"] is not None else 0.0
        bpm_raw = item["bpm_raw"] if item["bpm_raw"] is not None else 0.0
        bpm_confidence = item["bpm_confidence_normalized"] if item["bpm_confidence_normalized"] is not None else 0.0
        
        final_responses.append(BPMResponse(
            bpm=int(round(bpm_normalized)),
            bpm_raw=round(bpm_raw, 2),
            bpm_confidence=round(bpm_confidence, 2),
            bpm_method=item["bpm_method"],
            debug_info="\n".join(item["debug_info_parts"]),
            key=item["key"],
            scale=item["scale"],
            key_confidence=round(item["key_confidence_normalized"], 2),
        ))
        
        # Cleanup temp file
        if item["file_path"] and os.path.exists(item["file_path"]):
            try:
                os.unlink(item["file_path"])
            except Exception:
                pass
    
    return final_responses
