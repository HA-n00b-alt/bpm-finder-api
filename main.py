"""
BPM Finder API - Google Cloud Run microservice
Batch processing: Computes BPM and key from multiple audio preview URLs.
"""
import os
import tempfile
import asyncio
import io
import time
import uuid
import json
from typing import Optional, Tuple, List
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
import httpx
import aiofiles
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, HttpUrl, field_validator
from google.cloud import pubsub_v1, firestore
# Lazy load essentia - heavy C++ library that slows startup
# Will be imported in analyze_audio() when actually needed

# Google Cloud authentication for service-to-service calls
GCP_AUTH_AVAILABLE = False
GCP_AUTH_ERROR = None
try:
    import google.auth
    import google.auth.transport.requests
    import google.oauth2.id_token
    GCP_AUTH_AVAILABLE = True
except ImportError as e:
    # Capture the actual import error for debugging
    import sys
    import traceback
    GCP_AUTH_ERROR = f"ImportError: {str(e)}\nPython path: {sys.path}\nTraceback: {traceback.format_exc()}"
    print(f"Warning: Failed to import google-auth: {GCP_AUTH_ERROR}", file=sys.stderr)
except Exception as e:
    # Catch any other errors during import
    import sys
    import traceback
    GCP_AUTH_ERROR = f"Unexpected error importing google-auth: {str(e)}\nTraceback: {traceback.format_exc()}"
    print(f"Warning: Unexpected error importing google-auth: {GCP_AUTH_ERROR}", file=sys.stderr)

app = FastAPI(title="BPM Finder API")

# Google Cloud configuration
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT") or os.getenv("PROJECT_ID", "bpm-api-microservice")
PUBSUB_TOPIC = f"projects/{PROJECT_ID}/topics/bpm-analysis-tasks"

# Initialize Pub/Sub and Firestore clients
publisher = pubsub_v1.PublisherClient()
db = firestore.Client()

# Download limits
CONNECT_TIMEOUT = 5.0  # seconds
TOTAL_TIMEOUT = 20.0  # seconds
MAX_SIZE = 10 * 1024 * 1024  # 10MB
MAX_AUDIO_DURATION = 35.0  # seconds - cap analysis to first 35s

# Fallback Service Configuration
FALLBACK_SERVICE_URL = "https://bpm-fallback-service-340051416180.europe-west3.run.app"
FALLBACK_SERVICE_AUDIENCE = FALLBACK_SERVICE_URL
FALLBACK_TIMEOUT = 120.0  # seconds - total timeout for fallback request
FALLBACK_REQUEST_TIMEOUT_COLD_START = 120.0  # seconds - timeout for first attempt (handles cold starts ~95s)
FALLBACK_REQUEST_TIMEOUT_WARM = 60.0  # seconds - timeout for retries (warm instances are faster)
FALLBACK_MAX_RETRIES = 3  # Maximum retry attempts for transient failures
FALLBACK_RETRY_DELAY = 2.0  # seconds - delay between retries (increased for cold start recovery)

# Circuit breaker state for fallback service
class FallbackCircuitBreaker:
    """Simple circuit breaker for fallback service calls."""
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
    
    def record_success(self):
        """Record a successful call."""
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self):
        """Record a failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
    
    def can_attempt(self) -> bool:
        """Check if we can attempt a call."""
        if self.state == "closed":
            return True
        elif self.state == "open":
            # Check if recovery timeout has passed
            if self.last_failure_time and (time.time() - self.last_failure_time) >= self.recovery_timeout:
                self.state = "half_open"
                return True
            return False
        else:  # half_open
            return True

# Global circuit breaker instance
fallback_circuit_breaker = FallbackCircuitBreaker()

# Global HTTPX client with connection pooling for reuse across requests
_http_client: Optional[httpx.AsyncClient] = None

# ThreadPoolExecutor for CPU-bound Essentia analysis (prevents event loop blocking)
_essentia_executor: Optional[ThreadPoolExecutor] = None
_essentia_semaphore: Optional[asyncio.Semaphore] = None

def get_essentia_executor() -> ThreadPoolExecutor:
    """Get or create the global ThreadPoolExecutor for Essentia analysis."""
    global _essentia_executor
    if _essentia_executor is None:
        # Use env var or default to CPU count
        max_workers = int(os.getenv("ESSENTIA_MAX_CONCURRENCY", max(1, os.cpu_count() or 1)))
        _essentia_executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="essentia")
    return _essentia_executor

def get_essentia_semaphore() -> asyncio.Semaphore:
    """Get or create the global semaphore for Essentia analysis concurrency control."""
    global _essentia_semaphore
    if _essentia_semaphore is None:
        # Use same limit as executor
        max_concurrent = int(os.getenv("ESSENTIA_MAX_CONCURRENCY", max(1, os.cpu_count() or 1)))
        _essentia_semaphore = asyncio.Semaphore(max_concurrent)
    return _essentia_semaphore

# Per-request URL concurrency limit for batch processing
BATCH_URL_CONCURRENCY = int(os.getenv("BATCH_URL_CONCURRENCY", "20"))

def get_http_client() -> httpx.AsyncClient:
    """Get or create the global HTTPX AsyncClient with connection pooling."""
    global _http_client
    if _http_client is None:
        # Create client with connection pooling enabled
        # limits: max_keepalive_connections=20, max_connections=100
        _http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=20,  # Keep 20 connections alive for reuse
                max_connections=100,           # Maximum 100 total connections
            ),
            timeout=httpx.Timeout(
                connect=10.0,  # Default connect timeout
                read=30.0,     # Default read timeout (can be overridden per request)
                write=30.0,    # Default write timeout
                pool=10.0      # Pool timeout
            ),
            follow_redirects=False,  # We handle redirects manually for security
        )
    return _http_client

@app.on_event("startup")
async def startup_event():
    """Initialize HTTP client and check fallback service health at startup."""
    # Pre-initialize the HTTP client with connection pooling
    get_http_client()
    
    # Check fallback service health
    try:
        # Try to get auth headers (this will fail if auth is misconfigured)
        try:
            auth_headers = await get_auth_headers(FALLBACK_SERVICE_AUDIENCE)
        except Exception as e:
            print(f"Warning: Fallback service auth check failed at startup: {str(e)[:200]}")
            print("Fallback service calls will be skipped until auth is configured.")
            return
        
        # Try to reach the fallback service health endpoint using global client
        client = get_http_client()
        timeout = httpx.Timeout(connect=5.0, read=10.0)
        try:
            response = await client.get(
                f"{FALLBACK_SERVICE_URL}/health",
                headers=auth_headers,
                timeout=timeout
            )
            if response.status_code == 200:
                print("✅ Fallback service is reachable and healthy")
                fallback_circuit_breaker.record_success()
            else:
                print(f"⚠️  Fallback service returned HTTP {response.status_code} at startup")
                fallback_circuit_breaker.record_failure()
        except httpx.TimeoutException:
            print("⚠️  Fallback service health check timed out at startup")
            fallback_circuit_breaker.record_failure()
        except Exception as e:
            print(f"⚠️  Fallback service health check failed: {str(e)[:200]}")
            fallback_circuit_breaker.record_failure()
    except Exception as e:
        print(f"Warning: Startup fallback service check failed: {str(e)[:200]}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup HTTP client and executor on shutdown."""
    global _http_client, _essentia_executor
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None
    if _essentia_executor is not None:
        _essentia_executor.shutdown(wait=True)
        _essentia_executor = None


class BatchBPMRequest(BaseModel):
    urls: List[HttpUrl]
    max_confidence: Optional[float] = 0.65
    debug_level: Optional[str] = "normal"  # "minimal", "normal", "detailed"

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
    
    @field_validator("debug_level")
    @classmethod
    def validate_debug_level(cls, v):
        """Validate debug_level is a valid option."""
        if v not in ["minimal", "normal", "detailed"]:
            raise ValueError("debug_level must be one of: minimal, normal, detailed")
        return v


class BPMResponse(BaseModel):
    # Essentia BPM results
    bpm_essentia: Optional[int] = None
    bpm_raw_essentia: Optional[float] = None
    bpm_confidence_essentia: Optional[float] = None
    
    # Librosa BPM results (null if not used)
    bpm_librosa: Optional[int] = None
    bpm_raw_librosa: Optional[float] = None
    bpm_confidence_librosa: Optional[float] = None
    
    # Essentia key results
    key_essentia: Optional[str] = None
    scale_essentia: Optional[str] = None
    keyscale_confidence_essentia: Optional[float] = None
    
    # Librosa key results (null if not used)
    key_librosa: Optional[str] = None
    scale_librosa: Optional[str] = None
    keyscale_confidence_librosa: Optional[float] = None
    
    # Debug information
    debug_txt: Optional[str] = None


class BatchSubmissionResponse(BaseModel):
    batch_id: str
    total_urls: int
    status: str
    stream_url: str


class BatchStatusResponse(BaseModel):
    batch_id: str
    status: str
    total_urls: int
    processed: int
    results: dict


def validate_redirect_url(url: str) -> bool:
    """Validate that a redirect URL uses HTTPS."""
    return url.startswith("https://")


async def download_audio_async(url: str, output_path: str) -> None:
    """Download audio file with SSRF protection and size limits using async streaming."""
    # Use global HTTP client with connection pooling
    client = get_http_client()
    timeout = httpx.Timeout(
            connect=CONNECT_TIMEOUT,
            read=TOTAL_TIMEOUT,
            write=TOTAL_TIMEOUT,
            pool=CONNECT_TIMEOUT
    )
    
    # Use global client (connection pooling enabled)
    # Note: follow_redirects=False is set in global client, we handle redirects manually
    current_url = url
    redirect_count = 0
    max_redirects = 10
    
    # Follow redirects manually, validating each one
    while redirect_count < max_redirects:
        # httpx streams by default when using aiter_bytes() - no stream parameter needed
        response = await client.get(current_url, timeout=timeout)
        
        # If redirect, validate and follow
        if response.status_code in (301, 302, 303, 307, 308):
            redirect_url = response.headers.get("location")
            # Close redirect response (we don't need the body, only headers)
            await response.aclose()
            
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
            # Close response before raising error (we don't need the body)
            await response.aclose()
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
        
        # Stream download directly to file with size limit (async I/O using aiofiles)
        # aiofiles handles buffering internally, so we write chunks directly
        total_size = 0
        
        async with aiofiles.open(output_path, "wb") as f:
            async for chunk in response.aiter_bytes():
                total_size += len(chunk)
                if total_size > MAX_SIZE:
                    # Clean up partial file
                    await f.close()
                    try:
                        os.unlink(output_path)
                    except Exception:
                        pass
                    raise HTTPException(
                        status_code=400,
                        detail=f"File too large (max {MAX_SIZE / 1024 / 1024:.1f}MB)"
                    )
                
                # Write chunk directly - aiofiles handles buffering internally
                await f.write(chunk)
        
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
        Normalized BPM value (float, no rounding - rounding happens once at response level)
    """
    normalized = bpm
    
    # Apply corrections only for extreme outliers
    if normalized < 40:
        normalized *= 2
    elif normalized > 220:
        normalized /= 2
    
    return normalized


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
    # Lazy load essentia - only import when actually needed (reduces cold start time)
    import essentia.standard as es
    
    debug_lines = []
    
    try:
        # Load audio once - Essentia handles MP3/AAC decoding directly
        loader = es.MonoLoader(
            filename=audio_path,
            sampleRate=44100
        )
        audio = loader()
        
        # Cap duration to MAX_AUDIO_DURATION seconds by trimming the array
        max_samples = int(MAX_AUDIO_DURATION * 44100)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
            debug_lines.append(f"Audio loaded: {len(audio)/44100:.1f}s (trimmed from original, capped at {MAX_AUDIO_DURATION}s)")
        else:
            debug_lines.append(f"Audio loaded: {len(audio)/44100:.1f}s")
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
        
        debug_lines.append(f"BPM={bpm_raw:.2f} (normalized={round(bpm_normalized, 1):.1f})")
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
                # Try with profileType only (some Essentia versions don't support pcpSize/numHarmonics)
                key_extractor = es.KeyExtractor(profileType=profile)
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
) -> Tuple[int, dict]:
    """Process a single URL: download, analyze, return result.
    
    Returns:
        Tuple of (index, dict) with processing results and fallback flags
    """
    url_str = str(url)
    parsed = urlparse(url_str)
    input_ext = os.path.splitext(parsed.path)[1] or ".tmp"
    input_path = None
    debug_info_parts = []
    
    # Timing telemetry
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
        input_fd, input_path = tempfile.mkstemp(suffix=input_ext, dir="/tmp")
        os.close(input_fd)  # Close file descriptor, we'll write via path
        
        # Download audio (async streaming) with timing
        try:
            timing["download_start"] = time.time()
            await download_audio_async(url_str, input_path)
            timing["download_end"] = time.time()
            timing["download_duration"] = timing["download_end"] - timing["download_start"]
            debug_info_parts.append(f"URL fetch: SUCCESS ({url_str[:50]}...)")
        except Exception as e:
            # Clean up temp file on download failure
            if input_path:
                try:
                    loop = asyncio.get_event_loop()
                    file_exists = await loop.run_in_executor(None, os.path.exists, input_path)
                    if file_exists:
                        await loop.run_in_executor(None, os.unlink, input_path)
                except Exception:
                    pass  # Ignore cleanup errors
                input_path = None
            
            error_msg = f"URL fetch error: {str(e)}"
            debug_info_parts.append(error_msg)
            return index, {
                "index": index,
                "url": url_str,
                "file_path": None,
                "bpm_essentia": None,
                "bpm_raw_essentia": None,
                "bpm_confidence_essentia": None,
                "bpm_librosa": None,
                "bpm_raw_librosa": None,
                "bpm_confidence_librosa": None,
                "key_essentia": "unknown",
                "scale_essentia": "unknown",
                "keyscale_confidence_essentia": 0.0,
                "key_librosa": None,
                "scale_librosa": None,
                "keyscale_confidence_librosa": None,
                "need_fallback_bpm": True,
                "need_fallback_key": True,
                "debug_info_parts": debug_info_parts,
                "timing": timing,
            }
        
        # Analyze audio (BPM + key from same loaded array) with timing
        # Offload CPU-bound Essentia analysis to thread pool to prevent event loop blocking
        timing["essentia_start"] = time.time()
        essentia_executor = get_essentia_executor()
        essentia_sem = get_essentia_semaphore()
        loop = asyncio.get_event_loop()
        
        async with essentia_sem:
            (
                bpm_normalized, bpm_raw, bpm_confidence_normalized, bpm_quality, bpm_method,
                key, scale, key_strength_raw, key_confidence_normalized,
                need_fallback_bpm, need_fallback_key,
                analysis_debug
            ) = await loop.run_in_executor(
                essentia_executor,
                analyze_audio,
                input_path,
                max_confidence
            )
        timing["essentia_end"] = time.time()
        timing["essentia_duration"] = timing["essentia_end"] - timing["essentia_start"]
        
        debug_info_parts.append(f"Max confidence threshold: {max_confidence:.2f}")
        debug_info_parts.append("=== Analysis (Essentia) ===")
        debug_info_parts.append(analysis_debug)
        
        # Store for potential fallback batch call
        # Return result with fallback flags and separate Essentia/Librosa fields
        return index, {
            "index": index,
            "url": url_str,
            "file_path": input_path,  # Keep file for batch fallback if needed
            # Essentia results
            "bpm_essentia": int(round(bpm_normalized)) if bpm_normalized is not None else None,
            "bpm_raw_essentia": round(bpm_raw, 2) if bpm_raw is not None else None,
            "bpm_confidence_essentia": round(bpm_confidence_normalized, 2) if bpm_confidence_normalized is not None else None,
            "key_essentia": key if key else "unknown",
            "scale_essentia": scale if scale else "unknown",
            "keyscale_confidence_essentia": round(key_confidence_normalized, 2) if key_confidence_normalized is not None else 0.0,
            # Librosa results (will be populated by fallback if needed)
            "bpm_librosa": None,
            "bpm_raw_librosa": None,
            "bpm_confidence_librosa": None,
            "key_librosa": None,
            "scale_librosa": None,
            "keyscale_confidence_librosa": None,
            # Fallback flags
            "need_fallback_bpm": need_fallback_bpm,
            "need_fallback_key": need_fallback_key,
            "debug_info_parts": debug_info_parts,
            "timing": timing,
        }
    
    except Exception as e:
        error_msg = f"Processing error: {str(e)}"
        debug_info_parts.append(error_msg)
        return index, {
            "index": index,
            "url": url_str,
            "file_path": None,
            "bpm_essentia": None,
            "bpm_raw_essentia": None,
            "bpm_confidence_essentia": None,
            "bpm_librosa": None,
            "bpm_raw_librosa": None,
            "bpm_confidence_librosa": None,
            "key_essentia": "unknown",
            "scale_essentia": "unknown",
            "keyscale_confidence_essentia": 0.0,
            "key_librosa": None,
            "scale_librosa": None,
            "keyscale_confidence_librosa": None,
            "need_fallback_bpm": True,
            "need_fallback_key": True,
            "debug_info_parts": debug_info_parts,
            "timing": timing,
        }
    
    finally:
        # Note: Don't delete input_path here - we may need it for fallback batch
        # Cleanup will happen after fallback processing
        pass


async def get_auth_headers(audience: str) -> dict:
    """Generates an OIDC token for the target audience for internal Cloud Run calls."""
    if not GCP_AUTH_AVAILABLE:
        error_msg = "GCP_AUTH_AVAILABLE is False - google-auth library not imported"
        if GCP_AUTH_ERROR:
            error_msg += f"\nImport error details: {GCP_AUTH_ERROR[:500]}"
        raise Exception(error_msg)
    
    # Run synchronous auth operations in a thread to avoid blocking
    def _get_token_sync():
        try:
            # Get default credentials (uses the service account attached to the Cloud Run service)
            credentials, project = google.auth.default()
            
            if not credentials:
                raise Exception(f"No credentials found for audience: {audience}")
            
            # Refresh credentials if needed
            if not credentials.valid:
                auth_request = google.auth.transport.requests.Request()
                credentials.refresh(auth_request)
            
            # Create a request object for fetching the ID token
            auth_request = google.auth.transport.requests.Request()
            
            # Fetch the ID token with the audience (fallback service URL)
            token = google.oauth2.id_token.fetch_id_token(auth_request, audience)
            
            if not token:
                raise Exception(f"Empty token generated for audience: {audience}")
            
            return {"Authorization": f"Bearer {token}"}
        except Exception as e:
            # Re-raise with context
            raise Exception(f"Error generating auth token for {audience}: {str(e)}")
    
    try:
        # Run in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _get_token_sync)
    except Exception as e:
        # Log the full error for debugging
        import traceback
        error_details = f"Error in get_auth_headers for {audience}: {str(e)}\n{traceback.format_exc()}"
        print(error_details)
        raise  # Re-raise so caller can see the error




@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"ok": True}


def generate_debug_output(debug_info_parts: List[str], timing: dict, fallback_timing: Optional[dict], debug_level: str) -> str:
    """Generate debug output based on debug_level.
    
    Args:
        debug_info_parts: List of debug message strings
        timing: Dict with download and essentia timing
        fallback_timing: Optional dict with fallback service timing
        debug_level: "minimal", "normal", or "detailed"
    
    Returns:
        Formatted debug string
    """
    if debug_level == "minimal":
        # Only errors and final results
        filtered = [line for line in debug_info_parts if "error" in line.lower() or "SUCCESS" in line or "Fallback" in line]
        if filtered:
            return "\n".join(filtered)
        return ""
    
    elif debug_level == "normal":
        # Include telemetry summary
        output = "\n".join(debug_info_parts)
        telemetry = []
        if timing.get("download_duration"):
            telemetry.append(f"Download: {timing['download_duration']:.2f}s")
        if timing.get("essentia_duration"):
            telemetry.append(f"Essentia analysis: {timing['essentia_duration']:.2f}s")
        if fallback_timing and fallback_timing.get("duration"):
            telemetry.append(f"Fallback service: {fallback_timing['duration']:.2f}s")
        if telemetry:
            output += f"\n=== Telemetry ===\n" + ", ".join(telemetry)
        return output
    
    else:  # detailed
        # Full output with all timing details
        output = "\n".join(debug_info_parts)
        telemetry = []
        if timing.get("download_start") and timing.get("download_end"):
            telemetry.append(f"Download: {timing['download_duration']:.3f}s (start: {timing['download_start']:.3f}, end: {timing['download_end']:.3f})")
        if timing.get("essentia_start") and timing.get("essentia_end"):
            telemetry.append(f"Essentia analysis: {timing['essentia_duration']:.3f}s (start: {timing['essentia_start']:.3f}, end: {timing['essentia_end']:.3f})")
        if fallback_timing:
            if fallback_timing.get("start") and fallback_timing.get("end"):
                telemetry.append(f"Fallback service: {fallback_timing['duration']:.3f}s (start: {fallback_timing['start']:.3f}, end: {fallback_timing['end']:.3f})")
            elif fallback_timing.get("duration"):
                telemetry.append(f"Fallback service: {fallback_timing['duration']:.3f}s")
        if telemetry:
            output += f"\n=== Telemetry ===\n" + "\n".join(telemetry)
        return output


@app.post("/analyze/batch", response_model=BatchSubmissionResponse)
async def analyze_batch(request: BatchBPMRequest):
    """Submit batch for processing: publish tasks to Pub/Sub and return batch_id immediately.
    
    Processing happens asynchronously via worker service. Use /stream/{batch_id} to receive results.
    """
    max_confidence = request.max_confidence if request.max_confidence is not None else 0.65
    debug_level = request.debug_level if request.debug_level else "normal"
    
    # Generate batch_id
    batch_id = str(uuid.uuid4())
    total_urls = len(request.urls)
    
    # Create batch document in Firestore
    batch_ref = db.collection('batches').document(batch_id)
    batch_ref.set({
        'batch_id': batch_id,
        'total_urls': total_urls,
        'processed': 0,
        'status': 'processing',
        'created_at': firestore.SERVER_TIMESTAMP,
        'results': {}
    })
    
    # Publish each URL as a separate message to Pub/Sub
    for index, url in enumerate(request.urls):
        message_data = {
            'batch_id': batch_id,
            'url': str(url),
            'index': index,
            'max_confidence': max_confidence,
            'debug_level': debug_level
        }
        message_bytes = json.dumps(message_data).encode('utf-8')
        
        try:
            future = publisher.publish(PUBSUB_TOPIC, message_bytes)
            future.result()  # Wait for publish to complete
        except Exception as e:
            # Log error but continue with other URLs
            print(f"Error publishing message for URL {index}: {str(e)}")
            # Update Firestore with error
            batch_ref.update({
                f'results.{index}': {
                    'index': index,
                    'url': str(url),
                    'error': f'Failed to publish to Pub/Sub: {str(e)}'
                }
            })
    
    return BatchSubmissionResponse(
        batch_id=batch_id,
        total_urls=total_urls,
        status='processing',
        stream_url=f'/stream/{batch_id}'
    )


@app.get("/stream/{batch_id}")
async def stream_batch_results(batch_id: str):
    """Stream batch results as NDJSON (Newline Delimited JSON).
    
    Polls Firestore every 500ms and yields results as they become available.
    """
    batch_ref = db.collection('batches').document(batch_id)
    sent_indices = set()
    
    async def generate_stream():
        loop = asyncio.get_event_loop()
        start_time = time.time()
        last_keepalive = start_time
        max_stream_time = 600  # 10 minutes max stream time
        
        while True:
            try:
                current_time = time.time()
                elapsed = current_time - start_time
                
                # Check max stream time
                if elapsed > max_stream_time:
                    yield json.dumps({
                        "type": "error",
                        "message": f"Stream timeout after {max_stream_time}s"
                    }) + "\n"
                    break
                
                # Send keepalive every 30 seconds to prevent connection timeout
                if current_time - last_keepalive >= 30:
                    yield json.dumps({
                        "type": "keepalive",
                        "elapsed": int(elapsed)
                    }) + "\n"
                    last_keepalive = current_time
                
                # Get batch document from Firestore (run in executor to avoid blocking)
                batch_doc = await loop.run_in_executor(None, batch_ref.get)
                
                if not batch_doc.exists:
                    # Batch not found
                    yield json.dumps({
                        "type": "error",
                        "message": f"Batch {batch_id} not found"
                    }) + "\n"
                    break
                
                batch_data = batch_doc.to_dict()
                status = batch_data.get('status', 'processing')
                total_urls = batch_data.get('total_urls', 0)
                processed = batch_data.get('processed', 0)
                results = batch_data.get('results', {})
                
                # Send status update (only if status changed or every 10 seconds)
                should_send_status = (
                    len(sent_indices) == 0 or  # First status
                    processed > len(sent_indices) or  # New results available
                    int(elapsed) % 10 == 0  # Every 10 seconds
                )
                
                if should_send_status:
                    yield json.dumps({
                        "type": "status",
                        "status": status,
                        "total": total_urls,
                        "processed": processed,
                        "elapsed": int(elapsed)
                    }) + "\n"
                
                # Send new results
                for index_str, result in results.items():
                    index = int(index_str)
                    if index not in sent_indices:
                        sent_indices.add(index)
                        # Format result for streaming
                        stream_result = {
                            "type": "result",
                            "index": index,
                            "url": result.get("url"),
                            "bpm_essentia": result.get("bpm_essentia"),
                            "bpm_raw_essentia": result.get("bpm_raw_essentia"),
                            "bpm_confidence_essentia": result.get("bpm_confidence_essentia"),
                            "bpm_librosa": result.get("bpm_librosa"),
                            "bpm_raw_librosa": result.get("bpm_raw_librosa"),
                            "bpm_confidence_librosa": result.get("bpm_confidence_librosa"),
                            "key_essentia": result.get("key_essentia"),
                            "scale_essentia": result.get("scale_essentia"),
                            "keyscale_confidence_essentia": result.get("keyscale_confidence_essentia"),
                            "key_librosa": result.get("key_librosa"),
                            "scale_librosa": result.get("scale_librosa"),
                            "keyscale_confidence_librosa": result.get("keyscale_confidence_librosa"),
                            "debug_txt": result.get("debug_txt"),
                            "error": result.get("error")
                        }
                        yield json.dumps(stream_result) + "\n"
                
                # Send progress update only when processed count changes
                if processed > 0 and processed != len(sent_indices):
                    yield json.dumps({
                        "type": "progress",
                        "processed": processed,
                        "total": total_urls
                    }) + "\n"
                
                # Check if complete
                if status == "completed":
                    yield json.dumps({
                        "type": "complete",
                        "batch_id": batch_id,
                        "total": total_urls,
                        "elapsed": int(elapsed)
                    }) + "\n"
                    break
                
                # Wait before next poll
                await asyncio.sleep(0.5)
                
            except Exception as e:
                import traceback
                error_msg = f"Error streaming batch: {str(e)}\n{traceback.format_exc()}"
                yield json.dumps({
                    "type": "error",
                    "message": error_msg[:500]  # Limit error message length
                }) + "\n"
                break
    
    return StreamingResponse(
        generate_stream(),
        media_type="application/x-ndjson"
    )


@app.get("/batch/{batch_id}", response_model=BatchStatusResponse)
async def get_batch_status(batch_id: str):
    """Get batch status and results from Firestore."""
    batch_ref = db.collection('batches').document(batch_id)
    loop = asyncio.get_event_loop()
    batch_doc = await loop.run_in_executor(None, batch_ref.get)
    
    if not batch_doc.exists:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")
    
    batch_data = batch_doc.to_dict()
    
    return BatchStatusResponse(
        batch_id=batch_id,
        status=batch_data.get('status', 'processing'),
        total_urls=batch_data.get('total_urls', 0),
        processed=batch_data.get('processed', 0),
        results=batch_data.get('results', {})
    )
