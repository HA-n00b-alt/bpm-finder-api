"""
Shared processing functions for BPM analysis
Used by both main service and worker service
"""
import os
import time
import asyncio
import tempfile
from typing import Optional, Tuple, List
from urllib.parse import urlparse
import httpx
import aiofiles

# Pre-import Essentia at module level to avoid hanging on first use
# This is a heavy import but necessary for analysis
try:
    import essentia.standard as es
    ESSENTIA_AVAILABLE = True
except ImportError:
    es = None
    ESSENTIA_AVAILABLE = False

# Download limits
CONNECT_TIMEOUT = 5.0
TOTAL_TIMEOUT = 20.0
MAX_SIZE = 10 * 1024 * 1024  # 10MB
MAX_AUDIO_DURATION = 35.0

# Fallback Service Configuration
FALLBACK_SERVICE_URL = "https://bpm-fallback-service-340051416180.europe-west3.run.app"
FALLBACK_SERVICE_AUDIENCE = FALLBACK_SERVICE_URL
FALLBACK_REQUEST_TIMEOUT_COLD_START = 120.0
FALLBACK_REQUEST_TIMEOUT_WARM = 60.0
FALLBACK_MAX_RETRIES = 3
FALLBACK_RETRY_DELAY = 2.0


class FallbackCircuitBreaker:
    """Simple circuit breaker for fallback service calls."""
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"
    
    def record_success(self):
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
    
    def can_attempt(self) -> bool:
        if self.state == "closed":
            return True
        elif self.state == "open":
            if self.last_failure_time and (time.time() - self.last_failure_time) >= self.recovery_timeout:
                self.state = "half_open"
                return True
            return False
        else:  # half_open
            return True


def validate_redirect_url(url: str) -> bool:
    """Validate that a redirect URL uses HTTPS."""
    return url.startswith("https://")


async def download_audio_async(url: str, output_path: str, client: httpx.AsyncClient) -> None:
    """Download audio file with SSRF protection and size limits."""
    timeout = httpx.Timeout(
        connect=CONNECT_TIMEOUT,
        read=TOTAL_TIMEOUT,
        write=TOTAL_TIMEOUT,
        pool=CONNECT_TIMEOUT
    )
    
    current_url = url
    redirect_count = 0
    max_redirects = 10
    
    while redirect_count < max_redirects:
        response = await client.get(current_url, timeout=timeout)
        
        if response.status_code in (301, 302, 303, 307, 308):
            redirect_url = response.headers.get("location")
            await response.aclose()
            
            if not redirect_url:
                raise Exception("Invalid redirect: missing Location header")
            
            if redirect_url.startswith("/"):
                parsed = urlparse(current_url)
                redirect_url = f"{parsed.scheme}://{parsed.netloc}{redirect_url}"
            elif not redirect_url.startswith("http"):
                parsed = urlparse(current_url)
                redirect_url = f"{parsed.scheme}://{parsed.netloc}/{redirect_url.lstrip('/')}"
            
            if not validate_redirect_url(redirect_url):
                raise Exception("Redirect to non-HTTPS URL not allowed")
            
            current_url = redirect_url
            redirect_count += 1
            continue
        
        if response.status_code != 200:
            await response.aclose()
            raise Exception(f"Download failed: HTTP {response.status_code}")
        
        final_url = str(response.url)
        if not validate_redirect_url(final_url):
            raise Exception("Final URL must use HTTPS")
        
        content_length = response.headers.get("content-length")
        if content_length and int(content_length) > MAX_SIZE:
            raise Exception(f"File too large (max {MAX_SIZE / 1024 / 1024:.1f}MB)")
        
        total_size = 0
        try:
            async with aiofiles.open(output_path, "wb") as f:
                async for chunk in response.aiter_bytes():
                    total_size += len(chunk)
                    if total_size > MAX_SIZE:
                        await f.close()
                        try:
                            os.unlink(output_path)
                        except Exception:
                            pass
                        await response.aclose()
                        raise Exception(f"File too large (max {MAX_SIZE / 1024 / 1024:.1f}MB)")
                    await f.write(chunk)
        finally:
            await response.aclose()
        
        return
    
    raise Exception("Too many redirects")


def normalize_confidence(confidence: float) -> Tuple[float, str]:
    """Normalize confidence value to 0-1 range."""
    if confidence < 0:
        return 0.0, "very low"
    
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
    
    normalized = min(1.0, confidence / MAX_CONFIDENCE)
    return float(normalized), quality


def normalize_bpm(bpm: float) -> float:
    """Normalize BPM by applying corrections for extreme outliers."""
    normalized = bpm
    if normalized < 40:
        normalized *= 2
    elif normalized > 220:
        normalized /= 2
    return normalized


def analyze_audio(audio_path: str, max_confidence: float) -> Tuple:
    """Analyze audio file: compute BPM and key."""
    import sys
    import time
    start_time = time.time()
    print(f"[ANALYZE] Starting analysis of {audio_path} at {start_time}", file=sys.stderr, flush=True)
    
    # Check if Essentia is available (should be pre-imported at module level)
    if not ESSENTIA_AVAILABLE or es is None:
        error_msg = "Essentia not available (import failed at module level)"
        print(f"[ANALYZE] ERROR: {error_msg}", file=sys.stderr, flush=True)
        return None, None, None, None, "error", "unknown", "unknown", 0.0, 0.0, True, True, error_msg
    
    print(f"[ANALYZE] Essentia available (pre-imported)", file=sys.stderr, flush=True)
    
    debug_lines = []
    
    try:
        print(f"[ANALYZE] Creating MonoLoader...", file=sys.stderr, flush=True)
        loader = es.MonoLoader(filename=audio_path, sampleRate=44100)
        print(f"[ANALYZE] Loading audio file...", file=sys.stderr, flush=True)
        audio = loader()
        print(f"[ANALYZE] Audio loaded: {len(audio)} samples", file=sys.stderr, flush=True)
        
        max_samples = int(MAX_AUDIO_DURATION * 44100)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
            debug_lines.append(f"Audio loaded: {len(audio)/44100:.1f}s (trimmed, capped at {MAX_AUDIO_DURATION}s)")
        else:
            debug_lines.append(f"Audio loaded: {len(audio)/44100:.1f}s")
        print(f"[ANALYZE] Audio prepared: {len(audio)} samples ({len(audio)/44100:.1f}s)", file=sys.stderr, flush=True)
    except Exception as e:
        error_msg = f"Essentia audio loading error: {str(e)}"
        print(f"[ANALYZE] ERROR loading audio: {error_msg}", file=sys.stderr, flush=True)
        import traceback
        print(f"[ANALYZE] Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
        debug_lines.append(error_msg)
        return None, None, None, None, "error", "unknown", "unknown", 0.0, 0.0, True, True, "\n".join(debug_lines)
    
    # BPM extraction
    bpm_normalized = None
    bpm_raw = None
    bpm_confidence_normalized = None
    bpm_quality = None
    bpm_method = "multifeature"
    need_fallback_bpm = False
    
    try:
        print(f"[ANALYZE] Creating RhythmExtractor2013...", file=sys.stderr, flush=True)
        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        print(f"[ANALYZE] Extracting BPM...", file=sys.stderr, flush=True)
        bpm_raw, beats, confidence_raw, _, beats_intervals = rhythm_extractor(audio)
        print(f"[ANALYZE] BPM extracted: {bpm_raw:.2f}, confidence: {confidence_raw:.3f}", file=sys.stderr, flush=True)
        
        bpm_confidence_normalized, bpm_quality = normalize_confidence(float(confidence_raw))
        bpm_normalized = normalize_bpm(float(bpm_raw))
        
        debug_lines.append(f"BPM={bpm_raw:.2f} (normalized={round(bpm_normalized, 1):.1f})")
        debug_lines.append(f"Confidence: raw={confidence_raw:.3f}, normalized={bpm_confidence_normalized:.3f}, quality={bpm_quality}")
        
        if bpm_confidence_normalized >= max_confidence:
            debug_lines.append(f"BPM confidence ({bpm_confidence_normalized:.3f}) >= threshold ({max_confidence:.2f})")
        else:
            need_fallback_bpm = True
            debug_lines.append(f"BPM confidence ({bpm_confidence_normalized:.3f}) < threshold ({max_confidence:.2f}) - fallback needed")
    except Exception as e:
        error_msg = f"BPM extraction error: {str(e)}"
        print(f"[ANALYZE] ERROR in BPM extraction: {error_msg}", file=sys.stderr, flush=True)
        import traceback
        print(f"[ANALYZE] BPM Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
        debug_lines.append(error_msg)
        need_fallback_bpm = True
    
    # Key extraction
    key = "unknown"
    scale = "unknown"
    key_strength_raw = 0.0
    key_confidence_normalized = 0.0
    need_fallback_key = False
    
    try:
        print(f"[ANALYZE] Starting key extraction...", file=sys.stderr, flush=True)
        key_profiles = ['temperley', 'krumhansl', 'edma', 'edmm']
        results = []
        
        for profile in key_profiles:
            try:
                print(f"[ANALYZE] Trying key profile: {profile}", file=sys.stderr, flush=True)
                key_extractor = es.KeyExtractor(profileType=profile)
                key_result, scale_result, strength = key_extractor(audio)
                results.append((str(key_result), str(scale_result), float(strength), profile))
                debug_lines.append(f"key_profile={profile}: key={key_result} {scale_result}, strength={strength:.3f}")
                print(f"[ANALYZE] Profile {profile}: {key_result} {scale_result}, strength={strength:.3f}", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[ANALYZE] Profile {profile} failed: {str(e)}", file=sys.stderr, flush=True)
                continue
        
        if results:
            results.sort(key=lambda x: x[2], reverse=True)
            key, scale, key_strength_raw, winning_profile = results[0]
            key_confidence_normalized = min(1.0, max(0.0, key_strength_raw))
            debug_lines.append(f"Winner: {winning_profile} (strength={key_strength_raw:.3f})")
            print(f"[ANALYZE] Key winner: {winning_profile} - {key} {scale}, strength={key_strength_raw:.3f}", file=sys.stderr, flush=True)
        else:
            try:
                print(f"[ANALYZE] No profile results, trying default KeyExtractor", file=sys.stderr, flush=True)
                key_extractor = es.KeyExtractor()
                key, scale, key_strength_raw = key_extractor(audio)
                key_confidence_normalized = min(1.0, max(0.0, key_strength_raw))
                debug_lines.append(f"Fallback: default KeyExtractor")
                print(f"[ANALYZE] Default KeyExtractor: {key} {scale}, strength={key_strength_raw:.3f}", file=sys.stderr, flush=True)
            except Exception as e:
                error_msg = f"KeyExtractor fallback error: {str(e)}"
                print(f"[ANALYZE] ERROR in default KeyExtractor: {error_msg}", file=sys.stderr, flush=True)
                import traceback
                print(f"[ANALYZE] KeyExtractor Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
                debug_lines.append(error_msg)
                need_fallback_key = True
        
        if key_confidence_normalized >= max_confidence:
            debug_lines.append(f"Key strength ({key_confidence_normalized:.3f}) >= threshold")
        else:
            need_fallback_key = True
            debug_lines.append(f"Key strength ({key_confidence_normalized:.3f}) < threshold - fallback needed")
    except Exception as e:
        error_msg = f"Key computation error: {str(e)}"
        print(f"[ANALYZE] ERROR in key computation: {error_msg}", file=sys.stderr, flush=True)
        import traceback
        print(f"[ANALYZE] Key computation Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
        debug_lines.append(error_msg)
        need_fallback_key = True
    
    total_time = time.time() - start_time
    print(f"[ANALYZE] Analysis complete: BPM={bpm_normalized}, Key={key} {scale} (total time: {total_time:.2f}s)", file=sys.stderr, flush=True)
    
    result = (
        bpm_normalized, bpm_raw, bpm_confidence_normalized, bpm_quality, bpm_method,
        key, scale, key_strength_raw, key_confidence_normalized,
        need_fallback_bpm, need_fallback_key,
        "\n".join(debug_lines)
    )
    print(f"[ANALYZE] Returning result tuple (length: {len(result)})", file=sys.stderr, flush=True)
    return result


async def get_auth_headers(audience: str) -> dict:
    """Generate OIDC token for service-to-service auth."""
    try:
        import google.auth
        import google.auth.transport.requests
        import google.oauth2.id_token
    except ImportError as e:
        raise Exception(f"google-auth not available: {str(e)}")
    
    def _get_token_sync():
        credentials, project = google.auth.default()
        if not credentials.valid:
            auth_request = google.auth.transport.requests.Request()
            credentials.refresh(auth_request)
        
        auth_request = google.auth.transport.requests.Request()
        token = google.oauth2.id_token.fetch_id_token(auth_request, audience)
        
        if not token:
            raise Exception(f"Empty token for audience: {audience}")
        
        return {"Authorization": f"Bearer {token}"}
    
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _get_token_sync)


def generate_debug_output(debug_info_parts: List[str], timing: dict, fallback_timing: Optional[dict], debug_level: str) -> str:
    """Generate debug output based on debug_level."""
    if debug_level == "minimal":
        filtered = [line for line in debug_info_parts if "error" in line.lower() or "SUCCESS" in line or "Fallback" in line]
        return "\n".join(filtered) if filtered else ""
    
    elif debug_level == "normal":
        output = "\n".join(debug_info_parts)
        telemetry = []
        if timing.get("download_duration"):
            telemetry.append(f"Download: {timing['download_duration']:.2f}s")
        if timing.get("essentia_duration"):
            telemetry.append(f"Essentia: {timing['essentia_duration']:.2f}s")
        if fallback_timing and fallback_timing.get("duration"):
            telemetry.append(f"Fallback: {fallback_timing['duration']:.2f}s")
        if telemetry:
            output += f"\n=== Telemetry ===\n" + ", ".join(telemetry)
        return output
    
    else:  # detailed
        output = "\n".join(debug_info_parts)
        telemetry = []
        if timing.get("download_duration"):
            telemetry.append(f"Download: {timing['download_duration']:.3f}s")
        if timing.get("essentia_duration"):
            telemetry.append(f"Essentia: {timing['essentia_duration']:.3f}s")
        if fallback_timing and fallback_timing.get("duration"):
            telemetry.append(f"Fallback: {fallback_timing['duration']:.3f}s")
        if telemetry:
            output += f"\n=== Telemetry ===\n" + "\n".join(telemetry)
        return output
