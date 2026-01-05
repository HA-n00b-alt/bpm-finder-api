"""
BPM Finder API - Google Cloud Run microservice
Computes BPM and key from 30s audio preview URLs.
"""
import os
import tempfile
import subprocess
import math
from typing import Optional, Tuple
from urllib.parse import urlparse
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl, field_validator
import numpy as np
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

# Fallback Service Configuration
FALLBACK_SERVICE_URL = "https://bpm-fallback-service-340051416180.europe-west3.run.app"
FALLBACK_SERVICE_AUDIENCE = FALLBACK_SERVICE_URL


class BPMRequest(BaseModel):
    url: HttpUrl
    max_confidence: Optional[float] = 0.65

    @field_validator("url")
    @classmethod
    def validate_url(cls, v):
        """Validate URL scheme."""
        url_str = str(v)
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


def download_audio(url: str, output_path: str) -> None:
    """Download audio file with SSRF protection and size limits."""
    with httpx.Client(
        timeout=httpx.Timeout(
            connect=CONNECT_TIMEOUT,
            read=TOTAL_TIMEOUT,
            write=TOTAL_TIMEOUT,
            pool=CONNECT_TIMEOUT
        ),
        follow_redirects=False,  # Manual redirect handling for validation
        max_redirects=10,
    ) as client:
        current_url = url
        max_redirects = 10
        redirect_count = 0
        
        # Follow redirects manually, validating each one
        while redirect_count < max_redirects:
            response = client.get(current_url)
            
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
            
            # Download with size limit
            content = b""
            for chunk in response.iter_bytes():
                content += chunk
                if len(content) > MAX_SIZE:
                    raise HTTPException(
                        status_code=400,
                        detail=f"File too large (max {MAX_SIZE / 1024 / 1024:.1f}MB)"
                    )
            
            # Write to file
            with open(output_path, "wb") as f:
                f.write(content)
            
            return
        
        # Too many redirects
        raise HTTPException(
            status_code=400,
            detail="Too many redirects"
        )


def convert_to_wav(input_path: str, output_path: str) -> None:
    """Convert audio to mono 44100Hz 16-bit PCM WAV using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "1",  # mono
        "-y",  # overwrite output
        output_path,
    ]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
    )
    
    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"ffmpeg conversion failed: {result.stderr[:200]}"
        )


def compute_bpm(wav_path: str) -> Tuple[float, float, float, str, str]:
    """Compute BPM using Essentia RhythmExtractor2013 with multifeature method.
    
    Args:
        wav_path: Path to the WAV audio file
    
    Returns:
        (normalized_bpm, raw_bpm, normalized_confidence, quality_level, debug_info) tuple
    """
    debug_lines = []
    
    try:
        # Load audio
        loader = es.MonoLoader(filename=wav_path, sampleRate=44100)
        audio = loader()
    except Exception as e:
        error_msg = f"Essentia audio loading error: {str(e)}"
        debug_lines.append(error_msg)
        raise Exception(error_msg)
    
    # BPM extraction using multifeature method
    try:
        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        bpm_raw, beats, confidence_raw, _, beats_intervals = rhythm_extractor(audio)
        
        # Normalize confidence and get quality level
        confidence_normalized, quality = normalize_confidence(float(confidence_raw))
        bpm_normalized = normalize_bpm(float(bpm_raw))
        
        debug_lines.append(f"BPM={bpm_raw:.2f} (normalized={bpm_normalized:.1f})")
        debug_lines.append(f"Confidence: raw={confidence_raw:.3f} (range: 0-5.32), normalized={confidence_normalized:.3f} (0-1), quality={quality}")
        
        return bpm_normalized, float(bpm_raw), confidence_normalized, quality, "\n".join(debug_lines)
    except Exception as e:
        error_msg = f"BPM extraction error: {str(e)}"
        debug_lines.append(error_msg)
        raise Exception(error_msg)


def compute_key(wav_path: str) -> Tuple[str, str, float, float, str]:
    """Compute musical key using Essentia KeyExtractor with multiple profile types.
    
    Tries multiple key profile types and returns the result with the highest strength value.
    
    Args:
        wav_path: Path to the WAV audio file
    
    Returns:
        (key, scale, raw_strength, normalized_strength, debug_info) tuple
        - key: The detected key (e.g., "C", "D", "E", etc.)
        - scale: The detected scale ("major" or "minor")
        - raw_strength: Raw strength value from Essentia
        - normalized_strength: Normalized strength (0.0-1.0), assuming KeyExtractor returns 0-1 range
        - debug_info: Debug information string
    """
    debug_lines = []
    
    try:
        # Load audio
        loader = es.MonoLoader(filename=wav_path, sampleRate=44100)
        audio = loader()
    except Exception as e:
        error_msg = f"Essentia audio loading error (key): {str(e)}"
        debug_lines.append(error_msg)
        raise Exception(error_msg)
    
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
            key, scale, strength = key_extractor(audio)
            results.append((str(key), str(scale), float(strength), profile))
            debug_lines.append(f"key_profile={profile}: key={key} {scale}, strength={strength:.3f}")
        except Exception as e:
            # Profile not available in this Essentia version, skip
            debug_lines.append(f"key_profile={profile}: error={str(e)}")
            continue
    
    # If we have results, return the one with highest strength
    if results:
        # Sort by strength (index 2) in descending order
        results.sort(key=lambda x: x[2], reverse=True)
        key, scale, raw_strength, winning_profile = results[0]
        
        # Normalize strength (assume KeyExtractor returns 0-1, but clamp if > 1)
        normalized_strength = min(1.0, max(0.0, raw_strength))
        
        debug_lines.append(f"Winner: {winning_profile} profile (strength={raw_strength:.3f}, normalized={normalized_strength:.3f})")
        if len(results) > 1:
            strength_range = results[0][2] - results[-1][2]
            debug_lines.append(f"Key ensemble: {len(results)} profiles, strength_range={strength_range:.3f}")
        return key, scale, float(raw_strength), float(normalized_strength), "\n".join(debug_lines)
    
    # Fall back to default KeyExtractor if no profiles worked
    try:
        key_extractor = es.KeyExtractor()
        key, scale, raw_strength = key_extractor(audio)
        normalized_strength = min(1.0, max(0.0, raw_strength))
        debug_lines.append(f"Fallback: default KeyExtractor, key={key} {scale}, strength={raw_strength:.3f} (normalized={normalized_strength:.3f})")
        return str(key), str(scale), float(raw_strength), float(normalized_strength), "\n".join(debug_lines)
    except Exception as e:
        error_msg = f"KeyExtractor fallback error: {str(e)}"
        debug_lines.append(error_msg)
        raise Exception(error_msg)


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


@app.post("/bpm", response_model=BPMResponse)
async def compute_bpm_from_url(request: BPMRequest):
    """Compute BPM and key from audio preview URL."""
    url_str = str(request.url)
    parsed = urlparse(url_str)
    
    # Create temporary files
    input_ext = os.path.splitext(parsed.path)[1] or ".tmp"
    input_file = None
    output_file = None
    debug_info_parts = []
    
    try:
        # Create temp files
        input_fd, input_path = tempfile.mkstemp(suffix=input_ext, dir="/tmp")
        input_file = os.fdopen(input_fd, "wb")
        input_file.close()
        
        output_fd, output_path = tempfile.mkstemp(suffix=".wav", dir="/tmp")
        output_file = os.fdopen(output_fd, "wb")
        output_file.close()
        
        # Download audio
        try:
            download_audio(url_str, input_path)
            debug_info_parts.append(f"URL fetch: SUCCESS ({url_str[:50]}...)")
        except Exception as e:
            error_msg = f"URL fetch error: {str(e)}"
            debug_info_parts.append(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Convert to WAV
        try:
            convert_to_wav(input_path, output_path)
            debug_info_parts.append("Audio conversion: SUCCESS")
        except Exception as e:
            error_msg = f"Audio conversion error: {str(e)}"
            debug_info_parts.append(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Get max_confidence threshold (default 0.65)
        max_confidence = request.max_confidence if request.max_confidence is not None else 0.65
        debug_info_parts.append(f"Max confidence threshold: {max_confidence:.2f}")
        
        # Compute BPM using multifeature method
        bpm_normalized = None
        bpm_raw = None
        bpm_confidence_normalized = None
        bpm_quality = None
        bpm_method = "multifeature"
        need_fallback_bpm = False
        
        try:
            bpm_normalized, bpm_raw, bpm_confidence_normalized, bpm_quality, bpm_debug = compute_bpm(output_path)
            debug_info_parts.append("=== BPM Analysis (Essentia) ===")
            debug_info_parts.append(bpm_debug)
            
            # Check if BPM confidence meets threshold
            if bpm_confidence_normalized >= max_confidence:
                debug_info_parts.append(f"BPM confidence ({bpm_confidence_normalized:.3f}) >= threshold ({max_confidence:.2f}) - using Essentia result")
            else:
                need_fallback_bpm = True
                debug_info_parts.append(f"BPM confidence ({bpm_confidence_normalized:.3f}) < threshold ({max_confidence:.2f}) - fallback needed")
        except Exception as e:
            error_msg = f"BPM computation error: {str(e)}"
            debug_info_parts.append(error_msg)
            need_fallback_bpm = True  # Need fallback if computation failed
        
        # Compute key using Essentia
        key = "unknown"
        scale = "unknown"
        key_strength_raw = 0.0
        key_confidence_normalized = 0.0
        need_fallback_key = False
        
        try:
            key, scale, key_strength_raw, key_confidence_normalized, key_debug = compute_key(output_path)
            debug_info_parts.append("=== Key Analysis (Essentia) ===")
            debug_info_parts.append(key_debug)
            
            # Check if key strength meets threshold
            if key_confidence_normalized >= max_confidence:
                debug_info_parts.append(f"Key strength ({key_confidence_normalized:.3f}) >= threshold ({max_confidence:.2f}) - using Essentia result")
            else:
                need_fallback_key = True
                debug_info_parts.append(f"Key strength ({key_confidence_normalized:.3f}) < threshold ({max_confidence:.2f}) - fallback needed")
        except Exception as e:
            error_msg = f"Key computation error: {str(e)}"
            debug_info_parts.append(error_msg)
            need_fallback_key = True  # Need fallback if computation failed
        
        # Call fallback service selectively
        if need_fallback_bpm or need_fallback_key:
            fallback_items = []
            if need_fallback_bpm:
                fallback_items.append("BPM")
            if need_fallback_key:
                fallback_items.append("key")
            
            debug_info_parts.append(f"=== Fallback Service Call ({', '.join(fallback_items)}) ===")
            
            try:
                # Get authentication headers
                auth_headers = await get_auth_headers(FALLBACK_SERVICE_AUDIENCE)
                
                # Read the WAV file content into memory
                with open(output_path, "rb") as wav_file:
                    wav_content = wav_file.read()
                
                files = {"audio_file": ("audio.wav", wav_content, "audio/wav")}
                data = {
                    "url": url_str,
                    "process_bpm": need_fallback_bpm,
                    "process_key": need_fallback_key
                }
                
                # Make async POST request to fallback service
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(
                        f"{FALLBACK_SERVICE_URL}/bpm",
                        files=files,
                        data=data,
                        headers=auth_headers
                    )
                    
                    if response.status_code == 200:
                        fallback_data = response.json()
                        debug_info_parts.append("Fallback service: SUCCESS")
                        
                        # Update BPM if fallback was needed and response contains BPM data
                        if need_fallback_bpm and fallback_data.get("bpm_normalized") is not None:
                            bpm_normalized = fallback_data["bpm_normalized"]
                            bpm_raw = fallback_data["bpm_raw"]
                            bpm_confidence_normalized = fallback_data["confidence"]
                            bpm_method = "librosa_hpss_fallback"
                            debug_info_parts.append(f"Fallback BPM: {bpm_normalized:.1f} (confidence={bpm_confidence_normalized:.3f})")
                        elif need_fallback_bpm:
                            debug_info_parts.append("Fallback BPM: No data returned from fallback service")
                        
                        # Update key if fallback was needed and response contains key data
                        if need_fallback_key and fallback_data.get("key") is not None:
                            key = fallback_data["key"]
                            scale = fallback_data["scale"]
                            key_confidence_normalized = fallback_data["key_confidence"]
                            debug_info_parts.append(f"Fallback key: {key} {scale} (confidence={key_confidence_normalized:.3f})")
                        elif need_fallback_key:
                            debug_info_parts.append("Fallback key: No data returned from fallback service")
                    else:
                        debug_info_parts.append(f"Fallback service: HTTP {response.status_code} - {response.text[:200]}")
                        # Continue with Essentia results if fallback fails
            except Exception as e:
                # Log error but don't fail - proceed with Essentia results
                debug_info_parts.append(f"Fallback service error: {str(e)[:200]}")
                # Continue with Essentia results if fallback fails
        
        # Ensure we have valid values (use defaults if fallback failed and Essentia also failed)
        if bpm_normalized is None:
            bpm_normalized = 0.0
            bpm_raw = 0.0
            bpm_confidence_normalized = 0.0
            bpm_method = "error"
        
        return BPMResponse(
            bpm=int(round(bpm_normalized)),
            bpm_raw=round(bpm_raw, 2),
            bpm_confidence=round(bpm_confidence_normalized, 2),
            bpm_method=bpm_method,
            debug_info="\n".join(debug_info_parts),
            key=key,
            scale=scale,
            key_confidence=round(key_confidence_normalized, 2),
        )
    
    except HTTPException:
        raise
    except Exception as e:
        debug_info_parts.append(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)[:200]}"
        )
    
    finally:
        # Clean up temp files
        for path in [input_path, output_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception:
                    pass  # Ignore cleanup errors

