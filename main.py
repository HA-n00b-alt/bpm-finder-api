"""
BPM Finder API - Google Cloud Run microservice
Computes BPM and key from 30s audio preview URLs.
"""
import os
import tempfile
import subprocess
from typing import Optional, Tuple
from urllib.parse import urlparse
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl, field_validator
import numpy as np
import essentia.standard as es

app = FastAPI(title="BPM Finder API")

# Download limits
CONNECT_TIMEOUT = 5.0  # seconds
TOTAL_TIMEOUT = 20.0  # seconds
MAX_SIZE = 10 * 1024 * 1024  # 10MB


class BPMRequest(BaseModel):
    url: HttpUrl

    @field_validator("url")
    @classmethod
    def validate_url(cls, v):
        """Validate URL scheme."""
        url_str = str(v)
        if not url_str.startswith("https://"):
            raise ValueError("Only HTTPS URLs are allowed")
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
    """Compute BPM using Essentia with three-method ensemble approach.
    
    Implements a robust three-method BPM extraction: multifeature, degara, and onset-based.
    Normalizes BPM values before comparing confidence scores to prevent octave errors,
    and returns the result with highest confidence.
    
    Args:
        wav_path: Path to the WAV audio file
    
    Returns:
        (normalized_bpm, raw_bpm, confidence, method_name, debug_info) tuple
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
    
    # Method 1: Multifeature
    bpm_mf_raw = None
    bpm_mf_normalized = None
    beats_confidence_mf = 0.0
    mf_error = None
    try:
        rhythm_extractor_mf = es.RhythmExtractor2013(method="multifeature")
        bpm_mf_raw, beats_mf, beats_confidence_mf, _, beats_intervals_mf = rhythm_extractor_mf(audio)
        bpm_mf_normalized = normalize_bpm(float(bpm_mf_raw))
        debug_lines.append(f"multifeature: BPM={bpm_mf_raw:.2f} (norm={bpm_mf_normalized:.1f}), confidence={beats_confidence_mf:.3f}")
    except Exception as e:
        mf_error = f"multifeature error: {str(e)}"
        debug_lines.append(mf_error)
        beats_confidence_mf = -1.0  # Mark as failed
    
    # Method 2: Degara
    bpm_dg_raw = None
    bpm_dg_normalized = None
    beats_confidence_dg = 0.0
    dg_error = None
    try:
        rhythm_extractor_dg = es.RhythmExtractor2013(method="degara")
        bpm_dg_raw, beats_dg, beats_confidence_dg, _, beats_intervals_dg = rhythm_extractor_dg(audio)
        bpm_dg_normalized = normalize_bpm(float(bpm_dg_raw))
        debug_lines.append(f"degara: BPM={bpm_dg_raw:.2f} (norm={bpm_dg_normalized:.1f}), confidence={beats_confidence_dg:.3f}")
    except Exception as e:
        dg_error = f"degara error: {str(e)}"
        debug_lines.append(dg_error)
        beats_confidence_dg = -1.0  # Mark as failed
    
    # Method 3: Onset-based
    bpm_onset_raw = None
    bpm_onset_norm = None
    beats_confidence_onset = 0.0
    onset_error = None
    try:
        od = es.OnsetDetection(method='complex')
        onsets = od(audio)
        bt = es.BeatTrackerDegara()
        bpm_onset_raw, _, beats_confidence_onset, _, _ = bt(onsets)
        bpm_onset_norm = normalize_bpm(float(bpm_onset_raw))
        debug_lines.append(f"onset: BPM={bpm_onset_raw:.2f} (norm={bpm_onset_norm:.1f}), confidence={beats_confidence_onset:.3f}")
    except Exception as e:
        onset_error = f"onset error: {str(e)}"
        debug_lines.append(onset_error)
        beats_confidence_onset = -1.0  # Mark as failed
    
    # Analyze confidence scores for insights
    valid_confidences = [c for c in [beats_confidence_mf, beats_confidence_dg, beats_confidence_onset] if c >= 0]
    if valid_confidences:
        avg_confidence = sum(valid_confidences) / len(valid_confidences)
        min_confidence = min(valid_confidences)
        max_confidence = max(valid_confidences)
        
        if min_confidence < 0.3:
            debug_lines.append(f"WARNING: Low confidence detected (min={min_confidence:.3f})")
        if max_confidence > 0.8:
            debug_lines.append(f"INFO: High confidence detected (max={max_confidence:.3f})")
        if max_confidence - min_confidence > 0.4:
            debug_lines.append(f"INFO: High confidence variance (range={max_confidence - min_confidence:.3f})")
        debug_lines.append(f"Ensemble: avg_confidence={avg_confidence:.3f}, range=[{min_confidence:.3f}, {max_confidence:.3f}]")
    
    # Compare confidence scores and return normalized BPM of winning method
    if beats_confidence_mf >= beats_confidence_dg and beats_confidence_mf >= beats_confidence_onset and beats_confidence_mf >= 0:
        debug_lines.append(f"Winner: multifeature (confidence={beats_confidence_mf:.3f})")
        return bpm_mf_normalized, float(bpm_mf_raw), float(beats_confidence_mf), "multifeature", "\n".join(debug_lines)
    elif beats_confidence_dg >= beats_confidence_onset and beats_confidence_dg >= 0:
        debug_lines.append(f"Winner: degara (confidence={beats_confidence_dg:.3f})")
        return bpm_dg_normalized, float(bpm_dg_raw), float(beats_confidence_dg), "degara", "\n".join(debug_lines)
    elif beats_confidence_onset >= 0:
        debug_lines.append(f"Winner: onset (confidence={beats_confidence_onset:.3f})")
        return bpm_onset_norm, float(bpm_onset_raw), float(beats_confidence_onset), "onset", "\n".join(debug_lines)
    else:
        # All methods failed, return first available or raise error
        error_msg = "All BPM extraction methods failed"
        debug_lines.append(error_msg)
        raise Exception(f"{error_msg}\n" + "\n".join(debug_lines))


def compute_key(wav_path: str) -> Tuple[str, str, float, str]:
    """Compute musical key using Essentia KeyExtractor with multiple profile types.
    
    Tries multiple key profile types and returns the result with the highest strength value.
    
    Args:
        wav_path: Path to the WAV audio file
    
    Returns:
        (key, scale, confidence, debug_info) tuple
        - key: The detected key (e.g., "C", "D", "E", etc.)
        - scale: The detected scale ("major" or "minor")
        - confidence: Confidence score (0.0-1.0)
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
        key, scale, strength, winning_profile = results[0]
        debug_lines.append(f"Winner: {winning_profile} profile (strength={strength:.3f})")
        if len(results) > 1:
            strength_range = results[0][2] - results[-1][2]
            debug_lines.append(f"Key ensemble: {len(results)} profiles, strength_range={strength_range:.3f}")
        return key, scale, strength, "\n".join(debug_lines)
    
    # Fall back to default KeyExtractor if no profiles worked
    try:
        key_extractor = es.KeyExtractor()
        key, scale, strength = key_extractor(audio)
        debug_lines.append(f"Fallback: default KeyExtractor, key={key} {scale}, strength={strength:.3f}")
        return str(key), str(scale), float(strength), "\n".join(debug_lines)
    except Exception as e:
        error_msg = f"KeyExtractor fallback error: {str(e)}"
        debug_lines.append(error_msg)
        raise Exception(error_msg)


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
        
        # Compute BPM using three-method ensemble
        try:
            bpm_normalized, bpm_raw, confidence, bpm_method, bpm_debug = compute_bpm(output_path)
            debug_info_parts.append("=== BPM Analysis ===")
            debug_info_parts.append(bpm_debug)
        except Exception as e:
            error_msg = f"BPM computation error: {str(e)}"
            debug_info_parts.append(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Compute key
        try:
            key, scale, key_confidence, key_debug = compute_key(output_path)
            debug_info_parts.append("=== Key Analysis ===")
            debug_info_parts.append(key_debug)
        except Exception as e:
            error_msg = f"Key computation error: {str(e)}"
            debug_info_parts.append(error_msg)
            # Don't fail the whole request if key fails, but include error in debug
            key = "unknown"
            scale = "unknown"
            key_confidence = 0.0
        
        return BPMResponse(
            bpm=int(round(bpm_normalized)),
            bpm_raw=round(bpm_raw, 2),
            bpm_confidence=round(confidence, 2),
            bpm_method=bpm_method,
            debug_info="\n".join(debug_info_parts),
            key=key,
            scale=scale,
            key_confidence=round(key_confidence, 2),
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

