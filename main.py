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
    confidence: float
    key: str
    scale: str
    key_confidence: float
    source_url_host: str


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


def compute_bpm(percussive_audio: np.ndarray) -> Tuple[float, float, float]:
    """Compute BPM using Essentia RhythmExtractor2013 with dual-method approach.
    
    Uses percussive component from HPSS for improved BPM accuracy. Tries both multifeature
    and degara methods, normalizes BPM values before comparing confidence scores to prevent
    octave errors, and returns the result with higher confidence.
    
    Args:
        percussive_audio: Percussive audio component from HPSS (numpy array)
    
    Returns:
        (normalized_bpm, raw_bpm, confidence) tuple
    """
    # Try multifeature method
    rhythm_extractor_mf = es.RhythmExtractor2013(method="multifeature")
    bpm_mf_raw, beats_mf, beats_confidence_mf, _, beats_intervals_mf = rhythm_extractor_mf(percussive_audio)
    bpm_mf_normalized = normalize_bpm(float(bpm_mf_raw))
    
    # Try degara method
    rhythm_extractor_dg = es.RhythmExtractor2013(method="degara")
    bpm_dg_raw, beats_dg, beats_confidence_dg, _, beats_intervals_dg = rhythm_extractor_dg(percussive_audio)
    bpm_dg_normalized = normalize_bpm(float(bpm_dg_raw))
    
    # Compare confidence scores and return normalized BPM of winning method
    if beats_confidence_mf >= beats_confidence_dg:
        return bpm_mf_normalized, float(bpm_mf_raw), float(beats_confidence_mf)
    else:
        return bpm_dg_normalized, float(bpm_dg_raw), float(beats_confidence_dg)


def compute_key(harmonic_audio: np.ndarray) -> Tuple[str, str, float]:
    """Compute musical key using Essentia KeyExtractor with multiple profile types.
    
    Uses harmonic component from HPSS for improved key detection accuracy. Tries multiple
    key profile types and returns the result with the highest strength value.
    
    Args:
        harmonic_audio: Harmonic audio component from HPSS (numpy array)
    
    Returns:
        (key, scale, confidence) tuple
        - key: The detected key (e.g., "C", "D", "E", etc.)
        - scale: The detected scale ("major" or "minor")
        - confidence: Confidence score (0.0-1.0)
    """
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
            key, scale, strength = key_extractor(harmonic_audio)
            results.append((str(key), str(scale), float(strength), profile))
        except Exception:
            # Profile not available in this Essentia version, skip
            continue
    
    # If we have results, return the one with highest strength
    if results:
        # Sort by strength (index 2) in descending order
        results.sort(key=lambda x: x[2], reverse=True)
        key, scale, strength, _ = results[0]
        return key, scale, strength
    
    # Fall back to default KeyExtractor if no profiles worked
    key_extractor = es.KeyExtractor()
    key, scale, strength = key_extractor(harmonic_audio)
    
    return str(key), str(scale), float(strength)


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
    source_host = parsed.hostname or ""
    
    # Create temporary files
    input_ext = os.path.splitext(parsed.path)[1] or ".tmp"
    input_file = None
    output_file = None
    
    try:
        # Create temp files
        input_fd, input_path = tempfile.mkstemp(suffix=input_ext, dir="/tmp")
        input_file = os.fdopen(input_fd, "wb")
        input_file.close()
        
        output_fd, output_path = tempfile.mkstemp(suffix=".wav", dir="/tmp")
        output_file = os.fdopen(output_fd, "wb")
        output_file.close()
        
        # Download audio
        download_audio(url_str, input_path)
        
        # Convert to WAV
        convert_to_wav(input_path, output_path)
        
        # Load audio and apply HPSS for improved accuracy
        loader = es.MonoLoader(filename=output_path, sampleRate=44100)
        audio = loader()
        
        # Apply Harmonic-Percussive Source Separation
        hpss = es.HPSS()
        harmonic, percussive = hpss(audio)
        
        # Compute BPM using percussive component
        bpm_normalized, bpm_raw, confidence = compute_bpm(percussive)
        
        # Compute key using harmonic component
        key, scale, key_confidence = compute_key(harmonic)
        
        return BPMResponse(
            bpm=int(round(bpm_normalized)),
            bpm_raw=round(bpm_raw, 2),
            confidence=round(confidence, 2),
            key=key,
            scale=scale,
            key_confidence=round(key_confidence, 2),
            source_url_host=source_host,
        )
    
    except HTTPException:
        raise
    except Exception as e:
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

