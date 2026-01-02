"""
BPM Finder API - Google Cloud Run microservice
Computes BPM from 30s audio preview URLs.
"""
import os
import tempfile
import subprocess
from typing import Optional, Tuple
from urllib.parse import urlparse
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl, field_validator
import essentia.standard as es

app = FastAPI(title="BPM Finder API")

# Allowed host suffixes for SSRF protection
ALLOWED_HOST_SUFFIXES = [
    ".mzstatic.com",  # Apple previews
    ".scdn.co",  # Spotify previews
    ".deezer.com",
    ".dzcdn.net",  # Deezer previews
]

# Download limits
CONNECT_TIMEOUT = 5.0  # seconds
TOTAL_TIMEOUT = 20.0  # seconds
MAX_SIZE = 10 * 1024 * 1024  # 10MB


class BPMRequest(BaseModel):
    url: HttpUrl

    @field_validator("url")
    @classmethod
    def validate_url(cls, v):
        """Validate URL scheme and host."""
        url_str = str(v)
        if not url_str.startswith("https://"):
            raise ValueError("Only HTTPS URLs are allowed")
        
        parsed = urlparse(url_str)
        host = parsed.hostname or ""
        
        # Check if host ends with any allowed suffix
        allowed = any(host.endswith(suffix) for suffix in ALLOWED_HOST_SUFFIXES)
        if not allowed:
            raise ValueError(
                f"Host must end with one of: {', '.join(ALLOWED_HOST_SUFFIXES)}"
            )
        
        return v


class BPMResponse(BaseModel):
    bpm: int
    bpm_raw: float
    confidence: float
    source_url_host: str


def validate_redirect_url(url: str) -> bool:
    """Validate that a redirect URL is allowed."""
    parsed = urlparse(url)
    host = parsed.hostname or ""
    return any(host.endswith(suffix) for suffix in ALLOWED_HOST_SUFFIXES)


def download_audio(url: str, output_path: str) -> None:
    """Download audio file with SSRF protection and size limits."""
    with httpx.Client(
        timeout=httpx.Timeout(connect=CONNECT_TIMEOUT, read=TOTAL_TIMEOUT),
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
                if not redirect_url.startswith("https://"):
                    raise HTTPException(
                        status_code=400,
                        detail="Redirect to non-HTTPS URL not allowed"
                    )
                
                if not validate_redirect_url(redirect_url):
                    raise HTTPException(
                        status_code=400,
                        detail="Redirect to non-allowed host detected"
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
            
            # Check final URL
            final_url = str(response.url)
            if not validate_redirect_url(final_url):
                raise HTTPException(
                    status_code=400,
                    detail="Final URL host not allowed"
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


def compute_bpm(wav_path: str) -> Tuple[float, float]:
    """Compute BPM using Essentia RhythmExtractor2013.
    
    Returns:
        (bpm, confidence) tuple
    """
    loader = es.MonoLoader(filename=wav_path)
    audio = loader()
    
    rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
    bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)
    
    return float(bpm), float(beats_confidence)


def normalize_bpm(bpm: float) -> int:
    """Normalize BPM to reasonable range (70-200) by doubling/halving."""
    normalized = bpm
    
    # Double if too slow
    while normalized < 70:
        normalized *= 2
    
    # Halve if too fast
    while normalized > 200:
        normalized /= 2
    
    return round(normalized)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"ok": True}


@app.post("/bpm", response_model=BPMResponse)
async def compute_bpm_from_url(request: BPMRequest):
    """Compute BPM from audio preview URL."""
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
        
        # Compute BPM
        bpm_raw, confidence = compute_bpm(output_path)
        bpm_normalized = normalize_bpm(bpm_raw)
        
        return BPMResponse(
            bpm=bpm_normalized,
            bpm_raw=round(bpm_raw, 2),
            confidence=round(confidence, 2),
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

