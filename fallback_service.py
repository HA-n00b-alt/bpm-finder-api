"""
BPM Fallback Service - Google Cloud Run microservice
High-accuracy, high-cost fallback service for low-confidence primary service results.
Accepts pre-processed audio files directly via file upload (batch processing).
"""
import io
import os
import tempfile
import asyncio
import time
import warnings
import multiprocessing as mp
import resource
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Tuple, List
from fastapi import FastAPI, UploadFile, HTTPException, Request
from pydantic import BaseModel
# Lazy load heavy libraries - only import when actually needed (reduces cold start time)
import aiofiles

app = FastAPI(title="BPM Fallback Service")
PROCESS_START = time.time()
_first_request = True

# ProcessPoolExecutor for CPU-bound work (bypasses GIL)
# Use env var or default to CPU count (clamped to >=1)
_process_pool = None

def get_process_pool():
    """Get or create the ProcessPoolExecutor singleton."""
    global _process_pool
    if _process_pool is None:
        max_workers = int(os.getenv("PROCESS_POOL_WORKERS", max(1, os.cpu_count() or 1)))
        # Use spawn to avoid fork-related crashes with native libs (librosa/essentia/numba).
        ctx = mp.get_context("spawn")
        _process_pool = ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx)
    return _process_pool


def log_telemetry(message: str) -> None:
    print(f"[TELEMETRY] {message}", flush=True)


def get_rss_kb() -> Optional[int]:
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return None


@app.on_event("startup")
async def startup_event():
    startup_s = time.time() - PROCESS_START
    log_telemetry(f"fallback_startup_s={startup_s:.2f}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup ProcessPoolExecutor on shutdown."""
    global _process_pool
    if _process_pool is not None:
        _process_pool.shutdown(wait=True)
        _process_pool = None


class FallbackResponse(BaseModel):
    """Response from fallback service.
    
    Fields are Optional to support selective processing:
    - If only BPM is requested: BPM fields populated, key fields None
    - If only key is requested: key fields populated, BPM fields None
    - If both are requested: all fields populated
    
    Note: If a field is requested (process_bpm/process_key=True), it will always
    be populated unless an error occurs (which raises HTTPException).
    """
    bpm_normalized: Optional[float] = None
    bpm_raw: Optional[float] = None
    confidence: Optional[float] = None
    key: Optional[str] = None
    scale: Optional[str] = None
    key_confidence: Optional[float] = None


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


def extract_key_from_chroma(chroma) -> Tuple[str, str, float]:
    """Extract musical key from chroma features using Krumhansl-Schmuckler algorithm.
    
    Improved version: uses chroma_cqt for better stability, drops low-energy frames.
    
    Args:
        chroma: Chroma features array (12 x time frames)
    
    Returns:
        (key, scale, confidence) tuple
    """
    # Lazy load numpy - only import when actually needed
    import numpy as np
    
    # Krumhansl-Schmuckler key profile templates (12-element arrays)
    major_template = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_template = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    
    # Normalize templates
    major_template = major_template / major_template.sum()
    minor_template = minor_template / minor_template.sum()
    
    # Average chroma across time, but drop low-energy frames for stability
    chroma_energy = np.sum(chroma, axis=0)
    energy_threshold = np.percentile(chroma_energy, 25)  # Keep top 75% of frames
    valid_frames = chroma_energy >= energy_threshold
    
    if np.any(valid_frames):
        chroma_profile = np.mean(chroma[:, valid_frames], axis=1)
    else:
        chroma_profile = np.mean(chroma, axis=1)
    
    chroma_profile = chroma_profile / (chroma_profile.sum() + 1e-10)  # Normalize
    
    # Calculate correlation with all 12 transpositions of Major and Minor templates
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    correlations = []
    
    for i, key in enumerate(keys):
        # Major template transposed to key i
        major_transposed = np.roll(major_template, i)
        major_corr = np.corrcoef(chroma_profile, major_transposed)[0, 1]
        if np.isnan(major_corr):
            major_corr = 0.0
        correlations.append((key, 'major', major_corr))
        
        # Minor template transposed to key i
        minor_transposed = np.roll(minor_template, i)
        minor_corr = np.corrcoef(chroma_profile, minor_transposed)[0, 1]
        if np.isnan(minor_corr):
            minor_corr = 0.0
        correlations.append((key, 'minor', minor_corr))
    
    # Find the highest correlation to determine final key and scale
    correlations.sort(key=lambda x: x[2], reverse=True)
    best_key, best_scale, best_corr = correlations[0]
    
    # Set confidence to the highest correlation score (normalized to 0-1 range)
    # Correlation ranges from -1 to 1, so we normalize: (corr + 1) / 2
    confidence = max(0.0, min(1.0, (best_corr + 1) / 2))
    
    return best_key, best_scale, confidence


def process_audio_array(
    audio,
    sr: int,
    process_bpm: bool,
    process_key: bool,
    telemetry: dict,
    overall_start: float
) -> dict:
    """Process a pre-loaded audio array."""
    try:
        # Suppress librosa warnings (PySoundFile fallback, deprecation warnings)
        # These are harmless but noisy in logs
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
            warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
            
            # Lazy load heavy libraries - only import when actually needed (reduces cold start time)
            telemetry["librosa_import_start"] = time.time()
            import librosa
            import numpy as np
            telemetry["librosa_import_end"] = time.time()
            telemetry["librosa_import_duration"] = telemetry["librosa_import_end"] - telemetry["librosa_import_start"]
        
        # Initialize response values
        bpm_normalized = None
        bpm_raw = None
        confidence = None
        key = None
        scale = None
        key_confidence = None
        
        # Ensure audio is a 1D float32 array
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)

        # Downsample for faster processing (prefer cheap decimation when possible)
        target_sr = int(os.getenv("FALLBACK_TARGET_SR", "22050"))
        if sr > target_sr:
            if sr % target_sr == 0:
                factor = sr // target_sr
                audio = audio[::factor]
                sr = target_sr
                log_telemetry(f"fallback_downsample method=decimate factor={factor} sr={sr}")
            else:
                resample_start = time.time()
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
                resample_s = time.time() - resample_start
                log_telemetry(f"fallback_downsample method=resample sr={sr} s={resample_s:.2f}")
        
        # Limit to reduce compute time
        max_seconds = float(os.getenv("FALLBACK_MAX_SECONDS", "30"))
        max_samples = int(max_seconds * sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
            print(f"[TELEMETRY] Audio trimmed to {max_seconds:.1f}s: {len(audio)} samples", flush=True)
        
        # Only perform HPSS if both BPM and key are needed (expensive operation)
        if process_bpm and process_key:
            try:
                telemetry["hpss_start"] = time.time()
                harmonic, percussive = librosa.effects.hpss(
                    audio,
                    margin=(1.0, 5.0),
                    kernel_size=31
                )
                telemetry["hpss_end"] = time.time()
                telemetry["hpss_duration"] = telemetry["hpss_end"] - telemetry["hpss_start"]
                print(f"[TELEMETRY] HPSS: {telemetry['hpss_duration']:.2f}s", flush=True)
            except Exception as e:
                telemetry["total_duration"] = time.time() - overall_start
                return {
                    "error": True,
                    "error_type": "hpss_failed",
                    "error_message": f"HPSS processing failed: {str(e)[:200]}",
                    "bpm_normalized": None,
                    "bpm_raw": None,
                    "confidence": None,
                    "key": None,
                    "scale": None,
                    "key_confidence": None,
                    "telemetry": telemetry,
                }
        elif process_bpm:
            percussive = audio
            harmonic = None
        elif process_key:
            harmonic = audio
            percussive = None
        else:
            telemetry["total_duration"] = time.time() - overall_start
            return {
                "bpm_normalized": None,
                "bpm_raw": None,
                "confidence": None,
                "key": None,
                "scale": None,
                "key_confidence": None,
                "telemetry": telemetry,
            }
        
        if process_bpm:
            try:
                telemetry["bpm_start"] = time.time()
                bpm_audio = percussive if percussive is not None else audio
                hop_length = int(os.getenv("FALLBACK_BPM_HOP_LENGTH", "1024"))
                tempo, beats = librosa.beat.beat_track(
                    y=bpm_audio,
                    sr=sr,
                    units='time',
                    hop_length=hop_length
                )
                bpm_raw = float(tempo)
                bpm_normalized = normalize_bpm(bpm_raw)
                telemetry["bpm_end"] = time.time()
                telemetry["bpm_duration"] = telemetry["bpm_end"] - telemetry["bpm_start"]
                print(f"[TELEMETRY] BPM processing: {telemetry['bpm_duration']:.2f}s", flush=True)
                
                if len(beats) > 0:
                    beat_intervals = np.diff(beats)
                    if len(beat_intervals) > 1:
                        interval_std = np.std(beat_intervals)
                        interval_mean = np.mean(beat_intervals)
                        if interval_mean > 0:
                            cv = interval_std / interval_mean
                            confidence = max(0.0, min(1.0, 1.0 - cv * 2))
                        else:
                            confidence = 0.5
                    else:
                        confidence = 0.5
                else:
                    confidence = 0.0
            except Exception as e:
                telemetry["total_duration"] = time.time() - overall_start
                return {
                    "error": True,
                    "error_type": "bpm_processing_failed",
                    "error_message": f"BPM processing failed: {str(e)[:200]}",
                    "bpm_normalized": None,
                    "bpm_raw": None,
                    "confidence": None,
                    "key": key if key else None,
                    "scale": scale if scale else None,
                    "key_confidence": key_confidence if key_confidence else None,
                    "telemetry": telemetry,
                }
        
        if process_key:
            try:
                telemetry["key_start"] = time.time()
                key_audio = harmonic if harmonic is not None else audio
                chroma = librosa.feature.chroma_cqt(
                    y=key_audio,
                    sr=sr,
                    hop_length=2048,
                    fmin=65.41,
                    n_octaves=5,
                    bins_per_octave=12
                )
                key, scale, key_confidence = extract_key_from_chroma(chroma)
                telemetry["key_end"] = time.time()
                telemetry["key_duration"] = telemetry["key_end"] - telemetry["key_start"]
                print(f"[TELEMETRY] Key processing: {telemetry['key_duration']:.2f}s", flush=True)
            except Exception as e:
                telemetry["total_duration"] = time.time() - overall_start
                return {
                    "error": True,
                    "error_type": "key_processing_failed",
                    "error_message": f"Key processing failed: {str(e)[:200]}",
                    "bpm_normalized": bpm_normalized,
                    "bpm_raw": round(bpm_raw, 2) if bpm_raw is not None else None,
                    "confidence": round(confidence, 2) if confidence is not None else None,
                    "key": None,
                    "scale": None,
                    "key_confidence": None,
                    "telemetry": telemetry,
                }
        
        telemetry["total_duration"] = time.time() - overall_start
        
        print(f"[TELEMETRY] Total processing time: {telemetry['total_duration']:.2f}s", flush=True)
        print(f"[TELEMETRY] Breakdown:", flush=True)
        if telemetry.get("librosa_import_duration"):
            print(f"  - Librosa import: {telemetry['librosa_import_duration']:.2f}s ({telemetry['librosa_import_duration']/telemetry['total_duration']*100:.1f}%)", flush=True)
        if telemetry.get("audio_load_duration"):
            print(f"  - Audio load: {telemetry['audio_load_duration']:.2f}s ({telemetry['audio_load_duration']/telemetry['total_duration']*100:.1f}%)", flush=True)
        if telemetry.get("hpss_duration"):
            print(f"  - HPSS: {telemetry['hpss_duration']:.2f}s ({telemetry['hpss_duration']/telemetry['total_duration']*100:.1f}%)", flush=True)
        if telemetry.get("bpm_duration"):
            print(f"  - BPM processing: {telemetry['bpm_duration']:.2f}s ({telemetry['bpm_duration']/telemetry['total_duration']*100:.1f}%)", flush=True)
        if telemetry.get("key_duration"):
            print(f"  - Key processing: {telemetry['key_duration']:.2f}s ({telemetry['key_duration']/telemetry['total_duration']*100:.1f}%)", flush=True)
        log_telemetry(
            f"fallback_processing payload={telemetry.get('payload_type', 'unknown')} "
            f"audio_load_s={telemetry.get('audio_load_duration')} "
            f"bpm_s={telemetry.get('bpm_duration')} "
            f"key_s={telemetry.get('key_duration')} "
            f"total_s={telemetry.get('total_duration')}"
        )
        
        return {
            "bpm_normalized": bpm_normalized,
            "bpm_raw": round(bpm_raw, 2) if bpm_raw is not None else None,
            "confidence": round(confidence, 2) if confidence is not None else None,
            "key": key,
            "scale": scale,
            "key_confidence": round(key_confidence, 2) if key_confidence is not None else None,
            "telemetry": telemetry,
        }
    except Exception as e:
        import traceback
        error_details = f"{str(e)}\n{traceback.format_exc()}"
        telemetry["total_duration"] = time.time() - overall_start
        print(f"[ERROR] Unexpected error in process_audio_array: {error_details[:500]}", flush=True)
        return {
            "error": True,
            "error_type": "unexpected_error",
            "error_message": f"Unexpected error: {str(e)[:200]}",
            "bpm_normalized": None,
            "bpm_raw": None,
            "confidence": None,
            "key": None,
            "scale": None,
            "key_confidence": None,
            "telemetry": telemetry,
        }


def process_single_audio(
    audio_file_path: str,
    process_bpm: bool,
    process_key: bool
) -> dict:
    """Process a single audio file from disk path.
    
    Args:
        audio_file_path: Path to audio file on disk
        process_bpm: Whether to process BPM
        process_key: Whether to process key
    
    Returns:
        dict with requested fields populated (to reduce pickling overhead)
        OR dict with "error": True if processing failed
    """
    # Telemetry tracking
    overall_start = time.time()
    telemetry = {
        "librosa_import_start": None,
        "librosa_import_end": None,
        "librosa_import_duration": None,
        "audio_load_start": None,
        "audio_load_end": None,
        "audio_load_duration": None,
        "hpss_start": None,
        "hpss_end": None,
        "hpss_duration": None,
        "bpm_start": None,
        "bpm_end": None,
        "bpm_duration": None,
        "key_start": None,
        "key_end": None,
        "key_duration": None,
        "total_duration": None,
        "payload_type": "file"
    }
    
    try:
        # Load audio - use Essentia MonoLoader for fast decoding (C++), then pass to librosa
        telemetry["audio_load_start"] = time.time()
        
        # Try to use Essentia MonoLoader for fast decoding (if available)
        audio = None
        sr = 44100
        use_essentia = False
        
        try:
            import essentia.standard as es
            # Use Essentia MonoLoader - fast C++ implementation using ffmpeg directly
            loader = es.MonoLoader(filename=audio_file_path, sampleRate=sr)
            audio = loader()
            use_essentia = True
            print(f"[TELEMETRY] Audio decoded via Essentia MonoLoader: {len(audio)} samples", flush=True)
        except ImportError:
            # Essentia not available - fall back to librosa
            print(f"[TELEMETRY] Essentia not available, using librosa", flush=True)
        except Exception as e:
            # Essentia failed - fall back to librosa
            print(f"[WARNING] Essentia MonoLoader failed: {str(e)[:100]}, falling back to librosa", flush=True)
        
        # If Essentia didn't work, use librosa (with warning suppression)
        if audio is None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
                warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
                # Load only first 60 seconds - sufficient for accurate BPM/key detection
                import librosa
                audio, sr = librosa.load(audio_file_path, sr=sr, mono=True, duration=60.0)
        
        telemetry["audio_load_end"] = time.time()
        telemetry["audio_load_duration"] = telemetry["audio_load_end"] - telemetry["audio_load_start"]
        load_method = "Essentia" if use_essentia else "librosa"
        print(f"[TELEMETRY] Audio load ({load_method}): {telemetry['audio_load_duration']:.2f}s ({len(audio)/sr:.1f}s)", flush=True)
        return process_audio_array(audio, sr, process_bpm, process_key, telemetry, overall_start)
    except Exception as e:
        # Catch any unexpected errors anywhere in the function
        # CRITICAL: Cannot raise HTTPException here - it's not picklable and breaks ProcessPoolExecutor
        import traceback
        error_details = f"{str(e)}\n{traceback.format_exc()}"
        telemetry["total_duration"] = time.time() - overall_start
        print(f"[ERROR] Unexpected error in process_single_audio: {error_details[:500]}", flush=True)
        return {
            "error": True,
            "error_type": "unexpected_error",
            "error_message": f"Unexpected error: {str(e)[:200]}",
            "bpm_normalized": None,
            "bpm_raw": None,
            "confidence": None,
            "key": None,
            "scale": None,
            "key_confidence": None,
            "telemetry": telemetry,
        }


def process_single_audio_npy(
    npy_bytes: bytes,
    sample_rate: int,
    process_bpm: bool,
    process_key: bool
) -> dict:
    """Process a single audio payload from a numpy .npy buffer."""
    overall_start = time.time()
    telemetry = {
        "librosa_import_start": None,
        "librosa_import_end": None,
        "librosa_import_duration": None,
        "audio_load_start": None,
        "audio_load_end": None,
        "audio_load_duration": None,
        "hpss_start": None,
        "hpss_end": None,
        "hpss_duration": None,
        "bpm_start": None,
        "bpm_end": None,
        "bpm_duration": None,
        "key_start": None,
        "key_end": None,
        "key_duration": None,
        "total_duration": None,
        "payload_type": "pcm_npy"
    }
    
    try:
        telemetry["audio_load_start"] = time.time()
        import numpy as np
        audio = np.load(io.BytesIO(npy_bytes), allow_pickle=False)
        telemetry["audio_load_end"] = time.time()
        telemetry["audio_load_duration"] = telemetry["audio_load_end"] - telemetry["audio_load_start"]
        print(f"[TELEMETRY] Audio load (PCM): {telemetry['audio_load_duration']:.2f}s ({len(audio)/sample_rate:.1f}s)", flush=True)
        
        return process_audio_array(audio, sample_rate, process_bpm, process_key, telemetry, overall_start)
    except Exception as e:
        import traceback
        error_details = f"{str(e)}\n{traceback.format_exc()}"
        telemetry["total_duration"] = time.time() - overall_start
        print(f"[ERROR] Unexpected error in process_single_audio_npy: {error_details[:500]}", flush=True)
        return {
            "error": True,
            "error_type": "unexpected_error",
            "error_message": f"Unexpected error: {str(e)[:200]}",
            "bpm_normalized": None,
            "bpm_raw": None,
            "confidence": None,
            "key": None,
            "scale": None,
            "key_confidence": None,
            "telemetry": telemetry,
        }


@app.get("/health")
async def health():
    """Health check endpoint for Cloud Run startup probe."""
    return {"ok": True, "service": "bpm-fallback-service"}


@app.post("/process_batch", response_model=List[FallbackResponse])
async def process_batch(
    request: Request
):
    """Batch process multiple audio files from memory.
    
    Accepts multipart form data with:
    - audio_files: List of uploaded audio files (multiple files with same name)
    - process_bpm_{i}: Whether to process BPM for file i (as string "true"/"false")
    - process_key_{i}: Whether to process key for file i (as string "true"/"false")
    - url_{i}: Optional URL for file i
    
    Returns:
        List of FallbackResponse, one per input file
    """
    global _first_request
    request_start = time.time()
    cold_start = False
    if _first_request:
        cold_start = True
        _first_request = False
    log_telemetry(f"fallback_request_start cold_start={str(cold_start).lower()}")
    temp_files = {}  # Track temp files for cleanup: {index: path}
    try:
        # Get form data
        form = await request.form()
        
        # Extract audio files and optional PCM payloads
        # Use multi_items() to get all items including duplicates
        audio_files = []
        pcm_files = {}
        sample_rates = {}
        form_items = list(form.multi_items())
        
        # Debug: log what we received
        print(f"DEBUG: Form multi_items count: {len(form_items)}")
        for key, value in form_items:
            if key == "audio_files":
                print(f"DEBUG: Found audio_files, type: {type(value)}")
                if hasattr(value, 'read'):
                    audio_files.append(value)
                elif isinstance(value, list):
                    audio_files.extend(value)
                else:
                    audio_files.append(value)
            elif key.startswith("pcm_npy_"):
                print(f"DEBUG: Found pcm_npy, key={key}, type={type(value)}")
                try:
                    idx = int(key.split("_")[-1])
                    if hasattr(value, 'read'):
                        pcm_files[idx] = value
                except Exception:
                    print(f"WARNING: Invalid pcm_npy key: {key}", flush=True)
            elif key.startswith("sample_rate_"):
                try:
                    idx = int(key.split("_")[-1])
                    sample_rates[idx] = int(value)
                except Exception:
                    print(f"WARNING: Invalid sample_rate value for {key}: {value}", flush=True)
            else:
                print(f"DEBUG: Form key: {key}, type: {type(value)}")
        
        print(f"DEBUG: Total audio_files extracted: {len(audio_files)}")
        print(f"DEBUG: Total pcm_npy extracted: {len(pcm_files)}")
        log_telemetry(
            f"fallback_request_inputs audio_files={len(audio_files)} pcm_npy={len(pcm_files)}"
        )
        
        if not audio_files and not pcm_files:
            raise HTTPException(
                status_code=400,
                detail="No audio files or PCM payloads provided in request"
            )
        
        # Extract processing flags
        process_flags = {}
        for key, value in form.items():
            if key.startswith("process_bpm_"):
                idx = int(key.split("_")[-1])
                process_flags.setdefault(idx, {})["bpm"] = value.lower() == "true"
            elif key.startswith("process_key_"):
                idx = int(key.split("_")[-1])
                process_flags.setdefault(idx, {})["key"] = value.lower() == "true"
        
        # Stream files to temp disk and collect paths (avoid loading full files into RAM)
        # This reduces memory usage and pickling overhead when passing to process pool
        
        async def stream_to_temp(audio_file, index, process_bpm, process_key):
            """Stream upload to temp file and return path."""
            temp_path = None
            stream_start = time.time()
            try:
                # Determine file extension from filename or content_type
                suffix = '.tmp'
                if audio_file.filename:
                    # Extract extension from filename
                    _, ext = os.path.splitext(audio_file.filename)
                    if ext:
                        suffix = ext
                elif hasattr(audio_file, 'content_type') and audio_file.content_type:
                    # Map content type to extension
                    content_type = audio_file.content_type
                    if content_type == 'audio/mpeg':
                        suffix = '.mp3'
                    elif content_type in ('audio/mp4', 'audio/aac'):
                        suffix = '.m4a'
                    elif content_type == 'audio/wav':
                        suffix = '.wav'
                
                # Create temp file
                temp_fd, temp_path = tempfile.mkstemp(suffix=suffix, dir='/tmp')
                os.close(temp_fd)  # Close fd, we'll write via path
                
                # Stream file content to disk (async I/O) - read in chunks to avoid loading full file into RAM
                total_bytes = 0
                write_start = time.time()
                async with aiofiles.open(temp_path, 'wb') as f:
                    # FastAPI UploadFile.read() can be called multiple times in a loop
                    while True:
                        chunk = await audio_file.read(8192)  # 8KB chunks
                        if not chunk:
                            break
                        await f.write(chunk)
                        total_bytes += len(chunk)
                write_duration = time.time() - write_start
                stream_duration = time.time() - stream_start
                
                # Verify file was written (not empty)
                if total_bytes == 0:
                    raise ValueError(f"Uploaded file for index {index} is empty")
                
                print(f"[TELEMETRY] File streaming for index {index}: {stream_duration:.2f}s (write: {write_duration:.2f}s, size: {total_bytes} bytes)", flush=True)
                
                # No fsync needed for ephemeral /tmp files
                return (index, temp_path, process_bpm, process_key)
            except Exception as e:
                # Cleanup on error
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except Exception:
                        pass
                # Log error for debugging
                import traceback
                print(f"ERROR: stream_to_temp failed for index {index}: {str(e)}")
                print(traceback.format_exc())
                raise
        
        async def read_pcm_npy(pcm_file, index, process_bpm, process_key):
            """Read PCM numpy payload into memory and return bytes."""
            try:
                payload = await pcm_file.read()
                if not payload:
                    raise ValueError(f"PCM payload for index {index} is empty")
                sr = sample_rates.get(index, 44100)
                print(f"[TELEMETRY] PCM payload for index {index}: {len(payload)} bytes (sr={sr})", flush=True)
                return (index, payload, sr, process_bpm, process_key)
            except Exception as e:
                import traceback
                print(f"ERROR: read_pcm_npy failed for index {index}: {str(e)}")
                print(traceback.format_exc())
                raise
        
        # Stream all inputs concurrently
        stream_tasks = []
        pcm_tasks = []
        for i, audio_file in enumerate(audio_files):
            if i in pcm_files:
                continue
            flags = process_flags.get(i, {})
            process_bpm = flags.get("bpm", True)
            process_key = flags.get("key", True)
            
            if not process_bpm and not process_key:
                future = asyncio.Future()
                future.set_result((i, None, False, False))
                stream_tasks.append(future)
            else:
                stream_tasks.append(stream_to_temp(audio_file, i, process_bpm, process_key))
        
        for i, pcm_file in pcm_files.items():
            flags = process_flags.get(i, {})
            process_bpm = flags.get("bpm", True)
            process_key = flags.get("key", True)
            
            if not process_bpm and not process_key:
                future = asyncio.Future()
                future.set_result((i, None, None, False, False))
                pcm_tasks.append(future)
            else:
                pcm_tasks.append(read_pcm_npy(pcm_file, i, process_bpm, process_key))
        
        # Wait for all streaming to complete
        streamed_files = await asyncio.gather(*stream_tasks, return_exceptions=True)
        pcm_items = await asyncio.gather(*pcm_tasks, return_exceptions=True)
        
        # Process files concurrently using ProcessPoolExecutor (CPU-bound work)
        process_pool = get_process_pool()
        loop = asyncio.get_event_loop()
        
        # Initialize results array with empty FallbackResponse objects upfront
        indices = set(process_flags.keys())
        indices.update(pcm_files.keys())
        if audio_files:
            indices.update(range(len(audio_files)))
        total_count = max(indices) + 1 if indices else len(audio_files)
        results = [FallbackResponse() for _ in range(total_count)]
        
        # Use dict to track processing tasks - maintains explicit index-to-future mapping
        # This prevents index misalignment that can occur with separate lists
        processing_tasks = {}
        
        for item in streamed_files:
            if isinstance(item, Exception):
                # Streaming failed - keep empty response (already initialized)
                print(f"[ERROR] Streaming failed: {str(item)[:200]}", flush=True)
                continue
                
            if isinstance(item, tuple):
                i, temp_path, process_bpm, process_key = item
                if temp_path is None:
                    # Skipped item (no processing requested) - keep empty response
                    # Use "SKIP" marker to distinguish from errors
                    continue
                else:
                    # Track temp file for cleanup
                    temp_files[i] = temp_path
                    # Submit CPU-bound work to process pool (pass path string, not bytes)
                    future = loop.run_in_executor(
                        process_pool,
                        process_single_audio,
                        temp_path,
                        process_bpm,
                        process_key
                    )
                    # Map index to future explicitly
                    processing_tasks[i] = future
        
        for item in pcm_items:
            if isinstance(item, Exception):
                print(f"[ERROR] PCM read failed: {str(item)[:200]}", flush=True)
                continue
            if isinstance(item, tuple):
                i, payload, sr, process_bpm, process_key = item
                if payload is None:
                    continue
                future = loop.run_in_executor(
                    process_pool,
                    process_single_audio_npy,
                    payload,
                    sr,
                    process_bpm,
                    process_key
                )
                processing_tasks[i] = future
        
        # Wait for all tasks to complete concurrently, handling errors per item
        if processing_tasks:
            # Create list of futures and track their indices
            task_indices = list(processing_tasks.keys())
            task_futures = [processing_tasks[idx] for idx in task_indices]
            completed = await asyncio.gather(*task_futures, return_exceptions=True)
            
            # Process results in order, handling exceptions and converting dicts to Pydantic models
            for idx, result in zip(task_indices, completed):
                if isinstance(result, Exception):
                    # Exception from process pool - log and keep empty response
                    import traceback
                    tb = ""
                    if getattr(result, "__traceback__", None):
                        tb = "".join(traceback.format_exception(type(result), result, result.__traceback__))
                    error_details = f"Process pool error: {str(result)}\n{tb}"
                    print(f"[ERROR] Process pool exception for index {idx}: {error_details[:500]}", flush=True)
                    # Keep empty response (already initialized)
                elif result is None:
                    # process_single_audio unexpectedly returned None - log and keep empty response
                    print(f"[ERROR] process_single_audio returned None for index {idx}", flush=True)
                    # Keep empty response (already initialized)
                elif isinstance(result, dict) and result.get("error"):
                    # Error dict returned from process_single_audio (e.g., audio load failed, BPM/key processing failed)
                    error_type = result.get("error_type", "unknown_error")
                    error_message = result.get("error_message", "Unknown error")
                    print(f"[ERROR] Processing error for index {idx} ({error_type}): {error_message}", flush=True)
                    # Keep empty response (already initialized)
                elif isinstance(result, dict):
                    # Success - convert dict to Pydantic model (built in parent process to reduce pickling)
                    # Only include fields that are in the FallbackResponse model
                    valid_fields = {
                        "bpm_normalized", "bpm_raw", "confidence", 
                        "key", "scale", "key_confidence"
                    }
                    result_clean = {k: v for k, v in result.items() if k in valid_fields}
                    try:
                        results[idx] = FallbackResponse(**result_clean)
                    except Exception as e:
                        # If validation fails, log and keep empty response
                        print(f"[ERROR] Failed to create FallbackResponse for index {idx}: {str(e)}", flush=True)
                        print(f"[ERROR] Result dict keys: {list(result.keys())}", flush=True)
                        print(f"[ERROR] Filtered keys: {list(result_clean.keys())}", flush=True)
                        # Keep empty response (already initialized)
                else:
                    # Unexpected result type - log and keep empty response
                    print(f"[ERROR] Unexpected result type for index {idx}: {type(result)}", flush=True)
                    # Keep empty response (already initialized)
        
    except Exception as e:
        # Log the full error for debugging
        import traceback
        error_msg = f"ERROR in process_batch: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        # Cleanup temp files on error
        for temp_path in temp_files.values():
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception:
                pass
        # Return error response
        raise HTTPException(
            status_code=500,
            detail=f"Batch processing failed: {str(e)[:200]}"
        )
    finally:
        # Cleanup temp files
        for temp_path in temp_files.values():
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception:
                pass  # Ignore cleanup errors
        request_s = time.time() - request_start
        log_telemetry(
            f"fallback_request_done_s={request_s:.2f} rss_kb={get_rss_kb()}"
        )
    
    return results
