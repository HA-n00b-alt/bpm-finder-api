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
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Tuple, List
from fastapi import FastAPI, UploadFile, HTTPException, Request
from pydantic import BaseModel
# Lazy load heavy libraries - only import when actually needed (reduces cold start time)
import aiofiles

app = FastAPI(title="BPM Fallback Service")

# ProcessPoolExecutor for CPU-bound work (bypasses GIL)
# Use env var or default to CPU count (clamped to >=1)
_process_pool = None

def get_process_pool():
    """Get or create the ProcessPoolExecutor singleton."""
    global _process_pool
    if _process_pool is None:
        max_workers = int(os.getenv("PROCESS_POOL_WORKERS", max(1, os.cpu_count() or 1)))
        _process_pool = ProcessPoolExecutor(max_workers=max_workers)
    return _process_pool

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
    """
    # Telemetry tracking
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
        "total_duration": None
    }
    overall_start = time.time()
    
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
    
    # Load audio using librosa
    # OPTIMIZATION: Limit duration to 60 seconds - enough for BPM/key detection, dramatically reduces load time
    # For preview files, this is typically sufficient. For full tracks, we only need a sample.
    try:
        telemetry["audio_load_start"] = time.time()
        # Load only first 60 seconds - sufficient for accurate BPM/key detection
        # This can reduce load time from 40s to ~5-10s for long files
        audio, sr = librosa.load(audio_file_path, sr=44100, mono=True, duration=60.0)
        telemetry["audio_load_end"] = time.time()
        telemetry["audio_load_duration"] = telemetry["audio_load_end"] - telemetry["audio_load_start"]
        print(f"[TELEMETRY] Audio load: {telemetry['audio_load_duration']:.2f}s (limited to 60s)", flush=True)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Audio loading failed: {str(e)[:200]}"
        )
    
    # Only perform HPSS if both BPM and key are needed (expensive operation)
    # If only one is needed, use the original audio directly
    if process_bpm and process_key:
        # Both needed: perform HPSS once
        # OPTIMIZATION: Use faster HPSS parameters
        # - margin: Controls separation strength (default 1.0), lower = faster but less separation
        # - kernel_size: Size of median filter (default 31), smaller = faster
        telemetry["hpss_start"] = time.time()
        harmonic, percussive = librosa.effects.hpss(
            audio,
            margin=(1.0, 5.0),  # Default, but explicit
            kernel_size=31  # Default, but explicit
        )
        telemetry["hpss_end"] = time.time()
        telemetry["hpss_duration"] = telemetry["hpss_end"] - telemetry["hpss_start"]
        print(f"[TELEMETRY] HPSS: {telemetry['hpss_duration']:.2f}s", flush=True)
    elif process_bpm:
        # Only BPM: use original audio (beat tracking works on full audio)
        percussive = audio
        harmonic = None
    elif process_key:
        # Only key: use original audio (chroma works on full audio)
        harmonic = audio
        percussive = None
    else:
        # Should not happen (validated at endpoint), but return empty dict
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
    
    # BPM Extraction - only if requested
    if process_bpm:
        try:
            telemetry["bpm_start"] = time.time()
            # Use percussive component if HPSS was done, otherwise use original audio
            bpm_audio = percussive if percussive is not None else audio
            # OPTIMIZATION: Use faster parameters for beat tracking
            # - units='time' is faster than default 'frames'
            # - start_bpm and std_bpm can help if we have a rough estimate, but we don't
            # - hop_length=512 is default, but we can use larger for speed (less accuracy)
            # For now, keep default accuracy but optimize HPSS if both are needed
            tempo, beats = librosa.beat.beat_track(
                y=bpm_audio,
                sr=sr,
                units='time',  # Return beats in time (seconds) - slightly faster
                hop_length=512  # Default, but explicit for clarity
            )
            bpm_raw = float(tempo)
            bpm_normalized = normalize_bpm(bpm_raw)
            telemetry["bpm_end"] = time.time()
            telemetry["bpm_duration"] = telemetry["bpm_end"] - telemetry["bpm_start"]
            print(f"[TELEMETRY] BPM processing: {telemetry['bpm_duration']:.2f}s", flush=True)
            
            # Calculate confidence based on beat tracking quality
            # Use the consistency of detected beats as confidence indicator
            if len(beats) > 0:
                # Calculate confidence from beat consistency
                beat_intervals = np.diff(beats)
                if len(beat_intervals) > 1:
                    # Lower variance in beat intervals = higher confidence
                    interval_std = np.std(beat_intervals)
                    interval_mean = np.mean(beat_intervals)
                    if interval_mean > 0:
                        cv = interval_std / interval_mean  # Coefficient of variation
                        # Calculate confidence from beat consistency
                        # Lower coefficient of variation = more consistent beats = higher confidence
                        # Formula: confidence = 1.0 - cv * 2
                        # When cv = 0 (perfect consistency), confidence = 1.0
                        # When cv = 0.5 (50% variation), confidence = 0.0
                        confidence = max(0.0, min(1.0, 1.0 - cv * 2))
                    else:
                        confidence = 0.5
                else:
                    confidence = 0.5
            else:
                confidence = 0.0
        except Exception as e:
            # If BPM processing fails and it was requested, raise error
            raise HTTPException(
                status_code=500,
                detail=f"BPM processing failed: {str(e)[:200]}"
            )
    
    # Key Extraction - only if requested
    if process_key:
        try:
            telemetry["key_start"] = time.time()
            # Use harmonic component if HPSS was done, otherwise use original audio
            key_audio = harmonic if harmonic is not None else audio
            # Use chroma_cqt for better stability (more robust to timbre variations)
            # Optimized parameters for speed: larger hop_length (less time resolution needed for key),
            # reduced n_octaves (key detection doesn't need full frequency range),
            # fmin set to C2 (65.41 Hz) to avoid processing very low frequencies
            chroma = librosa.feature.chroma_cqt(
                y=key_audio,
                sr=sr,
                hop_length=2048,      # Default: 512, 4x larger = ~4x faster, still sufficient for key detection
                fmin=65.41,            # C2 - reasonable lower bound for musical key detection
                n_octaves=5,           # Default: 6, reduced by 1 octave = faster, still covers full musical range
                bins_per_octave=12     # Default: 12, keep for accuracy
            )
            key, scale, key_confidence = extract_key_from_chroma(chroma)
            telemetry["key_end"] = time.time()
            telemetry["key_duration"] = telemetry["key_end"] - telemetry["key_start"]
            print(f"[TELEMETRY] Key processing: {telemetry['key_duration']:.2f}s", flush=True)
        except Exception as e:
            # If key processing fails and it was requested, raise error
            raise HTTPException(
                status_code=500,
                detail=f"Key processing failed: {str(e)[:200]}"
            )
    
    # Calculate total duration
    telemetry["total_duration"] = time.time() - overall_start
    
    # Log comprehensive telemetry
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
    
    # Return plain dict to reduce pickling overhead (Pydantic model built in parent process)
    return {
        "bpm_normalized": bpm_normalized,
        "bpm_raw": round(bpm_raw, 2) if bpm_raw is not None else None,
        "confidence": round(confidence, 2) if confidence is not None else None,
        "key": key,
        "scale": scale,
        "key_confidence": round(key_confidence, 2) if key_confidence is not None else None,
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
    temp_files = {}  # Track temp files for cleanup: {index: path}
    try:
        # Get form data
        form = await request.form()
        
        # Extract audio files - FastAPI handles multiple files with same name
        # Use multi_items() to get all items including duplicates
        audio_files = []
        form_items = list(form.multi_items())
        
        # Debug: log what we received
        print(f"DEBUG: Form multi_items count: {len(form_items)}")
        for key, value in form_items:
            if key == "audio_files":
                print(f"DEBUG: Found audio_files, type: {type(value)}")
                # value should be an UploadFile object
                if hasattr(value, 'read'):  # It's an UploadFile
                    audio_files.append(value)
                elif isinstance(value, list):
                    audio_files.extend(value)
                else:
                    audio_files.append(value)
            else:
                print(f"DEBUG: Form key: {key}, type: {type(value)}")
        
        print(f"DEBUG: Total audio_files extracted: {len(audio_files)}")
        
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
        
        # Stream all files concurrently
        stream_tasks = []
        for i, audio_file in enumerate(audio_files):
            flags = process_flags.get(i, {})
            process_bpm = flags.get("bpm", True)
            process_key = flags.get("key", True)
            
            # Validate that at least one processing option is requested
            if not process_bpm and not process_key:
                # Empty response marker - create completed future
                future = asyncio.Future()
                future.set_result((i, None, False, False))
                stream_tasks.append(future)
            else:
                stream_tasks.append(stream_to_temp(audio_file, i, process_bpm, process_key))
        
        # Wait for all streaming to complete
        streamed_files = await asyncio.gather(*stream_tasks, return_exceptions=True)
        
        # Process files concurrently using ProcessPoolExecutor (CPU-bound work)
        # This bypasses the GIL and allows true parallelism on multiple CPU cores
        process_pool = get_process_pool()
        loop = asyncio.get_event_loop()
        
        # Create tasks for concurrent processing
        futures = []
        task_indices = []  # Track original index for each task
        for idx, item in enumerate(streamed_files):
            if isinstance(item, Exception):
                # Streaming failed - return empty response
                future = asyncio.Future()
                future.set_result({"error": True})
                futures.append(future)
                task_indices.append(idx)
                continue
                
            if isinstance(item, tuple):
                i, temp_path, process_bpm, process_key = item
                if temp_path is None:
                    # Empty response - create a completed future
                    future = asyncio.Future()
                    future.set_result({"error": True})
                    futures.append(future)
                    task_indices.append(i)
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
                    futures.append(future)
                    task_indices.append(i)
        
        # Wait for all tasks to complete concurrently, handling errors per item
        results = [None] * len(audio_files)
        completed = await asyncio.gather(*futures, return_exceptions=True)
        
        # Process results in order, handling exceptions and converting dicts to Pydantic models
        for idx, result in zip(task_indices, completed):
            if isinstance(result, HTTPException):
                # Re-raise HTTP exceptions (these are expected errors)
                raise result
            elif isinstance(result, Exception) or (isinstance(result, dict) and result.get("error")):
                # Per-item error handling: return empty response for failed items
                results[idx] = FallbackResponse()
            elif isinstance(result, dict):
                # Convert dict to Pydantic model (built in parent process to reduce pickling)
                results[idx] = FallbackResponse(**result)
            else:
                results[idx] = result
        
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
    
    return results
