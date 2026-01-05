"""
BPM Fallback Service - Google Cloud Run microservice
High-accuracy, high-cost fallback service for low-confidence primary service results.
Accepts pre-processed audio files directly via file upload (batch processing).
"""
import io
import os
import tempfile
import asyncio
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Tuple, List
from fastapi import FastAPI, UploadFile, HTTPException, Request
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="BPM Fallback Service")

# Import librosa at module level (required for Cloud Run startup)
import librosa

# ProcessPoolExecutor for CPU-bound work (bypasses GIL)
# Use 2 workers to match Cloud Run's 2 CPU cores
_process_pool = None

def get_process_pool():
    """Get or create the ProcessPoolExecutor singleton."""
    global _process_pool
    if _process_pool is None:
        _process_pool = ProcessPoolExecutor(max_workers=2)
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


def extract_key_from_chroma(chroma: np.ndarray) -> Tuple[str, str, float]:
    """Extract musical key from chroma features using Krumhansl-Schmuckler algorithm.
    
    Improved version: uses chroma_cqt for better stability, drops low-energy frames.
    
    Args:
        chroma: Chroma features array (12 x time frames)
    
    Returns:
        (key, scale, confidence) tuple
    """
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
    audio_content: bytes,
    process_bpm: bool,
    process_key: bool
) -> FallbackResponse:
    """Process a single audio file from memory.
    
    Args:
        audio_content: Audio file content as bytes
        process_bpm: Whether to process BPM
        process_key: Whether to process key
    
    Returns:
        FallbackResponse with requested fields populated
    """
    # Initialize response values
    bpm_normalized = None
    bpm_raw = None
    confidence = None
    key = None
    scale = None
    key_confidence = None
    
    # Load audio from memory
    # For compressed formats (MP3/AAC), librosa needs a file path, not BytesIO
    # Use a temporary file that gets cleaned up immediately
    temp_file = None
    temp_fd = None
    try:
        # Create temporary file
        temp_fd, temp_file = tempfile.mkstemp(suffix='.tmp', dir='/tmp')
        os.close(temp_fd)  # Close file descriptor, we'll write via path
        temp_fd = None  # Mark as closed
        
        # Write audio content to temporary file
        try:
            with open(temp_file, 'wb') as f:
                f.write(audio_content)
                f.flush()
                os.fsync(f.fileno())  # Ensure data is written to disk
        except Exception as e:
            # If write fails, ensure file is cleaned up
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass
            raise HTTPException(
                status_code=500,
                detail=f"Audio file write failed: {str(e)[:200]}"
            )
        
        # Load audio using librosa (can now detect format from file extension/path)
        try:
            audio, sr = librosa.load(temp_file, sr=44100, mono=True)
        except Exception as e:
            # If load fails, ensure file is cleaned up
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass
            raise HTTPException(
                status_code=500,
                detail=f"Audio loading failed: {str(e)[:200]}"
            )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Catch any other unexpected errors
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)[:200]}"
        )
    finally:
        # Clean up temporary file and file descriptor in all cases
        if temp_fd is not None:
            try:
                os.close(temp_fd)
            except Exception:
                pass
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception:
                pass  # Ignore cleanup errors
    
    # Only perform HPSS if we need either BPM or key
    if process_bpm or process_key:
        # Perform Harmonic-Percussive Source Separation
        harmonic, percussive = librosa.effects.hpss(audio)
    else:
        # Should not happen (validated at endpoint), but return empty
        return FallbackResponse()
    
    # BPM Extraction (Percussive component) - only if requested
    if process_bpm:
        try:
            tempo, beats = librosa.beat.beat_track(y=percussive, sr=sr)
            bpm_raw = float(tempo)
            bpm_normalized = normalize_bpm(bpm_raw)
            
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
                        # Apply cap to prevent over-reporting confidence
                        confidence = min(0.85, max(0.0, 1.0 - cv * 2))
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
    
    # Key Extraction (Harmonic component) - only if requested
    if process_key:
        try:
            # Use chroma_cqt for better stability (more robust to timbre variations)
            # Optimized parameters for speed: larger hop_length (less time resolution needed for key),
            # reduced n_octaves (key detection doesn't need full frequency range),
            # fmin set to C2 (65.41 Hz) to avoid processing very low frequencies
            chroma = librosa.feature.chroma_cqt(
                y=harmonic,
                sr=sr,
                hop_length=2048,      # Default: 512, 4x larger = ~4x faster, still sufficient for key detection
                fmin=65.41,            # C2 - reasonable lower bound for musical key detection
                n_octaves=5,           # Default: 6, reduced by 1 octave = faster, still covers full musical range
                bins_per_octave=12     # Default: 12, keep for accuracy
            )
            key, scale, key_confidence = extract_key_from_chroma(chroma)
        except Exception as e:
            # If key processing fails and it was requested, raise error
            raise HTTPException(
                status_code=500,
                detail=f"Key processing failed: {str(e)[:200]}"
            )
    
    return FallbackResponse(
        bpm_normalized=bpm_normalized,
        bpm_raw=round(bpm_raw, 2) if bpm_raw is not None else None,
        confidence=round(confidence, 2) if confidence is not None else None,
        key=key,
        scale=scale,
        key_confidence=round(key_confidence, 2) if key_confidence is not None else None,
    )


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
    
    # Read all file contents first (I/O bound, can be done concurrently)
    file_contents = []
    for i, audio_file in enumerate(audio_files):
        try:
            content = await audio_file.read()
            flags = process_flags.get(i, {})
            process_bpm = flags.get("bpm", True)
            process_key = flags.get("key", True)
            
            # Validate that at least one processing option is requested
            if not process_bpm and not process_key:
                file_contents.append((i, None, False, False))  # Empty response marker
            else:
                file_contents.append((i, content, process_bpm, process_key))
        except Exception as e:
            # If reading fails, mark for empty response
            file_contents.append((i, None, False, False))
    
    # Process files concurrently using ProcessPoolExecutor (CPU-bound work)
    # This bypasses the GIL and allows true parallelism on multiple CPU cores
    process_pool = get_process_pool()
    loop = asyncio.get_event_loop()
    
    # Create tasks for concurrent processing
    futures = []
    task_indices = []  # Track original index for each task
    for i, content, process_bpm, process_key in file_contents:
        if content is None:
            # Empty response - create a completed future with None
            future = asyncio.Future()
            future.set_result(FallbackResponse())
            futures.append(future)
            task_indices.append(i)
        else:
            # Submit CPU-bound work to process pool
            future = loop.run_in_executor(
                process_pool,
                process_single_audio,
                content,
                process_bpm,
                process_key
            )
            futures.append(future)
            task_indices.append(i)
    
    # Wait for all tasks to complete concurrently, handling errors per item
    results = [None] * len(audio_files)
    completed = await asyncio.gather(*futures, return_exceptions=True)
    
    # Process results in order, handling exceptions
    for idx, result in zip(task_indices, completed):
        if isinstance(result, HTTPException):
            # Re-raise HTTP exceptions (these are expected errors)
            raise result
        elif isinstance(result, Exception):
            # Per-item error handling: return empty response for failed items
            results[idx] = FallbackResponse()
        else:
            results[idx] = result
    
    return results
