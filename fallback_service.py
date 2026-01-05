"""
BPM Fallback Service - Google Cloud Run microservice
High-accuracy, high-cost fallback service for low-confidence primary service results.
Accepts pre-processed WAV audio files directly via file upload.
"""
import os
import tempfile
from typing import Optional, Tuple, Literal
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
import numpy as np

app = FastAPI(title="BPM Fallback Service")

# Import librosa at module level (required for Cloud Run startup)
# This ensures the import happens during container startup
import librosa


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


def extract_key_from_chroma(chroma: np.ndarray) -> Tuple[str, str, float]:
    """Extract musical key from chroma features using Krumhansl-Schmuckler algorithm.
    
    Implements complete template matching with all 12 transpositions of Major and Minor templates.
    
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
    
    # Average chroma across time to get 12-element profile
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


@app.get("/health")
async def health():
    """Health check endpoint for Cloud Run startup probe."""
    return {"ok": True, "service": "bpm-fallback-service"}


@app.post("/bpm", response_model=FallbackResponse)
async def process_bpm(
    audio_file: UploadFile = File(...),
    url: Optional[str] = Form(None),
    process_bpm: bool = Form(True),
    process_key: bool = Form(True)
):
    """Process BPM and/or key from uploaded WAV audio file using high-accuracy librosa methods.
    
    Args:
        audio_file: Uploaded WAV audio file (mono, 44100Hz)
        url: Optional original URL for logging/debug purposes
        process_bpm: Whether to process BPM (default: True)
        process_key: Whether to process key (default: True)
    
    Returns:
        FallbackResponse with BPM and/or key analysis (only requested fields populated)
    """
    # Validate that at least one processing option is requested
    if not process_bpm and not process_key:
        raise HTTPException(
            status_code=400,
            detail="At least one of process_bpm or process_key must be True"
        )
    
    temp_path = None
    
    try:
        # Read and save file with robust file handling
        content = await audio_file.read()
        
        # Create temporary file with proper context management
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav', dir='/tmp')
        try:
            with os.fdopen(temp_fd, 'wb') as f:
                f.write(content)
                f.flush()  # Ensure data is written to disk
                os.fsync(f.fileno())  # Force write to disk
        except Exception as e:
            # Clean up on write error
            try:
                os.unlink(temp_path)
            except Exception:
                pass
            raise HTTPException(status_code=500, detail=f"File write failed: {str(e)}")
        
        # Load audio with librosa (guaranteed to be mono, 44100Hz WAV)
        audio, sr = librosa.load(temp_path, sr=44100, mono=True)
        
        # Initialize response values
        bpm_normalized = None
        bpm_raw = None
        confidence = None
        key = None
        scale = None
        key_confidence = None
        
        # Perform Harmonic-Percussive Source Separation
        # We always need HPSS if either BPM or key is requested (already validated above)
        harmonic, percussive = librosa.effects.hpss(audio)
        
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
                chroma = librosa.feature.chroma_stft(y=harmonic, sr=sr)
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
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)[:200]}"
        )
    
    finally:
        # Cleanup temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass  # Ignore cleanup errors

