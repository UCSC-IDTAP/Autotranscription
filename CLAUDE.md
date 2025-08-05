# Auto-Transcription Pipeline for Indian Classical Music

## Project Overview
This project implements an automatic transcription pipeline specifically designed for Indian classical music. The pipeline processes audio files through multiple stages to extract pitch information and generate visual/audio representations of the melodic content.

## Development Philosophy
This project follows a modular architecture prioritizing:
- **Separation of Concerns**: Each module handles a single responsibility
- **Maintainability**: Easy to find, understand, and modify specific functionality
- **Readability**: Clear, focused modules with reasonable file sizes
- **Testability**: Components can be tested and developed in isolation
- **Reusability**: Modules can be imported and used independently

## Architecture

### Modular Components

1. **AutoTranscriptionPipeline** (`src/autotranscription/pipeline.py`)
   - Main orchestrator for the entire transcription process (259 lines)
   - Manages workspace organization and file tracking
   - Implements resumable pipeline with stage completion tracking
   - Coordinates all specialized modules

2. **AudioSeparator** (`src/autotranscription/audio_separator.py`)
   - Uses Demucs models (htdemucs, htdemucs_ft) for vocal/instrumental separation
   - Separates input audio into vocals and instrumental tracks
   - Creates organized output structure with timing information

3. **PitchExtractor** (`src/autotranscription/pitch_extraction.py`)
   - Multi-algorithm pitch estimation (Essentia, CREPE, SWIPE, PyWorld)
   - Algorithm-specific confidence thresholds and optimization
   - Handles data persistence and statistical analysis

4. **TonicEstimator** (`src/autotranscription/tonic_estimation.py`)
   - Tonic frequency estimation using Essentia's TonicIndianArtMusic
   - Comparison between original and instrumental tracks
   - Western note name conversion utilities

5. **SargamVisualizer** (`src/autotranscription/visualization.py`)
   - Matplotlib visualizations with Indian classical music notation
   - Sargam pitch labels (S r R g G m M P d D n N) relative to tonic
   - Logarithmic frequency scaling with octave diacritical marks

6. **AudioProcessor** (`src/autotranscription/audio_processing.py`)
   - Audio utilities for envelope extraction and sine wave generation
   - Amplitude envelope preservation for realistic audio outputs
   - Phase-continuous sine wave synthesis

7. **RagaEstimator** (`src/autotranscription/raga_estimation.py`)
   - TDMS (Time-Delayed Melody Surfaces) implementation for raga recognition
   - Phase space embedding and delay coordinate generation
   - Rule-based classification with confidence scoring for 15+ ragas
   - Clean output format with candidate probabilities

### Pipeline Stages

#### Stage 1: Audio Separation
- Uses Demucs to separate vocals from instrumental tracks
- Creates `vocals.wav` and `no_vocals.wav` files
- Stores separation metadata and timing

#### Stage 2: Tonic Estimation
- Uses Essentia's `TonicIndianArtMusic` algorithm
- Estimates tonic frequency on both original and instrumental tracks
- Compares results and calculates frequency/cents differences

#### Stage 3: Pitch Trace Extraction
- **Primary Algorithm**: Essentia's `PredominantPitchMelodia` (optimized for quiet sections)
- **Optional Algorithms**: CREPE, SWIPE, PyWorld
- Parameters tuned for Indian classical music (55-1760 Hz range)
- Confidence-based filtering with algorithm-specific thresholds

#### Stage 4: Raga Estimation using TDMS
- **Time-Delayed Melody Surfaces**: Phase space embedding of pitch sequences
- **Delay Coordinates**: 3D vectors capturing melodic patterns and temporal relationships
- **Tonic Normalization**: Pitch data converted to cents relative to estimated tonic
- **Pattern Matching**: Rule-based classification against known raga characteristics
- **Support for 15 ragas**: Yaman, Bhairav, Bhimpalasi, Malkauns, Bageshri, etc.

### Key Features

#### Workspace Management
- Organized directory structure under `pipeline_workspace/`
- Unique numeric IDs for each audio file
- JSON metadata tracking for resumable processing
- Stage completion tracking to avoid reprocessing

#### Directory Structure
```
pipeline_workspace/
├── audio/           # Copied input files
├── metadata/        # JSON files with processing metadata
├── separated/       # Demucs output (vocals/instrumental)
├── data/
│   └── pitch_traces/  # Numpy arrays with pitch data
├── visualizations/  # PNG plots comparing algorithms
└── audio_outputs/   # Sine wave reconstructions
```

#### Pitch Extraction Optimization
- **Essentia**: Optimized for quiet sections with `voicingTolerance=0.4`, `guessUnvoiced=True`, `magnitudeThreshold=20`
- **Confidence Thresholds**: Essentia (0.05), CREPE (0.7), SWIPE (0.5), PyWorld (0.5)
- **Frequency Range**: 55-1760 Hz (appropriate for Indian classical vocals)

#### Visualization & Audio Output
- **Sargam Visualization**: Single-axis plots with Indian classical music notation
  - Y-axis shows sargam labels (S r R g G m M P d D n N) relative to detected tonic
  - Range: 1 octave below to 2 octaves above tonic (48 chromatic divisions)
  - Octave notation with diacritical marks: dots above (upper), dots below (lower)
  - Clean y-axis without scientific notation, only musical labels
- **Audio Outputs**: Sine wave reconstructions with original amplitude envelope applied
- **Disconnected Segments**: Low-confidence regions create gaps (no interpolation)

## Usage Patterns

### Basic Usage
```python
from autotranscription.pipeline import run_pipeline

# Process a single file
results = run_pipeline("audio_file.mp3", workspace_dir="my_workspace")

# Resume processing (automatically detects existing work)
results = run_pipeline("audio_file.mp3", workspace_dir="my_workspace")
```

### Advanced Usage
```python
# Use specific algorithms
pipeline = AutoTranscriptionPipeline(
    workspace_dir="workspace",
    algorithms=['essentia', 'crepe']  # Default: ['essentia']
)
results = pipeline.run("audio_file.mp3")

# List processed files
audio_files = pipeline.list_audio_files()
```

## Dependencies
- **Audio Processing**: demucs, essentia, librosa, soundfile
- **Pitch Estimation**: crepe, pysptk (SWIPE), pyworld
- **Scientific**: numpy, scipy, matplotlib
- **Environment**: pipenv for dependency management

## Current State
- ✅ **Modular Architecture**: Refactored into focused, maintainable modules
- ✅ **Complete Pipeline**: Audio separation, tonic estimation, pitch extraction, and raga estimation
- ✅ **TDMS Raga Recognition**: Time-Delayed Melody Surfaces for raga classification
- ✅ **Sargam Visualization**: Indian classical music notation with clean y-axis
- ✅ **Indian Classical Optimization**: Tuned for characteristics of this musical tradition
- ✅ **Resumable Processing**: Metadata tracking with stage completion
- ✅ **Multi-Algorithm Support**: Essentia, CREPE, SWIPE, PyWorld with confidence filtering
- ✅ **Professional Output**: High-quality visualizations and audio reconstructions
- ✅ **Organized Workspace**: Structured file management and data persistence

## Testing
- Uses 1-minute excerpt of "Kishori Amonkar - Babul Mora" for testing
- Pipeline creates comprehensive outputs in `pipeline_workspace/`
- Visualizations and sine wave reconstructions available for quality assessment

## Notes for Future Development
- **Modular Design**: Each component can be developed and tested independently
- **Algorithm Selection**: Default uses Essentia for efficiency; others available for comparison
- **Configuration**: Confidence thresholds and parameters tuned for Indian classical music
- **Extensibility**: New algorithms or visualizations can be added to respective modules
- **Code Quality**: Maintain separation of concerns and reasonable file sizes
- **Documentation**: Each module is self-documenting with clear responsibilities

## Commands to Remember
- **Test Pipeline**: `pipenv run python src/autotranscription/pipeline.py`
- **Test Audio Separator**: `pipenv run python src/autotranscription/audio_separator.py`
- **Environment Setup**: `pipenv install` then `pipenv shell`