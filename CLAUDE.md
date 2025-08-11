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

8. **PitchSegmenter** (`src/autotranscription/pitch_segmentation.py`)
   - Two-stage pitch segmentation using local minima/maxima detection
   - Sophisticated vibrato detection with frequency range and oscillation pattern analysis
   - Intelligent segment classification (fixed, moving, vibrato, silence)
   - Post-processing to reclassify false vibrato as fixed pitch segments
   - Comprehensive segment metadata including frequency statistics

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

#### Stage 5: Pitch Segmentation
- **Two-Stage Approach**: Fine-grained segmentation followed by intelligent merging
  - **Stage 1**: Local minima/maxima detection creates fine segments at every pitch direction change
  - **Stage 2**: Vibrato detection and merging of consecutive segments with same target pitch
- **Sophisticated Vibrato Detection**: Multi-criteria analysis including:
  - Minimum 3-6 consecutive short segments (< 0.15-0.20s each)
  - Frequency range within ~1.5 semitones (0.125 log units)
  - True oscillation pattern with ≥4 direction changes
  - Mix of fixed and moving segment types
- **Vibrato Reclassification**: Post-processing step that converts vibrato segments to fixed pitch when:
  - All frequencies in segment data stay within threshold of center pitch
  - Prevents false vibrato detections on stable pitch regions
- **Comprehensive Metadata**: Each segment includes detailed frequency statistics (min/max freq, log freq spans)
- **IDTAP Integration**: Uses raga-specific target pitches for accurate classification

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
│   ├── pitch_traces/  # Numpy arrays with pitch data
│   └── segmentation/  # Individual segment data arrays for each stage
├── visualizations/  # PNG plots comparing algorithms and segmentation stages
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
- ✅ **Complete Pipeline**: Audio separation, tonic estimation, pitch extraction, raga estimation, and pitch segmentation
- ✅ **Advanced Pitch Segmentation**: Two-stage min/max approach with sophisticated vibrato detection
- ✅ **Intelligent Vibrato Detection**: Multi-criteria analysis with frequency range and oscillation pattern detection
- ✅ **Vibrato Reclassification**: Post-processing to correct false vibrato detections as fixed pitch segments
- ✅ **Comprehensive Segment Metadata**: Detailed frequency statistics and classification data for each segment
- ✅ **TDMS Raga Recognition**: Time-Delayed Melody Surfaces for raga classification
- ✅ **IDTAP Integration**: Uses raga-specific target pitches for accurate pitch analysis and segmentation
- ✅ **Sargam Visualization**: Indian classical music notation with clean y-axis
- ✅ **Multi-Stage Visualization**: Comparative plots showing Stage 1 and Stage 2 segmentation results
- ✅ **Indian Classical Optimization**: Tuned for characteristics of this musical tradition
- ✅ **Resumable Processing**: Metadata tracking with stage completion
- ✅ **Multi-Algorithm Support**: Essentia, CREPE, SWIPE, PyWorld with confidence filtering
- ✅ **Professional Output**: High-quality visualizations and audio reconstructions
- ✅ **Organized Workspace**: Structured file management and data persistence

## Testing
- Uses 1-minute excerpt of "Kishori Amonkar - Babul Mora" for testing
- Pipeline creates comprehensive outputs in `pipeline_workspace/`
- Multiple visualization types: pitch traces, segmentation stages, comparative analysis
- Individual segment data arrays saved for detailed analysis
- Comprehensive JSON metadata with frequency statistics and segment classifications
- Visualizations and sine wave reconstructions available for quality assessment

## Recent Improvements (2025)

### Vibrato Detection Algorithm Enhancement
The pitch segmentation module has undergone significant improvements to better detect and classify vibrato in Indian classical music:

#### Key Algorithm Changes:
1. **Multi-Criteria Vibrato Detection** (`_find_vibrato_sequence`):
   - **Frequency Range Check**: Uses actual segment data min/max rather than endpoint frequencies
   - **Oscillation Pattern Detection**: Requires ≥4 direction changes for true vibrato
   - **Greedy Sequential Search**: Finds the earliest valid vibrato subsequence and commits immediately
   - **Flexible Segment Length**: 3-6+ consecutive short segments (< 0.15-0.20s each)

2. **Vibrato Reclassification** (`_reclassify_vibrato_within_threshold`):
   - **Post-Processing Step**: Analyzes detected vibrato segments after initial classification
   - **Statistical Validation**: Checks if min/max log frequencies stay within threshold of center pitch
   - **False Positive Correction**: Converts stable-frequency "vibrato" back to fixed pitch segments
   - **Preserves Metadata**: Maintains all timing and frequency information during reclassification

#### Technical Details:
- **Frequency Range Threshold**: 0.125 log units (~1.5 semitones) maximum for vibrato sequences
- **Oscillation Detection**: Uses endpoint frequencies to track actual pitch movement patterns
- **Center Frequency Calculation**: Statistical analysis to determine vibrato center pitch
- **Data-Driven Validation**: Uses actual segment pitch data rather than theoretical endpoints

#### Problem Solved:
Previously, the algorithm had issues with segments 408-413 where a single outlier frequency (segment 412 at 587 Hz vs others at 434-487 Hz) prevented valid vibrato detection. The improved algorithm now handles such outliers gracefully and makes more accurate classifications.

## Notes for Future Development
- **Modular Design**: Each component can be developed and tested independently
- **Algorithm Selection**: Default uses Essentia for efficiency; others available for comparison
- **Configuration**: Confidence thresholds and parameters tuned for Indian classical music
- **Extensibility**: New algorithms or visualizations can be added to respective modules
- **Code Quality**: Maintain separation of concerns and reasonable file sizes
- **Documentation**: Each module is self-documenting with clear responsibilities
- **Vibrato Algorithm**: Continue refinining vibrato detection based on musicological analysis of Indian classical performances

## Commands to Remember
- **Test Pipeline**: `pipenv run python src/autotranscription/pipeline.py`
- **Test Audio Separator**: `pipenv run python src/autotranscription/audio_separator.py`
- **Environment Setup**: `pipenv install` then `pipenv shell`

## Current Workspace Status
The pipeline has been extensively tested with multiple audio files in `pipeline_workspace/`:
- **Processed Files**: 6 audio files (IDs 0-5) with complete pipeline processing
- **Active Test File**: `0.mp3` (Kishori Amonkar - Babul Mora_1min.mp3) - Primary development and testing file
- **Stage Completion**: All files have completed all 6 pipeline stages:
  1. Audio Separation (Demucs)
  2. Tonic Estimation (Essentia TonicIndianArtMusic)
  3. Pitch Trace Extraction (Essentia PredominantPitchMelodia)
  4. Pitch Cleaning (confidence-based filtering)
  5. Raga Estimation (TDMS algorithm)
  6. Pitch Segmentation (two-stage min/max with vibrato detection)

### Output Files Available:
- **Visualizations**: Multiple segmentation comparison plots, sargam plots, stage-specific visualizations
- **Data Arrays**: Individual segment data for each stage (original, stage1, stage2)
- **Metadata**: Comprehensive JSON files with detailed segment statistics and frequency information
- **Audio Outputs**: Sine wave reconstructions for selected files

### Pipeline Robustness:
- **Resumable**: Can restart from any completed stage
- **Incremental**: Only processes missing stages
- **Data Integrity**: All intermediate results preserved for analysis
- **Debugging Support**: Comprehensive logging and detailed metadata for troubleshooting