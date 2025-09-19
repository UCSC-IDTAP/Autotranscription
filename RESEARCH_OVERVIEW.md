# Auto-Transcription Pipeline for Indian Classical Music
## Research Overview & Onboarding Guide

### Executive Summary

This project implements a comprehensive **automatic transcription pipeline** specifically designed for **Indian classical music**. The pipeline processes audio recordings through multiple sophisticated stages to extract melodic information, identify the raga, and segment the performance into meaningful musical units with detailed pitch analysis.

**Key Innovation**: The system combines modern AI-driven audio separation (Demucs) with specialized Indian classical music algorithms including Time-Delayed Melody Surfaces (TDMS) for raga recognition and multi-stage pitch segmentation with advanced vibrato detection.

---

## 🎯 Project Status & Capabilities

### ✅ Fully Operational Pipeline
- **6 audio files** have been successfully processed through the complete pipeline
- **All 6 stages** are working: separation, tonic estimation, pitch extraction, cleaning, raga estimation, and segmentation
- **Comprehensive test data** available in `pipeline_workspace/` with detailed visualizations and outputs
- **Recent focus**: Advanced vibrato detection and segment classification improvements

### 🔬 Current Research Focus
Based on recent commits, the team has been focused on:
1. **Vibrato Detection Algorithm**: Sophisticated multi-criteria vibrato detection with frequency range analysis
2. **Pitch Segmentation Refinement**: Two-stage min/max approach with intelligent merging
3. **False Positive Correction**: Post-processing to reclassify stable-frequency "vibrato" as fixed pitch
4. **Visualization Quality**: High-resolution, detailed segmentation plots with raga pitch references

---

## 🏗️ Architecture Overview

### Modular Design Philosophy
The codebase follows a **modular architecture** with clear separation of concerns:

```
src/autotranscription/
├── pipeline.py              # Main orchestrator (1,117 lines)
├── audio_separator.py       # Demucs-based vocal/instrumental separation
├── pitch_extraction.py      # Multi-algorithm pitch estimation
├── pitch_cleaning.py        # Octave jump correction and artifacts removal
├── tonic_estimation.py      # Tonic frequency detection
├── raga_estimation.py       # TDMS-based raga recognition
├── pitch_segmentation.py    # Advanced pitch segmentation (1,681 lines)
├── visualization.py         # Sargam-based plotting
└── audio_processing.py      # Audio utilities and synthesis
```

### Core Pipeline Stages

#### Stage 1: Audio Separation
- **Technology**: Demucs neural source separation (htdemucs model)
- **Output**: Clean vocals track + instrumental track
- **Performance**: ~22 seconds processing time for 1-minute audio

#### Stage 2: Tonic Estimation
- **Algorithm**: Essentia's `TonicIndianArtMusic`
- **Validation**: Compares tonic estimation on original vs. instrumental tracks
- **Output**: Tonic frequency in Hz + Western note name

#### Stage 3: Pitch Trace Extraction
- **Primary Algorithm**: Essentia's `PredominantPitchMelodia` (optimized for Indian classical)
- **Alternative Algorithms**: CREPE, SWIPE, PyWorld available for comparison
- **Optimization**: Tuned for 55-1760 Hz range with confidence-based filtering
- **Confidence Thresholds**: Algorithm-specific (Essentia: 0.05, CREPE: 0.7, etc.)

#### Stage 4: Pitch Cleaning
- **Purpose**: Remove octave jumps and algorithmic artifacts
- **Method**: Statistical analysis to detect and correct frequency jumps
- **Output**: Cleaned pitch traces with correction statistics

#### Stage 5: Raga Estimation (TDMS)
- **Technology**: Time-Delayed Melody Surfaces for pattern recognition
- **Process**: Phase space embedding → delay coordinates → pattern matching
- **Database**: 15+ ragas including Yaman, Bhairav, Bhimpalasi, Malkauns, etc.
- **Integration**: Creates IDTAP Raga objects with API-sourced definitions when available

#### Stage 6: Pitch Segmentation (Research Innovation)
- **Two-Stage Approach**:
  - **Stage 1**: Fine-grained segmentation at every pitch direction change
  - **Stage 2**: Intelligent merging with advanced vibrato detection
- **Segment Types**: Fixed pitch, moving pitch, silence, vibrato
- **Vibrato Detection**: Multi-criteria analysis:
  - Minimum 3-6 consecutive short segments (< 0.15-0.20s each)
  - Frequency range within ~1.5 semitones (0.125 log units)
  - True oscillation pattern with ≥4 direction changes
  - Post-processing reclassification for false positives

---

## 🔬 Technical Innovations

### 1. Advanced Vibrato Detection
The recent work has focused on solving challenging vibrato detection problems:

**Problem**: Previous algorithm had issues with outlier frequencies preventing valid vibrato detection
**Solution**: Multi-criteria approach using actual segment data rather than theoretical endpoints

**Key Algorithm Features**:
- Uses actual min/max frequencies from segment data
- Greedy sequential search for vibrato patterns
- Post-processing step to reclassify false vibrato as fixed pitch
- Statistical validation using center pitch calculations

### 2. IDTAP Integration
- **Raga Object Creation**: Integrates with IDTAP (Indian Digital Traditional Arts Project) API
- **Pitch Targeting**: Uses raga-specific target pitches for accurate segmentation
- **Sargam Notation**: Outputs use traditional Indian musical notation (S r R g G m M P d D n N)

### 3. High-Quality Visualizations
- **Sargam Y-Axis**: Plots use traditional Indian notation instead of frequency/MIDI
- **Raga Reference Lines**: Shows target pitches with threshold boundaries
- **Multi-Stage Comparison**: Visualizes segmentation evolution across processing stages
- **Extreme Detail**: 120+ inch wide plots for maximum temporal resolution

---

## 📊 Current Test Data & Results

### Processed Audio Files
- **Primary Test**: Kishori Amonkar - Babul Mora (1 minute excerpt)
- **Tonic**: D#3 (157.6 Hz)
- **Processing Stats**: 63% high-confidence pitch frames
- **Segmentation**: Generates 100+ detailed segments with comprehensive metadata

### Available Outputs
```
pipeline_workspace/
├── audio/                   # Copied input files (6 files)
├── separated/               # Demucs vocal/instrumental tracks
├── data/
│   ├── pitch_traces/       # Numpy arrays with pitch/confidence/time data
│   └── segmentation/       # Individual segment arrays (3000+ files)
├── visualizations/         # High-resolution PNG plots
│   ├── *_pitch_sargam.png         # Traditional sargam notation plots
│   ├── *_pitch_segmentation_*.png # Multi-stage segmentation analysis
│   └── *_segmentation_comparison.png
├── audio_outputs/          # Sine wave reconstructions
└── metadata/              # Complete JSON results with statistics
```

---

## 🛠️ Development Environment

### Dependencies
- **Audio Processing**: demucs, essentia, librosa, soundfile
- **Pitch Estimation**: crepe, pysptk (SWIPE), pyworld
- **Scientific Computing**: numpy, scipy, matplotlib
- **Machine Learning**: tensorflow (for CREPE)
- **Project-Specific**: idtap (local editable package)

### Setup
```bash
pipenv install              # Install all dependencies
pipenv shell               # Activate environment
python src/autotranscription/pipeline.py  # Run test
```

### File Organization
- **Resumable Processing**: Metadata tracks completed stages
- **Incremental Updates**: Only processes missing stages
- **Data Persistence**: All intermediate results saved for analysis
- **Organized Workspace**: Structured file hierarchy with unique IDs

---

## 🎯 Research Opportunities

### Immediate Areas for Investigation

#### 1. Vibrato Algorithm Refinement
- **Current Challenge**: Balancing sensitivity vs. false positives
- **Research Question**: Can we improve musical accuracy of vibrato vs. fixed pitch classification?
- **Data Available**: 3000+ segment data files for algorithm validation

#### 2. Raga Recognition Accuracy
- **Current State**: TDMS implementation with rule-based classification
- **Potential**: Machine learning approaches using segmented pitch data
- **Dataset**: Multiple processed recordings with raga ground truth

#### 3. Performance Analysis
- **Metrics Needed**: Accuracy validation against human annotations
- **Tools Available**: Detailed segment-level data for comparison
- **Visualization**: Existing plots show algorithm decisions for manual verification

#### 4. Multi-Algorithm Fusion
- **Current**: Primarily uses Essentia, others available for comparison
- **Opportunity**: Ensemble methods combining multiple pitch extractors
- **Infrastructure**: Pipeline already supports multiple algorithms

### Advanced Research Directions

#### 1. Temporal Pattern Analysis
- **Data**: High-resolution segmentation with timing information
- **Applications**: Phrase structure analysis, rhythmic pattern detection
- **Integration**: Could enhance raga recognition with temporal features

#### 2. Performance Style Analysis
- **Opportunity**: Artist-specific ornamental pattern recognition
- **Data**: Detailed vibrato and movement analysis available
- **Applications**: Performance comparison, style classification

#### 3. Real-time Processing
- **Current**: Batch processing pipeline
- **Research**: Streaming adaptation for live performance analysis
- **Challenges**: Memory management, latency optimization

---

## 📚 Getting Started as a Researcher

### 1. Explore Existing Results
```bash
# Look at processed data
ls pipeline_workspace/visualizations/
ls pipeline_workspace/data/segmentation/

# Examine a metadata file
cat pipeline_workspace/metadata/0.json | head -50
```

### 2. Run the Pipeline
```bash
# Test with existing audio
pipenv run python src/autotranscription/pipeline.py

# Process new audio file
pipenv run python -c "from src.autotranscription.pipeline import run_pipeline; run_pipeline('path/to/audio.mp3')"
```

### 3. Understand the Code
- **Start with**: `pipeline.py` for overall architecture
- **Key Innovation**: `pitch_segmentation.py` for current research focus
- **Visualization**: `visualization.py` for output generation
- **Data Flow**: Follow the 6-stage progression in the pipeline

### 4. Analyze Segmentation Results
```python
# Load segment data for analysis
import numpy as np
import json

# Load detailed segment metadata
with open('pipeline_workspace/data/segmentation/0_segments_stage2.json') as f:
    segments = json.load(f)

# Load individual segment pitch data
segment_data = np.load('pipeline_workspace/data/segmentation/0_segment_000_data_stage2.npy')
```

---

## 🎵 Musical Context

### Indian Classical Music Specifics
- **Microtonal System**: 22 shruti (microtones) per octave
- **Raga Structure**: Specific pitch relationships and melodic rules
- **Ornaments**: Extensive use of gamakas (pitch bends) and vibrato
- **Performance Style**: Improvisational within strict tonal frameworks

### Why This Matters for Transcription
- **Standard Western Tools Fail**: Equal temperament assumptions break down
- **Context-Sensitive Analysis**: Same frequency has different meanings in different ragas
- **Temporal Patterns**: Ornamental patterns carry structural significance
- **Cultural Preservation**: Digital archival of traditional performance practices

---

## 🚀 Next Steps for New Researchers

1. **Familiarize with Results**: Examine existing visualizations and understand what the pipeline produces
2. **Validate Algorithms**: Compare algorithm outputs with your musical knowledge
3. **Identify Improvement Areas**: Look for segmentation or classification errors in the data
4. **Propose Experiments**: Design studies to validate or improve specific components
5. **Contribute to Codebase**: The modular architecture makes it easy to enhance individual components

---

## 📞 Technical Support

- **Code Documentation**: Each module has comprehensive docstrings
- **Test Data**: Extensive workspace with multiple processed examples
- **Visualization**: Detailed plots show algorithm decisions for debugging
- **Modular Design**: Each component can be tested and developed independently

**Welcome to the project! The pipeline is mature, well-tested, and ready for advanced research.**