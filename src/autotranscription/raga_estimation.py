"""
Raga estimation module using Time-Delayed Melody Surfaces (TDMS).

Implements phase space embedding and delay coordinate generation for raga recognition
in Indian classical music based on the research by Gulati et al. (ISMIR 2016).
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from scipy.spatial.distance import cdist
from scipy.stats import entropy
import pickle


class RagaEstimator:
    """
    Raga estimation using Time-Delayed Melody Surfaces (TDMS).
    
    Based on the approach by Gulati et al. that uses delay coordinates to create
    phase space embeddings that capture both tonal and temporal characteristics
    of melodies for raga recognition.
    """
    
    def __init__(self, models_dir: Path = None):
        """
        Initialize the raga estimator.
        
        Args:
            models_dir: Directory containing trained raga models (optional)
        """
        self.models_dir = models_dir if models_dir else Path.cwd() / "raga_models"
        self.models_dir.mkdir(exist_ok=True)
        
        # TDMS parameters (based on research paper)
        self.delay_coordinates = 3  # Number of delay coordinates
        self.delay_step = 1  # Delay step size
        self.octave_folding = True  # Fold to single octave
        self.grid_resolution = 50  # Resolution for 2D representation
        
        # Common ragas for recognition (can be extended)
        self.known_ragas = [
            'Yaman', 'Bhairav', 'Bhimpalasi', 'Malkauns', 'Bageshri',
            'Darbari', 'Kedar', 'Marwa', 'Puriya', 'Todi',
            'Alhaiya_Bilaval', 'Vrindavani_Sarang', 'Jaunpuri', 'Khamaj', 'Bihag'
        ]
        
        # Load any existing models
        self.raga_models = self._load_models()
    
    def estimate_raga_from_pitch_data(self, pitch_file: str, confidence_file: str, 
                                    times_file: str, tonic_frequency: float,
                                    confidence_threshold: float = 0.1) -> Dict[str, Any]:
        """
        Estimate raga from pitch trace data using TDMS.
        
        Args:
            pitch_file: Path to pitch data numpy file
            confidence_file: Path to confidence data numpy file  
            times_file: Path to times data numpy file
            tonic_frequency: Tonic frequency in Hz
            confidence_threshold: Minimum confidence for valid pitch points
            
        Returns:
            Dictionary with raga estimation results
        """
        # Load pitch data
        pitch = np.load(pitch_file)
        confidence = np.load(confidence_file)
        times = np.load(times_file)
        
        # Filter for high confidence regions
        valid_mask = (confidence > confidence_threshold) & (pitch > 0)
        
        if np.sum(valid_mask) < 10:  # Need minimum valid points
            return {
                "error": "Insufficient pitch data for raga estimation"
            }
        
        # Extract valid pitch sequence
        valid_pitch = pitch[valid_mask]
        valid_times = times[valid_mask]
        
        # Normalize to tonic (convert to cents relative to tonic)
        normalized_pitch = self._normalize_to_tonic(valid_pitch, tonic_frequency)
        
        # Generate TDMS representation
        tdms_surface = self._generate_tdms(normalized_pitch)
        
        # Estimate raga using pattern matching or classification
        raga_results = self._classify_raga(tdms_surface, normalized_pitch)
        
        return raga_results
    
    def _normalize_to_tonic(self, pitch_hz: np.ndarray, tonic_hz: float) -> np.ndarray:
        """
        Normalize pitch to tonic in cents.
        
        Args:
            pitch_hz: Pitch values in Hz
            tonic_hz: Tonic frequency in Hz
            
        Returns:
            Pitch values in cents relative to tonic
        """
        # Convert to cents relative to tonic
        cents = 1200 * np.log2(pitch_hz / tonic_hz)
        
        # Fold to single octave if enabled
        if self.octave_folding:
            cents = cents % 1200
        
        return cents
    
    def _generate_tdms(self, pitch_cents: np.ndarray) -> np.ndarray:
        """
        Generate Time-Delayed Melody Surface from pitch data.
        
        Args:
            pitch_cents: Pitch sequence in cents
            
        Returns:
            2D TDMS surface representation
        """
        # Generate delay coordinates
        delay_coords = self._get_delay_coordinates(pitch_cents)
        
        # Create 2D representation
        tdms_surface = self._create_2d_representation(delay_coords)
        
        return tdms_surface
    
    def _get_delay_coordinates(self, signal: np.ndarray) -> np.ndarray:
        """
        Extract delay coordinates from input signal.
        
        Args:
            signal: Input pitch sequence
            
        Returns:
            Matrix of delay coordinates [n_points, n_delays]
        """
        n_points = len(signal) - (self.delay_coordinates - 1) * self.delay_step
        if n_points <= 0:
            return np.array([])
        
        delay_matrix = np.zeros((n_points, self.delay_coordinates))
        
        for i in range(self.delay_coordinates):
            start_idx = i * self.delay_step
            end_idx = start_idx + n_points
            delay_matrix[:, i] = signal[start_idx:end_idx]
        
        return delay_matrix
    
    def _create_2d_representation(self, delay_coords: np.ndarray) -> np.ndarray:
        """
        Create 2D surface representation from delay coordinates.
        
        Args:
            delay_coords: Matrix of delay coordinates
            
        Returns:
            2D surface representation
        """
        if len(delay_coords) == 0:
            return np.zeros((self.grid_resolution, self.grid_resolution))
        
        # Use first two delay coordinates for 2D representation
        x_coords = delay_coords[:, 0]  # Current pitch
        y_coords = delay_coords[:, 1] if delay_coords.shape[1] > 1 else delay_coords[:, 0]  # Previous pitch
        
        # Create 2D histogram/surface
        x_range = (0, 1200) if self.octave_folding else (x_coords.min(), x_coords.max())
        y_range = (0, 1200) if self.octave_folding else (y_coords.min(), y_coords.max())
        
        surface, _, _ = np.histogram2d(
            x_coords, y_coords,
            bins=self.grid_resolution,
            range=[x_range, y_range],
            density=True
        )
        
        return surface
    
    def _classify_raga(self, tdms_surface: np.ndarray, 
                      pitch_sequence: np.ndarray) -> Dict[str, Any]:
        """
        Classify raga based on TDMS surface and pitch characteristics.
        
        Args:
            tdms_surface: 2D TDMS surface
            pitch_sequence: Normalized pitch sequence in cents
            
        Returns:
            Raga classification results
        """
        # Calculate pitch histogram (svara distribution)
        svara_hist = self._calculate_svara_histogram(pitch_sequence)
        
        # Extract surface features
        surface_features = self._extract_surface_features(tdms_surface)
        
        # If we have trained models, use them
        if self.raga_models:
            classification_result = self._model_based_classification(
                tdms_surface, svara_hist, surface_features
            )
        else:
            # Use rule-based classification as fallback
            classification_result = self._rule_based_classification(
                svara_hist, surface_features
            )
        
        return classification_result
    
    def _calculate_svara_histogram(self, pitch_cents: np.ndarray, 
                                 n_bins: int = 120) -> np.ndarray:
        """
        Calculate histogram of svaras (pitch classes).
        
        Args:
            pitch_cents: Pitch sequence in cents
            n_bins: Number of bins for histogram
            
        Returns:
            Normalized histogram of pitch distribution
        """
        hist, _ = np.histogram(pitch_cents, bins=n_bins, range=(0, 1200), density=True)
        return hist
    
    def _extract_surface_features(self, surface: np.ndarray) -> Dict[str, float]:
        """
        Extract features from TDMS surface for classification.
        
        Args:
            surface: 2D TDMS surface
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Basic statistical features
        features['surface_mean'] = float(surface.mean())
        features['surface_std'] = float(surface.std())
        features['surface_max'] = float(surface.max())
        features['surface_entropy'] = float(entropy(surface.flatten() + 1e-10))
        
        # Spatial features
        center_of_mass = np.unravel_index(surface.argmax(), surface.shape)
        features['peak_x'] = float(center_of_mass[0] / surface.shape[0])
        features['peak_y'] = float(center_of_mass[1] / surface.shape[1])
        
        # Distribution features
        features['surface_skewness'] = float(self._calculate_skewness(surface.flatten()))
        features['surface_kurtosis'] = float(self._calculate_kurtosis(surface.flatten()))
        
        return features
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _model_based_classification(self, tdms_surface: np.ndarray,
                                  svara_hist: np.ndarray,
                                  features: Dict[str, float]) -> Dict[str, Any]:
        """
        Classify raga using trained models.
        
        Args:
            tdms_surface: 2D TDMS surface
            svara_hist: Svara histogram
            features: Extracted features
            
        Returns:
            Classification results with confidence scores
        """
        # This would use trained models - placeholder for now
        # In practice, you'd load ML models trained on labeled raga data
        
        # For now, return a simple pattern-based result
        return self._rule_based_classification(svara_hist, features)
    
    def _rule_based_classification(self, svara_hist: np.ndarray,
                                 features: Dict[str, float]) -> Dict[str, Any]:
        """
        Rule-based raga classification using pitch distribution patterns.
        
        Args:
            svara_hist: Svara histogram
            features: Surface features
            
        Returns:
            Classification results
        """
        # Define simplified raga characteristics based on prominent svaras
        raga_patterns = {
            'Yaman': [0, 200, 400, 600, 700, 900, 1100],  # Sa Re Ga Ma# Pa Dha Ni
            'Bhairav': [0, 100, 300, 500, 700, 800, 1000],  # Sa Re♭ Ga♭ Ma Pa Dha♭ Ni♭
            'Malkauns': [0, 300, 500, 800, 1000],  # Sa Ga♭ Ma Dha♭ Ni♭
            'Bageshri': [0, 200, 300, 500, 700, 800, 1100],  # Sa Re Ga♭ Ma Pa Dha♭ Ni
            'Bhimpalasi': [0, 200, 300, 500, 700, 800, 1000],  # Sa Re Ga♭ Ma Pa Dha♭ Ni♭
        }
        
        # Calculate similarity scores
        scores = {}
        for raga_name, pattern in raga_patterns.items():
            # Create expected histogram for this raga
            expected_hist = np.zeros_like(svara_hist)
            for cent_value in pattern:
                bin_idx = int(cent_value / 1200 * len(expected_hist))
                if 0 <= bin_idx < len(expected_hist):
                    expected_hist[bin_idx] = 1.0
            
            # Smooth the expected histogram
            expected_hist = self._smooth_histogram(expected_hist)
            
            # Calculate correlation with observed histogram
            correlation = np.corrcoef(svara_hist, expected_hist)[0, 1]
            scores[raga_name] = max(0, correlation) if not np.isnan(correlation) else 0
        
        # Sort by score
        sorted_ragas = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Prepare results
        top_raga = sorted_ragas[0] if sorted_ragas else ('Unknown', 0.0)
        
        results = {
            'predicted_raga': top_raga[0],
            'confidence': float(top_raga[1]),
            'candidates': [
                {'raga': raga, 'confidence': float(score)} 
                for raga, score in sorted_ragas[:5]  # Top 5 with scores
            ]
        }
        
        return results
    
    def _smooth_histogram(self, hist: np.ndarray, sigma: float = 2.0) -> np.ndarray:
        """
        Smooth histogram using Gaussian convolution.
        
        Args:
            hist: Input histogram
            sigma: Gaussian sigma parameter
            
        Returns:
            Smoothed histogram
        """
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(hist, sigma=sigma, mode='wrap')
        return smoothed / (smoothed.sum() + 1e-10)  # Normalize
    
    def _load_models(self) -> Dict[str, Any]:
        """
        Load pre-trained raga models if available.
        
        Returns:
            Dictionary of loaded models
        """
        models = {}
        model_file = self.models_dir / "raga_models.pkl"
        
        if model_file.exists():
            try:
                with open(model_file, 'rb') as f:
                    models = pickle.load(f)
                print(f"Loaded raga models for {len(models)} ragas")
            except Exception as e:
                print(f"Failed to load raga models: {e}")
        
        return models
    
    def save_models(self, models: Dict[str, Any]):
        """
        Save trained raga models.
        
        Args:
            models: Dictionary of models to save
        """
        model_file = self.models_dir / "raga_models.pkl"
        
        try:
            with open(model_file, 'wb') as f:
                pickle.dump(models, f)
            print(f"Saved raga models to {model_file}")
        except Exception as e:
            print(f"Failed to save raga models: {e}")
    
    def get_supported_ragas(self) -> List[str]:
        """
        Get list of supported ragas.
        
        Returns:
            List of supported raga names
        """
        return self.known_ragas.copy()