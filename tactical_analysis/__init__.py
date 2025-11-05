"""
Tactical Analysis Module

Provides homography transformations and field mapping for soccer analysis.
"""

from tactical_analysis.homography import HomographyTransformer
from tactical_analysis.global_homography_mapper import GlobalHomographyMapper
from tactical_analysis.confidence_visualizer import ConfidenceVisualizer
from tactical_analysis.sports_compat import ViewTransformer, SoccerPitchConfiguration, draw_pitch

__all__ = [
    'HomographyTransformer',
    'GlobalHomographyMapper',
    'ConfidenceVisualizer',
    'ViewTransformer',
    'SoccerPitchConfiguration',
    'draw_pitch'
]
