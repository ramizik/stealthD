"""
Fast feature extraction for team assignment during inference.
Uses color histograms instead of slow SigLIP embeddings.
"""

import cv2
import numpy as np
import supervision as sv
from PIL import Image

class FastFeatureExtractor:
    """
    Extract fast color-based features for team assignment.
    Much faster than SigLIP embeddings (< 1ms vs 2600ms per player).
    """

    def __init__(self, bins=(8, 8, 8), torso_only=True):
        """
        Initialize fast feature extractor.

        Args:
            bins: Number of bins for each color channel (H, S, V)
            torso_only: If True, crop only upper 50% of player bbox (jersey focus)
        """
        self.bins = bins
        self.feature_dim = bins[0] + bins[1] + bins[2]  # Concatenated histogram
        self.torso_only = torso_only

    def extract_color_histogram(self, image):
        """
        Extract color histogram from image.

        Args:
            image: BGR image (numpy array)

        Returns:
            Normalized color histogram feature vector
        """
        # Validate image is not empty
        if image is None or image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
            # Return zero vector for invalid images
            total_bins = sum(self.bins)
            return np.zeros(total_bins)

        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Extract histograms for each channel
        h_hist = cv2.calcHist([hsv], [0], None, [self.bins[0]], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [self.bins[1]], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [self.bins[2]], [0, 256])

        # Normalize histograms
        h_hist = h_hist.flatten() / (h_hist.sum() + 1e-7)
        s_hist = s_hist.flatten() / (s_hist.sum() + 1e-7)
        v_hist = v_hist.flatten() / (v_hist.sum() + 1e-7)

        # Concatenate all histograms
        feature = np.concatenate([h_hist, s_hist, v_hist])

        return feature

    def get_player_features(self, frame, player_detections):
        """
        Extract fast features from all players in a frame.

        Args:
            frame: Input video frame
            player_detections: Player detection results

        Returns:
            Numpy array of features (N, feature_dim)
        """
        if len(player_detections.xyxy) == 0:
            return np.array([]).reshape(0, self.feature_dim)

        features = []
        for bbox in player_detections.xyxy:
            # Crop player region
            cropped = sv.crop_image(frame, bbox)

            # Apply torso-only cropping to focus on jersey colors
            if self.torso_only:
                height = cropped.shape[0]
                cropped = cropped[0:int(height/2), :]  # Upper 50% only

            # Extract color histogram
            feature = self.extract_color_histogram(cropped)
            features.append(feature)

        return np.array(features)

    def get_features_from_crops(self, crops):
        """
        Extract fast features from pre-cropped player images (PIL format).
        Used during training when we have crops instead of full frames.

        Args:
            crops: List of PIL Image objects (player crops)

        Returns:
            Numpy array of features (N, feature_dim)
        """
        if len(crops) == 0:
            return np.array([]).reshape(0, self.feature_dim)

        features = []
        invalid_count = 0

        for crop in crops:
            # Convert PIL to numpy BGR
            if isinstance(crop, Image.Image):
                # PIL image - convert to BGR
                img_array = np.array(crop)
                if img_array.shape[2] == 3:  # RGB
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                elif img_array.shape[2] == 4:  # RGBA
                    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                else:
                    img_bgr = img_array
            else:
                # Already numpy array
                img_bgr = crop

            # Validate crop before processing
            if img_bgr is None or img_bgr.size == 0 or img_bgr.shape[0] == 0 or img_bgr.shape[1] == 0:
                # Return zero vector for invalid crops
                total_bins = sum(self.bins)
                features.append(np.zeros(total_bins))
                invalid_count += 1
                continue

            # Apply torso-only cropping to focus on jersey colors
            if self.torso_only:
                height = img_bgr.shape[0]
                img_bgr = img_bgr[0:int(height/2), :]  # Upper 50% only

                # Check again after torso cropping
                if img_bgr.shape[0] == 0 or img_bgr.shape[1] == 0:
                    total_bins = sum(self.bins)
                    features.append(np.zeros(total_bins))
                    invalid_count += 1
                    continue

            # Extract color histogram
            feature = self.extract_color_histogram(img_bgr)
            features.append(feature)

        # Print warning if invalid crops detected
        if invalid_count > 0:
            print(f"  ⚠️ Warning: {invalid_count}/{len(crops)} crops were empty/invalid (filled with zeros)")

        return np.array(features)
