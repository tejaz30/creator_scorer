"""
Attractiveness analysis module.
Computes multi-cue attractiveness scores for video reels.
Now supports both legacy and composite scoring systems.
"""
import os
import math
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from utils.video_utils import sample_frames_from_video
from config import FRAME_SAMPLE_COUNT

# Import the new composite attractiveness analyzer
try:
    # DISABLED: Heavy model loading - use lightweight legacy system only
    # from features.attractiveness_composite import composite_attractiveness_analyzer
    COMPOSITE_ANALYZER_AVAILABLE = False
    print("ℹ️ Using lightweight attractiveness analysis (composite models disabled)")
except ImportError:
    COMPOSITE_ANALYZER_AVAILABLE = False
    print("⚠️ Composite attractiveness analyzer not available, using legacy system")

# Import aesthetic predictor (assuming it exists in your environment)
try:
    from aesthetic_predictor import predict_aesthetic
except ImportError:
    def predict_aesthetic(img):
        """Fallback aesthetic predictor."""
        return 5.0  # Default score

# Use a standard OpenCV Haar cascade for faces
_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

@dataclass
class DetectedFace:
    x: int
    y: int
    w: int
    h: int

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

class AttractivenessAnalyzer:
    """Analyzes attractiveness of video reels using multiple visual cues."""
    
    def __init__(self, frames_per_reel: int = FRAME_SAMPLE_COUNT):
        self.frames_per_reel = frames_per_reel
    
    def detect_faces_in_frame(self, frame_bgr: np.ndarray) -> List[DetectedFace]:
        """Detect faces using Haar cascade."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = _CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return [DetectedFace(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
    
    def select_best_face_frame(self, frames: List[np.ndarray]) -> Tuple[Optional[int], Optional[np.ndarray], Optional[DetectedFace]]:
        """
        From a list of frames, pick the one with the 'best' face (largest area).
        Returns: (best_frame_index, best_frame, best_face_object)
        """
        best_idx = None
        best_frame = None
        best_face = None
        best_area = 0.0

        for idx, frame in enumerate(frames):
            faces = self.detect_faces_in_frame(frame)
            if not faces:
                continue
            for f in faces:
                area = f.w * f.h
                if area > best_area:
                    best_area = area
                    best_idx = idx
                    best_frame = frame
                    best_face = f

        return best_idx, best_frame, best_face
    
    def compute_lighting_score(self, frame_bgr: np.ndarray) -> float:
        """Lighting score based on average brightness of the frame."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        v_channel = hsv[..., 2].astype(np.float32)  # 0–255
        mean_v = float(v_channel.mean())
        # Normalize 0–255 → 0–1
        score = max(0.0, min(1.0, mean_v / 255.0))
        return score
    
    def compute_sharpness_score(self, frame_bgr: np.ndarray) -> float:
        """Sharpness score based on variance of Laplacian."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        var = float(lap.var())

        # Heuristic normalization: 0–500 → 0–1, clamp
        norm = var / 500.0
        norm = max(0.0, min(1.0, norm))
        return norm
    
    def face_cues(self, frame_shape: Tuple[int, int, int], bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """
        Compute face area fraction and center offset normalization.
        """
        h, w = frame_shape[:2]
        x, y, bw, bh = bbox

        face_area = float(bw * bh)
        frame_area = float(w * h) if (w > 0 and h > 0) else 1.0
        face_area_frac = face_area / frame_area

        frame_cx, frame_cy = w / 2.0, h / 2.0
        face_cx, face_cy = x + bw / 2.0, y + bh / 2.0

        dx = face_cx - frame_cx
        dy = face_cy - frame_cy
        dist = math.sqrt(dx * dx + dy * dy)

        # Max possible distance is corner to center
        max_dist = math.sqrt(frame_cx ** 2 + frame_cy ** 2) or 1.0
        center_offset_norm = dist / max_dist  # 0 center → 1 corner

        return float(face_area_frac), float(center_offset_norm)
    
    def crop_face(self, frame_bgr: np.ndarray, face: DetectedFace) -> Image.Image:
        """Crop the face region and return a PIL Image (RGB)."""
        x, y, w, h = face.bbox
        h_img, w_img = frame_bgr.shape[:2]

        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(w_img, x + w)
        y1 = min(h_img, y + h)

        face_bgr = frame_bgr[y0:y1, x0:x1]
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(face_rgb)
    
    def aesthetic_score(self, img) -> float:
        """Compute aesthetic score using the aesthetic predictor."""
        # If it's an OpenCV frame or cropped region (NumPy array)
        if isinstance(img, np.ndarray):
            # assume BGR and convert → RGB PIL
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        elif isinstance(img, Image.Image):
            # already PIL; ensure RGB
            img = img.convert("RGB")
        else:
            # Fallback: try to wrap whatever it is as PIL
            img = Image.fromarray(np.array(img)).convert("RGB")

        # Call the aesthetic predictor
        score = predict_aesthetic(img)   # returns 0–10
        return float(score)
    
    def multi_cue_attractiveness(
        self,
        aesthetic_face_0_10: float,
        aesthetic_full_0_10: float,
        lighting_0_1: float,
        sharpness_0_1: float,
    ) -> float:
        """
        Fuse cues with weighted combination:
        0.65 * face aesthetic + 0.20 * full-frame aesthetic + 0.10 * lighting + 0.05 * sharpness
        Returns a 0–10 score.
        """
        # Normalize aesthetics to [0, 1]
        face_norm = np.clip((aesthetic_face_0_10 or 0.0) / 10.0, 0.0, 1.0)
        full_norm = np.clip((aesthetic_full_0_10 or 0.0) / 10.0, 0.0, 1.0)

        # Clamp lighting & sharpness
        lt = np.clip(lighting_0_1 or 0.0, 0.0, 1.0)
        sh = np.clip(sharpness_0_1 or 0.0, 0.0, 1.0)

        score_0_1 = (
            0.65 * face_norm +
            0.20 * full_norm +
            0.10 * lt +
            0.05 * sh
        )

        score_0_1 = float(np.clip(score_0_1, 0.0, 1.0))
        return score_0_1 * 10.0
    
    def compute_attractiveness_for_reel(self, video_path: str) -> Dict[str, float]:
        """
        Main entry point for computing attractiveness metrics for a reel.
        Uses composite scoring if available, otherwise falls back to legacy system.
        """
        # Try composite scoring first
        if COMPOSITE_ANALYZER_AVAILABLE:
            try:
                composite_results = composite_attractiveness_analyzer.compute_attractiveness_for_reel(video_path)
                
                # Also compute legacy metrics for backward compatibility
                legacy_results = self._compute_legacy_attractiveness(video_path)
                
                # Combine results, prioritizing composite scores
                combined_results = {**legacy_results, **composite_results}
                
                # Map composite score to legacy field for compatibility
                if "composite_aesthetic_score" in composite_results:
                    combined_results["multi_cue_attr_0_10"] = composite_results["composite_aesthetic_score"]
                
                return combined_results
                
            except Exception as e:
                print(f"    ⚠️ Composite scoring failed, falling back to legacy: {e}")
        
        # Fallback to legacy system
        return self._compute_legacy_attractiveness(video_path)
    
    def _compute_legacy_attractiveness(self, video_path: str) -> Dict[str, float]:
        """
        Legacy attractiveness computation method.
        """
        empty = {
            "best_frame_idx": None,
            "lighting": np.nan,
            "sharpness": np.nan,
            "face_area_frac": np.nan,
            "center_offset_norm": np.nan,
            "aesthetic_face_0_10": np.nan,
            "aesthetic_full_0_10": np.nan,
            "multi_cue_attr_0_10": np.nan,
        }

        if not video_path or not os.path.exists(video_path):
            print("    ✗ Video path does not exist:", video_path)
            return empty

        # 1) Sample frames
        frames = sample_frames_from_video(video_path, max_frames=self.frames_per_reel)
        if not frames:
            print("    ✗ No frames sampled from video")
            return empty

        # 2) Select best face frame
        best_idx, best_frame, best_face = self.select_best_face_frame(frames)
        if best_frame is None or best_face is None:
            print("    ✗ No face detected in sampled frames")
            return empty

        # 3) Visual cues
        lighting = self.compute_lighting_score(best_frame)    # 0–1
        sharpness = self.compute_sharpness_score(best_frame)  # 0–1
        face_area_frac, center_offset_norm = self.face_cues(best_frame.shape, best_face.bbox)

        # 4) Aesthetic scores (face + full frame)
        face_img = self.crop_face(best_frame, best_face)      # PIL image
        aest_face = self.aesthetic_score(face_img)            # 0–10
        # full-frame aesthetic uses the whole frame
        full_rgb = cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB)
        full_img = Image.fromarray(full_rgb)
        aest_full = self.aesthetic_score(full_img)            # 0–10

        # 5) Fuse
        fused_score = self.multi_cue_attractiveness(
            aest_face,
            aest_full,
            lighting,
            sharpness,
        )

        return {
            "best_frame_idx": best_idx,
            "lighting": float(lighting),
            "sharpness": float(sharpness),
            "face_area_frac": float(face_area_frac),
            "center_offset_norm": float(center_offset_norm),
            "aesthetic_face_0_10": float(aest_face),
            "aesthetic_full_0_10": float(aest_full),
            "multi_cue_attr_0_10": float(fused_score),
        }

# Global analyzer instance
attractiveness_analyzer = AttractivenessAnalyzer()