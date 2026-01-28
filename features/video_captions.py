"""
Video caption detection module.
Detects and analyzes dynamic captions in video content using OCR.
"""
import os
import cv2
import difflib
import numpy as np
from typing import List, Tuple, Dict, Any

from utils.video_utils import sample_bottom_frames
from config import (
    TARGET_FPS, BOTTOM_CROP_RATIO, MIN_TEXT_LEN, SIMILARITY_SAME_SEGMENT,
    MIN_SEGMENT_DURATION, MAX_SEGMENT_DURATION, CAPTION_MIN_COVERAGE,
    STATIC_OVERLAY_MAX_SEGMENTS, STATIC_DOMINANCE_RATIO, DEVICE
)

# EasyOCR import
try:
    from easyocr import Reader
    ocr = Reader(['en'], gpu=False if DEVICE == 'cpu' else True)
    OCR_AVAILABLE = True
    print("EasyOCR reader initialised for caption detection.")
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: EasyOCR not available, caption detection will be disabled")

class VideoCaptionAnalyzer:
    """Analyzes dynamic captions in video content."""
    
    def __init__(self):
        self.ocr_available = OCR_AVAILABLE
        self.target_fps = TARGET_FPS
        self.bottom_crop_ratio = BOTTOM_CROP_RATIO
        self.min_text_len = MIN_TEXT_LEN
        self.similarity_threshold = SIMILARITY_SAME_SEGMENT
        self.min_segment_duration = MIN_SEGMENT_DURATION
        self.max_segment_duration = MAX_SEGMENT_DURATION
        self.caption_min_coverage = CAPTION_MIN_COVERAGE
        self.static_max_segments = STATIC_OVERLAY_MAX_SEGMENTS
        self.static_dominance_ratio = STATIC_DOMINANCE_RATIO
    
    def ocr_caption_text(self, frame_bgr) -> str:
        """
        Run OCR using EasyOCR on a bottom-band BGR frame.
        Returns a cleaned, lowercased text string (or "" if none / error).
        """
        if not self.ocr_available:
            return ""
        
        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # EasyOCR expects an RGB numpy array
            results = ocr.readtext(frame_rgb, detail=1)

            if not results:
                return ""

            texts = []
            for item in results:
                # EasyOCR sometimes returns (bbox, text, conf) or [bbox, text, conf]
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                text = item[1]
                if not isinstance(text, str):
                    continue

                text_clean = text.strip()
                if len(text_clean) < self.min_text_len:
                    continue

                texts.append(text_clean)

            if not texts:
                return ""

            joined = " ".join(texts)
            joined = joined.lower()
            joined = " ".join(joined.split())
            return joined

        except Exception:
            # Frame-level OCR errors should not kill the whole reel
            return ""
    
    def build_caption_timeline(self, video_path: str) -> Tuple[List[Tuple[float, str]], float]:
        """
        For a video with OPTIMIZED sampling:
          1. Sample frames from the bottom band with smart sampling.
          2. OCR each sampled frame with early exit optimization.

        Returns:
          timeline: list of (time_sec, text) (text may be "")
          duration: total video duration in seconds
        """
        frames, duration = sample_bottom_frames(
            video_path, 
            target_fps=self.target_fps,
            bottom_ratio=self.bottom_crop_ratio
        )
        if not frames or duration <= 0:
            return [], 0.0

        timeline: List[Tuple[float, str]] = []
        
        # OPTIMIZATION: Smart sampling with early exit
        caption_found = False
        consecutive_empty = 0
        max_consecutive_empty = 5  # Stop after 5 consecutive empty frames
        
        for i, (t_sec, frame) in enumerate(frames):
            if frame is None:
                timeline.append((t_sec, ""))
                consecutive_empty += 1
            else:
                text = self.ocr_caption_text(frame)
                timeline.append((t_sec, text))
                
                if text.strip():
                    caption_found = True
                    consecutive_empty = 0
                else:
                    consecutive_empty += 1
            
            # OPTIMIZATION: Early exit if we found captions and then many empty frames
            if caption_found and consecutive_empty >= max_consecutive_empty:
                # Fill remaining timeline with empty entries for consistency
                remaining_frames = frames[i+1:]
                for t_sec_remaining, _ in remaining_frames:
                    timeline.append((t_sec_remaining, ""))
                break

        return timeline, duration
    
    def text_similarity(self, a: str, b: str) -> float:
        """Simple normalized similarity between two strings in [0, 1]."""
        return difflib.SequenceMatcher(None, a, b).ratio()
    
    def timeline_to_segments(
        self,
        timeline: List[Tuple[float, str]],
        duration: float,
    ) -> List[Dict[str, Any]]:
        """
        Convert (time, text) timeline into segments where text is mostly the same.
        Each segment: {"text": str, "start": float, "end": float}
        """
        if not timeline or duration <= 0:
            return []

        segments: List[Dict[str, Any]] = []

        current_text: str | None = None
        current_start: float | None = None
        last_time: float | None = None

        for item in timeline:
            # Extra defensive unpack
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            t_sec, raw_text = item
            text = (raw_text or "").strip()
            if len(text) < self.min_text_len:
                text = ""

            # If we have no active segment yet
            if current_text is None:
                if text:
                    current_text = text
                    current_start = t_sec
                    last_time = t_sec
                else:
                    last_time = t_sec
                continue

            # We are in a segment already
            sim = self.text_similarity(text, current_text) if text and current_text else 0.0

            if text and sim >= self.similarity_threshold:
                # same logical caption, extend segment
                last_time = t_sec
            else:
                # close previous segment
                end_time = last_time if last_time is not None else t_sec
                seg_duration = max(0.0, end_time - (current_start or 0.0))
                if seg_duration >= self.min_segment_duration and current_text:
                    segments.append(
                        {
                            "text": current_text,
                            "start": current_start,
                            "end": min(end_time, duration),
                        }
                    )

                # start new segment if there is new text; otherwise reset
                if text:
                    current_text = text
                    current_start = t_sec
                    last_time = t_sec
                else:
                    current_text = None
                    current_start = None
                    last_time = t_sec

        # close final open segment
        if current_text is not None and current_start is not None:
            end_time = last_time if last_time is not None else duration
            seg_duration = max(0.0, end_time - current_start)
            if seg_duration >= self.min_segment_duration:
                segments.append(
                    {
                        "text": current_text,
                        "start": current_start,
                        "end": min(end_time, duration),
                    }
                )

        # Clip extremely long segments
        for seg in segments:
            if seg["end"] - seg["start"] > self.max_segment_duration:
                seg["end"] = seg["start"] + self.max_segment_duration

        return segments
    
    def classify_caption_style(
        self,
        segments: List[Dict[str, Any]],
        duration: float,
    ) -> Dict[str, Any]:
        """
        Decide if the reel has dynamic captions, static overlay, or no captions.
        """
        if duration <= 0 or not segments:
            return {
                "has_dynamic_captions": False,
                "style": "none",
                "num_segments": 0,
                "caption_coverage": 0.0,
                "segments": [],
            }

        # compute durations per segment
        seg_durations = []
        total_caption_time = 0.0
        for seg in segments:
            d = max(0.0, float(seg["end"] - seg["start"]))
            seg_durations.append(d)
            total_caption_time += d

        if total_caption_time <= 0.0:
            return {
                "has_dynamic_captions": False,
                "style": "none",
                "num_segments": len(segments),
                "caption_coverage": 0.0,
                "segments": segments,
            }

        caption_coverage = float(total_caption_time / duration)
        num_segments = len(segments)

        # Very tiny coverage → treat as no captions
        if caption_coverage < self.caption_min_coverage:
            return {
                "has_dynamic_captions": False,
                "style": "none",
                "num_segments": num_segments,
                "caption_coverage": caption_coverage,
                "segments": segments,
            }

        dominant_ratio = max(seg_durations) / total_caption_time if total_caption_time > 0 else 0.0

        # Heuristic: if one segment dominates and there are few segments → static overlay
        if num_segments <= self.static_max_segments and dominant_ratio >= self.static_dominance_ratio:
            style = "static"
            has_dynamic = False
        else:
            style = "dynamic"
            has_dynamic = True

        return {
            "has_dynamic_captions": has_dynamic,
            "style": style,
            "num_segments": num_segments,
            "caption_coverage": caption_coverage,
            "segments": segments,
        }
    
    def analyze_reel_captions(self, video_path: str) -> Dict[str, Any]:
        """
        Wrapper: given a local mp4 path, compute dynamic caption classification.
        """
        timeline, duration = self.build_caption_timeline(video_path)
        segments = self.timeline_to_segments(timeline, duration)
        classification = self.classify_caption_style(segments, duration)
        classification["duration"] = duration
        return classification
    
    def compute_video_caption_flag_for_reel(self, video_path: str) -> Dict[str, Any]:
        """
        Joint-pipeline friendly wrapper for video caption analysis.
        """
        if not video_path or not os.path.exists(video_path):
            print("    ✗ Video path does not exist:", video_path)
            return {
                "has_dynamic_captions": np.nan,
                "caption_style": "none",
                "num_segments": 0,
                "caption_coverage": np.nan,
            }

        if not self.ocr_available:
            print("    ✗ OCR not available, skipping caption analysis")
            return {
                "has_dynamic_captions": np.nan,
                "caption_style": "none",
                "num_segments": 0,
                "caption_coverage": np.nan,
            }

        try:
            info = self.analyze_reel_captions(video_path)
        except Exception as e:
            print(f"[ERROR] Caption detection failed for {video_path}: {e}")
            return {
                "has_dynamic_captions": np.nan,
                "caption_style": "none",
                "num_segments": 0,
                "caption_coverage": np.nan,
            }

        return {
            # Cast to int for easy averaging
            "has_dynamic_captions": int(bool(info.get("has_dynamic_captions", False))),
            "caption_style": info.get("style", "none"),
            "num_segments": int(info.get("num_segments", 0)),
            "caption_coverage": float(info.get("caption_coverage", 0.0)),
        }

# Global analyzer instance
video_caption_analyzer = VideoCaptionAnalyzer()