"""
Accessories detection module.
Detects various accessories and objects in video frames using YOLO.
"""
import os
from collections import defaultdict
from typing import Dict, List
import cv2
import numpy as np

from utils.video_utils import sample_frames_from_video
from config import ACCESSORY_CLASSES, CLASS_BUCKET, FRAME_SAMPLE_COUNT

# YOLO import
try:
    from ultralytics import YOLO
    # Load YOLO model (you may need to specify the correct model path)
    yolo_model = YOLO('yolov8n.pt')  # or your custom model
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: YOLO not available, accessory detection will be disabled")

class AccessoryAnalyzer:
    """Analyzes accessories and objects in video frames."""
    
    def __init__(self, frames_per_reel: int = FRAME_SAMPLE_COUNT):
        self.frames_per_reel = frames_per_reel
        self.yolo_available = YOLO_AVAILABLE
        self.accessory_classes = ACCESSORY_CLASSES
        self.class_bucket = CLASS_BUCKET
    
    def detect_accessories_in_frame(self, frame_bgr: np.ndarray) -> Dict[str, int]:
        """
        Detect accessories in a single frame using YOLO.
        Returns count of each detected accessory class.
        """
        if not self.yolo_available:
            return {}
        
        try:
            # Run YOLO detection
            results = yolo_model(frame_bgr, verbose=False)
            
            # Count detections by class
            class_counts = defaultdict(int)
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Get class name
                        class_id = int(box.cls[0])
                        class_name = yolo_model.names[class_id]
                        
                        # Check if it's an accessory we're interested in
                        if class_name in self.accessory_classes:
                            class_counts[class_name] += 1
            
            return dict(class_counts)
            
        except Exception as e:
            print(f"    ✗ Error in accessory detection: {e}")
            return {}
    
    def aggregate_to_buckets(self, class_counts: Dict[str, int]) -> Dict[str, int]:
        """
        Aggregate individual class counts into broader buckets.
        """
        bucket_counts = defaultdict(int)
        
        for class_name, count in class_counts.items():
            bucket = self.class_bucket.get(class_name, "other")
            bucket_counts[bucket] += count
        
        return dict(bucket_counts)
    
    def compute_accessories_for_reel(self, video_path: str) -> Dict[str, float]:
        """
        Compute accessory metrics for a video reel with BATCH PROCESSING OPTIMIZATION.
        
        Returns:
            Dict with average counts per bucket (clothing, jewellery, gadgets, etc.)
        """
        if not video_path or not os.path.exists(video_path):
            print("    ✗ Video path does not exist:", video_path)
            return {
                "avg_clothing_per_reel": np.nan,
                "avg_jewellery_per_reel": np.nan,
                "avg_gadgets_per_reel": np.nan,
                "avg_vehicles_per_reel": np.nan,
                "avg_travel_gear_per_reel": np.nan,
            }
        
        if not self.yolo_available:
            print("    ✗ YOLO not available, skipping accessory detection")
            return {
                "avg_clothing_per_reel": np.nan,
                "avg_jewellery_per_reel": np.nan,
                "avg_gadgets_per_reel": np.nan,
                "avg_vehicles_per_reel": np.nan,
                "avg_travel_gear_per_reel": np.nan,
            }
        
        # Sample frames from video
        frames = sample_frames_from_video(video_path, max_frames=self.frames_per_reel)
        
        if not frames:
            print("    ✗ No frames sampled from video")
            return {
                "avg_clothing_per_reel": np.nan,
                "avg_jewellery_per_reel": np.nan,
                "avg_gadgets_per_reel": np.nan,
                "avg_vehicles_per_reel": np.nan,
                "avg_travel_gear_per_reel": np.nan,
            }
        
        # OPTIMIZATION: Batch YOLO processing instead of frame-by-frame
        total_bucket_counts = defaultdict(int)
        
        try:
            # Process all frames in a single batch call
            results = yolo_model(frames, verbose=False)
            
            # Process results for each frame
            for result in results:
                class_counts = defaultdict(int)
                
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        class_name = yolo_model.names.get(class_id, "unknown")
                        
                        if class_name in self.accessory_classes:
                            class_counts[class_name] += 1
                
                # Aggregate to buckets
                bucket_counts = self.aggregate_to_buckets(class_counts)
                for bucket, count in bucket_counts.items():
                    total_bucket_counts[bucket] += count
                    
        except Exception as e:
            print(f"    ⚠️ Batch YOLO processing failed, falling back to individual frames: {e}")
            # Fallback to individual frame processing
            for frame in frames:
                class_counts = self.detect_accessories_in_frame(frame)
                bucket_counts = self.aggregate_to_buckets(class_counts)
                
                for bucket, count in bucket_counts.items():
                    total_bucket_counts[bucket] += count
        
        # Calculate averages per frame
        num_frames = len(frames)
        avg_counts = {}
        
        for bucket in ["clothing", "jewellery", "gadgets", "vehicles", "travel_gear"]:
            avg_counts[f"avg_{bucket}_per_reel"] = float(total_bucket_counts[bucket] / num_frames)
        
        return avg_counts

# Global analyzer instance
accessory_analyzer = AccessoryAnalyzer()