"""
Updated main entry point for the Instagram Reel Feature Extraction System.
Incorporates all the new features and improvements from the updated notebook.
"""
import os
import re
import json
import time
import cv2
import queue
import threading
import requests
import torch
import whisper
import pandas as pd
import numpy as np
import yt_dlp
import hashlib
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from easyocr import Reader
from dotenv import load_dotenv
from apify_client import ApifyClient
from ultralytics import YOLO
from mine_redis import get_files_gem

from features.creativity import creativity_analyzer
from features.gemini_analysis import gemini_analyzer
from features.marketing_tendency import analyze_creator_marketing_tendency
from config import (
    GEMINI_API_KEY, APIFY_API_KEY, REEL_DOWNLOAD_DIR, APIFY_CACHE_DIR, DEVICE,
    MAX_REELS_PER_CREATOR, FRAME_SAMPLE_COUNT, MAX_DOWNLOAD_WORKERS, MAX_POSTS_FOR_MARKETING_ANALYSIS,
    COMMENTS_CACHE_DIR
)

# --- CONFIGURATION ---
load_dotenv()

# Paths
INPUT_CSV = "new_creators.csv"
OUTPUT_CSV = "final_creator_scores.csv"

# Updated paths to match new structure
REEL_DOWNLOAD_DIR = "./_reel_cache"
APIFY_CACHE_DIR = "cache_apify"
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

CHECKPOINT_REELS = CHECKPOINT_DIR / "processed_reels.parquet"
CHECKPOINT_FRAMES = CHECKPOINT_DIR / "processed_frames.parquet"
CHECKPOINT_PROGRESS = CHECKPOINT_DIR / "progress.txt"

os.makedirs(REEL_DOWNLOAD_DIR, exist_ok=True)
os.makedirs(APIFY_CACHE_DIR, exist_ok=True)
os.makedirs(COMMENTS_CACHE_DIR, exist_ok=True)

# Performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.download_stats = defaultdict(list)
        self.processing_times = defaultdict(list)
        self.start_time = time.time()
    
    def log_download_batch(self, creator, total_reels, successful_downloads, batch_time):
        success_rate = successful_downloads / total_reels if total_reels > 0 else 0
        self.download_stats[creator] = {
            'total': total_reels,
            'successful': successful_downloads,
            'success_rate': success_rate,
            'batch_time': batch_time,
            'downloads_per_second': successful_downloads / batch_time if batch_time > 0 else 0
        }
        
        print(f"📊 Download Performance for @{creator}:")
        print(f"   Success Rate: {success_rate:.1%} ({successful_downloads}/{total_reels})")
        print(f"   Batch Time: {batch_time:.1f}s")
        print(f"   Speed: {successful_downloads / batch_time:.1f} downloads/sec")

perf_monitor = PerformanceMonitor()

# --- GLOBAL MODEL LOADING ---
# Configuration values imported from config.py
MAX_POSTS_PER_CREATOR = 20

print(f"🚀 Loading Whisper (Medium) on {DEVICE}...")
whisper_model = whisper.load_model("medium", device=DEVICE)

print("🚀 Loading EasyOCR...")
ocr_reader = Reader(['en'], gpu=(DEVICE == "cuda"))

print("🚀 Loading Face Cascades...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

if APIFY_API_KEY:
    apify_client = ApifyClient(APIFY_API_KEY)

# Load YOLO model for accessories
print("🚀 Loading YOLO model for accessories...")
yolo_model = YOLO('yolov8n.pt')

# Aesthetic predictor stub (replace with actual implementation if available)
def predict_aesthetic(img):
    """Stub for aesthetic predictor - returns a random score between 0-10"""
    import random
    return random.uniform(0, 10)

# Transcript cache
_transcript_cache = {}

def normalize_creator(x: str) -> str:
    return str(x).strip().lstrip("@").lower()

# =============================================================================
# CHECKPOINT SYSTEM
# =============================================================================
def save_checkpoint(all_rows, sun_frame_rows, last_completed_creator):
    """Save current progress to checkpoint files"""
    if all_rows:
        df_reels = pd.DataFrame(all_rows)
        df_reels.to_parquet(CHECKPOINT_REELS, index=False)
        print(f"💾 Saved {len(all_rows)} reels to checkpoint")
    
    if sun_frame_rows:
        df_frames = pd.DataFrame(sun_frame_rows)
        df_frames.to_parquet(CHECKPOINT_FRAMES, index=False)
        print(f"💾 Saved {len(sun_frame_rows)} frame records to checkpoint")
    
    # Save progress
    with open(CHECKPOINT_PROGRESS, 'w') as f:
        f.write(last_completed_creator)
    print(f"💾 Progress saved: last completed creator = {last_completed_creator}")

def load_checkpoint():
    """Load existing checkpoint data"""
    all_rows = []
    sun_frame_rows = []
    last_completed = None
    
    if CHECKPOINT_REELS.exists():
        df_reels = pd.read_parquet(CHECKPOINT_REELS)
        all_rows = df_reels.to_dict('records')
        print(f"📂 Loaded {len(all_rows)} existing reels from checkpoint")
    
    if CHECKPOINT_FRAMES.exists():
        df_frames = pd.read_parquet(CHECKPOINT_FRAMES)
        sun_frame_rows = df_frames.to_dict('records')
        print(f"📂 Loaded {len(sun_frame_rows)} existing frame records from checkpoint")
    
    if CHECKPOINT_PROGRESS.exists():
        with open(CHECKPOINT_PROGRESS, 'r') as f:
            last_completed = f.read().strip()
        print(f"📂 Last completed creator: {last_completed}")
    
    return all_rows, sun_frame_rows, last_completed

def get_remaining_creators(creator_list, last_completed):
    """Get list of creators still to process"""
    if not last_completed:
        return creator_list
    
    try:
        last_idx = creator_list.index(last_completed)
        remaining = creator_list[last_idx + 1:]
        print(f"🔄 Resuming from creator #{last_idx + 2}: {remaining[0] if remaining else 'DONE'}")
        return remaining
    except ValueError:
        print(f"⚠️ Last completed creator '{last_completed}' not found in list, starting from beginning")
        return creator_list

# =============================================================================
# APIFY SCRAPING WITH CACHING
# =============================================================================
def flatten_comments(comment_list, max_n: int = 50):
    """Convert a single comment list (Apify objects) → simple text list."""
    if not isinstance(comment_list, list):
        return []
    out = []
    for c in comment_list[:max_n]:
        if isinstance(c, dict):
            txt = c.get("text") or c.get("body") or ""
            txt = (txt or "").strip()
            if txt:
                out.append(txt)
    return out

def load_or_fetch_reels_cached(creator: str, max_items: int) -> pd.DataFrame:
    """Cache wrapper around fetch_reels_from_apify."""
    # Clean creator name for safe file path
    safe_creator = creator.replace('/', '_').replace('\\', '_').replace(':', '_').replace('?', '_').replace('*', '_').replace('<', '_').replace('>', '_').replace('|', '_').replace('"', '_')
    cache_path = Path(APIFY_CACHE_DIR) / f"{safe_creator}_max{max_items}.parquet"

    if cache_path.exists():
        print(f"📂 Using cached Apify reels for @{creator} from {cache_path}")
        return pd.read_parquet(cache_path)

    # Fallback: real network call once
    df = fetch_reels_from_apify(creator, max_items=max_items)
    if not df.empty:
        df.to_parquet(cache_path, index=False)
        print(f"💾 Cached Apify reels for @{creator} → {cache_path}")
    else:
        print(f"⚠️ No reels for @{creator}, nothing cached.")
    return df

def fetch_reels_from_apify(handle: str, max_items: int = MAX_REELS_PER_CREATOR) -> pd.DataFrame:
    print(f"\n📸 Fetching reels for @{handle} via Apify...")

    try:
        run_input = {
            "username": [handle],
            "resultsLimit": max_items,
        }

        run = apify_client.actor("xMc5Ga1oCONPmWJIa").call(run_input=run_input)
        items = apify_client.dataset(run["defaultDatasetId"]).list_items().items

        if not items:
            print("  ✗ No items returned.")
            return pd.DataFrame()

        df = pd.DataFrame(items)

        # Process reel_url
        if "url" in df.columns:
            df["reel_url"] = df["url"]
        elif "shortcode" in df.columns:
            df["reel_url"] = "https://www.instagram.com/reel/" + df["shortcode"].astype(str) + "/"
        else:
            df["reel_url"] = None

        # Process caption
        if "caption" in df.columns:
            df["caption_norm"] = df["caption"].fillna("")
        else:
            df["caption_norm"] = ""

        # Process comments
        comment_fields = ["latestComments", "comments", "deepLatestComments", "deepComments"]
        comment_fields = [c for c in comment_fields if c in df.columns]

        if comment_fields:
            def collect_all_comments(row, max_total=100):
                texts = []
                for col in comment_fields:
                    texts.extend(flatten_comments(row.get(col), max_n=50))
                # de-duplicate while preserving order
                seen = set()
                uniq = []
                for t in texts:
                    if t not in seen:
                        seen.add(t)
                        uniq.append(t)
                    if len(uniq) >= max_total:
                        break
                return uniq

            df["flat_comments"] = df.apply(collect_all_comments, axis=1)
        else:
            df["flat_comments"] = [[] for _ in range(len(df))]

        # Filter valid URLs
        mask = (
            df["reel_url"].notna()
            & (
                df["reel_url"].str.contains("/reel/")
                | df["reel_url"].str.contains("/p/")
            )
        )

        out = df.loc[mask, ["reel_url", "caption_norm", "flat_comments"]].copy()
        out = out.rename(columns={"caption_norm": "caption"})
        out = out.reset_index(drop=True)

        print(f"  ✓ {len(out)} valid reels for @{handle}")
        return out

    except Exception as e:
        print(f"  ✗ Apify error for @{handle}: {e}")
        return pd.DataFrame()

# =============================================================================
# DOWNLOAD SYSTEM
# =============================================================================
_download_cache = {}

def download_reel_cached(reel_url: str, reel_no: int, task_id: str = "joint") -> str | None:
    """Download reel with caching using get_files_gem"""
    if reel_url in _download_cache:
        cached_path = _download_cache[reel_url]
        print(f"    💾 Using cached download: {cached_path}")
        if os.path.exists(cached_path):
            return cached_path
        else:
            print(f"    ❌ Cached file missing: {cached_path}")
            del _download_cache[reel_url]

    print(f"    🔽 Downloading: {reel_url}")
    try:
        # Import get_files_gem dynamically to avoid import errors if not available
        try:
            from mine_redis_simple import get_files_gem
        except ImportError:
            from mine_redis import get_files_gem
        
        out = get_files_gem(REEL_URL=reel_url, REEL_NO=str(reel_no), task_id=task_id)
        
        if not out:
            print(f"    ❌ Download failed: get_files_gem returned None/False")
            return None

        # get_files_gem may return a path or a dict with 'path'
        if isinstance(out, dict):
            path = out.get("path")
        else:
            path = out

        if not path or not os.path.exists(path):
            print(f"    ❌ Invalid path or file doesn't exist: {path}")
            return None

        local_path = os.path.abspath(path)
        print(f"    ✅ Valid download: {local_path}")
        _download_cache[reel_url] = local_path
        return local_path
        
    except ImportError:
        print(f"    ❌ get_files_gem not available, skipping download")
        return None
    except Exception as e:
        print(f"    ❌ Download failed: {e}")
        return None

# =============================================================================
# POSTS VS REELS RATIO FEATURE
# =============================================================================
def _norm_creator(x: str) -> str:
    return str(x).strip().lstrip("@").lower()

def fetch_all_posts_for_creators(creators, max_posts: int = MAX_POSTS_PER_CREATOR) -> dict:
    """
    For a list of IG handles, fetch up to `max_posts` recent posts per handle
    using Apify instagram-scraper.
    """
    if not APIFY_API_KEY:
        raise RuntimeError("Missing APIFY_API_KEY for posts ratio module")
    
    client = ApifyClient(APIFY_API_KEY)
    profile_urls = [f"https://www.instagram.com/{u.lstrip('@')}/" for u in creators]
    print(f"🚀 Scraping up to {max_posts} posts for {len(creators)} creators...")

    run_input = {
        "directUrls": profile_urls,
        "resultsLimit": max_posts,
        "addUserInfo": True,
        "addLocation": False,
        "addLikes": False,
        "addVideoThumbnails": False,
        "proxyConfiguration": {"useApifyProxy": True},
    }

    try:
        run = client.actor("apify/instagram-scraper").call(run_input=run_input)
        items = client.dataset(run["defaultDatasetId"]).list_items().items

        grouped = {}
        for it in items:
            if not isinstance(it, dict):
                continue
            user = (it.get("ownerUsername") or it.get("username") or "unknown")
            user = _norm_creator(user)
            grouped.setdefault(user, []).append(it)

        print(f"✓ Got {len(items)} posts across {len(grouped)} creators")
        return grouped
    except Exception as e:
        print(f"✗ Error fetching posts: {e}")
        return {}

def _is_reel_item(it: dict) -> bool:
    """Robust reel detection"""
    if not isinstance(it, dict):
        return False

    # 1) Explicit boolean
    if it.get("isReel") is True:
        return True

    # 2) productType (commonly 'clips' for reels)
    pt = str(it.get("productType") or it.get("product_type") or "").lower()
    if "clips" in pt or pt in ("reel", "reels"):
        return True

    # 3) type / __typename strings
    t = str(it.get("type") or "").lower()
    tn = str(it.get("__typename") or "").lower()
    if "reel" in t or "reel" in tn:
        return True

    return False

def summarise_posts_by_type(posts_by_user: dict) -> pd.DataFrame:
    """Summarize posts by type for each creator"""
    rows = []

    for user, items in posts_by_user.items():
        user = _norm_creator(user)
        total_posts = len(items)

        reel_count = 0
        static_count = 0
        other_count = 0

        for it in items:
            if not isinstance(it, dict):
                continue

            if _is_reel_item(it):
                reel_count += 1
                continue

            # Non-reel buckets
            ig_type = str(it.get("type") or "").lower()
            typename = str(it.get("__typename") or "").lower()

            if ("graphimage" in ig_type) or ("graphimage" in typename):
                static_count += 1
            elif ("graphsidecar" in ig_type) or ("graphsidecar" in typename):
                static_count += 1
            elif ("graphvideo" in ig_type) or ("graphvideo" in typename):
                static_count += 1  # feed video (non-reel)
            else:
                other_count += 1

        rows.append({
            "creator": user,
            "total_posts_sampled": int(total_posts),
            "reels_sampled": int(reel_count),
            "static_posts_sampled": int(static_count),
            "other_posts_sampled": int(other_count),
        })

    return pd.DataFrame(rows)

def compute_static_reel_ratios(posts_by_user: dict) -> pd.DataFrame:
    """Compute ratios from posts data"""
    df = summarise_posts_by_type(posts_by_user)
    if df.empty:
        return df

    denom = df["total_posts_sampled"].replace({0: np.nan})
    df["reels_ratio"] = df["reels_sampled"] / denom
    df["static_ratio"] = df["static_posts_sampled"] / denom
    df["creator_norm"] = df["creator"].apply(_norm_creator)

    return df

# =============================================================================
# FEATURE EXTRACTION MODULES
# =============================================================================

def transcribe_reel(video_path: str, reel_url: str | None = None) -> str:
    """Transcribe audio from video using Whisper with debug info."""
    
    print(f"    🔍 Checking video file: {video_path}")
    if not os.path.exists(video_path):
        print(f"    ❌ File does not exist: {video_path}")
        return ""
    
    file_size = os.path.getsize(video_path)
    print(f"    📁 File exists, size: {file_size} bytes")
    
    if file_size == 0:
        print(f"    ❌ File is empty (0 bytes)")
        return ""
    
    # Check cache first
    cache_key = video_path
    if cache_key in _transcript_cache:
        print(f"    💾 Using cached transcript")
        return _transcript_cache[cache_key]
    
    try:
        print(f"    🎤 Starting Whisper transcription...")
        use_fp16 = torch.cuda.is_available()
        
        result = whisper_model.transcribe(video_path, fp16=use_fp16)
        
        text = (result.get("text") or "").strip()
        print(f"    ✅ Transcription complete: {len(text)} characters")
        
        # Cache the result
        _transcript_cache[cache_key] = text
        return text
        
    except Exception as e:
        print(f"    ✗ Whisper transcription failed: {e}")
        return ""

def compute_spoken_word_count(transcript: str | None) -> int:
    """Return the number of spoken 'word' tokens in the transcript."""
    if not transcript:
        return 0
    
    # Clean transcript (remove music indicators)
    cleaned = transcript.lower()
    music_indicators = ["[music]", "[applause]", "bgm", "instrumental"]
    for indicator in music_indicators:
        cleaned = cleaned.replace(indicator, "")
    
    tokens = re.findall(r"\w+", cleaned)
    return len(tokens)

def is_music_only_transcript(transcript: str) -> bool:
    """Check if transcript is music-only."""
    if not transcript:
        return True
    
    word_count = compute_spoken_word_count(transcript)
    music_indicators = ["[music]", "[applause]", "bgm", "instrumental"]
    
    if word_count < 5 or any(indicator in transcript.lower() for indicator in music_indicators):
        return True
    
    return False

def english_percentage(transcript: str) -> float:
    """Compute English percentage in transcript."""
    if not transcript:
        return 0.0
    
    # Simple heuristic based on common English words
    words = re.findall(r'\w+', transcript.lower())
    if not words:
        return 0.0
    
    common_english = {"the", "is", "and", "to", "of", "a", "in", "it", "my", "so", "you", "that", "have"}
    english_count = sum(1 for word in words if word in common_english)
    
    return (english_count / len(words)) * 100.0

def detect_series_from_text(caption: str, transcript: str, comments=None) -> dict:
    """Detect series/episode patterns in text."""
    combined = (caption + " " + transcript).lower()
    
    # Look for series indicators
    series_patterns = [r"(part|episode|day)\s*\d+", r"series", r"ep\s*\d+"]
    
    series_flag = 0
    episode_number = None
    
    for pattern in series_patterns:
        match = re.search(pattern, combined)
        if match:
            series_flag = 1
            # Try to extract episode number
            num_match = re.search(r"\d+", match.group())
            if num_match:
                episode_number = int(num_match.group())
            break
    
    return {
        "series_flag": series_flag,
        "episode_number": episode_number
    }

def compute_sun_exposure_for_reel(video_path: str) -> dict:
    """Compute sun exposure metrics for a reel."""
    if not os.path.exists(video_path):
        return {
            "sun_exposure_raw_A": 0.0,
            "sun_exposure_0_10_A": 0.0,
            "sun_frame_scores": []
        }
    
    # Sample frames and compute sun exposure
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "sun_exposure_raw_A": 0.0,
            "sun_exposure_0_10_A": 0.0,
            "sun_frame_scores": []
        }
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return {
            "sun_exposure_raw_A": 0.0,
            "sun_exposure_0_10_A": 0.0,
            "sun_frame_scores": []
        }
    
    # Sample frames uniformly
    sample_count = min(FRAME_SAMPLE_COUNT, total_frames)
    indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)
    
    frame_scores = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Simple sun exposure calculation based on brightness
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            brightness = hsv[:, :, 2].mean()
            sun_score = min(brightness / 255.0 * 10.0, 10.0)
            frame_scores.append(sun_score)
    
    cap.release()
    
    avg_score = np.mean(frame_scores) if frame_scores else 0.0
    
    return {
        "sun_exposure_raw_A": float(avg_score),
        "sun_exposure_0_10_A": float(min(avg_score, 10.0)),
        "sun_frame_scores": frame_scores
    }

def compute_eye_contact_for_reel(video_path: str) -> dict:
    """Compute eye contact metrics for a reel."""
    if not os.path.exists(video_path):
        return {
            "eye_contact_ratio": 0.0,
            "eye_contact_score_0_10": 0.0
        }
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "eye_contact_ratio": 0.0,
            "eye_contact_score_0_10": 0.0
        }
    
    total_frames = 0
    eye_contact_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        total_frames += 1
        if total_frames % 3 != 0:  # Sample every 3rd frame
            continue
        
        # Simple eye contact detection using face cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
        
        if len(faces) > 0:
            # If face detected, assume eye contact for simplicity
            eye_contact_frames += 1
    
    cap.release()
    
    ratio = eye_contact_frames / max(total_frames // 3, 1)
    score = ratio * 10.0
    
    return {
        "eye_contact_ratio": float(ratio),
        "eye_contact_score_0_10": float(score)
    }

def compute_video_caption_flag_for_reel(video_path: str) -> dict:
    """Compute video caption detection for a reel."""
    if not os.path.exists(video_path):
        return {
            "has_dynamic_captions": 0,
            "caption_style": "none",
            "num_segments": 0,
            "caption_coverage": 0.0
        }
    
    # Simple caption detection using OCR on bottom portion
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "has_dynamic_captions": 0,
            "caption_style": "none", 
            "num_segments": 0,
            "caption_coverage": 0.0
        }
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    caption_frames = 0
    
    # Sample a few frames
    for pos in [0.2, 0.5, 0.8]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * pos))
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            # Crop bottom 40% for captions
            bottom_crop = frame[int(h * 0.6):h, 0:w]
            
            try:
                results = ocr_reader.readtext(bottom_crop, detail=0)
                if any(len(text) > 2 for text in results):
                    caption_frames += 1
            except:
                pass
    
    cap.release()
    
    has_captions = 1 if caption_frames >= 2 else 0
    
    return {
        "has_dynamic_captions": has_captions,
        "caption_style": "dynamic" if has_captions else "none",
        "num_segments": caption_frames,
        "caption_coverage": caption_frames / 3.0
    }

def compute_accessories_for_reel(video_path: str) -> dict:
    """Compute accessory detection for a reel using YOLO."""
    if not os.path.exists(video_path):
        return {"total_accessories": 0}
    
    # Sample a few frames and run YOLO detection
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"total_accessories": 0}
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    accessory_counts = defaultdict(int)
    
    # Sample frames
    sample_indices = np.linspace(0, total_frames - 1, min(5, total_frames), dtype=int)
    
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            try:
                results = yolo_model(frame, verbose=False)
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            class_id = int(box.cls[0])
                            class_name = yolo_model.names[class_id]
                            accessory_counts[class_name] += 1
            except:
                pass
    
    cap.release()
    
    # Convert to individual accessory counts
    accessory_dict = {}
    accessory_classes = [
        "backpack", "handbag", "hat", "scarf", "sunglasses", "glasses",
        "necklace", "earrings", "watch", "bracelet", "ring", "wallet", "belt",
        "mobile_phone", "laptop", "tablet", "smartwatch", "headphones", "camera",
        "car", "sports_car", "motorcycle", "bike", "airplane", "boat",
        "suitcase", "luggage", "surfboard", "skis", "horse",
        "dress", "coat", "suit", "high_heels"
    ]
    
    for acc_class in accessory_classes:
        accessory_dict[acc_class] = accessory_counts.get(acc_class, 0)
    
    accessory_dict["total_accessories"] = sum(accessory_counts.values())
    
    return accessory_dict

def compute_attractiveness_for_reel(video_path: str) -> dict:
    """Compute attractiveness metrics for a reel using the enhanced composite system."""
    try:
        # Import the attractiveness analyzer
        from features.attractiveness import attractiveness_analyzer
        
        # Use the enhanced analyzer which includes composite scoring
        result = attractiveness_analyzer.compute_attractiveness_for_reel(video_path)
        
        # Map to expected output format for backward compatibility
        return {
            "attractiveness_score": result.get("multi_cue_attr_0_10", 0.0),
            "face_area_frac": result.get("face_area_frac", 0.0),
            "center_offset_norm": result.get("center_offset_norm", 1.0),
            "lighting_score": result.get("lighting", 0.0),
            "sharpness_score": result.get("sharpness", 0.0),
            "aesthetic_face": result.get("aesthetic_face_0_10", 0.0),
            "aesthetic_full": result.get("aesthetic_full_0_10", 0.0),
            # New composite metrics
            "face_aesthetic_score_0_10": result.get("face_aesthetic_score_0_10", 0.0),
            "bg_clutter_score": result.get("bg_clutter_score", 0.0),
            "bg_brightness_score": result.get("bg_brightness_score", 0.0),
            "bg_saturation_score": result.get("bg_saturation_score", 0.0),
            "composite_aesthetic_score": result.get("composite_aesthetic_score", 0.0),
            "face_samples_count": result.get("face_samples_count", 0),
            "bg_samples_count": result.get("bg_samples_count", 0),
        }
        
    except Exception as e:
        print(f"    ✗ Attractiveness analysis failed: {e}")
        return {
            "attractiveness_score": 0.0,
            "face_area_frac": 0.0,
            "center_offset_norm": 1.0,
            "lighting_score": 0.0,
            "sharpness_score": 0.0,
            "aesthetic_face": 0.0,
            "aesthetic_full": 0.0,
            "face_aesthetic_score_0_10": 0.0,
            "bg_clutter_score": 0.0,
            "bg_brightness_score": 0.0,
            "bg_saturation_score": 0.0,
            "composite_aesthetic_score": 0.0,
            "face_samples_count": 0,
            "bg_samples_count": 0,
        }

def fetch_deep_comments_apify(reel_url: str, max_comments: int = 150) -> list:
    """Fetch deep comments for a reel using Apify with caching."""
    
    # Create cache key from URL
    url_hash = hashlib.md5(reel_url.encode()).hexdigest()
    cache_file = Path(COMMENTS_CACHE_DIR) / f"{url_hash}_max{max_comments}.json"
    
    # Check cache first
    if cache_file.exists():
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_comments = json.load(f)
            print(f"    💾 Using cached comments: {len(cached_comments)} comments")
            return cached_comments
        except Exception as e:
            print(f"    ⚠️ Cache read failed: {e}, fetching fresh...")
    
    print(f"    💬 Fetching deep comments (max {max_comments})...")
    
    try:
        run_input = {
            "directUrls": [reel_url],
            "resultsLimit": max_comments,
        }
        
        run = apify_client.actor("apify/instagram-comment-scraper").call(run_input=run_input)
        items = apify_client.dataset(run["defaultDatasetId"]).list_items().items
        
        comments = []
        for item in items:
            if isinstance(item, dict):
                text = item.get("text", "").strip()
                if text:
                    comments.append(text)
        
        # Cache the results
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(comments, f, ensure_ascii=False, indent=2)
            print(f"    💾 Cached {len(comments)} comments to {cache_file}")
        except Exception as e:
            print(f"    ⚠️ Cache write failed: {e}")
        
        print(f"    ✓ Deep comments fetched: {len(comments)}")
        return comments
        
    except Exception as e:
        print(f"    ✗ Deep comments fetch failed: {e}")
        return []

def filter_top_comments_for_gemini(comments, max_total: int = 100, max_after_filter: int = 30) -> list:
    """Filter comments for Gemini analysis."""
    if not comments:
        return []
    
    # Convert to list of strings
    if isinstance(comments, (list, tuple)):
        comments_list = list(comments)
    else:
        comments_list = [comments]
    
    # Clean and filter
    cleaned = []
    for c in comments_list:
        if c is None:
            continue
        s = str(c).strip()
        if s and len(s) > 2:  # Filter out very short comments
            cleaned.append(s)
    
    # Take top comments and filter out emoji-only
    top_comments = cleaned[:max_total]
    
    # Simple emoji filter - remove comments that are mostly emojis
    filtered = []
    for comment in top_comments:
        # Count alphanumeric characters
        alnum_count = sum(1 for c in comment if c.isalnum())
        if alnum_count >= 3:  # At least 3 alphanumeric characters
            filtered.append(comment)
    
    return filtered[:max_after_filter]

# =============================================================================
# GEMINI HELPERS
# =============================================================================
def parse_gemini_raw(x):
    """Safely parse gemini_raw (string/dict) -> dict."""
    if isinstance(x, dict):
        return x
    if not isinstance(x, str) or not x.strip():
        return {}
    try:
        out = json.loads(x)
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}

def flatten_dict(d, parent_key="", sep="_"):
    """Flatten nested dicts into a single-level dict."""
    out = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        k = str(k)
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            out.update(flatten_dict(v, parent_key=new_key, sep=sep))
        else:
            out[new_key] = v
    return out

def call_gemini_for_reel(caption: str, transcript: str, comments: list) -> str:
    """Call Gemini for reel analysis."""
    if not gemini_client:
        return "{}"
    
    try:
        prompt = f"""
        Analyze this Instagram reel and return a JSON object with these fields:
        
        Caption: {caption}
        Transcript: {transcript}
        Comments: {comments[:10]}
        
        Return JSON with:
        {{
            "genz_word_count": <count of Gen-Z slang words>,
            "is_marketing": <0 or 1>,
            "is_educational": <0 or 1>,
            "is_vlog": <0 or 1>,
            "has_humour": <0 or 1>,
            "comment_sentiment_counts": {{
                "questioning": <count>,
                "agreeing": <count>,
                "appreciating": <count>,
                "negative": <count>,
                "neutral": <count>
            }}
        }}
        """
        
        response = gemini_client.models.generate_content(
            model="models/gemini-2.0-flash-001",
            contents=[{"parts": [{"text": prompt}]}]
        )
        
        if response and response.candidates:
            response_text = response.candidates[0].content.parts[0].text.strip()
            
            # Clean response
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            elif response_text.startswith("```"):
                response_text = response_text.replace("```", "").strip()
            
            # Validate JSON
            json.loads(response_text)
            return response_text
        
        return "{}"
        
    except Exception as e:
        print(f"    ✗ Gemini analysis failed: {e}")
        return "{}"



# =============================================================================
# MAIN PROCESSING LOOP
# =============================================================================
def process_creator_with_checkpoints(creator: str, all_rows: list, sun_frame_rows: list, creator_post_ratios: dict = None):
    """Process a single creator with all the new features."""
    creator_norm = normalize_creator(creator)
    
    print(f"\n{'='*60}")
    print(f"🎯 Processing creator: @{creator}")
    print('='*60)
    
    # 1) Get reels (Apify + manifest cache)
    df_reels = load_or_fetch_reels_cached(creator, max_items=MAX_REELS_PER_CREATOR)
    if df_reels.empty:
        print(f"❗ No reels for @{creator}, skipping.")
        return None
    
    df_batch = df_reels.head(MAX_REELS_PER_CREATOR).copy()
    print(f"📋 Found {len(df_batch)} reels for @{creator}")
    
    # 2) PRE-DOWNLOAD IN PARALLEL
    print(f"🔽 Pre-downloading reels for @{creator} with {MAX_DOWNLOAD_WORKERS} workers...")
    download_results = {}
    
    with ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS) as executor:
        future_to_idx = {}
        for reel_idx, row in df_batch.iterrows():
            reel_url = row["reel_url"]
            fut = executor.submit(
                download_reel_cached,
                reel_url=reel_url,
                reel_no=reel_idx,
                task_id=f"joint_{creator}_{reel_idx}",
            )
            future_to_idx[fut] = reel_idx
        
        downloads_completed = 0
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                path = fut.result()
                downloads_completed += 1
                if downloads_completed % 5 == 0:
                    print(f"  📥 Downloaded {downloads_completed}/{len(df_batch)} reels...")
            except Exception as e:
                print(f"  ✗ Download failed for @{creator} reel #{idx}: {e}")
                path = None
            download_results[idx] = path
    
    print(f"✅ Download phase complete: {sum(1 for p in download_results.values() if p)} successful")
    
    # 3) METRIC LOOP
    creator_reels_processed = 0
    creator_frame_data = []
    reel_results = []  # Store individual reel results
    
    for reel_idx, row in df_batch.iterrows():
        reel_url = row["reel_url"]
        caption = row.get("caption") or ""
        
        print(f"\n📊 Processing reel {creator_reels_processed + 1}/{len(df_batch)}: #{reel_idx}")
        
        video_path = download_results.get(reel_idx)
        if not video_path:
            print("  ✗ No local video_path for this reel (download failed or missing).")
            continue
        
        try:
            # Extract all features
            raw_comments = fetch_deep_comments_apify(reel_url, max_comments=150)
            transcript = transcribe_reel(video_path, reel_url=reel_url)
            
            is_music = is_music_only_transcript(transcript)
            word_count = compute_spoken_word_count(transcript)
            
            sun = compute_sun_exposure_for_reel(video_path)
            frame_scores = sun.pop("sun_frame_scores", [])
            
            eye = compute_eye_contact_for_reel(video_path)
            creat = creativity_analyzer.compute_creativity_for_reel(video_path)
            
            # Compute attractiveness metrics
            attract = compute_attractiveness_for_reel(video_path)
            
            series_info = detect_series_from_text(caption, transcript, raw_comments)
            caps_info = compute_video_caption_flag_for_reel(video_path)
            acc_counts = compute_accessories_for_reel(video_path)
            english_pct = english_percentage(transcript)
            
            filtered_comments_for_gemini = filter_top_comments_for_gemini(
                raw_comments,
                max_total=100,
                max_after_filter=30,
            )
            
            gemini_raw = call_gemini_for_reel(caption, transcript, filtered_comments_for_gemini)
            gemini_obj = parse_gemini_raw(gemini_raw)
            gemini_flat = flatten_dict(gemini_obj)
            
            # Build row output
            row_out = {
                "creator": creator,
                "reel_idx": reel_idx,
                "reel_url": reel_url,
                "caption": caption,
                "transcript": transcript,
                "flat_comments": filtered_comments_for_gemini,
                "raw_comments_full": raw_comments,
                "english_pct": english_pct,
                "word_count": word_count,
                "is_music_only": is_music,
                "series_flag": series_info.get("series_flag", 0),
                "episode_number": series_info.get("episode_number"),
            }
            
            # Add post ratios if available
            if creator_post_ratios:
                for key, value in creator_post_ratios.items():
                    if key not in ['creator', 'creator_norm']:  # Avoid duplicating creator info
                        row_out[key] = value
            
            # Add gemini_* columns
            for k, v in gemini_flat.items():
                row_out[f"gemini_{k}"] = v
            
            # Add other metrics
            row_out.update(sun)
            row_out.update(eye)
            row_out.update(creat)
            row_out.update(attract)
            row_out.update(caps_info)
            row_out.update(acc_counts)
            
            all_rows.append(row_out)
            reel_results.append(row_out)  # Store for aggregation
            
            # Add frame data
            for frame_idx, s in enumerate(frame_scores):
                creator_frame_data.append({
                    "creator": creator,
                    "reel_idx": reel_idx,
                    "frame_idx": frame_idx,
                    "sun_exposure_raw_A": float(s),
                })
            
            sun_frame_rows.extend(creator_frame_data)
            creator_reels_processed += 1
            
            print(f"  ✅ Reel #{reel_idx} processed successfully")
            
        except Exception as e:
            print(f"  ✗ Error processing reel #{reel_idx}: {e}")
            continue
    
    print(f"\n✅ Completed @{creator}: {creator_reels_processed} reels processed")
    
    # 4) ANALYZE MARKETING TENDENCY ACROSS ALL POSTS
    print(f"\n📊 Analyzing overall marketing tendency for @{creator}...")
    marketing_analysis = analyze_creator_marketing_tendency(
        creator, 
        gemini_client=gemini_client, 
        apify_client=apify_client, 
        max_posts=MAX_POSTS_FOR_MARKETING_ANALYSIS
    )
    
    # 5) COMPUTE OUTLIER RATIO FOR THIS CREATOR
    from features.creativity import compute_outlier_2sigma_ratio
    word_counts = [row.get('word_count', 0) for row in reel_results]
    outlier_ratio = compute_outlier_2sigma_ratio(word_counts) if word_counts else 0.0
    
    # 6) AGGREGATE RESULTS (like your example)
    if not reel_results:
        return None
    
    df = pd.DataFrame(reel_results)
    nm = df[~df['is_music_only']]  # Non-music reels
    
    # Return aggregated result (like your example script)
    result = {
        "creator": creator,
        "eye_contact_avg_score_0_10": df.get('eye_contact_score_0_10', pd.Series([0])).mean(),
        "series_reel_mean": df['series_flag'].mean(),
        "avg_captioned_reels": df.get('has_dynamic_captions', pd.Series([0])).mean(),
        "avg_english_pct_non_music": nm['english_pct'].mean() if not nm.empty else 0,
        "gemini_genz_word_count": df.get("gemini_genz_word_count", pd.Series([0])).mean(),
        "gemini_is_marketing": df.get("gemini_is_marketing", pd.Series([0])).mean(),
        "gemini_is_educational": df.get("gemini_is_educational", pd.Series([0])).mean(),
        "gemini_has_humour": df.get("gemini_has_humour", pd.Series([0])).mean(),
        "gemini_comment_sentiment_counts.agreeing": df.get("gemini_comment_sentiment_counts_agreeing", pd.Series([0])).mean(),
        "gemini_comment_sentiment_counts.appreciating": df.get("gemini_comment_sentiment_counts_appreciating", pd.Series([0])).mean(),
        "gemini_comment_sentiment_counts.neutral": df.get("gemini_comment_sentiment_counts_neutral", pd.Series([0])).mean(),
        "mean_hist_score": df.get("hist_score_0_10", pd.Series([0])).mean(),
        "mean_scene_score": df.get("scene_score_0_10", pd.Series([0])).mean(),
        "mean_face_density": df.get("face_frame_density", pd.Series([0])).mean(),
        # Additional metrics for completeness
        "avg_word_count": df['word_count'].mean(),
        "marketing_ratio": df.get("gemini_is_marketing", pd.Series([0])).mean(),
        "educational_ratio": df.get("gemini_is_educational", pd.Series([0])).mean(),
        "vlog_ratio": df.get("gemini_is_vlog", pd.Series([0])).mean(),
        "has_humour_any": df.get("gemini_has_humour", pd.Series([0])).max(),
        "comments_questioning": df.get("gemini_comment_sentiment_counts_questioning", pd.Series([0])).sum(),
        "comments_agreeing": df.get("gemini_comment_sentiment_counts_agreeing", pd.Series([0])).sum(),
        "comments_appreciating": df.get("gemini_comment_sentiment_counts_appreciating", pd.Series([0])).sum(),
        "comments_negative": df.get("gemini_comment_sentiment_counts_negative", pd.Series([0])).sum(),
        "comments_neutral": df.get("gemini_comment_sentiment_counts_neutral", pd.Series([0])).sum(),
        "avg_english_pct": nm['english_pct'].mean() if not nm.empty else 0,
        "mean_clip_score": df.get("clip_score_0_10", pd.Series([0])).mean(),
        "avg_sun_exposure": df.get("sun_exposure_0_10_A", pd.Series([0])).mean(),
        "avg_attractiveness": df.get("composite_aesthetic_score", pd.Series([0])).mean(),
        "avg_total_accessories": df.get("total_accessories", pd.Series([0])).mean(),
    }
    
    # Add marketing tendency analysis to the result
    result.update(marketing_analysis)
    
    # Add outlier ratio
    result['outlier_2sigma_ratio'] = outlier_ratio
    
    return result

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # 1. Validation
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Input file {INPUT_CSV} not found.")
        exit()
    
    df = pd.read_csv(INPUT_CSV)
    if 'creator' not in df.columns:
        print("❌ Missing 'creator' column")
        exit()
    
    # Get creator list
    creator_list = df['creator'].dropna().astype(str).apply(normalize_creator).unique().tolist()
    
    # Load checkpoint
    print("🔄 Loading checkpoint data...")
    all_rows, sun_frame_rows, last_completed_creator = load_checkpoint()
    
    # Get remaining creators
    remaining_creators = get_remaining_creators(creator_list, last_completed_creator)
    
    # Check for existing progress (resume capability)
    processed_users = set()
    if os.path.exists(OUTPUT_CSV):
        try:
            # Read existing file to see who is done
            existing_df = pd.read_csv(OUTPUT_CSV)
            if 'creator' in existing_df.columns:
                processed_users = set(existing_df['creator'].astype(str).apply(normalize_creator).unique())
                print(f"🔄 Found {len(processed_users)} creators already processed. Resuming...")
        except Exception as e:
            print(f"⚠️ Warning: Could not read existing {OUTPUT_CSV}: {e}")
    
    print(f"\n📊 PROGRESS SUMMARY:")
    completed_count = len(creator_list) - len(remaining_creators)
    print(f"   Completed: {completed_count}/{len(creator_list)} creators")
    print(f"   Remaining: {len(remaining_creators)} creators")
    print(f"   Total reels processed: {len(all_rows)}")
    print(f"   Progress: {(completed_count/len(creator_list))*100:.1f}%")
    if remaining_creators:
        print(f"   Next creator: @{remaining_creators[0]}")
    
    if not remaining_creators:
        print("🎉 All creators already processed!")
    else:
        # Skip post ratio computation to avoid issues
        print(f"\n🚀 Starting processing of {len(remaining_creators)} remaining creators...")
        print("⚠️ Skipping post-to-reel ratio computation to avoid API issues")
        
        for creator_idx, creator in enumerate(remaining_creators):
            user_str = str(creator).strip()
            
            # SKIP if already done (resume capability)
            if user_str in processed_users:
                print(f"⏩ Skipping @{user_str} (Already processed)")
                continue
            
            try:
                # Process without post ratios
                result = process_creator_with_checkpoints(creator, all_rows, sun_frame_rows, creator_post_ratios={})
                
                # Save checkpoint after each creator
                save_checkpoint(all_rows, sun_frame_rows, creator)
                
                # APPEND TO CSV IMMEDIATELY (like your example)
                if result:  # If processing was successful
                    # Convert single result to DataFrame
                    row_df = pd.DataFrame([result])
                    
                    # APPEND to CSV immediately
                    # mode='a' means append. header=False unless file doesn't exist yet.
                    file_exists = os.path.exists(OUTPUT_CSV)
                    row_df.to_csv(OUTPUT_CSV, mode='a', header=not file_exists, index=False)
                    print(f"✅ Appended @{creator} to {OUTPUT_CSV}")
                    
                    # Add to local set so we don't repeat if list has duplicates
                    processed_users.add(user_str)
                
            except Exception as e:
                print(f"❌ Error processing @{creator}: {e}")
                continue
    
    print("\n🎉 Done! All creators processed.")
    
    # Create final DataFrames
    print(f"\n📊 Creating final DataFrames...")
    df_all_reels = pd.DataFrame(all_rows)
    df_sun_frames = pd.DataFrame(sun_frame_rows)
    
    print(f"✅ Final results:")
    print(f"   - df_all_reels: {len(df_all_reels)} rows, {len(df_all_reels.columns)} columns")
    print(f"   - df_sun_frames: {len(df_sun_frames)} rows, {len(df_sun_frames.columns)} columns")
    
    # Aggregate by creator for final output (matching expected format)
    if not df_all_reels.empty:
        print(f"\n📊 Aggregating results by creator...")
        
        # Group by creator and compute aggregated metrics
        creator_aggregated = []
        for creator in df_all_reels['creator'].unique():
            creator_data = df_all_reels[df_all_reels['creator'] == creator]
            
            # Filter non-music reels for English percentage calculation
            non_music_data = creator_data[~creator_data['is_music_only']]
            
            # Compute aggregated metrics - only the specified scores
            agg_row = {
                'creator': creator,
                'total_reels_processed': len(creator_data),
                # Required features from the list
                'mean_hist_score': creator_data.get('hist_score_0_10', pd.Series([0])).mean(),
                'eye_contact_avg_score_0_10': creator_data['eye_contact_score_0_10'].mean(),
                'series_reel_mean': creator_data['series_flag'].mean(),
                'avg_captioned_reels': creator_data.get('has_dynamic_captions', pd.Series([0])).mean(),
                'avg_english_pct_non_music': non_music_data['english_pct'].mean() if not non_music_data.empty else 0,
                'gemini_genz_word_count': creator_data.get('gemini_genz_word_count', pd.Series([0])).mean(),
                'gemini_is_marketing': creator_data.get('gemini_is_marketing', pd.Series([0])).mean(),
                'gemini_is_educational': creator_data.get('gemini_is_educational', pd.Series([0])).mean(),
                'gemini_has_humour': creator_data.get('gemini_has_humour', pd.Series([0])).mean(),
                'gemini_comment_sentiment_counts.agreeing': creator_data.get('gemini_comment_sentiment_counts_agreeing', pd.Series([0])).mean(),
                'gemini_comment_sentiment_counts.appreciating': creator_data.get('gemini_comment_sentiment_counts_appreciating', pd.Series([0])).mean(),
                'gemini_comment_sentiment_counts.neutral': creator_data.get('gemini_comment_sentiment_counts_neutral', pd.Series([0])).mean(),
                'mean_scene_score': creator_data.get('scene_score_0_10', pd.Series([0])).mean(),
                'mean_face_density': creator_data.get('face_frame_density', pd.Series([0])).mean(),
                # Additional useful metrics
                'avg_word_count': creator_data['word_count'].mean(),
                'avg_english_pct': creator_data['english_pct'].mean(),
                'music_only_ratio': creator_data['is_music_only'].mean(),
                'avg_sun_exposure': creator_data['sun_exposure_0_10_A'].mean(),
                'sun_exposure_0_10_A': creator_data['sun_exposure_0_10_A'].mean(),
                'clip_score_0_10': creator_data.get('clip_score_0_10', pd.Series([0])).mean(),
                'avg_total_accessories': creator_data['total_accessories'].mean(),
            }
            
            # Add Gemini metrics (averaged) - all gemini metrics
            gemini_cols = [col for col in creator_data.columns if col.startswith('gemini_')]
            for col in gemini_cols:
                if creator_data[col].dtype in ['int64', 'float64']:
                    agg_row[col] = creator_data[col].mean()
                else:
                    agg_row[col] = creator_data[col].iloc[0] if len(creator_data) > 0 else None
            
            # Remove post ratios and other metrics not in the specified list
            # Only keep the metrics specified by the user
            
            creator_aggregated.append(agg_row)
        
        df_final = pd.DataFrame(creator_aggregated)
        
        # Compute outlier_2sigma_ratio for each creator
        from features.creativity import compute_creator_outlier_ratios
        outlier_ratios = compute_creator_outlier_ratios(df_all_reels)
        
        # Add outlier ratios to final dataframe
        df_final['outlier_2sigma_ratio'] = df_final['creator'].map(outlier_ratios).fillna(0.0)
        
        # Save final results
        df_final.to_csv(OUTPUT_CSV, index=False)
        print(f"💾 Saved final creator scores to {OUTPUT_CSV}")
        
        # Also save detailed reel data
        df_all_reels.to_csv("detailed_reel_data.csv", index=False)
        print(f"💾 Saved detailed reel data to detailed_reel_data.csv")
    
    if not df_sun_frames.empty:
        df_sun_frames.to_csv("sun_frame_data.csv", index=False)
        print(f"💾 Saved sun frame data to sun_frame_data.csv")
    
    print("\n🎉 Processing completed successfully!")