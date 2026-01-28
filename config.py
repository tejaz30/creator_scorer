"""
Configuration settings for the feature extraction system.
"""
import os
import torch
from dotenv import load_dotenv

load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
APIFY_API_KEY = os.getenv("APIFY_API_KEY")

# Device configuration
USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"

# Directory settings
REEL_DOWNLOAD_DIR = "_reel_cache"
APIFY_CACHE_DIR = "apify_metadata_cache"
COMMENTS_CACHE_DIR = "cache_comments"

# Processing limits (OPTIMIZED for single high-performance instance)
MAX_REELS_PER_CREATOR = 10
FRAME_SAMPLE_COUNT = 16  # Keep original value for quality
MAX_DOWNLOAD_WORKERS = 8  # Reduced from 10 to prevent overload
MAX_FRAMES_PER_REEL = 16  # Keep original value for quality
MAX_POSTS_FOR_MARKETING_ANALYSIS = 10

# Gemini model
GEMINI_MODEL = "models/gemini-2.0-flash-001"

# Video caption detection settings (REVERTED OCR sampling to original)
TARGET_FPS = 3  # Reverted from 1 to 3 (original value)
SMART_SAMPLING = True  # Keep smart sampling logic
BOTTOM_CROP_RATIO = 0.4
MIN_TEXT_LEN = 3
SIMILARITY_SAME_SEGMENT = 0.6
MIN_SEGMENT_DURATION = 0.0
MAX_SEGMENT_DURATION = 15.0
CAPTION_MIN_COVERAGE = 0.05
STATIC_OVERLAY_MAX_SEGMENTS = 2
STATIC_DOMINANCE_RATIO = 0.7

# Accessory detection classes
ACCESSORY_CLASSES = [
    "backpack", "handbag", "hat", "scarf", "sunglasses", "glasses",
    "necklace", "earrings", "watch", "bracelet", "ring", "wallet", "belt",
    "mobile_phone", "laptop", "tablet", "smartwatch", "headphones", "camera",
    "car", "sports_car", "motorcycle", "bike", "airplane", "boat",
    "suitcase", "luggage", "surfboard", "skis", "horse",
    "dress", "coat", "suit", "high_heels",
]

CLASS_BUCKET = {
    "backpack": "travel_gear",
    "handbag": "clothing",
    "hat": "clothing",
    "scarf": "clothing",
    "sunglasses": "clothing",
    "glasses": "clothing",
    "necklace": "jewellery",
    "earrings": "jewellery",
    "watch": "jewellery",
    "bracelet": "jewellery",
    "ring": "jewellery",
    "wallet": "gadgets",
    "belt": "clothing",
    "mobile_phone": "gadgets",
    "laptop": "gadgets",
    "tablet": "gadgets",
    "smartwatch": "gadgets",
    "headphones": "gadgets",
    "camera": "gadgets",
    "car": "vehicles",
    "sports_car": "vehicles",
    "motorcycle": "vehicles",
    "bike": "vehicles",
    "airplane": "vehicles",
    "boat": "vehicles",
    "suitcase": "travel_gear",
    "luggage": "travel_gear",
    "surfboard": "travel_gear",
    "skis": "travel_gear",
    "horse": "vehicles",
    "dress": "clothing",
    "coat": "clothing",
    "suit": "clothing",
    "high_heels": "clothing",
}

# Series detection keywords
SERIES_KEYWORDS = [
    r"\bseries\b",
    r"\bserie\b",
    r"\bepisode\b",
    r"\bep\b",
    r"\bpart\b",
    r"\bpt\b",
    r"\bseason\b",
]

EP_PATTERNS = [
    r"(?:episode|ep|ep\.)\s*(\d+)",
    r"(?:part|pt|pt\.)\s*(\d+)",
    r"s(?:eason)?\s*\d+\s*(?:episode|ep)\s*(\d+)",
]

# Create directories
os.makedirs(REEL_DOWNLOAD_DIR, exist_ok=True)
os.makedirs(APIFY_CACHE_DIR, exist_ok=True)

# Attractiveness analysis settings
BETA_VAE_MODEL_PATH = os.getenv("BETA_VAE_MODEL_PATH", "models/beta_vae_utkface.pth")
GOLD_VECTOR_PATH = os.getenv("GOLD_VECTOR_PATH", "models/gold_standard_female.pth")
MIN_FACE_SIZE = 90
MIN_SAMPLES_REQUIRED = 3
ATTRACTIVENESS_LATENT_DIM = 128

# Composite scoring weights
FACE_WEIGHT = 0.5
BRIGHTNESS_WEIGHT = 0.25
CLEANLINESS_WEIGHT = 0.25  # Inverted clutter score