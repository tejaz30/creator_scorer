# pyright: reportMissingImports=false, reportArgumentType=false, reportOptionalSubscript=false, reportGeneralTypeIssues=false
"""
Redis + Celery version of mine.py with 10 workers for parallel processing.

To run this script:
1. Install Redis: brew install redis (macOS)
2. Start Redis: redis-server
3. Install required packages: pip install celery redis
4. Start Celery workers: celery -A mine_redis worker --loglevel=info --concurrency=10
5. Run the script: python mine_redis.py
"""

import time
import pandas as pd
import os
from google import genai
from google.genai.errors import APIError
from google.api_core.exceptions import FailedPrecondition
from dotenv import load_dotenv
import shutil
import requests
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from gspread_dataframe import set_with_dataframe
from apify_client import ApifyClient
from celery import Celery, group
from celery.result import AsyncResult
import redis
import instaloader
from config import APIFY_API_KEY, GEMINI_API_KEY

# Import the prompt from the separate file
# from gemini_prompt import GEMINI_PROMPT

# Suppress logging
logging.getLogger("apify_client").setLevel(logging.ERROR)
logging.getLogger("celery").setLevel(logging.INFO)

# Load environment variables
load_dotenv()
gemini_api = GEMINI_API_KEY
apifyapi = APIFY_API_KEY
sheet_url = os.environ.get("SHEET_URL")
redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# Initialize Celery with Redis as broker and backend
app = Celery('mine_redis', 
             broker=redis_url,
             backend=redis_url)

# Celery configuration
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour time limit per task
    worker_prefetch_multiplier=1,  # Fetch one task at a time per worker
    worker_max_tasks_per_child=50,  # Restart worker after 50 tasks to prevent memory leaks
)

# Initialize Redis client
redis_client = redis.from_url(redis_url)

# Create a single Gemini client object
client = genai.Client(api_key=gemini_api)

# Initialize models
model_name = "models/gemini-2.0-flash-001"
generation_config = {"temperature": 0.05}

# Initialize Google Sheets with error handling
scope = ["https://spreadsheets.google.com/feeds", 
         "https://www.googleapis.com/auth/drive",
         "https://www.googleapis.com/auth/spreadsheets"]

def init_google_sheets():
    """Initialize Google Sheets connection"""
    try:
        print(f"Loading credentials from credentials.json...")
        creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
        gspread_client = gspread.authorize(creds)  
        
        print(f"Attempting to open sheet: {sheet_url}")
        if not sheet_url:
            raise ValueError("SHEET_URL environment variable is not set!")
        
        spreadsheet = gspread_client.open_by_url(sheet_url)
        print(f"Sheet opened successfully: {spreadsheet.title}")
        
        with open("credentials.json", "r") as f:
            import json
            creds_json = json.load(f)
            service_account_email = creds_json.get("client_email")
            print(f"Service Account Email: {service_account_email}")
            print(f" Make sure this email is shared with permission on your Google Sheet!")
        
        worksheet = spreadsheet.sheet1
        print(f"Worksheet loaded: {worksheet.title}")
        
        return gspread_client, spreadsheet, worksheet
        
    except Exception as e:
        print(f" ERROR: Failed to initialize Google Sheets: {e}")
        raise

# ============= HELPER FUNCTIONS =============
def ensure_expected_columns(df):
    expected_cols = [
        "orig_reel", "creator name", "creator url", "timelog", "aspirational",
        "relatable", "cool", "credible", "communication", "wavy",
        "straight", "curly", "overall", "location", "creator type",
        "comments", "likes", "views", "3xmedian", "5xmedian", "followers",
        "engagement", "pViews", "eViews", "price", "brandfilter",
        "hero_attr", "hair_routine", "hair_style", "qualify_for_ads"
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = None
    return df

# ============= FOLLOWER RETRIEVAL FUNCTION =============
def get_follower_count(creator_username):
    """
    Fetch follower count for a creator using Instaloader.
    Args:
        creator_username: The Instagram username (not URL)
    Returns:
        Follower count or 0 if failed
    """
    try:
        print(f"  Fetching follower count for: {creator_username}")
        L = instaloader.Instaloader()
        follower_data = instaloader.Profile.from_username(L.context, creator_username)
        follower_count = follower_data.followers
        print(f"  ✓ Follower count: {int(follower_count):,}")
        return follower_count
    except Exception as e:
        print(f"  ✗ Error fetching follower count for {creator_username}: {e}")
        return 0

# ============= GEMINI API FUNCTIONS =============
def get_scores_gem(folder_path, flag):
    """
    Refactored function to upload files and generate content using the new Google GenAI SDK.
    """
    uploaded_files = []
    
    try:
        filenames = sorted(os.listdir(folder_path))
        
        # 1. File Upload Logic
        for filename in filenames:
            if filename[0].isdigit():
                if (flag == 1 and int(filename[0]) <= 5) or (flag == 0 and int(filename[0]) > 5):
                    file_path = os.path.join(folder_path, filename)
                    if os.path.isfile(file_path):
                        try:
                            uploaded = client.files.upload(file=file_path)
                            uploaded_files.append(uploaded)
                            print(f"Uploaded file: {filename} (Name: {uploaded.name})")
                        except Exception as e:
                            print(f"Error uploading file {filename}: {e}")
            else:
                if flag == 0:
                    file_path = os.path.join(folder_path, filename)
                    if os.path.isfile(file_path):
                        try:
                            uploaded = client.files.upload(file=file_path)
                            uploaded_files.append(uploaded)
                            print(f"Uploaded file: {filename} (Name: {uploaded.name})")
                        except Exception as e:
                            print(f"Error uploading file {filename}: {e}")

        # Wait for all files to be in ACTIVE state
        print("Waiting for files to become active...")
        for uploaded_file in uploaded_files:
            max_wait_time = 10
            wait_interval = 2
            elapsed_time = 0
            
            while elapsed_time < max_wait_time:
                try:
                    file_info = client.files.get(name=uploaded_file.name)
                    if file_info.state == "ACTIVE":
                        print(f"File {uploaded_file.name} is now ACTIVE")
                        break
                    elif file_info.state == "PROCESSING":
                        print(f"File {uploaded_file.name} is still PROCESSING, waiting...")
                        time.sleep(wait_interval)
                        elapsed_time += wait_interval
                    else:
                        print(f"File {uploaded_file.name} is in unexpected state: {file_info.state}")
                        break
                except Exception as e:
                    print(f"Error checking file status for {uploaded_file.name}: {e}")
                    break
            
            if elapsed_time >= max_wait_time:
                print(f"Warning: File {uploaded_file.name} did not become ACTIVE within {max_wait_time} seconds")

        # 2. Content Generation Logic
        for attempt in range(5):
            try:
                response = client.models.generate_content(
                    model=model_name, 
                    contents=[GEMINI_PROMPT] + uploaded_files,  
                    config=generation_config
                )
                return response.text
            except APIError as e:
                print(f"Gemini API error (attempt {attempt + 1}): {e}")
                time.sleep(2.5)
            except Exception as e:
                print(f"Unknown error (attempt {attempt + 1}): {e}")
                time.sleep(2)
        
        return "lalala"

    finally:
        # 3. File Cleanup
        for uploaded_file in uploaded_files:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception as e:
                print(f"Error deleting file : {e}")


def reset_files_gem(task_id):
    """Reset reels folder for specific task"""
    folder = os.path.join(os.getcwd(), 'reels', task_id)
    try:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"Removed existing reels folder: {folder}")
        os.makedirs(folder)
        print(f"Created new reels folder: {folder}")
    except Exception as e:
        print(f"Error resetting reels folder: {e}")

# ============= DOWNLOAD FUNCTIONS =============
def get_cdn_url(REEL_URL: str, API_BASE) -> str | None:
    try:
        endpoint = f"{API_BASE}/api/video"
        print(f"Fetching CDN URL for: {REEL_URL}")
        resp = requests.get(endpoint, params={"postUrl": REEL_URL}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "success":
            cdn_url = data["data"]["videoUrl"]
            print(f"CDN URL obtained: {cdn_url}")
            return cdn_url
        else:
            print(f"Failed to get CDN URL. Response: {data}")
    except Exception as e:
        print(f"Error getting CDN URL: {e}")
    return None


def download_reel(video_url: str, REEL_NO: str, task_id: str):
    max_bytes = 4 * 1024 * 1024
    
    try:
        reels_dir = os.path.join("reels", task_id)
        os.makedirs(reels_dir, exist_ok=True)
        path = os.path.join(reels_dir, REEL_NO + '.mp4')
        
        # Check if file already exists and is valid
        if os.path.exists(path):
            # Quick validation - check file size
            file_size = os.path.getsize(path)
            if file_size > 1024:  # More than 1KB
                print(f"Reel {REEL_NO} already exists: {path}")
                return path
            else:
                print(f"Existing file too small ({file_size} bytes), re-downloading...")
                os.remove(path)
        
        print(f"Downloading reel {REEL_NO} from: {video_url}")
        headers = {"Range": f"bytes=0-{max_bytes - 1}"}

        with requests.get(video_url, headers=headers, stream=True, timeout=20) as r:
            if r.status_code not in (200, 206):
                print(f"Failed to download reel. Status code: {r.status_code}")
                return None

            with open(path, "wb") as f:
                downloaded = 0
                for chunk in r.iter_content(chunk_size=1048576):  
                    if not chunk:
                        break
                    if downloaded + len(chunk) > max_bytes:
                        chunk = chunk[:max_bytes - downloaded]
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded >= max_bytes:
                        break
            print(f" Reel {REEL_NO} downloaded successfully")
            return path
    except Exception as e:
        print(f" Error downloading reel {REEL_NO}: {e}")
    return None


def get_files_gem(REEL_URL, REEL_NO='0', task_id='default'):
    API_BASE = "https://reeldownload-zeta.vercel.app/"
    cdn = get_cdn_url(REEL_URL, API_BASE)
    if cdn:
        path = download_reel(cdn, REEL_NO, task_id)
        return path
    return None


def download_reels_parallel(reels_list, task_id, max_workers=4):
    """
    Download multiple reels in parallel using ThreadPoolExecutor.
    
    Args:
        reels_list: List of tuples [(reel_url, reel_number), ...]
        task_id: Unique task identifier for folder isolation
        max_workers: Number of concurrent downloads (default 4)
    
    Returns:
        List of successfully downloaded file paths
    """
    downloaded_files = []
    
    def download_with_retry(reel_url, reel_no):
        try:
            path = get_files_gem(reel_url, str(reel_no), task_id)
            return path
        except Exception as e:
            print(f" Error downloading reel {reel_no}: {e}")
            return None
    
    print(f' Downloading {len(reels_list)} reels in parallel (max_workers={max_workers})...')
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(download_with_retry, reel_url, reel_no) 
            for reel_no, reel_url in enumerate(reels_list)
        ]
        
        for future in futures:
            try:
                result = future.result(timeout=60)
                if result:
                    downloaded_files.append(result)
            except Exception as e:
                print(f" Download future failed: {e}")
    
    print(f"✓ Downloaded {len(downloaded_files)}/{len(reels_list)} reels successfully")
    return downloaded_files

# ============= PARSING FUNCTIONS =============
def extract_res_content(text):
    start_tag = "<res>"
    end_tag = "</res>"
    start_idx = text.find(start_tag)
    end_idx = text.find(end_tag)
    
    if start_idx == -1:
        return -1
    if end_idx == -1:
        return text[start_idx + len(start_tag):]
    return text[start_idx + len(start_tag):end_idx]


def first_valid_word(s):
    tokens = re.findall(r"\w+", s)            
    banned = {"straight", "wavy", "curly"}
    return next((t for t in tokens if not is_number(t) and t.lower() not in banned), None)


def is_number(tok):
    try:
        float(tok)
        return True
    except ValueError:
        return False


def extract_additional_fields(text):
    # brandfilter = ""
    hero_attr = ""
    hair_routine = ""
    hair_style = ""
    qualify_for_ads = ""
    
    # Extract additional fields from the text
    res_start = text.find("<res>")
    res_end = text.find("</res>")
    
    if res_start != -1 and res_end != -1:
        res_content = text[res_start+5:res_end].strip()
        parts = res_content.split()
        
        if len(parts) >= 10:
            hero_attr = parts[7] if len(parts) > 7 else ""
            hair_routine = parts[8] if len(parts) > 8 else ""
            hair_style = parts[9] if len(parts) > 9 else ""
    
    #  brandfilter
    return hero_attr, hair_routine, hair_style, qualify_for_ads

# ============= APIFY FUNCTIONS =============
def get_profile_from_username(ig, num=1, act=1, cols=["ownerUsername"]):
    if len(ig) == 0:
        return pd.DataFrame()
    
    client_apify = ApifyClient(apifyapi)
    run = None
    
    try:
        if act == 1:
            run_input = {"username": ig, "resultsLimit": num}
            run = client_apify.actor("xMc5Ga1oCONPmWJIa").call(run_input=run_input)
        elif act == 2:
            run_input = {"startUrls": ig, "maxItems": num, "customMapFunction": "(object) => { return {...object} }"}
            run = client_apify.actor("culc72xb7MP3EbaeX").call(run_input=run_input)
        elif act == 3:
            run_input={"usernames":[ig]}
            run = client_apify.actor("apify/instagram-profile-scraper").call(run_input=run_input)

        data = client_apify.dataset(run["defaultDatasetId"]).list_items().items 
        
        for col in data:
            if "error" in col:
                return pd.DataFrame()
        
        df_data = pd.DataFrame(data)
        print(f" Apify returned {len(df_data)} items")
        print(f"Available columns: {list(df_data.columns)}")
        
        # Enhanced column mapping
        column_mapping = {
            'likesCount': ['likeCount', 'likes', 'likescount'],
            'commentsCount': ['commentCount', 'comments', 'commentscount'],
            'videoPlayCount': ['playCount', 'videoViewCount', 'viewCount', 'views', 'videoplaycount'],
            'locationName': ['location', 'locationname'],
            'inputUrl': ['url', 'inputurl'],
            'url': ['videoUrl', 'postUrl', 'shortCode']
        }
        
        mapped_cols = []
        for col in cols:
            if col in df_data.columns:
                mapped_cols.append(col)
            elif col in column_mapping:
                found = False
                for alt_col in column_mapping[col]:
                    if alt_col in df_data.columns:
                        df_data[col] = df_data[alt_col]
                        mapped_cols.append(col)
                        found = True
                        print(f"Mapped {alt_col} → {col}")
                        break
                if not found:
                    df_data[col] = None
                    mapped_cols.append(col)
                    print(f"  Column '{col}' not found, created empty column")
            else:
                df_data[col] = None
                mapped_cols.append(col)
        
        result = df_data[mapped_cols]
        
        # Print data quality summary
        print(f"\n DATA QUALITY CHECK:")
        for col in ['likesCount', 'commentsCount', 'videoPlayCount']:
            if col in result.columns:
                non_zero = (result[col] > 0).sum()
                total = len(result)
                print(f"  {col}: {non_zero}/{total} non-zero values ({non_zero/total*100:.1f}%)")
        
        return result
        
    except Exception as e:
        print(f" Apify error: {e}")
        return pd.DataFrame()

# ============= ENGAGEMENT SCORE FUNCTION =============
def calculate_engagement_score(likes, views, comments, followers):
    """
    Calculate engagement score from 0 to 1.
    
    Engagement Score = (Engagement Rate + Comment Rate + Viral Factor) / 3
    """
    if likes is None or views is None or comments is None or followers is None:
        return 0, 0, 0, 0
    
    try:
        likes = float(likes) if likes > 0 else 0
        views = float(views) if views > 0 else 0
        comments = float(comments) if comments > 0 else 0
        followers = float(followers) if followers > 0 else 1
    except (TypeError, ValueError):
        return 0, 0, 0, 0
    
    # Engagement Rate: (likes + comments) / views (capped at 0.1)
    engagement_rate = min((likes + comments) / views, 0.1) if views > 0 else 0
    
    # Comment Rate: comments / likes (capped at 0.5)
    comment_rate = min(comments / likes, 0.5) if likes > 0 else 0
    
    # Viral Factor: (views / followers) normalized (capped at 1)
    viral_factor = min((views / followers) / 100, 1) if followers > 0 else 0
    
    # Calculate overall engagement score (0-1)
    engagement_score = (engagement_rate * 10 + comment_rate * 2 + viral_factor) / 3
    engagement_score = min(max(engagement_score, 0), 1)
    
    return engagement_score, engagement_rate, comment_rate, viral_factor


def categorize_brand_filter(engagement_score, overall_score=None):
    """
    Categorize content into brand filter categories based on engagement score.
    
    Categories:
    - offbrand: engagement_score < 0.2
    - promising: 0.2 <= engagement_score < 0.5
    - onbrand: engagement_score >= 0.5
    """
    if engagement_score is None:
        return "unknown", 0
    
    try:
        engagement_score = float(engagement_score)
    except (TypeError, ValueError):
        return "unknown", 0
    
    engagement_score = max(0, min(engagement_score, 1))
    
    if engagement_score >= 0.5:
        category = "onbrand"
        confidence = engagement_score
    elif engagement_score >= 0.2:
        category = "promising"
        confidence = 0.5 + abs(engagement_score - 0.5)
    else:
        category = "offbrand"
        confidence = 1 - engagement_score
    
    return category, confidence

# ============= USERNAME RETRIEVAL FUNCTION =============
def get_username(url):
    """
    Get Instagram username from a reel URL using Instaloader.
    Falls back to parsing username from profile URL if reel parsing fails.
    Args:
        url: The Instagram reel URL or profile URL
    Returns:
        Username string or None if failed
    """
    try:
        L = instaloader.Instaloader()
        # Extract the shortcode from the link
        shortcode = url.strip("/").split("/")[-1]
        post = instaloader.Post.from_shortcode(L.context, shortcode)
        return post.owner_username
    except Exception as e:
        print(f"    Failed to fetch username from reel: {e}")
        
        # Fallback: Check if URL is a profile URL (e.g., https://www.instagram.com/username/)
        try:
            # Clean the URL and extract potential username
            clean_url = url.strip().strip("/")
            
            # Check if it matches Instagram profile URL pattern
            if "instagram.com/" in clean_url:
                # Extract the part after instagram.com/
                parts = clean_url.split("instagram.com/")
                if len(parts) > 1:
                    potential_username = parts[1].split("/")[0].split("?")[0]
                    
                    # Validate that it's not a special path (like 'reel', 'p', 'tv', etc.)
                    special_paths = ['reel', 'reels', 'p', 'tv', 'stories', 'accounts', 'explore']
                    if potential_username and potential_username not in special_paths:
                        print(f"Extracted username from profile URL: {potential_username}")
                        return potential_username
            
            print(f"Could not extract username from URL: {url}")
            return None
        except Exception as fallback_error:
            print(f"Fallback username extraction also failed: {fallback_error}")
            return None

# ============= CELERY TASKS =============
@app.task(bind=True, name='mine_redis.process_creator')
def process_creator(self, index, orig_reel, sheet_url):
    """
    Celery task to process a single creator.
    This runs in a worker process with isolation.
    """
    task_id = self.request.id
    print(f"\n{'='*60}")
    print(f"[TASK {task_id}] Processing creator at index {index}")
    print(f"{'='*60}")
    
    try:
        # Initialize Google Sheets for this task
        gspread_client, spreadsheet, worksheet = init_google_sheets()
        
        # Get current data
        records = worksheet.get_all_records()
        df = pd.DataFrame(records)
        df = ensure_expected_columns(df)
        
        # Verify the row still needs processing
        creator_name = df.at[index, 'creator name']
        if pd.notna(creator_name) and creator_name != '' and creator_name != 'None':
            print(f"[TASK {task_id}] Row already processed, skipping")
            return {"status": "skipped", "index": index, "reason": "already_processed"}
        
        previous_users = set(df['creator name'].dropna().unique()) if 'creator name' in df.columns else set()
        
        print(f"[TASK {task_id}] Processing reel: {orig_reel}")
        
        # Get username from reel using Instaloader
        username = get_username(orig_reel)
        
        if not username:
            df.at[index, 'creator name'] = -2
            update_sheet_row(worksheet, index, df.iloc[index])
            return {"status": "error", "index": index, "error": "Could not fetch username"}
        
        # Check for duplicates
        if username in previous_users:
            df.at[index, 'creator name'] = 'duplicate'
            update_sheet_row(worksheet, index, df.iloc[index])
            print(f"[TASK {task_id}]   Marked as duplicate: {username}")
            return {"status": "duplicate", "index": index, "username": username}
        
        # Process the creator
        df.at[index, 'creator name'] = username
        creator_url = f"https://www.instagram.com/{username}/"
        df.at[index, 'creator url'] = creator_url
        
        print(f"[TASK {task_id}]  Username: {username}")
        print(f"[TASK {task_id}] Fetching reels...")
        
        reels_data = get_profile_from_username([creator_url], 20, 1, 
            ['inputUrl', 'videoPlayCount', 'likesCount', 'commentsCount', 'url', 'locationName'])
        
        if len(reels_data) == 0:
            df.at[index, 'aspirational'] = "ERROR: No reels data"
            update_sheet_row(worksheet, index, df.iloc[index])
            return {"status": "error", "index": index, "error": "No reels data"}
        
        # Analyze the creator
        result = analyse_from_index(index, reels_data, df, task_id)
        
        # Update Google Sheet
        update_sheet_row(worksheet, index, df.iloc[index])
        
        print(f"[TASK {task_id}]  Analysis complete")
        
        # Cleanup task-specific folder
        reset_files_gem(task_id)
        
        return {
            "status": "success", 
            "index": index, 
            "username": username,
            "task_id": task_id,
            "result": result
        }
        
    except Exception as e:
        print(f"[TASK {task_id}]  Error: {e}")
        return {"status": "error", "index": index, "error": str(e), "task_id": task_id}


def analyse_from_index(index, data, df, task_id):
    """
    Analyze creator data and populate the dataframe row.
    Modified to use task_id for folder isolation.
    """
    print(f"\n[TASK {task_id}] Analyzing data...")
    
    # Filter valid reels
    if 'url' not in data.columns:
        print(f"[TASK {task_id}]  ERROR: 'url' column not found")
        df.at[index, 'aspirational'] = "ERROR: No URL column"
        return None
    
    valid_data = data[data['url'].notna() & (data['url'] != '')].copy()
    
    if 'likesCount' in valid_data.columns and 'videoPlayCount' in valid_data.columns:
        with_data = valid_data[(valid_data['likesCount'] > 0) | (valid_data['videoPlayCount'] > 0)]
        if len(with_data) >= 10:
            valid_data = with_data
    
    if len(valid_data) == 0:
        df.at[index, 'aspirational'] = "ERROR: No valid reels"
        return None
    
    reels_to_analyze = valid_data.head(20)
    reels = list(reels_to_analyze['url'])
    
    print(f"[TASK {task_id}]  Processing {len(reels)} reels")
    
    # Extract engagement metrics
    def get_valid_values(column_name):
        if column_name in reels_to_analyze.columns:
            values = reels_to_analyze[column_name].replace([0, -1, np.nan], pd.NA).dropna()
            return list(values.astype(int))
        return []
    
    commentsCount = get_valid_values('commentsCount')
    likesCount = get_valid_values('likesCount')
    viewsCount = get_valid_values('videoPlayCount')
    
    # Calculate medians
    median_likes = int(np.median(likesCount)) if len(likesCount) > 0 else 0
    median_views = int(np.median(viewsCount)) if len(viewsCount) > 0 else 0
    median_comments = int(np.median(commentsCount)) if len(commentsCount) > 0 else 0
    
    print(f"[TASK {task_id}] Median - Likes: {median_likes:,} | Views: {median_views:,} | Comments: {median_comments:,}")
    
    # Calculate 3x and 5x median metrics
    threexmedian = 0
    fivexmedian = 0
    pViews = 0
    
    if len(viewsCount) > 0 and median_views > 0:
        threexmedian = sum(1 for v in viewsCount if v >= 3 * median_views)
        fivexmedian = sum(1 for v in viewsCount if v >= 5 * median_views)
        pViews = (threexmedian / len(viewsCount)) * median_views
    
    # Store metrics
    try:
        if 'locationName' in reels_to_analyze.columns:
            location = list(reels_to_analyze['locationName'].dropna())
            if location:
                df.at[index, 'location'] = str(location)
        
        df.at[index, 'comments'] = median_comments
        df.at[index, 'likes'] = median_likes
        df.at[index, 'views'] = median_views
        df.at[index, '3xmedian'] = threexmedian
        df.at[index, '5xmedian'] = fivexmedian
        df.at[index, 'pViews'] = int(pViews)
        
        # Fetch follower count
        creator_name = df.at[index, 'creator name']
        if creator_name and creator_name != '' and creator_name != 'duplicate':
            follower_count = get_follower_count(creator_name)
            df.at[index, 'followers'] = follower_count
        
    except Exception as e:
        print(f'[TASK {task_id}]  Error storing metrics: {e}')
    
    # Download reels
    reels_to_download = reels[:6]
    downloaded_files = download_reels_parallel(reels_to_download, task_id, max_workers=4)
    
    if not downloaded_files:
        df.at[index, 'aspirational'] = "ERROR: No reels downloaded"
        return None
    
    # AI Analysis
    folder_path = os.path.join(os.getcwd(), 'reels', task_id)
    print(f'[TASK {task_id}]  Uploading to Gemini for AI analysis...')
    
    aspirational = []
    relatable = []
    cool = []
    credible = []
    communication = []
    creator_type = []
    straight = []
    curly = []
    wavy = []
    
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(get_scores_gem, folder_path, 0),
                executor.submit(get_scores_gem, folder_path, 1)
            ]
            results = [future.result() for future in futures]
            
            valid_results = [res for res in results if res and res != 'lalala']
            if not valid_results:
                df.at[index, 'aspirational'] = "ERROR: No valid Gemini results"
                return None
            
            hero_attrs = []
            hair_routines = []
            hair_styles = []
            
            for res in valid_results:
                text = extract_res_content(res)
                if text != -1 and res != 'lalala':
                    numbers = re.findall(r'\d+', text)
                    
                    if 'straight' in text.lower():
                        straight.append(1)
                    if 'curly' in text.lower():
                        curly.append(1)
                    if 'wavy' in text.lower():
                        wavy.append(1)
                    
                    if len(numbers) >= 5:
                        numbers = list(map(int, numbers[:5]))
                        aspirational.append(numbers[0])
                        relatable.append(numbers[1])
                        cool.append(numbers[2])
                        credible.append(numbers[3])
                        communication.append(numbers[4])
                    
                    creator_type.append(first_valid_word(text))
                    
                    _, hero_attr, hair_routine, hair_style, _ = extract_additional_fields(res)
                    if hero_attr:
                        hero_attrs.append(hero_attr)
                    if hair_routine:
                        hair_routines.append(hair_routine)
        
        if len(aspirational) > 0:
            df.at[index, 'aspirational'] = str(np.percentile(aspirational, 75))
            df.at[index, 'relatable'] = str(np.percentile(relatable, 75))
            df.at[index, 'cool'] = str(np.percentile(cool, 75))
            df.at[index, 'credible'] = str(np.percentile(credible, 75))
            df.at[index, 'communication'] = str(np.percentile(communication, 75))
            
            df.at[index, 'creator type'] = str(list(set(creator_type)))
            df.at[index, 'curly'] = str(len(curly))
            df.at[index, 'straight'] = str(len(straight))
            df.at[index, 'wavy'] = str(len(wavy))
            
            if hero_attrs:
                df.at[index, 'hero_attr'] = '; '.join(list(set(hero_attrs)))
            if hair_routines:
                df.at[index, 'hair_routine'] = '; '.join(list(set(hair_routines)))
            if hair_styles:
                df.at[index, 'hair_style'] = '; '.join(list(set(hair_styles)))
            
            overall = (np.percentile(aspirational, 75) + np.percentile(relatable, 75) + 
                      np.percentile(cool, 75) + np.percentile(credible, 75))
            df.at[index, 'overall'] = str(overall)
            
            # Calculate engagement score
            eng_score, eng_rate, comment_rate, viral_factor = calculate_engagement_score(
                median_likes, median_views, median_comments, df.at[index, 'followers']
            )
            # brand_category, confidence = categorize_brand_filter(eng_score, overall)
            
            df.at[index, 'engagement'] = round(eng_score, 3)
            # df.at[index, 'brandfilter'] = brand_category
            
            print(f"[TASK {task_id}]  AI Analysis Complete - Overall: {overall}")
            return {
                "overall": overall,
                "engagement": eng_score,
                # "brandfilter": brand_category
            }
        else:
            df.at[index, 'aspirational'] = "ERROR: No scores extracted"
            return None
    
    except Exception as e:
        print(f"[TASK {task_id}]  Error during Gemini analysis: {e}")
        df.at[index, 'aspirational'] = f"ERROR: {str(e)}"
        return None


def update_sheet_row(worksheet, row_index, row_data):
    """
    Update only a specific row in the Google Sheet.
    """
    try:
        columns = list(row_data.index)
        values = [str(row_data[col]) if pd.notna(row_data[col]) else "" for col in columns]
        
        sheet_row_number = row_index + 2
        
        def num_to_col_letter(num):
            result = ""
            while num > 0:
                num -= 1
                result = chr(65 + (num % 26)) + result
                num //= 26
            return result
        
        end_col_letter = num_to_col_letter(len(columns))
        range_name = f'A{sheet_row_number}:{end_col_letter}{sheet_row_number}'
        
        worksheet.update(range_name=range_name, values=[values])
        print(f" Updated row {sheet_row_number} in Google Sheet")
        return True
    except Exception as e:
        print(f" Error updating sheet row: {e}")
        return False

# ============= MAIN EXECUTION =============
def main():
    """
    Main function that orchestrates the parallel processing using Celery.
    Workers continuously process tasks as they become available - no idle time!
    """
    print(f"\n{'='*60}")
    print(f"REDIS + CELERY CONTINUOUS PROCESSOR")
    print(f"Workers process tasks continuously with no idle time")
    print(f"{'='*60}\n")
    
    # Test Redis connection
    try:
        redis_client.ping()
        print("✓ Redis connection successful")
    except Exception as e:
        print(f" ERROR: Redis connection failed: {e}")
        print("Make sure Redis is running: redis-server")
        return
    
    # Initialize Google Sheets
    gspread_client, spreadsheet, worksheet = init_google_sheets()
    
    submitted_tasks = {}  # Track active tasks: {task_id: (index, AsyncResult)}
    max_concurrent_tasks = 10
    
    print(f"Maximum concurrent tasks: {max_concurrent_tasks}")
    print(f"Checking for completed tasks every 5 seconds\n")
    
    while True:
        try:
            # Refresh data from sheet
            records = worksheet.get_all_records()
            df = pd.DataFrame(records)
            df = ensure_expected_columns(df)
            
            # Find unprocessed rows
            unprocessed_rows = []
            for index, row in df.iterrows():
                creator_name = row.get('creator name', None)
                if pd.isnull(creator_name) or creator_name == '' or creator_name == 'None':  
                    orig_reel = row.get('orig_reel', None)
                    if orig_reel is not None and orig_reel != '':
                        unprocessed_rows.append((index, orig_reel))
            
            # Check completed tasks and remove them from tracking
            completed_tasks = []
            for task_id, (index, async_result) in list(submitted_tasks.items()):
                if async_result.ready():
                    completed_tasks.append(task_id)
                    try:
                        result = async_result.get(timeout=1)
                        status = result.get('status', 'unknown')
                        username = result.get('username', 'N/A')
                        
                        if status == 'success':
                            print(f"COMPLETED: Row {index} - {username}")
                        elif status == 'duplicate':
                            print(f"DUPLICATE: Row {index} - {username}")
                        elif status == 'error':
                            error_msg = result.get('error', 'Unknown error')
                            print(f" ERROR: Row {index} - {error_msg}")
                        elif status == 'skipped':
                            print(f" SKIPPED: Row {index}")
                    except Exception as e:
                        print(f" TASK FAILED: Row {index} - {e}")
                    
                    del submitted_tasks[task_id]
            
            # Submit new tasks to fill available worker slots
            available_slots = max_concurrent_tasks - len(submitted_tasks)
            
            if available_slots > 0 and unprocessed_rows:
                # Submit tasks for available slots
                tasks_to_submit = unprocessed_rows[:available_slots]
                
                for index, orig_reel in tasks_to_submit:
                    # Double-check this row hasn't been claimed by another task
                    if any(idx == index for idx, _ in submitted_tasks.values()):
                        continue
                    
                    # Submit task asynchronously
                    async_result = process_creator.apply_async(
                        args=[index, orig_reel, sheet_url]
                    )
                    submitted_tasks[async_result.id] = (index, async_result)
                    print(f"🚀 SUBMITTED: Row {index} (Task ID: {async_result.id[:8]}...)")
            
            # Status update
            active_count = len(submitted_tasks)
            pending_count = len(unprocessed_rows)
            
            if active_count > 0 or pending_count > 0:
                print(f"\n STATUS: {active_count} active tasks | {pending_count} pending rows")
            
            # Exit condition: no active tasks and no pending rows
            if active_count == 0 and pending_count == 0:
                print(f"\n{'='*60}")
                print(f"✓ ALL ROWS PROCESSED! Script will now exit.")
                print(f"{'='*60}\n")
                break
            
            # Wait before next check
            time.sleep(15)
            
        except KeyboardInterrupt:
            print(f"\n\n⚠ Interrupted by user. Waiting for active tasks to complete...")
            for task_id, (index, async_result) in submitted_tasks.items():
                try:
                    async_result.revoke(terminate=True)
                    print(f"Cancelled task for row {index}")
                except:
                    pass
            break
            
        except Exception as e:
            print(f"\n Error in main loop: {e}")
            print(f"Retrying in 10 seconds...")
            time.sleep(10)


if __name__ == "__main__":    
    main()
