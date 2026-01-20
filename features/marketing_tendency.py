"""
Marketing Tendency Analysis for Instagram Creators.

This module analyzes a creator's marketing tendency by examining their latest posts
(both reels and static posts) to determine what percentage are marketing content.
"""

import os
import json
import time
import pandas as pd
from typing import Dict, List, Any
from apify_client import ApifyClient

from config import APIFY_API_KEY

# Default value if not defined in config
try:
    from config import MAX_POSTS_FOR_MARKETING_ANALYSIS
except ImportError:
    MAX_POSTS_FOR_MARKETING_ANALYSIS = 10


class MarketingTendencyAnalyzer:
    """Analyzes creator marketing tendency across their recent posts."""
    
    def __init__(self, gemini_client=None, apify_client=None):
        self.gemini_client = gemini_client
        self.apify_client = apify_client or (ApifyClient(APIFY_API_KEY) if APIFY_API_KEY else None)
    
    def _norm_creator(self, x: str) -> str:
        """Normalize creator name."""
        return str(x).strip().lstrip("@").lower()
    
    def flatten_comments(self, comment_list, max_n: int = 50):
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
    
    def _is_reel_item(self, it: dict) -> bool:
        """Robust reel detection."""
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
    
    def fetch_recent_posts(self, creator: str, max_posts: int = MAX_POSTS_FOR_MARKETING_ANALYSIS) -> List[Dict]:
        """
        Fetch recent posts for a creator using Apify instagram-scraper.
        Returns both reels and static posts in chronological order (most recent first).
        """
        if not self.apify_client:
            print(f"  ⚠️ No Apify client available for @{creator}")
            return []
        
        profile_url = f"https://www.instagram.com/{creator.lstrip('@')}/"
        print(f"  🔍 Fetching up to {max_posts} recent posts for @{creator}...")

        run_input = {
            "directUrls": [profile_url],
            "resultsLimit": max_posts * 2,  # Get more to ensure we have enough recent ones
            "addUserInfo": True,
            "addLocation": False,
            "addLikes": False,
            "addVideoThumbnails": False,
            "proxyConfiguration": {"useApifyProxy": True},
        }

        try:
            run = self.apify_client.actor("apify/instagram-scraper").call(run_input=run_input)
            items = self.apify_client.dataset(run["defaultDatasetId"]).list_items().items

            # Filter items for this creator
            creator_posts = []
            creator_norm = self._norm_creator(creator)
            
            for item in items:
                if not isinstance(item, dict):
                    continue
                user = (item.get("ownerUsername") or item.get("username") or "unknown")
                user = self._norm_creator(user)
                
                if user == creator_norm:
                    # Add timestamp for sorting if available
                    timestamp = item.get("timestamp") or item.get("taken_at_timestamp") or 0
                    item["sort_timestamp"] = timestamp
                    creator_posts.append(item)

            # Sort by timestamp (most recent first) and take the requested number
            creator_posts.sort(key=lambda x: x.get("sort_timestamp", 0), reverse=True)
            recent_posts = creator_posts[:max_posts]

            print(f"  ✓ Found {len(recent_posts)} recent posts for @{creator}")
            return recent_posts
            
        except Exception as e:
            print(f"  ✗ Error fetching posts for @{creator}: {e}")
            return []
    
    def parse_gemini_raw(self, x):
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
    
    def call_gemini_for_post_marketing(self, caption: str, transcript: str = "", comments: list = None) -> str:
        """Call Gemini to analyze if a post (reel or static) is marketing content using same criteria as reel analysis."""
        if not self.gemini_client:
            return "{}"
        
        try:
            comments_text = comments[:10] if comments else []  # Use same number as reel analysis
            
            prompt = f"""
            Analyze this Instagram post and return a JSON object with these fields:
            
            Caption: {caption}
            Transcript: {transcript}
            Comments: {comments_text}
            
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
            
            Consider it marketing if it:
            - Promotes products/services
            - Contains affiliate links or discount codes
            - Has clear call-to-actions for purchasing
            - Mentions brands/sponsors
            - Uses promotional language
            - Has pricing information
            - Encourages followers to buy/visit/subscribe
            """
            
            response = self.gemini_client.models.generate_content(
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
            print(f"    ✗ Gemini post marketing analysis failed: {e}")
            return "{}"
    
    def analyze_creator_marketing_tendency(self, creator: str, max_posts: int = MAX_POSTS_FOR_MARKETING_ANALYSIS) -> Dict[str, Any]:
        """
        Analyze a creator's marketing tendency by looking at their latest posts (both reels and static posts).
        
        Returns:
            dict: Marketing analysis results including:
                - marketing_posts_analyzed: Total posts analyzed
                - marketing_posts_count: Number of marketing posts found
                - marketing_tendency_ratio: Ratio of marketing posts (0.0 to 1.0)
                - marketing_reels_count: Marketing posts that were reels
                - marketing_static_posts_count: Marketing posts that were static
        """
        print(f"📊 Analyzing marketing tendency for @{creator}...")
        
        try:
            # Fetch recent posts
            posts = self.fetch_recent_posts(creator, max_posts=max_posts)
            
            if not posts:
                print(f"  ⚠️ No posts found for @{creator}")
                return {
                    "marketing_posts_analyzed": 0,
                    "marketing_posts_count": 0,
                    "marketing_tendency_ratio": 0.0,
                    "marketing_reels_count": 0,
                    "marketing_static_posts_count": 0
                }
            
            print(f"  📋 Analyzing {len(posts)} recent posts...")
            
            marketing_count = 0
            marketing_reels = 0
            marketing_static = 0
            total_analyzed = 0
            
            for post in posts:
                if not isinstance(post, dict):
                    continue
                    
                # Get caption
                caption = post.get("caption", "") or ""
                
                # Get comments (if available)
                comments = []
                if "latestComments" in post:
                    comments = self.flatten_comments(post["latestComments"], max_n=10)
                
                # For reels, try to get transcript if we have the URL
                transcript = ""
                post_url = post.get("url", "")
                if self._is_reel_item(post) and post_url:
                    # Note: We can't easily get transcript here without downloading the video
                    # So we'll analyze reels the same way as static posts for consistency
                    pass
                
                # Analyze with Gemini using same format as reel analysis
                gemini_result = self.call_gemini_for_post_marketing(caption, transcript, comments)
                gemini_obj = self.parse_gemini_raw(gemini_result)
                
                is_marketing = gemini_obj.get("is_marketing", 0)
                
                if is_marketing:
                    marketing_count += 1
                    
                    # Count by type
                    if self._is_reel_item(post):
                        marketing_reels += 1
                    else:
                        marketing_static += 1
                
                total_analyzed += 1
                
                # Add small delay to avoid rate limiting
                time.sleep(0.1)
            
            marketing_ratio = marketing_count / total_analyzed if total_analyzed > 0 else 0.0
            
            result = {
                "marketing_posts_analyzed": total_analyzed,
                "marketing_posts_count": marketing_count,
                "marketing_tendency_ratio": marketing_ratio,
                "marketing_reels_count": marketing_reels,
                "marketing_static_posts_count": marketing_static
            }
            
            print(f"  ✅ Marketing analysis complete: {marketing_count}/{total_analyzed} posts are marketing ({marketing_ratio:.1%})")
            return result
            
        except Exception as e:
            print(f"  ✗ Error analyzing marketing tendency for @{creator}: {e}")
            return {
                "marketing_posts_analyzed": 0,
                "marketing_posts_count": 0,
                "marketing_tendency_ratio": 0.0,
                "marketing_reels_count": 0,
                "marketing_static_posts_count": 0
            }


# Global analyzer instance
marketing_analyzer = MarketingTendencyAnalyzer()


def analyze_creator_marketing_tendency(creator: str, gemini_client=None, apify_client=None, max_posts: int = MAX_POSTS_FOR_MARKETING_ANALYSIS) -> Dict[str, Any]:
    """
    Convenience function to analyze a creator's marketing tendency.
    
    Args:
        creator: Instagram creator handle
        gemini_client: Gemini client for analysis
        apify_client: Apify client for data fetching
        max_posts: Maximum number of recent posts to analyze
    
    Returns:
        dict: Marketing analysis results
    """
    analyzer = MarketingTendencyAnalyzer(gemini_client=gemini_client, apify_client=apify_client)
    return analyzer.analyze_creator_marketing_tendency(creator, max_posts=max_posts)