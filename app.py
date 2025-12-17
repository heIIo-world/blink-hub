"""
Blink Hub v2.3 - Local Web App
Downloads all videos from your Blink camera system to local storage.
"""

import asyncio
import json
import logging
import os
import re
import sys
import sqlite3
import traceback
import smtplib
import shutil
import subprocess
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta, time as dt_time, timezone
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response
from starlette.background import BackgroundTask
from pydantic import BaseModel
from aiohttp import ClientSession, ClientTimeout
from blinkpy.blinkpy import Blink
from blinkpy.auth import Auth

# Try to import the 2FA exception (name varies by version)
try:
    from blinkpy.blinkpy import BlinkTwoFARequiredError
except ImportError:
    BlinkTwoFARequiredError = None


# =============================================================================
# Configuration
# =============================================================================

VERSION = "3.1.0"
BUILD_DATE = "2025-12-16T00:00:00Z"

# Determine base path (works for both script and PyInstaller exe)
if getattr(sys, 'frozen', False):
    # Running as compiled exe
    BASE_DIR = Path(sys.executable).parent
else:
    # Running as script
    BASE_DIR = Path(__file__).parent

# Setup logging to file
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "blink_app.log"

# Circular buffer for live log streaming
from collections import deque
import threading

class LogBuffer:
    """Thread-safe circular buffer for log entries."""
    def __init__(self, maxlen=1000):
        self.buffer = deque(maxlen=maxlen)
        self.lock = threading.Lock()
        self.listeners = []
    
    def add(self, entry):
        with self.lock:
            self.buffer.append(entry)
    
    def get_all(self):
        with self.lock:
            return list(self.buffer)
    
    def get_recent(self, n=100):
        with self.lock:
            return list(self.buffer)[-n:]
    
    def clear(self):
        with self.lock:
            self.buffer.clear()

log_buffer = LogBuffer(maxlen=2000)

class BufferHandler(logging.Handler):
    """Custom handler that writes to log buffer."""
    def emit(self, record):
        try:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "level": record.levelname,
                "message": self.format(record)
            }
            log_buffer.add(entry)
        except Exception:
            pass

# Custom file handler that flushes after every write
class FlushingFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        FlushingFileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Add buffer handler for live streaming
buffer_handler = BufferHandler()
buffer_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
buffer_handler.setLevel(logging.INFO)  # Only INFO and above for UI
logging.getLogger().addHandler(buffer_handler)

logger = logging.getLogger(__name__)

# Log startup
logger.info(f"=" * 50)
logger.info(f"Blink Hub v{VERSION} (build {BUILD_DATE}) starting")
logger.info(f"Base directory: {BASE_DIR}")
logger.info(f"Log file: {LOG_FILE}")

# Check timezone support
try:
    ZoneInfo("America/Los_Angeles")
    logger.info("Timezone support: OK (tzdata available)")
except Exception as e:
    logger.warning(f"Timezone support: LIMITED - {e}")
    logger.warning("Install tzdata package for full timezone support: pip install tzdata")

DEFAULT_DOWNLOADS_DIR = BASE_DIR / "downloads"
DB_PATH = BASE_DIR / "blink_downloads.db"
CREDENTIALS_PATH = BASE_DIR / "credentials.json"
SETTINGS_PATH = BASE_DIR / "settings.json"
DELAY_BETWEEN_DOWNLOADS = 2  # seconds between API calls

# Global state
blink_instance: Optional[Blink] = None
blink_session: Optional[ClientSession] = None
scheduler_task: Optional[asyncio.Task] = None
retention_task: Optional[asyncio.Task] = None
continuous_sync_task: Optional[asyncio.Task] = None
download_status = {
    "running": False,
    "total": 0,
    "downloaded": 0,
    "skipped": 0,
    "current_camera": "",
    "current_file": "",
    "bytes_downloaded": 0,
    "last_run": None,
    "error": None
}
sync_status = {
    "enabled": False,
    "running": False,
    "last_sync": None,
    "next_sync": None,
    "interval_minutes": 5,
    "videos_synced": 0,
    "error": None
}

# Toast notifications queue (for UI)
notifications = []


# =============================================================================
# Settings Management
# =============================================================================

def get_default_settings() -> dict:
    """Get default settings."""
    return {
        "download_dir": str(DEFAULT_DOWNLOADS_DIR.absolute()),
        "scheduler_enabled": False,
        "scheduler_time": "03:00",
        "scheduler_days": 7,
        # Email settings
        "email_enabled": False,
        "email_smtp_server": "",
        "email_smtp_port": 587,
        "email_username": "",
        "email_password": "",
        "email_from": "",
        "email_to": "",
        "email_on_success": True,
        "email_on_error": True,
        # Retention settings
        "retention_enabled": False,
        "retention_days": 30,
        # Disk space warning
        "disk_warning_enabled": True,
        "disk_warning_gb": 10,
        # UI preferences
        "theme": "dark",
        "show_thumbnails": True,
        # Timezone settings (for video filenames)
        "timezone": "auto",  # "auto" = detect from browser, or specific like "America/Los_Angeles"
    }


def load_settings() -> dict:
    """Load settings from file."""
    defaults = get_default_settings()
    if SETTINGS_PATH.exists():
        try:
            with open(SETTINGS_PATH, 'r') as f:
                saved = json.load(f)
                defaults.update(saved)
        except:
            pass
    return defaults


def save_settings(settings: dict):
    """Save settings to file."""
    with open(SETTINGS_PATH, 'w') as f:
        json.dump(settings, f, indent=2)


def get_download_dir() -> Path:
    """Get the current download directory."""
    settings = load_settings()
    return Path(settings.get("download_dir", str(DEFAULT_DOWNLOADS_DIR.absolute())))


def get_configured_timezone() -> str:
    """Get the configured timezone from settings."""
    settings = load_settings()
    return settings.get("timezone", "auto")


def get_timezone_for_filename() -> str:
    """Get the timezone to use for filenames.
    
    Returns timezone string like 'America/Los_Angeles' or 'UTC' if auto-detect fails.
    """
    tz_setting = get_configured_timezone()
    if tz_setting == "auto":
        # When auto, we'll use UTC as server-side default
        # The browser will send its detected timezone in API requests
        return "UTC"
    return tz_setting


def convert_utc_to_timezone(utc_dt: datetime, tz_name: str) -> datetime:
    """Convert a UTC datetime to the specified timezone.
    
    Args:
        utc_dt: datetime in UTC (may or may not have tzinfo)
        tz_name: timezone name like 'America/Los_Angeles' or 'UTC'
    
    Returns:
        datetime in the target timezone (naive, for filename use)
    """
    try:
        # Ensure we have a UTC datetime
        if utc_dt.tzinfo is None:
            utc_dt = utc_dt.replace(tzinfo=timezone.utc)
        elif utc_dt.tzinfo != timezone.utc:
            utc_dt = utc_dt.astimezone(timezone.utc)
        
        # Handle UTC specially (no tzdata needed)
        if tz_name == "UTC":
            return utc_dt.replace(tzinfo=None)
        
        # Convert to target timezone
        target_tz = ZoneInfo(tz_name)
        local_dt = utc_dt.astimezone(target_tz)
        
        # Return naive datetime for filename formatting
        return local_dt.replace(tzinfo=None)
    except Exception as e:
        logger.warning(f"Timezone conversion failed ({tz_name}): {e}, using UTC")
        return utc_dt.replace(tzinfo=None) if utc_dt.tzinfo else utc_dt


def is_valid_timezone(tz_name: str) -> bool:
    """Check if a timezone name is valid."""
    if tz_name == "auto" or tz_name == "UTC":
        return True
    try:
        ZoneInfo(tz_name)
        return True
    except Exception:
        return False


# Common US timezones for the dropdown
TIMEZONE_OPTIONS = [
    ("auto", "Auto-detect from browser"),
    ("America/New_York", "Eastern Time (ET)"),
    ("America/Chicago", "Central Time (CT)"),
    ("America/Denver", "Mountain Time (MT)"),
    ("America/Los_Angeles", "Pacific Time (PT)"),
    ("America/Anchorage", "Alaska Time (AKT)"),
    ("Pacific/Honolulu", "Hawaii Time (HT)"),
    ("UTC", "UTC (Coordinated Universal Time)"),
]


# =============================================================================
# Database Functions
# =============================================================================

def init_db():
    """Initialize SQLite database to track downloaded videos."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Main videos table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS downloaded_videos (
            id TEXT PRIMARY KEY,
            camera_name TEXT,
            created_at TEXT,
            file_path TEXT,
            file_size INTEGER DEFAULT 0,
            thumbnail_path TEXT,
            downloaded_at TEXT,
            duration INTEGER DEFAULT 0,
            starred INTEGER DEFAULT 0,
            reviewed INTEGER DEFAULT 0
        )
    """)
    
    # Download log
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS download_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            videos_downloaded INTEGER,
            videos_skipped INTEGER,
            bytes_downloaded INTEGER DEFAULT 0,
            status TEXT
        )
    """)
    
    # Tags table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            color TEXT DEFAULT '#3b82f6',
            created_at TEXT
        )
    """)
    
    # Video-tag relationship (many-to-many)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS video_tags (
            video_id TEXT,
            tag_id INTEGER,
            created_at TEXT,
            PRIMARY KEY (video_id, tag_id),
            FOREIGN KEY (video_id) REFERENCES downloaded_videos(id),
            FOREIGN KEY (tag_id) REFERENCES tags(id)
        )
    """)
    
    # Camera groups
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS camera_groups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            color TEXT DEFAULT '#6b7280',
            created_at TEXT
        )
    """)
    
    # Camera-group membership
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS camera_group_members (
            camera_name TEXT,
            group_id INTEGER,
            PRIMARY KEY (camera_name, group_id),
            FOREIGN KEY (group_id) REFERENCES camera_groups(id)
        )
    """)
    
    # Migration: Add columns if they don't exist (MUST run before indexes)
    migrations = [
        ("downloaded_videos", "file_size", "INTEGER DEFAULT 0"),
        ("downloaded_videos", "thumbnail_path", "TEXT"),
        ("downloaded_videos", "duration", "INTEGER DEFAULT 0"),
        ("downloaded_videos", "starred", "INTEGER DEFAULT 0"),
        ("downloaded_videos", "reviewed", "INTEGER DEFAULT 0"),
        ("downloaded_videos", "filename_timezone", "TEXT"),  # Track which timezone was used for filename
        ("downloaded_videos", "notes", "TEXT"),  # User notes for videos
        ("download_log", "bytes_downloaded", "INTEGER DEFAULT 0"),
    ]
    
    for table, column, col_type in migrations:
        try:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        except:
            pass
    
    # Create indexes for faster queries (after migrations add columns)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_videos_camera ON downloaded_videos(camera_name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_videos_created ON downloaded_videos(created_at)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_videos_starred ON downloaded_videos(starred)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_video_tags_video ON video_tags(video_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_video_tags_tag ON video_tags(tag_id)")
    
    conn.commit()
    conn.close()


def is_video_downloaded(video_id: str) -> bool:
    """Check if a video has already been downloaded."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM downloaded_videos WHERE id = ?", (video_id,))
    result = cursor.fetchone() is not None
    conn.close()
    return result


def generate_thumbnail(video_path: str, thumbnail_dir: Path) -> str:
    """Generate a thumbnail from a video file using ffmpeg.
    
    Returns the path to the generated thumbnail, or empty string on failure.
    """
    try:
        video_path = Path(video_path)
        if not video_path.exists():
            return ""
        
        # Create thumbnail filename based on video filename
        thumb_name = video_path.stem + ".jpg"
        thumb_path = thumbnail_dir / thumb_name
        
        # Skip if thumbnail already exists
        if thumb_path.exists():
            return str(thumb_path)
        
        # Try ffmpeg first (best quality)
        try:
            result = subprocess.run([
                'ffmpeg', '-y', '-i', str(video_path),
                '-ss', '00:00:01',  # Capture at 1 second
                '-vframes', '1',
                '-vf', 'scale=320:-1',  # 320px width, maintain aspect
                '-q:v', '3',  # Good quality
                str(thumb_path)
            ], capture_output=True, timeout=30)
            
            if thumb_path.exists():
                return str(thumb_path)
        except FileNotFoundError:
            # ffmpeg not available, try alternative
            pass
        except Exception as e:
            logger.warning(f"ffmpeg thumbnail failed: {e}")
        
        # Fallback: try using moviepy if available
        try:
            from moviepy.editor import VideoFileClip
            with VideoFileClip(str(video_path)) as clip:
                # Get frame at 1 second or midpoint for very short clips
                t = min(1, clip.duration / 2)
                frame = clip.get_frame(t)
                
                # Save using PIL
                from PIL import Image
                img = Image.fromarray(frame)
                img.thumbnail((320, 180))
                img.save(thumb_path, "JPEG", quality=85)
                
                if thumb_path.exists():
                    return str(thumb_path)
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"moviepy thumbnail failed: {e}")
        
        return ""
        
    except Exception as e:
        logger.warning(f"Thumbnail generation failed for {video_path}: {e}")
        return ""


def mark_video_downloaded(video_id: str, camera_name: str, created_at: str, file_path: str, file_size: int = 0, thumbnail_path: str = "", duration: int = 0, filename_timezone: str = None):
    """Mark a video as downloaded.
    
    Args:
        video_id: Unique video ID from Blink
        camera_name: Name of the camera
        created_at: Original UTC timestamp from Blink
        file_path: Local file path
        file_size: Size in bytes
        thumbnail_path: Path to thumbnail image
        duration: Video duration in seconds
        filename_timezone: Timezone used for the filename (None/NULL = UTC/legacy)
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """INSERT OR REPLACE INTO downloaded_videos 
           (id, camera_name, created_at, file_path, file_size, thumbnail_path, downloaded_at, duration, filename_timezone) 
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (video_id, camera_name, created_at, file_path, file_size, thumbnail_path, datetime.now().isoformat(), duration, filename_timezone)
    )
    conn.commit()
    conn.close()


def get_downloaded_videos(limit: int = 100) -> list:
    """Get list of recently downloaded videos."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT camera_name, created_at, file_path, file_size, thumbnail_path, downloaded_at FROM downloaded_videos ORDER BY downloaded_at DESC LIMIT ?",
        (limit,)
    )
    results = cursor.fetchall()
    conn.close()
    return [
        {"camera": r[0], "created_at": r[1], "file_path": r[2], "file_size": r[3], "thumbnail_path": r[4], "downloaded_at": r[5]}
        for r in results
    ]


def get_download_stats() -> dict:
    """Get download statistics."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Total videos downloaded
    cursor.execute("SELECT COUNT(*), SUM(file_size) FROM downloaded_videos")
    row = cursor.fetchone()
    total = row[0] or 0
    total_bytes = row[1] or 0
    
    # Videos by camera
    cursor.execute("""
        SELECT camera_name, COUNT(*) as count, SUM(file_size) as size
        FROM downloaded_videos 
        GROUP BY camera_name 
        ORDER BY count DESC
    """)
    by_camera = {row[0]: {"count": row[1], "size": row[2] or 0} for row in cursor.fetchall()}
    
    # Last download time
    cursor.execute("SELECT MAX(downloaded_at) FROM downloaded_videos")
    last_download = cursor.fetchone()[0]
    
    # Most recent video date
    cursor.execute("SELECT MAX(created_at) FROM downloaded_videos WHERE created_at != ''")
    newest_video = cursor.fetchone()[0]
    
    # Downloads today
    today = datetime.now().strftime("%Y-%m-%d")
    cursor.execute("SELECT COUNT(*) FROM downloaded_videos WHERE downloaded_at LIKE ?", (f"{today}%",))
    today_count = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        "total_videos": total,
        "total_bytes": total_bytes,
        "by_camera": by_camera,
        "last_download": last_download,
        "newest_video": newest_video,
        "today_count": today_count
    }


def log_download_run(downloaded: int, skipped: int, bytes_downloaded: int, status: str):
    """Log a download run."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO download_log (timestamp, videos_downloaded, videos_skipped, bytes_downloaded, status) VALUES (?, ?, ?, ?, ?)",
        (datetime.now().isoformat(), downloaded, skipped, bytes_downloaded, status)
    )
    conn.commit()
    conn.close()


def get_download_history(limit: int = 20) -> list:
    """Get download history."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT timestamp, videos_downloaded, videos_skipped, bytes_downloaded, status FROM download_log ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    results = cursor.fetchall()
    conn.close()
    return [
        {"timestamp": r[0], "downloaded": r[1], "skipped": r[2], "bytes": r[3], "status": r[4]}
        for r in results
    ]


def get_videos_for_retention(days: int) -> list:
    """Get videos older than specified days."""
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, file_path, thumbnail_path FROM downloaded_videos WHERE downloaded_at < ?",
        (cutoff,)
    )
    results = cursor.fetchall()
    conn.close()
    return [{"id": r[0], "file_path": r[1], "thumbnail_path": r[2]} for r in results]


def delete_video_record(video_id: str):
    """Delete a video record from database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Also delete tag associations
    cursor.execute("DELETE FROM video_tags WHERE video_id = ?", (video_id,))
    cursor.execute("DELETE FROM downloaded_videos WHERE id = ?", (video_id,))
    conn.commit()
    conn.close()


# =============================================================================
# Video Browsing Functions
# =============================================================================

def browse_local_videos(
    cameras: list = None,
    camera_groups: list = None,
    start_date: str = None,
    end_date: str = None,
    start_time: str = None,
    end_time: str = None,
    tags: list = None,
    starred_only: bool = False,
    unreviewed_only: bool = False,
    min_duration: int = None,
    max_duration: int = None,
    search_text: str = None,
    sort_by: str = "created_at",
    sort_order: str = "DESC",
    page: int = 1,
    per_page: int = 50
) -> dict:
    """Browse local videos with comprehensive filtering."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Build query
    conditions = []
    params = []
    
    # Camera filter
    if cameras and len(cameras) > 0:
        placeholders = ','.join('?' * len(cameras))
        conditions.append(f"v.camera_name IN ({placeholders})")
        params.extend(cameras)
    
    # Camera groups filter - get cameras in specified groups
    if camera_groups and len(camera_groups) > 0:
        placeholders = ','.join('?' * len(camera_groups))
        cursor.execute(f"""
            SELECT DISTINCT camera_name FROM camera_group_members 
            WHERE group_id IN (SELECT id FROM camera_groups WHERE name IN ({placeholders}))
        """, camera_groups)
        group_cameras = [row[0] for row in cursor.fetchall()]
        if group_cameras:
            placeholders = ','.join('?' * len(group_cameras))
            conditions.append(f"v.camera_name IN ({placeholders})")
            params.extend(group_cameras)
    
    # Date range filter
    if start_date:
        if start_time:
            conditions.append("v.created_at >= ?")
            params.append(f"{start_date}T{start_time}")
        else:
            conditions.append("v.created_at >= ?")
            params.append(f"{start_date}T00:00:00")
    
    if end_date:
        if end_time:
            conditions.append("v.created_at <= ?")
            params.append(f"{end_date}T{end_time}")
        else:
            conditions.append("v.created_at <= ?")
            params.append(f"{end_date}T23:59:59")
    
    # Tags filter
    if tags and len(tags) > 0:
        placeholders = ','.join('?' * len(tags))
        conditions.append(f"""v.id IN (
            SELECT video_id FROM video_tags 
            WHERE tag_id IN (SELECT id FROM tags WHERE name IN ({placeholders}))
        )""")
        params.extend(tags)
    
    # Starred/reviewed filters
    if starred_only:
        conditions.append("v.starred = 1")
    if unreviewed_only:
        conditions.append("v.reviewed = 0")
    
    # Duration filter
    if min_duration:
        conditions.append("v.duration >= ?")
        params.append(min_duration)
    if max_duration:
        conditions.append("v.duration <= ?")
        params.append(max_duration)
    
    # Text search (camera name)
    if search_text:
        conditions.append("v.camera_name LIKE ?")
        params.append(f"%{search_text}%")
    
    # Build WHERE clause
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    
    # Validate sort parameters
    valid_sorts = ["created_at", "camera_name", "duration", "file_size", "downloaded_at"]
    if sort_by not in valid_sorts:
        sort_by = "created_at"
    if sort_order.upper() not in ["ASC", "DESC"]:
        sort_order = "DESC"
    
    # Get total count
    cursor.execute(f"SELECT COUNT(*) FROM downloaded_videos v WHERE {where_clause}", params)
    total_count = cursor.fetchone()[0]
    
    # Calculate pagination
    offset = (page - 1) * per_page
    total_pages = (total_count + per_page - 1) // per_page if total_count > 0 else 1
    
    # Get videos with tags
    query = f"""
        SELECT v.id, v.camera_name, v.created_at, v.file_path, v.file_size, 
               v.thumbnail_path, v.downloaded_at, v.duration, v.starred, v.reviewed, v.notes,
               GROUP_CONCAT(t.name, ',') as tags
        FROM downloaded_videos v
        LEFT JOIN video_tags vt ON v.id = vt.video_id
        LEFT JOIN tags t ON vt.tag_id = t.id
        WHERE {where_clause}
        GROUP BY v.id
        ORDER BY v.{sort_by} {sort_order}
        LIMIT ? OFFSET ?
    """
    params.extend([per_page, offset])
    
    cursor.execute(query, params)
    results = cursor.fetchall()
    conn.close()
    
    videos = []
    for r in results:
        videos.append({
            "id": r[0],
            "camera_name": r[1],
            "created_at": r[2],
            "file_path": r[3],
            "file_size": r[4] or 0,
            "thumbnail_path": r[5],
            "downloaded_at": r[6],
            "duration": r[7] or 0,
            "starred": bool(r[8]),
            "reviewed": bool(r[9]),
            "notes": r[10] or "",
            "tags": r[11].split(',') if r[11] else []
        })
    
    return {
        "videos": videos,
        "total": total_count,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages
    }


def get_video_by_id(video_id: str) -> dict:
    """Get a single video by ID with its tags."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT v.id, v.camera_name, v.created_at, v.file_path, v.file_size, 
               v.thumbnail_path, v.downloaded_at, v.duration, v.starred, v.reviewed, v.notes
        FROM downloaded_videos v WHERE v.id = ?
    """, (video_id,))
    
    r = cursor.fetchone()
    if not r:
        conn.close()
        return None
    
    # Get tags
    cursor.execute("""
        SELECT t.name, t.color FROM tags t
        JOIN video_tags vt ON t.id = vt.tag_id
        WHERE vt.video_id = ?
    """, (video_id,))
    tags = [{"name": row[0], "color": row[1]} for row in cursor.fetchall()]
    
    conn.close()
    
    return {
        "id": r[0],
        "camera_name": r[1],
        "created_at": r[2],
        "file_path": r[3],
        "file_size": r[4] or 0,
        "thumbnail_path": r[5],
        "downloaded_at": r[6],
        "duration": r[7] or 0,
        "starred": bool(r[8]),
        "reviewed": bool(r[9]),
        "notes": r[10] or "",
        "tags": tags
    }


def update_video_flags(video_id: str, starred: bool = None, reviewed: bool = None):
    """Update video starred/reviewed flags."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    updates = []
    params = []
    
    if starred is not None:
        updates.append("starred = ?")
        params.append(1 if starred else 0)
    if reviewed is not None:
        updates.append("reviewed = ?")
        params.append(1 if reviewed else 0)
    
    if updates:
        params.append(video_id)
        cursor.execute(f"UPDATE downloaded_videos SET {', '.join(updates)} WHERE id = ?", params)
        conn.commit()
    
    conn.close()


def get_unique_cameras() -> list:
    """Get list of unique camera names from downloaded videos."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT camera_name FROM downloaded_videos ORDER BY camera_name")
    results = [row[0] for row in cursor.fetchall()]
    conn.close()
    return results


# =============================================================================
# Tag Functions
# =============================================================================

def get_all_tags() -> list:
    """Get all tags."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT t.id, t.name, t.color, COUNT(vt.video_id) as video_count
        FROM tags t
        LEFT JOIN video_tags vt ON t.id = vt.tag_id
        GROUP BY t.id
        ORDER BY t.name
    """)
    results = [{"id": r[0], "name": r[1], "color": r[2], "video_count": r[3]} for r in cursor.fetchall()]
    conn.close()
    return results


def create_tag(name: str, color: str = "#3b82f6") -> int:
    """Create a new tag."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO tags (name, color, created_at) VALUES (?, ?, ?)",
        (name, color, datetime.now().isoformat())
    )
    tag_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return tag_id


def delete_tag(tag_id: int):
    """Delete a tag and its associations."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM video_tags WHERE tag_id = ?", (tag_id,))
    cursor.execute("DELETE FROM tags WHERE id = ?", (tag_id,))
    conn.commit()
    conn.close()


def update_tag(tag_id: int, name: str = None, color: str = None):
    """Update a tag."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    updates = []
    params = []
    if name:
        updates.append("name = ?")
        params.append(name)
    if color:
        updates.append("color = ?")
        params.append(color)
    
    if updates:
        params.append(tag_id)
        cursor.execute(f"UPDATE tags SET {', '.join(updates)} WHERE id = ?", params)
        conn.commit()
    
    conn.close()


def add_tag_to_video(video_id: str, tag_id: int):
    """Add a tag to a video."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT OR IGNORE INTO video_tags (video_id, tag_id, created_at) VALUES (?, ?, ?)",
            (video_id, tag_id, datetime.now().isoformat())
        )
        conn.commit()
    except:
        pass
    conn.close()


def remove_tag_from_video(video_id: str, tag_id: int):
    """Remove a tag from a video."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM video_tags WHERE video_id = ? AND tag_id = ?", (video_id, tag_id))
    conn.commit()
    conn.close()


def add_tag_to_videos(video_ids: list, tag_id: int):
    """Add a tag to multiple videos (bulk operation)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    for video_id in video_ids:
        try:
            cursor.execute(
                "INSERT OR IGNORE INTO video_tags (video_id, tag_id, created_at) VALUES (?, ?, ?)",
                (video_id, tag_id, now)
            )
        except:
            pass
    conn.commit()
    conn.close()


# =============================================================================
# Camera Group Functions
# =============================================================================

def get_all_camera_groups() -> list:
    """Get all camera groups with their members."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, name, color FROM camera_groups ORDER BY name")
    groups = []
    for row in cursor.fetchall():
        cursor.execute("SELECT camera_name FROM camera_group_members WHERE group_id = ?", (row[0],))
        members = [r[0] for r in cursor.fetchall()]
        groups.append({
            "id": row[0],
            "name": row[1],
            "color": row[2],
            "cameras": members
        })
    
    conn.close()
    return groups


def create_camera_group(name: str, color: str = "#6b7280", cameras: list = None) -> int:
    """Create a new camera group."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO camera_groups (name, color, created_at) VALUES (?, ?, ?)",
        (name, color, datetime.now().isoformat())
    )
    group_id = cursor.lastrowid
    
    if cameras:
        for camera in cameras:
            cursor.execute(
                "INSERT OR IGNORE INTO camera_group_members (camera_name, group_id) VALUES (?, ?)",
                (camera, group_id)
            )
    
    conn.commit()
    conn.close()
    return group_id


def update_camera_group(group_id: int, name: str = None, color: str = None, cameras: list = None):
    """Update a camera group."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if name:
        cursor.execute("UPDATE camera_groups SET name = ? WHERE id = ?", (name, group_id))
    if color:
        cursor.execute("UPDATE camera_groups SET color = ? WHERE id = ?", (color, group_id))
    
    if cameras is not None:
        # Replace all members
        cursor.execute("DELETE FROM camera_group_members WHERE group_id = ?", (group_id,))
        for camera in cameras:
            cursor.execute(
                "INSERT INTO camera_group_members (camera_name, group_id) VALUES (?, ?)",
                (camera, group_id)
            )
    
    conn.commit()
    conn.close()


def delete_camera_group(group_id: int):
    """Delete a camera group."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM camera_group_members WHERE group_id = ?", (group_id,))
    cursor.execute("DELETE FROM camera_groups WHERE id = ?", (group_id,))
    conn.commit()
    conn.close()


def get_camera_groups_for_camera(camera_name: str) -> list:
    """Get all groups that a camera belongs to."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT cg.id, cg.name, cg.color 
        FROM camera_groups cg
        JOIN camera_group_members cgm ON cg.id = cgm.group_id
        WHERE cgm.camera_name = ?
    """, (camera_name,))
    groups = [{"id": row[0], "name": row[1], "color": row[2]} for row in cursor.fetchall()]
    conn.close()
    return groups


# =============================================================================
# Disk Space Functions
# =============================================================================

def get_disk_space(path: Path) -> dict:
    """Get disk space information for a path."""
    try:
        if os.name == 'nt':  # Windows
            import ctypes
            free_bytes = ctypes.c_ulonglong(0)
            total_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                ctypes.c_wchar_p(str(path)), 
                None, 
                ctypes.pointer(total_bytes), 
                ctypes.pointer(free_bytes)
            )
            return {
                "free": free_bytes.value,
                "total": total_bytes.value,
                "used": total_bytes.value - free_bytes.value,
                "percent_used": round((total_bytes.value - free_bytes.value) / total_bytes.value * 100, 1) if total_bytes.value > 0 else 0
            }
        else:  # Unix
            stat = os.statvfs(path)
            free = stat.f_bavail * stat.f_frsize
            total = stat.f_blocks * stat.f_frsize
            return {
                "free": free,
                "total": total,
                "used": total - free,
                "percent_used": round((total - free) / total * 100, 1) if total > 0 else 0
            }
    except Exception as e:
        print(f"Error getting disk space: {e}")
        return {"free": 0, "total": 0, "used": 0, "percent_used": 0}


def format_bytes(bytes_val: int) -> str:
    """Format bytes to human readable string."""
    if bytes_val < 1024:
        return f"{bytes_val} B"
    elif bytes_val < 1024 * 1024:
        return f"{bytes_val / 1024:.1f} KB"
    elif bytes_val < 1024 * 1024 * 1024:
        return f"{bytes_val / (1024 * 1024):.1f} MB"
    else:
        return f"{bytes_val / (1024 * 1024 * 1024):.2f} GB"


# =============================================================================
# Email Notifications
# =============================================================================

def send_email_notification(subject: str, body: str):
    """Send email notification."""
    settings = load_settings()
    
    if not settings.get("email_enabled"):
        logger.info("Email notifications disabled, skipping")
        return
    
    smtp_server = settings.get("email_smtp_server", "")
    smtp_port = settings.get("email_smtp_port", 587)
    username = settings.get("email_username", "")
    password = settings.get("email_password", "")
    email_from = settings.get("email_from", "") or username
    email_to = settings.get("email_to", "")
    
    logger.info(f"Email config: server={smtp_server}, port={smtp_port}, user={username}, from={email_from}, to={email_to}")
    
    if not all([smtp_server, username, password, email_to]):
        logger.error(f"Email not configured properly: server={smtp_server}, user={username}, to={email_to}, password={'set' if password else 'NOT SET'}")
        raise Exception("Email settings incomplete - check SMTP server, username, password, and recipient")
    
    try:
        msg = MIMEMultipart()
        msg['From'] = email_from
        msg['To'] = email_to
        msg['Subject'] = f"[Blink Downloader] {subject}"
        
        msg.attach(MIMEText(body, 'plain'))
        
        logger.info(f"Connecting to {smtp_server}:{smtp_port}...")
        server = smtplib.SMTP(smtp_server, smtp_port, timeout=30)
        server.set_debuglevel(1)  # Enable SMTP debug output
        
        logger.info("Starting TLS...")
        server.starttls()
        
        logger.info(f"Logging in as {username}...")
        server.login(username, password)
        
        logger.info(f"Sending email to {email_to}...")
        result = server.send_message(msg)
        logger.info(f"Send result: {result}")
        server.quit()
        
        logger.info(f"Email sent successfully: {subject}")
    except smtplib.SMTPAuthenticationError as e:
        logger.error(f"SMTP Authentication failed: {e}")
        raise Exception(f"Authentication failed - check username and password. For Gmail, use an App Password.")
    except smtplib.SMTPException as e:
        logger.error(f"SMTP error: {e}")
        raise Exception(f"SMTP error: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        logger.error(traceback.format_exc())
        raise


def notify_download_complete(downloaded: int, skipped: int, bytes_downloaded: int, errors: list = None):
    """Send notification when download completes."""
    settings = load_settings()
    
    if errors and settings.get("email_on_error"):
        subject = f"Download completed with errors - {downloaded} videos"
        body = f"""Blink Video Download Report

Downloaded: {downloaded} videos ({format_bytes(bytes_downloaded)})
Skipped: {skipped} videos
Errors: {len(errors)}

Error details:
{chr(10).join(errors[:10])}

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        send_email_notification(subject, body)
    elif downloaded > 0 and settings.get("email_on_success"):
        subject = f"Download complete - {downloaded} new videos"
        body = f"""Blink Video Download Report

Downloaded: {downloaded} videos ({format_bytes(bytes_downloaded)})
Skipped (already downloaded): {skipped} videos

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        send_email_notification(subject, body)


# =============================================================================
# Notification Queue (for UI toasts)
# =============================================================================

def add_notification(message: str, type: str = "info"):
    """Add a notification to the queue."""
    global notifications
    notifications.append({
        "id": datetime.now().timestamp(),
        "message": message,
        "type": type,
        "time": datetime.now().isoformat()
    })
    # Keep only last 10 notifications
    notifications = notifications[-10:]


def get_notifications() -> list:
    """Get and clear notifications."""
    global notifications
    result = notifications.copy()
    notifications = []
    return result


# =============================================================================
# Retention Policy
# =============================================================================

async def run_retention_policy():
    """Delete videos older than retention period."""
    settings = load_settings()
    
    if not settings.get("retention_enabled"):
        return {"deleted": 0}
    
    retention_days = settings.get("retention_days", 30)
    videos = get_videos_for_retention(retention_days)
    
    deleted = 0
    for video in videos:
        try:
            # Delete video file
            file_path = Path(video["file_path"])
            if file_path.exists():
                file_path.unlink()
                deleted += 1
            
            # Delete thumbnail if exists
            if video.get("thumbnail_path"):
                thumb_path = Path(video["thumbnail_path"])
                if thumb_path.exists():
                    thumb_path.unlink()
            
            # Remove from database
            delete_video_record(video["id"])
            
        except Exception as e:
            print(f"Error deleting {video['file_path']}: {e}")
    
    if deleted > 0:
        add_notification(f"Retention policy: deleted {deleted} old videos", "info")
        print(f"Retention policy: deleted {deleted} videos older than {retention_days} days")
    
    return {"deleted": deleted}


async def retention_scheduler():
    """Background task to run retention policy daily."""
    while True:
        try:
            await asyncio.sleep(3600 * 24)  # Check once per day
            await run_retention_policy()
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Retention scheduler error: {e}")


# =============================================================================
# Credentials Management
# =============================================================================

def save_credentials(auth_data: dict):
    """Save authentication credentials (legacy - for username/password only)."""
    with open(CREDENTIALS_PATH, 'w') as f:
        json.dump(auth_data, f)


async def save_blink_credentials():
    """Save full Blink credentials including auth token using blinkpy's save method."""
    global blink_instance
    if blink_instance:
        try:
            await blink_instance.save(str(CREDENTIALS_PATH))
            logger.info(f"Saved Blink credentials to {CREDENTIALS_PATH}")
            return True
        except Exception as e:
            logger.error(f"Failed to save Blink credentials: {e}")
            return False
    return False


def load_credentials() -> Optional[dict]:
    """Load saved credentials."""
    if CREDENTIALS_PATH.exists():
        try:
            with open(CREDENTIALS_PATH, 'r') as f:
                return json.load(f)
        except:
            pass
    return None


# =============================================================================
# Scheduler Functions
# =============================================================================

def get_next_scheduled_run() -> Optional[str]:
    """Calculate the next scheduled run time."""
    settings = load_settings()
    if not settings.get("scheduler_enabled"):
        return None
    
    try:
        scheduled_time = settings.get("scheduler_time", "03:00")
        hour, minute = map(int, scheduled_time.split(":"))
        
        now = datetime.now()
        next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        if next_run <= now:
            next_run += timedelta(days=1)
        
        return next_run.strftime("%Y-%m-%d %H:%M")
    except:
        return None


async def scheduler_loop():
    """Background scheduler that runs downloads at scheduled time."""
    global blink_instance
    print("Scheduler started")
    
    while True:
        try:
            settings = load_settings()
            
            if settings.get("scheduler_enabled") and blink_instance:
                scheduled_time = settings.get("scheduler_time", "03:00")
                hour, minute = map(int, scheduled_time.split(":"))
                
                now = datetime.now()
                if now.hour == hour and now.minute == minute:
                    print(f"Scheduler triggered at {now}")
                    add_notification("Scheduled download starting...", "info")
                    
                    # Run download
                    days = settings.get("scheduler_days", 7)
                    await download_videos_task(since_days=days, cameras=None, scheduled=True)
                    
                    # Run retention policy after download
                    await run_retention_policy()
                    
                    # Wait a minute to avoid re-triggering
                    await asyncio.sleep(60)
            
            # Check every 30 seconds
            await asyncio.sleep(30)
            
        except asyncio.CancelledError:
            print("Scheduler stopped")
            break
        except Exception as e:
            print(f"Scheduler error: {e}")
            await asyncio.sleep(60)


def start_scheduler():
    """Start the background scheduler."""
    global scheduler_task, retention_task
    
    if scheduler_task is None or scheduler_task.done():
        scheduler_task = asyncio.create_task(scheduler_loop())
        print("Scheduler task created")
    
    if retention_task is None or retention_task.done():
        retention_task = asyncio.create_task(retention_scheduler())
        print("Retention task created")


def stop_scheduler():
    """Stop the background scheduler."""
    global scheduler_task, retention_task
    
    if scheduler_task and not scheduler_task.done():
        scheduler_task.cancel()
    if retention_task and not retention_task.done():
        retention_task.cancel()


# =============================================================================
# Continuous Sync Functions
# =============================================================================

async def continuous_sync_loop():
    """Background task that continuously syncs videos from Blink cloud to local storage."""
    global blink_instance, sync_status
    
    logger.info("Continuous sync loop started")
    sync_status["enabled"] = True
    
    while True:
        try:
            settings = load_settings()
            interval = settings.get("sync_interval_minutes", 5)
            sync_status["interval_minutes"] = interval
            
            if not settings.get("continuous_sync_enabled", False):
                sync_status["enabled"] = False
                await asyncio.sleep(30)
                continue
            
            sync_status["enabled"] = True
            
            if blink_instance and blink_instance.cameras:
                sync_status["running"] = True
                sync_status["error"] = None
                
                try:
                    logger.info("Starting sync cycle...")
                    
                    # Download any new videos (last 1 day to catch recent)
                    await download_videos_task(since_days=1, cameras=None, scheduled=False)
                    
                    sync_status["last_sync"] = datetime.now().isoformat()
                    sync_status["videos_synced"] = download_status.get("downloaded", 0)
                    sync_status["next_sync"] = (datetime.now() + timedelta(minutes=interval)).isoformat()
                    
                    logger.info(f"Sync complete. Downloaded: {sync_status['videos_synced']}, Next sync in {interval} minutes")
                    
                except Exception as e:
                    logger.error(f"Sync error: {e}")
                    sync_status["error"] = str(e)
                
                sync_status["running"] = False
            else:
                sync_status["error"] = "Not connected to Blink"
            
            # Wait for next interval
            await asyncio.sleep(interval * 60)
            
        except asyncio.CancelledError:
            logger.info("Continuous sync stopped")
            sync_status["enabled"] = False
            break
        except Exception as e:
            logger.error(f"Sync loop error: {e}")
            sync_status["error"] = str(e)
            await asyncio.sleep(60)


def start_continuous_sync():
    """Start continuous sync background task."""
    global continuous_sync_task
    
    if continuous_sync_task is None or continuous_sync_task.done():
        continuous_sync_task = asyncio.create_task(continuous_sync_loop())
        logger.info("Continuous sync task created")


def stop_continuous_sync():
    """Stop continuous sync background task."""
    global continuous_sync_task
    
    if continuous_sync_task and not continuous_sync_task.done():
        continuous_sync_task.cancel()
        logger.info("Continuous sync task cancelled")


# =============================================================================
# FastAPI App Setup
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    init_db()
    DEFAULT_DOWNLOADS_DIR.mkdir(exist_ok=True)
    
    # Start scheduler if enabled
    settings = load_settings()
    if settings.get("scheduler_enabled"):
        start_scheduler()
    
    # Start continuous sync if enabled
    if settings.get("continuous_sync_enabled"):
        start_continuous_sync()
    
    yield
    
    # Shutdown
    stop_scheduler()
    stop_continuous_sync()


app = FastAPI(title="Blink Hub", version=VERSION, lifespan=lifespan)

# Request logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    # Log POST requests to important endpoints
    if request.method == "POST" and not request.url.path.startswith("/static"):
        logger.info(f"API Request: {request.method} {request.url.path}")
    response = await call_next(request)
    return response

# Serve static files
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Serve thumbnails
THUMBNAILS_DIR = BASE_DIR / "thumbnails"
THUMBNAILS_DIR.mkdir(exist_ok=True)
app.mount("/thumbnails", StaticFiles(directory=str(THUMBNAILS_DIR)), name="thumbnails")


# =============================================================================
# Pydantic Models
# =============================================================================

class LoginRequest(BaseModel):
    email: str
    password: str
    remember_me: bool = True


class PinRequest(BaseModel):
    pin: str
    remember_me: bool = True


# Track remember_me preference between login and 2FA
remember_me_preference = True


class DownloadRequest(BaseModel):
    since_days: int = 7
    cameras: list[str] = []
    browser_timezone: Optional[str] = None  # Timezone detected from browser for "auto" mode


class SyncRequest(BaseModel):
    cameras: list[str] = []
    browser_timezone: Optional[str] = None  # Timezone detected from browser for "auto" mode


class SettingsRequest(BaseModel):
    download_dir: str
    scheduler_enabled: bool
    scheduler_time: str
    scheduler_days: int
    # Email settings
    email_enabled: bool = False
    email_smtp_server: str = ""
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_from: str = ""
    email_to: str = ""
    email_on_success: bool = True
    email_on_error: bool = True
    # Retention settings
    retention_enabled: bool = False
    retention_days: int = 30
    # Disk space warning
    disk_warning_enabled: bool = True
    disk_warning_gb: int = 10
    # Timezone for video filenames
    timezone: str = "auto"


class TestEmailRequest(BaseModel):
    pass


# =============================================================================
# API Routes - Pages
# =============================================================================

@app.get("/manifest.json")
async def manifest():
    """Serve PWA manifest."""
    return JSONResponse({
        "name": "Blink Hub",
        "short_name": "Blink DL",
        "description": "Download and manage Blink camera videos",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#0a0a0f",
        "theme_color": "#6366f1",
        "icons": [
            {
                "src": "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><rect fill='%236366f1' rx='20' width='100' height='100'/><text x='50' y='65' text-anchor='middle' font-size='50'>📹</text></svg>",
                "sizes": "192x192",
                "type": "image/svg+xml"
            }
        ]
    })

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main page."""
    template_path = BASE_DIR / "templates" / "index.html"
    if template_path.exists():
        return template_path.read_text(encoding="utf-8")
    return "<h1>Blink Hub</h1><p>Template not found</p>"


# =============================================================================
# API Routes - Status & Info
# =============================================================================

@app.get("/api/status")
async def get_status():
    """Get current status of the downloader."""
    global blink_instance
    
    cameras = []
    logged_in = False
    
    # Check for demo query param or if no blink instance
    demo_mode = not blink_instance or not blink_instance.cameras
    
    if blink_instance:
        if blink_instance.cameras:
            cameras = list(blink_instance.cameras.keys())
            logged_in = True
        elif hasattr(blink_instance, 'key_required') and blink_instance.key_required:
            logged_in = False
    
    settings = load_settings()
    stats = get_download_stats()
    disk = get_disk_space(get_download_dir())
    
    # Check disk space warning
    disk_warning = None
    if settings.get("disk_warning_enabled"):
        warning_bytes = settings.get("disk_warning_gb", 10) * 1024 * 1024 * 1024
        if disk["free"] < warning_bytes:
            disk_warning = f"Low disk space: {format_bytes(disk['free'])} remaining"
    
    return {
        "version": VERSION,
        "logged_in": logged_in,
        "demo_mode": demo_mode and not logged_in,
        "awaiting_2fa": blink_instance is not None and not logged_in,
        "cameras": cameras if cameras else ["Front Door", "Backyard", "Living Room", "Garage", "Front Porch"],  # Demo cameras if not logged in
        "download_status": download_status,
        "history": get_download_history(10),
        "settings": settings,
        "stats": stats,
        "disk": {
            "free": disk["free"],
            "total": disk["total"],
            "percent_used": disk["percent_used"],
            "free_formatted": format_bytes(disk["free"]),
            "total_formatted": format_bytes(disk["total"])
        },
        "disk_warning": disk_warning,
        "notifications": get_notifications(),
        "scheduler_next_run": get_next_scheduled_run()
    }


@app.get("/api/logs")
async def get_logs(count: int = 200, level: str = None):
    """Get recent log entries from buffer."""
    logs = log_buffer.get_recent(count)
    
    # Filter by level if specified
    if level:
        level = level.upper()
        logs = [l for l in logs if l.get("level") == level]
    
    return {
        "logs": logs,
        "total": len(log_buffer.get_all()),
        "returned": len(logs)
    }


@app.post("/api/logs/clear")
async def clear_logs():
    """Clear the log buffer."""
    log_buffer.clear()
    logger.info("Log buffer cleared by user")
    return {"success": True, "message": "Log buffer cleared"}


@app.get("/api/videos")
async def get_videos(limit: int = 50):
    """Get list of downloaded videos with thumbnails."""
    videos = get_downloaded_videos(limit)
    return {"videos": videos}


# =============================================================================
# Local Video Browser API
# =============================================================================

@app.get("/api/local-videos")
async def browse_videos(
    cameras: str = None,  # Comma-separated camera names
    camera_groups: str = None,  # Comma-separated group names
    start_date: str = None,  # YYYY-MM-DD
    end_date: str = None,  # YYYY-MM-DD
    start_time: str = None,  # HH:MM
    end_time: str = None,  # HH:MM
    tags: str = None,  # Comma-separated tag names
    starred: bool = None,
    unreviewed: bool = None,
    min_duration: int = None,
    max_duration: int = None,
    search: str = None,
    sort_by: str = "created_at",
    sort_order: str = "desc",
    page: int = 1,
    per_page: int = 50
):
    """Browse locally downloaded videos with comprehensive filtering."""
    logger.info(f"Loading videos: cameras={cameras}, dates={start_date} to {end_date}, page={page}")
    result = browse_local_videos(
        cameras=cameras.split(',') if cameras else None,
        camera_groups=camera_groups.split(',') if camera_groups else None,
        start_date=start_date,
        end_date=end_date,
        start_time=start_time,
        end_time=end_time,
        tags=tags.split(',') if tags else None,
        starred_only=starred,
        unreviewed_only=unreviewed,
        min_duration=min_duration,
        max_duration=max_duration,
        search_text=search,
        sort_by=sort_by,
        sort_order=sort_order.upper(),
        page=page,
        per_page=per_page
    )
    logger.info(f"Found {len(result.get('videos', []))} videos (page {result.get('page', 1)} of {result.get('total_pages', 1)})")
    return result


@app.post("/api/local-videos/regenerate-thumbnails")
async def regenerate_all_thumbnails(background_tasks: BackgroundTasks):
    """Regenerate thumbnails for all local videos."""
    logger.info("Thumbnail regeneration requested")
    
    # First check if ffmpeg is available
    ffmpeg_available = False
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
        ffmpeg_available = result.returncode == 0
    except:
        pass
    
    if not ffmpeg_available:
        logger.warning("ffmpeg not found - thumbnails require ffmpeg to be installed")
        return {
            "success": False, 
            "message": "ffmpeg is not installed. Please install ffmpeg to generate thumbnails. Download from https://ffmpeg.org/download.html"
        }
    
    def regenerate_task():
        logger.info("Starting thumbnail regeneration task")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, file_path FROM downloaded_videos WHERE thumbnail_path IS NULL OR thumbnail_path = ''")
        videos = cursor.fetchall()
        conn.close()
        
        logger.info(f"Found {len(videos)} videos without thumbnails")
        
        regenerated = 0
        for video_id, file_path in videos:
            if file_path and Path(file_path).exists():
                thumb_path = generate_thumbnail(file_path, THUMBNAILS_DIR)
                if thumb_path:
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute("UPDATE downloaded_videos SET thumbnail_path = ? WHERE id = ?", (thumb_path, video_id))
                    conn.commit()
                    conn.close()
                    regenerated += 1
        
        logger.info(f"Regenerated {regenerated} thumbnails")
    
    background_tasks.add_task(regenerate_task)
    return {"success": True, "message": f"Thumbnail regeneration started in background"}


@app.get("/api/local-videos/{video_id}")
async def get_local_video(video_id: str):
    """Get details for a specific local video."""
    video = get_video_by_id(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return video


@app.post("/api/local-videos/{video_id}/star")
async def toggle_video_star(video_id: str, starred: bool = True):
    """Toggle starred status of a video."""
    update_video_flags(video_id, starred=starred)
    return {"success": True, "starred": starred}


@app.post("/api/local-videos/{video_id}/review")
async def mark_video_reviewed(video_id: str, reviewed: bool = True):
    """Mark a video as reviewed."""
    update_video_flags(video_id, reviewed=reviewed)
    return {"success": True, "reviewed": reviewed}


@app.delete("/api/local-videos/{video_id}")
async def delete_local_video(video_id: str):
    """Delete a local video file and database record."""
    video = get_video_by_id(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Delete file
    file_path = video.get("file_path")
    if file_path and Path(file_path).exists():
        try:
            Path(file_path).unlink()
        except Exception as e:
            logger.error(f"Error deleting video file: {e}")
    
    # Delete thumbnail if exists
    thumb_path = video.get("thumbnail_path")
    if thumb_path and Path(thumb_path).exists():
        try:
            Path(thumb_path).unlink()
        except:
            pass
    
    # Delete database record
    delete_video_record(video_id)
    
    return {"success": True}


@app.post("/api/local-videos/bulk-delete")
async def bulk_delete_videos(request: Request):
    """Delete multiple videos at once."""
    data = await request.json()
    video_ids = data.get("video_ids", [])
    
    deleted = 0
    errors = []
    
    for video_id in video_ids:
        try:
            video = get_video_by_id(video_id)
            if video:
                file_path = video.get("file_path")
                if file_path and Path(file_path).exists():
                    Path(file_path).unlink()
                delete_video_record(video_id)
                deleted += 1
        except Exception as e:
            errors.append(f"{video_id}: {str(e)}")
    
    return {"success": True, "deleted": deleted, "errors": errors}


@app.get("/api/local-videos/{video_id}/stream")
async def stream_local_video(video_id: str):
    """Stream a local video file."""
    video = get_video_by_id(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    file_path = video.get("file_path")
    if not file_path or not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(
        file_path,
        media_type="video/mp4",
        filename=Path(file_path).name
    )


@app.get("/api/local-videos/{video_id}/thumbnail")
async def get_local_video_thumbnail(video_id: str):
    """Get thumbnail for a local video."""
    video = get_video_by_id(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Check if thumbnail exists
    thumb_path = video.get("thumbnail_path")
    if thumb_path and Path(thumb_path).exists():
        return FileResponse(
            thumb_path,
            media_type="image/jpeg",
            filename=Path(thumb_path).name
        )
    
    # Try to generate thumbnail on-the-fly
    file_path = video.get("file_path")
    if file_path and Path(file_path).exists():
        thumb_path = generate_thumbnail(file_path, THUMBNAILS_DIR)
        if thumb_path and Path(thumb_path).exists():
            # Update database with new thumbnail path
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("UPDATE downloaded_videos SET thumbnail_path = ? WHERE id = ?", (thumb_path, video_id))
            conn.commit()
            conn.close()
            
            return FileResponse(
                thumb_path,
                media_type="image/jpeg",
                filename=Path(thumb_path).name
            )
    
    # Return placeholder
    raise HTTPException(status_code=404, detail="Thumbnail not available")


@app.get("/api/local-videos/{video_id}/download")
async def download_local_video(video_id: str):
    """Download a local video file."""
    video = get_video_by_id(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    file_path = video.get("file_path")
    if not file_path or not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(
        file_path,
        media_type="video/mp4",
        filename=Path(file_path).name,
        headers={"Content-Disposition": f"attachment; filename={Path(file_path).name}"}
    )


@app.post("/api/local-videos/{video_id}/notes")
async def save_video_notes(video_id: str, request: Request):
    """Save notes for a video."""
    data = await request.json()
    notes = data.get("notes", "")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE downloaded_videos SET notes = ? WHERE id = ?", (notes, video_id))
    conn.commit()
    conn.close()
    
    return {"success": True}


@app.get("/api/local-videos/day-counts")
async def get_video_day_counts(year: int, month: int):
    """Get video counts per day for a given month (for calendar view)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Format: YYYY-MM prefix
    month_prefix = f"{year}-{month:02d}"
    
    cursor.execute("""
        SELECT date(created_at) as day, COUNT(*) as count
        FROM downloaded_videos
        WHERE created_at LIKE ?
        GROUP BY date(created_at)
    """, (f"{month_prefix}%",))
    
    rows = cursor.fetchall()
    conn.close()
    
    counts = {row[0]: row[1] for row in rows}
    return {"counts": counts}


@app.post("/api/local-videos/export")
async def export_videos_zip(request: Request):
    """Export selected videos as a zip file."""
    import zipfile
    import tempfile
    
    data = await request.json()
    video_ids = data.get("video_ids", [])
    
    if not video_ids:
        raise HTTPException(status_code=400, detail="No videos selected")
    
    if len(video_ids) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 videos per export")
    
    # Get video file paths
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    placeholders = ",".join(["?" for _ in video_ids])
    cursor.execute(f"SELECT id, file_path, camera_name FROM downloaded_videos WHERE id IN ({placeholders})", video_ids)
    videos = cursor.fetchall()
    conn.close()
    
    if not videos:
        raise HTTPException(status_code=404, detail="No videos found")
    
    # Create zip file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    try:
        with zipfile.ZipFile(temp_file.name, 'w', zipfile.ZIP_DEFLATED) as zf:
            for vid_id, file_path, camera_name in videos:
                if file_path and Path(file_path).exists():
                    # Use camera name + original filename for organization
                    arcname = f"{camera_name}/{Path(file_path).name}"
                    zf.write(file_path, arcname)
        
        return FileResponse(
            temp_file.name,
            media_type="application/zip",
            filename=f"blink-videos-{datetime.now().strftime('%Y%m%d-%H%M%S')}.zip",
            background=BackgroundTask(lambda: Path(temp_file.name).unlink(missing_ok=True))
        )
    except Exception as e:
        Path(temp_file.name).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@app.get("/api/local-cameras")
async def get_local_cameras():
    """Get list of cameras that have local videos."""
    cameras = get_unique_cameras()
    return {"cameras": cameras}


# =============================================================================
# Tags API
# =============================================================================

@app.get("/api/tags")
async def list_tags():
    """Get all tags."""
    return {"tags": get_all_tags()}


class TagCreate(BaseModel):
    name: str
    color: str = "#3b82f6"


@app.post("/api/tags")
async def create_new_tag(tag: TagCreate):
    """Create a new tag."""
    try:
        tag_id = create_tag(tag.name, tag.color)
        return {"success": True, "id": tag_id}
    except Exception as e:
        if "UNIQUE constraint" in str(e):
            raise HTTPException(status_code=400, detail="Tag already exists")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/tags/{tag_id}")
async def update_existing_tag(tag_id: int, tag: TagCreate):
    """Update a tag."""
    update_tag(tag_id, tag.name, tag.color)
    return {"success": True}


@app.delete("/api/tags/{tag_id}")
async def delete_existing_tag(tag_id: int):
    """Delete a tag."""
    delete_tag(tag_id)
    return {"success": True}


@app.post("/api/local-videos/{video_id}/tags/{tag_id}")
async def add_video_tag(video_id: str, tag_id: int):
    """Add a tag to a video."""
    add_tag_to_video(video_id, tag_id)
    return {"success": True}


@app.delete("/api/local-videos/{video_id}/tags/{tag_id}")
async def remove_video_tag(video_id: str, tag_id: int):
    """Remove a tag from a video."""
    remove_tag_from_video(video_id, tag_id)
    return {"success": True}


@app.post("/api/local-videos/bulk-tag")
async def bulk_tag_videos(request: Request):
    """Add a tag to multiple videos at once."""
    data = await request.json()
    video_ids = data.get("video_ids", [])
    tag_id = data.get("tag_id")
    
    if not tag_id:
        raise HTTPException(status_code=400, detail="tag_id required")
    
    add_tag_to_videos(video_ids, tag_id)
    return {"success": True, "tagged": len(video_ids)}


# =============================================================================
# Camera Groups API
# =============================================================================

@app.get("/api/camera-groups")
async def list_camera_groups():
    """Get all camera groups."""
    return {"groups": get_all_camera_groups()}


class CameraGroupCreate(BaseModel):
    name: str
    color: str = "#6b7280"
    cameras: list = []


@app.post("/api/camera-groups")
async def create_new_camera_group(group: CameraGroupCreate):
    """Create a new camera group."""
    try:
        group_id = create_camera_group(group.name, group.color, group.cameras)
        return {"success": True, "id": group_id}
    except Exception as e:
        if "UNIQUE constraint" in str(e):
            raise HTTPException(status_code=400, detail="Group already exists")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/camera-groups/{group_id}")
async def update_existing_camera_group(group_id: int, group: CameraGroupCreate):
    """Update a camera group."""
    update_camera_group(group_id, group.name, group.color, group.cameras)
    return {"success": True}


@app.delete("/api/camera-groups/{group_id}")
async def delete_existing_camera_group(group_id: int):
    """Delete a camera group."""
    delete_camera_group(group_id)
    return {"success": True}


# =============================================================================
# Sync Status API
# =============================================================================

@app.get("/api/sync-status")
async def get_sync_status():
    """Get continuous sync status."""
    return sync_status


@app.post("/api/sync/toggle")
async def toggle_continuous_sync(request: Request):
    """Enable or disable continuous sync."""
    data = await request.json()
    enabled = data.get("enabled", False)
    interval = data.get("interval_minutes", 5)
    
    # Update settings
    settings = load_settings()
    settings["continuous_sync_enabled"] = enabled
    settings["sync_interval_minutes"] = interval
    save_settings(settings)
    
    if enabled:
        start_continuous_sync()
        sync_status["enabled"] = True
        sync_status["interval_minutes"] = interval
    else:
        stop_continuous_sync()
        sync_status["enabled"] = False
    
    return {"success": True, "enabled": enabled}


@app.get("/api/history")
async def get_history():
    """Get download history."""
    return {"history": get_download_history(50)}


@app.get("/api/cameras")
async def get_cameras():
    """Get real-time status of all cameras and sync modules."""
    # This endpoint now returns the same data as camera-status
    return await get_camera_status()


@app.get("/api/camera-stats")
async def get_camera_download_stats():
    """Get camera download statistics."""
    stats = get_download_stats()
    return {"cameras": stats.get("by_camera", {})}


@app.get("/api/camera-status")
async def get_camera_status():
    """Get real-time status of all cameras and sync modules."""
    global blink_instance
    
    # Check for demo mode
    demo_mode = not blink_instance or not blink_instance.cameras
    
    if demo_mode:
        # Return demo data so user can preview UI
        demo_cameras = [
            {
                "name": "Front Door",
                "camera_id": "12345",
                "serial": "G8T1-1234-5678",
                "camera_type": "outdoor",
                "product_type": "catalina",
                "motion_enabled": True,
                "motion_detected": False,
                "power_source": "battery",
                "power_display": "ok",
                "battery_state": "ok",
                "temperature": 62,
                "wifi_strength": 4,
                "last_record": "2025-12-08T10:30:00",
                "network_id": "1001",
            },
            {
                "name": "Backyard",
                "camera_id": "12346",
                "serial": "G8T1-2345-6789",
                "camera_type": "outdoor",
                "product_type": "catalina",
                "motion_enabled": True,
                "motion_detected": True,
                "power_source": "battery",
                "power_display": "low",
                "battery_state": "low",
                "temperature": 58,
                "wifi_strength": 2,
                "last_record": "2025-12-08T14:15:00",
                "network_id": "1001",
            },
            {
                "name": "Living Room",
                "camera_id": "12347",
                "serial": "M8T1-3456-7890",
                "camera_type": "mini",
                "product_type": "owl",
                "motion_enabled": False,
                "motion_detected": False,
                "power_source": "usb",
                "power_display": "USB",
                "battery_state": None,
                "temperature": 72,
                "wifi_strength": 5,
                "last_record": "2025-12-08T12:00:00",
                "network_id": "1001",
            },
            {
                "name": "Garage",
                "camera_id": "12348",
                "serial": "M8T1-4567-8901",
                "camera_type": "mini",
                "product_type": "owl",
                "motion_enabled": True,
                "motion_detected": False,
                "power_source": "usb",
                "power_display": "USB",
                "battery_state": None,
                "temperature": 55,
                "wifi_strength": 3,
                "last_record": "2025-12-07T22:45:00",
                "network_id": "1001",
            },
            {
                "name": "Front Porch",
                "camera_id": "12349",
                "serial": "D8T1-5678-9012",
                "camera_type": "doorbell",
                "product_type": "lotus",
                "motion_enabled": True,
                "motion_detected": False,
                "power_source": "usb",
                "power_display": "Wired",
                "battery_state": None,
                "temperature": 60,
                "wifi_strength": 4,
                "last_record": "2025-12-08T09:20:00",
                "network_id": "1001",
            },
        ]
        demo_sync_modules = [
            {
                "name": "Home Base",
                "sync_id": "5001",
                "serial": "S8T1-0001-0001",
                "status": "online",
                "network_id": "1001",
                "armed": True,
                "cameras_count": 4,
            },
        ]
        return {
            "cameras": demo_cameras,
            "sync_modules": demo_sync_modules,
            "camera_count": len(demo_cameras),
            "timestamp": datetime.now().isoformat(),
            "demo_mode": True,
            "error": None
        }
    
    try:
        # Refresh data from Blink servers (respects internal throttling)
        await blink_instance.refresh()
        
        cameras = []
        for name, camera in blink_instance.cameras.items():
            # Get attributes from camera
            attrs = camera.attributes if hasattr(camera, 'attributes') else {}
            
            # Determine camera type
            cam_type = getattr(camera, 'camera_type', '') or ''
            product_type = getattr(camera, 'product_type', '') or ''
            
            # Detect if camera is USB/wired powered (Minis and some others)
            # Blink Minis are "mini" or "owl", doorbells are typically wired too
            is_wired = cam_type.lower() in ['mini', 'owl', 'doorbell', 'lotus'] or \
                       product_type.lower() in ['owl', 'lotus', 'catalina']
            
            battery_state = getattr(camera, 'battery_state', None)
            battery_voltage = getattr(camera, 'battery_voltage', None)
            
            # Also check attributes dict for battery info
            if isinstance(attrs, dict):
                if not battery_state:
                    battery_state = attrs.get('battery_state') or attrs.get('battery')
                if not battery_voltage:
                    battery_voltage = attrs.get('battery_voltage')
            
            # Log battery info for debugging
            logger.debug(f"Camera {name} battery: state={battery_state}, voltage={battery_voltage}")
            
            # Determine online status - check multiple indicators
            wifi_strength = getattr(camera, 'wifi_strength', None) or attrs.get('wifi_strength') or attrs.get('signal_strength')
            
            # Get thumbnail URL and extract timestamp for staleness check
            thumbnail_url = getattr(camera, 'thumbnail', None) or attrs.get('thumbnail', '')
            thumbnail_age_hours = None
            thumbnail_ts = None
            if thumbnail_url and 'ts=' in thumbnail_url:
                try:
                    ts_match = re.search(r'ts=(\d+)', thumbnail_url)
                    if ts_match:
                        thumbnail_ts = int(ts_match.group(1))
                        current_ts = int(datetime.now().timestamp())
                        thumbnail_age_seconds = current_ts - thumbnail_ts
                        thumbnail_age_hours = thumbnail_age_seconds / 3600
                except Exception as e:
                    logger.debug(f"Failed to parse thumbnail timestamp: {e}")
            
            # Log all potentially relevant status attributes for debugging
            status_attrs = {
                'online': getattr(camera, 'online', None),
                'status': getattr(camera, 'status', None),
                'active': getattr(camera, 'active', None),
                'enabled': getattr(camera, 'enabled', None),
                'wifi_strength': wifi_strength,
                'battery_state': battery_state,
                'is_wired': is_wired,
                'cam_type': cam_type,
                'thumbnail_age_hours': round(thumbnail_age_hours, 1) if thumbnail_age_hours else None,
                'attrs_online': attrs.get('online') if isinstance(attrs, dict) else None,
                'attrs_active': attrs.get('active') if isinstance(attrs, dict) else None,
                'attrs_status': attrs.get('status') if isinstance(attrs, dict) else None,
            }
            logger.info(f"Camera {name} status check: {status_attrs}")
            
            # Check explicit online/status attributes
            camera_online = None  # None = unknown, True = online, False = offline
            if hasattr(camera, 'online') and camera.online is not None:
                camera_online = bool(camera.online)
                logger.debug(f"Camera {name} online from camera.online: {camera_online}")
            elif hasattr(camera, 'status') and camera.status is not None:
                status_val = str(getattr(camera, 'status', '')).lower()
                if status_val in ['online', 'true', '1', 'yes', 'active']:
                    camera_online = True
                elif status_val in ['offline', 'false', '0', 'no', 'inactive']:
                    camera_online = False
                logger.debug(f"Camera {name} online from camera.status ({status_val}): {camera_online}")
            elif isinstance(attrs, dict):
                if 'online' in attrs and attrs.get('online') is not None:
                    camera_online = bool(attrs.get('online'))
                    logger.debug(f"Camera {name} online from attrs['online']: {camera_online}")
                elif 'active' in attrs and attrs.get('active') is not None:
                    camera_online = bool(attrs.get('active'))
                    logger.debug(f"Camera {name} online from attrs['active']: {camera_online}")
            
            # PRIMARY CHECK: Use thumbnail timestamp to detect stale/offline cameras
            # If thumbnail is older than 24 hours, camera is likely offline regardless of other data
            # (Blink caches wifi_strength, battery, etc. even when camera is dead)
            STALE_THRESHOLD_HOURS = 24
            if camera_online is None and thumbnail_age_hours is not None:
                if thumbnail_age_hours > STALE_THRESHOLD_HOURS:
                    camera_online = False
                    logger.info(f"Camera {name} detected OFFLINE: thumbnail is {thumbnail_age_hours:.1f} hours old (>{STALE_THRESHOLD_HOURS}h threshold)")
                else:
                    camera_online = True
                    logger.debug(f"Camera {name} detected online: thumbnail is {thumbnail_age_hours:.1f} hours old")
            
            # FALLBACK: Infer online/offline status based on wifi_strength and battery_state
            # Key insight: if wifi_strength is present, camera is communicating (online)
            # If wifi_strength is missing BUT battery_state is present, it's a battery camera that's offline
            # If BOTH are missing, it's likely a Mini (USB) that just doesn't report these - don't assume offline
            if camera_online is None:
                if wifi_strength is not None and wifi_strength != 0 and wifi_strength != '':
                    # Has wifi signal = definitely online
                    camera_online = True
                    logger.debug(f"Camera {name} inferred online: wifi_strength={wifi_strength}")
                elif battery_state is not None and not is_wired:
                    # Battery camera (has battery_state) but no wifi = likely offline/dead battery
                    camera_online = False
                    logger.info(f"Camera {name} inferred offline: battery camera with no wifi_strength")
                else:
                    # No wifi_strength AND no battery_state = probably a Mini/USB camera
                    # Don't assume offline, leave as unknown (will show as online by default)
                    if is_wired:
                        camera_online = True  # Assume wired cameras are online
                        logger.debug(f"Camera {name} assumed online: wired camera")
                    else:
                        # Could be a Mini not detected as wired, or unknown type
                        # Default to online to avoid false positives
                        camera_online = True
                        logger.debug(f"Camera {name} assumed online: no battery_state (likely Mini/USB)")
            
            # Determine online status string
            if camera_online is True:
                online_status = "online"
            elif camera_online is False:
                online_status = "offline"
            else:
                online_status = "unknown"
            
            # If wired camera, show "USB Power" instead of battery
            if is_wired:
                power_source = "usb"
                power_display = "USB"
            elif battery_state:
                power_source = "battery"
                # If camera is offline and battery-powered, the battery is likely dead
                # regardless of what the (stale) cached battery_state says
                if camera_online is False:
                    power_display = "Offline"
                    logger.info(f"Camera {name} appears offline - battery may be dead (cached state: {battery_state})")
                else:
                    power_display = battery_state  # Show exactly what Blink returns
            else:
                power_source = "unknown"
                power_display = "--"
            
            # Recording info - try multiple sources for last_record
            last_record_val = None
            # Try direct attribute
            if getattr(camera, 'last_record', None):
                last_record_val = getattr(camera, 'last_record')
            # Try attributes dict
            elif isinstance(attrs, dict):
                last_record_val = attrs.get('last_record') or attrs.get('updated_at') or attrs.get('created_at')
            # Try clip info
            if not last_record_val and hasattr(camera, 'clip') and camera.clip:
                clip = camera.clip
                if isinstance(clip, dict):
                    last_record_val = clip.get('created_at') or clip.get('updated_at')
                elif hasattr(clip, 'created_at'):
                    last_record_val = getattr(clip, 'created_at', None)
            
            # Try extracting timestamp from thumbnail URL (Blink embeds it as ?ts=UNIX_TIMESTAMP)
            if not last_record_val:
                thumbnail_url = getattr(camera, 'thumbnail', None) or attrs.get('thumbnail', '')
                if thumbnail_url and 'ts=' in thumbnail_url:
                    try:
                        ts_match = re.search(r'ts=(\d+)', thumbnail_url)
                        if ts_match:
                            ts_val = int(ts_match.group(1))
                            # Convert Unix timestamp to ISO format
                            last_record_val = datetime.fromtimestamp(ts_val).isoformat()
                    except Exception as e:
                        logger.debug(f"Failed to parse thumbnail timestamp: {e}")
            
            camera_info = {
                "name": name,
                "camera_id": getattr(camera, 'camera_id', None),
                "serial": getattr(camera, 'serial', None),
                "camera_type": cam_type or "Camera",
                "product_type": product_type,
                
                # Status indicators
                "online_status": online_status,
                "thumbnail_age_hours": round(thumbnail_age_hours, 1) if thumbnail_age_hours is not None else None,
                "motion_enabled": getattr(camera, 'motion_enabled', None),
                "motion_detected": getattr(camera, 'motion_detected', False),
                "armed": getattr(camera, 'arm', None) if hasattr(camera, 'arm') else None,
                
                # Power info
                "power_source": power_source,
                "power_display": power_display,
                "battery_state": battery_state,
                "battery_voltage": battery_voltage,
                
                # Sensor data
                "temperature": getattr(camera, 'temperature', None),
                "temperature_calibrated": getattr(camera, 'temperature_calibrated', None),
                "wifi_strength": getattr(camera, 'wifi_strength', None) or attrs.get('wifi_strength') or attrs.get('signal_strength'),
                
                # Recording info
                "last_record": last_record_val,
                "thumbnail": getattr(camera, 'thumbnail', None),
                "thumbnail_time": last_record_val,
                "clip": getattr(camera, 'clip', None),
                
                # Network info
                "network_id": getattr(camera, 'network_id', None),
                "sync_module": attrs.get('sync_module', None) if isinstance(attrs, dict) else None,
            }
            
            # Try to find which sync module this camera belongs to
            for sync_name, sync_mod in blink_instance.sync.items():
                if hasattr(sync_mod, 'cameras') and name in sync_mod.cameras:
                    camera_info["sync_module"] = sync_name
                    break
            
            # Get groups this camera belongs to
            camera_info["groups"] = get_camera_groups_for_camera(name)
            
            # Convert temperature to Fahrenheit if available (Blink reports in Fahrenheit)
            if camera_info["temperature"] is not None:
                camera_info["temperature_f"] = camera_info["temperature"]
                camera_info["temperature_c"] = round((camera_info["temperature"] - 32) * 5/9, 1)
            
            cameras.append(camera_info)
        
        # Get sync module information
        sync_modules = []
        if hasattr(blink_instance, 'sync') and blink_instance.sync:
            for name, sync in blink_instance.sync.items():
                # Count cameras in this sync module
                cam_count = 0
                camera_names = []
                if hasattr(sync, 'cameras') and sync.cameras:
                    camera_names = list(sync.cameras.keys())
                    cam_count = len(camera_names)
                
                # Skip pseudo-sync modules for Minis (sync with 1 camera where camera name == sync name)
                if cam_count == 1 and camera_names[0] == name:
                    logger.debug(f"Skipping pseudo-sync for Mini camera: {name}")
                    continue
                
                # Determine online status - check multiple attributes
                online_status = "unknown"
                if hasattr(sync, 'online'):
                    online_status = "online" if sync.online else "offline"
                elif hasattr(sync, 'status'):
                    status_val = str(getattr(sync, 'status', '')).lower()
                    if status_val in ['online', 'true', '1', 'yes']:
                        online_status = "online"
                    elif status_val in ['offline', 'false', '0', 'no']:
                        online_status = "offline"
                    else:
                        online_status = status_val if status_val else "unknown"
                
                # Try multiple sources for wifi_strength
                sync_attrs = sync.attributes if hasattr(sync, 'attributes') else {}
                wifi_strength = (
                    getattr(sync, 'wifi_strength', None) or 
                    sync_attrs.get('wifi_strength') or 
                    sync_attrs.get('signal_strength') or
                    (sync_attrs.get('network_info', {}) or {}).get('wifi_strength')
                )
                
                # Get local storage info
                local_storage_enabled = getattr(sync, 'local_storage_enabled', None)
                local_storage_status = getattr(sync, 'local_storage_status', None)
                
                # Try to get WiFi SSID from network_info or attributes
                network_info = sync_attrs.get('network_info', {}) or {}
                wifi_ssid = network_info.get('ssid') or sync_attrs.get('wifi_ssid') or sync_attrs.get('ssid')
                
                # Get firmware version
                fw_version = sync_attrs.get('fw_version') or getattr(sync, 'fw_version', None)
                
                # Separate Mini cameras from sync module cameras
                mini_cameras = []
                sync_cameras = []
                for cam_name in camera_names:
                    cam_type = None
                    # Find the camera to check its type
                    for cam in cameras:
                        if cam.get('name') == cam_name:
                            cam_type = cam.get('camera_type', '')
                            break
                    if cam_type and 'mini' in cam_type.lower():
                        mini_cameras.append(cam_name)
                    else:
                        sync_cameras.append(cam_name)
                
                sync_info = {
                    "name": name,
                    "sync_id": getattr(sync, 'sync_id', None),
                    "serial": getattr(sync, 'serial', None),
                    "status": online_status,
                    "network_id": getattr(sync, 'network_id', None),
                    "armed": getattr(sync, 'arm', False) if hasattr(sync, 'arm') else False,
                    "cameras_count": cam_count,
                    "cameras": camera_names,
                    "sync_cameras": sync_cameras,  # Cameras that actually use the sync module
                    "mini_cameras": mini_cameras,  # Mini cameras on same network
                    "wifi_strength": wifi_strength,
                    "wifi_ssid": wifi_ssid,
                    "fw_version": fw_version,
                    "local_storage_enabled": local_storage_enabled,
                    "local_storage_status": local_storage_status,
                    "storage_used": getattr(sync, 'storage_used', None),
                }
                sync_modules.append(sync_info)
        
        logger.info(f"Camera status retrieved: {len(cameras)} cameras, {len(sync_modules)} sync modules")
        
        return {
            "cameras": cameras,
            "sync_modules": sync_modules,
            "camera_count": len(cameras),
            "timestamp": datetime.now().isoformat(),
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Error getting camera status: {e}")
        logger.error(traceback.format_exc())
        return {"cameras": [], "sync_modules": [], "camera_count": 0, "error": str(e)}


@app.post("/api/camera/{camera_name:path}/refresh-thumbnail")
async def refresh_camera_thumbnail(camera_name: str):
    """Request a new thumbnail from a specific camera."""
    global blink_instance
    
    if not blink_instance or not blink_instance.cameras:
        raise HTTPException(status_code=400, detail="Not logged in")
    
    if camera_name not in blink_instance.cameras:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_name}' not found")
    
    try:
        camera = blink_instance.cameras[camera_name]
        logger.info(f"Attempting snap_picture for {camera_name} (type: {camera.camera_type})")
        
        # snap_picture() requests a new thumbnail from the camera
        result = await camera.snap_picture()
        logger.info(f"snap_picture result for {camera_name}: {result}")
        
        # Give it a moment then refresh to get the new thumbnail URL
        await asyncio.sleep(2)
        await blink_instance.refresh()
        
        add_notification(f"Thumbnail refresh requested for {camera_name}", "success")
        return {"success": True, "message": f"Thumbnail refresh requested for {camera_name}", "result": str(result)}
    except Exception as e:
        logger.error(f"Error refreshing thumbnail for {camera_name}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/camera/{camera_name:path}/toggle-motion")
async def toggle_camera_motion(camera_name: str, enable: bool = True):
    """Enable or disable motion detection for a specific camera."""
    global blink_instance
    
    if not blink_instance or not blink_instance.cameras:
        raise HTTPException(status_code=400, detail="Not logged in")
    
    if camera_name not in blink_instance.cameras:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_name}' not found")
    
    try:
        camera = blink_instance.cameras[camera_name]
        
        # The set_motion_detect method is deprecated in newer blinkpy versions
        # For individual camera motion detection, we need to use the arm property
        # or the sync module's arm status
        # Try multiple approaches
        
        success = False
        error_msg = ""
        
        # Method 1: Try the arm property setter if available
        if hasattr(camera, 'arm') and not callable(getattr(camera, 'arm', None)):
            try:
                camera.arm = enable
                success = True
                logger.info(f"Set camera.arm = {enable} for {camera_name}")
            except Exception as e:
                error_msg = f"arm property failed: {e}"
                logger.warning(error_msg)
        
        # Method 2: Try async_arm if it exists on camera
        if not success and hasattr(camera, 'async_arm'):
            try:
                await camera.async_arm(enable)
                success = True
                logger.info(f"Called camera.async_arm({enable}) for {camera_name}")
            except Exception as e:
                error_msg = f"async_arm failed: {e}"
                logger.warning(error_msg)
        
        # Method 3: Try the deprecated set_motion_detect as fallback
        if not success and hasattr(camera, 'set_motion_detect'):
            try:
                await camera.set_motion_detect(enable)
                success = True
                logger.info(f"Called camera.set_motion_detect({enable}) for {camera_name}")
            except Exception as e:
                error_msg = f"set_motion_detect failed: {e}"
                logger.warning(error_msg)
        
        if not success:
            # For some camera types like doorbells, individual motion control isn't supported
            raise HTTPException(
                status_code=400, 
                detail=f"Motion detection toggle not supported for this camera type. {error_msg}"
            )
        
        await blink_instance.refresh()
        status = "enabled" if enable else "disabled"
        add_notification(f"Motion detection {status} for {camera_name}", "success")
        return {"success": True, "message": f"Motion detection {status} for {camera_name}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error toggling motion for {camera_name}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# Live view session tracking
live_view_sessions = {}

@app.post("/api/camera/{camera_name:path}/live-view/start")
async def start_live_view(camera_name: str):
    """Start a live view session for a camera (experimental)."""
    global blink_instance, live_view_sessions
    
    if not blink_instance:
        raise HTTPException(status_code=400, detail="Not logged in")
    
    camera = find_camera(camera_name)
    if not camera:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_name}' not found")
    
    try:
        session_id = f"{camera_name}_{int(time.time())}"
        
        # Try to get live view URL if available
        live_url = None
        if hasattr(camera, 'get_liveview'):
            try:
                live_url = await camera.get_liveview()
                logger.info(f"Got live view URL for {camera_name}")
            except Exception as e:
                logger.warning(f"get_liveview not available for {camera_name}: {e}")
        
        # If no live URL, fall back to rapid refresh mode
        if not live_url:
            # Request initial snapshot
            await camera.snap_picture()
            
            live_view_sessions[session_id] = {
                "camera_name": camera_name,
                "started_at": time.time(),
                "mode": "refresh",  # Use refresh mode instead of true streaming
                "frame_count": 0
            }
            
            return {
                "success": True,
                "session_id": session_id,
                "mode": "refresh",
                "message": "Live refresh mode started. Frames will be captured on demand.",
                "refresh_interval_ms": 1500
            }
        else:
            live_view_sessions[session_id] = {
                "camera_name": camera_name,
                "started_at": time.time(),
                "mode": "stream",
                "url": live_url,
                "frame_count": 0
            }
            
            return {
                "success": True,
                "session_id": session_id,
                "mode": "stream",
                "url": live_url,
                "message": "Live stream available"
            }
    except Exception as e:
        logger.error(f"Error starting live view for {camera_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/camera/{camera_name:path}/live-view/frame")
async def get_live_frame(camera_name: str):
    """Capture a new frame for live view (refresh mode)."""
    global blink_instance
    
    if not blink_instance:
        raise HTTPException(status_code=400, detail="Not logged in")
    
    camera = find_camera(camera_name)
    if not camera:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_name}' not found")
    
    try:
        # Request new snapshot
        await camera.snap_picture()
        # Small delay for camera to process
        await asyncio.sleep(0.5)
        # Refresh to get new thumbnail URL
        await blink_instance.refresh()
        
        return {
            "success": True,
            "timestamp": time.time(),
            "thumbnail_url": f"/api/camera/{camera_name}/thumbnail?t={int(time.time())}"
        }
    except Exception as e:
        logger.error(f"Error getting live frame for {camera_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/camera/{camera_name:path}/live-view/stop")
async def stop_live_view(camera_name: str, session_id: str = None):
    """Stop a live view session."""
    global live_view_sessions
    
    if session_id and session_id in live_view_sessions:
        del live_view_sessions[session_id]
    
    return {"success": True, "message": "Live view stopped"}


@app.post("/api/sync/{sync_name}/arm")
async def arm_sync_module(sync_name: str, arm: bool = True):
    """Arm or disarm a sync module."""
    global blink_instance
    
    if not blink_instance:
        raise HTTPException(status_code=400, detail="Not logged in")
    
    if not hasattr(blink_instance, 'sync') or sync_name not in blink_instance.sync:
        raise HTTPException(status_code=404, detail=f"Sync module '{sync_name}' not found")
    
    try:
        sync = blink_instance.sync[sync_name]
        await sync.async_arm(arm)
        await blink_instance.refresh()
        status = "armed" if arm else "disarmed"
        add_notification(f"Sync module {sync_name} {status}", "success")
        return {"success": True, "message": f"Sync module {sync_name} {status}"}
    except Exception as e:
        logger.error(f"Error arming sync module {sync_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/camera/{camera_name:path}/thumbnail")
async def get_camera_thumbnail(camera_name: str):
    """Get the thumbnail image for a camera (proxied through server)."""
    global blink_instance
    
    if not blink_instance or not blink_instance.cameras:
        raise HTTPException(status_code=400, detail="Not logged in")
    
    if camera_name not in blink_instance.cameras:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_name}' not found")
    
    try:
        camera = blink_instance.cameras[camera_name]
        
        # Get cached image if available
        if hasattr(camera, '_cached_image') and camera._cached_image:
            return Response(
                content=camera._cached_image,
                media_type="image/jpeg"
            )
        
        # Try to fetch the image from the thumbnail URL
        thumbnail_url = getattr(camera, 'thumbnail', None)
        if not thumbnail_url:
            raise HTTPException(status_code=404, detail="No thumbnail available")
        
        # Fetch the image through the authenticated session
        if blink_instance.auth and hasattr(blink_instance.auth, 'session'):
            async with blink_instance.auth.session.get(thumbnail_url) as resp:
                if resp.status == 200:
                    image_data = await resp.read()
                    return Response(
                        content=image_data,
                        media_type="image/jpeg"
                    )
        
        raise HTTPException(status_code=404, detail="Could not fetch thumbnail")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting thumbnail for {camera_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/cameras/refresh-all-thumbnails")
async def refresh_all_thumbnails():
    """Request all cameras to snap new thumbnails."""
    global blink_instance
    
    if not blink_instance or not blink_instance.cameras:
        raise HTTPException(status_code=400, detail="Not logged in")
    
    results = []
    for name, camera in blink_instance.cameras.items():
        try:
            await camera.snap_picture()
            results.append({"camera": name, "success": True})
            # Small delay between requests to not overwhelm the API
            await asyncio.sleep(0.5)
        except Exception as e:
            logger.warning(f"Failed to snap picture for {name}: {e}")
            results.append({"camera": name, "success": False, "error": str(e)})
    
    # Refresh to get new thumbnail URLs
    await blink_instance.refresh()
    
    return {"success": True, "results": results}


@app.get("/api/video-clips")
async def get_video_clips(
    camera: str = None,
    since_days: int = 7,
    page: int = 1,
    limit: int = 50
):
    """Get list of video clips from Blink cloud with filtering."""
    global blink_instance
    
    if not blink_instance or not blink_instance.cameras:
        raise HTTPException(status_code=400, detail="Not logged in")
    
    try:
        # Get videos from Blink
        from datetime import datetime, timedelta
        since_date = datetime.now() - timedelta(days=since_days)
        since_str = since_date.strftime("%Y/%m/%d %H:%M")
        
        # Use sync module to get video metadata
        all_clips = []
        
        for sync_name, sync_module in blink_instance.sync.items():
            # Check if sync module has last_records (recent clips)
            if hasattr(sync_module, 'last_records') and sync_module.last_records:
                for cam_name, records in sync_module.last_records.items():
                    for record in records:
                        if isinstance(record, dict):
                            clip_info = {
                                "camera": cam_name,
                                "sync_module": sync_name,
                                "clip_url": record.get("clip", record.get("media", "")),
                                "thumbnail_url": record.get("thumbnail", ""),
                                "created_at": record.get("created_at", record.get("time", "")),
                                "id": record.get("id", ""),
                            }
                            all_clips.append(clip_info)
            
            # Also check cameras for recent_clips
            if hasattr(sync_module, 'cameras'):
                for cam_name, cam in sync_module.cameras.items():
                    if hasattr(cam, 'recent_clips') and cam.recent_clips:
                        for clip in cam.recent_clips:
                            if isinstance(clip, dict):
                                clip_info = {
                                    "camera": cam_name,
                                    "sync_module": sync_name,
                                    "clip_url": clip.get("clip", clip.get("media", "")),
                                    "thumbnail_url": clip.get("thumbnail", ""),
                                    "created_at": clip.get("created_at", clip.get("time", "")),
                                    "id": clip.get("id", ""),
                                }
                                all_clips.append(clip_info)
        
        # Also check main cameras dict
        for cam_name, cam in blink_instance.cameras.items():
            if hasattr(cam, 'recent_clips') and cam.recent_clips:
                for clip in cam.recent_clips:
                    clip_info = {
                        "camera": cam_name,
                        "clip_url": getattr(clip, 'clip', '') if hasattr(clip, 'clip') else clip.get("clip", clip.get("media", "")) if isinstance(clip, dict) else "",
                        "thumbnail_url": getattr(clip, 'thumbnail', '') if hasattr(clip, 'thumbnail') else clip.get("thumbnail", "") if isinstance(clip, dict) else "",
                        "created_at": getattr(clip, 'created_at', '') if hasattr(clip, 'created_at') else clip.get("created_at", clip.get("time", "")) if isinstance(clip, dict) else "",
                    }
                    if clip_info["clip_url"]:
                        all_clips.append(clip_info)
            
            # Also get the current clip
            if hasattr(cam, 'clip') and cam.clip:
                clip_info = {
                    "camera": cam_name,
                    "clip_url": cam.clip,
                    "thumbnail_url": getattr(cam, 'thumbnail', ''),
                    "created_at": getattr(cam, 'last_record', ''),
                }
                # Avoid duplicates
                if not any(c["clip_url"] == clip_info["clip_url"] for c in all_clips):
                    all_clips.append(clip_info)
        
        # Filter by camera if specified
        if camera and camera != "all":
            all_clips = [c for c in all_clips if c.get("camera", "").lower() == camera.lower()]
        
        # Sort by date (newest first)
        all_clips.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        # Paginate
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated = all_clips[start_idx:end_idx]
        
        return {
            "clips": paginated,
            "total": len(all_clips),
            "page": page,
            "limit": limit,
            "has_more": end_idx < len(all_clips)
        }
        
    except Exception as e:
        logger.error(f"Error getting video clips: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/cloud-parity")
async def check_cloud_parity():
    """Check if local videos are in sync with cloud videos."""
    global blink_instance
    
    if not blink_instance or not blink_instance.cameras:
        return {"error": "Not logged in to Blink"}
    
    try:
        from datetime import datetime, timedelta
        
        # Get videos from cloud (last 30 days)
        cloud_videos = []
        
        # Use the media/changed API to get cloud videos
        try:
            since = datetime.now() - timedelta(days=30)
            since_str = since.strftime("%Y-%m-%dT%H:%M:%S+00:00")
            
            url = f"{blink_instance.urls.base_url}/api/v1/accounts/{blink_instance.account_id}/media/changed?since={since_str}&page=1"
            
            async with ClientSession() as session:
                headers = blink_instance.auth.header
                async with session.get(url, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        media = data.get('media', [])
                        for item in media:
                            cloud_videos.append({
                                'id': item.get('id'),
                                'camera': item.get('device_name', item.get('camera', 'Unknown')),
                                'date': item.get('created_at', '')[:10] if item.get('created_at') else '',
                                'created_at': item.get('created_at', '')
                            })
        except Exception as e:
            logger.warning(f"Could not fetch cloud videos: {e}")
            return {"error": f"Could not fetch cloud videos: {str(e)}"}
        
        # Get local video IDs from database
        local_video_ids = set()
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM downloaded_videos")
            for row in cursor.fetchall():
                local_video_ids.add(str(row['id']))
            conn.close()
        except Exception as e:
            logger.warning(f"Could not fetch local videos: {e}")
        
        # Find missing videos
        missing = []
        for cv in cloud_videos:
            if cv['id'] and str(cv['id']) not in local_video_ids:
                missing.append(cv)
        
        return {
            "cloud_count": len(cloud_videos),
            "local_count": len(local_video_ids),
            "missing_count": len(missing),
            "missing": missing[:100],  # Limit to first 100
            "checked_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error checking cloud parity: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}


@app.get("/api/camera/{camera_name:path}/details")
async def get_camera_details(camera_name: str):
    """Get detailed information about a specific camera."""
    global blink_instance
    
    if not blink_instance or not blink_instance.cameras:
        raise HTTPException(status_code=400, detail="Not logged in")
    
    if camera_name not in blink_instance.cameras:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_name}' not found")
    
    try:
        camera = blink_instance.cameras[camera_name]
        attrs = camera.attributes if hasattr(camera, 'attributes') else {}
        
        # Build comprehensive camera info
        details = {
            "name": camera_name,
            "camera_id": getattr(camera, 'camera_id', None),
            "serial": getattr(camera, 'serial', None),
            "camera_type": getattr(camera, 'camera_type', ''),
            "product_type": getattr(camera, 'product_type', ''),
            
            # Status
            "motion_enabled": getattr(camera, 'motion_enabled', None),
            "motion_detected": getattr(camera, 'motion_detected', False),
            "armed": getattr(camera, 'arm', None) if hasattr(camera, 'arm') else None,
            
            # Power
            "battery_state": getattr(camera, 'battery_state', None),
            "battery_voltage": getattr(camera, 'battery_voltage', None),
            "battery_level": getattr(camera, 'battery_level', None),
            
            # Sensors
            "temperature": getattr(camera, 'temperature', None),
            "temperature_calibrated": getattr(camera, 'temperature_calibrated', None),
            "wifi_strength": getattr(camera, 'wifi_strength', None),
            
            # Media
            "thumbnail": getattr(camera, 'thumbnail', None),
            "clip": getattr(camera, 'clip', None),
            "last_record": getattr(camera, 'last_record', None),
            
            # Network
            "network_id": getattr(camera, 'network_id', None),
            "sync_module": attrs.get('sync_module', None) if isinstance(attrs, dict) else None,
            
            # Recent clips
            "recent_clips": [],
            
            # Raw attributes for debugging
            "attributes": attrs if isinstance(attrs, dict) else str(attrs),
        }
        
        # Get recent clips for this camera
        if hasattr(camera, 'recent_clips') and camera.recent_clips:
            for clip in camera.recent_clips[:10]:  # Last 10 clips
                if isinstance(clip, dict):
                    details["recent_clips"].append({
                        "clip_url": clip.get("clip", clip.get("media", "")),
                        "thumbnail_url": clip.get("thumbnail", ""),
                        "created_at": clip.get("created_at", clip.get("time", "")),
                    })
        
        return details
        
    except Exception as e:
        logger.error(f"Error getting camera details for {camera_name}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/camera/{camera_name:path}/debug")
async def debug_camera(camera_name: str):
    """Debug endpoint to see all camera attributes."""
    global blink_instance
    
    if not blink_instance or not blink_instance.cameras:
        raise HTTPException(status_code=400, detail="Not logged in")
    
    # Find camera
    camera = None
    for name in blink_instance.cameras.keys():
        if name == camera_name or name.strip() == camera_name.strip():
            camera = blink_instance.cameras[name]
            break
    
    if not camera:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_name}' not found")
    
    # Get all attributes
    attrs = camera.attributes if hasattr(camera, 'attributes') else {}
    
    # Get direct attributes
    direct_attrs = {}
    for attr in dir(camera):
        if not attr.startswith('_'):
            try:
                val = getattr(camera, attr)
                if not callable(val):
                    direct_attrs[attr] = str(val)[:200]  # Truncate long values
            except:
                pass
    
    return {
        "camera_name": camera_name,
        "attributes_dict": attrs,
        "direct_attributes": direct_attrs,
    }


@app.get("/api/camera/{camera_name:path}/config")
async def get_camera_config(camera_name: str):
    """Get camera configuration/settings."""
    global blink_instance
    
    if not blink_instance or not blink_instance.cameras:
        raise HTTPException(status_code=400, detail="Not logged in")
    
    # Handle URL-encoded camera names and trailing spaces
    camera_name_decoded = camera_name.strip()
    
    # Find camera - try exact match first, then try with/without trailing space
    camera = None
    actual_name = None
    for name in blink_instance.cameras.keys():
        if name == camera_name_decoded or name.strip() == camera_name_decoded:
            camera = blink_instance.cameras[name]
            actual_name = name
            break
    
    if not camera:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_name}' not found")
    
    try:
        attrs = camera.attributes if hasattr(camera, 'attributes') else {}
        if not isinstance(attrs, dict):
            attrs = {}
        
        # Build config from local attributes with defaults
        config = {
            "name": actual_name,
            "camera_id": getattr(camera, 'camera_id', None) or attrs.get('id'),
            "network_id": getattr(camera, 'network_id', None) or attrs.get('network_id'),
            
            # Motion settings
            "motion_sensitivity": attrs.get("motion_sensitivity") or attrs.get("sensitivity") or 5,
            "motion_enabled": getattr(camera, 'motion_enabled', None),
            
            # Recording settings
            "clip_length": attrs.get("clip_length") or attrs.get("video_length") or 30,
            "retrigger_time": attrs.get("retrigger_time") or attrs.get("cooldown_time") or 10,
            "video_quality": attrs.get("video_quality") or "standard",
            
            # Night vision
            "night_vision": "auto",
            "illuminator_intensity": attrs.get("illuminator_intensity") or 5,
            
            # Other settings
            "flip_video": attrs.get("flip_video", False),
            "early_notification": attrs.get("early_notification", False),
            "end_clip_early": attrs.get("end_clip_early", False),
            
            # Temperature alert
            "temp_alert": attrs.get("temp_alert", False),
            
            # Volume (for doorbells)
            "volume": attrs.get("volume") or 5,
        }
        
        # Handle night vision - could be bool or string
        if attrs.get("illuminator_enable") is True:
            config["night_vision"] = "on"
        elif attrs.get("illuminator_enable") is False:
            config["night_vision"] = "off"
        elif attrs.get("night_vision"):
            config["night_vision"] = attrs.get("night_vision")
        
        # Try to fetch more detailed config from API if we have IDs
        if config["camera_id"] and config["network_id"]:
            try:
                url = f"{blink_instance.urls.base_url}/network/{config['network_id']}/camera/{config['camera_id']}/config"
                timeout = ClientTimeout(total=5)  # 5 second timeout
                async with ClientSession(timeout=timeout) as session:
                    async with session.get(url, headers=blink_instance.auth.header) as resp:
                        if resp.status == 200:
                            api_config = await resp.json()
                            camera_cfg = api_config.get("camera", [{}])
                            if isinstance(camera_cfg, list) and camera_cfg:
                                camera_cfg = camera_cfg[0]
                            elif not isinstance(camera_cfg, dict):
                                camera_cfg = {}
                            
                            # Update with API values
                            if camera_cfg.get("motion_sensitivity") is not None:
                                config["motion_sensitivity"] = camera_cfg["motion_sensitivity"]
                            if camera_cfg.get("clip_length") is not None:
                                config["clip_length"] = camera_cfg["clip_length"]
                            if camera_cfg.get("retrigger_time") is not None:
                                config["retrigger_time"] = camera_cfg["retrigger_time"]
                            if camera_cfg.get("video_quality") is not None:
                                config["video_quality"] = camera_cfg["video_quality"]
                            if camera_cfg.get("illuminator_enable") is not None:
                                config["night_vision"] = "on" if camera_cfg["illuminator_enable"] else "off"
                            if camera_cfg.get("illuminator_intensity") is not None:
                                config["illuminator_intensity"] = camera_cfg["illuminator_intensity"]
            except Exception as e:
                logger.warning(f"Could not fetch detailed config from Blink API: {e}")
        
        return config
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting camera config for {camera_name}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


class CameraConfigUpdate(BaseModel):
    motion_sensitivity: Optional[int] = None
    clip_length: Optional[int] = None
    retrigger_time: Optional[int] = None
    video_quality: Optional[str] = None
    night_vision: Optional[str] = None
    illuminator_intensity: Optional[int] = None


@app.post("/api/camera/{camera_name:path}/config")
async def update_camera_config(camera_name: str, config: CameraConfigUpdate):
    """Update camera configuration/settings."""
    global blink_instance
    
    if not blink_instance or not blink_instance.cameras:
        raise HTTPException(status_code=400, detail="Not logged in")
    
    # Handle URL-encoded camera names and trailing spaces
    camera_name_decoded = camera_name.strip()
    
    # Find camera - try exact match first, then try with/without trailing space
    camera = None
    actual_name = None
    for name in blink_instance.cameras.keys():
        if name == camera_name_decoded or name.strip() == camera_name_decoded:
            camera = blink_instance.cameras[name]
            actual_name = name
            break
    
    if not camera:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_name}' not found")
    
    try:
        attrs = camera.attributes if hasattr(camera, 'attributes') else {}
        camera_id = getattr(camera, 'camera_id', None) or attrs.get('camera_id') or attrs.get('id')
        network_id = getattr(camera, 'network_id', None) or attrs.get('network_id')
        
        if not camera_id or not network_id:
            raise HTTPException(status_code=400, detail="Camera ID or Network ID not available")
        
        # Build update payload
        update_data = {}
        
        if config.motion_sensitivity is not None:
            if not 1 <= config.motion_sensitivity <= 10:
                raise HTTPException(status_code=400, detail="Motion sensitivity must be 1-10")
            update_data["motion_sensitivity"] = config.motion_sensitivity
        
        if config.clip_length is not None:
            if not 5 <= config.clip_length <= 60:
                raise HTTPException(status_code=400, detail="Clip length must be 5-60 seconds")
            update_data["clip_length"] = config.clip_length
        
        if config.retrigger_time is not None:
            if not 10 <= config.retrigger_time <= 60:
                raise HTTPException(status_code=400, detail="Retrigger time must be 10-60 seconds")
            update_data["retrigger_time"] = config.retrigger_time
        
        if config.video_quality is not None:
            if config.video_quality not in ["saver", "standard", "best"]:
                raise HTTPException(status_code=400, detail="Video quality must be saver, standard, or best")
            update_data["video_quality"] = config.video_quality
        
        if config.night_vision is not None:
            if config.night_vision not in ["on", "off", "auto"]:
                raise HTTPException(status_code=400, detail="Night vision must be on, off, or auto")
            update_data["illuminator_enable"] = config.night_vision != "off"
        
        if config.illuminator_intensity is not None:
            if not 1 <= config.illuminator_intensity <= 10:
                raise HTTPException(status_code=400, detail="Illuminator intensity must be 1-10")
            update_data["illuminator_intensity"] = config.illuminator_intensity
        
        if not update_data:
            return {"success": True, "message": "No changes to apply"}
        
        # Send update to Blink API directly
        logger.info(f"Updating camera {actual_name} with: {update_data}")
        logger.info(f"Camera ID: {camera_id}, Network ID: {network_id}")
        
        # Try the standard endpoint first
        url = f"{blink_instance.urls.base_url}/network/{network_id}/camera/{camera_id}/update"
        timeout = ClientTimeout(total=10)
        
        async with ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=blink_instance.auth.header, json=update_data) as resp:
                response_text = await resp.text()
                logger.info(f"Camera config response: {resp.status} - {response_text[:200]}")
                
                if resp.status == 200:
                    logger.info(f"Camera config updated for {actual_name}: {update_data}")
                    return {"success": True, "updated": update_data}
                elif resp.status == 404:
                    # Camera settings may not be supported for this camera type
                    logger.warning(f"Camera settings endpoint not found for {actual_name} - this camera type may not support settings changes via API")
                    return {
                        "success": False, 
                        "error": "Camera settings are not supported for this camera type via the Blink API. Changes can only be made through the official Blink app.",
                        "attempted": update_data
                    }
                else:
                    logger.error(f"Failed to update camera config: {resp.status} - {response_text}")
                    return {"success": False, "error": f"Blink API returned {resp.status}: {response_text}", "attempted": update_data}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating camera config for {camera_name}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sync-modules")
async def get_sync_modules_details():
    """Get detailed information about all sync modules."""
    global blink_instance
    
    if not blink_instance:
        raise HTTPException(status_code=400, detail="Not logged in")
    
    if not hasattr(blink_instance, 'sync') or not blink_instance.sync:
        return {"sync_modules": []}
    
    try:
        modules = []
        for name, sync in blink_instance.sync.items():
            # Skip pseudo-sync modules for Minis (sync with 1 camera where camera name == sync name)
            if hasattr(sync, 'cameras') and sync.cameras:
                camera_names = list(sync.cameras.keys())
                if len(camera_names) == 1 and camera_names[0] == name:
                    # This is a Mini camera masquerading as a sync module - skip it
                    logger.debug(f"Skipping pseudo-sync for Mini camera: {name}")
                    continue
            
            # Determine online status - check multiple attributes
            online_status = "unknown"
            if hasattr(sync, 'online'):
                online_status = "online" if sync.online else "offline"
            elif hasattr(sync, 'status'):
                status_val = str(getattr(sync, 'status', '')).lower()
                if status_val in ['online', 'true', '1', 'yes']:
                    online_status = "online"
                elif status_val in ['offline', 'false', '0', 'no']:
                    online_status = "offline"
                else:
                    online_status = status_val if status_val else "unknown"
            
            # Get all available attributes
            module_info = {
                "name": name,
                "sync_id": getattr(sync, 'sync_id', None),
                "serial": getattr(sync, 'serial', None),
                "status": online_status,
                "network_id": getattr(sync, 'network_id', None),
                "armed": getattr(sync, 'arm', False) if hasattr(sync, 'arm') else False,
                "wifi_strength": getattr(sync, 'wifi_strength', None),
                "local_storage": getattr(sync, 'local_storage', None),
                "local_storage_enabled": getattr(sync, 'local_storage_enabled', None),
                "local_storage_status": getattr(sync, 'local_storage_status', None),
                "camera_count": len(sync.cameras) if hasattr(sync, 'cameras') else 0,
                "cameras": list(sync.cameras.keys()) if hasattr(sync, 'cameras') else [],
            }
            modules.append(module_info)
        
        return {"sync_modules": modules}
        
    except Exception as e:
        logger.error(f"Error getting sync module details: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/video/{video_id}/stream")
async def stream_video(video_id: str, url: str = None):
    """Proxy video stream from Blink cloud."""
    global blink_instance
    
    if not blink_instance:
        raise HTTPException(status_code=400, detail="Not logged in")
    
    if not url:
        raise HTTPException(status_code=400, detail="Video URL required")
    
    try:
        # Fetch video through authenticated session
        if blink_instance.auth and hasattr(blink_instance.auth, 'session'):
            async with blink_instance.auth.session.get(url) as resp:
                if resp.status == 200:
                    video_data = await resp.read()
                    return Response(
                        content=video_data,
                        media_type="video/mp4"
                    )
        
        raise HTTPException(status_code=404, detail="Could not fetch video")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error streaming video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/login")
async def login(request: LoginRequest):
    """Login to Blink using blinkpy's OAuth v2 flow."""
    global blink_instance, blink_session, remember_me_preference
    
    from blinkpy.auth import BlinkTwoFARequiredError
    
    # Store remember_me preference for use after 2FA
    remember_me_preference = request.remember_me
    
    try:
        logger.info(f"Login attempt for: {request.email} (remember_me: {request.remember_me})")
        
        # Close any existing session
        if blink_session:
            try:
                await blink_session.close()
            except:
                pass
        
        blink_session = ClientSession()
        blink_instance = Blink(session=blink_session)
        
        # Create Auth with no_prompt=True so blinkpy doesn't try to read stdin
        auth = Auth({
            "username": request.email,
            "password": request.password
        }, no_prompt=True, session=blink_session)
        
        blink_instance.auth = auth
        
        logger.info("Starting blinkpy OAuth v2 login flow...")
        logger.info(f"Auth object created, login_url: {getattr(auth, 'login_url', 'N/A')}")
        
        try:
            # This calls the OAuth v2 flow internally
            # If 2FA is needed, it raises BlinkTwoFARequiredError
            logger.info("Calling blink_instance.start()...")
            result = await blink_instance.start()
            logger.info(f"blink_instance.start() returned: {result}")
            logger.info(f"blink_instance.available: {blink_instance.available}")
            logger.info(f"blink_instance.cameras count: {len(blink_instance.cameras) if blink_instance.cameras else 0}")
            
            # Check auth state
            if hasattr(auth, 'login_response'):
                logger.info(f"auth.login_response: {auth.login_response}")
            if hasattr(auth, 'region_id'):
                logger.info(f"auth.region_id: {auth.region_id}")
            if hasattr(blink_instance, 'last_refresh'):
                logger.info(f"blink_instance.last_refresh: {blink_instance.last_refresh}")
            
            # CRITICAL: Check if start() failed (returns False on failure)
            if result is False or not blink_instance.available:
                logger.error("start() returned False or blink not available - login failed")
                # Check if 2FA might be needed but error not raised properly
                if hasattr(auth, 'check_key_required') and auth.check_key_required:
                    logger.info("2FA appears to be required but exception wasn't raised")
                    return {
                        "success": False, 
                        "needs_2fa": True, 
                        "message": "2FA required - check your phone for SMS PIN"
                    }
                raise HTTPException(
                    status_code=401, 
                    detail="Login failed - please check your email and password"
                )
            
            # If we get here, login succeeded without 2FA
            logger.info(f"Login successful without 2FA - cameras: {len(blink_instance.cameras) if blink_instance.cameras else 0}")
            if blink_instance.cameras:
                await save_blink_credentials()
                add_notification("Logged in successfully", "success")
            return {"success": True, "message": "Logged in"}
                
        except BlinkTwoFARequiredError:
            # This is the expected path when 2FA is required
            # The OAuth flow should have triggered SMS at this point
            logger.info("BlinkTwoFARequiredError caught - 2FA required, SMS should be sent")
            return {
                "success": False, 
                "needs_2fa": True, 
                "message": "2FA required - check your phone for SMS PIN"
            }
        except HTTPException:
            raise
        except Exception as start_error:
            error_name = type(start_error).__name__
            error_str = str(start_error).lower()
            logger.info(f"start() raised {error_name}: {start_error}")
            
            # Check if it's an auth error
            if "login" in error_str or "auth" in error_str or "credential" in error_str or "password" in error_str:
                raise HTTPException(status_code=401, detail="Invalid email or password")
            
            # Re-raise other errors
            raise start_error
        
    except HTTPException:
        raise
    except BlinkTwoFARequiredError:
        # Catch it here too in case it bubbles up
        logger.info("BlinkTwoFARequiredError caught at outer level")
        return {"success": False, "needs_2fa": True, "message": "2FA required - check your phone for SMS PIN"}
    except Exception as e:
        error_msg = str(e)
        error_name = type(e).__name__
        logger.error(f"Login error ({error_name}): {error_msg}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=error_msg)


@app.post("/api/verify-pin")
async def verify_pin(request: PinRequest):
    """Verify 2FA PIN using blinkpy's OAuth v2 flow."""
    global blink_instance, blink_session, remember_me_preference
    
    if not blink_instance or not blink_instance.auth:
        raise HTTPException(status_code=400, detail="Not logged in - please login first")
    
    # Use request value if provided, otherwise use global preference from login
    should_save = request.remember_me and remember_me_preference
    
    try:
        logger.info(f"Verifying 2FA PIN using blinkpy OAuth v2... (remember_me: {should_save})")
        
        # Use blinkpy's built-in 2FA completion method
        if hasattr(blink_instance.auth, 'complete_2fa_login'):
            logger.info("Using auth.complete_2fa_login()")
            result = await blink_instance.auth.complete_2fa_login(request.pin)
            logger.info(f"complete_2fa_login result: {result}")
            
            if result:
                # Complete the setup after 2FA
                logger.info("2FA verified, completing setup...")
                
                # CRITICAL FIX: After complete_2fa_login(), we have valid tokens but
                # setup_login_ids() and setup_urls() were never called (they need data
                # from the login response). The cleanest fix is to call start() again -
                # it will use the existing valid tokens and complete without triggering 2FA.
                try:
                    logger.info("Re-running start() to complete initialization...")
                    await blink_instance.start()
                    logger.info(f"start() complete - cameras: {len(blink_instance.cameras) if blink_instance.cameras else 0}")
                except Exception as start_err:
                    # start() might raise an exception but still have succeeded partially
                    # Check if we have what we need
                    logger.warning(f"start() raised exception: {start_err}")
                    
                    # If URLs are still not set, try manual setup
                    if blink_instance.urls is None:
                        logger.info("URLs still None, trying manual setup...")
                        try:
                            # Try to get homescreen which should set up remaining state
                            await blink_instance.setup_post_verify()
                        except Exception as manual_err:
                            logger.error(f"Manual setup also failed: {manual_err}")
                            # If we still have no cameras, this is a real failure
                            if not blink_instance.cameras:
                                raise start_err
                
                # Check if we successfully got cameras
                camera_count = len(blink_instance.cameras) if blink_instance.cameras else 0
                logger.info(f"Final camera count: {camera_count}")
                
                if camera_count > 0 or blink_instance.sync:
                    if should_save:
                        await save_blink_credentials()
                    add_notification(f"2FA verified - found {camera_count} cameras", "success")
                    return {"success": True, "message": "Verified", "cameras": camera_count}
                else:
                    # May have sync modules but no cameras yet - still save credentials
                    if should_save:
                        await save_blink_credentials()
                    add_notification("2FA verified - syncing cameras...", "success")
                    return {"success": True, "message": "Verified", "cameras": 0}
            else:
                raise HTTPException(status_code=400, detail="Invalid PIN - please try again")
        
        # Fallback for older blinkpy versions
        elif hasattr(blink_instance.auth, 'send_auth_key'):
            logger.info("Using legacy send_auth_key()")
            await blink_instance.auth.send_auth_key(blink_instance, request.pin)
            await blink_instance.setup_post_verify()
            
            if blink_instance.cameras:
                if should_save:
                    await save_blink_credentials()
                return {"success": True, "message": "Verified"}
            else:
                if should_save:
                    await save_blink_credentials()
                return {"success": True, "message": "Verified"}
        else:
            raise HTTPException(status_code=500, detail="No supported 2FA verification method found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PIN verification error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Verification failed: {str(e)}")


@app.post("/api/resend-pin")
async def resend_pin():
    """Request a new 2FA PIN by re-triggering the blinkpy OAuth v2 flow."""
    global blink_instance, blink_session
    
    if not blink_instance or not blink_instance.auth:
        raise HTTPException(status_code=400, detail="Not logged in - please login first")
    
    try:
        logger.info("Attempting to resend 2FA PIN by re-triggering OAuth v2 flow...")
        
        # Get stored credentials
        login_data = blink_instance.auth.login_attributes
        
        if not login_data or 'username' not in login_data:
            raise HTTPException(status_code=400, detail="No login data available - please login again")
        
        username = login_data.get('username')
        password = login_data.get('password')
        
        if not username or not password:
            raise HTTPException(status_code=400, detail="Credentials not available - please login again")
        
        logger.info(f"Re-triggering OAuth v2 flow for {username}")
        
        # Re-start the OAuth flow which should trigger a new SMS
        try:
            await blink_instance.auth.startup()
        except Exception as e:
            error_name = type(e).__name__
            # BlinkTwoFARequiredError means the flow triggered 2FA again
            if "BlinkTwoFARequiredError" in error_name or "2fa" in error_name.lower():
                logger.info("OAuth v2 flow re-triggered 2FA - new SMS should be sent")
                add_notification("New PIN sent to your phone via SMS", "success")
                return {"success": True, "message": "New PIN sent to your phone via SMS"}
            # Re-raise if it's a different error
            raise
        
        # If we get here without 2FA error, something unusual happened
        return {"success": True, "message": "PIN request processed - check your phone"}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Resend PIN error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Failed to resend PIN: {str(e)}")


@app.post("/api/login/saved")
async def login_saved():
    """Login using saved credentials (with auth token)."""
    global blink_instance, blink_session
    
    if not CREDENTIALS_PATH.exists():
        raise HTTPException(status_code=400, detail="No saved credentials found")
    
    try:
        logger.info(f"Attempting login with saved credentials from {CREDENTIALS_PATH}")
        
        # Close any existing session
        if blink_session:
            try:
                await blink_session.close()
            except:
                pass
        
        blink_session = ClientSession()
        
        # Load credentials using blinkpy's json_load helper
        from blinkpy.helpers.util import json_load
        creds = await json_load(str(CREDENTIALS_PATH))
        
        if not creds:
            raise HTTPException(status_code=400, detail="Credentials file is empty or invalid")
        
        logger.info(f"Loaded credentials with keys: {list(creds.keys())}")
        
        # Check if we have a token (indicates previously successful auth)
        has_token = 'token' in creds or 'access_token' in creds
        logger.info(f"Has auth token: {has_token}")
        
        # Create Blink instance with session
        blink_instance = Blink(session=blink_session)
        
        # Create Auth with credentials AND session
        auth = Auth(creds, no_prompt=True, session=blink_session)
        blink_instance.auth = auth
        
        two_fa_required = False
        try:
            logger.info("Calling blink_instance.start()...")
            await blink_instance.start()
            logger.info(f"start() completed. Cameras: {len(blink_instance.cameras) if blink_instance.cameras else 0}")
        except Exception as start_error:
            error_name = type(start_error).__name__
            error_str = str(start_error).lower()
            logger.info(f"start() raised {error_name}: {start_error}")
            
            if "2fa" in error_name.lower() or "2fa" in error_str or "key required" in error_str:
                logger.info("2FA required even with saved credentials (token may have expired)")
                two_fa_required = True
            else:
                raise start_error
        
        if two_fa_required:
            return {"success": False, "needs_2fa": True, "message": "Token expired - 2FA required"}
        
        # Even if start() succeeded, try a refresh to ensure data is loaded
        if not blink_instance.cameras or len(blink_instance.cameras) == 0:
            logger.info("No cameras after start(), trying explicit refresh...")
            try:
                await blink_instance.refresh(force=True)
                logger.info(f"After refresh: {len(blink_instance.cameras) if blink_instance.cameras else 0} cameras")
            except Exception as refresh_error:
                logger.error(f"Refresh failed: {refresh_error}")
        
        if blink_instance.cameras and len(blink_instance.cameras) > 0:
            # Re-save to update any refreshed tokens
            await save_blink_credentials()
            add_notification("Logged in with saved credentials", "success")
            logger.info(f"Successfully logged in with saved credentials, {len(blink_instance.cameras)} cameras found")
            return {"success": True}
        
        # Check sync modules as alternative indicator of success
        if hasattr(blink_instance, 'sync') and blink_instance.sync and len(blink_instance.sync) > 0:
            logger.info(f"No cameras but {len(blink_instance.sync)} sync modules found - login succeeded")
            await save_blink_credentials()
            add_notification("Logged in (sync modules found, checking cameras...)", "success")
            return {"success": True}
        
        logger.warning("No cameras or sync modules found after login with saved credentials")
        return {"success": False, "needs_2fa": True, "message": "No cameras found - token may be expired"}
        
    except Exception as e:
        logger.error(f"Saved login error: {e}")
        logger.error(traceback.format_exc())
        if "2fa" in str(e).lower():
            return {"success": False, "needs_2fa": True}
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/logout")
async def logout():
    """Logout from Blink - clears session but keeps saved credentials for quick re-login."""
    global blink_instance, blink_session
    
    print("Logging out - clearing session state (keeping saved credentials)...")
    
    # Try to properly close blinkpy if it has a method
    if blink_instance:
        try:
            if hasattr(blink_instance, 'logout'):
                await blink_instance.logout()
            elif hasattr(blink_instance, 'close'):
                await blink_instance.close()
        except Exception as e:
            print(f"Error during blink logout: {e}")
    
    # Close the aiohttp session
    if blink_session:
        try:
            await blink_session.close()
        except Exception as e:
            print(f"Error closing session: {e}")
        blink_session = None
    
    # Clear the instance
    blink_instance = None
    
    # Keep credentials for quick re-login
    # Use /api/forget-credentials to clear them
    
    add_notification("Logged out", "info")
    return {"success": True}


@app.post("/api/forget-credentials")
async def forget_credentials():
    """Delete saved credentials to force fresh 2FA on next login."""
    if CREDENTIALS_PATH.exists():
        try:
            CREDENTIALS_PATH.unlink()
            logger.info("Deleted saved credentials")
            add_notification("Saved credentials deleted", "info")
            return {"success": True, "message": "Credentials deleted - you'll need 2FA on next login"}
        except Exception as e:
            logger.error(f"Error deleting credentials: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to delete credentials: {e}")
    else:
        return {"success": True, "message": "No saved credentials found"}


# =============================================================================
# API Routes - Settings
# =============================================================================

@app.get("/api/settings")
async def get_settings():
    """Get current settings."""
    settings = load_settings()
    # Don't send password back to client
    settings["email_password"] = "***" if settings.get("email_password") else ""
    return {"settings": settings}


@app.post("/api/settings")
async def update_settings(request: SettingsRequest):
    """Update settings."""
    download_path = Path(request.download_dir)
    
    try:
        download_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot create directory: {str(e)}")
    
    # Validate time format
    try:
        hour, minute = map(int, request.scheduler_time.split(":"))
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError()
    except:
        raise HTTPException(status_code=400, detail="Invalid time format. Use HH:MM")
    
    # Load existing settings to preserve password if not changed
    existing = load_settings()
    
    settings = {
        "download_dir": str(download_path.absolute()),
        "scheduler_enabled": request.scheduler_enabled,
        "scheduler_time": request.scheduler_time,
        "scheduler_days": request.scheduler_days,
        "email_enabled": request.email_enabled,
        "email_smtp_server": request.email_smtp_server,
        "email_smtp_port": request.email_smtp_port,
        "email_username": request.email_username,
        "email_password": request.email_password if request.email_password != "***" else existing.get("email_password", ""),
        "email_from": request.email_from,
        "email_to": request.email_to,
        "email_on_success": request.email_on_success,
        "email_on_error": request.email_on_error,
        "retention_enabled": request.retention_enabled,
        "retention_days": request.retention_days,
        "disk_warning_enabled": request.disk_warning_enabled,
        "disk_warning_gb": request.disk_warning_gb,
        "timezone": request.timezone,
    }
    save_settings(settings)
    
    # Start or stop scheduler based on settings
    if request.scheduler_enabled:
        start_scheduler()
    else:
        stop_scheduler()
    
    add_notification("Settings saved", "success")
    
    return {"success": True, "settings": settings}


@app.post("/api/test-email")
async def test_email():
    """Send a test email."""
    logger.info("Test email requested")
    settings = load_settings()
    
    if not settings.get("email_enabled"):
        logger.warning("Email not enabled in settings")
        raise HTTPException(status_code=400, detail="Email notifications not enabled - enable it in settings first")
    
    logger.info(f"Email settings: server={settings.get('email_smtp_server')}, user={settings.get('email_username')}, to={settings.get('email_to')}")
    
    try:
        send_email_notification(
            "Test Email",
            "This is a test email from Blink Hub.\n\nIf you received this, your email settings are configured correctly!"
        )
        add_notification("Test email sent", "success")
        logger.info("Test email sent successfully")
        return {"success": True, "message": "Test email sent"}
    except Exception as e:
        logger.error(f"Test email failed: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to send email: {str(e)}")


@app.get("/api/timezones")
async def get_timezones():
    """Get list of available timezones for the dropdown."""
    # Check if tzdata is available
    tzdata_available = is_valid_timezone("America/Los_Angeles")
    
    return {
        "timezones": [{"value": tz[0], "label": tz[1]} for tz in TIMEZONE_OPTIONS],
        "current": load_settings().get("timezone", "auto"),
        "tzdata_available": tzdata_available
    }


@app.post("/api/rename-videos-to-local-time")
async def rename_videos_to_local_time(background_tasks: BackgroundTasks, request: Request):
    """Rename existing videos from UTC to local timezone.
    
    Only renames videos where filename_timezone is NULL (legacy UTC filenames).
    """
    try:
        body = await request.json()
        target_timezone = body.get("timezone", "America/Los_Angeles")
    except:
        target_timezone = "America/Los_Angeles"
    
    # Validate timezone
    if not is_valid_timezone(target_timezone):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid timezone: {target_timezone}. If timezone support is not working, please restart the application to install the tzdata package."
        )
    
    # Get videos that need renaming (filename_timezone IS NULL)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, camera_name, created_at, file_path, file_size, thumbnail_path, duration
        FROM downloaded_videos 
        WHERE filename_timezone IS NULL AND created_at IS NOT NULL AND created_at != ''
    """)
    videos = cursor.fetchall()
    conn.close()
    
    if not videos:
        return {"success": True, "message": "No legacy videos to rename", "renamed": 0, "errors": 0}
    
    renamed_count = 0
    error_count = 0
    
    for video in videos:
        video_id, camera_name, created_at, file_path, file_size, thumbnail_path, duration = video
        
        try:
            old_path = Path(file_path)
            if not old_path.exists():
                logger.warning(f"File not found: {file_path}")
                error_count += 1
                continue
            
            # Parse UTC timestamp
            try:
                dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            except:
                dt = datetime.strptime(created_at[:19], "%Y-%m-%dT%H:%M:%S")
                dt = dt.replace(tzinfo=timezone.utc)
            
            # Convert to target timezone
            local_dt = convert_utc_to_timezone(dt, target_timezone)
            new_filename = local_dt.strftime("%Y%m%d_%H%M%S") + ".mp4"
            new_path = old_path.parent / new_filename
            
            # Skip if new filename is the same or target exists
            if old_path.name == new_filename:
                # Just update the database to mark as converted
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE downloaded_videos SET filename_timezone = ? WHERE id = ?",
                    (target_timezone, video_id)
                )
                conn.commit()
                conn.close()
                renamed_count += 1
                continue
            
            if new_path.exists():
                logger.warning(f"Target file already exists: {new_path}")
                error_count += 1
                continue
            
            # Rename the file
            old_path.rename(new_path)
            
            # Update database
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE downloaded_videos SET file_path = ?, filename_timezone = ? WHERE id = ?",
                (str(new_path), target_timezone, video_id)
            )
            conn.commit()
            conn.close()
            
            renamed_count += 1
            logger.info(f"Renamed: {old_path.name} -> {new_filename}")
            
        except Exception as e:
            logger.error(f"Error renaming video {video_id}: {e}")
            error_count += 1
    
    message = f"Renamed {renamed_count} videos to {target_timezone}"
    if error_count > 0:
        message += f" ({error_count} errors)"
    
    add_notification(message, "success" if error_count == 0 else "warning")
    
    return {
        "success": True,
        "message": message,
        "renamed": renamed_count,
        "errors": error_count
    }


@app.get("/api/legacy-video-count")
async def get_legacy_video_count():
    """Get count of videos with legacy UTC filenames that can be renamed."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*) FROM downloaded_videos 
        WHERE filename_timezone IS NULL AND created_at IS NOT NULL AND created_at != ''
    """)
    count = cursor.fetchone()[0]
    conn.close()
    
    return {"count": count}


# =============================================================================
# API Routes - Downloads
# =============================================================================

@app.post("/api/download")
async def start_download(request: DownloadRequest, background_tasks: BackgroundTasks):
    """Start downloading videos."""
    global blink_instance, download_status
    
    if not blink_instance:
        raise HTTPException(status_code=400, detail="Not logged in")
    
    if not blink_instance.cameras:
        raise HTTPException(status_code=400, detail="No cameras found")
    
    if download_status["running"]:
        raise HTTPException(status_code=400, detail="Download already in progress")
    
    background_tasks.add_task(
        download_videos_task,
        since_days=request.since_days,
        cameras=request.cameras if request.cameras else None,
        browser_timezone=request.browser_timezone
    )
    
    add_notification("Download started", "info")
    return {"success": True, "message": "Download started"}


@app.get("/api/download/status")
async def get_download_status():
    """Get current download status."""
    global download_status
    return download_status


@app.post("/api/sync")
async def sync_new_videos(request: SyncRequest, background_tasks: BackgroundTasks):
    """Sync only new videos since last download."""
    global blink_instance, download_status
    
    if not blink_instance:
        raise HTTPException(status_code=400, detail="Not logged in")
    
    if not blink_instance.cameras:
        raise HTTPException(status_code=400, detail="No cameras found")
    
    if download_status["running"]:
        raise HTTPException(status_code=400, detail="Download already in progress")
    
    background_tasks.add_task(
        download_videos_task,
        since_days=1,
        cameras=request.cameras if request.cameras else None,
        browser_timezone=request.browser_timezone
    )
    
    add_notification("Sync started", "info")
    return {"success": True, "message": "Sync started"}


class DownloadMissingRequest(BaseModel):
    browser_timezone: str = None


@app.post("/api/download-missing")
async def download_missing_videos(request: DownloadMissingRequest, background_tasks: BackgroundTasks):
    """Download only videos that are in the cloud but not downloaded locally."""
    global blink_instance, download_status
    
    if not blink_instance:
        raise HTTPException(status_code=400, detail="Not logged in")
    
    if not blink_instance.cameras:
        raise HTTPException(status_code=400, detail="No cameras found")
    
    if download_status["running"]:
        raise HTTPException(status_code=400, detail="Download already in progress")
    
    # Get the list of missing videos
    try:
        from datetime import datetime, timedelta
        
        # Get videos from cloud (last 30 days)
        cloud_videos = []
        since = datetime.now() - timedelta(days=30)
        since_str = since.strftime("%Y-%m-%dT%H:%M:%S+00:00")
        
        url = f"{blink_instance.urls.base_url}/api/v1/accounts/{blink_instance.account_id}/media/changed?since={since_str}&page=1"
        
        async with ClientSession() as session:
            headers = blink_instance.auth.header
            async with session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    media = data.get('media', [])
                    for item in media:
                        cloud_videos.append({
                            'id': item.get('id'),
                            'camera': item.get('device_name', item.get('camera', 'Unknown')),
                            'created_at': item.get('created_at', ''),
                            'media': item.get('media', ''),
                            'thumbnail': item.get('thumbnail', '')
                        })
        
        # Get local video IDs
        local_video_ids = set()
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM downloaded_videos")
        for row in cursor.fetchall():
            local_video_ids.add(str(row['id']))
        conn.close()
        
        # Find missing videos
        missing_videos = [cv for cv in cloud_videos if cv['id'] and str(cv['id']) not in local_video_ids]
        
        if not missing_videos:
            return {"success": True, "message": "No missing videos to download", "missing_count": 0}
        
        logger.info(f"Found {len(missing_videos)} missing videos to download")
        
        # Start background download task for missing videos only
        background_tasks.add_task(
            download_missing_videos_task,
            missing_videos=missing_videos,
            browser_timezone=request.browser_timezone
        )
        
        add_notification(f"Downloading {len(missing_videos)} missing videos", "info")
        return {"success": True, "message": f"Starting download of {len(missing_videos)} missing videos", "missing_count": len(missing_videos)}
        
    except Exception as e:
        logger.error(f"Error starting missing video download: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/open-folder")
async def open_download_folder():
    """Open the download folder in the system file explorer."""
    import subprocess
    
    downloads_dir = get_download_dir()
    downloads_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if sys.platform == 'win32':
            # Try to open explorer - this works when running interactively
            # but fails silently when running as a service (Session 0)
            try:
                # Use start command which sometimes works better
                subprocess.Popen(f'explorer "{str(downloads_dir)}"', shell=True)
            except:
                pass
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', str(downloads_dir)])
        else:
            subprocess.Popen(['xdg-open', str(downloads_dir)])
        
        # Always return the path so UI can show it
        return {"success": True, "path": str(downloads_dir)}
    except Exception as e:
        # Still return path even on error
        return {"success": True, "path": str(downloads_dir)}


@app.post("/api/browse-folder")
async def browse_folder():
    """Open folder browser dialog and return selected path."""
    # Note: This won't work when running as a Windows service (Session 0)
    # because services can't show UI dialogs. User must type path manually.
    import subprocess
    
    try:
        if sys.platform == 'win32':
            # Try PowerShell dialog - only works in interactive session
            ps_script = '''
$shell = New-Object -ComObject Shell.Application
$folder = $shell.BrowseForFolder(0, "Select download folder for Blink videos", 0, 0)
if ($folder) {
    $folder.Self.Path
}
'''
            result = subprocess.run(
                ['powershell', '-WindowStyle', 'Hidden', '-Command', ps_script],
                capture_output=True,
                text=True,
                timeout=120
            )
            selected_path = result.stdout.strip()
            logger.info(f"Folder browser result: '{selected_path}', stderr: {result.stderr}")
            
            if selected_path and selected_path != '' and not selected_path.startswith('Exception'):
                return {"success": True, "path": selected_path}
            else:
                # Running as service - dialog can't be shown
                return {"success": False, "message": "Running as service - type the folder path manually (e.g. E:\\BlinkVideos)"}
        else:
            return {"success": False, "message": "Folder browser only supported on Windows"}
    except subprocess.TimeoutExpired:
        return {"success": False, "message": "Dialog timed out"}
    except Exception as e:
        logger.error(f"Browse folder error: {e}")
        return {"success": False, "message": "Running as service - type the folder path manually"}


@app.post("/api/retention/run")
async def run_retention_now():
    """Manually run retention policy."""
    result = await run_retention_policy()
    return {"success": True, "deleted": result["deleted"]}


# =============================================================================
# Download Task
# =============================================================================

async def download_videos_task(since_days: int, cameras: Optional[list[str]] = None, scheduled: bool = False, browser_timezone: str = None):
    """Background task to download videos with proper duplicate detection.
    
    Args:
        since_days: Number of days back to download
        cameras: Optional list of camera names to filter
        scheduled: Whether this is a scheduled run
        browser_timezone: Timezone detected from browser (for auto mode)
    """
    global blink_instance, download_status
    
    download_status["running"] = True
    download_status["total"] = 0
    download_status["downloaded"] = 0
    download_status["skipped"] = 0
    download_status["bytes_downloaded"] = 0
    download_status["error"] = None
    download_status["current_file"] = ""
    
    errors = []
    
    # Determine timezone to use for filenames
    settings = load_settings()
    tz_setting = settings.get("timezone", "auto")
    if tz_setting == "auto":
        # Use browser-detected timezone, fall back to UTC
        filename_tz = browser_timezone if browser_timezone else "UTC"
    else:
        filename_tz = tz_setting
    
    logger.info(f"Using timezone '{filename_tz}' for video filenames")
    
    downloads_dir = get_download_dir()
    downloads_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        await blink_instance.refresh(force=True)
        
        since_date = datetime.now() - timedelta(days=since_days)
        since_str = since_date.strftime("%Y/%m/%d %H:%M")
        
        download_status["current_camera"] = "Fetching video list..."
        
        print(f"Cameras: {list(blink_instance.cameras.keys())}")
        print(f"Sync modules: {list(blink_instance.sync.keys())}")
        
        all_videos = []
        
        # Method: Request media list directly via API
        try:
            download_status["current_camera"] = "Fetching media list from API..."
            
            for sync_name, sync_module in blink_instance.sync.items():
                url = f"{blink_instance.urls.base_url}/api/v1/accounts/{blink_instance.account_id}/media/changed?since={since_str}&page=1"
                
                print(f"Requesting: {url}")
                
                async with ClientSession() as session:
                    async with session.get(url, headers=blink_instance.auth.header) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            media_list = data.get('media', [])
                            print(f"API returned {len(media_list)} media items")
                            
                            for item in media_list:
                                cam_name = item.get('device_name', 'Unknown')
                                if cameras and cam_name not in cameras:
                                    continue
                                all_videos.append({
                                    'camera': cam_name,
                                    'video': item
                                })
                        else:
                            print(f"API request failed: {resp.status}")
                
                break
                
        except Exception as e:
            print(f"Error fetching media list: {e}")
            print(traceback.format_exc())
        
        download_status["total"] = len(all_videos)
        print(f"Found {len(all_videos)} total videos to process")
        
        # Process each video
        for item in all_videos:
            video = item.get('video', {})
            camera_name = item.get('camera') or video.get('device_name') or video.get('camera_name') or 'Unknown'
            
            download_status["current_camera"] = camera_name
            
            video_id = str(video.get('id') or video.get('clip_id') or video.get('media_id') or '')
            
            if not video_id:
                created_at = video.get('created_at', '')
                video_id = f"{camera_name}_{created_at}".replace(':', '-').replace(' ', '_')
            
            if is_video_downloaded(video_id):
                download_status["skipped"] += 1
                continue
            
            # Check date range
            try:
                created_at = video.get('created_at', '')
                if created_at:
                    try:
                        video_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    except:
                        video_time = datetime.strptime(created_at[:19], "%Y-%m-%dT%H:%M:%S")
                    
                    if video_time.tzinfo:
                        video_time = video_time.replace(tzinfo=None)
                    
                    if video_time < since_date:
                        download_status["skipped"] += 1
                        continue
            except Exception as e:
                print(f"Error parsing date {created_at}: {e}")
            
            # Create camera directory
            safe_camera_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in camera_name).strip()
            camera_dir = downloads_dir / safe_camera_name
            camera_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename (using configured timezone)
            try:
                created_at = video.get('created_at', '')
                if created_at:
                    try:
                        dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    except:
                        dt = datetime.strptime(created_at[:19], "%Y-%m-%dT%H:%M:%S")
                        dt = dt.replace(tzinfo=timezone.utc)
                    
                    # Convert UTC to configured timezone for filename
                    local_dt = convert_utc_to_timezone(dt, filename_tz)
                    filename = local_dt.strftime("%Y%m%d_%H%M%S") + ".mp4"
                else:
                    filename = f"{video_id}.mp4"
            except Exception as e:
                logger.warning(f"Filename generation error: {e}")
                filename = f"{video_id}.mp4"
            
            file_path = camera_dir / filename
            download_status["current_file"] = filename
            
            if file_path.exists():
                file_size = file_path.stat().st_size
                thumb_path = generate_thumbnail(str(file_path), THUMBNAILS_DIR)
                # File already exists - we don't know what timezone was used originally
                mark_video_downloaded(video_id, camera_name, video.get('created_at', ''), str(file_path), file_size, thumb_path, filename_timezone=None)
                download_status["skipped"] += 1
                continue
            
            # Download the video
            try:
                media_url = video.get('media') or video.get('clip_url') or video.get('url')
                
                if media_url:
                    if not media_url.startswith('http'):
                        # Need blink_instance.urls to construct full URL
                        if not blink_instance or not blink_instance.urls:
                            errors.append(f"{camera_name}/{filename}: Session expired - please refresh")
                            download_status["skipped"] += 1
                            continue
                        media_url = f"{blink_instance.urls.base_url}{media_url}"
                    
                    # Also check we have auth headers
                    if not blink_instance or not blink_instance.auth or not blink_instance.auth.header:
                        errors.append(f"{camera_name}/{filename}: Not authenticated - please login again")
                        download_status["skipped"] += 1
                        continue
                    
                    print(f"Downloading: {camera_name} -> {filename}")
                    
                    async with ClientSession() as session:
                        async with session.get(media_url, headers=blink_instance.auth.header) as resp:
                            if resp.status == 200:
                                content = await resp.read()
                                file_size = len(content)
                                
                                with open(file_path, 'wb') as f:
                                    f.write(content)
                                
                                # Generate thumbnail
                                thumb_path = generate_thumbnail(str(file_path), THUMBNAILS_DIR)
                                
                                mark_video_downloaded(video_id, camera_name, video.get('created_at', ''), str(file_path), file_size, thumb_path, filename_timezone=filename_tz)
                                download_status["downloaded"] += 1
                                download_status["bytes_downloaded"] += file_size
                                print(f"  Saved: {file_path} ({format_bytes(file_size)})")
                            else:
                                print(f"  Failed: HTTP {resp.status}")
                                errors.append(f"{camera_name}/{filename}: HTTP {resp.status}")
                                download_status["skipped"] += 1
                else:
                    print(f"  No media URL for video {video_id}")
                    download_status["skipped"] += 1
                    
            except Exception as e:
                print(f"Error downloading video {video_id}: {e}")
                errors.append(f"{camera_name}/{filename}: {str(e)}")
                download_status["skipped"] += 1
            
            await asyncio.sleep(DELAY_BETWEEN_DOWNLOADS)
        
        # Fallback to blinkpy's built-in method if no videos found
        if download_status["total"] == 0:
            download_status["current_camera"] = "Using fallback downloader..."
            print("No videos found via API, trying built-in download method...")
            
            try:
                temp_dir = downloads_dir / "_temp_download"
                temp_dir.mkdir(parents=True, exist_ok=True)
                
                print(f"Downloading to temp: {temp_dir}")
                
                await blink_instance.download_videos(
                    str(temp_dir),
                    since=since_str,
                    delay=DELAY_BETWEEN_DOWNLOADS,
                    camera='all'
                )
                
                print(f"Download complete, organizing files...")
                
                # Process flat files
                for f in temp_dir.glob("*.mp4"):
                    filename = f.stem
                    parts = filename.rsplit('_', 2)
                    
                    if len(parts) >= 3:
                        camera_name = parts[0]
                        new_filename = f"{parts[1]}_{parts[2]}.mp4"
                    else:
                        camera_name = "Unknown"
                        new_filename = f.name
                    
                    safe_camera_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in camera_name).strip()
                    camera_dir = downloads_dir / safe_camera_name
                    camera_dir.mkdir(parents=True, exist_ok=True)
                    
                    new_path = camera_dir / new_filename
                    video_id = f"{camera_name}_{f.stem}"
                    
                    if not is_video_downloaded(video_id) and not new_path.exists():
                        file_size = f.stat().st_size
                        shutil.move(str(f), str(new_path))
                        thumb_path = generate_thumbnail(str(new_path), THUMBNAILS_DIR)
                        # Fallback method - filename from blinkpy, unknown timezone
                        mark_video_downloaded(video_id, camera_name, "", str(new_path), file_size, thumb_path, filename_timezone=None)
                        download_status["downloaded"] += 1
                        download_status["bytes_downloaded"] += file_size
                        print(f"  Moved: {f.name} -> {camera_name}/{new_filename}")
                    else:
                        f.unlink()
                        download_status["skipped"] += 1
                
                # Process subfolders
                for subdir in temp_dir.iterdir():
                    if subdir.is_dir():
                        camera_name = subdir.name
                        safe_camera_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in camera_name).strip()
                        camera_dir = downloads_dir / safe_camera_name
                        camera_dir.mkdir(parents=True, exist_ok=True)
                        
                        for f in subdir.glob("*.mp4"):
                            video_id = f"{camera_name}_{f.stem}"
                            new_path = camera_dir / f.name
                            
                            if not is_video_downloaded(video_id) and not new_path.exists():
                                file_size = f.stat().st_size
                                shutil.move(str(f), str(new_path))
                                thumb_path = generate_thumbnail(str(new_path), THUMBNAILS_DIR)
                                # Fallback method - filename from blinkpy, unknown timezone
                                mark_video_downloaded(video_id, camera_name, "", str(new_path), file_size, thumb_path, filename_timezone=None)
                                download_status["downloaded"] += 1
                                download_status["bytes_downloaded"] += file_size
                            else:
                                f.unlink()
                                download_status["skipped"] += 1
                
                # Cleanup
                try:
                    shutil.rmtree(str(temp_dir))
                except:
                    pass
                        
            except Exception as e:
                print(f"Fallback download error: {e}")
                print(traceback.format_exc())
                errors.append(f"Fallback download: {str(e)}")
        
        status = "success" if not errors else f"completed with {len(errors)} errors"
        log_download_run(download_status["downloaded"], download_status["skipped"], download_status["bytes_downloaded"], status)
        download_status["last_run"] = datetime.now().isoformat()
        
        # Send notification
        if download_status["downloaded"] > 0 or errors:
            notify_download_complete(
                download_status["downloaded"],
                download_status["skipped"],
                download_status["bytes_downloaded"],
                errors if errors else None
            )
            add_notification(
                f"Download complete: {download_status['downloaded']} new videos ({format_bytes(download_status['bytes_downloaded'])})",
                "success" if not errors else "warning"
            )
        
    except Exception as e:
        print(f"Download task error: {e}")
        print(traceback.format_exc())
        download_status["error"] = str(e)
        log_download_run(download_status["downloaded"], download_status["skipped"], download_status["bytes_downloaded"], f"error: {str(e)}")
        add_notification(f"Download error: {str(e)}", "error")
    
    finally:
        download_status["running"] = False
        download_status["current_camera"] = ""
        download_status["current_file"] = ""


async def download_missing_videos_task(missing_videos: list, browser_timezone: str = None):
    """Background task to download only missing videos.
    
    Args:
        missing_videos: List of video dicts from cloud parity check
        browser_timezone: Timezone detected from browser (for auto mode)
    """
    global blink_instance, download_status
    
    download_status["running"] = True
    download_status["total"] = len(missing_videos)
    download_status["downloaded"] = 0
    download_status["skipped"] = 0
    download_status["bytes_downloaded"] = 0
    download_status["error"] = None
    download_status["current_file"] = ""
    
    errors = []
    
    # Determine timezone to use for filenames
    settings = load_settings()
    tz_setting = settings.get("timezone", "auto")
    if tz_setting == "auto":
        filename_tz = browser_timezone if browser_timezone else "UTC"
    else:
        filename_tz = tz_setting
    
    logger.info(f"Downloading {len(missing_videos)} missing videos using timezone '{filename_tz}'")
    
    downloads_dir = get_download_dir()
    downloads_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        for video in missing_videos:
            video_id = str(video.get('id', ''))
            camera_name = video.get('camera', 'Unknown')
            created_at = video.get('created_at', '')
            media_url = video.get('media', '')
            
            download_status["current_camera"] = camera_name
            
            if not video_id or not media_url:
                download_status["skipped"] += 1
                continue
            
            # Double-check it's not already downloaded
            if is_video_downloaded(video_id):
                download_status["skipped"] += 1
                continue
            
            # Create camera directory
            safe_camera_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in camera_name).strip()
            camera_dir = downloads_dir / safe_camera_name
            camera_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            try:
                if created_at:
                    try:
                        dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    except:
                        dt = datetime.strptime(created_at[:19], "%Y-%m-%dT%H:%M:%S")
                        dt = dt.replace(tzinfo=timezone.utc)
                    
                    local_dt = convert_utc_to_timezone(dt, filename_tz)
                    filename = local_dt.strftime("%Y%m%d_%H%M%S") + ".mp4"
                else:
                    filename = f"{video_id}.mp4"
            except Exception as e:
                logger.warning(f"Filename generation error: {e}")
                filename = f"{video_id}.mp4"
            
            file_path = camera_dir / filename
            download_status["current_file"] = filename
            
            # Skip if file exists
            if file_path.exists():
                file_size = file_path.stat().st_size
                thumb_path = generate_thumbnail(str(file_path), THUMBNAILS_DIR)
                mark_video_downloaded(video_id, camera_name, created_at, str(file_path), file_size, thumb_path, filename_timezone=None)
                download_status["skipped"] += 1
                continue
            
            # Download the video
            try:
                if not media_url.startswith('http'):
                    if not blink_instance or not blink_instance.urls:
                        errors.append(f"{camera_name}/{filename}: Session expired")
                        download_status["skipped"] += 1
                        continue
                    media_url = f"{blink_instance.urls.base_url}{media_url}"
                
                if not blink_instance or not blink_instance.auth or not blink_instance.auth.header:
                    errors.append(f"{camera_name}/{filename}: Not authenticated")
                    download_status["skipped"] += 1
                    continue
                
                logger.info(f"Downloading missing: {camera_name} -> {filename}")
                
                async with ClientSession() as session:
                    async with session.get(media_url, headers=blink_instance.auth.header) as resp:
                        if resp.status == 200:
                            content = await resp.read()
                            file_size = len(content)
                            
                            with open(file_path, 'wb') as f:
                                f.write(content)
                            
                            thumb_path = generate_thumbnail(str(file_path), THUMBNAILS_DIR)
                            mark_video_downloaded(video_id, camera_name, created_at, str(file_path), file_size, thumb_path, filename_timezone=filename_tz)
                            download_status["downloaded"] += 1
                            download_status["bytes_downloaded"] += file_size
                            logger.info(f"  Saved: {file_path} ({format_bytes(file_size)})")
                        else:
                            logger.warning(f"  Failed: HTTP {resp.status}")
                            errors.append(f"{camera_name}/{filename}: HTTP {resp.status}")
                            download_status["skipped"] += 1
                            
            except Exception as e:
                logger.error(f"Error downloading video {video_id}: {e}")
                errors.append(f"{camera_name}/{filename}: {str(e)}")
                download_status["skipped"] += 1
            
            await asyncio.sleep(DELAY_BETWEEN_DOWNLOADS)
        
        status = "success" if not errors else f"completed with {len(errors)} errors"
        log_download_run(download_status["downloaded"], download_status["skipped"], download_status["bytes_downloaded"], status)
        download_status["last_run"] = datetime.now().isoformat()
        
        if download_status["downloaded"] > 0 or errors:
            notify_download_complete(
                download_status["downloaded"],
                download_status["skipped"],
                download_status["bytes_downloaded"],
                errors if errors else None
            )
            add_notification(
                f"Missing videos download: {download_status['downloaded']} new ({format_bytes(download_status['bytes_downloaded'])})",
                "success" if not errors else "warning"
            )
        
    except Exception as e:
        logger.error(f"Download missing task error: {e}")
        logger.error(traceback.format_exc())
        download_status["error"] = str(e)
        log_download_run(download_status["downloaded"], download_status["skipped"], download_status["bytes_downloaded"], f"error: {str(e)}")
        add_notification(f"Download error: {str(e)}", "error")
    
    finally:
        download_status["running"] = False
        download_status["current_camera"] = ""
        download_status["current_file"] = ""


# =============================================================================
# Main Entry Point
# =============================================================================

def run_server():
    """Run the uvicorn server."""
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="warning")

if __name__ == "__main__":
    import threading
    
    # Check if we should run in desktop mode (pywebview) or browser mode
    desktop_mode = False
    try:
        import webview
        desktop_mode = True
    except ImportError:
        desktop_mode = False
    
    if desktop_mode and (getattr(sys, 'frozen', False) or '--desktop' in sys.argv):
        # Run as native desktop app with pywebview
        print("Starting Blink Downloader (Desktop Mode)...")
        
        # Start server in background thread
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Give server time to start
        import time
        time.sleep(1)
        
        # Create native window
        window = webview.create_window(
            'Blink Hub',
            'http://127.0.0.1:8080',
            width=1200,
            height=800,
            min_size=(900, 600),
            background_color='#0f0f1a'
        )
        webview.start()
    else:
        # Run as browser-based server
        import uvicorn
        import webbrowser
        
        def open_browser():
            import time
            time.sleep(1.5)
            webbrowser.open("http://127.0.0.1:8080")
        
        if getattr(sys, 'frozen', False):
            threading.Thread(target=open_browser, daemon=True).start()
        
        print("=" * 50)
        print("Blink Hub v2.0")
        print("=" * 50)
        print("Server running at: http://127.0.0.1:8080")
        print("=" * 50)
        
        uvicorn.run(app, host="127.0.0.1", port=8080)
