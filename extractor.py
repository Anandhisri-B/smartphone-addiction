"""
extractor.py  (v7 - Multi-Brand Android + Stock Android + iPhone Support)
=================================================================
Supports:
  1. Samsung Digital Wellbeing  → Category-level pages (Social, Dashboard, Video...)
  2. Stock Android / Nokia / OnePlus / Realme / Oppo / Vivo / Pixel
       → "App activity details" dashboard with app list
       → Individual app detail pages (AppName + "X minutes" / "X hrs, Y mins")
  3. iPhone / iOS Screen Time
       → Day/Week overview page (total + Social / Entertainment / Creativity)
       → Most Used app list page (WhatsApp, Phone, YouTube, etc.)
       → Individual app detail pages (app name + today usage + daily average)
       → Pickups page (total pickups = unlocks, first pickup time)
       → Notifications summary page (total notifications count)

Brand is auto-detected from OCR text — no manual input needed.

Nokia/Stock Android screenshot types:
  TYPE A: App activity details (LIST PAGE)  ← MOST IMPORTANT
    Nav bar: "App activity details"
    Big time: "6 hrs, 22 mins"
    App rows: "YouTube" then "2 hrs, 18 mins" (two-line format)

  TYPE B: Individual app detail page
    App name big: "Chrome" / "WhatsApp" / "Instagram"
    Buttons: "Screen time ▼  Hourly ▼"
    Big time: "26 minutes" / "1 hr, 21 mins"
    Date: "Mon, 2 Feb"

iPhone / iOS Screen Time screenshot types:
  TYPE A: Day overview (MOST IMPORTANT)
    Header: "iPhone"  toggle: "Week / Day"
    Big time: "4h 25m"
    Category labels at bottom: "Social  1h 14m", "Entertainment  25m", "Creativity  20m"
  TYPE B: Most Used app list
    Header: "Most Used   Show Categories"
    App rows: "WhatsApp  1h 11m", "Phone  48m", "YouTube  25m"
    Footer: "Show More"
  TYPE C: Individual app detail
    Nav: "< YouTube"  then "Screen Time"
    Big time: "25m"
    "Daily Average   18m"
    "Category   Entertainment"
  TYPE D: Pickups page
    Header: "Pickups"
    Big number: "193"
    "First Pickup   2:49 AM"   "Total Pickups   193"
  TYPE E: Notifications page
    Header: "Notifications"
    Big number: "14"
    App row: "Phone   14"

Duplicate removal (two layers):
  1. Text hash   — exact same OCR content
  2. pHash       — visually near-identical images (hamming distance <= 10)

Which screenshots to upload:
  iPhone:
    MUST HAVE:  Day overview page (total + category breakdown)
    RECOMMENDED: Most Used list (individual app breakdown)
    OPTIONAL:   Individual app detail pages for apps NOT in Most Used list
    OPTIONAL:   Pickups page (maps to unlocks)
    OPTIONAL:   Notifications page
  Nokia/Stock Android:
    MUST HAVE:  App activity details main page (gives total + all top apps)
    OPTIONAL:   Individual app pages for apps NOT visible in main list
    NOT NEEDED: Duplicate main pages, weekly if daily exists
  Samsung:
    MUST HAVE:  Dashboard page (total + notifications + unlocks)
    MUST HAVE:  Social / Productivity / Video category pages
    NOT NEEDED: Duplicate pages, same category daily + weekly (daily preferred)
"""

import re
import os
import cv2
import numpy as np
import pytesseract
from pathlib import Path


# ── Windows: Auto-set Tesseract path ─────────────────────────────────────────
def _set_tesseract_path():
    import platform
    if platform.system() != 'Windows':
        return
    candidates = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        os.path.join(os.environ.get('LOCALAPPDATA', ''),
                     r'Programs\Tesseract-OCR\tesseract.exe'),
        os.path.join(os.environ.get('USERPROFILE', ''),
                     r'AppData\Local\Programs\Tesseract-OCR\tesseract.exe'),
    ]
    for path in candidates:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return

_set_tesseract_path()


# ═══════════════════════════════════════════════════════════════════════════════
#  APP → CATEGORY MAPPING
# ═══════════════════════════════════════════════════════════════════════════════
APP_CATEGORY = {
    # Social / messaging
    'whatsapp': 'social', 'instagram': 'social', 'facebook': 'social',
    'messenger': 'social', 'telegram': 'social', 'snapchat': 'social',
    'twitter': 'social', 'x': 'social', 'tiktok': 'social',
    'linkedin': 'social', 'discord': 'social', 'signal': 'social',
    'wechat': 'social', 'viber': 'social', 'hike': 'social',
    'sharechat': 'social', 'moj': 'social', 'josh': 'social',
    'phone': 'social', 'truecaller': 'social', 'contacts': 'social',
    'dialer': 'social', 'messages': 'social', 'sms': 'social',
    'kik': 'social', 'skype': 'social', 'hangouts': 'social',

    # Video / streaming
    'youtube': 'video', 'netflix': 'video', 'hotstar': 'video',
    'disney': 'video', 'amazon prime': 'video', 'prime video': 'video',
    'mx player': 'video', 'vlc': 'video', 'zee5': 'video',
    'sony liv': 'video', 'sonyliv': 'video', 'voot': 'video',
    'jio cinema': 'video', 'jiocinema': 'video', 'mxplayer': 'video',
    'dailymotion': 'video', 'twitch': 'video',

    # Gaming
    'pubg': 'gaming', 'bgmi': 'gaming', 'free fire': 'gaming',
    'freefire': 'gaming', 'clash of clans': 'gaming', 'clash royale': 'gaming',
    'candy crush': 'gaming', 'ludo king': 'gaming', 'among us': 'gaming',
    'minecraft': 'gaming', 'roblox': 'gaming', 'mobile legends': 'gaming',
    'pokemon': 'gaming', 'game': 'gaming', 'games': 'gaming',
    'play games': 'gaming',

    # Productivity / tools
    'chrome': 'productivity', 'google': 'productivity', 'gmail': 'productivity',
    'drive': 'productivity', 'docs': 'productivity', 'sheets': 'productivity',
    'slides': 'productivity', 'maps': 'productivity', 'calendar': 'productivity',
    'keep': 'productivity', 'tasks': 'productivity', 'chatgpt': 'productivity',
    'copilot': 'productivity', 'edge': 'productivity', 'firefox': 'productivity',
    'brave': 'productivity', 'opera': 'productivity', 'calculator': 'productivity',
    'clock': 'productivity', 'settings': 'productivity', 'files': 'productivity',
    'gallery': 'productivity', 'photos': 'productivity', 'camera': 'productivity',
    'notes': 'productivity', 'microsoft': 'productivity', 'outlook': 'productivity',
    'teams': 'productivity', 'zoom': 'productivity', 'meet': 'productivity',
    'where is my train': 'productivity', 'train': 'productivity',
    'paytm': 'productivity', 'gpay': 'productivity', 'phonepe': 'productivity',
    'amazon': 'productivity', 'flipkart': 'productivity',
    'swiggy': 'productivity', 'zomato': 'productivity',
    'ola': 'productivity', 'uber': 'productivity',
}

# Samsung nav bar → feature category
SAMSUNG_CATEGORY_MAP = {
    'social': 'social',
    'productivity and finance': 'productivity',
    'productivity': 'productivity',
    'video': 'video',
    'gaming': 'gaming',
    'dashboard': 'dashboard',
    'entertainment': 'video',
    'communication': 'social',
    'utilities': 'productivity',
    'travel': 'productivity',
    'health and fitness': 'others',
    'health': 'others',
    'shopping': 'others',
    'food and drink': 'others',
    'music and audio': 'others',
    'news and magazines': 'others',
    'books and reference': 'others',
    'education': 'productivity',
}


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def parse_mins(s):
    """
    Convert any time string to total minutes.
    Handles all formats including OCR noise around values:
      Stock Android: '26 minutes', '1 hr, 21 mins', '2 hrs, 18 mins', '1 minute'
                     'SS 31 minutes @O'  (OCR icon noise around value)
                     '6 1 minute g'      (OCR icon prefix)
                     '* 13 minutes'      (bullet OCR noise)
      Samsung:       '2h 29m', '5h14m', '18m', '1h 6m / day'
    """
    s = str(s).strip().lower()

    # "X hrs, Y mins" / "X hr, Y min" (Stock Android)
    m = re.search(r'(\d+)\s*hr[s]?\s*[,]?\s*(\d+)\s*min', s)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))

    # "X hrs" / "X hr" only
    m = re.search(r'(\d+)\s*hr[s]?\b', s)
    if m:
        return int(m.group(1)) * 60

    # "X minutes" / "X mins" / "X minute" — robust: allows noise around the number
    m = re.search(r'\b(\d+)\s*min(?:ute[s]?)?(?!\w)', s)
    if m:
        v = int(m.group(1))
        if 1 <= v <= 720:
            return v

    # Samsung compact: "2h 29m", "5h14m"
    m = re.search(r'(\d+)\s*h\s*(\d+)\s*m', s)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))

    m = re.search(r'(\d+)\s*h\b', s)
    if m:
        return int(m.group(1)) * 60

    m = re.search(r'(\d+)\s*m(?![a-z])', s)
    if m:
        v = int(m.group(1))
        if 1 <= v <= 720:
            return v

    return 0


def ocr_image(path):
    """Preprocess and OCR a single image."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    if gray2.mean() < 128:
        gray2 = cv2.bitwise_not(gray2)
    _, thresh = cv2.threshold(gray2, 200, 255, cv2.THRESH_BINARY)
    return pytesseract.image_to_string(thresh, config='--psm 6 --oem 3')


def phash(img_path, hash_size=8):
    """Perceptual hash for near-duplicate visual detection."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    resized = cv2.resize(img, (hash_size * 4, hash_size * 4))
    dct_result = cv2.dct(np.float32(resized))
    dct_low = dct_result[:hash_size, :hash_size]
    med = np.median(dct_low)
    return ''.join(['1' if v > med else '0' for v in dct_low.flatten()])


def hamming(h1, h2):
    """Hamming distance between two binary hash strings."""
    return sum(c1 != c2 for c1, c2 in zip(h1, h2))


def app_to_category(app_name):
    """Map an app name to a feature category."""
    name = app_name.lower().strip()
    if name in APP_CATEGORY:
        return APP_CATEGORY[name]
    for key, cat in APP_CATEGORY.items():
        if key in name or name in key:
            return cat
    return 'others'


def detect_daily(lines):
    """Return True if screenshot shows a daily (not weekly) view."""
    for line in lines[:10]:
        ll = line.lower()
        # iOS Screen Time: explicit "yesterday" or "today" label
        if 'yesterday' in ll or re.search(r'\btoday\b', ll):
            return True
        # iOS "Day" tab is selected (appears before "Week" in OCR or toggled)
        if re.search(r'\bweek\b.*\bday\b', ll) or re.search(r'\bday\b', ll):
            # "Week  Day" layout — Day is right-side (selected) when viewing daily
            if re.search(r'\bweek\b', ll):
                return True
        # Daily indicator: specific weekday name visible (Android)
        if re.search(r'\b(mon|tue|wed|thu|fri|sat|sun)\b', ll):
            # But NOT the weekly bar chart label row (has multiple days)
            day_count = len(re.findall(r'\b(mon|tue|wed|thu|fri|sat|sun)\b', ll))
            if day_count == 1:
                return True
        if 'weekly view' in ll:
            return True   # "Weekly View" button = currently on daily page
        if 'daily view' in ll:
            return False  # "Daily View" button = currently on weekly page
    return True  # default: assume daily


# ═══════════════════════════════════════════════════════════════════════════════
#  BRAND + PAGE TYPE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_ocr(text):
    """
    Fix common OCR merging errors from Stock Android screenshots:
      'Appactivity' → 'app activity'
      'Screentime'  → 'screen time'
      'AppActivity' → 'app activity'
    """
    t = text.lower()
    t = re.sub(r'appactivity', 'app activity', t)
    t = re.sub(r'screentime', 'screen time', t)
    t = re.sub(r'app\s+activity\s+details', 'app activity details', t)
    return t


def detect_brand(lines):
    """
    Auto-detect phone brand and page type.

    Returns: (brand, page_type)
      brand:     'samsung' | 'stock_android' | 'iphone'
      page_type: 'samsung_dashboard' | 'samsung_category' |
                 'stock_list' | 'stock_app_detail' |
                 'iphone_day_summary' | 'iphone_most_used' |
                 'iphone_app_detail' | 'iphone_pickups' |
                 'iphone_notifications' | 'unknown'

    Stock Android covers: Nokia, OnePlus, Realme, Oppo, Vivo, Pixel,
                          Motorola, Xiaomi (with stock-like UI)
    iPhone covers: iOS Screen Time pages
    """
    full_text = _normalize_ocr(' '.join(lines[:12]))
    full_lower = full_text.lower()

    # ── iPhone / iOS Screen Time markers ────────────────────────────────────

    # Pickups page: "Pickups" heading anywhere in the page
    # "First Pickup" / "Total Pickups" may be beyond the first 12 lines, so
    # scan ALL lines for confirmation signals
    full_all = _normalize_ocr(' '.join(lines)).lower()
    if 'pickups' in full_all and (
        'first pickup' in full_all or 'total pickups' in full_all
    ):
        return 'iphone', 'iphone_pickups'

    # Notifications page: "Notifications" heading in first lines
    if 'notifications' in full_lower:
        return 'iphone', 'iphone_notifications'

    # Day/Week summary: "Screen Time" label + "Show Today" / category labels (Social, Entertainment, Creativity)
    if 'screen time' in full_lower and (
        'show today' in full_lower or 'show categories' in full_lower or
        ('social' in full_lower and 'entertainment' in full_lower)
    ):
        return 'iphone', 'iphone_day_summary'

    # Most Used list: "Most Used" heading + "Show Categories" + app list with time
    if 'most used' in full_lower and 'show categories' in full_lower:
        return 'iphone', 'iphone_most_used'

    # Individual app detail: "Screen Time" section + known iOS app info fields
    if 'screen time' in full_lower and (
        'daily average' in full_lower or
        ('category' in full_lower and 'age rating' in full_lower)
    ):
        return 'iphone', 'iphone_app_detail'

    # ── Stock Android markers ────────────────────────────────────────────────
    # Main list page: nav bar says "App activity details" (may be OCR-merged)
    if 'app activity details' in full_text or 'appactivity' in ' '.join(lines[:3]).lower():
        return 'stock_android', 'stock_list'

    # Individual app page: has "Screen time" + "Hourly" pill buttons
    if ('screen time' in full_text or 'screentime' in full_text) and 'hourly' in full_text:
        return 'stock_android', 'stock_app_detail'

    # ── Samsung markers ──────────────────────────────────────────────────────
    for line in lines[:5]:
        ll = re.sub(r'^[<\s€]+', '', line.lower().strip()).strip()
        if ll == 'dashboard':
            return 'samsung', 'samsung_dashboard'
        for key in SAMSUNG_CATEGORY_MAP:
            if ll == key or ll.startswith(key):
                return 'samsung', 'samsung_category'

    return 'unknown', 'unknown'


# ═══════════════════════════════════════════════════════════════════════════════
#  STOCK ANDROID PARSERS
# ═══════════════════════════════════════════════════════════════════════════════

# Lines to skip when parsing app lists
_STOCK_SKIP = re.compile(
    r'^(\d+h|[<>]|screen\s?time|hourly|app\s+activity|show\s+sites|'
    r'sun\s+mon|mon\s+tue|\d+:\d+|settings|app\s+timer|no\s+timer|'
    r'manage\s+notif|app\s+used\s+in)',
    re.IGNORECASE
)


def _clean_app_name(name):
    """
    Strip OCR icon/noise characters from start/end of app names.
    Examples:
      'gy youtube 2'  → 'youtube'
      '© Chrome R'    → 'chrome'
      '& 31 minutes'  → '' (invalid — is a time line)
      '-») Photos'    → 'photos'
      'SS 31 minutes' → '' (skip — this is a time line with noise)
    """
    # Remove trailing OCR noise (single chars, icons, digits)
    name = re.sub(r'[\s\d©@\.\•\*\-\»\(\)R\|l\[\]]{0,3}\s*$', '', name).strip()
    # Remove leading OCR noise (non-letter chars and short noise tokens)
    name = re.sub(r'^[^a-zA-Z]+', '', name).strip()
    # Remove short noise tokens at start (2 chars or fewer before real name)
    name = re.sub(r'^[a-z]{1,2}\s+(?=[a-zA-Z])', '', name).strip()
    return name.lower()


def parse_stock_list_page(lines, is_daily):
    """
    Parse "App activity details" main list page (Stock Android).

    Returns: {'total_min': int, 'apps': {app_name_lower: minutes}}

    Two-line format (most common):
      "YouTube"
      "2 hrs, 18 mins"

    One-line format (OCR sometimes merges):
      "YouTube  2 hrs, 18 mins"
    """
    result = {'total_min': 0, 'apps': {}}

    # ── Total screen time ─────────────────────────────────────────────────
    for line in lines[1:7]:
        ll = line.lower().strip()
        if _STOCK_SKIP.match(ll):
            continue
        if re.match(r'^(mon|tue|wed|thu|fri|sat|sun),?\s+\d+', ll):
            continue
        mins = parse_mins(ll)
        if mins >= 30:
            result['total_min'] = mins
            break

    # ── Find where app list starts (after weekly bar chart) ───────────────
    list_start = 0
    for i, line in enumerate(lines):
        ll = line.lower()
        # Row with multiple day abbreviations = bar chart label row
        if re.search(r'\bsun\b.*\bmon\b|\bmon\b.*\btue\b', ll):
            list_start = i + 2   # skip chart + nav row
            break
        # Navigation row like "< Mon, 2 Feb >"
        if re.match(r'^<\s+(mon|tue|wed|thu|fri|sat|sun)', ll):
            list_start = i + 1
            break

    app_lines = [l.strip() for l in lines[list_start:]
                 if l.strip() and not _STOCK_SKIP.match(l.strip().lower())]

    # ── Parse app entries ─────────────────────────────────────────────────
    last_app_name = None  # track last seen app name for orphan time lines

    i = 0
    while i < len(app_lines):
        line = app_lines[i]
        ll = line.lower()

        # Skip date lines like "Mon, 2 Feb"
        if re.match(r'^(mon|tue|wed|thu|fri|sat|sun),?\s+\d+', ll):
            i += 1
            continue

        # Skip "Show sites that you visit" button
        if 'show sites' in ll:
            i += 1
            continue

        # One-line format: "AppName   X hrs, Y mins"
        one = re.match(
            r'^(.+?)\s{2,}(\d+\s*hr[s]?,?\s*\d+\s*min[s]?|\d+\s*hr[s]?'
            r'|\d+\s*min(?:ute[s]?)?)\s*$',
            line, re.IGNORECASE
        )
        if one:
            app_name = _clean_app_name(one.group(1))
            mins = parse_mins(one.group(2))
            if mins > 0 and len(app_name) >= 2:
                result['apps'][app_name] = mins
                last_app_name = app_name
            i += 1
            continue

        # Check if current line is a time line (possibly with OCR noise)
        cur_mins = parse_mins(ll)
        if cur_mins > 0 and re.search(r'\bmin(?:ute[s]?)?|\bhr[s]?\b', ll):
            # Standalone time line — belongs to last seen app name
            if last_app_name:
                result['apps'][last_app_name] = cur_mins
            i += 1
            continue

        # Two-line format: app name on this line, time on next line
        if i + 1 < len(app_lines):
            next_line = app_lines[i + 1]
            next_mins = parse_mins(next_line.lower())
            if next_mins > 0 and re.search(r'\bmin(?:ute[s]?)?|\bhr[s]?\b', next_line.lower()):
                app_name = _clean_app_name(line)
                if re.search(r'[a-zA-Z]', app_name) and len(app_name) >= 2:
                    result['apps'][app_name] = next_mins
                    last_app_name = app_name
                i += 2
                continue

        # This line is an app name without a parseable time following it
        potential_name = _clean_app_name(line)
        if re.search(r'[a-zA-Z]{2,}', potential_name):
            last_app_name = potential_name

        i += 1

    return result


def parse_stock_app_detail(lines, is_daily):
    """
    Parse individual app detail page (Stock Android).
    Returns: (app_name_str, minutes_int)

    Layout:
      ← [back arrow]
      AppName          ← big heading
      Screen time ▼  Hourly ▼
      26 minutes       ← usage
      Mon, 2 Feb
    """
    app_name = None
    minutes = 0

    # App name: second or third non-empty non-system line
    for line in lines[:8]:
        ll = line.lower().strip()
        if any(kw in ll for kw in ['screen time', 'hourly', 'app activity',
                                    'settings', 'app timer', 'manage']):
            continue
        if re.match(r'^\d+:\d+', ll):  # clock time
            continue
        if re.match(r'^[€<\s]+$', ll):  # back arrow only
            continue
        if re.search(r'[a-zA-Z]{2,}', line) and not re.match(r'^\d+', line.strip()):
            app_name = re.sub(r'^[^a-zA-Z]+', '', line).strip()
            if app_name:
                break

    # Usage time: first valid time in lines 2-9
    for line in lines[2:10]:
        ll = line.lower().strip()
        if any(kw in ll for kw in ['screen time', 'hourly', 'settings',
                                    'app timer', 'manage', 'background']):
            continue
        if re.match(r'^(mon|tue|wed|thu|fri|sat|sun)', ll):
            continue
        mins = parse_mins(ll)
        if mins > 0:
            minutes = mins
            break

    return app_name, minutes


# ═══════════════════════════════════════════════════════════════════════════════
#  iPHONE / iOS SCREEN TIME PARSERS
# ═══════════════════════════════════════════════════════════════════════════════

# iOS Screen Time category → internal category
IPHONE_CATEGORY_MAP = {
    'social':        'social',
    'entertainment': 'video',
    'creativity':    'productivity',
    'productivity':  'productivity',
    'utilities':     'productivity',
    'games':         'gaming',
    'gaming':        'gaming',
    'education':     'productivity',
    'health':        'others',
    'finance':       'others',
    'shopping':      'others',
    'food & drink':  'others',
    'music':         'others',
    'news':          'others',
    'reference':     'others',
    'travel':        'others',
    'weather':       'others',
}

# iOS app page category field → internal category
IPHONE_APP_CATEGORY_MAP = {
    'social networking': 'social',
    'social':            'social',
    'entertainment':     'video',
    'photo & video':     'video',
    'productivity':      'productivity',
    'utilities':         'productivity',
    'games':             'gaming',
    'education':         'productivity',
    'health & fitness':  'others',
    'finance':           'others',
    'shopping':          'others',
    'food & drink':      'others',
    'music':             'others',
    'news':              'others',
    'travel':            'others',
    'weather':           'others',
    'reference':         'others',
    'business':          'productivity',
    'developer tools':   'productivity',
    'lifestyle':         'others',
    'sports':            'others',
}


def _iphone_parse_time(s):
    """
    Parse iOS Screen Time time strings to minutes.
    Handles:
      '25m'       → 25
      '1h 11m'    → 71
      '4h 25m'    → 265
      '48m'       → 48
      '2m'        → 2
      '24s'       → 0  (seconds → 0 min, below threshold)
    """
    s = s.strip().lower()
    # "Xh Ym"
    m = re.search(r'(\d+)\s*h\s*(\d+)\s*m', s)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    # "Xh"
    m = re.search(r'(\d+)\s*h\b', s)
    if m:
        return int(m.group(1)) * 60
    # "Xm"
    m = re.search(r'(\d+)\s*m\b', s)
    if m:
        v = int(m.group(1))
        if 1 <= v <= 720:
            return v
    # seconds → treat as < 1 min
    if re.search(r'\d+\s*s\b', s):
        return 0
    return 0


def parse_iphone_day_summary(lines):
    """
    Parse iOS Screen Time day/week overview page.

    Real OCR output (Image 6):
      '< iPhone'
      'Week Day'
      'Screen Time Show Today'
      'Yesterday, 9 March'
      '4h 25m'
      ...
      'Social Entertainment Creativity'   ← all category labels on ONE line
      '1h 14m 25m 20m'                   ← all times on the NEXT line

    Also handles the inline format (if OCR keeps them together):
      'Social   1h 14m'
    """
    result = {'total_min': 0, 'social': 0, 'productivity': 0,
              'video': 0, 'gaming': 0, 'others': 0,
              'notifications': -1, 'unlocks': -1}

    # Total: first large time value (>= 30 min) in first 15 lines
    for line in lines[:15]:
        ll = line.lower().strip()
        if any(kw in ll for kw in ['daily average', 'show today', 'show categories',
                                    'screen time', 'week', 'day', 'yesterday', 'today']):
            continue
        mins = _iphone_parse_time(ll)
        if mins >= 30:
            result['total_min'] = mins
            break

    # Category parsing — two strategies:
    #   A) Inline: "Social   1h 14m"  (category + time on same line)
    #   B) Split:  "Social Entertainment Creativity" then "1h 14m 25m 20m"
    for i, line in enumerate(lines):
        ll = line.lower().strip()

        # Strategy A: inline — category keyword + time on same line
        for cat_key, cat_val in IPHONE_CATEGORY_MAP.items():
            m = re.search(
                rf'\b{re.escape(cat_key)}\b.*?(\d+\s*h\s*\d+\s*m|\d+\s*h\b|\d+\s*m\b)',
                ll, re.IGNORECASE
            )
            if m:
                mins = _iphone_parse_time(m.group(1))
                if mins > 0:
                    result[cat_val] = max(result.get(cat_val, 0), mins)

        # Strategy B: label row with multiple categories, times on the next line
        # Detect a line that contains ≥2 category keywords but NO time values
        cat_keys_found = [k for k in IPHONE_CATEGORY_MAP if re.search(rf'\b{re.escape(k)}\b', ll)]
        has_time = bool(re.search(r'\d+\s*[hm]\b', ll))
        if len(cat_keys_found) >= 2 and not has_time and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            # Extract all time tokens from the next line in order
            times = re.findall(r'\d+\s*h\s*\d+\s*m|\d+\s*h\b|\d+\s*m\b', next_line, re.IGNORECASE)
            for j, cat_key in enumerate(cat_keys_found):
                if j < len(times):
                    mins = _iphone_parse_time(times[j])
                    cat_val = IPHONE_CATEGORY_MAP[cat_key]
                    if mins > 0:
                        result[cat_val] = max(result.get(cat_val, 0), mins)

    return result


def parse_iphone_most_used(lines):
    """
    Parse iOS "Most Used" app list page.

    Example lines:
      Most Used    Show Categories
      WhatsApp
      1h 11m
      Phone
      48m
      YouTube
      25m
      Google Photos
      19m
      Messages
      2m
      Camera
      1m
      Screenshots
      24s
      Show More

    Returns: {'app_name': minutes, ...}
    """
    apps = {}
    # Find where the app list starts (after "Most Used" / "Show Categories" header)
    start = 0
    for i, line in enumerate(lines):
        if 'most used' in line.lower():
            start = i + 1
            break

    i = start
    while i < len(lines):
        line = lines[i].strip()
        ll = line.lower()

        # Skip UI chrome lines
        if any(kw in ll for kw in ['most used', 'show categories', 'show more',
                                    'screen time', 'add limit']):
            i += 1
            continue
        # Skip clock / status bar lines
        if re.match(r'^\d+:\d+', ll):
            i += 1
            continue

        # Try two-line format: app name line, then time line
        if i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            next_mins = _iphone_parse_time(next_line.lower())
            time_pattern = re.search(r'\d+\s*(h|m|s)\b', next_line.lower())
            if next_mins >= 0 and time_pattern and re.search(r'[a-zA-Z]{2,}', line):
                app_name = line.lower().strip()
                # Skip if the "app name" is itself a time string
                if not re.match(r'^\d+\s*(h|m|s)\b', app_name):
                    apps[app_name] = next_mins
                    i += 2
                    continue

        # Try one-line format: "AppName   1h 11m"
        one = re.match(
            r'^(.+?)\s{2,}(\d+\s*h\s*\d+\s*m|\d+\s*h|\d+\s*m|\d+\s*s)\s*$',
            line, re.IGNORECASE
        )
        if one:
            app_name = one.group(1).lower().strip()
            mins = _iphone_parse_time(one.group(2))
            if re.search(r'[a-zA-Z]{2,}', app_name):
                apps[app_name] = mins
            i += 1
            continue

        i += 1

    return apps


def parse_iphone_app_detail(lines):
    """
    Parse individual iOS app Screen Time detail page.

    Example layout:
      ← YouTube
      Screen Time
      25m
      [bar chart]
      Daily Average   18m
      Updated today at 9:24 PM
      Info
      YouTube
      Category    Entertainment
      Age Rating  13+
      Developer   Google
      Limits
      Add Limit

    Returns: (app_name_str, today_minutes, daily_avg_minutes, ios_category_str)
    """
    app_name = None
    today_mins = 0
    daily_avg = 0
    ios_category = None

    # App name: in the nav bar — "< YouTube" → "YouTube"
    # Skip: status bar (clock like "9:28 all 5G"), screen time label, iphone, settings
    for line in lines[:6]:
        ll = line.strip()
        cleaned = re.sub(r'^[<\s]+', '', ll).strip()
        # Skip status bar lines: start with clock time "9:28" or contain signal/battery noise
        if re.match(r'^\d+:\d+', cleaned):
            continue
        if not re.search(r'[a-zA-Z]{2,}', cleaned):
            continue
        if any(kw in cleaned.lower() for kw in ['screen time', 'iphone', 'settings',
                                                  'all 5g', 'all 4g', '5g', '4g']):
            continue
        app_name = cleaned
        break

    # Today's usage: first time value that is NOT the daily average
    daily_avg_seen = False
    for i, line in enumerate(lines):
        ll = line.lower().strip()
        if 'daily average' in ll:
            daily_avg_seen = True
            # Try to grab the value from the same line or next
            m = re.search(r'(\d+\s*h\s*\d+\s*m|\d+\s*h|\d+\s*m)', ll)
            if m:
                daily_avg = _iphone_parse_time(m.group(1))
            elif i + 1 < len(lines):
                daily_avg = _iphone_parse_time(lines[i + 1].strip())
            continue
        if 'category' in ll:
            # "Category    Entertainment"
            m = re.search(r'category\s+(.+)', ll)
            if m:
                ios_category = m.group(1).strip()
            continue
        if any(kw in ll for kw in ['screen time', 'updated', 'info', 'limits',
                                    'add limit', 'age rating', 'developer', 'daily average']):
            continue
        if re.match(r'^\d+:\d+', ll):
            continue
        mins = _iphone_parse_time(ll)
        if mins > 0 and not daily_avg_seen and today_mins == 0:
            today_mins = mins

    return app_name, today_mins, daily_avg, ios_category


def _parse_pickup_hour(hour_raw, ampm):
    """
    Convert OCR'd hour + am/pm to 24h int, handling OCR noise.
    '42:49 AM' → raw=42, but real hour is 2 (▲ glyph OCR'd as '4' prefix).
    Clamp: if raw > 12 take last digit(s) mod 12.
    """
    h = hour_raw
    if h > 12:
        # Strip leading OCR-noise digit(s): e.g. 42 → 2, 112 → 2
        h = int(str(hour_raw)[-1])  # last digit
        if h > 12:
            h = hour_raw % 12
    if ampm == 'pm' and h != 12:
        h += 12
    if ampm == 'am' and h == 12:
        h = 0
    return h


def parse_iphone_pickups(lines):
    """
    Parse iOS Pickups page.

    Real OCR output:
      '< iPhone'
      'Screenshots'           ← noise from scroll position
      'Pickups'
      'Yesterday, 9 March'
      'a ns'                  ← '193' completely garbled by OCR
      '0'                     ← chart axis
      'S M TT W TT FS'        ← chart day labels
      '50'                    ← chart Y-axis
      '25'                    ← chart Y-axis
      '12AM GAM 12PM 6PM'
      'First Pickup Total Pickups'   ← both labels on one line
      '42:49 AM 193'                 ← '▲2:49 AM  193' — '4' is OCR noise from ▲

    Strategy:
      1. Primary: grab the last number from the line after the label row
      2. Hour: clamp raw OCR hour if > 12 (▲ glyph adds a leading digit)
    """
    total_pickups = 0
    first_pickup_hour = None

    for i, line in enumerate(lines):
        ll = line.lower().strip()

        # "First Pickup Total Pickups" labels — value is on the NEXT line
        if 'first pickup' in ll or 'total pickups' in ll:
            # Try current line last number
            m = re.search(r'(\d{2,4})\s*$', line.strip())
            if m:
                v = int(m.group(1))
                if 10 <= v <= 2000:
                    total_pickups = max(total_pickups, v)

            # Check next line: "42:49 AM 193" or "▲2:49 AM  193"
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                tm = re.search(r'(\d{1,2}):(\d{2})\s*(am|pm)', next_line.lower())
                if tm:
                    first_pickup_hour = _parse_pickup_hour(int(tm.group(1)), tm.group(3))
                nm = re.search(r'(\d{2,4})\s*$', next_line)
                if nm:
                    v = int(nm.group(1))
                    if 10 <= v <= 2000:
                        total_pickups = max(total_pickups, v)
            continue

        # Catch "42:49 AM 193" style lines directly (time + count inline)
        tm_cnt = re.search(r'(\d{1,2}):(\d{2})\s*(am|pm)\s+(\d{2,4})\s*$', ll)
        if tm_cnt:
            first_pickup_hour = _parse_pickup_hour(int(tm_cnt.group(1)), tm_cnt.group(3))
            v = int(tm_cnt.group(4))
            if 10 <= v <= 2000:
                total_pickups = max(total_pickups, v)

    return total_pickups, first_pickup_hour
    """
    Parse iOS Pickups page.

    Real OCR output:
      '< iPhone'
      'Screenshots'           ← noise from scroll position
      'Pickups'
      'Yesterday, 9 March'
      'a ns'                  ← '193' completely garbled by OCR
      '0'                     ← chart axis
      'S M TT W TT FS'        ← chart day labels
      '50'                    ← chart Y-axis
      '25'                    ← chart Y-axis
      '12AM GAM 12PM 6PM'
      'First Pickup Total Pickups'   ← both labels on one line
      '42:49 AM 193'                 ← '▲2:49 AM  193' — count is last number

    Strategy:
      1. Primary: grab the last number from the line that contains
         'first pickup' OR 'total pickups' context (next line after label row)
      2. Fallback: largest plausible standalone integer on the page
    """
    total_pickups = 0
    first_pickup_hour = None

    # Scan all lines for the label+value patterns
    for i, line in enumerate(lines):
        ll = line.lower().strip()

        # "First Pickup Total Pickups" labels — value is on the NEXT line
        if 'first pickup' in ll or 'total pickups' in ll:
            # Value line may be current line (inline) or next line
            # Try current line last number first
            m = re.search(r'(\d{2,4})\s*$', line.strip())
            if m:
                v = int(m.group(1))
                if 10 <= v <= 2000:
                    total_pickups = max(total_pickups, v)

            # Also check next line: "42:49 AM 193" or "▲2:49 AM  193"
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                # Extract first-pickup time (HH:MM AM/PM anywhere in next line)
                tm = re.search(r'(\d{1,2}):(\d{2})\s*(am|pm)', next_line.lower())
                if tm:
                    hour = int(tm.group(1))
                    ampm = tm.group(3)
                    if ampm == 'pm' and hour != 12:
                        hour += 12
                    if ampm == 'am' and hour == 12:
                        hour = 0
                    first_pickup_hour = hour
                # Total pickups = last number on the next line
                nm = re.search(r'(\d{2,4})\s*$', next_line)
                if nm:
                    v = int(nm.group(1))
                    if 10 <= v <= 2000:
                        total_pickups = max(total_pickups, v)
            continue

        # Also catch "42:49 AM 193" style lines directly (time + count inline)
        # Pattern: optional noise + time + spaces + 2-4 digit count at end
        tm_cnt = re.search(
            r'(\d{1,2}):(\d{2})\s*(am|pm)\s+(\d{2,4})\s*$', ll
        )
        if tm_cnt:
            hour = int(tm_cnt.group(1))
            ampm = tm_cnt.group(3)
            if ampm == 'pm' and hour != 12:
                hour += 12
            if ampm == 'am' and hour == 12:
                hour = 0
            first_pickup_hour = hour
            v = int(tm_cnt.group(4))
            if 10 <= v <= 2000:
                total_pickups = max(total_pickups, v)

    return total_pickups, first_pickup_hour


def parse_iphone_notifications(lines):
    """
    Parse iOS Notifications summary page.

    Real OCR output:
      'Notifications'
      'Yesterday, 9 March'
      '14°'          ← big count with OCR degree-symbol noise
      '20'           ← chart Y-axis label (higher than real count)
      '0'            ← chart axis
      'S M T W T FS' ← chart day labels
      '10'           ← chart Y-axis
      '5'            ← chart Y-axis
      'Phone'        ← app name
      '\\. e114 >'   ← garbled app row (not a real app name — rejected)

    Strategy: collect all standalone numeric candidates after the date line,
    then take the MINIMUM — the real count (14) is always ≤ chart axis labels
    (20, 10, 5...). Garbled app rows like '\\. e114' are rejected because
    they don't start with real letters.
    """
    candidates = []
    date_seen = False

    for line in lines:
        ll = line.strip()
        lower = ll.lower()

        if re.search(r'\byesterday\b|\btoday\b', lower):
            date_seen = True
            continue

        # Skip known non-count lines
        if any(kw in lower for kw in ['notification', '12 am', '6 am', '12 pm',
                                       '6 pm', 'average', 'weekly',
                                       'show more', 'add limit']):
            continue
        # Skip chart day-label rows (multiple single-letter tokens: S M T W T F S)
        if re.search(r'\b[smtwtfs]\b.*\b[smtwtfs]\b', lower):
            continue
        # Skip time-axis rows like "12AM 6AM 12PM 6PM"
        if re.search(r'\d+\s*[ap]m.*\d+\s*[ap]m', lower):
            continue

        if not date_seen:
            continue

        # Standalone number line (e.g. "14°", "20", "10", "5")
        # Strip trailing OCR noise then check if purely numeric
        cleaned = re.sub(r'[°•·\s>)]+$', '', ll).strip()
        cleaned = re.sub(r'^[^0-9]+', '', cleaned).strip()
        if cleaned.isdigit():
            v = int(cleaned)
            if 1 <= v <= 2000:
                candidates.append(v)
            continue

        # App row: must start with real word letters (≥2), not OCR junk like '\. e'
        # e.g. "Phone   14" → valid;  "\\. e114 >" → rejected
        if re.match(r'^[A-Za-z]{2,}', ll):
            m = re.search(r'(\d+)\s*[>\s]*$', ll)
            if m:
                v = int(m.group(1))
                if 1 <= v <= 500:
                    candidates.append(v)

    # True count = first candidate after the date line.
    # It appears immediately after "Yesterday/Today" as the big displayed number.
    # Chart Y-axis labels and app rows come later and are ignored once we have it.
    if candidates:
        return candidates[0]
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
#  SAMSUNG PARSERS
# ═══════════════════════════════════════════════════════════════════════════════

def samsung_detect_category(lines):
    for line in lines[:5]:
        ll = re.sub(r'^[<\s€]+', '', line.lower().strip()).strip()
        for key, cat in SAMSUNG_CATEGORY_MAP.items():
            if ll == key or ll.startswith(key):
                return cat
    return 'unknown'


def samsung_parse_category_time(lines, is_daily):
    for line in lines[3:12]:
        ll = line.lower()
        if any(kw in ll for kw in ['app timer', 'view all', 'notification',
                                    'app open', 'unlock', 'weekly', 'daily',
                                    'dashboard', 'show app']):
            continue
        clean = re.sub(r'[^0-9hHmM\s]', '', line).strip()
        if is_daily:
            mins = parse_mins(clean)
            if mins >= 5:
                return mins
        else:
            m = re.search(r'(\d+\s*h\s*\d+\s*m|\d+\s*m)\s*/\s*day', ll)
            if m:
                return parse_mins(m.group(1))
    return 0


def samsung_parse_notifications_unlocks(lines, is_daily):
    notifs, unlocks = -1, -1
    for line in lines:
        ll = line.lower().strip()
        m = re.match(r'^(\d+)\s+notification', ll)
        if m:
            v = int(m.group(1))
            if 0 <= v <= 2000:
                notifs = max(notifs, v)
            continue
        m = re.match(r'^(\d+)\s+unlock', ll)
        if m:
            v = int(m.group(1))
            if 0 <= v <= 1000:
                unlocks = v
    return max(notifs, 0), max(unlocks, 0)


def samsung_parse_dashboard(lines, is_daily):
    result = {'total_min': 0, 'social': 0, 'productivity': 0,
              'video': 0, 'gaming': 0, 'others': 0,
              'notifications': -1, 'unlocks': -1}
    for line in lines[3:10]:
        ll = line.lower()
        if any(kw in ll for kw in ['notification', 'unlock', 'social',
                                    'productivity', 'video', 'dashboard']):
            continue
        clean = re.sub(r'[^0-9hHmM\s]', '', line).strip()
        mins = parse_mins(clean)
        if is_daily and mins >= 10:
            result['total_min'] = mins
            break
    CAT_ROW = re.compile(
        r'[@\.\•\*\s]*(social|productivity and finance|productivity|video|gaming)'
        r'\s+(\d+\s*h\s*\d+\s*m|\d+\s*h|\d+\s*m)',
        re.IGNORECASE
    )
    for line in lines:
        m = CAT_ROW.search(line.lower())
        if m:
            cat_name = m.group(1).strip()
            mins = parse_mins(m.group(2))
            mapped = SAMSUNG_CATEGORY_MAP.get(cat_name, 'others')
            if mapped in result and mins > 0:
                result[mapped] = max(result.get(mapped, 0), mins)
    n, u = samsung_parse_notifications_unlocks(lines, is_daily)
    result['notifications'] = n
    result['unlocks'] = u
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  DUPLICATE DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class DuplicateDetector:
    """
    Two-layer duplicate detection:
      Layer 1: Text hash  — identical OCR content
      Layer 2: pHash      — visually near-identical (hamming distance <= threshold)
    """
    def __init__(self, phash_threshold=10):
        self.text_hashes = set()
        self.phashes = []          # list of (hash_str, img_name)
        self.threshold = phash_threshold

    def is_duplicate(self, text, img_path):
        # Layer 1: text content hash
        th = hash(text[:400].strip())
        if th in self.text_hashes:
            return True, 'text_duplicate'
        self.text_hashes.add(th)

        # Layer 2: perceptual image hash
        ph = phash(img_path)
        if ph:
            for existing_ph, existing_name in self.phashes:
                dist = hamming(ph, existing_ph)
                if dist <= self.threshold:
                    return True, f'visual_duplicate_of_{existing_name}'
            self.phashes.append((ph, Path(img_path).name))

        return False, None


# ═══════════════════════════════════════════════════════════════════════════════
#  MASTER EXTRACTION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_user_data(user_folder):
    """
    Process all screenshots in a folder → 18-feature dict.

    Auto-detects brand and page type per screenshot.
    Removes duplicates automatically (text + visual hash).
    Merges data: daily beats weekly, list page beats individual app pages.
    """
    folder = Path(user_folder)
    exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    images = sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])

    if not images:
        raise ValueError(f"No images found in {folder}")

    # ── Accumulators ─────────────────────────────────────────────────────────
    daily  = {'total': 0, 'social': 0, 'productivity': 0, 'video': 0,
              'gaming': 0, 'others': 0, 'notifications': -1, 'unlocks': -1}
    weekly = {k: v for k, v in daily.items()}

    # For Stock Android: track if we have the authoritative list page
    list_page_seen = False
    list_page_app_names = set()   # app names already captured from list page

    dup = DuplicateDetector(phash_threshold=10)

    for img_path in images:
        try:
            text = ocr_image(str(img_path))
        except Exception as e:
            print(f"    Skipping {img_path.name}: {e}")
            continue

        # ── Duplicate check ──────────────────────────────────────────────────
        is_dup, reason = dup.is_duplicate(text, str(img_path))
        if is_dup:
            print(f"    {img_path.name} → DUPLICATE ({reason}) — skipped")
            continue

        lines = [l.strip() for l in text.split('\n') if l.strip()]
        brand, page_type = detect_brand(lines)
        is_daily = detect_daily(lines)
        target = daily if is_daily else weekly

        print(f"    {img_path.name} → brand={brand}  page={page_type}  daily={is_daily}")

        # ════════════════════════════════════════════════════════════════════
        #  STOCK ANDROID
        # ════════════════════════════════════════════════════════════════════
        if brand == 'stock_android':

            if page_type == 'stock_list':
                parsed = parse_stock_list_page(lines, is_daily)

                # Store total
                if parsed['total_min'] > 0:
                    target['total'] = max(target['total'], parsed['total_min'])
                    if is_daily:
                        list_page_seen = True

                # Accumulate per-category from app list
                cat_totals = {}
                for app_name, mins in parsed['apps'].items():
                    cat = app_to_category(app_name)
                    cat_totals[cat] = cat_totals.get(cat, 0) + mins
                    if is_daily:
                        list_page_app_names.add(app_name)

                for cat, total in cat_totals.items():
                    if cat in target:
                        target[cat] = max(target[cat], total)

                print(f"      total={parsed['total_min']}  "
                      f"apps_found={len(parsed['apps'])}  "
                      f"sample={list(parsed['apps'].items())[:4]}")

            elif page_type == 'stock_app_detail':
                app_name, mins = parse_stock_app_detail(lines, is_daily)
                if app_name and mins > 0:
                    app_lower = app_name.lower().strip()
                    # Skip if this app was already captured from the list page
                    if list_page_seen and app_lower in list_page_app_names:
                        print(f"      {app_name} ({mins}m) — already in list page, skipped")
                    else:
                        cat = app_to_category(app_name)
                        target[cat] = max(target[cat], mins)
                        print(f"      app={app_name}  mins={mins}  cat={cat}")

        # ════════════════════════════════════════════════════════════════════
        #  iPHONE / iOS SCREEN TIME
        # ════════════════════════════════════════════════════════════════════
        elif brand == 'iphone':

            if page_type == 'iphone_day_summary':
                parsed = parse_iphone_day_summary(lines)
                # Use the explicit big-number total only — never sum categories into it
                if parsed['total_min'] > 0:
                    target['total'] = max(target['total'], parsed['total_min'])
                # Category breakdowns go into their slots independently
                for cat in ('social', 'productivity', 'video', 'gaming', 'others'):
                    if parsed[cat] > 0:
                        target[cat] = max(target.get(cat, 0), parsed[cat])
                print(f"      total={parsed['total_min']} (explicit only)  "
                      f"social={parsed['social']}  video={parsed['video']}  "
                      f"productivity={parsed['productivity']}")

            elif page_type == 'iphone_most_used':
                apps = parse_iphone_most_used(lines)
                cat_totals = {}
                for app_name, mins in apps.items():
                    cat = app_to_category(app_name)
                    cat_totals[cat] = cat_totals.get(cat, 0) + mins
                for cat, total in cat_totals.items():
                    if cat in target:
                        target[cat] = max(target.get(cat, 0), total)
                # Also accumulate total from app list if not already set
                # Only use summed app times as total if no explicit total captured yet
                list_total = sum(apps.values())
                if list_total > 0 and target['total'] == 0:
                    target['total'] = list_total
                print(f"      apps_found={len(apps)}  "
                      f"sample={list(apps.items())[:4]}  derived_total={list_total}")

            elif page_type == 'iphone_app_detail':
                app_name, today_mins, daily_avg, ios_category = parse_iphone_app_detail(lines)
                if app_name and today_mins > 0:
                    cat = app_to_category(app_name)
                    # Refine category using iOS's own category label if available
                    if ios_category:
                        mapped = IPHONE_APP_CATEGORY_MAP.get(ios_category.lower(), cat)
                        cat = mapped
                    target[cat] = max(target.get(cat, 0), today_mins)
                    print(f"      app={app_name}  today={today_mins}m  "
                          f"avg={daily_avg}m  ios_cat={ios_category}  → {cat}")

            elif page_type == 'iphone_pickups':
                pickups, first_hour = parse_iphone_pickups(lines)
                if pickups > 0:
                    target['unlocks'] = max(target.get('unlocks', -1), pickups)
                print(f"      pickups={pickups}  first_pickup_hour={first_hour}")

            elif page_type == 'iphone_notifications':
                notifs = parse_iphone_notifications(lines)
                if notifs > 0:
                    target['notifications'] = max(target.get('notifications', -1), notifs)
                print(f"      notifications={notifs}")

        # ════════════════════════════════════════════════════════════════════
        #  SAMSUNG
        # ════════════════════════════════════════════════════════════════════
        elif brand == 'samsung':

            if page_type == 'samsung_dashboard':
                dash = samsung_parse_dashboard(lines, is_daily)
                if is_daily:
                    if dash['total_min'] > 0:
                        daily['total'] = max(daily['total'], dash['total_min'])
                    for cat in ('social', 'productivity', 'video', 'gaming', 'others'):
                        if dash[cat] > 0:
                            daily[cat] = max(daily[cat], dash[cat])
                    if dash['notifications'] > 0:
                        daily['notifications'] = max(daily['notifications'], dash['notifications'])
                    if dash['unlocks'] >= 0:
                        daily['unlocks'] = max(daily['unlocks'], dash['unlocks'])

            elif page_type == 'samsung_category':
                cat = samsung_detect_category(lines)
                cat_time = samsung_parse_category_time(lines, is_daily)
                if cat_time > 0:
                    target[cat] = max(target[cat], cat_time)
                n, _ = samsung_parse_notifications_unlocks(lines, is_daily)
                if n > 0:
                    target['notifications'] = max(target['notifications'], n)

        else:
            print(f"      Unknown brand/page — attempting generic extraction")
            for line in lines[2:6]:
                mins = parse_mins(line.lower())
                if mins >= 20:
                    target['total'] = max(target['total'], mins)
                    break

    # ── Merge daily + weekly (daily wins) ─────────────────────────────────────
    def pick(key):
        dv = daily.get(key) or 0
        wv = weekly.get(key) or 0
        return dv if dv > 0 else wv

    def pick_meta(key):
        dv = daily.get(key, -1)
        wv = weekly.get(key, -1)
        if dv is not None and dv >= 0:
            return dv
        if wv is not None and wv >= 0:
            return wv
        return 0

    social        = pick('social')
    productivity  = pick('productivity')
    video         = pick('video')
    gaming        = pick('gaming')
    others        = pick('others')
    total_notifs  = pick_meta('notifications')
    total_unlocks = pick_meta('unlocks')

    total = daily.get('total') or 0
    if total == 0:
        total = weekly.get('total') or 0
    if total == 0:
        total = social + productivity + video + gaming + others
    total = max(total, 1)

    total_hours  = max(total / 60, 0.1)
    prod_safe    = max(productivity, 1)
    sprod        = round(social / prod_safe, 2)
    night_min    = round(social * 0.12 + video * 0.10)

    print(f"\n    FINAL → total={total}  social={social}  prod={productivity}  "
          f"video={video}  gaming={gaming}  notifs={total_notifs}  unlocks={total_unlocks}")

    return {
        'total_screen_time_min':     total,
        'social_time_min':           social,
        'productivity_time_min':     productivity,
        'video_time_min':            video,
        'gaming_time_min':           gaming,
        'others_time_min':           others,
        'night_10pm_2am_min':        night_min,
        'early_morning_2am_6am_min': round(night_min * 0.3),
        'morning_6am_12pm_min':      round(total * 0.18),
        'afternoon_12pm_6pm_min':    round(total * 0.45),
        'evening_6pm_10pm_min':      round(total * 0.25),
        'total_notifications':       total_notifs,
        'total_unlocks':             total_unlocks,
        'social_prod_ratio':         min(sprod, 9.99),
        'night_usage_pct':           round(night_min / total * 100, 2),
        'gaming_pct':                round(gaming   / total * 100, 2),
        'notification_per_hour':     round(total_notifs  / total_hours, 2),
        'unlock_frequency_per_hour': round(total_unlocks / total_hours, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else '.'
    result = extract_user_data(folder)
    print("\n── Extracted Features ──")
    for k, v in result.items():
        print(f"  {k}: {v}")