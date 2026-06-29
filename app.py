import os
import re
import uuid
import json
import time
import wave
import struct
import shutil
import hashlib
import hmac
import subprocess
import threading
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).parent

# Cloud par dono folders /tmp mein
IS_CLOUD = bool(os.environ.get('RAILWAY_ENVIRONMENT') or os.environ.get('RENDER') or os.environ.get('RAILWAY_STATIC_URL'))
STATIC_DIR   = Path('/tmp/sc_static')    if IS_CLOUD else BASE_DIR / 'static'
DOWNLOADS_DIR = Path('/tmp/sc_downloads') if IS_CLOUD else BASE_DIR / 'downloads'

STATIC_DIR.mkdir(exist_ok=True, parents=True)
DOWNLOADS_DIR.mkdir(exist_ok=True, parents=True)

# ─── Razorpay Config ────────────────────────────────────────────────
# Paste your Razorpay keys here:
RAZORPAY_KEY_ID     = 'rzp_test_T75Oq87HC0lH1y'   # Replace with your Key ID
RAZORPAY_KEY_SECRET = '33EKelT2CkQsvhVsPxCjhsoo'  # Replace with your Key Secret

# ─── License Key Storage ─────────────────────────────────────────────
# Stored in licenses.json inside the app folder
LICENSES_FILE = BASE_DIR / 'licenses.json'

def load_licenses():
    try:
        if LICENSES_FILE.exists():
            with open(LICENSES_FILE, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def save_licenses(data):
    try:
        with open(LICENSES_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

def generate_license_key():
    """Generate a unique license key: SC-XXXX-XXXX-XXXX-XXXX"""
    parts = [uuid.uuid4().hex[:4].upper() for _ in range(4)]
    return 'SC-' + '-'.join(parts)

def validate_license(key):
    """
    Returns: ('pro', expiry_date_str) if valid Pro
             ('free', None)           if expired or invalid
    """
    licenses = load_licenses()
    entry = licenses.get(key)
    if not entry:
        return ('free', None)
    expiry = datetime.fromisoformat(entry['expiry'])
    if datetime.now() > expiry:
        return ('expired', entry['expiry'])
    return ('pro', entry['expiry'])

def activate_license(key, plan='monthly', email=''):
    """Activate a new license key."""
    licenses = load_licenses()
    if plan == 'yearly':
        expiry = datetime.now() + timedelta(days=365)
    else:
        expiry = datetime.now() + timedelta(days=30)
    licenses[key] = {
        'email': email,
        'plan': plan,
        'activated': datetime.now().isoformat(),
        'expiry': expiry.isoformat(),
    }
    save_licenses(licenses)
    return expiry.strftime('%d %b %Y')

# ─── Premium Feature Limits ──────────────────────────────────────────
FREE_LIMITS = {
    'max_clips': 3,
    'max_quality': '720',
    'subtitle_styles': ['static', 'karaoke'],
    'face_tracking': False,
    'max_langs': ['en'],
    'watermark': True,
}

PRO_LIMITS = {
    'max_clips': 15,
    'max_quality': '2160',
    'subtitle_styles': ['static', 'karaoke', 'popin', 'beasty', 'mozi', 'deepdiver', 'popline'],
    'face_tracking': True,
    'max_langs': ['en','es','fr','de','ru','ar','ja','ko','pt','it','tr','nl','pl','id','vi','th'],
    'watermark': False,
}

jobs = {}
manual_sessions = {}

# Lazy-loaded Whisper model
_whisper_model = None
_whisper_lock = threading.Lock()


def get_whisper_model():
    global _whisper_model
    with _whisper_lock:
        if _whisper_model is None:
            import whisper
            _whisper_model = whisper.load_model("base")
        return _whisper_model


def check_whisper():
    try:
        import whisper  # noqa
        return True
    except Exception:
        return False


def check_mediapipe():
    try:
        import mediapipe  # noqa
        import cv2  # noqa
        return True
    except Exception:
        return False


def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
        return True
    except Exception:
        return False


def check_ytdlp():
    try:
        import yt_dlp  # noqa
        return True
    except Exception:
        return False


def update_job(job_id, **kwargs):
    if job_id in jobs:
        jobs[job_id].update(kwargs)


def set_stage(job_id, stage_idx, status):
    if job_id in jobs:
        jobs[job_id]['stages'][stage_idx]['status'] = status


def validate_youtube_url(url):
    pattern = r'^(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w-]+'
    return bool(re.match(pattern, url))


def get_video_id(url):
    patterns = [
        r'youtube\.com/watch\?v=([\w-]+)',
        r'youtu\.be/([\w-]+)',
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def analyze_audio_loudness(wav_path, clip_duration, num_clips):
    with wave.open(str(wav_path), 'rb') as wf:
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        fmt = f'{len(raw)//2}h'
        samples = np.array(struct.unpack(fmt, raw), dtype=np.float32)
    else:
        samples = np.frombuffer(raw, dtype=np.int8).astype(np.float32)

    if n_channels > 1:
        samples = samples[::n_channels]

    total_seconds = int(n_frames / framerate)

    rms_per_sec = []
    for sec in range(total_seconds):
        start = int(sec * framerate)
        end = int((sec + 1) * framerate)
        chunk = samples[start:end]
        if len(chunk) == 0:
            rms_per_sec.append(0.0)
        else:
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            rms_per_sec.append(rms)

    window = clip_duration
    scored = []
    for i in range(len(rms_per_sec) - window):
        score = sum(rms_per_sec[i:i + window])
        scored.append((i, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    selected = []
    min_gap = max(10, clip_duration)
    for start_sec, score in scored:
        overlap = False
        for s, _ in selected:
            if abs(start_sec - s) < (clip_duration + min_gap):
                overlap = True
                break
        if not overlap:
            selected.append((start_sec, score))
        if len(selected) >= num_clips:
            break

    if selected:
        max_score = max(s for _, s in selected)
        min_score = min(s for _, s in selected)
        rng = max_score - min_score if max_score != min_score else 1
        selected = [(start, int(((score - min_score) / rng) * 60 + 40)) for start, score in selected]

    selected.sort(key=lambda x: x[0])
    return selected, total_seconds


def format_srt_timestamp(seconds):
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_srt_from_clip(clip_path, srt_path):
    """Run Whisper on the clip audio and write an SRT subtitle file."""
    model = get_whisper_model()
    result = model.transcribe(str(clip_path), fp16=False, task='translate')

    segments = result.get('segments', [])
    with open(srt_path, 'w', encoding='utf-8') as f:
        for idx, seg in enumerate(segments, start=1):
            start_ts = format_srt_timestamp(seg['start'])
            end_ts = format_srt_timestamp(seg['end'])
            text = seg['text'].strip()
            if text:
                f.write(f"{idx}\n{start_ts} --> {end_ts}\n{text}\n\n")

    return len(segments) > 0

    return True


def hex_to_ass(hex_color):
    """Convert #RRGGBB hex color to ASS format &H00BBGGRR (reversed)."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r = hex_color[0:2]
        g = hex_color[2:4]
        b = hex_color[4:6]
        return f"&H00{b}{g}{r}".upper()
    return "&H00FFFFFF"


def burn_subtitles(input_path, srt_path, output_path, color='#FFFFFF', font='Arial'):
    """Convert SRT to ASS then burn — gives full control over wrap and font size."""
    import re as _re

    ass_color = hex_to_ass(color)

    # Read SRT
    try:
        with open(srt_path, 'r', encoding='utf-8') as f:
            srt_content = f.read()
    except Exception:
        return False

    # Parse SRT blocks
    blocks = re.split(r'\n\s*\n', srt_content.strip())
    ass_events = []
    for block in blocks:
        lines_b = block.strip().splitlines()
        if len(lines_b) < 3:
            continue
        # Find timestamp line
        ts_line = None
        text_start = 0
        for li, l in enumerate(lines_b):
            if '-->' in l:
                ts_line = l
                text_start = li + 1
                break
        if not ts_line:
            continue
        ts = ts_line.strip()
        # Convert SRT timestamps (HH:MM:SS,mmm) to ASS (H:MM:SS.cc)
        def srt_ts_to_ass(ts_str):
            ts_str = ts_str.strip()
            m = re.match(r'(\d+):(\d+):(\d+)[,.](\d+)', ts_str)
            if not m:
                return '0:00:00.00'
            h, mn, s, ms = m.groups()
            cs = int(ms[:3]) // 10
            return f"{int(h)}:{int(mn):02d}:{int(s):02d}.{cs:02d}"
        parts_ts = ts.split('-->')
        if len(parts_ts) != 2:
            continue
        start_ass = srt_ts_to_ass(parts_ts[0])
        end_ass = srt_ts_to_ass(parts_ts[1])
        text = ' '.join(lines_b[text_start:]).strip()
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        if not text:
            continue
        ass_events.append(
            f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{text}"
        )

    if not ass_events:
        return False

    # Hindi text needs Nirmala UI font — override user font for Devanagari support
    has_hindi = any('\u0900' <= ch <= '\u097F' for line in ass_events for ch in line)
    font_name = 'Nirmala UI' if has_hindi else font

    # Write ASS with small font and proper margins
    ass_content = f"""\
[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 1

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},38,{ass_color},&H00FFFFFF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,2,0,2,120,120,120,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""" + '\n'.join(ass_events) + '\n'

    ass_path_tmp = Path(str(srt_path).replace('.srt', '_static.ass'))
    try:
        with open(ass_path_tmp, 'w', encoding='utf-8') as f:
            f.write(ass_content)
    except Exception:
        return False

    # Burn using ASS filter (better wrap support than subtitles filter)
    import re as _re2
    ass_str = str(ass_path_tmp).replace('\\', '/')
    ass_str = _re2.sub(r'^([A-Za-z]):', r'\1\\:', ass_str)
    vf = f"ass='{ass_str}'"

    cmd = [
        'ffmpeg', '-y',
        '-i', str(input_path),
        '-vf', vf,
        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '26',
        '-profile:v', 'main', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        '-c:a', 'aac', '-b:a', '96k',
        str(output_path)
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=180)
    if r.returncode != 0:
        print("[FFmpeg static subtitle error]")
        print(r.stderr.decode('utf-8', errors='ignore')[-800:])

    try:
        ass_path_tmp.unlink(missing_ok=True)
    except Exception:
        pass

    return r.returncode == 0 and Path(output_path).exists()


def format_ass_time(seconds):
    """Convert seconds to ASS timestamp: H:MM:SS.cc"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds - int(seconds)) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def generate_dynamic_ass(clip_path, ass_path, style='karaoke', color='#FFFFFF', font='Arial'):
    """
    Run Whisper with word_timestamps=True and generate an ASS file.
    Styles: karaoke, popin, beasty, mozi, deepdiver, popline
    """
    model = get_whisper_model()
    result = model.transcribe(
        str(clip_path), fp16=False,
        task='translate',
        word_timestamps=True
    )

    segments = result.get('segments', [])
    if not segments:
        return False

    all_words = []
    for seg in segments:
        for w in seg.get('words', []):
            word = w.get('word', '').strip()
            if word:
                all_words.append({
                    'word': word,
                    'start': float(w.get('start', seg['start'])),
                    'end': float(w.get('end', seg['end']))
                })

    if not all_words:
        return False

    CHUNK_SIZE = 4

    def make_chunks(words, size):
        chunks = []
        for i in range(0, len(words), size):
            chunks.append(words[i:i + size])
        return chunks

    chunks = make_chunks(all_words, CHUNK_SIZE)

    main_color = hex_to_ass(color)
    highlight_color = "&H0000FFFF"   # Yellow
    dim_color = "&H00808080"          # Grey

    all_text = ' '.join(w['word'] for w in all_words)
    has_hindi = any('\u0900' <= ch <= '\u097F' for ch in all_text)
    font_name = 'Nirmala UI' if has_hindi else font

    # --- Style-specific ASS header ---
    if style == 'beasty':
        # Bold white text, thick black outline — simple & clean
        style_line = f"Style: Default,{font_name},56,{main_color},&H00FFFFFF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,4,0,2,80,80,150,1"
    elif style == 'mozi':
        # Word-by-word, highlighted word gets colored box bg
        style_line = f"Style: Default,{font_name},54,{main_color},&H00FFFFFF,&H00000000,&HAA000000,1,0,0,0,100,100,0,0,1,3,0,2,80,80,150,1"
    elif style == 'deepdiver':
        # Dark background box behind each line
        style_line = f"Style: Default,{font_name},50,{main_color},&H00FFFFFF,&H00000000,&HCC000000,1,0,0,0,100,100,0,0,4,0,0,2,80,80,150,1"
    elif style == 'popline':
        # Colored text, bold, no outline — clean colorful look
        style_line = f"Style: Default,{font_name},58,{highlight_color},&H00FFFFFF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,2,0,2,80,80,150,1"
    else:
        # karaoke / popin default
        style_line = f"Style: Default,{font_name},52,{main_color},&H0000FFFF,&H00000000,&HAA000000,1,0,0,0,100,100,2,0,1,3,0,2,80,80,150,1"

    ass_header = f"""\
[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{style_line}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    lines = []

    if style == 'karaoke':
        for chunk in chunks:
            chunk_end = chunk[-1]['end']
            for wi, active_w in enumerate(chunk):
                ev_start = active_w['start']
                ev_end = chunk[wi + 1]['start'] if wi + 1 < len(chunk) else chunk_end
                if ev_end <= ev_start:
                    ev_end = ev_start + 0.1
                parts = []
                for j, w in enumerate(chunk):
                    if j < wi:
                        parts.append(f'{{\\c{dim_color}}}' + w['word'])
                    elif j == wi:
                        parts.append(f'{{\\c{highlight_color}\\fs58\\b1}}' + w['word'] + '{\\fs52\\b1}')
                    else:
                        parts.append(f'{{\\c{main_color}}}' + w['word'])
                    if j < len(chunk) - 1:
                        parts.append(' ')
                text = r'{\an2}' + ''.join(parts)
                lines.append(f"Dialogue: 0,{format_ass_time(ev_start)},{format_ass_time(ev_end)},Default,,0,0,0,,{text}")

    elif style == 'popin':
        for ci, chunk in enumerate(chunks):
            chunk_display_end = chunks[ci + 1][0]['start'] - 0.05 if ci + 1 < len(chunks) else chunk[-1]['end'] + 0.5
            for wi, active_w in enumerate(chunk):
                ev_start = active_w['start']
                ev_end = chunk[wi + 1]['start'] - 0.03 if wi + 1 < len(chunk) else chunk_display_end
                if ev_end <= ev_start:
                    ev_end = ev_start + 0.15
                text = (
                    r'{\an2}'
                    + f'{{\\c{highlight_color}\\fscx0\\fscy0'
                    + r'\t(0,100,\fscx115\fscy115)\t(100,180,\fscx100\fscy100)\b1}'
                    + active_w['word'] + r'{\b0}'
                )
                lines.append(f"Dialogue: 0,{format_ass_time(ev_start)},{format_ass_time(ev_end)},Default,,0,0,0,,{text}")

    elif style == 'beasty':
        # Full line shown all at once per chunk — bold & clean like MrBeast
        for chunk in chunks:
            ev_start = chunk[0]['start']
            ev_end = chunk[-1]['end'] + 0.1
            text = r'{\an2}' + ' '.join(w['word'] for w in chunk)
            lines.append(f"Dialogue: 0,{format_ass_time(ev_start)},{format_ass_time(ev_end)},Default,,0,0,0,,{text}")

    elif style == 'mozi':
        # Each word highlighted in color, rest white — Mozi style
        for chunk in chunks:
            chunk_end = chunk[-1]['end']
            for wi, active_w in enumerate(chunk):
                ev_start = active_w['start']
                ev_end = chunk[wi + 1]['start'] if wi + 1 < len(chunk) else chunk_end
                if ev_end <= ev_start:
                    ev_end = ev_start + 0.1
                parts = []
                for j, w in enumerate(chunk):
                    if j == wi:
                        # Highlighted word: colored + slight scale up
                        parts.append(f'{{\\c{highlight_color}\\fs60}}' + w['word'] + f'{{\\fs54\\c{main_color}}}')
                    else:
                        parts.append(f'{{\\c{main_color}}}' + w['word'])
                    if j < len(chunk) - 1:
                        parts.append(' ')
                text = r'{\an2}' + ''.join(parts)
                lines.append(f"Dialogue: 0,{format_ass_time(ev_start)},{format_ass_time(ev_end)},Default,,0,0,0,,{text}")

    elif style == 'deepdiver':
        # Full line, dark box background — Deep Diver style
        for chunk in chunks:
            ev_start = chunk[0]['start']
            ev_end = chunk[-1]['end'] + 0.15
            text = r'{\an2}' + ' '.join(w['word'] for w in chunk)
            lines.append(f"Dialogue: 0,{format_ass_time(ev_start)},{format_ass_time(ev_end)},Default,,0,0,0,,{text}")

    elif style == 'popline':
        # Each word pops in colored — Popline style
        for ci, chunk in enumerate(chunks):
            chunk_display_end = chunks[ci + 1][0]['start'] - 0.05 if ci + 1 < len(chunks) else chunk[-1]['end'] + 0.4
            for wi, active_w in enumerate(chunk):
                ev_start = active_w['start']
                ev_end = chunk[wi + 1]['start'] - 0.03 if wi + 1 < len(chunk) else chunk_display_end
                if ev_end <= ev_start:
                    ev_end = ev_start + 0.15
                text = (
                    r'{\an2\fscx0\fscy0\t(0,80,\fscx110\fscy110)\t(80,140,\fscx100\fscy100)}'
                    + active_w['word']
                )
                lines.append(f"Dialogue: 0,{format_ass_time(ev_start)},{format_ass_time(ev_end)},Default,,0,0,0,,{text}")

    with open(ass_path, 'w', encoding='utf-8') as f:
        f.write(ass_header)
        f.write('\n'.join(lines) + '\n')

    return True


def burn_dynamic_subtitles(input_path, ass_path, output_path):
    """Burn ASS dynamic subtitles into the video using ffmpeg."""
    # On Windows, FFmpeg needs forward slashes. Colon in drive letter needs special handling.
    ass_str = str(ass_path).replace('\\', '/')
    # Escape colon only in drive letter (e.g. C:/ → C\:/)
    import re as _re
    ass_str = _re.sub(r'^([A-Za-z]):', r'\1\\:', ass_str)
    vf = f"ass='{ass_str}'"
    cmd = [
        'ffmpeg', '-y',
        '-i', str(input_path),
        '-vf', vf,
        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '26', '-profile:v', 'main', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        '-c:a', 'aac', '-b:a', '96k',
        str(output_path)
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=300)
    if r.returncode != 0:
        print("[FFmpeg ASS error]", r.stderr.decode('utf-8', errors='ignore')[-800:])
    return r.returncode == 0 and Path(output_path).exists()



def face_reframe_clip(input_path, output_path, target_w=1080, target_h=1920, sample_every=8):
    """
    Detect faces using OpenCV Haar Cascade (works with all MediaPipe versions),
    track the largest face, smooth the crop, re-encode via Python->FFmpeg pipe.
    Returns True on success, False on failure (caller falls back to simple crop).
    """
    try:
        import cv2
    except ImportError:
        return False

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if vid_w == 0 or vid_h == 0 or total_frames == 0:
        cap.release()
        return False

    # Crop box: 9:16 ratio, fit inside video
    crop_h = vid_h
    crop_w = int(crop_h * target_w / target_h)
    if crop_w > vid_w:
        crop_w = vid_w
        crop_h = int(crop_w * target_h / target_w)

    # Load Haar cascade (ships with opencv-python, no extra download)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        cap.release()
        return False

    # Pass 1: sample every N frames, detect faces
    face_centers = {}
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
            )
            if len(faces) > 0:
                fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                face_centers[frame_idx] = (int(fx + fw / 2), int(fy + fh / 2))
            else:
                face_centers[frame_idx] = None
        frame_idx += 1
    cap.release()

    # Fill all frames by forward-fill
    all_cx = [vid_w // 2] * total_frames
    all_cy = [vid_h // 2] * total_frames
    last_x, last_y = vid_w // 2, vid_h // 2
    for i in range(total_frames):
        if i in face_centers and face_centers[i] is not None:
            last_x, last_y = face_centers[i]
        all_cx[i], all_cy[i] = last_x, last_y

    # Smooth with moving average (~1 second window)
    smooth_win = max(1, int(fps))
    def moving_avg(arr):
        out = []
        for i in range(len(arr)):
            lo = max(0, i - smooth_win // 2)
            hi = min(len(arr), i + smooth_win // 2 + 1)
            out.append(int(sum(arr[lo:hi]) / (hi - lo)))
        return out

    smooth_cx = moving_avg(all_cx)
    smooth_cy = moving_avg(all_cy)

    def clamp_crop(cx, cy):
        x = max(0, min(vid_w - crop_w, cx - crop_w // 2))
        y = max(0, min(vid_h - crop_h, cy - crop_h // 2))
        return x, y

    # Pass 2: pipe cropped frames to FFmpeg
    cap2 = cv2.VideoCapture(str(input_path))
    if not cap2.isOpened():
        return False

    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{crop_w}x{crop_h}',
        '-r', str(fps),
        '-i', 'pipe:0',
        '-i', str(input_path),
        '-map', '0:v', '-map', '1:a',
        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '26', '-profile:v', 'main', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        '-c:a', 'aac', '-b:a', '96k',
        '-s', f'{target_w}x{target_h}',
        '-shortest',
        str(output_path)
    ]

    try:
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        fi = 0
        while True:
            ret, frame = cap2.read()
            if not ret:
                break
            cx_i = smooth_cx[fi] if fi < len(smooth_cx) else vid_w // 2
            cy_i = smooth_cy[fi] if fi < len(smooth_cy) else vid_h // 2
            x, y = clamp_crop(cx_i, cy_i)
            cropped = frame[y:y+crop_h, x:x+crop_w]
            if cropped.shape[0] != crop_h or cropped.shape[1] != crop_w:
                cropped = cv2.resize(cropped, (crop_w, crop_h))
            try:
                proc.stdin.write(cropped.tobytes())
            except BrokenPipeError:
                break
            fi += 1
        proc.stdin.close()
        proc.wait()
        cap2.release()
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception:
        cap2.release()
        try:
            proc.kill()
        except Exception:
            pass
        return False


def process_video(job_id, url, clip_duration, num_clips, vertical_crop, add_subtitles, quality, face_reframe=False, subtitle_style='static', local_video_path=None, subtitle_color='#FFFFFF', subtitle_font='Arial'):
    try:
        set_stage(job_id, 0, 'running')

        if local_video_path:
            # Local file — skip download
            update_job(job_id, progress=10, message='Using uploaded video...')
            video_path = Path(local_video_path)
            if not video_path.exists():
                raise RuntimeError("Uploaded file not found")
            video_id = f"local_{job_id[:8]}"
            set_stage(job_id, 0, 'done')
        else:
            # YouTube download
            update_job(job_id, progress=5, message='Downloading video...')
            video_id = get_video_id(url)
            if not video_id:
                raise ValueError("Could not extract video ID")

            height_map = {'360': 360, '720': 720, '1080': 1080, '1440': 1440, '2160': 2160}
            max_height = height_map.get(str(quality), 1080)

            import yt_dlp
            ydl_opts = {
                'format': f'bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]/best',
                'merge_output_format': 'mp4',
                'outtmpl': str(DOWNLOADS_DIR / '%(id)s.%(ext)s'),
                'noplaylist': True,
                'quiet': True,
                'no_warnings': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                try:
                    ydl.download([url])
                except Exception as e:
                    raise RuntimeError(f"Download failed: {e}")

            candidates = list(DOWNLOADS_DIR.glob(f'{video_id}.*'))
            candidates = [c for c in candidates if c.suffix in ('.mp4', '.mkv', '.webm')]
            if not candidates:
                raise RuntimeError("Downloaded file not found")
            video_path = candidates[0]
            set_stage(job_id, 0, 'done')

        video_title = video_path.stem

        set_stage(job_id, 1, 'running')
        wav_path = DOWNLOADS_DIR / f'{video_id}.wav'
        cmd_audio = [
            'ffmpeg', '-y', '-i', str(video_path),
            '-vn', '-ar', '16000', '-ac', '1',   # 16kHz mono — faster extraction, enough for loudness analysis
            '-acodec', 'pcm_s16le',
            str(wav_path)
        ]
        r = subprocess.run(cmd_audio, capture_output=True, timeout=120)
        if r.returncode != 0:
            raise RuntimeError("Audio extraction failed")

        segments, total_seconds = analyze_audio_loudness(wav_path, clip_duration, num_clips)

        if total_seconds < 60:
            raise ValueError("Video too short - minimum 60 seconds required")

        if not segments:
            raise RuntimeError("Could not find any suitable segments")

        set_stage(job_id, 1, 'done')
        update_job(job_id, progress=40, message='Cutting clips...')

        set_stage(job_id, 2, 'running')
        clips = []
        raw_clip_paths = []

        # --- Parallel clip cutting ---
        def cut_single_clip(args):
            idx, start_sec, score = args
            clip_id = str(uuid.uuid4())[:8]
            raw_clip_path = DOWNLOADS_DIR / f'raw_{clip_id}.mp4'

            # Key speed trick: put -ss BEFORE -i for fast input seeking
            # ultrafast preset = 3-4x faster encoding than 'fast', negligible quality loss for short clips
            base_cmd = [
                'ffmpeg', '-y',
                '-ss', str(start_sec),          # seek BEFORE input = much faster
                '-i', str(video_path),
                '-t', str(clip_duration),
                '-avoid_negative_ts', '1',
            ]
            if vertical_crop:
                base_cmd += ['-vf', 'crop=ih*9/16:ih,scale=1080:1920']

            base_cmd += [
                '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '26',
                '-profile:v', 'main', '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                '-c:a', 'aac', '-b:a', '96k',  # 96k audio plenty for Shorts
                '-threads', '2',                 # limit per-process threads so parallel works better
                str(raw_clip_path)
            ]

            r = subprocess.run(base_cmd, capture_output=True, timeout=120)
            if r.returncode != 0:
                return None

            minutes = start_sec // 60
            seconds = start_sec % 60
            return {
                'idx': idx,
                'clip_id': clip_id,
                'raw_path': raw_clip_path,
                'start_sec': start_sec,
                'start_label': f'{minutes}:{seconds:02d}',
                'score': score,
            }

        # Run clip cuts in parallel (max 3 at a time to avoid overwhelming CPU)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        max_workers = min(3, len(segments))
        cut_args = [(idx, start_sec, score) for idx, (start_sec, score) in enumerate(segments)]

        results = [None] * len(cut_args)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(cut_single_clip, a): i for i, a in enumerate(cut_args)}
            done_count = 0
            for future in as_completed(future_map):
                i = future_map[future]
                result = future.result()
                results[i] = result
                done_count += 1
                pct = 40 + int((done_count / len(cut_args)) * 20)
                update_job(job_id, progress=pct)

        raw_clip_paths = [r for r in results if r is not None]

        set_stage(job_id, 2, 'done')

        # Stage 3: subtitles (if enabled) + finalize into static dir
        set_stage(job_id, 3, 'running')
        stage3_msg = 'Adding subtitles...' if add_subtitles else ('Face tracking...' if face_reframe else 'Finalizing clips...')
        update_job(job_id, progress=60, message=stage3_msg)

        total_clips = len(raw_clip_paths) or 1
        for i, item in enumerate(raw_clip_paths):
            clip_filename = f"clip_{item['clip_id']}.mp4"
            final_path = STATIC_DIR / clip_filename

            # Intermediate path for face reframe (before subtitles)
            if face_reframe:
                reframed_path = DOWNLOADS_DIR / f"reframed_{item['clip_id']}.mp4"
                success = face_reframe_clip(item['raw_path'], reframed_path)
                if success:
                    item['raw_path'].unlink(missing_ok=True)
                    item['raw_path'] = reframed_path
                # If face reframe fails, fall through to raw clip (no crash)

            if add_subtitles:
                subtitle_done = False
                if subtitle_style in ('karaoke', 'popin', 'beasty', 'mozi', 'deepdiver', 'popline'):
                    ass_path = DOWNLOADS_DIR / f"sub_{item['clip_id']}.ass"
                    try:
                        has_speech = generate_dynamic_ass(item['raw_path'], ass_path, style=subtitle_style, color=subtitle_color, font=subtitle_font)
                    except Exception as e:
                        print(f"[ASS generation error] {e}")
                        has_speech = False

                    if has_speech and ass_path.exists():
                        burned = burn_dynamic_subtitles(item['raw_path'], ass_path, final_path)
                        if burned:
                            subtitle_done = True
                        else:
                            print("[ASS burn failed, falling back to no subtitle]")

                    try:
                        ass_path.unlink(missing_ok=True)
                    except Exception:
                        pass

                else:
                    # Static SRT subtitles
                    srt_path = DOWNLOADS_DIR / f"sub_{item['clip_id']}.srt"
                    try:
                        has_speech = generate_srt_from_clip(item['raw_path'], srt_path)
                    except Exception as e:
                        print(f"[SRT generation error] {e}")
                        has_speech = False

                    if has_speech and srt_path.exists():
                        burned = burn_subtitles(item['raw_path'], srt_path, final_path, color=subtitle_color, font=subtitle_font)
                        if burned:
                            subtitle_done = True
                        else:
                            print("[SRT burn failed, falling back to no subtitle]")

                    try:
                        srt_path.unlink(missing_ok=True)
                    except Exception:
                        pass

                if not subtitle_done:
                    shutil.copy(str(item['raw_path']), str(final_path))
            else:
                shutil.copy(str(item['raw_path']), str(final_path))

            try:
                item['raw_path'].unlink(missing_ok=True)
            except Exception:
                pass

            clips.append({
                'id': item['clip_id'],
                'filename': clip_filename,
                'start_sec': item['start_sec'],
                'start_label': item['start_label'],
                'duration': clip_duration,
                'score': item['score'],
                'title': f"Clip {item['idx'] + 1}",
                'has_subtitles': add_subtitles,
                'subtitle_style': subtitle_style if add_subtitles else 'none',
                'face_reframed': face_reframe
            })

            progress_val = 60 + int(((i + 1) / total_clips) * 35)
            update_job(job_id, progress=progress_val)

        set_stage(job_id, 3, 'done')
        update_job(job_id, progress=100, message='Done!')

        try:
            # Always delete the wav. For local uploads, also delete the temp video copy.
            wav_path.unlink(missing_ok=True)
            if local_video_path:
                video_path.unlink(missing_ok=True)
            elif not local_video_path:
                video_path.unlink(missing_ok=True)
        except Exception:
            pass

        update_job(job_id,
                   status='done',
                   clips=clips,
                   video_title=video_title,
                   progress=100)

    except Exception as e:
        update_job(job_id, status='error', error=str(e), progress=0)
        for i in range(4):
            if jobs[job_id]['stages'][i]['status'] == 'running':
                set_stage(job_id, i, 'error')


@app.route('/')
def index():
    return send_file(str(BASE_DIR / 'index.html'), mimetype='text/html')


@app.route('/api/health')
def health():
    return jsonify({
        'status': 'ok',
        'ffmpeg': check_ffmpeg(),
        'ytdlp': check_ytdlp(),
        'whisper': check_whisper(),
        'mediapipe': check_mediapipe()
    })


# ─── License Routes ──────────────────────────────────────────────────

@app.route('/api/license/verify', methods=['POST'])
def license_verify():
    """Verify a license key entered by user."""
    data = request.get_json()
    key = str(data.get('key', '')).strip().upper()
    if not key:
        return jsonify({'error': 'No key provided'}), 400

    status, expiry = validate_license(key)
    if status == 'pro':
        return jsonify({
            'valid': True,
            'plan': 'pro',
            'expiry': expiry,
            'limits': PRO_LIMITS
        })
    elif status == 'expired':
        return jsonify({'valid': False, 'reason': 'expired', 'expiry': expiry})
    else:
        return jsonify({'valid': False, 'reason': 'invalid'})


@app.route('/api/license/activate', methods=['POST'])
def license_activate():
    """
    Admin route to generate + activate a new license key.
    Called after Razorpay payment is confirmed.
    Protected by admin secret.
    """
    data = request.get_json()
    secret = data.get('admin_secret', '')
    # Change this to your own secret password
    if secret != 'cortex_admin_2025':
        return jsonify({'error': 'Unauthorized'}), 403

    plan  = data.get('plan', 'monthly')
    email = data.get('email', '')
    key   = generate_license_key()
    expiry = activate_license(key, plan=plan, email=email)

    return jsonify({
        'key': key,
        'plan': plan,
        'email': email,
        'expiry': expiry
    })


@app.route('/api/razorpay/webhook', methods=['POST'])
def razorpay_webhook():
    """
    Razorpay webhook — auto-generates license on payment success.
    Set this URL in Razorpay Dashboard → Webhooks:
    http://YOUR_SERVER/api/razorpay/webhook
    """
    # Verify Razorpay signature
    webhook_secret = RAZORPAY_KEY_SECRET
    received_sig = request.headers.get('X-Razorpay-Signature', '')
    body = request.get_data()

    expected_sig = hmac.new(
        webhook_secret.encode(),
        body,
        hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(received_sig, expected_sig):
        return jsonify({'error': 'Invalid signature'}), 400

    event = request.get_json()
    if event.get('event') == 'payment.captured':
        payment = event['payload']['payment']['entity']
        email = payment.get('email', '')
        amount = payment.get('amount', 0)

        # Determine plan from amount
        plan = 'yearly' if amount >= 99900 else 'monthly'

        key = generate_license_key()
        expiry = activate_license(key, plan=plan, email=email)

        # Log it (in production, send email here)
        print(f"\n✅ Payment received!")
        print(f"   Email: {email}")
        print(f"   Plan:  {plan}")
        print(f"   Key:   {key}")
        print(f"   Valid until: {expiry}\n")

        # TODO: Send email with key to user
        # You can integrate smtplib or any email service here

    return jsonify({'status': 'ok'})


@app.route('/api/license/plans', methods=['GET'])
def get_plans():
    """Return available pricing plans."""
    return jsonify({
        'plans': [
            {
                'id': 'monthly',
                'name': 'Pro Monthly',
                'price': 199,
                'currency': 'INR',
                'duration': '30 days',
                'razorpay_link': 'https://rzp.io/l/shortscutter-pro',  # Update with your link
            },
            {
                'id': 'yearly',
                'name': 'Pro Yearly',
                'price': 999,
                'currency': 'INR',
                'duration': '365 days',
                'razorpay_link': 'https://rzp.io/l/shortscutter-yearly',  # Update with your link
                'savings': 'Save ₹1389/year',
            }
        ],
        'free_limits': FREE_LIMITS,
        'pro_limits': PRO_LIMITS,
    })



def upload_video():
    """Accept a local video file upload and save it to downloads dir."""
    if 'video' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    f = request.files['video']
    if not f.filename:
        return jsonify({'error': 'Empty filename'}), 400

    allowed = {'.mp4', '.mkv', '.webm', '.mov', '.avi', '.m4v', '.ts'}
    ext = Path(f.filename).suffix.lower()
    if ext not in allowed:
        return jsonify({'error': f'Unsupported file type: {ext}. Use mp4, mov, mkv, webm, avi'}), 400

    upload_id = str(uuid.uuid4())[:8]
    safe_name = f'upload_{upload_id}{ext}'
    save_path = DOWNLOADS_DIR / safe_name
    f.save(str(save_path))

    # Get duration via ffprobe
    probe_cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', str(save_path)
    ]
    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=15)
        duration = float(result.stdout.strip()) if result.stdout.strip() else 0
    except Exception:
        duration = 0

    return jsonify({
        'upload_id': upload_id,
        'filename': safe_name,
        'path': str(save_path),
        'original_name': f.filename,
        'duration': round(duration, 1),
        'size_mb': round(save_path.stat().st_size / (1024 * 1024), 1)
    })


@app.route('/api/process', methods=['POST'])
def process():
    data = request.get_json()
    url = data.get('url', '').strip()
    local_video_path = data.get('local_video_path', '').strip()
    clip_duration = int(data.get('clip_duration', 28))
    num_clips = int(data.get('num_clips', 3))

    # ─── License Check ───────────────────────────────────────────
    license_key = str(data.get('license_key', '')).strip().upper()
    lic_status, lic_expiry = validate_license(license_key) if license_key else ('free', None)
    is_pro = (lic_status == 'pro')
    limits = PRO_LIMITS if is_pro else FREE_LIMITS

    # Enforce free limits
    max_clips_allowed = limits['max_clips']
    num_clips = max(1, min(max_clips_allowed, num_clips))

    vertical_crop = bool(data.get('vertical_crop', True))
    add_subtitles = bool(data.get('add_subtitles', False))
    subtitle_style = str(data.get('subtitle_style', 'static'))

    # Free users: only static + karaoke
    if subtitle_style not in limits['subtitle_styles']:
        subtitle_style = 'static'

    subtitle_color = str(data.get('subtitle_color', '#FFFFFF'))
    if not subtitle_color.startswith('#') or len(subtitle_color) not in (4, 7):
        subtitle_color = '#FFFFFF'
    subtitle_font = str(data.get('subtitle_font', 'Arial'))
    import re as _re
    if not _re.match(r'^[\w\s]+$', subtitle_font):
        subtitle_font = 'Arial'

    # Face reframe — Pro only
    face_reframe = bool(data.get('face_reframe', False))
    if face_reframe and not is_pro:
        face_reframe = False

    quality = str(data.get('quality', '1080'))
    if quality not in ('360', '720', '1080', '1440', '2160'):
        quality = '1080'
    # Free: max 720p
    if not is_pro and quality in ('1080', '1440', '2160'):
        quality = '720'

    # Subtitle language — Free: English only
    subtitle_lang = str(data.get('subtitle_lang', 'en'))
    valid_langs = set(limits['max_langs'])
    if subtitle_lang not in valid_langs:
        subtitle_lang = 'en'

    # Validate: need either YouTube URL or local file
    use_local = bool(local_video_path)
    if not use_local and not validate_youtube_url(url):
        return jsonify({'error': 'Please provide a valid YouTube URL or upload a video file'}), 400

    if use_local:
        lp = Path(local_video_path)
        if not lp.exists():
            return jsonify({'error': 'Uploaded file not found. Please re-upload.'}), 400

    if not check_ffmpeg():
        return jsonify({'error': 'FFmpeg not installed'}), 500

    if not use_local and not check_ytdlp():
        return jsonify({'error': 'yt-dlp not installed'}), 500

    if add_subtitles and not check_whisper():
        return jsonify({'error': 'Whisper not installed. Run: pip install openai-whisper'}), 500

    if face_reframe and not check_mediapipe():
        return jsonify({'error': 'MediaPipe not installed. Run: pip install mediapipe opencv-python'}), 500

    subtitle_label = {
        'static': 'Adding subtitles',
        'karaoke': 'Adding karaoke subtitles',
        'popin': 'Adding pop-in subtitles',
        'beasty': 'Adding Beasty captions',
        'mozi': 'Adding Mozi captions',
        'deepdiver': 'Adding Deep Diver captions',
        'popline': 'Adding Popline captions'
    }.get(subtitle_style, 'Adding subtitles')
    stage3_name = subtitle_label if add_subtitles else ('Face tracking & reframe' if face_reframe else 'Finalizing clips')
    stage3_icon = '💬' if add_subtitles else ('🎯' if face_reframe else '📱')
    stage1_name = 'Reading local video' if use_local else 'Downloading video'

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        'status': 'running',
        'progress': 0,
        'message': 'Starting...',
        'clips': [],
        'video_title': '',
        'error': None,
        'stages': [
            {'name': stage1_name, 'icon': '📁' if use_local else '📥', 'status': 'waiting'},
            {'name': 'Analyzing audio peaks', 'icon': '🎵', 'status': 'waiting'},
            {'name': 'Cutting best moments', 'icon': '✂️', 'status': 'waiting'},
            {'name': stage3_name, 'icon': stage3_icon, 'status': 'waiting'},
        ]
    }

    t = threading.Thread(target=process_video,
                         args=(job_id, url, clip_duration, num_clips, vertical_crop,
                               add_subtitles, quality, face_reframe, subtitle_style,
                               local_video_path if use_local else None, subtitle_color, subtitle_font))
    t.daemon = True
    t.start()

    return jsonify({'job_id': job_id})


@app.route('/api/progress/<job_id>')
def progress(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(job)


@app.route('/api/download/<filename>')
def download(filename):
    safe = Path(filename).name
    path = STATIC_DIR / safe
    if not path.exists():
        return jsonify({'error': 'File not found'}), 404
    return send_file(str(path), as_attachment=True, download_name=safe)


@app.route('/api/thumbnail/<filename>')
def thumbnail(filename):
    safe = Path(filename).name
    video_path = STATIC_DIR / safe
    thumb_name = safe.replace('.mp4', '_thumb.jpg')
    thumb_path = STATIC_DIR / thumb_name

    if not thumb_path.exists():
        if not video_path.exists():
            return jsonify({'error': 'File not found'}), 404
        cmd = [
            'ffmpeg', '-y', '-i', str(video_path),
            '-ss', '00:00:01', '-vframes', '1',
            '-q:v', '2', str(thumb_path)
        ]
        subprocess.run(cmd, capture_output=True, timeout=15)

    if thumb_path.exists():
        return send_file(str(thumb_path), mimetype='image/jpeg')
    return jsonify({'error': 'Thumbnail failed'}), 500


@app.route('/api/trim', methods=['POST'])
def trim_clip():
    """Trim an already-generated clip to a new in/out point and save as a new clip."""
    data = request.get_json()
    source_filename = Path(data.get('filename', '')).name
    start = float(data.get('start', 0))
    end = float(data.get('end', 0))

    source_path = STATIC_DIR / source_filename
    if not source_path.exists():
        return jsonify({'error': 'Source clip not found'}), 404

    if end <= start:
        return jsonify({'error': 'End time must be after start time'}), 400

    duration = end - start
    if duration < 1:
        return jsonify({'error': 'Trimmed clip must be at least 1 second long'}), 400

    new_id = str(uuid.uuid4())[:8]
    new_filename = f'clip_{new_id}.mp4'
    new_path = STATIC_DIR / new_filename

    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start),
        '-i', str(source_path),
        '-t', str(duration),
        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '26', '-profile:v', 'main', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        '-c:a', 'aac', '-b:a', '96k',
        str(new_path)
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=120)
    if r.returncode != 0 or not new_path.exists():
        return jsonify({'error': 'Trim failed'}), 500

    return jsonify({
        'filename': new_filename,
        'duration': round(duration, 1)
    })


# ===== Manual Edit Mode =====
# Lets the user download the full original video and manually mark
# in/out points on a timeline instead of relying on auto-detection.

def download_full_video(session_id, url, quality):
    try:
        manual_sessions[session_id]['status'] = 'downloading'
        manual_sessions[session_id]['message'] = 'Downloading full video...'

        video_id = get_video_id(url)
        if not video_id:
            raise ValueError("Could not extract video ID")

        height_map = {'360': 360, '720': 720, '1080': 1080, '1440': 1440, '2160': 2160}
        max_height = height_map.get(str(quality), 1080)

        session_video_id = f"manual_{session_id}"
        import yt_dlp
        ydl_opts = {
            'format': f'bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]/best',
            'merge_output_format': 'mp4',
            'outtmpl': str(DOWNLOADS_DIR / f'{session_video_id}.%(ext)s'),
            'noplaylist': True,
            'quiet': True,
            'no_warnings': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download([url])
            except Exception as e:
                raise RuntimeError(f"Download failed: {e}")

        candidates = list(DOWNLOADS_DIR.glob(f'{session_video_id}.*'))
        candidates = [c for c in candidates if c.suffix in ('.mp4', '.mkv', '.webm')]
        if not candidates:
            raise RuntimeError("Downloaded file not found")
        video_path = candidates[0]

        cmd_probe = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)
        ]
        probe_result = subprocess.run(cmd_probe, capture_output=True, text=True, timeout=15)
        try:
            total_duration = float(probe_result.stdout.strip())
        except Exception:
            total_duration = 0

        manual_sessions[session_id].update({
            'status': 'ready',
            'message': 'Ready to edit',
            'video_path': str(video_path),
            'video_filename': video_path.name,
            'duration': round(total_duration, 1)
        })

    except Exception as e:
        manual_sessions[session_id].update({
            'status': 'error',
            'error': str(e)
        })


@app.route('/api/manual/start', methods=['POST'])
def manual_start():
    data = request.get_json()
    url = data.get('url', '').strip()
    local_video_path = data.get('local_video_path', '').strip()
    quality = str(data.get('quality', '1080'))
    if quality not in ('360', '720', '1080', '1440', '2160'):
        quality = '1080'

    use_local = bool(local_video_path)

    if use_local:
        lp = Path(local_video_path)
        if not lp.exists():
            return jsonify({'error': 'Uploaded file not found. Please re-upload.'}), 400
    elif not validate_youtube_url(url):
        return jsonify({'error': 'Invalid YouTube URL'}), 400

    if not check_ffmpeg():
        return jsonify({'error': 'FFmpeg not installed'}), 500

    if not use_local and not check_ytdlp():
        return jsonify({'error': 'yt-dlp not installed'}), 500

    session_id = str(uuid.uuid4())
    manual_sessions[session_id] = {
        'status': 'downloading',
        'message': 'Starting...',
        'video_path': None,
        'video_filename': None,
        'duration': 0,
        'error': None
    }

    if use_local:
        # No download needed — use local file directly, just probe duration
        def setup_local_session(sid, vpath):
            try:
                manual_sessions[sid]['status'] = 'downloading'
                manual_sessions[sid]['message'] = 'Preparing local video...'
                probe_cmd = [
                    'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1', vpath
                ]
                try:
                    result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=15)
                    total_duration = float(result.stdout.strip()) if result.stdout.strip() else 0
                except Exception:
                    total_duration = 0
                manual_sessions[sid].update({
                    'status': 'ready',
                    'message': 'Ready to edit',
                    'video_path': vpath,
                    'video_filename': Path(vpath).name,
                    'duration': round(total_duration, 1)
                })
            except Exception as e:
                manual_sessions[sid].update({'status': 'error', 'error': str(e)})

        t = threading.Thread(target=setup_local_session, args=(session_id, local_video_path))
    else:
        t = threading.Thread(target=download_full_video, args=(session_id, url, quality))

    t.daemon = True
    t.start()

    return jsonify({'session_id': session_id})


@app.route('/api/manual/progress/<session_id>')
def manual_progress(session_id):
    session = manual_sessions.get(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    return jsonify({
        'status': session['status'],
        'message': session.get('message', ''),
        'duration': session.get('duration', 0),
        'error': session.get('error')
    })


@app.route('/api/manual/video/<session_id>')
def manual_video(session_id):
    session = manual_sessions.get(session_id)
    if not session or session.get('status') != 'ready':
        return jsonify({'error': 'Video not ready'}), 404

    video_path = Path(session['video_path'])
    if not video_path.exists():
        return jsonify({'error': 'Video file missing'}), 404

    range_header = request.headers.get('Range')
    file_size = video_path.stat().st_size

    if not range_header:
        return send_file(str(video_path), mimetype='video/mp4')

    byte_range = range_header.replace('bytes=', '').split('-')
    start = int(byte_range[0]) if byte_range[0] else 0
    end = int(byte_range[1]) if len(byte_range) > 1 and byte_range[1] else file_size - 1
    end = min(end, file_size - 1)
    length = end - start + 1

    with open(video_path, 'rb') as f:
        f.seek(start)
        chunk = f.read(length)

    from flask import Response
    rv = Response(chunk, 206, mimetype='video/mp4', direct_passthrough=True)
    rv.headers.add('Content-Range', f'bytes {start}-{end}/{file_size}')
    rv.headers.add('Accept-Ranges', 'bytes')
    rv.headers.add('Content-Length', str(length))
    return rv


@app.route('/api/manual/cut', methods=['POST'])
def manual_cut():
    data = request.get_json()
    session_id = data.get('session_id', '')
    start = float(data.get('start', 0))
    end = float(data.get('end', 0))
    vertical_crop = bool(data.get('vertical_crop', True))

    session = manual_sessions.get(session_id)
    if not session or session.get('status') != 'ready':
        return jsonify({'error': 'Session not ready'}), 404

    video_path = Path(session['video_path'])
    if not video_path.exists():
        return jsonify({'error': 'Video file missing'}), 404

    if end <= start:
        return jsonify({'error': 'End time must be after start time'}), 400

    duration = end - start
    if duration < 1:
        return jsonify({'error': 'Clip must be at least 1 second long'}), 400

    clip_id = str(uuid.uuid4())[:8]
    clip_filename = f'clip_{clip_id}.mp4'
    clip_path = STATIC_DIR / clip_filename

    if vertical_crop:
        vf = 'crop=ih*9/16:ih,scale=1080:1920'
        cmd_cut = [
            'ffmpeg', '-y',
            '-ss', str(start),
            '-i', str(video_path),
            '-t', str(duration),
            '-vf', vf,
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '26', '-profile:v', 'main', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            '-c:a', 'aac', '-b:a', '96k',
            str(clip_path)
        ]
    else:
        cmd_cut = [
            'ffmpeg', '-y',
            '-ss', str(start),
            '-i', str(video_path),
            '-t', str(duration),
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '26', '-profile:v', 'main', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            '-c:a', 'aac', '-b:a', '96k',
            str(clip_path)
        ]

    r = subprocess.run(cmd_cut, capture_output=True, timeout=120)
    if r.returncode != 0 or not clip_path.exists():
        return jsonify({'error': 'Cutting failed'}), 500

    return jsonify({
        'filename': clip_filename,
        'duration': round(duration, 1)
    })


@app.route('/api/manual/cleanup/<session_id>', methods=['DELETE'])
def manual_cleanup(session_id):
    session = manual_sessions.get(session_id)
    if session and session.get('video_path'):
        try:
            Path(session['video_path']).unlink(missing_ok=True)
        except Exception:
            pass
    manual_sessions.pop(session_id, None)
    return jsonify({'deleted': True})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("\n🎬 ShortsCutter is starting...")
    print(f"🔗 Open your browser at: http://localhost:{port}\n")
    app.run(debug=False, host='0.0.0.0', port=port)
