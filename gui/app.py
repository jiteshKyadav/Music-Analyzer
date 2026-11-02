import customtkinter as ctk
from tkinter import messagebox
import os
import subprocess
import threading
import time
import pygame

from main import analyze_audio as run_analysis


# ============================================================
# MUSIC ANALYZER — PREMIUM DARK / AMBER UI
# ============================================================

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")


# -----------------------------
# Colors
# -----------------------------
BG = "#050608"
SURFACE = "#0B0D10"
SURFACE_2 = "#101318"
BORDER = "#252A32"
BORDER_AMBER = "#5A3A08"

TEXT = "#F5F5F7"
TEXT_DIM = "#9A9DA5"
TEXT_MUTED = "#666A73"

AMBER = "#FFB000"
AMBER_BRIGHT = "#FFC43D"
AMBER_DARK = "#8A5A00"
GREEN = "#42E88A"
RED = "#FF5D5D"


# -----------------------------
# Audio mixer
# -----------------------------
_mixer_ok = False
_mixer_init_error = ""

try:
    pygame.mixer.init()
    _mixer_ok = True
except pygame.error as exc:
    _mixer_init_error = str(exc)


# -----------------------------
# Main window
# -----------------------------
app = ctk.CTk()
app.title("Music Analyzer")
app.geometry("1600x880")
app.minsize(1200, 820)
app.configure(fg_color=BG)


# -----------------------------
# State
# -----------------------------
selected_audio_path = None

player_state = "no_file"  # no_file | stopped | playing | paused
_playback_start_ts = None
_playback_offset = 0.0
_total_duration = None
_updating_seek = False

_is_analyzing = False


# ============================================================
# Helpers
# ============================================================

def _fmt_mmss(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _shorten_filename(name, max_len=58):
    if len(name) <= max_len:
        return name

    stem, ext = os.path.splitext(name)
    available = max_len - len(ext) - 3

    if available < 8:
        return name[:max_len - 3] + "..."

    head = int(available * 0.68)
    tail = available - head

    return f"{stem[:head]}...{stem[-tail:]}{ext}"


def _detect_duration(file_path):
    try:
        import soundfile as sf
        info = sf.info(file_path)
        if info.duration and info.duration > 0:
            return float(info.duration)
    except Exception:
        pass

    if os.path.splitext(file_path)[1].lower() == ".wav":
        try:
            import wave
            with wave.open(file_path, "rb") as wav:
                rate = wav.getframerate()
                if rate:
                    return wav.getnframes() / float(rate)
        except Exception:
            pass

    if _mixer_ok:
        try:
            snd = pygame.mixer.Sound(file_path)
            length = snd.get_length()
            if length and length > 0:
                return float(length)
        except Exception:
            pass

    return None


def _current_position_seconds():
    if player_state == "playing" and _playback_start_ts is not None:
        return _playback_offset + (time.monotonic() - _playback_start_ts)

    return _playback_offset


def _refresh_timer():
    pos = _current_position_seconds()

    if _total_duration:
        pos = min(max(pos, 0), _total_duration)
        timer_label.configure(
            text=f"{_fmt_mmss(pos)} / {_fmt_mmss(_total_duration)}"
        )
    else:
        timer_label.configure(
            text=f"{_fmt_mmss(pos)} / --:--"
        )


def _refresh_seek_slider():
    global _updating_seek

    if not _total_duration or _total_duration <= 0:
        return

    pos = min(max(_current_position_seconds(), 0), _total_duration)

    _updating_seek = True
    try:
        seek_slider.set(pos)
    finally:
        _updating_seek = False


def _tick():
    global player_state, _playback_start_ts, _playback_offset

    try:
        if player_state == "playing" and _mixer_ok:
            if not pygame.mixer.music.get_busy():
                if _total_duration is not None:
                    _playback_offset = _total_duration
                elif _playback_start_ts is not None:
                    _playback_offset += time.monotonic() - _playback_start_ts

                _playback_start_ts = None
                player_state = "stopped"

                if not _is_analyzing:
                    status_label.configure(text="●  Status : Stopped")

        _refresh_timer()
        _refresh_seek_slider()

    finally:
        app.after(100, _tick)


# ============================================================
# Windows native file picker
# ============================================================

# ============================================================
# Windows native file picker
# ============================================================

# ============================================================
# Windows native file picker
# ============================================================

# ============================================================
# Windows native file picker
# ============================================================

# ============================================================
# Audio File Picker
# ============================================================

# ============================================================
# Windows Native File Picker
# ============================================================

# ============================================================
# Windows Native File Picker
# ============================================================

def _open_windows_file_picker():
    ps_script = r"""
$ErrorActionPreference = 'Stop'

Add-Type -AssemblyName System.Windows.Forms | Out-Null
Add-Type -AssemblyName System.Drawing | Out-Null

# ------------------------------------------------------------
# Win32 helper
# ------------------------------------------------------------
Add-Type @'
using System;
using System.Runtime.InteropServices;

public class MusicAnalyzerWindowHelper
{
    [DllImport("user32.dll")]
    public static extern IntPtr GetForegroundWindow();

    [DllImport("user32.dll")]
    public static extern IntPtr FindWindow(
        string lpClassName,
        string lpWindowName
    );

    [DllImport("user32.dll")]
    public static extern bool SetForegroundWindow(IntPtr hWnd);

    [DllImport("user32.dll")]
    public static extern bool BringWindowToTop(IntPtr hWnd);

    [DllImport("user32.dll")]
    public static extern bool SetWindowPos(
        IntPtr hWnd,
        IntPtr hWndInsertAfter,
        int X,
        int Y,
        int cx,
        int cy,
        uint uFlags
    );

    public static readonly IntPtr HWND_TOPMOST =
        new IntPtr(-1);

    public static readonly IntPtr HWND_NOTOPMOST =
        new IntPtr(-2);

    public const uint SWP_NOSIZE = 0x0001;
    public const uint SWP_NOMOVE = 0x0002;
    public const uint SWP_SHOWWINDOW = 0x0040;
}
'@

# ------------------------------------------------------------
# Find Music Analyzer window
#
# First try the currently active window.
# If that is not Music Analyzer, search by title.
# ------------------------------------------------------------
$guiHandle = [MusicAnalyzerWindowHelper]::GetForegroundWindow()

$musicAnalyzerHandle = [MusicAnalyzerWindowHelper]::FindWindow(
    $null,
    'Music Analyzer'
)

if ($musicAnalyzerHandle -ne [IntPtr]::Zero) {
    $guiHandle = $musicAnalyzerHandle
}

# ------------------------------------------------------------
# Create a NativeWindow wrapper around the WSL GUI HWND
# ------------------------------------------------------------
$owner = $null

if ($guiHandle -ne [IntPtr]::Zero) {
    $owner = New-Object System.Windows.Forms.NativeWindow
    $owner.AssignHandle($guiHandle)
}

# ------------------------------------------------------------
# Native Windows File Picker
# ------------------------------------------------------------
$dlg = New-Object System.Windows.Forms.OpenFileDialog

$dlg.Title = 'Select Audio File'
$dlg.Filter = 'Audio Files (*.wav;*.mp3)|*.wav;*.mp3'
$dlg.Multiselect = $false
$dlg.CheckFileExists = $true
$dlg.RestoreDirectory = $true

# ------------------------------------------------------------
# Show dialog WITH Music Analyzer as owner
# ------------------------------------------------------------
if ($owner -ne $null) {
    $result = $dlg.ShowDialog($owner)
}
else {
    $result = $dlg.ShowDialog()
}

$fileName = $null

if ($result -eq [System.Windows.Forms.DialogResult]::OK) {
    $fileName = $dlg.FileName
}

# ------------------------------------------------------------
# Release owner handle
# ------------------------------------------------------------
if ($owner -ne $null) {
    try {
        $owner.ReleaseHandle()
    } catch {}
}

# ------------------------------------------------------------
# Return selected Windows path
# ------------------------------------------------------------
if ($fileName) {
    Write-Output $fileName
}
"""

    try:
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-STA",
                "-WindowStyle",
                "Hidden",
                "-Command",
                ps_script,
            ],
            capture_output=True,
            text=True,
        )

    except FileNotFoundError as exc:
        raise RuntimeError(
            "powershell.exe was not found. "
            "This GUI expects to run under WSL."
        ) from exc

    if result.returncode != 0:
        err = (result.stderr or "").strip() or "Unknown error"
        raise RuntimeError(
            f"Windows file picker failed: {err}"
        )

    return (result.stdout or "").strip() or None

def _windows_path_to_wsl(win_path):
    result = subprocess.run(
        ["wslpath", "-u", win_path],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        err = (result.stderr or "").strip() or "Unknown error"
        raise RuntimeError(
            f"Path conversion failed: {err}"
        )

    path = (result.stdout or "").strip()

    if not path:
        raise RuntimeError(
            "Path conversion returned an empty path."
        )

    return path

# ============================================================
# Browse
# ============================================================

def browse_audio():
    global selected_audio_path
    global player_state, _playback_start_ts
    global _playback_offset, _total_duration

    try:
        win_path = _open_windows_file_picker()
    except Exception as exc:
        messagebox.showerror(
            "File Picker Error",
            str(exc)
        )
        return

    if not win_path:
        return

    try:
        file_path = _windows_path_to_wsl(win_path)
    except Exception as exc:
        messagebox.showerror(
            "File Picker Error",
            str(exc)
        )
        return
        
    if _mixer_ok:
        try:
            pygame.mixer.music.stop()
        except pygame.error:
            pass

    selected_audio_path = file_path
    player_state = "stopped"
    _playback_start_ts = None
    _playback_offset = 0.0
    _total_duration = _detect_duration(file_path)

    filename = os.path.basename(file_path)

    filename_label.configure(
        text=_shorten_filename(filename),
        text_color=AMBER_BRIGHT
    )

    track_meta.configure(
        text="WAV / MP3  •  Ready for analysis"
    )

    if _total_duration:
        seek_slider.configure(
            from_=0,
            to=max(_total_duration, 1)
        )

    seek_slider.set(0)

    _refresh_timer()

    for label in value_labels.values():
        label.configure(text="--")

    progress_bar.set(0)
    progress_percent.configure(text="0%")
    analysis_caption.configure(text="Ready for analysis")
    status_label.configure(text="●  Status : Audio Selected")

    complete_badge.configure(
        text="●  READY",
        text_color=TEXT_DIM
    )


# ============================================================
# Playback
# ============================================================

def _warn_no_file():
    messagebox.showwarning(
        "No Audio Selected",
        "Please select an audio file first."
    )


def _player_error():
    messagebox.showerror(
        "Audio Player Error",
        _mixer_init_error or "Audio playback is unavailable."
    )


def play_audio():
    global player_state, _playback_start_ts, _playback_offset

    if selected_audio_path is None:
        _warn_no_file()
        return

    if not _mixer_ok:
        _player_error()
        return

    try:
        if player_state == "paused":
            pygame.mixer.music.unpause()
            _playback_start_ts = time.monotonic()

        else:
            pygame.mixer.music.load(selected_audio_path)
            pygame.mixer.music.play()

            _playback_start_ts = time.monotonic()
            _playback_offset = 0.0

    except pygame.error as exc:
        messagebox.showerror("Playback Error", str(exc))
        return

    player_state = "playing"

    if not _is_analyzing:
        status_label.configure(text="●  Status : Playing")


def pause_audio():
    global player_state, _playback_start_ts, _playback_offset

    if selected_audio_path is None:
        _warn_no_file()
        return

    if not _mixer_ok:
        _player_error()
        return

    if player_state != "playing":
        return

    try:
        pygame.mixer.music.pause()
    except pygame.error as exc:
        messagebox.showerror("Playback Error", str(exc))
        return

    if _playback_start_ts is not None:
        _playback_offset += time.monotonic() - _playback_start_ts

    _playback_start_ts = None
    player_state = "paused"

    if not _is_analyzing:
        status_label.configure(text="●  Status : Paused")


def resume_audio():
    global player_state, _playback_start_ts

    if selected_audio_path is None:
        _warn_no_file()
        return

    if not _mixer_ok:
        _player_error()
        return

    if player_state != "paused":
        return

    try:
        pygame.mixer.music.unpause()
    except pygame.error as exc:
        messagebox.showerror("Playback Error", str(exc))
        return

    _playback_start_ts = time.monotonic()
    player_state = "playing"

    if not _is_analyzing:
        status_label.configure(text="●  Status : Playing")


def stop_audio():
    global player_state, _playback_start_ts, _playback_offset

    if selected_audio_path is None:
        _warn_no_file()
        return

    if not _mixer_ok:
        _player_error()
        return

    try:
        pygame.mixer.music.stop()
    except pygame.error:
        pass

    _playback_start_ts = None
    _playback_offset = 0.0
    player_state = "stopped"

    seek_slider.set(0)
    _refresh_timer()

    if not _is_analyzing:
        status_label.configure(text="●  Status : Stopped")


def seek_audio(value):
    global _playback_start_ts, _playback_offset, player_state

    if _updating_seek:
        return

    if selected_audio_path is None or not _mixer_ok:
        return

    if not _total_duration:
        return

    new_pos = max(0.0, min(float(value), _total_duration))
    was_playing = player_state == "playing"

    try:
        pygame.mixer.music.load(selected_audio_path)
        pygame.mixer.music.play(start=new_pos)

        _playback_offset = new_pos
        _playback_start_ts = time.monotonic()

        player_state = "playing" if was_playing else "paused"

        if not was_playing:
            pygame.mixer.music.pause()
            _playback_start_ts = None

    except (pygame.error, TypeError):
        pass


# ============================================================
# Analysis
# ============================================================

def _format_duration(seconds):
    total = int(round(float(seconds)))
    return f"{total // 60:02d}:{total % 60:02d}"


def _format_results(results):
    best_key = results.get("best_key")

    if best_key:
        key_text = f"{best_key['key']} {str(best_key['scale']).capitalize()}"
    else:
        key_text = "Not detected"

    genre = results.get("genre")

    if genre:
        genre_name = genre.get("genre", "Not detected")
        confidence = genre.get("confidence")

        if confidence is not None:
            genre_text = genre_name
        else:
            genre_text = genre_name
    else:
        genre_text = "Not detected"

    return {
        "BPM": f"{float(results['bpm']['bpm']):.2f}",
        "Key / Scale": key_text,
        "Loudness": f"{float(results['loudness']):.2f} LUFS",
        "Dynamic Range": f"{float(results['dynamic_range']):.2f} dB",
        "Duration": _format_duration(results["duration"]),
        "Sample Rate": f"{int(results['sample_rate'])} Hz",
        "Genre": genre_text,
    }


def _analysis_worker(file_path):
    try:
        results = run_analysis(file_path)
        app.after(0, _analysis_success, results)
    except Exception as exc:
        app.after(0, _analysis_error, exc)


def analyze_audio():
    global _is_analyzing

    if selected_audio_path is None:
        _warn_no_file()
        return

    if _is_analyzing:
        return

    _is_analyzing = True

    browse_btn.configure(state="disabled")
    analyze_btn.configure(state="disabled")

    complete_badge.configure(
        text="●  ANALYZING",
        text_color=AMBER
    )

    status_label.configure(
        text="●  Status : Analyzing..."
    )

    analysis_caption.configure(
        text="Extracting measurable audio characteristics..."
    )

    progress_bar.set(0)
    progress_percent.configure(text="0%")

    for label in value_labels.values():
        label.configure(text="--")

    threading.Thread(
        target=_analysis_worker,
        args=(selected_audio_path,),
        daemon=True
    ).start()

    # Visual progress animation.
    _animate_progress(0.0)


def _animate_progress(value):
    if not _is_analyzing:
        return

    # Deliberately approaches 96% instead of freezing early.
    # Final 4% is reserved for result delivery.
    if value < 0.96:
        step = 0.006 if value < 0.75 else 0.0025
        new_value = min(value + step, 0.96)

        progress_bar.set(new_value)
        progress_percent.configure(
            text=f"{int(new_value * 100)}%"
        )

        app.after(45, _animate_progress, new_value)


def _analysis_success(results):
    global _is_analyzing

    try:
        formatted = _format_results(results)

        for key, value in formatted.items():
            value_labels[key].configure(text=value)

        progress_bar.set(1.0)
        progress_percent.configure(text="100%")

        analysis_caption.configure(
            text="Analysis completed successfully"
        )

        status_label.configure(
            text="●  Status : Analysis Completed",
            text_color=GREEN
        )

        complete_badge.configure(
            text="✓  COMPLETE",
            text_color=GREEN
        )

    except Exception as exc:
        _analysis_error(exc)
        return

    finally:
        _is_analyzing = False
        browse_btn.configure(state="normal")
        analyze_btn.configure(state="normal")


def _analysis_error(exc):
    global _is_analyzing

    messagebox.showerror(
        "Analysis Error",
        str(exc) or "Unable to analyze the selected audio file."
    )

    progress_bar.set(0)
    progress_percent.configure(text="0%")

    analysis_caption.configure(text="Analysis failed")

    status_label.configure(
        text="●  Status : Error",
        text_color=RED
    )

    complete_badge.configure(
        text="●  ERROR",
        text_color=RED
    )

    _is_analyzing = False
    browse_btn.configure(state="normal")
    analyze_btn.configure(state="normal")


# ============================================================
# Window close
# ============================================================

def on_closing():
    if messagebox.askyesno("Exit", "Are you sure you want to exit?"):
        try:
            if _mixer_ok:
                pygame.mixer.music.stop()
                pygame.mixer.quit()
        except Exception:
            pass

        app.destroy()


# ============================================================
# UI Components — LARGE PREMIUM DESKTOP LAYOUT
# ============================================================

root = ctk.CTkFrame(
    app,
    fg_color=BG,
    corner_radius=0
)
root.pack(fill="both", expand=True)


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
header = ctk.CTkFrame(root, fg_color="transparent")
header.pack(fill="x", padx=46, pady=(24, 14))

brand = ctk.CTkFrame(header, fg_color="transparent")
brand.pack(side="left")

brand_icon = ctk.CTkLabel(
    brand,
    text="◒",
    font=("Segoe UI Symbol", 38, "bold"),
    text_color=AMBER
)
brand_icon.pack(side="left", padx=(0, 14))

brand_text = ctk.CTkFrame(brand, fg_color="transparent")
brand_text.pack(side="left")

title_label = ctk.CTkLabel(
    brand_text,
    text="Music Analyzer",
    font=("Segoe UI", 38, "bold"),
    text_color=TEXT,
    anchor="w"
)
title_label.pack(anchor="w")

subtitle_label = ctk.CTkLabel(
    brand_text,
    text="Audio Intelligence Studio",
    font=("Segoe UI", 13),
    text_color=AMBER,
    anchor="w"
)
subtitle_label.pack(anchor="w", pady=(1, 0))

complete_badge = ctk.CTkLabel(
    header,
    text="✓  COMPLETE",
    font=("Segoe UI", 13, "bold"),
    text_color=GREEN,
    fg_color="#0D1013",
    corner_radius=18,
    padx=17,
    pady=9
)
complete_badge.pack(side="right", pady=(3, 0))


# ------------------------------------------------------------
# Card helpers
# ------------------------------------------------------------
def card(parent, **kwargs):
    defaults = dict(
        fg_color=SURFACE,
        corner_radius=17,
        border_width=1,
        border_color=BORDER
    )
    defaults.update(kwargs)
    return ctk.CTkFrame(parent, **defaults)


def section_label(parent, text):
    return ctk.CTkLabel(
        parent,
        text=text.upper(),
        font=("Segoe UI", 13, "bold"),
        text_color=TEXT_DIM,
        anchor="w"
    )


# ------------------------------------------------------------
# Track information — large horizontal card
# ------------------------------------------------------------
track_card = card(root, border_color=BORDER_AMBER)
track_card.pack(
    fill="x",
    padx=46,
    pady=(2, 14)
)

track_card.grid_columnconfigure(1, weight=1)

track_top = ctk.CTkFrame(track_card, fg_color="transparent")
track_top.grid(
    row=0,
    column=0,
    columnspan=3,
    sticky="ew",
    padx=22,
    pady=(15, 4)
)

section_label(track_top, "Track Information").pack(side="left")

ctk.CTkLabel(
    track_top,
    text="WAV / MP3",
    font=("Segoe UI", 11),
    text_color=TEXT_MUTED
).pack(side="right")

track_body = ctk.CTkFrame(track_card, fg_color="transparent")
track_body.grid(
    row=1,
    column=0,
    columnspan=3,
    sticky="ew",
    padx=22,
    pady=(5, 18)
)
track_body.grid_columnconfigure(1, weight=1)

track_icon = ctk.CTkLabel(
    track_body,
    text="♫",
    font=("Segoe UI Symbol", 38, "bold"),
    text_color=AMBER,
    fg_color="#17130A",
    corner_radius=14,
    width=70,
    height=64
)
track_icon.grid(row=0, column=0, padx=(0, 18))

track_text = ctk.CTkFrame(track_body, fg_color="transparent")
track_text.grid(row=0, column=1, sticky="ew")

filename_label = ctk.CTkLabel(
    track_text,
    text="No audio selected",
    font=("Segoe UI", 18, "bold"),
    text_color=TEXT_DIM,
    anchor="w"
)
filename_label.pack(anchor="w")

track_meta = ctk.CTkLabel(
    track_text,
    text="WAV / MP3  •  Ready for analysis",
    font=("Segoe UI", 13),
    text_color=TEXT_MUTED,
    anchor="w"
)
track_meta.pack(anchor="w", pady=(5, 0))

browse_btn = ctk.CTkButton(
    track_body,
    text="Choose Audio   ⌁",
    width=155,
    height=44,
    corner_radius=11,
    fg_color="#12151A",
    hover_color="#1A1E24",
    border_width=1,
    border_color=BORDER,
    text_color=TEXT,
    font=("Segoe UI", 14, "bold"),
    command=browse_audio
)
browse_btn.grid(row=0, column=2, padx=(18, 0))


# ------------------------------------------------------------
# Playback + analysis — two large equal panels
# ------------------------------------------------------------
middle = ctk.CTkFrame(root, fg_color="transparent")
middle.pack(
    fill="x",
    padx=46,
    pady=(0, 14)
)

middle.grid_columnconfigure(0, weight=1, uniform="middle")
middle.grid_columnconfigure(1, weight=1, uniform="middle")


# Playback
playback_card = card(middle)
playback_card.grid(
    row=0,
    column=0,
    sticky="nsew",
    padx=(0, 7)
)

playback_header = ctk.CTkFrame(playback_card, fg_color="transparent")
playback_header.pack(fill="x", padx=22, pady=(18, 0))

section_label(playback_header, "Playback Control").pack(side="left")

ctk.CTkLabel(
    playback_header,
    text="AUDIO TRANSPORT",
    font=("Segoe UI", 11),
    text_color=TEXT_MUTED
).pack(side="right")

seek_slider = ctk.CTkSlider(
    playback_card,
    from_=0,
    to=1,
    height=16,
    button_length=22,
    button_corner_radius=10,
    fg_color="#20242B",
    progress_color=AMBER,
    button_color=AMBER_BRIGHT,
    button_hover_color="#FFD36B",
    command=seek_audio
)
seek_slider.pack(fill="x", padx=22, pady=(19, 3))
seek_slider.set(0)

playback_bottom = ctk.CTkFrame(playback_card, fg_color="transparent")
playback_bottom.pack(fill="x", padx=22, pady=(3, 18))

timer_label = ctk.CTkLabel(
    playback_bottom,
    text="00:00 / --:--",
    font=("Consolas", 12, "bold"),
    text_color=TEXT_DIM
)
timer_label.pack(side="left")

controls = ctk.CTkFrame(playback_bottom, fg_color="transparent")
controls.pack(side="right")

play_btn = ctk.CTkButton(
    controls,
    text="▶",
    width=48,
    height=38,
    corner_radius=12,
    fg_color=AMBER,
    hover_color=AMBER_BRIGHT,
    text_color="#111111",
    font=("Segoe UI Symbol", 14, "bold"),
    command=play_audio
)
play_btn.pack(side="left", padx=4)

pause_btn = ctk.CTkButton(
    controls,
    text="Ⅱ",
    width=48,
    height=38,
    corner_radius=12,
    fg_color="#171A20",
    hover_color="#22262D",
    border_width=1,
    border_color=BORDER,
    font=("Segoe UI Symbol", 15, "bold"),
    command=pause_audio
)
pause_btn.pack(side="left", padx=4)

resume_btn = ctk.CTkButton(
    controls,
    text="↻",
    width=48,
    height=38,
    corner_radius=12,
    fg_color="#171A20",
    hover_color="#22262D",
    border_width=1,
    border_color=BORDER,
    font=("Segoe UI Symbol", 13, "bold"),
    command=resume_audio
)
resume_btn.pack(side="left", padx=4)

stop_btn = ctk.CTkButton(
    controls,
    text="■",
    width=48,
    height=38,
    corner_radius=12,
    fg_color="#171A20",
    hover_color="#22262D",
    border_width=1,
    border_color=BORDER,
    font=("Segoe UI Symbol", 15, "bold"),
    command=stop_audio
)
stop_btn.pack(side="left", padx=4)


# Analysis
analysis_card = card(middle, border_color=BORDER_AMBER)
analysis_card.grid(
    row=0,
    column=1,
    sticky="nsew",
    padx=(7, 0)
)

analysis_header = ctk.CTkFrame(analysis_card, fg_color="transparent")
analysis_header.pack(fill="x", padx=22, pady=(18, 0))

section_label(analysis_header, "Analysis Engine").pack(side="left")

analysis_caption = ctk.CTkLabel(
    analysis_card,
    text="Ready for analysis",
    font=("Segoe UI", 13),
    text_color=TEXT_DIM,
    anchor="w"
)
analysis_caption.pack(fill="x", padx=22, pady=(16, 9))

progress_bar = ctk.CTkProgressBar(
    analysis_card,
    height=9,
    corner_radius=5,
    fg_color="#1A1D22",
    progress_color=AMBER
)
progress_bar.pack(fill="x", padx=22)
progress_bar.set(0)

analysis_bottom = ctk.CTkFrame(analysis_card, fg_color="transparent")
analysis_bottom.pack(fill="x", padx=22, pady=(8, 18))

progress_percent = ctk.CTkLabel(
    analysis_bottom,
    text="0%",
    font=("Consolas", 13, "bold"),
    text_color=AMBER
)
progress_percent.pack(side="right")


# ------------------------------------------------------------
# Analyze button row
# ------------------------------------------------------------
analyze_row = ctk.CTkFrame(root, fg_color="transparent")
analyze_row.pack(
    fill="x",
    padx=46,
    pady=(0, 13)
)

analyze_btn = ctk.CTkButton(
    analyze_row,
    text="Analyze Audio   →",
    width=170,
    height=46,
    corner_radius=12,
    fg_color=AMBER,
    hover_color=AMBER_BRIGHT,
    text_color="#111111",
    font=("Segoe UI", 14, "bold"),
    command=analyze_audio
)
analyze_btn.pack(side="right")


# ------------------------------------------------------------
# Results heading
# ------------------------------------------------------------
results_heading = ctk.CTkFrame(root, fg_color="transparent")
results_heading.pack(
    fill="x",
    padx=46,
    pady=(0, 8)
)

ctk.CTkLabel(
    results_heading,
    text="ANALYSIS RESULTS",
    font=("Segoe UI", 14, "bold"),
    text_color=TEXT_DIM
).pack(side="left")

ctk.CTkLabel(
    results_heading,
    text="LIVE METRICS",
    font=("Segoe UI", 11),
    text_color=TEXT_MUTED
).pack(side="right")


# ------------------------------------------------------------
# LARGE 3 x 3 metric grid
# ------------------------------------------------------------
results_grid = ctk.CTkFrame(root, fg_color="transparent")
results_grid.pack(
    fill="x",
    padx=46
)

for col in range(3):
    results_grid.grid_columnconfigure(
        col,
        weight=1,
        uniform="metric"
    )

for row in range(3):
    results_grid.grid_rowconfigure(
        row,
        weight=1,
        uniform="metric_row"
    )


result_specs = [
    ("BPM", "♬"),
    ("Key / Scale", "◉"),
    ("Loudness", "∿"),
    ("Dynamic Range", "▥"),
    ("Duration", "◷"),
    ("Sample Rate", "∿"),
    ("Genre", "♫"),
]


# ------------------------------------------------------------
# EXACT CARD POSITIONS
#
# Row 0:
# BPM | Key / Scale | Loudness
#
# Row 1:
# Dynamic Range | Duration | Sample Rate
#
# Row 2:
#        Genre
# ------------------------------------------------------------
card_positions = {
    "BPM": (0, 0),
    "Key / Scale": (0, 1),
    "Loudness": (0, 2),

    "Dynamic Range": (1, 0),
    "Duration": (1, 1),
    "Sample Rate": (1, 2),

    "Genre": (2, 1),
}


value_labels = {}


for key, icon in result_specs:

    row, col = card_positions[key]

    result_card = ctk.CTkFrame(
        results_grid,
        fg_color=SURFACE,
        corner_radius=16,
        border_width=1,
        border_color=BORDER
    )

    result_card.grid(
        row=row,
        column=col,
        sticky="nsew",
        padx=5,
        pady=5
    )

    # Icon
    ctk.CTkLabel(
        result_card,
        text=icon,
        font=("Segoe UI Symbol", 23, "bold"),
        text_color=AMBER
    ).pack(
        anchor="w",
        padx=20,
        pady=(15, 3)
    )

    # Metric name
    ctk.CTkLabel(
        result_card,
        text=key.upper(),
        font=("Segoe UI", 12, "bold"),
        text_color=TEXT_DIM
    ).pack(
        anchor="w",
        padx=20
    )

    # Metric value
    value = ctk.CTkLabel(
        result_card,
        text="--",
        font=("Segoe UI", 24, "bold"),
        text_color=TEXT,
        anchor="w"
    )

    value.pack(
        anchor="w",
        padx=20,
        pady=(7, 4)
    )

    # Subtle amber line for visual weight
    ctk.CTkFrame(
        result_card,
        height=2,
        fg_color=AMBER
    ).pack(
        fill="x",
        padx=20,
        pady=(3, 15)
    )

    value_labels[key] = value


# ------------------------------------------------------------
# Status
# ------------------------------------------------------------
# ------------------------------------------------------------
# Status
# ------------------------------------------------------------
status_label = ctk.CTkLabel(
    results_grid,
    text="●  Status : Ready",
    font=("Segoe UI", 13),
    text_color=TEXT_MUTED,
    anchor="w"
)

status_label.grid(
    row=2,
    column=0,
    sticky="w",
    padx=(5, 5),
    pady=(5, 5)
)


# ------------------------------------------------------------
# Start
# ------------------------------------------------------------
app.protocol("WM_DELETE_WINDOW", on_closing)

_tick()

app.mainloop()