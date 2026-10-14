"""Audio decoding, polishing and stitching.

Why not just concatenate TTS clips? Equal gaps and mismatched loudness are what
make multi-voice TTS sound synthetic. We therefore:
  * decode WAV or MP3 (via ffmpeg) into mono float32 at one sample rate;
  * trim leading/trailing silence from each clip so pauses are under our control;
  * loudness-match every turn to the same RMS target;
  * vary the gap between turns (shorter after quick reactions, longer between
    segments), with short fades to avoid clicks;
  * peak-limit and export MP3 for download/streaming.
Per-turn timestamps are returned so the transcript can follow the audio.
"""
from __future__ import annotations

import io
import random
import subprocess
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from ..config import get_settings


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # pragma: no cover - fall back to system ffmpeg
        return "ffmpeg"


def decode_audio(data: bytes, sr: int | None = None) -> np.ndarray:
    """Decode WAV/FLAC/MP3/OGG bytes to mono float32 at ``sr``."""
    sr = sr or get_settings().sample_rate
    try:
        audio, in_sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        return resample(audio, in_sr, sr)
    except Exception:
        pass
    proc = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", str(sr), "pipe:1"],
        input=data, capture_output=True, timeout=120,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise ValueError("could not decode audio: " + proc.stderr.decode(errors="ignore")[:200])
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def resample(audio: np.ndarray, in_sr: int, out_sr: int) -> np.ndarray:
    if in_sr == out_sr or len(audio) == 0:
        return audio.astype(np.float32)
    n_out = int(round(len(audio) * out_sr / in_sr))
    x_old = np.linspace(0, 1, num=len(audio), endpoint=False)
    x_new = np.linspace(0, 1, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def trim_silence(audio: np.ndarray, sr: int, thresh_db: float = -45.0, pad_s: float = 0.04) -> np.ndarray:
    if len(audio) == 0:
        return audio
    frame = max(1, int(sr * 0.01))
    n = len(audio) // frame
    if n == 0:
        return audio
    rms = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    loud = np.where(db > thresh_db)[0]
    if len(loud) == 0:
        return audio
    pad = int(pad_s * sr)
    start = max(0, loud[0] * frame - pad)
    end = min(len(audio), (loud[-1] + 1) * frame + pad)
    return audio[start:end]


def loudness_match(audio: np.ndarray, target_dbfs: float = -20.0) -> np.ndarray:
    """RMS-normalise speech-active frames to a common level so voices sit together."""
    if len(audio) == 0:
        return audio
    frame = 480
    n = max(1, len(audio) // frame)
    frames = audio[: n * frame].reshape(n, -1) if len(audio) >= frame else audio.reshape(1, -1)
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    active = rms[rms > np.max(rms) * 0.1] if np.max(rms) > 0 else rms
    level = float(np.sqrt(np.mean(active**2))) if len(active) else 1e-6
    gain = 10 ** (target_dbfs / 20) / max(level, 1e-6)
    return (audio * min(gain, 20.0)).astype(np.float32)


def fade(audio: np.ndarray, sr: int, ms: float = 12.0) -> np.ndarray:
    n = min(len(audio) // 2, int(sr * ms / 1000))
    if n <= 0:
        return audio
    out = audio.copy()
    ramp = np.linspace(0, 1, n, dtype=np.float32)
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out


def limit(audio: np.ndarray, ceiling: float = 0.95) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    return audio * (ceiling / peak) if peak > ceiling else audio


def prepare_clip(data: bytes, sr: int) -> np.ndarray:
    return fade(loudness_match(trim_silence(decode_audio(data, sr), sr)), sr)


def turn_gap(prev_text: str, next_text: str, rng: random.Random) -> float:
    """Pause before the next turn. Quick reactions follow fast; questions get a beat."""
    nxt_words = len(next_text.split())
    if nxt_words <= 4:  # "Right." / "Wait, really?" - snappy back-channel
        return rng.uniform(0.12, 0.22)
    if prev_text.rstrip().endswith("?"):
        return rng.uniform(0.30, 0.45)
    return rng.uniform(0.22, 0.38)


@dataclass
class StitchResult:
    audio: np.ndarray
    spans: list[tuple[float, float]]  # (start, end) per input clip, seconds
    duration: float


def stitch_turns(clips: list[np.ndarray], texts: list[str], sr: int, seed: int = 0, lead_in: float = 0.0) -> StitchResult:
    rng = random.Random(seed)
    parts: list[np.ndarray] = []
    spans: list[tuple[float, float]] = []
    t = lead_in
    if lead_in > 0:
        parts.append(np.zeros(int(lead_in * sr), dtype=np.float32))
    for i, clip in enumerate(clips):
        if i > 0:
            gap = turn_gap(texts[i - 1], texts[i], rng)
            parts.append(np.zeros(int(gap * sr), dtype=np.float32))
            t += gap
        parts.append(clip)
        spans.append((t, t + len(clip) / sr))
        t += len(clip) / sr
    audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    return StitchResult(audio=audio, spans=spans, duration=len(audio) / sr)


def sting(sr: int, seconds: float = 2.2, rising: bool = True) -> np.ndarray:
    """A short, soft synthesised intro/outro chime (no licensing issues, no asset files)."""
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    notes = [523.25, 659.25, 783.99, 1046.5] if rising else [1046.5, 783.99, 659.25, 523.25]
    out = np.zeros_like(t, dtype=np.float32)
    step = seconds / (len(notes) + 1)
    for k, f in enumerate(notes):
        start = k * step
        env = np.clip((t - start) / 0.02, 0, 1) * np.exp(-np.clip(t - start, 0, None) * 3.0) * (t >= start)
        out += (0.5 * np.sin(2 * np.pi * f * t) + 0.15 * np.sin(2 * np.pi * 2 * f * t)) * env
    out *= 10 ** (-26 / 20) / (np.sqrt(np.mean(out**2)) + 1e-9)
    return fade(out.astype(np.float32), sr, 30)


def concat_segments(segment_audio: list[np.ndarray], sr: int, gap_s: float = 0.9, with_stings: bool = True) -> tuple[np.ndarray, list[float]]:
    """Join segments with a longer topic-change pause; returns audio and each segment's start offset."""
    parts: list[np.ndarray] = []
    offsets: list[float] = []
    t = 0.0
    if with_stings:
        s = sting(sr)
        parts += [s, np.zeros(int(0.3 * sr), dtype=np.float32)]
        t += len(s) / sr + 0.3
    for i, a in enumerate(segment_audio):
        if i > 0:
            parts.append(np.zeros(int(gap_s * sr), dtype=np.float32))
            t += gap_s
        offsets.append(t)
        parts.append(a)
        t += len(a) / sr
    if with_stings:
        parts += [np.zeros(int(0.6 * sr), dtype=np.float32), sting(sr, rising=False)]
    audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    return limit(audio), offsets


def encode_mp3(audio: np.ndarray, sr: int, bitrate: str = "96k") -> bytes:
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    proc = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
         "-codec:a", "libmp3lame", "-b:a", bitrate, "-f", "mp3", "pipe:1"],
        input=pcm, capture_output=True, timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError("mp3 encode failed: " + proc.stderr.decode(errors="ignore")[:200])
    return proc.stdout
