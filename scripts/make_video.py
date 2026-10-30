"""Build the hackathon video locally from per-beat voice recordings and the Video Board.

Pipeline (all offline, nothing uploaded):
  1. voice clips (one per beat, sorted by file name) -> decoded to 48 kHz mono
  2. faster-whisper word timestamps -> filler words ("um", "uh", ...) cut out,
     long pauses shortened, head/tail silence trimmed, 12 ms crossfades at every cut
  3. RNNoise denoise (default; --no-denoise / --mix), high-pass, loudness normalised to -16 LUFS
  4. one slide per board section, captured with Edge headless at 1920 px wide
     (tall sections scroll slowly over the beat)
  5. one MP4 segment per beat (H.264 1080p30 + AAC), concatenated -> one video

Usage
  python -P -E scripts/make_video.py                       # real recordings in final_submission/voice/
  python -P -E scripts/make_video.py --tts                 # synthetic Windows voice from the script (preview)
  python -P -E scripts/make_video.py --voice DIR --out DIR --model small.en --no-denoise --mix 0.8 --no-clean
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BOARD = ROOT / "final_submission" / "video_board.html"
SCRIPT_MD = ROOT / "final_submission" / "video_script.md"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
SR = 48_000
N_SECTIONS = 7
AUDIO_EXT = {".m4a", ".mp3", ".wav", ".webm", ".ogg", ".aac", ".mp4", ".opus", ".flac"}
FILLERS = {"um", "umm", "uh", "uhh", "uhm", "erm", "er", "hmm", "hm", "mm", "mhm", "ah", "eh"}


def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd[0]}\n{r.stderr[-2000:]}")
    return r


# ---------------------------------------------------------------- audio I/O
def decode(path: Path) -> np.ndarray:
    r = subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
        capture_output=True,
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode(errors="replace")[-1000:])
    return np.frombuffer(r.stdout, dtype=np.float32).copy()


def write_wav(path: Path, x: np.ndarray) -> None:
    subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-", str(path)],
        input=x.astype(np.float32).tobytes(),
        check=True,
    )


# ---------------------------------------------------------------- cleaning
def transcribe_words(wav: Path, model_name: str):
    from faster_whisper import WhisperModel

    """Two passes. The plain pass is the authority on real words; a pass primed with filler
    words finds extra "um"s, but a filler is only accepted where the plain pass hears no real
    word (the primer otherwise makes Whisper relabel real speech as "Um")."""

    def go(device: str, compute: str):
        model = WhisperModel(model_name, device=device, compute_type=compute)
        out = []
        for prompt in (None, "Um, uh, so, hmm... I mean, like, you know, erm."):
            segments, _ = model.transcribe(
                str(wav), word_timestamps=True, initial_prompt=prompt,
                condition_on_previous_text=False, vad_filter=False,
            )
            out.append([(w.start, w.end, w.word) for seg in segments for w in (seg.words or [])])
        return out

    global _WHISPER_DEVICE
    plain = primed = None
    if _WHISPER_DEVICE != "cpu":
        try:
            plain, primed = go("cuda", "float16")
            _WHISPER_DEVICE = "cuda"
        except Exception as exc:  # missing CUDA runtime libraries etc.
            print(f"  whisper: GPU unavailable ({str(exc)[:60]}), using CPU")
            _WHISPER_DEVICE = "cpu"
    if plain is None:
        plain, primed = go("cpu", "int8")

    real = [(s, e) for s, e, w in plain if not is_filler(w)]
    words = list(plain)
    for s, e, w in primed:
        if not is_filler(w):
            continue
        if any(rs < e - 0.05 and re_ > s + 0.05 for rs, re_ in real):
            continue  # overlaps a real word in the plain pass: not a filler
        if any(is_filler(pw) and ps < e and pe > s for ps, pe, pw in plain):
            continue  # already found by the plain pass
        words.append((s, e, w))
    words.sort()
    return words


def is_filler(word: str) -> bool:
    return re.sub(r"[^a-z]", "", word.lower()) in FILLERS


_WHISPER_DEVICE = "auto"


def silence_spans(x: np.ndarray, frame_s: float = 0.02) -> list[tuple[float, float]]:
    n = int(frame_s * SR)
    k = len(x) // n
    rms = np.sqrt((x[: k * n].reshape(k, n) ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    floor = np.percentile(db, 10)
    thr = max(-48.0, floor + 8.0)
    quiet = db < thr
    spans, start = [], None
    for i, q in enumerate(quiet):
        if q and start is None:
            start = i
        elif not q and start is not None:
            spans.append((start * frame_s, i * frame_s))
            start = None
    if start is not None:
        spans.append((start * frame_s, k * frame_s))
    return spans


def plan_cuts(x: np.ndarray, words, *, max_pause=0.9, keep_pause=0.55, head=0.25, tail=0.4):
    """Return (cut spans, report). Each span is (start, end) in seconds to remove."""
    total = len(x) / SR
    cuts, fillers = [], []
    for s, e, w in words:
        if is_filler(w):
            cuts.append((max(0.0, s - 0.03), min(total, e + 0.03)))
            fillers.append((round(s, 2), w.strip()))
    pauses = 0.0
    for s, e in silence_spans(x):
        if s < 0.05:  # leading silence
            if e > head:
                cuts.append((0.0, e - head))
        elif e > total - 0.05:  # trailing silence
            if total - s > tail:
                cuts.append((s + tail, total))
        elif e - s > max_pause:
            mid = (s + e) / 2
            half = (e - s - keep_pause) / 2
            cuts.append((mid - half, mid + half))
            pauses += e - s - keep_pause
    cuts.sort()
    merged: list[list[float]] = []
    for s, e in cuts:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged], {"fillers": fillers, "pause_seconds_removed": round(pauses, 2)}


def apply_cuts(x: np.ndarray, cuts, fade_s: float = 0.012) -> np.ndarray:
    if not cuts:
        return x
    f = int(fade_s * SR)
    ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
    out = np.zeros(0, dtype=np.float32)
    pos = 0
    for s, e in cuts:
        a, b = int(s * SR), int(e * SR)
        piece = x[pos:a]
        out = crossfade_join(out, piece, ramp)
        pos = b
    out = crossfade_join(out, x[pos:], ramp)
    return out


def crossfade_join(a: np.ndarray, b: np.ndarray, ramp: np.ndarray) -> np.ndarray:
    f = len(ramp)
    if len(a) < f or len(b) < f:
        return np.concatenate([a, b])
    mixed = a[-f:] * (1 - ramp) + b[:f] * ramp
    return np.concatenate([a[:-f], mixed, b[f:]])


RNNOISE_MODEL = ROOT / "scripts" / "rnnoise" / "cb.rnnn"  # public-domain model from rnnoise-models


def denoise(src: Path, dst: Path, mix: float) -> None:
    """RNNoise (speech-trained recurrent net) via ffmpeg's arnndn; runs before transcription
    and cutting so the pause detector sees the cleaned signal. mix=1 full strength."""
    model = RNNOISE_MODEL.relative_to(ROOT).as_posix()  # a drive-letter colon breaks the filter string
    run([ffmpeg_exe(), "-v", "error", "-y", "-i", str(src), "-ac", "1", "-ar", str(SR),
         "-af", f"arnndn=m={model}:mix={mix}", str(dst)], cwd=ROOT)


def master(src: Path, dst: Path, target_lufs: float = -16.0) -> None:
    """High-pass, then a single static gain to the target loudness. (ffmpeg's one-pass loudnorm
    is a dynamic normaliser: it pumps the gaps between words up and makes room noise louder.)"""
    probe = subprocess.run(
        [ffmpeg_exe(), "-i", str(src), "-af", "highpass=f=80,loudnorm=print_format=json", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    mjson = probe.stderr[probe.stderr.rfind("{"):probe.stderr.rfind("}") + 1]
    measured = float(json.loads(mjson)["input_i"])
    gain = target_lufs - measured
    chain = ["highpass=f=80", f"volume={gain:.2f}dB", "alimiter=limit=0.891:level=false:attack=5:release=50"]  # -1 dBTP ceiling
    run([ffmpeg_exe(), "-v", "error", "-y", "-i", str(src), "-af", ",".join(chain), "-ar", str(SR), str(dst)])


def duration(path: Path) -> float:
    return len(decode(path)) / SR


# ---------------------------------------------------------------- slides
def capture_slides(out_dir: Path) -> list[Path]:
    html = BOARD.read_text(encoding="utf-8")
    slides = []
    with tempfile.TemporaryDirectory(prefix="iaa_slides_") as td:
        for i in range(1, N_SECTIONS + 1):
            hide = ",".join(f"section#s{j}" for j in range(1, N_SECTIONS + 1) if j != i)
            style = (
                f"<style>nav,footer{{display:none!important}}{hide}{{display:none!important}}"
                f"section,main{{min-height:0!important}}body{{padding-top:24px!important}}</style>"
            )
            page = Path(td) / f"s{i}.html"
            page.write_text(html.replace("</head>", style + "</head>", 1) if "</head>" in html else style + html,
                            encoding="utf-8")
            raw = Path(td) / f"s{i}.png"
            run([
                str(EDGE), "--headless", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
                "--force-color-profile=srgb", f"--user-data-dir={td}\\profile", "--window-size=1280,5000",
                "--force-device-scale-factor=1.5", f"--screenshot={raw}", page.as_uri(),
            ])
            # Edge returns before the file is flushed; wait for it to appear and settle.
            size, waited = -1, 0.0
            while waited < 90:
                time.sleep(0.5)
                waited += 0.5
                if raw.exists() and raw.stat().st_size == size and size > 0:
                    break
                size = raw.stat().st_size if raw.exists() else -1
            else:
                raise RuntimeError(f"Edge produced no screenshot for section {i}")
            img = Image.open(raw).convert("RGB")
            arr = np.asarray(img)
            bg = arr[-1, -1]
            rows = np.any(np.abs(arr.astype(int) - bg.astype(int)).sum(axis=2) > 24, axis=1)
            last = int(np.max(np.nonzero(rows)[0])) if rows.any() else arr.shape[0]
            h = max(1080, min(arr.shape[0], last + 90))
            slide = out_dir / f"slide_{i}.png"
            img.crop((0, 0, 1920, h)).save(slide)
            slides.append(slide)
    return slides


def bg_hex(slide: Path) -> str:
    r, g, b = Image.open(slide).convert("RGB").getpixel((5, 5))
    return f"0x{r:02x}{g:02x}{b:02x}"


# ---------------------------------------------------------------- video
def render_segment(slide: Path, wav: Path, dst: Path, gap: float = 0.6) -> float:
    d = duration(wav)
    h = Image.open(slide).height
    scroll = max(0, h - 1080)
    hold = 1.2
    span = max(d - 2 * hold, 0.1)
    y = f"'{scroll}*clip((t-{hold})/{span:.3f},0,1)'" if scroll else "0"
    vf = f"[0:v]pad=1920:{max(h, 1080)}:0:0:{bg_hex(slide)},crop=1920:1080:0:{y},format=yuv420p[v];[1:a]apad=pad_dur={gap}[a]"
    run([
        ffmpeg_exe(), "-v", "error", "-y", "-loop", "1", "-framerate", "30", "-i", str(slide), "-i", str(wav),
        "-filter_complex", vf, "-map", "[v]", "-map", "[a]", "-t", f"{d + gap:.3f}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-r", "30",
        "-c:a", "aac", "-b:a", "160k", "-ar", str(SR), "-movflags", "+faststart", str(dst),
    ])
    return d + gap


def concat(segments: list[Path], dst: Path) -> None:
    lst = dst.with_suffix(".txt")
    lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in segments), encoding="utf-8")
    run([ffmpeg_exe(), "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(dst)])
    lst.unlink()


# ---------------------------------------------------------------- TTS preview
def say_lines() -> list[str]:
    """The 'Say' column of the script table; cells may span several lines while being edited."""
    text = SCRIPT_MD.read_text(encoding="utf-8")
    body = text.split("Notes for recording")[0]
    chunks = re.split(r"\n(?=\|\s*\d+\s*\|)", body)
    out = []
    for ch in chunks:
        if not re.match(r"\|\s*\d+\s*\|", ch):
            continue
        cells = ch.split("|")
        if len(cells) < 5:
            continue
        out.append(re.sub(r"\s+", " ", cells[4]).strip())
    return out


def tts(out_dir: Path, inject_fillers: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, text in enumerate(say_lines(), 1):
        if inject_fillers and i in (2, 5):
            text = text.replace(". ", ". Um, ", 1).replace(", ", ", uh, ", 1)
        txt = out_dir / f"beat{i}.txt"
        txt.write_text(text, encoding="utf-8")
        wav = out_dir / f"beat{i}.wav"
        ps = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$s.SelectVoice('Microsoft Zira Desktop'); $s.Rate = 0; "
            f"$s.SetOutputToWaveFile('{wav}'); "
            f"$s.Speak([IO.File]::ReadAllText('{txt}')); $s.Dispose()"
        )
        run(["powershell", "-NoProfile", "-Command", ps])


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default=str(ROOT / "final_submission" / "voice"))
    ap.add_argument("--out", default=str(ROOT / "final_submission" / "video"))
    ap.add_argument("--model", default="small.en")
    ap.add_argument("--no-denoise", action="store_true", help="skip the RNNoise pass (on by default)")
    ap.add_argument("--mix", type=float, default=1.0, help="RNNoise strength 0-1 (1 = full)")
    ap.add_argument("--no-clean", action="store_true", help="skip filler/pause removal")
    ap.add_argument("--tts", action="store_true", help="synthesise the script with a Windows voice first")
    ap.add_argument("--name", default=None, help="output file name (default iaa_video.mp4 or preview_tts.mp4)")
    a = ap.parse_args()

    voice = Path(a.voice)
    out = Path(a.out)
    if a.tts:
        voice = out / "voice_tts"
        tts(voice, inject_fillers=True)
    clips = sorted(p for p in voice.iterdir() if p.suffix.lower() in AUDIO_EXT) if voice.exists() else []
    if not clips:
        print(f"no voice clips in {voice} (expected {N_SECTIONS} files, sorted by name)")
        return 1
    if len(clips) != N_SECTIONS:
        print(f"warning: {len(clips)} clips for {N_SECTIONS} sections; using the first {min(len(clips), N_SECTIONS)}")
    clips = clips[:N_SECTIONS]

    work = out / "work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    report = {"clips": [], "total_seconds": 0.0}

    print("slides ...")
    slides = capture_slides(work)

    segs = []
    for i, clip in enumerate(clips, 1):
        if not a.no_denoise:
            dn = work / f"dn_{i}.wav"
            denoise(clip, dn, a.mix)
            clip = dn
        x = decode(clip)
        raw = work / f"raw_{i}.wav"
        write_wav(raw, x)
        rep = {"file": clip.name, "raw_seconds": round(len(x) / SR, 2)}
        if not a.no_clean:
            words = transcribe_words(raw, a.model)
            cuts, info = plan_cuts(x, words)
            x = apply_cuts(x, cuts)
            rep.update(info)
            rep["transcript"] = "".join(w for _, _, w in words).strip()
        cut = work / f"cut_{i}.wav"
        write_wav(cut, x)
        clean = work / f"clean_{i}.wav"
        master(cut, clean)
        seg = work / f"seg_{i}.mp4"
        d = render_segment(slides[i - 1], clean, seg)
        rep["clean_seconds"] = round(d, 2)
        report["clips"].append(rep)
        report["total_seconds"] += d
        segs.append(seg)
        print(f"  beat {i}: {rep['raw_seconds']}s -> {rep['clean_seconds']}s"
              + (f", fillers removed {len(rep.get('fillers', []))}" if not a.no_clean else ""))

    name = a.name or ("preview_tts.mp4" if a.tts else "iaa_video.mp4")
    final = out / name
    concat(segs, final)
    report["total_seconds"] = round(report["total_seconds"], 1)
    report["output"] = str(final)
    (out / final.stem).with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    m, s = divmod(int(report["total_seconds"]), 60)
    print(f"written {final}  ({m}:{s:02d})" + ("  ** over 5:00 **" if report["total_seconds"] > 300 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
