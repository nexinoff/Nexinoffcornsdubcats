"""
Вся обработка видео: crop → ASR → перевод → TTS → баннер.
Держим отдельно от bot.py, чтобы можно было гонять и тестировать
из командной строки без Telegram:

    python pipeline.py source.mp4 output.mp4
"""

import subprocess
import json
import sys
import os
import re
from pathlib import Path

import requests
import static_ffmpeg
static_ffmpeg.add_paths()

from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator, MyMemoryTranslator, LingvaTranslator
from gtts import gTTS

FISH_API_KEY = os.environ.get("FISH_API_KEY")
FISH_VOICE_ID = os.environ.get("FISH_VOICE_ID")
EDGE_VOICE = os.environ.get("EDGE_VOICE", "ru-RU-DmitryNeural")

_whisper_model = None


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        # "base" — легче по памяти для бесплатного Railway-тира
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    return _whisper_model


def _ffprobe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(out.stdout)["format"]["duration"])


def _ffprobe_dims(path: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams",
         "-select_streams", "v:0", str(path)],
        capture_output=True, text=True, check=True,
    )
    s = json.loads(out.stdout)["streams"][0]
    return int(s["width"]), int(s["height"])


def _detect_crop(src: Path):
    """Ищет чёрные полосы и возвращает строку crop=W:H:X:Y или None."""
    out = subprocess.run(
        ["ffmpeg", "-i", str(src), "-vf", "cropdetect=24:2:0",
         "-frames:v", "50", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    crops = re.findall(r"crop=(\d+:\d+:\d+:\d+)", out.stderr)
    return crops[-1] if crops else None


def crop_to_9x16(src: Path, dst: Path):
    """Режет чёрные полосы, приводит к 9:16 (720x1280) и замазывает субтитры."""
    w, h = _ffprobe_dims(src)
    target_ratio = 9 / 16

    pre = ""
    cw, ch = w, h
    c = _detect_crop(src)
    if c:
        cw, ch, cx, cy = (int(x) for x in c.split(":"))
        if cw * ch < 0.9 * w * h:
            pre = f"crop={c},"
        else:
            cw, ch = w, h

    # куда ляжет контент в кадре 720x1280 — от этого считаем плашку субтитров
    r = min(720 / cw, 1280 / ch)
    fg_h = int(ch * r) // 2 * 2
    oy = (1280 - fg_h) // 2
    y0 = oy + int(fg_h * 0.78)
    h0 = fg_h - int(fg_h * 0.78)
    if h0 < 8:
        y0, h0 = 1272, 8
    box = f"drawbox=x=0:y={y0}:w=720:h={h0}:color=black@1:t=fill"

    if not pre and abs((w / h) - target_ratio) < 0.01:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src),
             "-vf", f"scale=720:1280,{box}",
             "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "28", "-threads", "2", "-c:a", "aac", str(dst)],
            check=True,
        )
        return

    filt = (
        f"[0:v]{pre}split=2[bg][fg];"
        "[bg]scale=360:640:force_original_aspect_ratio=increase,crop=360:640,"
        "gblur=sigma=8,scale=720:1280[blurred];"
        "[fg]scale=720:1280:force_original_aspect_ratio=decrease[fgs];"
        f"[blurred][fgs]overlay=(W-w)/2:(H-h)/2[ov];"
        f"[ov]{box}[out]"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-filter_complex", filt,
         "-map", "[out]", "-map", "0:a", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "28", "-threads", "2", "-c:a", "aac", str(dst)],
        check=True,
    )


def _clean_hallucination(text: str, limit: int = 1000) -> str:
    """Режет висперовую бурмалду: лимит длины и детектор зацикленных повторов."""
    if not text:
        return ""
    if len(text) > limit:
        text = text[:limit]
    chunk = text[:20]
    if chunk and text.count(chunk) > 5:
        return ""
    return text


def transcribe_zh(video_path: Path) -> str:
    model = _get_whisper()
    segments, _info = model.transcribe(
        str(video_path),
        language="zh",
        condition_on_previous_text=False,
        temperature=0.0,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
    )
    text = "".join(seg.text for seg in segments).strip()
    return _clean_hallucination(text)


def _google_gtx(chunk: str) -> str:
    """Прямой эндпоинт Google, который редко банят на серверных IP."""
    try:
        r = requests.get(
            "https://translate.googleapis.com/translate_a/single",
            params={"client": "gtx", "sl": "zh-CN", "tl": "ru", "dt": "t", "q": chunk},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        return "".join(p[0] for p in data[0] if p and p[0])
    except Exception:
        return ""


def _translate_chunk(chunk: str) -> str:
    r = _google_gtx(chunk)
    if r:
        return r
    engines = (
        lambda: GoogleTranslator(source="zh-CN", target="ru").translate(chunk),
        lambda: LingvaTranslator(source="zh", target="ru").translate(chunk),
        lambda: MyMemoryTranslator(source="zh-CN", target="ru").translate(chunk),
    )
    for eng in engines:
        try:
            r = eng()
            if r:
                return r
        except Exception:
            continue
    return ""


def translate_zh_to_ru(text: str) -> str:
    if not text:
        return ""
    chunks = [text[i:i + 900] for i in range(0, len(text), 900)]
    parts = [_translate_chunk(ch) for ch in chunks]
    out = " ".join(p for p in parts if p).strip()
    if not out:
        raise RuntimeError("Перевод не удался ни одним движком, текста нет")
    return out


def synthesize_ru(text: str, out_mp3: Path):
    if not text:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
             "-t", "3", str(out_mp3)], check=True,
        )
        return

    if FISH_API_KEY:
        try:
            resp = requests.post(
                "https://api.fish.audio/v1/tts",
                headers={
                    "Authorization": f"Bearer {FISH_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "text": text,
                    "reference_id": FISH_VOICE_ID,
                    "format": "mp3",
                    "model": "s2.1-pro-free",
                },
                timeout=120,
            )
            resp.raise_for_status()
            out_mp3.write_bytes(resp.content)
            return
        except Exception:
            pass  # fish не дал — уходим на edge-tts

    try:
        subprocess.run(
            [sys.executable, "-m", "edge_tts", "--voice", EDGE_VOICE,
             "--text", text, "--write-media", str(out_mp3)],
            check=True, timeout=120,
        )
        return
    except Exception:
        pass  # edge не дал — последний шанс gTTS

    tts = gTTS(text=text, lang="ru")
    tts.save(str(out_mp3))


def mux_new_audio(video_path: Path, audio_path: Path, out_path: Path):
    """Заменяет звук в видео на новую озвучку, растягивая её под длину ролика."""
    video_dur = _ffprobe_duration(video_path)
    audio_dur = _ffprobe_duration(audio_path)
    tempo = max(0.5, min(2.0, audio_dur / video_dur)) if video_dur > 0 else 1.0

    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-i", str(audio_path),
         "-filter:a", f"atempo={tempo:.3f},apad",
         "-map", "0:v", "-map", "1:a",
         "-c:v", "copy", "-t", str(video_dur), str(out_path)],
        check=True,
    )


def insert_banner(video_path: Path, banner_path: Path, out_path: Path,
                   chroma_color: str = "0x00FF00", similarity: float = 0.12,
                   blend: float = 0.08):
    dur = _ffprobe_duration(video_path)
    banner_dur = _ffprobe_duration(banner_path)

    insert_times = []
    minute_start = 0.0
    while minute_start < dur:
        minute_end = min(minute_start + 60, dur)
        mid = minute_start + (minute_end - minute_start) / 2
        mid = min(mid, max(0.0, dur - banner_dur))
        insert_times.append(mid)
        minute_start += 60

    n = len(insert_times)
    inputs = ["-i", str(video_path)]
    for _ in range(n):
        inputs += ["-i", str(banner_path)]

    video_chains = []
    for i, start in enumerate(insert_times, start=1):
        video_chains.append(
            f"[{i}:v]chromakey={chroma_color}:{similarity}:{blend},"
            f"scale=720:1280:force_original_aspect_ratio=increase,"
            f"crop=720:1280,"
            f"setpts=PTS+{start}/TB[bnr{i}]"
        )

    overlay_chain = []
    prev = "0:v"
    for i, start in enumerate(insert_times, start=1):
        label = f"vtmp{i}" if i < n else "vout"
        overlay_chain.append(
            f"[{prev}][bnr{i}]overlay=enable='between(t,{start},{start + banner_dur})'[{label}]"
        )
        prev = label

    audio_mute_chain = []
    prev_a = "0:a"
    for i, start in enumerate(insert_times, start=1):
        label = f"amute{i}"
        audio_mute_chain.append(
            f"[{prev_a}]volume=0:enable='between(t,{start},{start + banner_dur})'[{label}]"
        )
        prev_a = label
    base_audio_label = prev_a

    delay_chains = []
    banner_audio_labels = []
    for i, start in enumerate(insert_times, start=1):
        ms = int(start * 1000)
        label = f"bda{i}"
        delay_chains.append(f"[{i}:a]adelay={ms}|{ms}[{label}]")
        banner_audio_labels.append(f"[{label}]")

    amix_inputs = f"[{base_audio_label}]" + "".join(banner_audio_labels)
    amix_chain = f"{amix_inputs}amix=inputs={n + 1}:duration=first:normalize=0[aout]"

    filter_complex = ";".join(video_chains + overlay_chain + audio_mute_chain + delay_chains + [amix_chain])

    subprocess.run(
        ["ffmpeg", "-y", *inputs, "-filter_complex", filter_complex,
         "-map", "[vout]", "-map", "[aout]",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-threads", "2",
         "-c:a", "aac", str(out_path)],
        check=True,
    )


def process_video(src_path: Path, out_path: Path, banner_path: Path | None = None,
                   progress_cb=lambda msg: None):
    work = src_path.parent

    cropped = work / "cropped.mp4"
    progress_cb("crop")
    crop_to_9x16(src_path, cropped)

    progress_cb("transcribe")
    zh_text = transcribe_zh(cropped)

    progress_cb("translate")
    ru_text = translate_zh_to_ru(zh_text)

    progress_cb("tts")
    ru_mp3 = work / "ru_voice.mp3"
    synthesize_ru(ru_text, ru_mp3)

    dubbed = work / "dubbed.mp4"
    progress_cb("mux")
    mux_new_audio(cropped, ru_mp3, dubbed)

    if banner_path and banner_path.exists():
        progress_cb("banner")
        insert_banner(dubbed, banner_path, out_path)
    else:
        dubbed.rename(out_path)

    return zh_text, ru_text


if __name__ == "__main__":
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    zh, ru = process_video(src, out, banner_path=None, progress_cb=print)
    print("ZH:", zh)
    print("RU:", ru)