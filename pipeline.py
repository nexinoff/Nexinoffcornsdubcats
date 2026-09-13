"""
Вся обработка видео: crop → ASR → LLM-редактура → TTS по таймкодам → баннер.
Держим отдельно от bot.py, чтобы можно было гонять и тестировать
из командной строки без Telegram:

    python pipeline.py source.mp4 output.mp4

ASR: если есть GROQ_API_KEY — whisper-large-v3-turbo через Groq (телефон не грузится),
иначе whisper.cpp (Termux) или faster-whisper (сервер).
Озвучка: ТОЛЬКО edge-tts, с тремя попытками против глюков майкрософта.
LLM: Groq переписывает перевод под контекст и под длину таймкода.
Звук: сплошная лента без дыр (склейка GLUE), темп через VOICE_SPEED.
Фон: BG_MODE=blur (блюр по бокам) или black (чёрные полосы).
"""

import subprocess
import json
import sys
import os
import re
import time
import shutil
from pathlib import Path
from urllib.parse import quote

import requests

if shutil.which("ffmpeg") is None:
    try:
        import static_ffmpeg
        static_ffmpeg.add_paths()
    except Exception as _e:
        print("static_ffmpeg init failed:", _e)

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None
try:
    from deep_translator import MyMemoryTranslator
except Exception:
    MyMemoryTranslator = None

EDGE_VOICE = os.environ.get("EDGE_VOICE", "ru-RU-DmitryNeural")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
WHISPER_CPP = os.environ.get("WHISPER_CPP")
WHISPER_CPP_MODEL = os.environ.get("WHISPER_CPP_MODEL")
VOICE_SPEED = float(os.environ.get("VOICE_SPEED", "1.2"))
BG_MODE = os.environ.get("BG_MODE", "blur")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
CHARS_PER_SEC = float(os.environ.get("CHARS_PER_SEC", "12.0"))
MAX_SILENCE = float(os.environ.get("MAX_SILENCE", "0.7"))
GLUE = float(os.environ.get("GLUE", "0.3"))

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}

_whisper_model = None
_edge_token_cache = {"tok": "", "exp": 0.0}
_eng_errors = []


def _log_err(name, e):
    code = getattr(getattr(e, "response", None), "status_code", None)
    _eng_errors.append(f"{name}:{code or type(e).__name__}")


def _get_whisper():
    global _whisper_model
    if _whisper_model is None and WhisperModel is not None:
        _whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    return _whisper_model


def _cpp_ready() -> bool:
    return bool(WHISPER_CPP and WHISPER_CPP_MODEL
                and Path(WHISPER_CPP).exists() and Path(WHISPER_CPP_MODEL).exists())


def _lat_ratio(t: str) -> float:
    """Доля латинских букв среди всех букв — детектор английской бурмалды."""
    letters = [c for c in t if c.isalpha()]
    if not letters:
        return 0.0
    lat = sum(1 for c in letters if "a" <= c.lower() <= "z")
    return lat / len(letters)


def _ffprobe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(out.stdout)["format"]["duration"])


def _audio_duration(path: Path) -> float:
    """Длительность звуковой дорожки; 0.0 если дорожки нет вообще."""
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams",
         "-select_streams", "a:0", str(path)],
        capture_output=True, text=True, check=True,
    )
    streams = json.loads(out.stdout).get("streams", [])
    if not streams:
        return 0.0
    return float(streams[0].get("duration", 0) or 0)


def _transcribe_groq(video_path: Path):
    """whisper-large-v3-turbo через Groq: телефон только отправляет mp3."""
    work = video_path.parent
    mp3 = work / "asr.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "libmp3lame", "-b:a", "64k", str(mp3)],
        capture_output=True, check=True,
    )
    with open(mp3, "rb") as f:
        r = requests.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files={"file": ("asr.mp3", f, "audio/mpeg")},
            data={"model": "whisper-large-v3-turbo", "language": "zh",
                  "response_format": "verbose_json",
                  "timestamp_granularities[]": "segment"},
            timeout=300,
        )
    r.raise_for_status()
    data = r.json()
    segs = []
    for item in data.get("segments", []):
        t = (item.get("text") or "").strip()
        if t:
            segs.append((float(item.get("start", 0)), float(item.get("end", 0)), t))
    # предохранитель: если приехали миллисекунды — нормализуем в секунды
    if segs:
        vd = _ffprobe_duration(video_path)
        if max(e for _s, e, _t in segs) > vd * 2:
            segs = [(s / 1000.0, e / 1000.0, t) for s, e, t in segs]
    return segs


def _llm_rephrase(zh: str, ru: str, budget: int) -> str:
    """Groq-ллм переписывает черновик живо и укладывает в бюджет символов."""
    if not GROQ_API_KEY:
        return ru
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                     "Content-Type": "application/json"},
            json={
                "model": GROQ_MODEL,
                "temperature": 0.4,
                "max_tokens": 500,
                "messages": [
                    {"role": "system", "content":
                     "Ты редактор закадрового перевода китайских видео на русский. "
                     "Пиши живо, разговорно, по-пацански, без канцелярита. "
                     "Верни ТОЛЬКО текст перевода, без пояснений и кавычек."},
                    {"role": "user", "content":
                     f"Оригинал (zh): {zh}\nЧерновик (ru): {ru}\n"
                     f"Уложись строго в {budget} символов, сохрани смысл и эмоцию."},
                ],
            },
            timeout=45,
        )
        r.raise_for_status()
        out = r.json()["choices"][0]["message"]["content"].strip()
        if out:
            return out
    except Exception as e:
        _log_err("groq-llm", e)
    return ru


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
    """9:16 (720x1280): BG_MODE=blur — блюр-фон, BG_MODE=black — чёрные полосы."""
    w, h = _ffprobe_dims(src)

    pre = ""
    cw, ch = w, h
    c = _detect_crop(src)
    if c:
        cw, ch, cx, cy = (int(x) for x in c.split(":"))
        if cw * ch < 0.9 * w * h:
            pre = f"crop={c},"
        else:
            cw, ch = w, h

    if BG_MODE == "black":
        filt = (
            f"[0:v]{pre}scale=720:1280:force_original_aspect_ratio=decrease[fgs];"
            "color=black:720x1280[bgc];"
            "[bgc][fgs]overlay=(W-w)/2:(H-h)/2[out]"
        )
    else:
        filt = (
            f"[0:v]{pre}split=2[bg][fg];"
            "[bg]scale=360:640:force_original_aspect_ratio=increase,crop=360:640,"
            "gblur=sigma=18,scale=720:1280[blurred];"
            "[fg]scale=720:1280:force_original_aspect_ratio=decrease[fgs];"
            "[blurred][fgs]overlay=(W-w)/2:(H-h)/2[out]"
        )

    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-filter_complex", filt,
         "-map", "[out]", "-map", "0:a", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "28", "-threads", "2", "-c:a", "aac", str(dst)],
        check=True,
    )


def _clean_hallucination(text: str, limit: int = 1200) -> str:
    """Режет висперовую бурмалду: лимит длины и детектор зацикленных повторов."""
    if not text:
        return ""
    if len(text) > limit:
        text = text[:limit]
    chunk = text[:20]
    if chunk and text.count(chunk) > 5:
        return ""
    return text


def _transcribe_cpp(video_path: Path):
    """whisper.cpp на телефоне: wav 16k → JSON с таймкодами."""
    work = video_path.parent
    wav = work / "asr.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-ar", "16000", "-ac", "1",
         "-c:a", "pcm_s16le", str(wav)],
        capture_output=True, check=True,
    )
    base = work / "asr"
    subprocess.run(
        [WHISPER_CPP, "-m", WHISPER_CPP_MODEL, "-f", str(wav),
         "-l", "zh", "-oj", "-of", str(base), "-t", "4"],
        capture_output=True, check=True,
    )
    data = json.loads((work / "asr.json").read_text())
    segs = []
    for item in data.get("transcription", []):
        t = (item.get("text") or "").strip()
        if not t:
            continue
        off = item.get("offsets", {})
        segs.append((off.get("from", 0) / 1000.0, off.get("to", 0) / 1000.0, t))
    return segs


def _transcribe_fw(video_path: Path):
    model = _get_whisper()
    segments, _info = model.transcribe(
        str(video_path),
        language="zh",
        condition_on_previous_text=False,
        temperature=0.0,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
    )
    return [(s.start, s.end, (s.text or "").strip())
            for s in segments if (s.text or "").strip()]


def _chunk_segments(segs):
    chunks = []
    cur = None
    for st, en, t in segs:
        if cur is None:
            cur = [st, en, t]
        elif len(cur[2]) + len(t) <= 220 and (en - cur[0]) <= 12:
            cur[1] = en
            cur[2] += t
        else:
            chunks.append(tuple(cur))
            cur = [st, en, t]
    if cur:
        chunks.append(tuple(cur))
    return chunks


def transcribe_segments(video_path: Path):
    """ASR по приоритету: Groq large-v3-turbo → whisper.cpp → faster-whisper."""
    if GROQ_API_KEY:
        try:
            segs = _transcribe_groq(video_path)
            if segs:
                segs = [s for s in segs if _lat_ratio(s[2]) <= 0.5]
                return _chunk_segments(segs)
        except Exception as e:
            _log_err("groq-asr", e)
    if _cpp_ready():
        segs = _transcribe_cpp(video_path)
    else:
        segs = _transcribe_fw(video_path)
    segs = [s for s in segs if _lat_ratio(s[2]) <= 0.5]
    return _chunk_segments(segs)
def _edge_token() -> str:
    now = time.time()
    if _edge_token_cache["tok"] and now < _edge_token_cache["exp"]:
        return _edge_token_cache["tok"]
    r = requests.get("https://edge.microsoft.com/translate/auth", headers=UA, timeout=30)
    r.raise_for_status()
    tok = r.text.strip()
    _edge_token_cache["tok"] = tok
    _edge_token_cache["exp"] = now + 300
    return tok


def _edge_translate(chunk: str) -> str:
    """Бесплатный переводчик Microsoft Edge: токен без ключа."""
    try:
        tok = _edge_token()
        r = requests.post(
            "https://api-edge.cognitive.microsofttranslator.com/translate",
            params={"api-version": "3.0", "from": "zh-Hans", "to": "ru"},
            headers={**UA, "Authorization": f"Bearer {tok}",
                     "Content-Type": "application/json"},
            json=[{"Text": chunk}],
            timeout=60,
        )
        r.raise_for_status()
        return r.json()[0]["translations"][0]["text"]
    except Exception as e:
        _log_err("edge", e)
        return ""


def _gtx(chunk: str, client: str) -> str:
    """Прямой эндпоинт Google с разным client-ом."""
    try:
        r = requests.get(
            "https://translate.googleapis.com/translate_a/single",
            params={"client": client, "sl": "zh-CN", "tl": "ru", "dt": "t", "q": chunk},
            headers=UA, timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        return "".join(p[0] for p in data[0] if p and p[0])
    except Exception as e:
        _log_err(f"gtx-{client}", e)
        return ""


LINGVA_HOSTS = ["lingva.ml", "lingva.garudalinux.org", "lingva.lunar.icu"]


def _lingva(chunk: str) -> str:
    """Lingva-инстансы напрямую через HTTP, без библиотек."""
    for host in LINGVA_HOSTS:
        try:
            r = requests.get(f"https://{host}/api/v1/zh/ru/{quote(chunk)}",
                             headers=UA, timeout=30)
            r.raise_for_status()
            t = r.json().get("translation")
            if t:
                return t
        except Exception as e:
            _log_err("lingva", e)
    return ""


def _dt_google(chunk: str) -> str:
    if GoogleTranslator is None:
        return ""
    try:
        return GoogleTranslator(source="zh-CN", target="ru").translate(chunk) or ""
    except Exception as e:
        _log_err("dt-google", e)
        return ""


def _dt_mymemory(chunk: str) -> str:
    if MyMemoryTranslator is None:
        return ""
    try:
        return MyMemoryTranslator(source="zh-CN", target="ru").translate(chunk) or ""
    except Exception as e:
        _log_err("mymemory", e)
        return ""


def _translate_chunk(chunk: str) -> str:
    engines = (
        _edge_translate,
        lambda c: _gtx(c, "gtx"),
        lambda c: _gtx(c, "android"),
        _lingva,
        _dt_google,
        _dt_mymemory,
    )
    for eng in engines:
        try:
            r = eng(chunk)
            if r:
                return r
        except Exception as e:
            _log_err("wrap", e)
    return ""


def translate_zh_to_ru(text: str) -> str:
    if not text:
        return ""
    chunks = [text[i:i + 900] for i in range(0, len(text), 900)]
    parts = [_translate_chunk(ch) for ch in chunks]
    return " ".join(p for p in parts if p).strip()


def synthesize_ru(text: str, out_mp3: Path) -> str:
    """Озвучка ТОЛЬКО edge-tts, с тремя попытками против глюков майкрософта."""
    last = None
    for attempt in range(3):
        try:
            subprocess.run(
                [sys.executable, "-m", "edge_tts", "--voice", EDGE_VOICE,
                 "--text", text, "--write-media", str(out_mp3)],
                check=True, timeout=120,
            )
            return "edge"
        except Exception as e:
            last = e
            time.sleep(3 + attempt * 3)
    raise RuntimeError(f"edge-tts сдох после трёх попыток: {last}")


def mux_segments(video_path: Path, items, out_path: Path):
    """Кладёт каждую озвучку в её таймкод, с микро-фейдами и ресемплом 44.1k."""
    video_dur = _ffprobe_duration(video_path)
    inputs = ["-i", str(video_path)]
    for _st, p, _t, _fd in items:
        inputs += ["-i", str(p)]

    chains = []
    labels = []
    for i, (st, _p, tempo, fd) in enumerate(items, start=1):
        ms = int(st * 1000)
        lab = f"a{i}"
        fade_out = max(0.0, fd - 0.08)
        chains.append(
            f"[{i}:a]atempo={tempo:.3f},aresample=44100,afade=t=in:d=0.06,"
            f"afade=t=out:d=0.08:st={fade_out:.3f},adelay={ms}|{ms}[{lab}]"
        )
        labels.append(f"[{lab}]")
    mix = f"{''.join(labels)}amix=inputs={len(items)}:normalize=0:duration=longest[aout]"
    fc = ";".join(chains + [mix])

    subprocess.run(
        ["ffmpeg", "-y", *inputs, "-filter_complex", fc,
         "-map", "0:v", "-map", "[aout]",
         "-c:v", "copy", "-t", str(video_dur), "-c:a", "aac", str(out_path)],
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
    _eng_errors.clear()

    cropped = work / "cropped.mp4"
    progress_cb("crop")
    crop_to_9x16(src_path, cropped)

    progress_cb("transcribe")
    chunks = transcribe_segments(cropped)
    zh_full = _clean_hallucination("".join(t for _s, _e, t in chunks))
    if not zh_full:
        raise RuntimeError("Распознавание не дало текста или дало бурмалду")

    progress_cb("translate+llm+tts")
    raw = []
    engine_used = ""
    for idx, (st, en, txt) in enumerate(chunks):
        ru = translate_zh_to_ru(txt)
        # английский мусор в переводе не озвучиваем
        if not ru or _lat_ratio(ru) > 0.4:
            continue
        span = max(1.0, en - st)
        budget = max(20, int(span * CHARS_PER_SEC))
        # ллм правит текст если он не ложится в таймкод
        if len(ru) > int(budget * 1.05) or len(ru) < int(budget * 0.75):
            ru = _llm_rephrase(txt, ru, budget)
        mp3 = work / f"seg_{idx}.mp3"
        eng = synthesize_ru(ru, mp3)
        engine_used = engine_used or eng
        dur = _ffprobe_duration(mp3)
        base = max(0.95, min(1.75, dur / span))
        tempo = min(2.0, base * VOICE_SPEED)
        raw.append([st, dur, mp3, ru, tempo])

    if not raw:
        raise RuntimeError("Перевод не удался, коды движков: " + "; ".join(_eng_errors[:8]))

    video_dur = _ffprobe_duration(cropped)

    def layout(k):
        """Сплошная лента: склейка GLUE, без дыр и без наложений."""
        t = raw[0][0]
        out = []
        for st, dur, mp3, ru, tempo in raw:
            fd = dur / (tempo * k)
            out.append((t, mp3, tempo * k, fd))
            t += fd + GLUE
        return out, t

    items, end_t = layout(1.0)
    if end_t > video_dur - 0.2:
        k = min(1.35, (end_t - raw[0][0]) / max(1.0, video_dur - 0.2 - raw[0][0]))
        items, end_t = layout(k)

    timing = " ".join(f"{s:.1f}+{f:.1f}" for s, _p, _t, f in items)

    dubbed = work / "dubbed.mp4"
    progress_cb("mux")
    mux_segments(cropped, items, dubbed)

    # предохранитель: тихую дорожку не отправляем, орем ошибкой
    ad = _audio_duration(dubbed)
    if ad < 2.0:
        raise RuntimeError(f"mux собрал тихую дорожку ({ad:.1f} c), видос не отправлю")

    if banner_path and banner_path.exists():
        progress_cb("banner")
        insert_banner(dubbed, banner_path, out_path)
    else:
        dubbed.rename(out_path)

    ru_full = f"[{engine_used}] " + " ".join(r[3] for r in raw) + f" || timing: {timing}"
    return zh_full, ru_full


if __name__ == "__main__":
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    zh, ru = process_video(src, out, banner_path=None, progress_cb=print)
    print("ZH:", zh)
    print("RU:", ru)