"""
Вся обработка видео: crop → ASR → перевод → TTS → баннер.
Держим отдельно от bot.py, чтобы можно было гонять и тестировать
из командной строки без Telegram:

    python pipeline.py source.mp4 output.mp4
"""

import subprocess
import json
import sys
from pathlib import Path

from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
from gtts import gTTS

_whisper_model = None


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        # "small" — норм баланс скорости/качества для CPU на бесплатном Railway-тире
        _whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
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


def crop_to_9x16(src: Path, dst: Path):
    """Если видео не 9:16 — добавляет блюр-подложку по бокам/сверху-снизу."""
    w, h = _ffprobe_dims(src)
    target_ratio = 9 / 16
    if abs((w / h) - target_ratio) < 0.01:
        # уже вертикальное — просто копируем
        subprocess.run(["ffmpeg", "-y", "-i", str(src), "-c", "copy", str(dst)], check=True)
        return

    filt = (
        "[0:v]split=2[bg][fg];"
        "[bg]scale=1080:1920,gblur=sigma=30,eq=brightness=-0.05[blurred];"
        "[fg]scale=1080:-2[fg_scaled];"
        "[blurred][fg_scaled]overlay=(W-w)/2:(H-h)/2[out]"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-filter_complex", filt,
         "-map", "[out]", "-map", "0:a", "-c:v", "libx264", "-preset", "fast",
         "-crf", "20", "-c:a", "aac", str(dst)],
        check=True,
    )


def transcribe_zh(video_path: Path) -> str:
    model = _get_whisper()
    segments, _info = model.transcribe(str(video_path), language="zh")
    return "".join(seg.text for seg in segments).strip()


def translate_zh_to_ru(text: str) -> str:
    if not text:
        return ""
    return GoogleTranslator(source="zh-CN", target="ru").translate(text)


def synthesize_ru(text: str, out_mp3: Path):
    if not text:
        # тишина, если распознать не получилось
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
             "-t", "3", str(out_mp3)], check=True,
        )
        return
    tts = gTTS(text=text, lang="ru")
    tts.save(str(out_mp3))


def mux_new_audio(video_path: Path, audio_path: Path, out_path: Path):
    """Заменяет звук в видео на новую озвучку, растягивая её под длину ролика."""
    video_dur = _ffprobe_duration(video_path)
    audio_dur = _ffprobe_duration(audio_path)
    tempo = max(0.5, min(2.0, audio_dur / video_dur)) if video_dur > 0 else 1.0

    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-i", str(audio_path),
         "-filter:a", f"atempo={tempo:.3f}",
         "-map", "0:v", "-map", "1:a",
         "-c:v", "copy", "-shortest", str(out_path)],
        check=True,
    )


def insert_banner(video_path: Path, banner_path: Path, out_path: Path,
                   chroma_color: str = "0x00FF00", similarity: float = 0.12,
                   blend: float = 0.08):
    """
    Накладывает баннер ПОВЕРХ видео (не встык), убирая зелёный фон (chromakey)
    и растягивая баннер на весь кадр 1080x1920.

    Если ролик > 60 сек — баннер накладывается в середину каждой минуты
    (по правилам SkinHouse). Звук базового видео на это время глушится,
    звук баннера подмешивается вместо него.

    chroma_color / similarity / blend — подстрой под свой конкретный баннер,
    если ключинг съедает часть картинки или оставляет зелёную окантовку:
      - similarity выше = вырезает больше оттенков зелёного (но может задеть сам объект)
      - blend выше = мягче края (меньше "рваного" контура)
    """
    dur = _ffprobe_duration(video_path)
    banner_dur = _ffprobe_duration(banner_path)

    # точки вставки: середина 0-60с, середина 60-120с, и т.д.
    insert_times = []
    minute_start = 0.0
    while minute_start < dur:
        minute_end = min(minute_start + 60, dur)
        mid = minute_start + (minute_end - minute_start) / 2
        mid = min(mid, max(0.0, dur - banner_dur))  # чтобы баннер не вылез за конец ролика
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
            # increase + crop = баннер растягивается на весь кадр без полей по бокам
            # (лишнее по краям обрезается, а не остаётся чёрным/прозрачным)
            f"scale=1080:1920:force_original_aspect_ratio=increase,"
            f"crop=1080:1920,"
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
         "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-c:a", "aac",
         str(out_path)],
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
