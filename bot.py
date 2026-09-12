"""
Telegram-бот: перегон китайских вирусных клипов под RU-TikTok
================================================================

Что делает:
1. Принимает от тебя .mp4 (китайский оригинал, любое соотношение сторон)
2. Приводит видео к 9:16 (блюр-подложка по бокам, если исходник 16:9)
3. Распознаёт китайскую речь (faster-whisper, работает бесплатно и локально)
4. Переводит текст на русский (Google Translate через deep-translator, бесплатно)
5. Озвучивает перевод: Fish Audio (если есть ключ), иначе Edge TTS (Microsoft,
   бесплатно и без ключей), иначе gTTS
6. Склеивает результат: новое видео + новая аудиодорожка
7. Врезает твой баннер (banners/banner.mp4) в середину ролика
   (если ролик длиннее 60 сек — вставляет баннер в середину КАЖДОЙ минуты)
8. Отправляет готовый файл тебе обратно в Telegram

Деплой: Railway / Render / Fly.io (бесплатный тир, инструкция в README.md)

Переменные окружения (задаются в Railway → Variables):
    BOT_TOKEN        — токен от @BotFather (обязательно)
    OWNER_CHAT_ID    — твой telegram chat id, бот отвечает только тебе (опционально)
    FISH_API_KEY     — ключ Fish Audio для озвучки клоном (опционально)
    FISH_VOICE_ID    — id голоса в Fish Audio (опционально)
    EDGE_VOICE       — голос Edge TTS, по умолчанию ru-RU-DmitryNeural (опционально)
"""

import os
import subprocess
import logging
import tempfile
from pathlib import Path

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    CommandHandler,
    filters,
)

from pipeline import process_video  # вся тяжёлая логика вынесена сюда

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dub_bot")

BOT_TOKEN = os.environ["BOT_TOKEN"]
OWNER_CHAT_ID = os.environ.get("OWNER_CHAT_ID")  # если задан — бот игнорит всех остальных
BANNER_PATH = Path(__file__).parent / "banners" / "banner.mp4"


def _allowed(update: Update) -> bool:
    if not OWNER_CHAT_ID:
        return True
    return str(update.effective_chat.id) == str(OWNER_CHAT_ID)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Скинь мне китайский .mp4 — верну готовый ролик 9:16 с русской "
        "озвучкой и вшитым баннером.\n\n"
        "Команда /banner — обновить файл баннера (пришли видео с подписью /banner)."
    )


async def set_banner(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _allowed(update):
        return
    video = update.message.video or update.message.document
    if not video:
        await update.message.reply_text("Пришли видео-баннер файлом вместе с командой /banner")
        return
    file = await context.bot.get_file(video.file_id)
    BANNER_PATH.parent.mkdir(exist_ok=True)
    await file.download_to_drive(str(BANNER_PATH))
    await update.message.reply_text("Баннер обновлён, заебись.")


async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _allowed(update):
        return

    video = update.message.video or update.message.document
    if not video:
        return

    status_msg = await update.message.reply_text("Принял. Начинаю обработку — это пара минут…")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        src_path = tmp / "source.mp4"
        out_path = tmp / "output.mp4"

        tg_file = await context.bot.get_file(video.file_id)
        await tg_file.download_to_drive(str(src_path))

        try:
            await status_msg.edit_text("Кроплю под 9:16 и распознаю речь…")
            zh_text, ru_text = process_video(
                src_path=src_path,
                out_path=out_path,
                banner_path=BANNER_PATH if BANNER_PATH.exists() else None,
                progress_cb=lambda msg: None,  # можно прокинуть live-обновления статуса
            )
        except Exception as e:
            log.exception("pipeline failed")
            await status_msg.edit_text(f"Хуйня, что-то сломалось: {e}")
            return

        await status_msg.edit_text("Готово, заливаю обратно…")
        caption = f"🇨 {zh_text[:200]}\n\n🇷 {ru_text[:200]}" if zh_text else None
        try:
            with open(out_path, "rb") as f:
                await update.message.reply_video(video=f, caption=caption)
            await status_msg.delete()
        except Exception as e:
            log.exception("upload failed")
            await status_msg.edit_text(f"Загрузка не удалась: {e}")


def main():
    app = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .read_timeout(300)
        .write_timeout(300)
        .media_write_timeout(900)
        .build()
    )
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.CaptionRegex(r"^/banner") & (filters.VIDEO | filters.Document.VIDEO), set_banner))
    app.add_handler(MessageHandler(filters.VIDEO | filters.Document.VIDEO, handle_video))
    log.info("Бот запущен")
    app.run_polling()


if __name__ == "__main__":
    main()