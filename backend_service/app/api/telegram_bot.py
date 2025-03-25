import logging
from aiogram import Bot
from aiogram.client.default import DefaultBotProperties  # Новый способ задания параметров
from dotenv import load_dotenv
import os
# Настройки
load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = None  # Определяем автоматически

# Создаем объект бота
bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode="HTML"))


async def get_chat_id():
    """Получение ID чата из последних сообщений бота."""
    global CHAT_ID
    try:
        updates = await bot.get_updates()
        for update in updates:
            if update.message:
                CHAT_ID = update.message.chat.id
                logging.info(f"✅ Автоматически установлен Chat ID: {CHAT_ID}")
                return
        logging.warning("⚠️ Напишите что-нибудь боту, чтобы получить `CHAT_ID`.")
    except Exception as e:
        logging.error(f"❌ Ошибка при получении Chat ID: {str(e)}")


async def send_telegram_message(user_id: int, description: str):
    """Отправка сообщения в Telegram о распознанном пользователе."""
    global CHAT_ID
    if CHAT_ID is None:
        await get_chat_id()  # Получаем Chat ID, если его ещё нет

    if CHAT_ID:
        message = (
            f"👤 <b>Распознан пользователь!</b>\n"
            f"🆔 <b>ID:</b> {user_id}\n"
            f"🔍 <b>Описание:</b> {description}\n"
        )
        try:
            await bot.send_message(chat_id=CHAT_ID, text=message)
        except Exception as e:
            logging.error(f"❌ Ошибка при отправке в Telegram: {str(e)}")
    else:
        logging.error("⚠️ `CHAT_ID` не установлен. Напишите что-нибудь боту.")
