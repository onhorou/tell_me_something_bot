import asyncio
import os

from aiogram import Bot, Dispatcher, types
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()  # загружает переменные из .env в окружение

API_TOKEN = os.getenv("API_TOKEN")

bot = Bot(token=API_TOKEN)
dp = Dispatcher()

# Загружаем модель Hugging Face
emotion_model = pipeline(
    model="seara/rubert-tiny2-russian-emotion-detection-ru-go-emotions"
)

@dp.message_handler(commands=["start", "help"])
async def start_cmd(message: types.Message):
    await message.reply("Привет! Отправьте текст, и я покажу эмоции.")

@dp.message_handler()
async def handle_text(message: types.Message):
    result = emotion_model(message.text, top_k=5)
    text_result = "\n".join([f"{r['label']}: {r['score']:.2f}" for r in result])
    await message.reply(f"Ваши эмоции:\n{text_result}")

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())