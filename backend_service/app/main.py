from fastapi import FastAPI
from contextlib import asynccontextmanager
from .db.db import Database
import logging
from .api.routes import router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

db = Database()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Контекстный менеджер для управления подключением к базе данных.

    Этот метод подключает базу данных в начале работы приложения и закрывает соединение в конце.
    """
    await db.connect()
    logger.info("Подключение к базе данных установлено.")

    app.state.db = db
    
    yield
    
    await db.close()
    logger.info("Соединение с базой данных закрыто.")

app = FastAPI(lifespan=lifespan)

app.include_router(router)
