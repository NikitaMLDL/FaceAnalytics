from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
from .db.db import Database, CRMDatabase
from .api.routes import router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Основная база данных
db = Database()

# База данных CRM
crm_db = CRMDatabase()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Контекстный менеджер для управления подключениями к базам данных.

    Подключает обе базы данных при старте приложения и закрывает их при завершении.
    """
    await db.connect()
    await crm_db.connect()
    logger.info("Database connections established.")

    app.state.db = db
    app.state.crm_db = crm_db

    yield

    await db.close()
    await crm_db.close()
    logger.info("Database connections closed.")

app = FastAPI(lifespan=lifespan)

app.include_router(router)