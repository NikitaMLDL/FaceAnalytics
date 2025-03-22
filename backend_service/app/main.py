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
    Context manager for managing the database connection.

    This method connects to the database at the start of the application 
    and closes the connection at the end.
    """
    await db.connect()
    logger.info("Database connection established.")

    app.state.db = db
    
    yield
    
    await db.close()
    logger.info("Database connection closed.")

app = FastAPI(lifespan=lifespan)

app.include_router(router)
