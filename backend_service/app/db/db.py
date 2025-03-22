import asyncpg
import logging
from typing import Optional
from dotenv import load_dotenv
import os

load_dotenv()

database_url = os.getenv("DATABASE_URL")

class Database:
    def __init__(self):
        """
        Инициализация класса Database.
        """
        self.pool = None
        self.logger = logging.getLogger(__name__)

    async def connect(self):
        """
        Подключение к базе данных и инициализация пула соединений.

        Этот метод устанавливает соединение с базой данных и создает таблицы, если они не существуют.
        """
        try:
            self.pool = await asyncpg.create_pool(dsn=database_url)
            self.logger.info("Подключение к базе данных успешно установлено.")
            await self.create_tables_if_not_exists()
        except Exception as e:
            self.logger.error(f"Ошибка при подключении к базе данных: {str(e)}")
            raise

    async def close(self):
        """
        Закрытие соединений с базой данных.

        Этот метод закрывает пул соединений с базой данных, если он был открыт.
        """
        if self.pool:
            await self.pool.close()
            self.logger.info("Соединение с базой данных закрыто.")
        else:
            self.logger.warning("Пул соединений уже закрыт.")

    async def create_tables_if_not_exists(self):
        """
        Создание таблицы, если она не существует.

        Этот метод выполняет SQL-запрос для создания таблицы пользователей.
        """
        create_table_query = """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            description TEXT NOT NULL
        );
        """
        try:
            async with self.pool.acquire() as connection:
                await connection.execute(create_table_query)
                self.logger.info("Таблица users успешно создана или уже существует.")
        except Exception as e:
            self.logger.error(f"Ошибка при создании таблицы: {str(e)}")

    async def get_description(self, user_id: int) -> Optional[str]:
        """
        Получение описания пользователя из базы данных.

        :param user_id: Идентификатор пользователя.
        :return: Описание пользователя, если оно существует, иначе None.
        """
        try:
            async with self.pool.acquire() as connection:
                result = await connection.fetchrow(
                    "SELECT description FROM users WHERE id=$1", user_id
                )
                if result:
                    return result['description']
                return None
        except Exception as e:
            self.logger.error(f"Ошибка при получении описания пользователя {user_id}: {str(e)}")
            return None

    async def add_description(self, user_id: int, description: str):
        """
        Добавление нового описания пользователя в базу данных.

        :param user_id: Идентификатор пользователя.
        :param description: Описание пользователя.
        """
        try:
            if self.pool is None:
                self.logger.error("Ошибка: соединение с базой данных не установлено (pool = None).")
                return

            async with self.pool.acquire() as connection:
                await connection.execute(
                    "INSERT INTO users (id, description) VALUES ($1, $2)",
                    user_id, description
                )
                self.logger.info(f"Описание для пользователя {user_id} успешно добавлено.")
        except Exception as e:
            self.logger.error(f"Ошибка при добавлении описания для пользователя {user_id}: {str(e)}")

    async def update_description(self, user_id: int, description: str):
        """
        Обновление описания пользователя в базе данных.

        :param user_id: Идентификатор пользователя.
        :param description: Новое описание пользователя.
        """
        try:
            async with self.pool.acquire() as connection:
                result = await connection.execute(
                    "UPDATE users SET description=$1 WHERE id=$2",
                    description, user_id
                )
                if result:
                    self.logger.info(f"Описание для пользователя {user_id} успешно обновлено.")
                else:
                    self.logger.warning(f"Пользователь с ID {user_id} не найден.")
        except Exception as e:
            self.logger.error(f"Ошибка при обновлении описания пользователя {user_id}: {str(e)}")

    async def user_exists(self, user_id: int) -> bool:
        """
        Проверка, существует ли пользователь в базе данных.

        :param user_id: Идентификатор пользователя.
        :return: True, если пользователь существует, иначе False.
        """
        try:
            async with self.pool.acquire() as connection:
                result = await connection.fetchrow(
                    "SELECT 1 FROM users WHERE id=$1", user_id
                )
                return result is not None
        except Exception as e:
            self.logger.error(f"Ошибка при проверке существования пользователя {user_id}: {str(e)}")
            return False
