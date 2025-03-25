import asyncpg
import logging
from typing import Optional
from dotenv import load_dotenv
import os
from typing import List, Dict, Any

load_dotenv()

database_url = os.getenv("DATABASE_URL")


class Database:
    def __init__(self):
        """
        Initializes the Database class.
        """
        self.pool = None
        self.logger = logging.getLogger(__name__)

    async def connect(self):
        """
        Connects to the database and initializes the connection pool.

        This method establishes a connection to the database and creates tables if they do not exist.
        """
        try:
            self.pool = await asyncpg.create_pool(dsn=database_url)
            self.logger.info("Database connection successfully established.")
            await self.create_tables_if_not_exists()
        except Exception as e:
            self.logger.error(f"Error connecting to the database: {str(e)}")
            raise

    async def close(self):
        """
        Closes database connections.

        This method closes the connection pool if it was opened.
        """
        if self.pool:
            await self.pool.close()
            self.logger.info("Database connection closed.")
        else:
            self.logger.warning("Connection pool already closed.")

    async def create_tables_if_not_exists(self):
        """
        Creates the table if it does not exist.

        This method runs an SQL query to create a 'users' table if it does not already exist.
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
                self.logger.info("Users table created successfully or already exists.")
        except Exception as e:
            self.logger.error(f"Error creating the table: {str(e)}")

    async def get_description(self, user_id: int) -> Optional[str]:
        """
        Fetches the description of a user from the database.

        :param user_id: The user ID.
        :return: The user's description if it exists, otherwise None.
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
            self.logger.error(f"Error fetching description for user {user_id}: {str(e)}")
            return None

    async def add_description(self, user_id: int, description: str):
        """
        Adds a new description for a user in the database.

        :param user_id: The user ID.
        :param description: The description of the user.
        """
        try:
            if self.pool is None:
                self.logger.error("Error: database connection not established (pool = None).")
                return

            async with self.pool.acquire() as connection:
                await connection.execute(
                    "INSERT INTO users (id, description) VALUES ($1, $2)",
                    user_id, description
                )
                self.logger.info(f"Description for user {user_id} successfully added.")
        except Exception as e:
            self.logger.error(f"Error adding description for user {user_id}: {str(e)}")

    async def update_description(self, user_id: int, description: str):
        """
        Updates the description of a user in the database.

        :param user_id: The user ID.
        :param description: The new description for the user.
        """
        try:
            async with self.pool.acquire() as connection:
                result = await connection.execute(
                    "UPDATE users SET description=$1 WHERE id=$2",
                    description, user_id
                )
                if result:
                    self.logger.info(f"Description for user {user_id} successfully updated.")
                else:
                    self.logger.warning(f"User with ID {user_id} not found.")
        except Exception as e:
            self.logger.error(f"Error updating description for user {user_id}: {str(e)}")

    async def user_exists(self, user_id: int) -> bool:
        """
        Checks if a user exists in the database.

        :param user_id: The user ID.
        :return: True if the user exists, otherwise False.
        """
        try:
            async with self.pool.acquire() as connection:
                result = await connection.fetchrow(
                    "SELECT 1 FROM users WHERE id=$1", user_id
                )
                return result is not None
        except Exception as e:
            self.logger.error(f"Error checking existence of user {user_id}: {str(e)}")
            return False


class CRMDatabase:
    """
    Класс для управления базой данных CRM, хранящей информацию о покупках.
    """
    def __init__(self):
        self.pool = None
        self.logger = logging.getLogger(__name__)

    async def connect(self):
        """Подключение к CRM базе данных."""
        try:
            self.pool = await asyncpg.create_pool(dsn=database_url)
            self.logger.info("Подключение к CRM базе данных успешно.")
            await self.create_tables_if_not_exists()
        except Exception as e:
            self.logger.error(f"Ошибка подключения к CRM БД: {str(e)}")
            raise

    async def close(self):
        """Закрытие соединения с CRM базой данных."""
        if self.pool:
            await self.pool.close()
            self.logger.info("Соединение с CRM БД закрыто.")
        else:
            self.logger.warning("Пул соединений CRM уже закрыт.")

    async def create_tables_if_not_exists(self):
        """Создание таблицы покупок, если её нет."""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS purchases (
            purchase_id SERIAL PRIMARY KEY,
            client_id INT NOT NULL,
            product_name TEXT NOT NULL,
            quantity INT NOT NULL,
            purchase_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        try:
            async with self.pool.acquire() as connection:
                await connection.execute(create_table_query)
                self.logger.info("Таблица purchases создана или уже существует.")
        except Exception as e:
            self.logger.error(f"Ошибка при создании таблицы purchases: {str(e)}")

    async def add_purchase(self, client_id: int, product_name: str, quantity: int) -> int:
        """
        Добавляет запись о покупке в таблицу.

        :param client_id: ID клиента
        :param product_name: Название товара
        :param quantity: Количество купленного товара
        :return: ID созданной покупки
        """
        insert_query = """
        INSERT INTO purchases (client_id, product_name, quantity)
        VALUES ($1, $2, $3) RETURNING purchase_id;
        """
        try:
            async with self.pool.acquire() as connection:
                purchase_id = await connection.fetchval(insert_query, client_id, product_name, quantity)
                self.logger.info(f"Покупка успешно добавлена: ID {purchase_id}")
                return purchase_id
        except Exception as e:
            self.logger.error(f"Ошибка при добавлении покупки: {str(e)}")
            return -1

    async def get_purchases_by_user_id(self, client_id: int) -> List[Dict[str, Any]]:
        """
        Получает список покупок пользователя по его ID.

        :param client_id: ID клиента.
        :return: Список словарей с информацией о покупках.
        """
        select_query = """
        SELECT product_name, quantity, purchase_date FROM purchases WHERE client_id = $1;
        """
        try:
            async with self.pool.acquire() as connection:
                results = await connection.fetch(select_query, client_id)
                return [
                    {
                        "product_name": row["product_name"],
                        "quantity": row["quantity"],
                        "purchase_date": row["purchase_date"]  # Оставляем datetime
                    }
                    for row in results
                ] if results else []
        except Exception as e:
            self.logger.error(f"Ошибка при получении покупок пользователя {client_id}: {str(e)}")
            return []

    async def get_purchases_by_client(self, client_id: int) -> list:
        """
        Получает список всех покупок определенного клиента.

        :param client_id: ID клиента.
        :return: Список покупок клиента.
        """
        select_query = """
        SELECT * FROM purchases WHERE client_id = $1 ORDER BY purchase_date DESC;
        """
        try:
            async with self.pool.acquire() as connection:
                results = await connection.fetch(select_query, client_id)
                return [dict(record) for record in results]
        except Exception as e:
            self.logger.error(f"Ошибка при получении покупок клиента {client_id}: {str(e)}")
            return []

    async def update_purchase(self, purchase_id: int, product_name: str, quantity: int) -> bool:
        """
        Обновляет информацию о покупке.

        :param purchase_id: ID покупки.
        :param product_name: Новое название товара.
        :param quantity: Новое количество.
        :return: True, если обновление успешно, иначе False.
        """
        update_query = """
        UPDATE purchases SET product_name = $1, quantity = $2 WHERE purchase_id = $3;
        """
        try:
            async with self.pool.acquire() as connection:
                result = await connection.execute(update_query, product_name, quantity, purchase_id)
                return result == "UPDATE 1"
        except Exception as e:
            self.logger.error(f"Ошибка при обновлении покупки {purchase_id}: {str(e)}")
            return False

    async def delete_purchase(self, purchase_id: int) -> bool:
        """
        Удаляет запись о покупке.

        :param purchase_id: ID покупки.
        :return: True, если удаление успешно, иначе False.
        """
        delete_query = """
        DELETE FROM purchases WHERE purchase_id = $1;
        """
        try:
            async with self.pool.acquire() as connection:
                result = await connection.execute(delete_query, purchase_id)
                return result == "DELETE 1"
        except Exception as e:
            self.logger.error(f"Ошибка при удалении покупки {purchase_id}: {str(e)}")
            return False

    async def get_top_products(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Получает список самых популярных товаров.

        :param limit: Количество товаров в топе.
        :return: Список популярных товаров.
        """
        query = """
        SELECT product_name, SUM(quantity) AS total_sold
        FROM purchases
        GROUP BY product_name
        ORDER BY total_sold DESC
        LIMIT $1;
        """
        try:
            async with self.pool.acquire() as connection:
                results = await connection.fetch(query, limit)
                return [{"product_name": row["product_name"], "total_sold": row["total_sold"]} for row in results]
        except Exception as e:
            self.logger.error(f"Ошибка при получении популярных товаров: {str(e)}")
            return []

    async def get_frequently_bought_together(self, product_name: str, limit: int = 5) -> List[str]:
        """
        Получает список товаров, которые чаще всего покупают вместе с указанным товаром.

        :param product_name: Название товара.
        :param limit: Максимальное количество рекомендаций.
        :return: Список связанных товаров.
        """
        query = """
        SELECT p2.product_name, COUNT(*) AS frequency
        FROM purchases p1
        JOIN purchases p2 ON p1.client_id = p2.client_id
        WHERE p1.product_name = $1 AND p2.product_name <> $1
        GROUP BY p2.product_name
        ORDER BY frequency DESC
        LIMIT $2;
        """
        try:
            async with self.pool.acquire() as connection:
                results = await connection.fetch(query, product_name, limit)
                return [row["product_name"] for row in results]
        except Exception as e:
            self.logger.error(f"Ошибка при получении товаров, покупаемых вместе с {product_name}: {str(e)}")
            return []
    async def get_personal_recommendations(self, client_id: int, limit: int = 5) -> List[str]:
        """
        Получает персональные рекомендации на основе истории покупок клиента.

        :param client_id: ID клиента.
        :param limit: Количество рекомендаций.
        :return: Список рекомендованных товаров.
        """
        query = """
        WITH user_purchases AS (
            SELECT DISTINCT product_name
            FROM purchases
            WHERE client_id = $1
        )
        SELECT DISTINCT p2.product_name
        FROM purchases p1
        JOIN purchases p2 ON p1.client_id = p2.client_id
        WHERE p1.product_name IN (SELECT product_name FROM user_purchases)
        AND p2.product_name NOT IN (SELECT product_name FROM user_purchases)
        GROUP BY p2.product_name
        ORDER BY COUNT(*) DESC
        LIMIT $2;
        """
        try:
            async with self.pool.acquire() as connection:
                results = await connection.fetch(query, client_id, limit)
                return [row["product_name"] for row in results]
        except Exception as e:
            self.logger.error(f"Ошибка при получении персональных рекомендаций для клиента {client_id}: {str(e)}")
            return []
