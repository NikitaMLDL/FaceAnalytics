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
