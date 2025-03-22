import faiss
import numpy as np
import os
import logging
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDBService:
    """
    Класс для работы с базой данных FAISS.
    """

    def __init__(self, index_file_path: str = "faiss_index.index"):
        """
        Инициализация базы данных FAISS.

        :param index_file_path: Путь к файлу для сохранения индекса.
        """
        self.index_file_path = index_file_path
        self.index = None
        self.load_index()

    def load_index(self):
        """
        Загрузка индекса из файла, если он существует.
        Если файл индекса не существует, создается новый индекс.
        """
        if os.path.exists(self.index_file_path):
            try:
                self.index = faiss.read_index(self.index_file_path)
                logger.info(f"Индекс загружен из {self.index_file_path}")
            except Exception as e:
                logger.error(f"Не удалось загрузить индекс: {str(e)}")
                self.index = faiss.IndexFlatL2(512)
                self.index = faiss.IndexIDMap(self.index)
        else:
            self.index = faiss.IndexFlatL2(512)
            self.index = faiss.IndexIDMap(self.index)

    def save_index(self):
        """
        Сохранение индекса в файл.
        """
        try:
            faiss.write_index(self.index, self.index_file_path)
            logger.info(f"Индекс сохранен в {self.index_file_path}")
        except Exception as e:
            logger.error(f"Ошибка сохранения индекса: {str(e)}")

    def add_embeddings(self, embeddings: np.ndarray, ids: List[int]):
        """
        Добавление эмбеддингов в FAISS.

        :param embeddings: Векторы эмбеддингов для добавления.
        :param ids: Список идентификаторов пользователей.
        """
        try:
            embeddings = embeddings.astype(np.float32)
            self.index.add_with_ids(embeddings, np.array(ids, dtype=np.int64))
            logger.info(f"Добавлено {len(ids)} эмбеддингов в FAISS.")
            self.save_index()
        except Exception as e:
            logger.error(f"Ошибка добавления эмбеддингов в FAISS: {str(e)}")

    def search_embedding(self, embedding: np.ndarray, k: int = 1) -> List[int]:
        """
        Поиск наиболее похожих эмбеддингов.

        :param embedding: Эмбеддинг, по которому нужно найти похожие.
        :param k: Количество ближайших соседей для поиска.
        :return: Список идентификаторов найденных пользователей с соответствующими confidence.
        """
        try:
            embedding = embedding.astype(np.float32).reshape(1, -1)
            distances, indices = self.index.search(embedding, k)
            logger.info(f"Найдено {k} ближайших соседей.")

            confidence = []
            for i in range(k):
                distance = distances[0][i]
                if distance < 0.6:
                    confidence.append(1.0)
                elif distance < 1.0:
                    confidence.append(0.95)
                else:
                    confidence.append(0.5)

            logger.info(f"Индексы пользователей: {indices[0].tolist()}")
            logger.info(f"Confidence: {confidence}")

            filtered_results = [(indices[0][i], confidence[i]) for i in range(k) if confidence[i] >= 0.8]
            return filtered_results

        except Exception as e:
            logger.error(f"Ошибка поиска в FAISS: {str(e)}")
            return []
