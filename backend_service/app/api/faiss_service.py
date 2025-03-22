import faiss
import numpy as np
import os
import logging
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDBService:
    """
    Class for working with the FAISS database.
    """

    def __init__(self, index_file_path: str = "faiss_index.index"):
        """
        Initializes the FAISS database.

        :param index_file_path: Path to the file for saving the index.
        """
        self.index_file_path = index_file_path
        self.index = None
        self.load_index()

    def load_index(self):
        """
        Loads the index from a file if it exists.
        If the index file does not exist, a new index is created.
        """
        if os.path.exists(self.index_file_path):
            try:
                self.index = faiss.read_index(self.index_file_path)
                logger.info(f"Index loaded from {self.index_file_path}")
            except Exception as e:
                logger.error(f"Failed to load index: {str(e)}")
                self.index = faiss.IndexFlatL2(512)
                self.index = faiss.IndexIDMap(self.index)
        else:
            self.index = faiss.IndexFlatL2(512)
            self.index = faiss.IndexIDMap(self.index)

    def save_index(self):
        """
        Saves the index to a file.
        """
        try:
            faiss.write_index(self.index, self.index_file_path)
            logger.info(f"Index saved to {self.index_file_path}")
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")

    def add_embeddings(self, embeddings: np.ndarray, ids: List[int]):
        """
        Adds embeddings to FAISS.

        :param embeddings: Embedding vectors to be added.
        :param ids: List of user identifiers.
        """
        try:
            embeddings = embeddings.astype(np.float32)
            self.index.add_with_ids(embeddings, np.array(ids, dtype=np.int64))
            logger.info(f"Added {len(ids)} embeddings to FAISS.")
            self.save_index()
        except Exception as e:
            logger.error(f"Error adding embeddings to FAISS: {str(e)}")

    def search_embedding(self, embedding: np.ndarray, k: int = 1) -> List[int]:
        """
        Searches for the most similar embeddings.

        :param embedding: The embedding to search for similar ones.
        :param k: The number of nearest neighbors to find.
        :return: A list of user IDs and their corresponding confidence levels.
        """
        try:
            embedding = embedding.astype(np.float32).reshape(1, -1)
            distances, indices = self.index.search(embedding, k)
            logger.info(f"Found {k} nearest neighbors.")

            confidence = []
            for i in range(k):
                distance = distances[0][i]
                if distance < 0.6:
                    confidence.append(1.0)  # 100% confidence
                elif distance < 1.0:
                    confidence.append(0.95)  # 95% confidence
                else:
                    confidence.append(0.5)  # Low confidence if the distance is large

            logger.info(f"User indices: {indices[0].tolist()}")
            logger.info(f"Confidence: {confidence}")

            filtered_results = [(indices[0][i], confidence[i]) for i in range(k) if confidence[i] >= 0.8]
            return filtered_results

        except Exception as e:
            logger.error(f"Error searching in FAISS: {str(e)}")
            return []
