from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime


class FaceRecognizeRequest(BaseModel):
    """
    Model for validating the input data in a face recognition request.

    Attributes:
    - image: str
        A base64-encoded string representing the image, or a path to the image
        that will be used for face recognition. This image is sent to the server
        for processing and recognition.
    """
    image: str  # Base64-encoded string of the image or a file path


class Purchase(BaseModel):
    product_name: str
    quantity: int
    purchase_date: datetime  # Оставляем как datetime


class PersonResponse(BaseModel):
    """
    Модель для представления данных о распознанном человеке, включая аналитику по покупкам.

    Attributes:
    - name: str
        Имя распознанного человека.
    - description: Optional[str]
        Описание человека (например, профессия или биография).
        Может быть пустым, если описание отсутствует.
    - confidence: float
        Уверенность модели в распознавании лица, значение от 0 до 1.
        Чем выше значение, тем более уверена модель, что лицо принадлежит распознанному человеку.
    - purchases: List[Purchase]
        Список покупок, ассоциированных с человеком.
    - daily_sales: Dict[str, float]
        Данные о продажах по дням. Ключ - дата, значение - суммарная сумма покупок в этот день.
    - product_sales: Dict[str, float]
        Данные о продажах по продуктам. Ключ - название продукта, значение - суммарная сумма продаж.
    """
    name: str
    description: Optional[str] = None
    confidence: float
    purchases: List[Purchase] = []  # Используем вложенную модель Purchase
    daily_sales: Dict[str, float] = {}  # Продажи по дням
    product_sales: Dict[str, float] = {}  # Продажи по продуктам