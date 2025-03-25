from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Request
from ..models.schemas import PersonResponse
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import io
import numpy as np
from .faiss_service import VectorDBService
import os
import logging
from collections import defaultdict
from typing import List, Dict
from .telegram_bot import send_telegram_message

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='casia-webface').eval()
vector_db_service = VectorDBService()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/add_new_person", response_model=PersonResponse)
async def add_new_person(request: Request, description: str = Form(...), file: UploadFile = File(...)):
    """
    Adds a new user.

    :param request: The FastAPI request object.
    :param description: Description of the user.
    :param file: Image of the user.
    :return: Response with the added user's information.
    """
    db = request.app.state.db
    try:
        logger.info(f"Processing image for the new user: {file.filename}")

        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data)).convert("RGB")

        faces, _ = mtcnn.detect(img)
        if faces is None:
            logger.warning("No face detected in the image.")
            raise HTTPException(status_code=400, detail="No face detected")

        aligned_face = mtcnn(img)
        if aligned_face is None:
            logger.warning("Failed to align the face.")
            raise HTTPException(status_code=400, detail="Failed to align the face")

        embedding = resnet(aligned_face.unsqueeze(0)).detach().cpu().numpy().astype(np.float32)

        user_id_index = vector_db_service.search_embedding(embedding)

        if user_id_index and user_id_index[0] != -1:
            logger.info(f"Found a similar face with ID: {user_id_index[0]}")
            description_from_db = await db.get_description(user_id_index[0]) or "No description"
            return PersonResponse(
                name=f"User {user_id_index[0]}",
                description=description_from_db,
                confidence=0.95
            )

        new_user_id = vector_db_service.index.ntotal
        await db.add_description(new_user_id, description)
        vector_db_service.add_embeddings(embedding, [new_user_id])

        logger.info(f"User with ID {new_user_id} successfully added.")

        return PersonResponse(
            name="New User",
            description=description,
            confidence=0.5
        )

    except Exception as e:
        logger.error(f"Error adding user: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding user: {str(e)}")


@router.post("/face_recognize", response_model=PersonResponse)
async def face_recognize(request: Request, file: UploadFile = File(...)):
    """
    Face recognition with additional purchase data.

    :param request: The FastAPI request object.
    :param file: Image for face recognition.
    :return: Response with information about the found user and their purchases.
    """
    db = request.app.state.db
    crm_db = request.app.state.crm_db  # Подключение к CRM базе
    try:
        logger.info(f"Processing image for recognition: {file.filename}")

        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data)).convert("RGB")

        faces, _ = mtcnn.detect(img)
        if faces is None:
            logger.warning("No face detected in the image.")
            raise HTTPException(status_code=400, detail="No face detected")

        aligned_face = mtcnn(img)
        if aligned_face is None:
            logger.warning("Failed to align the face.")
            raise HTTPException(status_code=400, detail="Failed to align the face")

        embedding = resnet(aligned_face.unsqueeze(0)).detach().cpu().numpy().astype(np.float32)

        user_results = vector_db_service.search_embedding(embedding)

        if user_results:
            user_id, confidence = user_results[0]
            description_from_db = await db.get_description(user_id) or "No description"
            
            # Получение списка покупок пользователя
            purchases = await crm_db.get_purchases_by_user_id(user_id)

            # Получаем данные о продажах по дням и по продуктам
            daily_sales, product_sales = calculate_sales(purchases)

            logger.info(f"User found with ID: {user_id}, confidence: {confidence}, purchases: {purchases}")

            await send_telegram_message(user_id, description_from_db)
            return PersonResponse(
                name=f"User {user_id}",
                description=description_from_db,
                confidence=confidence,
                purchases=purchases,
                daily_sales=daily_sales,  # Добавляем данные о продажах по дням
                product_sales=product_sales  # Добавляем данные о продажах по продуктам
            )

        logger.info("User not found. Please add a new one.")
        # await send_telegram_message(0, "Новый пользователь, добавьте описание")   # 1
        return PersonResponse(
            name="New User",
            description="Please enter a description for the user",
            confidence=0.5,
            purchases=[],  # Пустой список, если пользователь не найден
            daily_sales={},  # Пустые данные о продажах по дням
            product_sales={}  # Пустые данные о продажах по продуктам
        )

    except Exception as e:
        logger.error(f"Error processing the image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing the image: {str(e)}")


def calculate_sales(purchases: List[dict]) -> tuple[Dict[str, float], Dict[str, float]]:
    daily_sales = defaultdict(float)
    product_sales = defaultdict(float)

    for purchase in purchases:
        # Преобразуем дату в строку (например, '2025-03-23')
        purchase_date = purchase['purchase_date'].strftime('%Y-%m-%d')  # Преобразуем datetime в строку
        daily_sales[purchase_date] += purchase['quantity']  # Суммируем по количеству на каждый день

        # Для каждого продукта, считаем суммарные продажи по названию продукта
        product_sales[purchase['product_name']] += purchase['quantity']

    return dict(daily_sales), dict(product_sales)