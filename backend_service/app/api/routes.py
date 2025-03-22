from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Request
from app.models.schemas import PersonResponse
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import io
import numpy as np
from .faiss_service import VectorDBService
import os
import logging

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
    Добавление нового пользователя.

    :param request: Объект запроса FastAPI.
    :param description: Описание пользователя.
    :param file: Изображение пользователя.
    :return: Ответ с информацией о добавленном пользователе.
    """
    db = request.app.state.db
    try:
        logger.info(f"Начинаю обработку изображения для нового пользователя: {file.filename}")

        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data)).convert("RGB")

        faces, _ = mtcnn.detect(img)
        if faces is None:
            logger.warning("Лицо не найдено на изображении.")
            raise HTTPException(status_code=400, detail="Лицо не найдено")

        aligned_face = mtcnn(img)
        if aligned_face is None:
            logger.warning("Не удалось выровнять лицо.")
            raise HTTPException(status_code=400, detail="Не удалось выровнять лицо")

        embedding = resnet(aligned_face.unsqueeze(0)).detach().cpu().numpy().astype(np.float32)

        user_id_index = vector_db_service.search_embedding(embedding)

        if user_id_index and user_id_index[0] != -1:
            logger.info(f"Найдено схожее лицо с ID: {user_id_index[0]}")
            description_from_db = await db.get_description(user_id_index[0]) or "Нет описания"
            return PersonResponse(
                name=f"User {user_id_index[0]}",
                description=description_from_db,
                confidence=0.95
            )

        new_user_id = vector_db_service.index.ntotal
        await db.add_description(new_user_id, description)
        vector_db_service.add_embeddings(embedding, [new_user_id])

        logger.info(f"Пользователь с ID {new_user_id} успешно добавлен.")

        return PersonResponse(
            name="New User",
            description=description,
            confidence=0.5
        )

    except Exception as e:
        logger.error(f"Ошибка при добавлении пользователя: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка добавления пользователя: {str(e)}")


@router.post("/face_recognize", response_model=PersonResponse)
async def face_recognize(request: Request, file: UploadFile = File(...)):
    """
    Распознавание лица.

    :param request: Объект запроса FastAPI.
    :param file: Изображение для распознавания.
    :return: Ответ с информацией о найденном пользователе или о необходимости добавить нового.
    """
    db = request.app.state.db
    try:
        logger.info(f"Начинаю обработку изображения для распознавания: {file.filename}")

        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data)).convert("RGB")

        faces, _ = mtcnn.detect(img)
        if faces is None:
            logger.warning("Лицо не найдено на изображении.")
            raise HTTPException(status_code=400, detail="Лицо не найдено")

        aligned_face = mtcnn(img)
        if aligned_face is None:
            logger.warning("Не удалось выровнять лицо.")
            raise HTTPException(status_code=400, detail="Не удалось выровнять лицо")

        embedding = resnet(aligned_face.unsqueeze(0)).detach().cpu().numpy().astype(np.float32)

        user_results = vector_db_service.search_embedding(embedding)

        if user_results:
            user_id, confidence = user_results[0]
            description_from_db = await db.get_description(user_id) or "Нет описания"
            logger.info(f"Найден пользователь с ID: {user_id} с confidence: {confidence}")

            return PersonResponse(
                name=f"User {user_id}",
                description=description_from_db,
                confidence=confidence
            )

        logger.info("Пользователь не найден. Добавьте нового.")
        return PersonResponse(
            name="New User",
            description="Пожалуйста, введите описание пользователя",
            confidence=0.5
        )

    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки изображения: {str(e)}")
