import streamlit as st
import requests
import io
from PIL import Image


API_BASE_URL = "http://localhost:8000"

st.title("Распознавание лиц")

if "description" not in st.session_state:
    st.session_state.description = ""

uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Загруженное изображение", use_container_width=True)

    with st.spinner("Распознаем лицо..."):
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        files = {"file": buffered.getvalue()}
        response = requests.post(f"{API_BASE_URL}/face_recognize", files=files)

    if response.status_code == 200:
        result = response.json()
        st.success(f"Результат: {result['name']}")
        st.write(f"Описание: {result['description']}")

        if result['name'] == "New User":
            st.session_state.description = ""
            description = st.text_area("Введите описание для нового пользователя", value=st.session_state.description)

            if st.button("Добавить пользователя"):
                if not description.strip():
                    st.error("Описание должно быть обязательным!")
                else:
                    with st.spinner("Добавляем пользователя..."):
                        files = {"file": buffered.getvalue()}
                        response_add = requests.post(f"{API_BASE_URL}/add_new_person", data={"description": description}, files=files)

                    if response_add.status_code == 200:
                        st.success(f"Пользователь успешно добавлен!")
                    else:
                        st.error("Ошибка при добавлении пользователя")
