import streamlit as st
import requests
import io
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

API_BASE_URL = "http://localhost:8000"

st.title("Face Recognition")

if "description" not in st.session_state:
    st.session_state.description = ""

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Recognizing face..."):
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        files = {"file": buffered.getvalue()}
        response = requests.post(f"{API_BASE_URL}/face_recognize", files=files)

    if response.status_code == 200:
        result = response.json()
        st.success(f"Result: {result['name']}")
        st.write(f"Description: {result['description']}")

        # Отображение данных о покупках
        if result['purchases']:
            # Получение daily_sales и product_sales
            daily_sales = result['daily_sales']
            product_sales = result['product_sales']

            # 1. График продаж по дням
            if daily_sales:
                daily_sales_df = pd.DataFrame(list(daily_sales.items()), columns=["Date", "Sales"])
                daily_sales_df["Date"] = pd.to_datetime(daily_sales_df["Date"])
                st.subheader("Sales per Day")
                fig_daily_sales, ax_daily_sales = plt.subplots()
                ax_daily_sales.plot(daily_sales_df["Date"], daily_sales_df["Sales"], marker='o', color='b', label="Sales")
                ax_daily_sales.set_title("Daily Sales")
                ax_daily_sales.set_xlabel("Date")
                ax_daily_sales.set_ylabel("Sales Amount")
                ax_daily_sales.grid(True)
                st.pyplot(fig_daily_sales)

            # 2. График продаж по продуктам
            if product_sales:
                product_sales_df = pd.DataFrame(list(product_sales.items()), columns=["Product", "Sales"])
                st.subheader("Sales per Product")
                fig_product_sales, ax_product_sales = plt.subplots()
                ax_product_sales.bar(product_sales_df["Product"], product_sales_df["Sales"], color='g', label="Sales")
                ax_product_sales.set_title("Sales per Product")
                ax_product_sales.set_xlabel("Product")
                ax_product_sales.set_ylabel("Sales Amount")
                ax_product_sales.grid(True)
                st.pyplot(fig_product_sales)

        # Если пользователь новый, добавляем описание
        if result['name'] == "New User":
            st.session_state.description = ""
            description = st.text_area("Enter a description for the new user", value=st.session_state.description)

            if st.button("Add User"):
                if not description.strip():
                    st.error("Description is required!")
                else:
                    with st.spinner("Adding user..."):
                        files = {"file": buffered.getvalue()}
                        response_add = requests.post(f"{API_BASE_URL}/add_new_person", data={"description": description}, files=files)

                    if response_add.status_code == 200:
                        st.success(f"User added successfully!")
                    else:
                        st.error("Error while adding the user")
    else:
        st.error("Error recognizing face.")
