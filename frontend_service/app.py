import streamlit as st
import requests
import io
from PIL import Image


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
