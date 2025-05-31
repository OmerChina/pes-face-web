
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp

st.set_page_config(page_title="PES 2021 Yüz Rehberi", layout="centered")

st.title("🧠 PES 2021 Yüz Rehberi Oluşturucu (Demo)")
st.write("Yüz fotoğrafını yükle, biz senin için PES karakter değerlerini hesaplayalım.")

uploaded_file = st.file_uploader("📤 Yüz Fotoğrafı Yükle (Önden)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image_rgb, caption="Yüklenen Görsel", use_column_width=True)

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(image_rgb)
        if results.multi_face_landmarks:
            st.success("✅ Yüz algılandı! PES değerleri hesaplanıyor...")
            face_landmarks = results.multi_face_landmarks[0]

            # Örnek metrik: Gözler arası mesafe
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            eye_distance = np.linalg.norm(
                np.array([left_eye.x, left_eye.y]) - np.array([right_eye.x, right_eye.y])
            )

            # Normalize edilmiş örnek PES değerleri
            pes_values = {
                "Kafa Genişliği": round(eye_distance * 100, 2),
                "Göz Yüksekliği": 5,
                "Burun Derinliği": 4,
                "Çene Hattı": 6
            }

            for k, v in pes_values.items():
                st.write(f"**{k}**: {v}")
        else:
            st.error("❌ Yüz algılanamadı. Lütfen daha net bir fotoğraf yükleyin.")
