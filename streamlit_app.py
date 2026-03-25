import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

st.set_page_config(page_title="Triniverse AI Test", layout="centered")

st.title("🧬 TRINIVERSE AI: KIỂM CHỨNG CÔNG THỨC")
st.write("Hệ thống xác định Tư duy gốc dựa trên Nhân trắc học xương cứng.")

# Khởi tạo MediaPipe
@st.cache_resource
def load_mesh():
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, 
        max_num_faces=1, 
        refine_landmarks=True
    )

face_mesh = load_mesh()

uploaded_file = st.file_uploader("Tải ảnh chân dung rõ mặt...", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    h, w, _ = img_array.shape
    
    # Xử lý Face Mesh
    results = face_mesh.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        def g(i): return np.array([lm[i].x * w, lm[i].y * h])

        # --- THUẬT TOÁN 2 LỚP ---
        y1, y2, y3, y4 = g(10)[1], g(168)[1], g(2)[1], g(152)[1]
        t_h = y4 - y1
        t1, t2, t3 = (y2-y1)/t_h*100, (y3-y2)/t_h*100, (y4-y3)/t_h*100
        
        dist = np.linalg.norm
        r_top = dist(g(21)-g(251)) / dist(g(234)-g(454))
        r_mid = dist(g(234)-g(454)) / dist(g(21)-g(251))
        r_bot = dist(g(172)-g(397)) / dist(g(234)-g(454))

        # Logic phân loại
        res = ""
        if abs(t1-t2) <= 1.5 and abs(t2-t3) <= 1.5: res = "NHÓM 6"
        elif abs(t1-t2) <= 1.5 and t1 > t3: res = "NHÓM 6"
        else:
            if t1 >= t2 and t1 >= t3:
                if r_top < 0.82: res = "NHÓM 5"
                elif r_top > 0.88: res = "NHÓM 7"
                else: res = "NHÓM 6"
            elif t2 >= t1 and t2 >= t3:
                if r_mid < 1.05: res = "NHÓM 4"
                elif r_mid > 1.15: res = "NHÓM 3"
                else: res = "NHÓM 2"
            else:
                if r_bot < 0.90: res = "NHÓM 1"
                else: res = "NHÓM 8"

        # Hiển thị kết quả chuyên nghiệp
        st.divider()
        st.subheader(f"👉 KẾT QUẢ ĐỊNH DANH: {res}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("**Tỷ lệ tầng mặt (Dọc)**")
            st.write(f"Trán: {t1:.1f}%")
            st.write(f"Giữa: {t2:.1f}%")
            st.write(f"Cằm: {t3:.1f}%")
        with col2:
            st.warning("**Chỉ số xương (Ngang)**")
            st.write(f"R_Trán: {r_top:.2f}")
            st.write(f"R_Gò má: {r_mid:.2f}")
            st.write(f"R_Hàm: {r_bot:.2f}")
            
    else:
        st.error("Không tìm thấy khuôn mặt. Vui lòng chụp thẳng và rõ nét hơn.")

st.caption("Phát triển bởi Triniverse AI Team.")
