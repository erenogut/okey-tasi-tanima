import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

model = YOLO("best.pt")

st.title("📷 Okey Taşı Tanıma Uygulaması")

uploaded_image = st.file_uploader("Bir okey taşı fotoğrafı yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Yüklenen Görüntü", use_column_width=True)

    
    img_array = np.array(image)

    
    results = model(img_array)

  
    res_plotted = results[0].plot()

    st.image(res_plotted, caption="Tahmin Sonucu", use_column_width=True)


    
    results = model(frame)[0]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = model.names[cls]
        conf = float(box.conf[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
