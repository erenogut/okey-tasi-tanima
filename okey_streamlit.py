import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import numpy as np


model = YOLO("best.pt")

st.title("📷 Okey Taşı Tanıma Uygulaması")
st.markdown("Kameranı kullanarak gerçek zamanlı olarak okey taşlarını tanı!")


run = st.checkbox('Kamerayı Başlat')

FRAME_WINDOW = st.image([])

cap = None
if run:
    cap = cv2.VideoCapture(0)
else:
    st.warning("Kamerayı başlatmak için kutucuğu işaretle.")

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Kamera görüntüsü alınamadı.")
        break

    
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
