import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

st.title("Real-Time Object Detection UI")

model = YOLO("yolo11n.pt")

img = st.camera_input("Capture Image")

if img is not None:
    bytes_data = img.getvalue()
    np_img = np.frombuffer(bytes_data, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    results = model(frame)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(box.cls[0].cpu().numpy())
            class_name = model.names[cls_id]
            conf = float(box.conf[0].cpu().numpy())

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame, caption="Detected Objects", use_column_width=True)
