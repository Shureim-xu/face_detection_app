import cv2
import streamlit as st
import numpy as np
from datetime import datetime

# Setup 
# Load Haar Cascade (relative path)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

st.title("Face Detection using the Viola–Jones Algorithm")

# Instructions
st.markdown("""
### 👤 How to Use This App
1. Allow your browser to access the **webcam**.  
2. Adjust detection settings from the sidebar.  
3. Click **Start Face Detection**.  
4. Rectangles will appear around detected faces.  
5. Click **Stop** to end the webcam stream.
""")

#Sidebar Settings 
st.sidebar.header("⚙️ Detection Settings")
rectangle_color = st.sidebar.color_picker("Rectangle Color", "#00FF00")
scale_factor = st.sidebar.slider("Scale Factor", 1.1, 2.0, 1.3)
min_neighbors = st.sidebar.slider("minNeighbors", 1, 15, 5)
save_image_checkbox = st.sidebar.checkbox("Save detected image")

#Session State
if "running" not in st.session_state:
    st.session_state.running = False

# Start / Stop buttons
if st.button("Start Face Detection", key="start_button"):
    st.session_state.running = True

if st.button("Stop Detection", key="stop_button"):
    st.session_state.running = False

# Placeholder for video frames
frame_placeholder = st.empty()

# Webcam Loop 
if st.session_state.running:
    cap = cv2.VideoCapture(0)

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.error("Cannot access webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors
        )

        # Convert hex color → BGR
        hex_color = rectangle_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        color_bgr = (b, g, r)

        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 2)

        # Convert to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display frame
        frame_placeholder.image(frame_rgb, channels="RGB")

        # Save image if requested (only once per press)
        if save_image_checkbox and len(faces) > 0:
            filename = f"detected_faces_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            st.sidebar.success(f"Image saved as {filename}")
            save_image_checkbox = False  # Prevent saving every frame

    cap.release()
    st.success("Face detection stopped.")
