import cv2
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default .xml")

st.title("Real-Time Face Detection (Viola–Jones + WebRTC)")

st.markdown("""
This version works **on Streamlit Cloud** because it uses **WebRTC** instead of trying to access the server's webcam.
""")

# Sidebar controls
st.sidebar.header("⚙️ Detection Settings")

rectangle_color = st.sidebar.color_picker("Rectangle Color", "#00FF00")
scale_factor = st.sidebar.slider("Scale Factor", 1.1, 2.0, 1.3)
min_neighbors = st.sidebar.slider("minNeighbors", 1, 15, 5)


# Convert UI color to BGR tuple
hex_color = rectangle_color.lstrip('#')
r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
color_bgr = (b, g, r)


# Video Processor
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors
        )

        # Draw rectangles
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), color_bgr, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# Start WebRTC stream
webrtc_streamer(
    key="face-detection",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)
