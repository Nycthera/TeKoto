import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import av
import numpy as np
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode

# --- Set page config ---
st.set_page_config(page_title="ASL Live Translator", layout="centered")

# --- Load model ---
@st.cache_resource
def load_model():
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 29)  # 26 letters + 3 extra classes
    model.load_state_dict(torch.load("frontend/asl_model_best.pth", map_location="cpu"))
    model.eval()
    return model
    
model = load_model()

# --- Class names ---
class_names = [chr(i) for i in range(65, 91)] + ['del', 'nothing', 'space']

# --- Preprocessing ---
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- WebRTC config ---
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- Video processor ---
class ASLProcessor(VideoProcessorBase):
    def __init__(self):
        self.predicted_label = ""

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        img = cv2.flip(image, 1)

        # Simple color-based hand detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([20, 255, 255]))
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            hand = max(contours, key=cv2.contourArea)
            if cv2.contourArea(hand) > 1000:
                x, y, w, h = cv2.boundingRect(hand)
                roi = img[y:y+h, x:x+w]
                if roi.size > 0:
                    try:
                        input_tensor = preprocess(roi).unsqueeze(0)
                        with torch.no_grad():
                            outputs = model(input_tensor)
                            _, pred = torch.max(outputs, 1)
                            self.predicted_label = class_names[pred.item()]
                    except Exception:
                        self.predicted_label = ""

                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)

        cv2.putText(img, f"Prediction: {self.predicted_label}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Streamlit UI ---
st.title("ASL to Text Translator (Live)")
st.markdown("""
This app uses a ResNet-18 model trained on the ASL Alphabet dataset to translate hand gestures live from your webcam.
""")

# Start webcam streamer
ctx = webrtc_streamer(
    key="asl-translator",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=ASLProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

# Live update the predicted label into a text area
if ctx and ctx.video_processor:
    placeholder = st.empty()
    while ctx.state.playing:
        predicted = ctx.video_processor.predicted_label
        placeholder.text_area("Predicted Letter", value=predicted, height=70, key=f"predicted_text_{time.time()}")
        time.sleep(0.2)
