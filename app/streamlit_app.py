import streamlit as st
import cv2
import torch
import numpy as np
import os
import sys

# Adding project roots to Python Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import Config
from models.model import SignLanguageModel
from core.inference_engine import InferenceEngine
from core.preprocessing import mp_holistic
from services.text_to_speech import speak_text
from services.speech_to_text import listen_and_recognize
from services.text_to_sign import text_to_sign_video

# Streamlit Page Setting
st.set_page_config(page_title="Sign Language System", layout="wide")

@st.cache_resource
def load_model_and_classes():
    """Loads classes and model weights into memory"""
    if not os.path.exists(Config.LABEL_MAPPING_PATH):
        st.error("Label mapping not found. Please run preprocessing to generate labels.")
        return None, []
        
    label_mapping = np.load(Config.LABEL_MAPPING_PATH, allow_pickle=True).item()
    classes = {v: k for k, v in label_mapping.items()}
    class_list = [classes[i] for i in range(len(classes))]
    
    model = SignLanguageModel(
        input_size=Config.KEYPOINT_DIM,
        hidden_size=Config.HIDDEN_SIZE,
        num_layers=Config.NUM_LAYERS,
        num_classes=len(class_list),
        model_type=Config.MODEL_TYPE
    )
    
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
    else:
        st.warning("Model weights not found. You need to train the model first before inference.")
        
    return model, class_list

st.title("Hệ thống giao tiếp 2 chiều: Ngôn ngữ ký hiệu ↔ Văn bản/Giọng nói")

model, classes = load_model_and_classes()

tab1, tab2 = st.tabs(["Nhận diện Ngôn Ngữ Ký Hiệu", "Dịch sang Ngôn Ngữ Ký Hiệu"])

# ----------------- TAB 1 -----------------
with tab1:
    st.header("1. Nhận diện từ Webcam")
    st.markdown("Chức năng này sẽ quét hành động từ webcam và chuyển thành văn bản + phát thành tiếng.")
    
    run_camera = st.checkbox("Bật Camera")
    
    if run_camera and model is not None:
        FRAME_WINDOW = st.image([])
        text_output = st.empty()
        audio_placeholder = st.empty()
        
        # Init inference engine
        engine = InferenceEngine(model, classes)
        cap = cv2.VideoCapture(0)
        
        last_spoken = None
        
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from camera.")
                    break
                    
                image, pred_class, confidence = engine.predict_frame(frame, holistic)
                
                # Render logic
                if pred_class:
                    # Draw text over output frame
                    cv2.putText(image, f'{pred_class} ({confidence:.2f})', (10, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                                
                    if pred_class != last_spoken and confidence > Config.CONFIDENCE_THRESHOLD:
                        text_output.markdown(f"**Nhận diện:** {pred_class}")
                        
                        # TTS processing
                        audio_path = speak_text(pred_class)
                        if audio_path and os.path.exists(audio_path):
                            audio_bytes = open(audio_path, 'rb').read()
                            audio_placeholder.audio(audio_bytes, format='audio/mp3', autoplay=True)
                            
                        last_spoken = pred_class
                
                # Streamlit Display
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(image)
        
        cap.release()

# ----------------- TAB 2 -----------------
with tab2:
    st.header("2. Dịch Văn Bản / Giọng Nói sang Ký Hiệu")
    st.markdown("Nhập văn bản hoặc sử dụng micro để tạo video ký hiệu tương ứng.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        text_input = st.text_input("Nhập văn bản tiếng Việt:")
        st.write("Hoặc sử dụng giọng nói:")
        if st.button("🎤 Bắt đầu ghi âm"):
            recognized_text = listen_and_recognize()
            if recognized_text:
                st.success(f"Nhận diện được: {recognized_text}")
                text_input = recognized_text
            else:
                st.error("Không thể nhận diện giọng nói.")
                
        if st.button("Tạo Video Sign Language") and text_input:
            with st.spinner("Đang tạo video..."):
                output_path = text_to_sign_video(text_input, "output_sign.mp4")
                if output_path and os.path.exists(output_path):
                    st.success("Tạo video thành công!")
                    with col2:
                        video_bytes = open(output_path, 'rb').read()
                        st.video(video_bytes)
                else:
                    st.error("Không tìm thấy dữ liệu video cho các từ đã nhập.")
