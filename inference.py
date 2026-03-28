import cv2
import torch
import numpy as np
import argparse
import sys
import os

# Adding project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from configs.config import Config
from models.model import SignLanguageModel
from core.inference_engine import InferenceEngine
from core.preprocessing import mp_holistic
from services.text_to_speech import speak_text

def main():
    parser = argparse.ArgumentParser(description="Sign Language Inference via Webcam")
    parser.add_argument("--source", type=int, default=0, help="Camera API index, default is 0")
    args = parser.parse_args()

    # Determine Device
    device = "cpu"
    
    # Load Label Mapping
    if not os.path.exists(Config.LABEL_MAPPING_PATH):
        print("Label mapping not found. Please train and validate model first.")
        return
        
    label_mapping = np.load(Config.LABEL_MAPPING_PATH, allow_pickle=True).item()
    classes = {v: k for k, v in label_mapping.items()}
    class_list = [classes[i] for i in range(len(classes))]
    
    # Load Model structure and state dictionary
    model = SignLanguageModel(
        input_size=Config.KEYPOINT_DIM,
        hidden_size=Config.HIDDEN_SIZE,
        num_layers=Config.NUM_LAYERS,
        num_classes=len(class_list),
        model_type=Config.MODEL_TYPE
    )
    
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=torch.device(device)))
    else:
        print("Model configuration loaded but no saved weights found at", Config.BEST_MODEL_PATH)
        return
    
    engine = InferenceEngine(model, class_list, device=device)
    cap = cv2.VideoCapture(args.source)
    
    last_spoken = None
    
    print("\nStarting Open-CV WebCam Inferencing Interface ")
    print("Make sign languages towards the camera.")
    print("Press 'q' inside the window to quit execution.\n")
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed returning frame from camera feed.")
                break
                
            image, pred_class, confidence = engine.predict_frame(frame, holistic)
            
            if pred_class:
                # Text indicator
                cv2.putText(image, f'{pred_class} ({confidence:.2f})', (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
                            
                # Out-loud Speech Call trigger check
                if pred_class != last_spoken and confidence > Config.CONFIDENCE_THRESHOLD:
                    print(f"Detected Event Match: {pred_class} (Conf: {confidence:.2f})")
                    speak_text(pred_class)
                    last_spoken = pred_class
            
            # Show rendering window
            cv2.imshow("Real-Time Sign-Language Interpretation", image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
