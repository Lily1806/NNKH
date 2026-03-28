import os
import cv2
import torch
import numpy as np
from collections import deque
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import Config
from core.preprocessing import mediapipe_detection, extract_keypoints

class InferenceEngine:
    """
    Handles real-time webcam inference using the trained sign language model.
    """
    def __init__(self, model, classes, device="cpu"):
        self.model = model
        self.classes = classes
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        
        # Buffer to keep the last MAX_FRAMES
        self.sequence = deque(maxlen=Config.MAX_FRAMES)
        
        # Buffer to keep the last VOTE_WINDOW predictions for smoothing
        self.predictions = deque(maxlen=Config.VOTE_WINDOW)
        
    def predict_frame(self, frame, holistic_model):
        """
        Processes a single frame, updates buffers, and returns the prediction.
        """
        image, results = mediapipe_detection(frame, holistic_model)
        keypoints = extract_keypoints(results)
        
        # Append to sequence buffer
        self.sequence.append(keypoints)
        
        predicted_class = None
        confidence = 0.0
        
        # Only predict if we have collected enough frames
        if len(self.sequence) == Config.MAX_FRAMES:
            # Prepare input tensor: shape (1, MAX_FRAMES, KEYPOINT_DIM)
            input_data = np.expand_dims(self.sequence, axis=0)
            input_tensor = torch.tensor(input_data, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                res = self.model(input_tensor)
                probs = torch.softmax(res[0], dim=0)
                
            confidence, predicted_idx = torch.max(probs, dim=0)
            confidence = confidence.item()
            predicted_idx = predicted_idx.item()
            
            # Add to rolling predictions buffer for majority voting
            self.predictions.append(predicted_idx)
            
            # Apply Majority Voting smoothing
            if len(self.predictions) == Config.VOTE_WINDOW:
                unique_preds, counts = np.unique(self.predictions, return_counts=True)
                majority_idx = unique_preds[np.argmax(counts)]
                
                # Check confidence threshold for the majority vote
                if majority_idx == predicted_idx and confidence > Config.CONFIDENCE_THRESHOLD:
                    predicted_class = self.classes[majority_idx]
                    
        return image, predicted_class, confidence
