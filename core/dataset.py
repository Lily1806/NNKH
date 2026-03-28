import os
import glob
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import Config


class SignLanguageDataset(Dataset):
    """
    Dataset dùng label_mapping.pkl + folder structure
    """

    def __init__(self, processed_dir, label_mapping_path=Config.LABEL_MAPPING_PATH):
        self.processed_dir = processed_dir

        # =========================
        # 📌 LOAD LABEL MAPPING
        # =========================
        if not os.path.exists(label_mapping_path):
            raise FileNotFoundError(f"❌ Không tìm thấy mapping: {label_mapping_path}")

        with open(label_mapping_path, 'rb') as f:
            self.label_mapping = pickle.load(f)

        print(f"📌 Loaded {len(self.label_mapping)} classes from mapping")

        self.samples = []

        # =========================
        # 📌 LOAD FILES THEO FOLDER
        # =========================
        for cls_name, label_idx in self.label_mapping.items():

            cls_dir = os.path.join(self.processed_dir, cls_name)

            if not os.path.exists(cls_dir):
                print(f"⚠️ Missing folder: {cls_name}")
                continue

            npy_files = glob.glob(os.path.join(cls_dir, "*.npy"))

            if len(npy_files) == 0:
                print(f"⚠️ Empty class: {cls_name}")
                continue

            for file_path in npy_files:
                self.samples.append((file_path, label_idx))

        # =========================
        # 📌 CHECK
        # =========================
        if len(self.samples) == 0:
            raise ValueError("❌ No samples loaded! Mapping không khớp folder!")

        print(f"✅ Loaded {len(self.samples)} samples from {processed_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label_idx = self.samples[idx]

        keypoints = np.load(file_path)

        # 🔥 đảm bảo shape đúng
        if keypoints.shape != (Config.MAX_FRAMES, Config.KEYPOINT_DIM):
            keypoints = np.resize(keypoints, (Config.MAX_FRAMES, Config.KEYPOINT_DIM))

        x = torch.tensor(keypoints, dtype=torch.float32)
        y = torch.tensor(label_idx, dtype=torch.long)

        return x, y

    @property
    def num_classes(self):
        return len(self.label_mapping)

    @property
    def classes(self):
        # sort theo index
        inverse = {v: k for k, v in self.label_mapping.items()}
        return [inverse[i] for i in range(len(inverse))]