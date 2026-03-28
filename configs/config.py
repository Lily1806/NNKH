import os

class Config:
    """Central config"""

    # =========================
    # 📁 PATH (FIX CHO COLAB + DRIVE)
    # =========================
    PROJECT_ROOT = "/content/drive/MyDrive/NNKH/NhanDienNNKH"

    DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "dataraw")
    DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

    DATA_PROCESSED_TRAIN = os.path.join(DATA_PROCESSED_DIR, "train")
    DATA_PROCESSED_PUBLIC_TEST = os.path.join(DATA_PROCESSED_DIR, "public_test")
    DATA_PROCESSED_PRIVATE_TEST = os.path.join(DATA_PROCESSED_DIR, "private_test")

    MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "saved_models")
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")

    LABEL_MAPPING_PATH = os.path.join(DATA_RAW_DIR, "label_mapping.pkl")

    # =========================
    # 🎯 DATA CONFIG
    # =========================
    MAX_FRAMES = 30
    KEYPOINT_DIM = 258

    # =========================
    # 🧠 MODEL CONFIG (UPGRADE)
    # =========================
    HIDDEN_SIZE = 256       # tăng từ 128 → học tốt hơn
    NUM_LAYERS = 3          # tăng depth
    DROPOUT = 0.3
    MODEL_TYPE = "LSTM"

    # =========================
    # 🔥 TRAINING CONFIG
    # =========================
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 0.0005

    WEIGHT_DECAY = 1e-5     # chống overfit
    EARLY_STOPPING = 5      # dừng sớm

    # =========================
    # 🎯 INFERENCE
    # =========================
    CONFIDENCE_THRESHOLD = 0.7
    VOTE_WINDOW = 5

    @classmethod
    def setup_directories(cls):
        os.makedirs(cls.DATA_PROCESSED_TRAIN, exist_ok=True)
        os.makedirs(cls.DATA_PROCESSED_PUBLIC_TEST, exist_ok=True)
        os.makedirs(cls.DATA_PROCESSED_PRIVATE_TEST, exist_ok=True)
        os.makedirs(cls.MODEL_DIR, exist_ok=True)