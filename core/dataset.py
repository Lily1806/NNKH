import os
import glob
import torch
import numpy as np
import unicodedata
from torch.utils.data import Dataset, DataLoader, random_split

def normalize_text(text):
    """Chuẩn hóa chuỗi Unicode (để tránh lỗi tiếng Việt: 1 chữ có 2 cách mã hóa)."""
    return unicodedata.normalize('NFC', text).strip()

class SignLanguageDataset(Dataset):
    """
    Dataset cho nhận diện ngôn ngữ ký hiệu.
    Tự động đọc các thư mục con trong data_dir làm label, bỏ qua việc dùng file mapping ngoài.
    Thực hiện padding / truncation với seq_length = max_frames để đảm bảo RNN feed được đều data.
    """
    def __init__(self, data_dir, max_frames=30, keypoint_dim=258):
        self.data_dir = data_dir
        self.max_frames = max_frames
        self.keypoint_dim = keypoint_dim
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Thư mục không tồn tại: {data_dir}")
            
        # Lấy danh sách các thư mục con
        original_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        if len(original_dirs) == 0:
            raise ValueError(f"Không tìm thấy thư mục class (nhãn) trong {data_dir}")
            
        # Tạo mapping tự động có chuẩn hóa tên
        self.classes = [normalize_text(d) for d in original_dirs]
        self.label_mapping = {normalize_text(cls_name): idx for idx, cls_name in enumerate(original_dirs)}
        
        self.samples = []
        for orig_dir in original_dirs:
            cls_dir = os.path.join(data_dir, orig_dir)
            cls_name_norm = normalize_text(orig_dir)
            label_idx = self.label_mapping[cls_name_norm]
            
            npy_files = glob.glob(os.path.join(cls_dir, "*.npy"))
            for f in npy_files:
                self.samples.append((f, label_idx))
                
        if len(self.samples) == 0:
            raise ValueError(f"Không tìm thấy file .npy nào trong thư mục {data_dir} và các thư mục con.")
            
        print(f"✅ Đã load {len(self.samples)} samples. Số classes: {len(self.classes)}")
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        file_path, label_idx = self.samples[idx]
        
        try:
            keypoints = np.load(file_path)
            # Nếu keypoints có shape sai, force nó về (frames, keypoint_dim)
            if len(keypoints.shape) != 2 or keypoints.shape[1] != self.keypoint_dim:
                keypoints = np.zeros((self.max_frames, self.keypoint_dim))
        except Exception as e:
            print(f"⚠️ Lỗi đọc file {file_path}: {e}")
            keypoints = np.zeros((self.max_frames, self.keypoint_dim))
            
        # Xử lý missing frames (Padding hoặc Truncation)
        frames = keypoints.shape[0]
        if frames < self.max_frames:
            # Padding bằng 0 ở cuối seq
            padding = np.zeros((self.max_frames - frames, self.keypoint_dim))
            keypoints = np.vstack((keypoints, padding))
        elif frames > self.max_frames:
            # Truncating lấy đoạn đầu
            keypoints = keypoints[:self.max_frames, :]
            
        # Normalize (tuỳ chọn) MediaPipe trả về giá trị tọa độ x, y từ 0 -> 1 rồi. 
        # Z cũng được chuẩn hóa tương đối nên không cần thiết scale. Chỉ chuyển về float32.
        x = torch.tensor(keypoints, dtype=torch.float32)
        y = torch.tensor(label_idx, dtype=torch.long)
        
        return x, y

def get_dataloaders(data_dir, batch_size=32, max_frames=30, keypoint_dim=258, val_split=0.2, seed=42):
    """
    Tự động chia tập train/val đảm bảo tính ngẫu nhiên nhưng lặp lại (seed)
    Và tối ưu num_workers.
    """
    torch.manual_seed(seed)
    
    dataset = SignLanguageDataset(data_dir, max_frames, keypoint_dim)
    
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Tính số workers cho dataloader
    # Trong colab nên dùng tối thiểu cpucount.
    num_workers = min(os.cpu_count() or 2, 4) 
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        drop_last=True if train_size > batch_size else False,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, dataset.classes