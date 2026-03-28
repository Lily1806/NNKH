# Hệ Thống Giao Tiếp 2 Chiều: Ngôn Ngữ Ký Hiệu ↔ Văn Bản / Giọng Nói

![Hệ thống nhận diện ngôn ngữ ký hiệu thông minh]

Hệ thống cung cấp một giải pháp toàn diện cho người câm điếc và người bình thường:
1. Nhận diện ngôn ngữ ký hiệu từ video/webcam → chuyển thành văn bản + giọng nói sử dụng MediaPipe Holistic và mạng nơ-ron hồi quy học sâu (LSTM/GRU).
2. Chuyển văn bản/giọng nói → hiển thị lại bằng video ngôn ngữ ký hiệu.

---

## 📂 Tổ chức mã nguồn: Kiến trúc Model-View

Hệ thống được tổ chức với cấu hình logic module hoàn chỉnh:
```text
project_root/
├── dataraw/                # Dữ liệu gốc dùng để pre-processing thành keypoint và render. Mặc định có sẵn nội dung
├── data/
│   └── processed/          # Thư mục lưu .npy keypoints đã xử lý qua MediaPipe
├── models/
│   ├── saved_models/       # Chứa params saved pytorch models
│   └── model.py            # Khai báo kiến trúc Model Core Engine
├── core/                   # ROOT LOGIC CLASSES
│   ├── preprocessing.py    # Xử lý video pipeline -> keypoint
│   ├── dataset.py          # Abstract Pytorch Dataset 
│   ├── trainer.py          # Vòng lặp Training Core Process
│   ├── evaluator.py        # Abstract Metrics Evaluator
│   └── inference_engine.py # Vòng lặp dự đoán realtime
├── services/               # TIỆN ÍCH NGOÀI LỀ LOGIC MODEL
│   ├── speech_to_text.py   # SpeechRecognition API Wrapper
│   ├── text_to_speech.py   # gTTS API wrappers 
│   └── text_to_sign.py     # Ghép Clip Moviepy 
├── app/
│   └── streamlit_app.py    # GUI User Interface
├── configs/
│   └── config.py           # Khai báo Hằng số Configuration Global Settings
├── train.py                # Điểm chốt chạy Code Training
├── inference.py            # Điểm chốt chạy Code Inference Native Window
├── requirements.txt        # PIP Dependency
└── README.md               # Document file.
```

---

## 🚀 Hướng Dẫn Sử Dụng và Cài Đặt

### 1. Chuẩn Bị Môi Trường Cài Đặt
Bảo đảm thiết bị có python từ 3.8 ~ 3.10
Mở Terminal khởi chạy lệnh chèn thư viện sau:
```bash
pip install -r requirements.txt
```

### 2. Vận Hành Processing Data (Tiền xử lý học liệu ảnh)
Kiểm tra rằng thư mục `dataraw/train` đã có nội dung các thư mục lớp nhãn như "Ăn", "Uống" với sub-files là `.mp4`.
```bash
python core/preprocessing.py
```
> Thao tác này sẽ phân tích và tạo folder `data/processed` cùng dictionary numpy map path `dataraw/label_mapping.npy`.

### 3. Huấn luyện hệ thống Model (Tạo Trí Nhớ Nhân Tạo)
Để chạy quá trình train trên Custom PyTorch Train Loop sử dụng CPU/MPS/CUDA:
```bash
python train.py
```
> Trọng số tham số hiệu quả nhất sẽ đưa lưu theo file config tại `models/saved_models/best_model.pth`.

### 4. Khởi Chạy Giao Diện Ứng Dụng Đa Phương Tiện (Streamlit App)
GUI App đem lại 2 Tabs hỗ trợ đồng thời cả Nhận diện từ Camera và Dịch từ Tiếng nói/Văn Bản -> Video Clip:
```bash
streamlit run app/streamlit_app.py
```

### 5. (Tùy chọn) Chạy Hệ Thống OpenCV CLI Core
```bash
python inference.py
```

---
> ⚠️ **Ghi chú tính tương thích:** Đối với service `text_to_sign.py` cần sử dụng `moviepy` -> Vui lòng cài `ffmpeg` trên local PC/Mac thông qua Terminal/Brew nếu Video Render gặp Warning Codec.
