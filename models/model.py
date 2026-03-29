import torch
import torch.nn as nn

class BiLSTMAttention(nn.Module):
    """
    BiLSTM kết hợp Cơ chế Attention.
    Tại sao chọn model này:
    - BiLSTM: Cực kỳ mạnh trong việc nắm bắt yếu tố ngữ cảnh theo chiều thời gian tiến và lùi (những chuyển động tay sign language phụ thuộc cả trước và sau).
    - Attention: Các frame trong video có ý nghĩa khác nhau, frame đứng im không có giá trị, trong khi frame giơ tay/cử chỉ phức tạp chứa nhiều info.
    - Dropout + Layer Norm: Giúp chống hiệu ứng overfitting trên dữ liệu không quá lớn và làm quá trình học mượt mà hơn.
    """
    def __init__(self, input_size=258, hidden_size=256, num_layers=3, num_classes=10, dropout=0.3):
        super(BiLSTMAttention, self).__init__()
        
        # RNN layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True, 
                            dropout=dropout if num_layers > 1 else 0)
        
        # Layer norm giúp ổn định hơn Batch Norm khi dùng với chuỗi độ dài không cố định,
        # và kết hợp tốt với LSTM (dùng cho hidden size tổng sau bi-direction là hidden*2)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
        # Mạng Self-Attention siêu đơn giản bằng MLP 1 lớp
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)  # Softmax trên trục seq_len (time)
        )
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len=30, input_size=258)
        
        # Truyền qua BiLSTM
        lstm_out, (hn, cn) = self.lstm(x) 
        # lstm_out shape: (batch_size, seq_len, hidden_size * 2)
        
        # Chuẩn hóa trạng thái
        lstm_out = self.layer_norm(lstm_out)
        
        # Tính trọng số Attention cho từng frame
        attn_weights = self.attention(lstm_out)
        # attn_weights shape: (batch_size, seq_len, 1)
        
        # Nhân element-wise vs trạng thái gốc để tính context vector (tổng trên time dimension)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)
        # context_vector shape: (batch_size, hidden_size * 2)
        
        # Phân loại
        out = self.fc(context_vector)
        # out shape: (batch_size, num_classes)
        
        return out