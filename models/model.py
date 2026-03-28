import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden)
        energy = torch.tanh(self.attn(lstm_output))  # (batch, seq_len, hidden)
        attention = self.v(energy).squeeze(-1)       # (batch, seq_len)
        weights = torch.softmax(attention, dim=1)    # (batch, seq_len)

        # weighted sum
        context = torch.sum(lstm_output * weights.unsqueeze(-1), dim=1)
        return context, weights


class SignLanguageModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, model_type="LSTM"):
        super(SignLanguageModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )

        self.attention = Attention(hidden_size)

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq, feature)
        lstm_out, _ = self.lstm(x)

        context, attn_weights = self.attention(lstm_out)

        out = self.dropout(context)
        out = self.fc(out)

        return out