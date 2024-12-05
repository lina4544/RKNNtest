import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        # 입력 임베딩 레이어
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout)
        
        # Transformer 인코더 레이어 쌓기
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 출력 레이어
        self.fc_out = nn.Linear(embed_dim, input_dim)
    
    def forward(self, src):
        # 입력 + 위치 임베딩
        embedded = self.embedding(src) * torch.sqrt(torch.tensor(self.embedding.embedding_dim, dtype=torch.float32))
        embedded = self.positional_encoding(embedded)
        
        # Transformer 인코더 적용
        encoded_output = self.encoder(embedded)
        
        # 마지막 출력
        output = self.fc_out(encoded_output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 위치 인코딩 계산
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
