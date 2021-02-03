import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class CREncoder(nn.Module):
    def __init__(self, bits, layers = 6):
        super(CREncoder, self).__init__()
        self.posenc = PositionalEncoding(2 * 16, dropout = 0)
        # self.posenc = PositionalEncoding(16, dropout = 0)
        # self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(2, 2), 6)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(2 * 16, 8, dim_feedforward = 2048, dropout = 0, activation = "gelu"), layers)
        # self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(16, 8, dropout = 0), 6)
        self.fc = nn.Sequential(
            nn.Linear(768, bits // 4)
        )
        self.sig = nn.Sigmoid()


    def forward(self, x):
        # out = x.permute(2, 3, 0, 1).reshape(24 * 16, -1, 2)
        out = x.permute(2, 0, 1, 3).reshape(24, -1, 2 * 16)
        # out = x.permute(1, 2, 0, 3).reshape(2 * 24, -1, 16)
        out = self.posenc(out)
        out = self.enc(out)
        out = out.permute(1, 0, 2).reshape(x.shape[0], -1)
        out = self.fc(out)
        out = self.sig(out)
        return out

class CRDecoder(nn.Module):
    def __init__(self, bits, layers = 6):
        super(CRDecoder, self).__init__()   
        # self.posenc = PositionalEncoding(2)  
        self.posenc = PositionalEncoding(2 * 16, dropout = 0) 
        # self.posenc = PositionalEncoding(16, dropout = 0) 
        self.fc = nn.Sequential(
            nn.Linear(bits // 4, 768)
        )
        self.relu = nn.LeakyReLU(0.3)
        # self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(2, 2), 6)
        self.dec = nn.TransformerEncoder(nn.TransformerEncoderLayer(2 * 16, 8, dim_feedforward = 2048, dropout = 0, activation = "gelu"), layers)
        self.dec.layers[-1].norm2 = nn.Identity()
        # self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(16, 8, dropout = 0), 6)
        # self.fc2 = nn.Sequential(
        #     nn.Linear(768, 768)
        # )
        
    def forward(self, x):
        out = self.fc(x)
        out = self.relu(out)
        # out = out.reshape(-1, 24 * 16, 2).permute(1, 0, 2)
        out = out.reshape(-1, 24, 2 * 16).permute(1, 0, 2)
        # out = out.reshape(-1, 2 * 24, 16).permute(1, 0, 2)
        out = self.posenc(out)
        out = self.dec(out)
        # out = out.reshape(24, 16, -1, 2).permute(2, 3, 0, 1)
        out = out.reshape(24, -1, 2, 16).permute(1, 2, 0, 3)
        # out = out.reshape(2, 24, -1, 16).permute(2, 0, 1, 3)
        # out = out.reshape(-1, 768)
        # out = self.fc2(out)
        # out = out.reshape(-1, 2, 24, 16)
        return out


if __name__ == "__main__":
    from torchviz import make_dot
    e = CREncoder(400)
    d = CRDecoder(400)
    a = torch.randn(10, 2, 24, 16)
    b = e(a)
    make_dot(b, dict(e.named_parameters())).view("encoder", cleanup = True)
    pass
    b = torch.tensor(b.detach().numpy())
    c = d(b)
    make_dot(c, dict(d.named_parameters())).view("decoder", cleanup = True)
    pass
