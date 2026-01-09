import torch
import torch.nn as nn
import torchvision.models as models


# ---------------------------
# 1. IMAGE ENCODER (ResNet50)
# ---------------------------
class ImageEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        layers = list(resnet.children())[:-1]  # remove final FC layer
        self.encoder = nn.Sequential(*layers)
        self.output_dim = 2048  # ResNet50 final feature dimension

    def forward(self, x):
        x = self.encoder(x)    # (B, 2048, 1, 1)
        return x.squeeze()     # (B, 2048)


# ---------------------------
# 2. TEXT ENCODER (BiLSTM)
# ---------------------------
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.output_dim = hidden_dim * 2

    def forward(self, x):
        x = self.embed(x)
        outputs, _ = self.lstm(x)
        return outputs  # (B, seq_len, 1024)


# --------------------------------------------
# 3. BASELINE FUSION (Simple Concatenation)
# --------------------------------------------
class ConcatFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img_feat, txt_feat):
        # img_feat: (B, 2048)
        # txt_feat: (B, seq_len, 1024)
        img_expanded = img_feat.unsqueeze(1).expand(-1, txt_feat.size(1), -1)
        fused = torch.cat([img_expanded, txt_feat], dim=2)
        return fused  # (B, seq_len, 3072)


# --------------------------------------------
# 4. CROSS-MODAL ATTENTION (Innovation)
# --------------------------------------------
class CrossModalAttention(nn.Module):
    def __init__(self, img_dim=2048, txt_dim=1024, hidden_dim=512):
        super().__init__()
        self.key_layer = nn.Linear(img_dim, hidden_dim)
        self.query_layer = nn.Linear(txt_dim, hidden_dim)
        self.value_layer = nn.Linear(img_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, img_feat, txt_feat):
        # img_feat: (B, 2048)
        # convert to 1-token "visual memory"
        img_feat = img_feat.unsqueeze(1)  # (B, 1, 2048)

        K = self.key_layer(img_feat)      # (B, 1, H)
        V = self.value_layer(img_feat)    # (B, 1, H)
        Q = self.query_layer(txt_feat)    # (B, seq_len, H)

        attn_scores = torch.matmul(Q, K.transpose(1, 2)) / self.scale  # (B, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)               # attention over text

        attended = attn_weights * V                                     # (B, seq_len, H)
        fused = torch.cat([attended, txt_feat], dim=2)                  # (B, seq_len, H + txt_dim)

        return fused


# --------------------------------------------
# 5. SEQUENCE MODEL (GRU)
# --------------------------------------------
class SequenceModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        out, _ = self.gru(x)
        return out  # (B, seq_len, hidden_dim)


# --------------------------------------------
# 6. TEXT DECODER (LSTM)
# --------------------------------------------
class TextDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, captions):
        embedded = self.embed(captions)
        outputs, _ = self.lstm(embedded)
        logits = self.fc(outputs)
        return logits  # (B, seq_len, vocab)
