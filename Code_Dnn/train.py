import torch
import torch.nn as nn
from tqdm import tqdm


class Trainer:
    def __init__(self, text_encoder, decoder, fusion, sequence_model, device="cuda"):

        self.text_encoder = text_encoder
        self.decoder = decoder
        self.fusion = fusion
        self.sequence_model = sequence_model

        self.device = device

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        self.optimizer = torch.optim.Adam(
            list(text_encoder.parameters()) +
            list(decoder.parameters()) +
            list(fusion.parameters()) +
            list(sequence_model.parameters()),
            lr=1e-3
        )

        self.text_encoder.to(device)
        self.decoder.to(device)
        self.fusion.to(device)
        self.sequence_model.to(device)

    def train_epoch(self, dataloader):
        self.text_encoder.train()
        self.decoder.train()
        self.fusion.train()
        self.sequence_model.train()

        total_loss = 0

        for img_feats, captions in tqdm(dataloader, desc="Training"):
            img_feats = img_feats.to(self.device)       # (B, 2048)
            captions = captions.to(self.device)         # (B, seq)

            # Text encoder output (B, seq, 1024)
            txt_encoded = self.text_encoder(captions)

            # Fusion
            fused = self.fusion(img_feats, txt_encoded)

            # Sequence model
            seq_out = self.sequence_model(fused)

            # Decoder
            logits = self.decoder.fc(seq_out)

            # Teacher forcing loss
            loss = self.criterion(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                captions[:, 1:].reshape(-1)
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.text_encoder.eval()
        self.decoder.eval()
        self.fusion.eval()
        self.sequence_model.eval()

        total_loss = 0

        with torch.no_grad():
            for img_feats, captions in dataloader:
                img_feats = img_feats.to(self.device)
                captions = captions.to(self.device)

                txt_encoded = self.text_encoder(captions)
                fused = self.fusion(img_feats, txt_encoded)
                seq_out = self.sequence_model(fused)
                logits = self.decoder.fc(seq_out)

                loss = self.criterion(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    captions[:, 1:].reshape(-1)
                )

                total_loss += loss.item()

        return total_loss / len(dataloader)
