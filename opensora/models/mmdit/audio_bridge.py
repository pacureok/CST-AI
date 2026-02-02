import torch
import torch.nn as nn

class AudioLatentEncoder(nn.Module):
    def __init__(self, audio_channels=1, embed_dim=128):
        super().__init__()
        # Un codificador simple para convertir espectrogramas en vectores compatibles con CST
        self.encoder = nn.Sequential(
            nn.Conv1d(audio_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool1d(1) # Esto nos da un vector de contexto global del audio
        )

    def forward(self, waveform):
        # waveform: [Batch, Channels, Time]
        latents = self.encoder(waveform)
        return latents.squeeze(-1) # Output: [Batch, 128]