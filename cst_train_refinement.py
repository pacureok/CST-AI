import torch
import torch_xla.core.xla_model as xm
from cst_engine import cst
from cst_judge import judge

class CST_TPU_Refinement:
    def __init__(self, model):
        self.device = xm.xla_device()
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        print("[⚡] Refinement Engine: Modo TPU v5e activo.")

    def train_step(self, prompt, audio_path):
        self.model.train()
        self.optimizer.zero_grad()
        audio_latents = cst.get_audio_features(audio_path)
        if audio_latents is not None:
            audio_latents = audio_latents.to(self.device)
        output = self.model(prompt, audio_latents)
        quality_score = judge.evaluate_realism(output)
        loss = 1.0 - quality_score
        loss.backward()
        xm.optimizer_step(self.optimizer)
        xm.mark_step()
        return quality_score
