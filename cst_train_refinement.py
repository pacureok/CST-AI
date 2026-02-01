# /home/oficialpacureok/Open-Sora/cst_train_refinement.py
import torch
import torch.nn.functional as F
from cst_engine import cst

class CST_Refinement:
    def __init__(self, model):
        self.model = model
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        print("[*] CST Refinement: Sistema de auto-mejora activo.")

    def calculate_realism_loss(self, generated_frames, target_style_latent):
        """
        Compara la textura generada con el 'ideal' de realismo B.
        """
        # Loss de contraste local (para evitar que sea borroso)
        grad_gen = torch.abs(generated_frames[:, :, 1:] - generated_frames[:, :, :-1])
        loss = F.mse_loss(generated_frames, target_style_latent) + 0.1 * grad_gen.mean()
        return loss

    def train_step(self, prompt, audio_path):
        self.optimizer.zero_grad()
        
        # 1. Generar con audio (A)
        audio_latent = cst.get_audio_features(audio_path)
        output = self.model(prompt, audio_latent)
        
        # 2. Aplicar corrección de Realismo (B)
        loss = self.calculate_realism_loss(output, target_style_latent=None) # Aquí se usa el scraper data
        
        # 3. Optimizar solo las capas CST
        loss.backward()
        self.optimizer.step()
        
        print(f"[+] Step completado. Pérdida de realismo: {loss.item():.4f}")

# Inicialización
# refinement = CST_Refinement(mi_modelo_mmdit)