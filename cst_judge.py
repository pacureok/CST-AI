import torch
import torch.nn.functional as F

class CST_Judge_TPU:
    def __init__(self):
        self.precision = torch.bfloat16 
        print("[⚖️] Juez de Realismo: Operativo en TPU (BF16)")

    def evaluate_realism(self, video_tensor):
        # Análisis de alta frecuencia para detectar texturas reales
        with torch.no_grad():
            # Si el tensor es 5D [B, C, T, H, W], tomamos un frame central
            if video_tensor.dim() == 5:
                frame = video_tensor[:, :, video_tensor.size(2)//2, :, :]
            else:
                frame = video_tensor
                
            frame = frame.to(dtype=torch.float32)
            grad_x = torch.abs(frame[:, :, :, 1:] - frame[:, :, :, :-1]).mean()
            grad_y = torch.abs(frame[:, :, 1:, :] - frame[:, :, :-1, :]).mean()
            return (grad_x + grad_y).item()

judge = CST_Judge_TPU()
