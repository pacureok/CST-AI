import torch
import torch_xla.core.xla_model as xm
import os

class CST_AutoSafe:
    def __init__(self, model, save_path="/kaggle/working/checkpoints"):
        self.model = model
        self.save_path = save_path
        self.best_score = -1.0
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print(f"[🛡️] AutoSafe Activado. Guardando en: {save_path}")

    def check_and_save(self, current_score, epoch):
        if current_score > self.best_score:
            self.best_score = current_score
            filename = os.path.join(self.save_path, "cst_step4_best.pt")
            # xm.save es el estándar para serializar en TPU
            xm.save(self.model.state_dict(), filename)
            print(f"[💾] Record de Realismo: {current_score:.4f} (Epoch {epoch}). Guardado.")
            return True
        return False
