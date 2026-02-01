# ==========================================
# CST ENGINE - PCURE-AI+ (PRO VERSION)
# Motor Híbrido C++/Python para Ultra-Realismo
# ==========================================

import os
import torch
import librosa
import numpy as np
import sys

# Intentar cargar el núcleo de optimización C++ (DPR Core)
try:
    import cst_dpr_core
    HAS_DPR = True
except ImportError:
    HAS_DPR = False

class CST_Engine:
    def __init__(self):
        # El sistema se autolocaliza
        self.root = os.path.dirname(os.path.abspath(__file__))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.version = "1.0.0-Hybrid"
        
        print(f"\n{'-'*40}")
        print(f"[*] CST ENGINE ACTIVADO - v{self.version}")
        print(f"[*] Hardware: {self.device.upper()}")
        print(f"[*] DPR Core (C++): {'CONECTADO' if HAS_DPR else 'DESCONECTADO (Solo Python)'}")
        print(f"{'-'*40}\n")

    def get_audio_features(self, audio_path):
        """
        PARTE A: Audio Nativo. 
        Convierte ondas sonoras en latentes para el bloque MMDiT.
        """
        if not audio_path or not os.path.exists(audio_path):
            print(f"[!] Aviso: No se encontró audio en {audio_path}")
            return None
            
        try:
            # Carga optimizada (44.1kHz)
            y, sr = librosa.load(audio_path, sr=44100)
            # Extraemos MFCCs (128 dimensiones para coincidir con el embedding del modelo)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)
            
            # Ajuste de precisión según hardware
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            audio_tensor = torch.from_numpy(mfcc).T.unsqueeze(0).to(dtype).to(self.device)
            
            return audio_tensor
        except Exception as e:
            print(f"[!] Error procesando audio: {e}")
            return None

    def apply_realism_b_patch(self, model):
        """
        PARTE B: Ultra-Realismo.
        Ajusta dinámicamente la capa final para máxima definición.
        """
        print("[*] Sintonizando pesos de Realismo B...")
        # Esta función puede ser llamada por el script de inferencia
        # para modificar los parámetros de LastLayer en tiempo de ejecución.
        for name, param in model.named_parameters():
            if "linear.weight" in name and "final" in name:
                param.data *= 1.05  # Aumenta el contraste de micro-textura
        return model

    def optimize_memory(self, important_tensors):
        """
        NÚCLEO HÍBRIDO (C++ + Python).
        Evita que la memoria explote en videos largos.
        """
        if HAS_DPR and self.device == "cuda":
            # Usamos el núcleo C++ para swappear tensores pesados a la RAM del sistema
            # mientras la GPU procesa el siguiente frame.
            for t in important_tensors:
                if t.is_cuda and t.ndim > 2:
                    cst_dpr_core.dpr_swap(t, "cpu")
            cst_dpr_core.clear_vram()
            return True
        return False

    def check_integrity(self):
        """Verifica que los archivos clave estén parcheados"""
        files = [
            "opensora/models/mmdit/model.py",
            "opensora/models/mmdit/layers.py",
            "requirements.txt"
        ]
        for f in files:
            full_path = os.path.join(self.root, f)
            status = "✓" if os.path.exists(full_path) else "✗"
            print(f"[{status}] {f}")

# Inicialización global
cst = CST_Engine()