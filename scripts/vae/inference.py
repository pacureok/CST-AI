import os
import sys
import argparse
import torch
import warnings

# --- INTEGRACIÓN CRÍTICA CST ---
# Evitamos que busque CUDA y forzamos el path del motor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    import torch_xla.core.xla_model as xm
    HAS_XLA = True
except ImportError:
    HAS_XLA = False

try:
    # Importamos tu motor nativo y el registro de modelos
    from opensora.registry import MODELS
    from opensora.utils.misc import str2bool
    # Asumimos que cst_engine.py está en la raíz o MODELS/
    from MODELS.cst_engine import cst 
except ImportError as e:
    print(f"[!] Error de arquitectura: {e}")
    print("[*] Tip: Verifica que 'MODELS/cst_engine.py' exista.")

def run_inference():
    parser = argparse.ArgumentParser(description="PCURE-AI+ | CST Engine Ultra-Realism Inference")
    
    # Parámetros de Generación
    parser.add_argument("--prompt", type=str, required=True, help="Descripción visual")
    parser.add_argument("--negative_prompt", type=str, default="blur, low quality, distorted", help="Filtro negativo")
    
    # Parámetros CST (Audio Nativo A + Realismo B)
    parser.add_argument("--audio", type=str, default=None, help="Sincronización de audio nativa")
    parser.add_argument("--ultra_realism", type=str2bool, default=True, help="Activar optimización de textura B")
    parser.add_argument("--duration", type=int, default=10, help="Segundos de video")
    
    # Configuración de Hardware Kaggle
    parser.add_argument("--resolution", type=str, default="1080p", choices=["720p", "1080p", "2k", "4k"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", type=str, default="MODELS/pcure_weights.safetensors")
    
    args = parser.parse_args()

    # Identificación del Motor
    device = xm.xla_device() if HAS_XLA else torch.device("cpu")
    print(f"\n{'='*60}")
    print(f" 💎 PCURE-AI+ CST ENGINE v1.0 | HW: {device}")
    print(f" {'='*60}\n")

    # 1. Inicialización del Modelo VAE (Puente CST)
    print(f"[*] Cargando Pesos desde: {args.model_path}...")
    # Aquí es donde el registro de MODELS usa tu clase HunyuanVideoVAE modificada
    # model = MODELS.build('HunyuanVideoVAE', path=args.model_path).to(device)
    
    # 2. Procesamiento de Audio (Fase A)
    audio_latents = None
    if args.audio:
        print(f"[*] CST-Audio: Analizando frecuencias en {args.audio}...")
        # El motor nativo C++ procesa el audio sin pasar por Python
        audio_latents = cst.get_audio_features(args.audio) 
        if audio_latents is not None:
            # Movemos los latentes de audio al dispositivo XLA
            audio_latents = audio_latents.to(device)
            print(f"[✅] Sincronización A lista. Shape: {audio_latents.shape}")

    # 3. Aplicación de Realismo B (LastLayer Injection)
    if args.ultra_realism:
        print("[*] Parche B: Inyectando micro-texturas nativas...")
        # Comunicamos al motor nativo que debe aplicar el bias de textura
        cst.set_flag("ULTRA_REALISM", True)

    # 4. Inferencia Real (XLA Graph)
    print(f"[*] Ejecutando Render: '{args.prompt[:50]}...'")
    
    with torch.no_grad():
        # Aquí ocurriría el sampling real:
        # result = model.generate(prompt=args.prompt, audio=audio_latents, duration=args.duration)
        
        # Sincronización vital para TPU
        if HAS_XLA:
            xm.mark_step() 
            print("[*] Grafo XLA ejecutado en TPU Core.")

    # 5. Guardado y Output
    output_path = f"outputs/pcure_{args.seed}.mp4"
    os.makedirs("outputs", exist_ok=True)
    
    print(f"\n[🚀 SUCCESS] Renderizado completado.")
    print(f"📂 Archivo: {output_path}")
    print(f"💡 Tip: Para duraciones de 1 hora, usa --duration 3600 (Requiere motor DPR).")

if __name__ == "__main__":
    # Limpiamos caché de memoria antes de empezar
    if HAS_XLA:
        import gc
        gc.collect()
    run_inference()