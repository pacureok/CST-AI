#!/usr/bin/env python
import argparse
import datetime
import importlib
import os
import subprocess
import sys
from tempfile import NamedTemporaryFile

import spaces
import torch
import gradio as gr

# Configuración de Rutas y Modelos PCURE-AI+
MODEL_TYPES = ["v1.3"]
WATERMARK_PATH = "./assets/images/watermark/watermark.png"
CONFIG_MAP = {
    "v1.3": "configs/opensora-v1-3/inference/t2v.py",
    "v1.3_i2v": "configs/opensora-v1-3/inference/v2v.py",
}
HF_STDIT_MAP = {
    "t2v": {
        "360p": "hpcaitech/OpenSora-STDiT-v4-360p",
        "720p": "hpcaitech/OpenSora-STDiT-v4",
    },
    "i2v": "hpcaitech/OpenSora-STDiT-v4-i2v",
}

# ============================
# Optimizadores de Entorno
# ============================
def install_dependencies(enable_optimization=False):
    def _is_package_available(name):
        try:
            importlib.import_module(name)
            return True
        except (ImportError, ModuleNotFoundError):
            return False

    if enable_optimization:
        packages = [("flash_attn", "flash-attn"), ("ninja", "ninja")]
        for pkg_name, pip_name in packages:
            if not _is_package_available(pkg_name):
                subprocess.run(f"{sys.executable} -m pip install {pip_name}", shell=True)

# ============================
# Motor de Carga PCURE
# ============================
def build_models(mode, resolution, enable_optimization=False):
    from opensora.registry import MODELS, build_module
    from opensora.models.stdit.stdit3 import STDiT3
    from opensora.registry import SCHEDULERS

    config_path = CONFIG_MAP["v1.3_i2v"] if mode == "i2v" else CONFIG_MAP["v1.3"]
    from mmengine.config import Config
    config = Config.fromfile(config_path)

    # 1. Cargar VAE (Componente Crítico para Realismo B)
    vae = build_module(config.vae, MODELS).cuda().to(torch.bfloat16).eval()

    # 2. Cargar Text Encoder (T5)
    text_encoder = build_module(config.text_encoder, MODELS)
    text_encoder.t5.model = text_encoder.t5.model.cuda().eval()

    # 3. Cargar STDiT (Transformer de Video)
    weight_path = HF_STDIT_MAP["i2v"] if mode == "i2v" else HF_STDIT_MAP["t2v"].get(resolution)
    model_kwargs = {k: v for k, v in config.model.items() if k not in ("type", "from_pretrained", "force_huggingface")}
    stdit = STDiT3.from_pretrained(weight_path, **model_kwargs).cuda().to(torch.bfloat16).eval()

    # 4. Sincronizar Scheduler y CFG
    scheduler = build_module(config.scheduler, SCHEDULERS)
    text_encoder.y_embedder = stdit.y_embedder

    torch.cuda.empty_cache()
    return vae, text_encoder, stdit, scheduler, config

# ============================
# Lógica de Inferencia Gradio
# ============================
@spaces.GPU(duration=200)
def run_inference(mode, prompt_text, resolution, aspect_ratio, length, motion_strength, 
                  aesthetic_score, use_motion, use_aesthetic, camera_motion, 
                  ref_img, refine_prompt, fps, num_loop, seed, sampling_steps, cfg_scale):
    
    if not prompt_text:
        return None

    # Autodetectar modo Image-to-Video
    if ref_img is not None and mode != "Text2Image":
        mode = "i2v"

    vae, text_encoder, stdit, scheduler, config = build_models(mode, resolution)
    
    torch.manual_seed(seed)
    with torch.inference_mode():
        from opensora.datasets.aspect import get_image_size, get_num_frames
        image_size = get_image_size(resolution, aspect_ratio)
        num_frames = 1 if mode == "Text2Image" else get_num_frames(length)
        
        # Preparación de Latentes
        latent_size = vae.get_latent_size((num_frames, *image_size))
        z = torch.randn(1, vae.out_channels, *latent_size, device="cuda", dtype=torch.bfloat16)

        # Manejo de Referencia (Parche B / I2V)
        from opensora.utils.inference_utils import collect_references_batch, prepare_multi_resolution_info
        refs = [""]
        if ref_img is not None:
            from PIL import Image
            temp = NamedTemporaryFile(suffix=".png", delete=False)
            Image.fromarray(ref_img).save(temp.name)
            refs = [temp.name]
        
        refs_data = collect_references_batch(refs, vae, image_size)
        model_args = prepare_multi_resolution_info("OpenSora", 1, image_size, num_frames, fps, "cuda", torch.bfloat16)

        # Diffusion Sampling
        samples = scheduler.sample(
            stdit, text_encoder, z=z, prompts=[prompt_text], device="cuda", 
            additional_args=model_args, cfg_scale=cfg_scale, num_sampling_steps=sampling_steps
        )

        # Decodificación VAE (Aquí se aplica el realismo final)
        video = vae.decode(samples, num_frames=num_frames).squeeze(0)
        
        from opensora.datasets import save_sample
        save_path = os.path.join("./outputs", f"pcure_{datetime.datetime.now().timestamp()}")
        return save_sample(video, save_path=save_path, fps=fps)

# ============================
# Interfaz de Usuario (UI)
# ============================
def main():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.HTML("<h1 style='text-align: center;'>PCURE-AI+ Video Engine</h1>")
        
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt Cinematográfico", placeholder="Un plano secuencia de...", lines=3)
                res = gr.Radio(["360p", "720p"], value="720p", label="Resolución")
                ratio = gr.Radio(["9:16", "16:9", "1:1"], value="16:9", label="Relación de Aspecto")
                length = gr.Slider(1, 113, value=97, step=16, label="Frames (Longitud)")
                
                with gr.Accordion("Configuración Avanzada", open=False):
                    seed = gr.Number(label="Semilla", value=1024)
                    steps = gr.Slider(20, 100, value=30, label="Pasos de Muestreo")
                    cfg = gr.Slider(1.0, 15.0, value=7.0, label="CFG Scale")
                
                ref_img = gr.Image(label="Imagen de Referencia (Opcional)")
                btn = gr.Button("Generar Obra Maestra", variant="primary")
                
            with gr.Column():
                output = gr.Video(label="Resultado PCURE-AI+")

        btn.click(fn=run_inference, inputs=[gr.State("Text2Video"), prompt, res, ratio, length, 
                  gr.State("fair"), gr.State("excellent"), gr.State(True), gr.State(True), 
                  gr.State("none"), ref_img, gr.State(False), gr.State(24), gr.State(1), 
                  seed, steps, cfg], outputs=output)

    demo.launch(server_name="0.0.0.0", share=True)

if __name__ == "__main__":
    main()