import base64
import os
from mimetypes import guess_type
from openai import OpenAI

# ======================================================
# Prompts de Sistema (Instrucciones para la IA)
# ======================================================

# Instrucción para Text-to-Video (T2V)
sys_prompt_t2v = """Eres parte de un equipo que crea videos. Tu tarea es tomar prompts cortos 
y hacerlos extremadamente detallados y descriptivos. No solo los hagas más largos, 
hazlos visualmente ricos siguiendo los ejemplos proporcionados."""

# Instrucción para Image-to-Video (I2V)
sys_prompt_i2v = """Crea una descripción de video basada en una imagen de entrada. 
Describe el movimiento comenzando desde esa imagen. Debes incluir información dinámica 
(acciones, tramas) para asegurar que el video tenga movimiento real."""

# Instrucción para puntaje de movimiento (Motion Score)
sys_prompt_motion_score = """Predice un puntaje de movimiento VMAF (1-15) para el prompt. 
4 es para modelos en pasarela, 1 es para videos estáticos."""

def image_to_url(image_path):
    """Convierte una imagen local a un formato URL Base64 para que GPT-4o pueda verla."""
    mime_type, _ = guess_type(image_path)
    if mime_type is None: mime_type = "application/octet-stream"
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{base64_encoded_data}"

def refine_prompt(prompt: str, retry_times: int = 3, type: str = "t2v", image_path: str = None):
    """
    Utiliza la API de OpenAI (GPT-4o) para mejorar el prompt original.
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    text = prompt.strip()
    
    # Lógica de refinamiento según el tipo (t2v, t2i, i2v, motion_score)
    for i in range(retry_times):
        if type == "t2v":
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": f"{sys_prompt_t2v}"},
                    # ... (incluye ejemplos de Few-Shot en el código original para guiar a la IA)
                    {"role": "user", "content": f'Crea una descripción creativa en INGLÉS para: " {text} "'}
                ],
                model="gpt-4o",
                temperature=0.01,
                max_tokens=250,
            )
        # ... (lógica similar para i2v usando image_to_url)
        
        if response and response.choices:
            return response.choices[0].message.content
    return prompt

def refine_prompts(prompts: list[str], retry_times: int = 3, type: str = "t2v", image_paths: list[str] = None):
    """Procesa una lista de prompts de forma secuencial."""
    if image_paths is None: image_paths = [None] * len(prompts)
    return [refine_prompt(p, retry_times, type, img) for p, img in zip(prompts, image_paths)]