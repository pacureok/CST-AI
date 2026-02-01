# /home/oficialpacureok/Open-Sora/cst_scraper.py
import os

def generate_scraping_list(keywords=["4k cinematic ultra realistic", "nature documentary 8k"]):
    print("[*] Generando lista de entrenamiento para Pcure-AI+...")
    # Este comando usa yt-dlp (debes incluirlo en requirements) para buscar videos
    for kw in keywords:
        print(f"[+] Buscando contenido: {kw}")
        # Comando sugerido para la terminal:
        # yt-dlp --max-downloads 10 --format mp4 "ytsearchall:{kw}"

if __name__ == "__main__":
    generate_scraping_list()