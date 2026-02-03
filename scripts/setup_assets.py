from huggingface_hub import hf_hub_download
import os

def main():
    print("🛰️ Descargando Pesos de Open Sora v2 para CST-AI...")
    repo_id = "hpcai-tech/Open-Sora-v2"
    
    # Descarga solo el binario necesario
    hf_hub_download(repo_id=repo_id, filename="sora_v2.pth", local_dir="models/weights")
    hf_hub_download(repo_id=repo_id, filename="config.json", local_dir="models/config")
    
    print("✅ Activos de Pacur AI+ listos.")

if __name__ == "__main__":
    main()
