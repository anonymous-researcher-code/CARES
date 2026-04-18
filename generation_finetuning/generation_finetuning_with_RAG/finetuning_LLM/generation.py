import os
import subprocess

def main():
    print("[CARES] Initializing generative model inference and prediction...")

    config_file = "generate_args.yaml"
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"[CARES] Error: Configuration file '{config_file}' not found.")

    # In LLaMA-Factory, batch generation/evaluation via YAML uses the 'train' command; do_predict in YAML triggers inference.
    command = ["llamafactory-cli", "train", config_file]
    
    try:
        print(f"[CARES] Running generation with config: {config_file}")
        subprocess.run(command, check=True)
        print("[CARES] Model generation finished successfully. Predictions saved.")
    except subprocess.CalledProcessError as e:
        print(f"[CARES] Error during generation: {e}")

if __name__ == "__main__":
    main()