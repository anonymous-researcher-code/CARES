import os
import subprocess

def main():
    print("[CARES] Initializing generative model fine-tuning...")

    config_file = "training_args.yaml"
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file {config_file} not found.")

    command = ["llamafactory-cli", "train", config_file]
    
    try:
        subprocess.run(command, check=True)
        print("[CARES] Model fine-tuning finished successfully.")
    except subprocess.CalledProcessError as e:
        print(f"[CARES] Error during training: {e}")

if __name__ == "__main__":
    main()