import subprocess
import signal
import sys
import os

from dotenv import load_dotenv

load_dotenv("model.env")

torchserve_process = None

def stop_torchserve():
    if torchserve_process:
        print("Stopping TorchServe...")
        subprocess.run(["torchserve", "--stop"], check=True) # Stop TorchServe
        torchserve_process.terminate()
        torchserve_process.wait()

def signal_handler(sig, frame):
    stop_torchserve()
    sys.exit(0)

if __name__ == "__main__":

    # Handle termination signals
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Generate .mar files
    print("Generating .mar files...")
    subprocess.run(["python", "build_mars.py", "--models=opus-mt-da-en,opus-mt-en-da", "--overwrite_mar"], check=True)

    # Start TorchServe
    print("Starting TorchServe...")
    torchserve_process = subprocess.Popen([
        "torchserve",
        "--start",
        f"--model-store={os.path.join(os.environ["LOCAL_MODEL_DIR"], os.environ["MARS_DIR"])}",
        "--models=all",
        "--ts-config=config.properties"
        "--no-config-snapshot",
        "--disable-token-auth",
    ])

    # Keep the script running to catch signals
    signal.pause()