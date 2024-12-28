import argparse
from transformers import MarianMTModel, MarianTokenizer
import os
import subprocess

from dotenv import load_dotenv

load_dotenv("model.env")

# Get the comma separated models from the command argument
parser = argparse.ArgumentParser()
parser.add_argument("--models", required=True, type=str)
parser.add_argument("--overwrite_mar", default=False, action=argparse.BooleanOptionalAction)
modelNames = parser.parse_args().models.split(",")
overwrite_mar = parser.parse_args().overwrite_mar

# Download the models from huggingface, if they aren't already stored locally
for modelName in modelNames:
    # ------------------------------ Downloading ------------------------------
    model_local_path = os.path.join(os.environ["LOCAL_MODEL_DIR"], os.environ["RAW_DIR"], modelName)

    if not os.path.isdir(model_local_path):
        print(f"Model folder for '{modelName}' wasn't found. Beginning download...")

        model_name = f"Helsinki-NLP/{modelName}".lower()

        # Download the model and tokenizer
        model = MarianMTModel.from_pretrained(model_name)
        tokenizer = MarianTokenizer.from_pretrained(model_name)

        # Move parameters from the models config to the generation config to avoid a userwarning
        model.generation_config.max_length = model.config.max_length
        model.generation_config.num_beams = model.config.num_beams
        model.generation_config.bad_words_ids = model.config.bad_words_ids
        del model.config.max_length
        del model.config.num_beams
        del model.config.bad_words_ids

        # Save the model and tokenizer locally
        model.save_pretrained(model_local_path)
        tokenizer.save_pretrained(model_local_path)

    else:
        print(f"Model '{modelName}' already downloaded. Skipping download.")

    # ---------------------------- Create .mar file ----------------------------
    mar_file_path = os.path.join(os.environ["LOCAL_MODEL_DIR"], os.environ["MARS_DIR"], f"{modelName}.mar")

    if os.path.isfile(mar_file_path) and not overwrite_mar:
        print(f"Mar file for '{modelName}' is already built. Skipping build.\n")

    else:
        print(f"Building '{modelName}.mar'...")

        # Ensure a directory exists for the .mar files.
        mar_dir_path = os.path.join(os.environ["LOCAL_MODEL_DIR"], "mars")

        if not os.path.isdir(mar_dir_path):
            os.mkdir(mar_dir_path)

        # Add all the required files, exluding the model, in order to build the model archive
        extra_file_paths = []
        
        for file in os.listdir(model_local_path):
            if file != "model.safetensors":
                extra_file_paths.append(os.path.join(model_local_path, file))

        torch_model_archiver_command = [
            "torch-model-archiver",
            "--force",
            f"--model-name={modelName}",
            "--version=1.0",
            f"--serialized-file={os.path.join(model_local_path, "model.safetensors")}",
            f"--extra-files={",".join(extra_file_paths)}",
            f"--export-path={mar_dir_path}",
            f"--handler={os.path.join("handlers", "handler.py")}"
        ]

        result = subprocess.run(torch_model_archiver_command, stdout=subprocess.PIPE, check=True)
        
        if result.returncode == 0:
            print(f"Succesfully built '{modelName}.mar'.\n")
        
        else:
            print(f"Error encountered while building '{modelName}.mar'. Check output for details.\n")