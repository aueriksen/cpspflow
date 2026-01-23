import os
import shutil
from typing import List, Dict
import logging
import subprocess
import csv
import numpy as np 
import ants
import torch


def load_and_check_images(paths_dict, logger = None):
    """
    Verify that all paths exist, are .nii or .nii.gz, and load as ANTs images.

    Args:
        paths_dict (dict): {name: filepath} e.g.,
            {"dwi_b0": "data/subj01/b0.nii.gz", ...}

    Returns:
        dict: {name: ANTsImage}
    """
    loaded_images = {}

    for name, path in paths_dict.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"File for {name} does not exist: {path}")
        if not path.endswith((".nii", ".nii.gz")):
            raise ValueError(f"File for {name} is not a NIfTI: {path}")

        logger.info(f"Loading {name}: {path}")
        loaded_images[name] = load_img(path)

    logger.info("All images loaded successfully.")
    return loaded_images


def load_img(raw_img_path, orient="RAS"):
    ants_image = ants.image_read(raw_img_path, reorient=orient)
    return ants_image


def normalize_image(image):
    normalized_img = ants.iMath(image, "Normalize")
    return normalized_img


def delete_temp_files(files: List[str]):
    """Delete temporary files if the flag is set."""
    for file in files:
        if os.path.exists(file):
            os.remove(file)


def mirror_pain_mask(pain_mask_path, output_path):
    """
    Mirror a left-hemisphere pain mask to the right hemisphere
    and save a single NIfTI with both masks encoded:
        left = 1, right = 2

    Args:
        pain_mask_path (str): Path to input left-hemisphere mask (binary NIfTI)
        output_path (str): Path to save combined mirrored mask

    Returns:
        None
    """
    # Load the mask
    pain_mask = load_img(pain_mask_path)
    pain_data = pain_mask.numpy() > 0

    # Mirror mask across x-axis (left-right)
    pain_data_flipped = np.flip(pain_data, axis=0)

    # Initialize combined mask
    combined_mask = np.zeros_like(pain_data, dtype=np.uint8)

    # Assign left hemisphere voxels = 1
    combined_mask[pain_data] = 1

    # Assign right hemisphere voxels = 2
    combined_mask[pain_data_flipped] = 2

    # Convert back to ANTs image
    combined_img = ants.from_numpy(
        combined_mask,
        origin=pain_mask.origin,
        spacing=pain_mask.spacing,
        direction=pain_mask.direction
    )

    # Ensure output folder exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ants.image_write(combined_img, output_path)
    print(f"Saved mirrored pain mask to {output_path}")


def apply_transform(fixed, moving_mask, transform_list):
    """Apply transforms to a brain mask."""
    return ants.apply_transforms(
        fixed=fixed,
        moving=moving_mask,
        transformlist=transform_list
    )

def apply_mask(image, mask):
    """Apply mask to an image."""
    return image * mask



def housekeeping(output_dir: str, save_intermediate: bool, logger=None):
    """
    Handles cleanup of intermediate folders and relocation of final DeepISLES output.

    Actions:
    1. Move lesion_msk.nii.gz one folder up.
    2. Delete results/ folder.
    3. If save_intermediate=False → delete intermediate folders.
    """

    # Move lesion_msk.nii.gz up
    results_dir = os.path.join(output_dir, "subject_space_results", "results")
    lesion_src = os.path.join(results_dir, "lesion_msk.nii.gz")
    lesion_dst = os.path.join(output_dir, "subject_space_results", "lesion_msk.nii.gz")

    if os.path.exists(lesion_src):
        logger.info("Moving lesion_msk.nii.gz to output directory...")
        shutil.move(lesion_src, lesion_dst)
    else:
        logger.info("Warning: lesion_msk.nii.gz not found. DeepISLES may have failed.")

    # Delete results/ folder
    if os.path.exists(results_dir):
        logger.info("Deleting empty folder...")
        shutil.rmtree(results_dir, ignore_errors=True)

    # Delete intermediate folders
    if not save_intermediate:
        logger.info("Cleaning up intermediate folders...")
        intermediate_dirs = [
            "hd_bet_results",
            "within_subject_reg"
        ]

        for folder in intermediate_dirs:
            folder_path = os.path.join(output_dir, folder)
            if os.path.exists(folder_path):
                logger.info(f"Deleting: {folder_path}")
                shutil.rmtree(folder_path, ignore_errors=True)
    else:
        logger.info("Keeping intermediate files as requested.")

    logger.info("Housekeeping complete.\n")

def assert_gpu_available():
    if not torch.cuda.is_available():
        raise AssertionError(
            "Trying to run cpspflow, but no GPU was detected. DeepISLES requires an NVIDIA GPU."
        )


def create_logger(output_dir, name="pipeline_logger"):
    """
    Creates a logger that prints to console and writes to output_dir/pipeline.log
    """
    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # prevent double logging

    # If handlers already exist, do nothing (avoid duplicates)
    if len(logger.handlers) > 0:
        return logger

    log_path = os.path.join(output_dir, "pipeline.log")

    # Formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )

    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

def resolve_host_path(container_path):
    with open("/proc/self/mountinfo", "r") as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) >= 5:
                mount_point = fields[4]
                if mount_point == container_path:
                    # host path is the last field
                    return fields[-1]
    return None


def runtime_checks(logger):
    # Check GPU visibility
    try:
        out = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
        logger.info("GPU detected via nvidia-smi.")
    except Exception:
        logger.error("ERROR: No GPU detected inside container. Exiting.")
        raise RuntimeError("GPU not available in container.")

    # Check docker socket
    if not os.path.exists("/var/run/docker.sock"):
        logger.error("ERROR: Docker socket not mounted. DeepISLES cannot run.")
        raise RuntimeError("Host Docker socket not mounted.")

    # Check docker CLI exists
    if shutil.which("docker") is None:
        logger.error("ERROR: docker client not installed inside container.")
        raise RuntimeError("docker CLI missing in container.")

    logger.info("Runtime checks passed successfully.")


def save_results_to_csv(
    result: Dict,
    csv_path: str
) -> None:
    """
    Save a single result dictionary to a CSV file.
    If the CSV already exists, the result is appended.
    If not, the header is written automatically.

    Args:
        result (dict): Dictionary containing result values.
        csv_path (str): Full path to output CSV file.
    """

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    write_header = not os.path.exists(csv_path)

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())

        if write_header:
            writer.writeheader()

        writer.writerow(result)



if __name__ == "__main__":
    pass
