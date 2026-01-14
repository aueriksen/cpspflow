import os
# import subprocess
from src.pipeline.utils import load_img
import torch
# from HD_BET import paths
# paths.folder_with_parameter_files = "/app/hd-bet_params/release_2.0.0"
from HD_BET import hd_bet_prediction

# https://pmc.ncbi.nlm.nih.gov/articles/PMC6865732/

# Official code: https://github.com/MIC-DKFZ/HD-BET

# Code example:
#   https://github.com/Tabrisrei/ISLES22_Ensemble/tree/master/src/HD-BET/HD_BET
#   https://github.com/Tabrisrei/ISLES22_Ensemble/blob/master/src/utils.py#L101 - How to use the HD-BET model


def extract_brain(input_path: str, output_path: str, gpu: bool = True, save_mask: bool = True, verbose: bool = False, logger = None):

    logger.info(f"Starting HD-BET on: {input_path}")
    logger.info(f" --> Output will be saved to: {output_path}")
    logger.info(f" --> Using GPU: {gpu}")

    if gpu: 
        device= "cuda"
        disable_tta = False
    else: 
        device= "cpu"
        disable_tta = True
    
    predictor = hd_bet_prediction.get_hdbet_predictor(
        use_tta=disable_tta,
        device=torch.device(device),
        verbose=verbose
    )
    try:
        hd_bet_prediction.hdbet_predict(input_path, output_path, predictor, keep_brain_mask=save_mask, compute_brain_extracted_image=True)
    except Exception as e:
        logger.error(f"HD-BET failed for {input_path} --> {e}")
        raise

def extract_brain_dwi_flair(dwi_b0_img_path, flair_img_path, output_folder, logger = None):
    logger.info("Starting brain extraction for DWI b0 and FLAIR...")

    os.makedirs(output_folder, exist_ok=True)
    
    dwi_output_path = os.path.join(output_folder, "dwi_b0_brain.nii.gz")
    flair_output_path = os.path.join(output_folder, "flair_brain.nii.gz")

    extract_brain(input_path=dwi_b0_img_path, output_path=dwi_output_path, gpu= True, save_mask=True, verbose=False, logger = logger)
    extract_brain(input_path=flair_img_path, output_path=flair_output_path, gpu= True, save_mask=True, verbose=False, logger = logger)
    
    logger.info("Loading brain-extracted images and masks...")

    return {
        "dwi_b0_brain": load_img(dwi_output_path),
        "dwi_b0_brain_mask": load_img(os.path.join(output_folder,"dwi_b0_brain_bet.nii.gz")),
        "flair_brain": load_img(flair_output_path),
        "flair_brain_mask": load_img(os.path.join(output_folder,"flair_brain_bet.nii.gz")),
    }

