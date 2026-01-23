import os

from src.pipeline.dicom_to_nifti import dicom_to_nifti
from src.pipeline.utils import load_img, load_and_check_images, housekeeping, create_logger, runtime_checks, resolve_host_path, save_results_to_csv
from src.pipeline.subject_registration import compute_within_subject_transforms, apply_transforms_and_brain_masks
from src.pipeline.brain_extraction import extract_brain_dwi_flair
from src.pipeline.deepisles_segmentation import run_deepisles
from src.pipeline.mni_registration import register_subject_to_mni
from src.pipeline.analysis import run_overlap_analysis

def run_full_pipeline(
    subject_nifti_folder: str,
    dwi_b0_file_name: str,
    dwi_b1000_file_name: str,
    adc_file_name: str,
    flair_file_name: str,
    output_dir: str, # Full path
    csv_result_path: str = None, # If none, it creates it in the output folder
    save_intermediate: bool = False,
    symptom_mask_path: str = None,
    mni_template_path: str = None,
    mni_transform_type: str = "Affine", # Either Rigid, Affine, or SyN (SyN is non-linear)
    thr_analysis: float = 0.01,
    parallelize = True,
):
    """
    Full pipeline for preprocessing, brain extraction, within-subject registration,
    mask application, DeepISLES segmentation, MNI registration, and overlap analysis.
    """
    os.makedirs(output_dir, exist_ok=True)

    logger = create_logger(output_dir)

    runtime_checks(logger)

    logger.info("====== Starting full pipeline ======")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"save_intermediate = {save_intermediate}")
    resolve_host_path(output_dir)

    # Convert inputs to NIfTI if needed, optionally anonymize
    dwi_b0_path = dicom_to_nifti(os.path.join(subject_nifti_folder, dwi_b0_file_name), output_dir=output_dir, logger = logger)
    dwi_b1000_path = dicom_to_nifti(os.path.join(subject_nifti_folder, dwi_b1000_file_name), output_dir=output_dir, logger = logger)
    adc_path = dicom_to_nifti(os.path.join(subject_nifti_folder, adc_file_name), output_dir=output_dir, logger = logger)
    flair_path = dicom_to_nifti(os.path.join(subject_nifti_folder, flair_file_name), output_dir=output_dir, logger = logger)

    # Collect all paths in a dict for loading
    input_paths = {
        "dwi_b0": dwi_b0_path,
        "dwi_b1000": dwi_b1000_path,
        "adc": adc_path,
        "flair": flair_path,
        "symptom_mask": symptom_mask_path,
        "mni_template": mni_template_path
    }

    logger.info("Loading input images...")
    images = load_and_check_images(input_paths, logger = logger)

    # Brain extraction (HD-BET) on raw DWI b0 and FLAIR
    logger.info("=== Running HD-BET on DWI b0 + FLAIR ===")
    bet_output_dir = os.path.join(output_dir, "hd_bet_results")
    brain_masks_dict = extract_brain_dwi_flair(input_paths["dwi_b0"], input_paths["flair"], bet_output_dir, logger = logger)

    # Within-subject registration (raw scans with skull)
    logger.info("=== Within-subject registration ===")
    moving_dict = {
        "dwi_b0": images["dwi_b0"],
        "adc": images["adc"],
        "flair": images["flair"]
    }
    reg_folder = os.path.join(output_dir, "within_subject_reg")
    registered_images, transforms = compute_within_subject_transforms(
        fixed=images["dwi_b1000"],
        moving_dict=moving_dict,
        output_path=reg_folder,
        save=save_intermediate,
        logger = logger
    )

    subject_space_results = os.path.join(output_dir, "subject_space_results")
    logger.info("=== Applying transforms + brain masks ===")
    preprocessed = apply_transforms_and_brain_masks(registered=registered_images, dwi_b1000=images["dwi_b1000"], brain_masks=brain_masks_dict, transforms=transforms, output_dir=subject_space_results, logger = logger)

    # DeepISLES segmentation
    logger.info("=== Running DeepISLES segmentation ===")
    run_deepisles(
        subject_dir = os.path.join(os.environ["HOST_OUTPUT_DIR"], "subject_space_results"),
        dwi_file_name = "dwi_b1000_brain.nii.gz", 
        adc_file_name = "adc_brain.nii.gz",
        flair_file_name = "flair_brain.nii.gz",
        fast = False,
        save_team_outputs = False,
        skull_strip = False,
        parallelize = parallelize,
        results_mni = False,
        verbose = True,
        logger = logger
    )

    lesion_mask = load_img(os.path.join(subject_space_results, "results", "lesion_msk.nii.gz"))

    # Register everything to MNI space
    logger.info("=== Registering images + lesion to MNI ===")
    mni_outputs = register_subject_to_mni(
        images_to_register={
            "dwi_b0": preprocessed["dwi_b0_brain"],
            "dwi_b1000": preprocessed["dwi_b1000_brain"],
            "adc": preprocessed["adc_brain"],
            "flair": preprocessed["flair_brain"],
            "lesion": lesion_mask
        },
        mni_template=images["mni_template"],
        output_dir=output_dir, 
        type_of_transform=mni_transform_type,
        logger = logger
    )

    # Overlap analysis with symptom mask
    logger.info("=== Running lesion-symptom overlap analysis ===")
    analysis_results = run_overlap_analysis(
        mni_lesion=mni_outputs["lesion"],
        cpsp_mask=images["symptom_mask"],
        overlap_threshold=thr_analysis,
        logger = logger
    )

    logger.info(f"Overlap results: {analysis_results}")

    # Save it as a csv_file
    if csv_result_path is None:
        csv_result_path = os.path.join(output_dir, "cpsp_results.csv")
    analysis_results["subject_id"] = os.path.basename(subject_nifti_folder)
    save_results_to_csv(analysis_results, csv_result_path)

    logger.info("Housekeeping")
    housekeeping(output_dir=output_dir, save_intermediate=save_intermediate, logger = logger)

    logger.info("====== Pipeline completed successfully ======")

    return {
        "registered_images": registered_images,
        "masked_images": {
            "dwi_b0": preprocessed["dwi_b0_brain"],
            "adc": preprocessed["adc_brain"],
            "flair": preprocessed["flair_brain"]
        },
        "lesion_mask": lesion_mask,
        "mni_outputs": mni_outputs,
        "analysis_results": analysis_results
    }

