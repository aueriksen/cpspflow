import argparse
import os

from src.pipeline.utils import assert_gpu_available
from src.pipeline.full_pipeline import run_full_pipeline


# Default paths inside Docker image
DEFAULT_MNI = "/app/data/mni_icbm_avg_152_t1/icbm_avg_152_t1_tal_nlin_symmetric_VI_brain.nii"
DEFAULT_CPSP = "/app/data/lesion_symptom_mask/results_mask_mirrored_resampled.nii.gz"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CPSP MRI preprocessing + DeepISLES pipeline"
    )

    # New argument names
    parser.add_argument("--input_dir", required=True,
                        help="Folder containing the subject's NIfTI or DICOM files.")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory (absolute path).")

    parser.add_argument("--dwi_b0", required=True,
                        help="Filename for DWI b0 inside subject_dir.")
    parser.add_argument("--dwi_b1000", required=True,
                        help="Filename for DWI b1000 inside subject_dir.")
    parser.add_argument("--adc", required=True,
                        help="Filename for ADC inside subject_dir.")
    parser.add_argument("--flair", required=True,
                        help="Filename for FLAIR inside subject_dir.")
    
    parser.add_argument(
        "--csv_result_path",
        help="Path to CSV file where CPSP results will be saved."
    )

    # Optional, but can override defaults
    parser.add_argument(
        "--symptom_mask",
        default=os.getenv("CPSP_MASK", DEFAULT_CPSP),
        help="Path to CPSP mask file (default: bundled mask)."
    )

    parser.add_argument(
        "--mni_template",
        default=os.getenv("MNI_TEMPLATE", DEFAULT_MNI),
        help="Path to MNI template (default: bundled template)."
    )

    parser.add_argument("--save_intermediate", action="store_true",
                        help="Keep intermediate files.")
    
    parser.add_argument("--transform_type", default="Affine",
                        choices=["Rigid", "Affine", "SyN"],
                        help="Transformation type for MNI mapping.")

    parser.add_argument("--thr_analysis", type=float, default=0.51,
                        help="Lesion-symptom overlap threshold.")

    parser.add_argument("--parallelize", action="store_true",
                        help="Parallelize.")

    return parser.parse_args()


def main():

    # Ensure GPU exists BEFORE running Docker + HD-BET
    assert_gpu_available()

    args = parse_args()

    run_full_pipeline(
        subject_nifti_folder=args.input_dir,
        dwi_b0_file_name=args.dwi_b0,
        dwi_b1000_file_name=args.dwi_b1000,
        adc_file_name=args.adc,
        flair_file_name=args.flair,
        output_dir=args.output_dir,
        csv_result_path=args.csv_result_path,
        save_intermediate=args.save_intermediate,
        symptom_mask_path=args.symptom_mask,
        mni_template_path=args.mni_template,
        mni_transform_type=args.transform_type,
        thr_analysis=args.thr_analysis,
        parallelize=args.parallelize,
    )


if __name__ == "__main__":
    main()

