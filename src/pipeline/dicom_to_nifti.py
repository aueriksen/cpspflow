import os
import shutil
import tempfile
from pathlib import Path
import pydicom
import dicom2nifti
import dicom2nifti.settings as settings


def dicom_to_nifti(input_path: str, output_dir: str = None, logger=None) -> str:
    """
    Converts a DICOM file or folder to a correctly oriented NIfTI file
    using dicom2nifti for robust handling of direction cosines.

    Args:
        input_path: Path to a DICOM file or folder.
        output_dir: Directory to save the resulting NIfTI file.
        modality:   Name used to generate output filename.
        logger:     Optional logger.

    Returns:
        Path to the saved NIfTI file.
    """

    settings.disable_validate_slice_increment()
    settings.disable_validate_slicecount()
    settings.disable_validate_orientation()
    # settings.disable_validate_qform_code()
    settings.disable_validate_orthogonal()

    input_path = Path(input_path)

    if output_dir is None:
        output_dir = input_path.parent

    os.makedirs(output_dir, exist_ok=True)

    # If input is already NIfTI → just return it
    if input_path.suffix.lower() in [".nii", ".gz", ".nii.gz"]:
        if logger:
            logger.info(f"{input_path} is already NIfTI → skipping conversion.")
        return str(input_path.resolve())

    if logger:
        logger.info(f"Collecting DICOMs from {input_path}")

    # Collect DICOM files
    dicom_files = []

    if input_path.is_dir():
        for f in sorted(input_path.glob("*.dcm")):
            try:
                ds = pydicom.dcmread(str(f))
                dicom_files.append(ds)
            except Exception as e:
                if logger:
                    logger.warning(f"Warning: Could not read DICOM file {f}: {e}")
    else:
        dicom_files = [pydicom.dcmread(str(input_path))]

    if len(dicom_files) == 0:
        raise RuntimeError(f"No readable DICOM files found in {input_path}")

    if logger:
        logger.info(f"Found {len(dicom_files)} DICOM files")

    # -------------------------
    # Create temp folder of DICOMs
    # -------------------------
    temp_dir = tempfile.mkdtemp()
    if logger:
        logger.info(f"Created temp folder: {temp_dir}")

    try:
        for idx, dicom_file in enumerate(dicom_files):
            src = (
                dicom_file
                if isinstance(dicom_file, str)
                else dicom_file.filename
            )
            shutil.copy(src, os.path.join(temp_dir, f"dicom_{idx}.dcm"))

        # Output file
        output_file = os.path.join(output_dir, f"{input_path.stem}.nii.gz")

        if logger:
            logger.info(f"Running dicom2nifti on {len(dicom_files)} slices → {output_file}")

        dicom2nifti.convert_dicom.dicom_series_to_nifti(
            temp_dir,
            output_file,
            reorient_nifti=False   # preserve original orientation for ANTs
        )

        if logger:
            logger.info(f"Saved NIfTI to: {output_file}")

    finally:
        # Cleanup temp directory
        if logger:
            logger.info(f"Removing temp folder: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)

    return output_file