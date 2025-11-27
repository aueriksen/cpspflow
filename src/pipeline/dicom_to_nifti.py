import os
from pathlib import Path
import numpy as np
import nibabel as nib
import pydicom

def dicom_to_nifti(input_path: str, output_dir: str = None, logger = None) -> str:
    """
    Converts a DICOM file or folder to NIfTI using pydicom + nibabel.

    Args:
        input_path: Path to DICOM file or folder.
        output_dir: Directory to save NIfTI. Defaults to same folder as input.

    Returns:
        Path to the saved NIfTI file.
    """

    input_path = Path(input_path)
    if output_dir is None:
        output_dir = input_path.parent
    os.makedirs(output_dir, exist_ok=True)


    # Already NIfTI
    if input_path.suffix in [".nii", ".gz"]:
        return str(input_path.resolve())
    
    logger.info("Input is a directory --> scanning for DICOM files (*.dcm)")

    # Collect DICOM files
    dicom_files = []
    if input_path.is_dir():
        for f in sorted(input_path.glob("*.dcm")):
            try:
                dicom_files.append(pydicom.dcmread(str(f)))
            except Exception as e:
                logger.info(f"Warning: Could not read DICOM file {f}: {e}")
    else:
        dicom_files = [pydicom.dcmread(str(input_path))]

    # Stack slices into 3D array
    try:
        dicom_files.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except Exception:
        logger.info("Slice sorting using ImagePositionPatient failed --> keeping original order")


    pixel_arrays = [f.pixel_array for f in dicom_files]
    volume = np.stack(pixel_arrays, axis=-1)

    # Get affine from DICOM metadata
    first = dicom_files[0]
    try:
        spacing = [float(first.PixelSpacing[0]), float(first.PixelSpacing[1]), float(first.SliceThickness)]
        affine = np.diag(spacing + [1])
    except Exception:
        logger.info(f"Failed extracting spacing from DICOM metadata: {e}")
        affine = np.eye(4)
        logger.info(f"Fallback: affine = identity matrix")

    nifti_img = nib.Nifti1Image(volume, affine)

    # Save NIfTI
    nifti_path = os.path.join(output_dir, input_path.stem + ".nii.gz")
    nib.save(nifti_img, nifti_path)

    return nifti_path