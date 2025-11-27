import numpy as np

def run_overlap_analysis(mni_lesion, cpsp_mask, overlap_threshold=0.51, logger = None):
    """
    Check if a lesion overlaps with a (possibly mirrored) pain-related mask region.

    Args:
        mni_lesion (ANTsImage): Lesion mask in MNI space
        cpsp_mask (ANTsImage): Pain region mask (mirrored: left=1, right=2)
        overlap_threshold (float): Fraction of lesion voxels overlapping to consider True

    Returns:
        dict: {
            "left_overlap": bool,
            "right_overlap": bool,
            "overlap_fraction_left": float,
            "overlap_fraction_right": float
        }
    """
    lesion_data = mni_lesion.numpy() > 0
    pain_data = cpsp_mask.numpy()

    lesion_voxels = np.sum(lesion_data) # mni spacing is 1mm3 -> Each voxel equals 1 ml
    if lesion_voxels == 0:
        result = {
            "lesion_volume_mm3": 0,
            "left_overlap": False,
            "overlap_fraction_left": 0.0,
            "right_overlap": False,
            "overlap_fraction_right": 0.0
        }
        logger.info("Lesion mask is empty. No overlap detected.")
        return result

    # Left overlap: check voxels where pain mask == 1
    overlap_left_voxels = np.sum(np.logical_and(lesion_data, pain_data == 1))
    overlap_fraction_left = overlap_left_voxels / lesion_voxels

    # Right overlap: check voxels where pain mask == 2
    overlap_right_voxels = np.sum(np.logical_and(lesion_data, pain_data == 2))
    overlap_fraction_right = overlap_right_voxels / lesion_voxels

    result = {
        "lesion_volume_mm3": int(lesion_voxels),
        "left_overlap": overlap_fraction_left > overlap_threshold,
        "overlap_fraction_left": overlap_fraction_left,
        "right_overlap": overlap_fraction_right > overlap_threshold,
        "overlap_fraction_right": overlap_fraction_right
    }
    logger.info(f"Lesion volume: {result['lesion_volume_mm3']} mm3")
    logger.info(f"Left overlap: {result['left_overlap']} ({overlap_fraction_left:.2f})")
    logger.info(f"Right overlap: {result['right_overlap']} ({overlap_fraction_right:.2f})")

    return result
