import os
import ants
from src.pipeline.utils import apply_mask, apply_transform

def compute_within_subject_transforms(fixed, moving_dict, output_path, save=True, logger = None):
    """
    Compute within-subject registration transforms for multiple scans to a fixed reference.

    Args:
        fixed (ANTsImage or str): Reference image (e.g., DWI b1000)
        moving_dict (dict): Dictionary of {name: ANTsImage or path} for moving images
            Example: {"dwi_b0": b0_image, "adc": adc_image, "flair": flair_image}
        output_path (str): Folder to save registered images
        save (bool): Whether to write registered images to disk

    Returns:
        tuple:
            registered_images (dict): {name: registered ANTsImage}
            transforms (dict): {name: transform dictionary from ants.registration}
    """

    os.makedirs(output_path, exist_ok=True)

    registered_images = {}
    transforms = {}

    # Registration settings
    reg_kwargs = {
        "type_of_transform": "Rigid",
        "verbose": False,
    }

    for name, moving in moving_dict.items():
        outprefix = os.path.join(output_path, f"{name}_to_fixed_")
        logger.info(f"Registering {name} to fixed image...")
        try:
            reg = ants.registration(fixed=fixed, moving=moving, outprefix=outprefix, **reg_kwargs)
        except Exception as e:
            logger.error(f"Registration failed for {name}: {e}")
            raise
        # Store transformed image and transform dictionary
        registered_images[name] = reg["warpedmovout"]
        transforms[name] = reg["fwdtransforms"]

        # Save registered image if requested
        if save:
            ants.image_write(reg["warpedmovout"], os.path.join(output_path, f"{name}_registered.nii.gz"))
        
        logger.info("Within-subject registration completed for all moving images.")

    return registered_images, transforms


def apply_transforms_and_brain_masks(
    registered: dict,
    dwi_b1000,
    brain_masks: dict,
    transforms: dict,
    output_dir: str,
    logger=None
):
    """
    Applies registration transforms to BET masks, applies masks to registered images,
    and (optionally) saves intermediate results for DeepISLES.

    Returns:
        dict of masked images (ANTsImage objects)
    """

    logger.info("Applying registration transforms to brain masks...")

    # Transform BET masks into registered (b1000) space
    b0_mask_reg = apply_transform(
        fixed=registered["dwi_b0"],
        moving_mask=brain_masks["dwi_b0_brain_mask"],
        transform_list=transforms["dwi_b0"]
    )

    flair_mask_reg = apply_transform(
        fixed=registered["flair"],
        moving_mask=brain_masks["flair_brain_mask"],
        transform_list=transforms["flair"]
    )

    logger.info("Applying brain masks to registered images...")

    # Apply masks to aligned images
    masked = {
        "dwi_b1000_brain": apply_mask(dwi_b1000, b0_mask_reg),
        "dwi_b0_brain": apply_mask(registered["dwi_b0"], b0_mask_reg),
        "adc_brain": apply_mask(registered["adc"], b0_mask_reg),
        "flair_brain": apply_mask(registered["flair"], flair_mask_reg),
    }

    # Save required DeepISLES input images
    os.makedirs(output_dir, exist_ok=True)

    filename_map = {
        "dwi_b1000_brain": "dwi_b1000_brain.nii.gz",
        "dwi_b0_brain": "dwi_b0_brain.nii.gz",
        "adc_brain": "adc_brain.nii.gz",
        "flair_brain": "flair_brain.nii.gz",
    }

    for key, outname in filename_map.items():
        ants.image_write(masked[key], os.path.join(output_dir, outname))
    
    logger.info("Mask application and saving complete.")

    return masked