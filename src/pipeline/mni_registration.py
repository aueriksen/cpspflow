import os
import ants

def register_subject_to_mni(images_to_register, mni_template, output_dir, type_of_transform, logger = None):
    """
    Register subject scans to MNI space using a brain-extracted reference image.

    Args:
        images_to_register (dict): {name: ANTsImage} e.g.,
            {"dwi_b0": img, "dwi_b1000": img, "adc": img, "flair": img, "lesion": img}
        reference (str): brain-extracted DWI b0 in subject space
        mni_template_path (str): path to MNI template (e.g., brain-only template)
        output_dir (str): folder to save registered images

    Returns:
        dict: {name: registered ANTsImage in MNI space}
    """
    reg_path = os.path.join(output_dir, "mni_results")
    os.makedirs(reg_path, exist_ok=True)

    logger.info("Registering DWI b0 brain to MNI...")
    reg_b0 = ants.registration(
        fixed=mni_template,
        moving=images_to_register["dwi_b0"],
        type_of_transform=type_of_transform,  # or "Rigid+Affine", or "SyN" if nonlinear desired
        verbose=False,
        outprefix=os.path.join(reg_path, "dwi_b0_to_MNI_")
    )

    # Apply the same transform to all other images
    registered_images = {}
    transform_list = reg_b0['fwdtransforms']  # forward transforms from subject -> MNI

    for name, img in images_to_register.items():
        logger.info(f"Applying MNI transform to {name}...")
        warped = ants.apply_transforms(
            fixed=mni_template,
            moving=img,
            transformlist=transform_list,
            interpolator='linear' if name != "lesion" else 'nearestNeighbor'  # nearest for masks
        )
        registered_images[name] = warped

        # Save
        out_file = os.path.join(reg_path, f"{name}_MNI.nii.gz")
        ants.image_write(warped, out_file)

    logger.info("Successfully registered subject scans to MNI!")

    return registered_images
