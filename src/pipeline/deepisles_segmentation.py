import os
import subprocess


def run_deepisles(
    subject_dir: str, # Input should be the full path 
    dwi_file_name: str,
    adc_file_name: str,
    flair_file_name: str,
    fast: bool = False,
    save_team_outputs: bool = False,
    skull_strip: bool = False,
    parallelize: bool = True,
    results_mni: bool = False,
    verbose: bool = True,
    logger = None
):
    """
    Runs DeepISLES segmentation inside Docker.
    """

    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "-v", f"{subject_dir}:/app/data",
        "isleschallenge/deepisles",
        "--dwi_file_name", dwi_file_name,
        "--adc_file_name", adc_file_name,
        "--flair_file_name", flair_file_name,
    ]

    # Optional flags
    if fast:
        cmd.append("--fast")
    if save_team_outputs:
        cmd.append("--save_team_outputs")
    if skull_strip:
        cmd.append("--skull_strip")
    if parallelize:
        cmd.append("--parallelize")
    if results_mni:
        cmd.append("--results_mni")

    if verbose:
        message = f"Running DeepISLES command:\n{' '.join(cmd)}"
        logger.info(message)

    subprocess.run(cmd, check=True)

    if verbose:
        logger.info("DeepISLES segmentation completed successfully.")
