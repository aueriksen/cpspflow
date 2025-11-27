# CPSP Lesion-Symptom Mapping Pipeline

**Requires GPU acceleration.**

---

## Input Requirements

The pipeline accepts either:

- **NIfTI files** (`.nii` or `.nii.gz`)  
- **DICOM folders** (the pipeline will convert them to NIfTI automatically)

**Recommended folder structure for a subject:**

***NIfTI files (`.nii` or `.nii.gz`):***  
subject_folder/  
├── dwi_b0.nii.gz  
├── dwi_b1000.nii.gz  
├── adc.nii.gz  
└── flair.nii.gz  

***DICOM folders:***  
subject_folder/  
├── dwi_b0/ # DICOM folder  
├── dwi_b1000/ # DICOM folder  
├── adc/ # DICOM folder  
└── flair/ # DICOM folder  


---

## Installation

1. **Install [Docker](https://docs.docker.com/engine/install/)**  
2. **Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)**  
3. **Open a command line or terminal** (e.g., PowerShell on Windows, Terminal on Linux/Mac) and verify GPU access inside Docker:  
    ```bash
    docker run --rm --gpus all ubuntu nvidia-smi
    ```
    This should display your NVidia GPUs.  
4. **Pull the DeepISLES Docker image**  
    ```bash
    docker pull isleschallenge/deepisles
    ```
5. ***Build the CPSP pipeline image:***  
    ```bash
    docker build -t aeriksen/cpspflow:latest .
    ```

---

## Quickstart Example
Assuming your subject folder is data/subj01/ and contains NIfTI files (rename the filenames according to your files):  

docker run --rm --gpus all \
  -v "INSERT_INPUT_FOLDER:/app/input" \
  -v "INSERT_OUTPUT_FOLDER:/app/output" \
  -e HOST_OUTPUT_DIR="INSERT_OUTPUT_FOLDER" \
  -v /var/run/docker.sock:/var/run/docker.sock \
  aeriksen/cpspflow:latest \
    --input_dir /app/input \
    --output_dir /app/output \
    --dwi_b0 DWI_b0.nii.gz \
    --dwi_b1000 DWI_b1000.nii.gz \
    --adc ADC.nii.gz \
    --flair FLAIR.nii.gz \
    --save_intermediate \
    --transform_type Affine \
    --parallelize

Notes:  
* Replace INSERT_INPUT_FOLDER and INSERT_OUTPUT_FOLDER with the full paths on your system.  
* /input is mapped to your subject folder  
* /output will contain all results, logs, and final lesion masks  
* -v /var/run/docker.sock:/var/run/docker.sock is required so that CPSPFlow can call DeepISLES in a separate Docker container.  
* For DICOM, use the folder name instead of .nii.gz  

---

## Pipeline Steps

Inside the container, the following steps are executed automatically:

1. DICOM → NIfTI conversion (if needed)  
2. Brain extraction using HD-BET (GPU-accelerated)  
3. Within-subject registration (ANTsPy)  
4. Mask application & DeepISLES input preparation  
5. Lesion segmentation using DeepISLES models (GPU)  
6. Registration to MNI space using ANTsPy  
7. CPSP lesion–symptom overlap analysis  
8. Final housekeeping (optional cleanup)  

The pipeline runs inside a Docker container (CPSPFlow), but the DeepISLES segmentation step is executed via a separate Docker container on the host. Make sure the host system has Docker and GPU access configured.

---

## HD-BET Weights for Local Runs

For brain extraction using HD-BET on your **local machine** (outside Docker), you need to provide the model weights manually.

1. **Download the HD-BET weights**:  
   [Release v1.5.0](https://zenodo.org/records/14445620/files/release_v1.5.0.zip?download=1)

2. **Extract the weights** and place them in your CPSP pipeline folder, e.g.:  
   ```text
   ~/hd-bet_params/release_2.0.0/
   ```

3. **Update the Python script** that calls HD-BET (brain_extraction.py) to use these weights:
   ```text
   from HD_BET import paths
   paths.folder_with_parameter_files = "/path/to/hd-bet_weights/release_1.5.0"
   ```

**Notes**:  
* The folder should contain all files from the zip, unzipped.  
* This step is only required for running HD-BET locally; when running via Docker, the pipeline can use the pre-installed weights inside the container.

---

## References

* ANTsPy / ANTs: [Documentation](https://antspy.readthedocs.io/en/stable/)  
* DeepISLES (Docker-based segmentation): [DeepISLES repo](https://github.com/ezequieldlrosa/DeepIsles)  
* HD-BET: [Isensee et al., HD-BET: Robust Brain Extraction using Deep Learning](https://github.com/MIC-DKFZ/HD-BET/tree/master)  
* CPSP Pain Mask: [Dissecting neuropathic from poststroke pain: the white matter within](http://dx.doi.org/10.1097/j.pain.0000000000002427),  
* MNI Template: [Template reference](https://dx.doi.org/10.1007/11866763_8), [Download](https://nist.mni.mcgill.ca/mni-icbm152-non-linear-6th-generation-symmetric-average-brain-stereotaxic-registration-model/)  

---

## Notes  
* All processing requires GPU-enabled Docker.  
* Intermediate files can be saved or cleaned up automatically.  
* Lesion volumes are reported in mm³, and overlap is computed per hemisphere.  
