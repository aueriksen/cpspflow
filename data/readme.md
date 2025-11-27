# Data Sources for CPSP Pipeline

This folder contains reference masks and templates used in the lesion-symptom mapping pipeline.

---

## CPSP Mask

- **Description:** Pain-related region mask (mirrored: left=1, right=2).  
- **Citation / DOI:** [Dissecting neuropathic from poststroke pain: the white matter within](http://dx.doi.org/10.1097/j.pain.0000000000002427)  
- **Intended use:** Overlap analysis with subject lesion masks in MNI space.

---

## MNI Template

- **Name:** ICBM 152 2009c nonlinear symmetric average brain template  
- **Resolution:** 1 mm isotropic  
- **Citation / DOI:** [Template reference](https://dx.doi.org/10.1007/11866763_8)  
- **Download link:** [Official MNI template](https://nist.mni.mcgill.ca/mni-icbm152-non-linear-6th-generation-symmetric-average-brain-stereotaxic-registration-model/)  
- **Intended use:** Registration of subject scans to standard MNI space.

---

### Notes

- Ensure all files are in NIfTI format (`.nii` or `.nii.gz`).  
- All subject scans will be registered to the MNI template before lesion overlap analysis.