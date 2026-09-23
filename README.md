# Eardrum Detection and Segmentation

## Table of Contents

- #introduction
  - #eardrum-anatomy
  - #pipeline-overview
- [pre-processing
  - #input-requirements
  - #overview
  - [Head Localization and Cropping
  - [Reference Frame Construction]  - #landmark-detection
  - [Frankfort Plane Alignment](#- #ear-region-cropping
  - [Intensity Normalization](#- #output-files
- #segmentation

# Introduction

This repository contains a complete pipeline for automatic eardrum detection and segmentation from CT scans.

The workflow is divided into two main components:

1. A preprocessing pipeline that automatically localizes the ears, standardizes anatomical orientation, and extracts high-resolution ear-centered volumes.
2. An automatic segmentation model that identifies the tympanic membrane within the extracted ear volumes.

The preprocessing stage is designed to reduce anatomical variability across subjects and generate standardized inputs for downstream segmentation tasks.

The overall objective is to provide a fully automated workflow that starts from a raw head CT scan and produces accurate eardrum segmentations.

---

## Eardrum Anatomy

The tympanic membrane, commonly known as the eardrum, is a thin semitransparent membrane that separates the external auditory canal from the middle ear cavity.

Its primary functions are:

- Converting sound pressure waves into mechanical vibrations.
- Transmitting acoustic energy to the ossicular chain.
- Acting as a protective barrier between the external and middle ear.

The tympanic membrane is located at the medial end of the external auditory canal and forms one of the key anatomical structures of the hearing system.

Due to its small size and thin geometry, accurate identification of the eardrum in CT images requires both precise localization and high-resolution image processing.

---

## Pipeline Overview

The complete workflow consists of the following stages:

1. Intensity clipping.
2. Head localization.
3. Native-resolution head ROI extraction.
4. Reference-frame construction.
5. Anatomical landmark detection.
6. Frankfort plane alignment.
7. High-resolution ear extraction.
8. Intensity normalization.
9. Automatic eardrum segmentation.

The preprocessing pipeline combines the functionality of the original P1-P4 workflow into a single script while preserving compatibility with the existing landmark-detection model.

A temporary low-resolution volume is generated exclusively for landmark detection because the trained model expects a fixed 256 × 256 × 256 input grid. This representation is used only for landmark prediction and is discarded afterwards.

All final ear crops are extracted at a configurable isotropic resolution (0.2 mm by default) directly from a higher-resolution reference frame. Unlike the original preprocessing workflow, the final outputs are not generated from the low-resolution landmark-detection volume.

The pipeline additionally preserves compatibility with the coordinate conventions used by the original P1-P4 workflow, allowing integration with previously processed datasets and transformation logs.

---

# Pre-processing

## Input Requirements

The pipeline expects CT scans in NIfTI format:

```text
<patient_id>__CT.nii
```

or

```text
<patient_id>__CT.nii.gz
```

### Requirements

- Head CT scan containing both ears.
- NIfTI format (`.nii` or `.nii.gz`).
- Voxel spacing smaller than:

```text
(1.5, 1.5, 5.5) mm
```

Scans with coarser voxel sizes are automatically excluded.

### Required Model

The pipeline requires a trained landmark-detection model:

```text
best_model_*.pth
```

which predicts six anatomical landmarks used for head alignment and ear localization.

### Software Dependencies

Main dependencies include:

- PyTorch
- SimpleITK
- NiBabel
- NumPy
- SciPy
- PyVista
- TotalSegmentator

---

## Overview

This preprocessing pipeline is a high-resolution adaptation of the preprocessing workflow originally developed for tissue segmentation.

The same anatomical normalization principles are preserved:

- Head localization.
- Standardized anatomical orientation.
- Landmark-based alignment.
- Consistent coordinate systems.
- Standardized cropping.

The preprocessing workflow consists of:

1. Intensity clipping.
2. Head localization using TotalSegmentator.
3. Native-resolution head ROI extraction.
4. Construction of a standardized local reference frame.
5. Landmark detection using a 3D U-Net.
6. Frankfort-plane alignment.
7. High-resolution ear extraction.
8. Intensity normalization.

The key difference from the original P1-P4 workflow is that landmark detection is performed on a temporary low-resolution representation, while all final ear crops are generated in a high-resolution reference frame.

Final outputs are therefore produced without the coarse intermediate downsampling used in the original preprocessing workflow.

---

## Head Localization and Cropping

The first stage identifies the head region and removes unnecessary anatomy.

### Intensity Clipping

CT intensities are clipped to:

```text
[-1000, 2007] HU
```

to reduce the influence of extreme intensity values while preserving relevant anatomical structures.

### Ear Localization

TotalSegmentator is executed using the `head_glands_cavities` task.

The centroids of the left and right auditory canals are extracted from the segmentation output. The midpoint between both canals is calculated and used as the reference point for head localization.

### Head ROI Extraction

A fixed-size head ROI is cropped around the ear midpoint while preserving the native scan resolution.

Default ROI dimensions:

```text
200 × 270 × 250 mm
```

The superior direction includes an additional 50 mm margin to ensure sufficient anatomical coverage for subsequent landmark detection and alignment.

This step reduces computational cost while retaining all structures required for the remaining processing stages.

---

## Reference Frame Construction

After head localization, a standardized local reference frame is created.

The cropped head ROI is first resampled to a common local reference frame with:

```text
Spacing: 0.5 mm isotropic
Origin: (0, 0, 0)
Orientation signs: (-X, -Y, +Z)
```

The volume is automatically flipped when necessary so that all scans share the same orientation convention prior to landmark detection.

### Landmark Detection Volume

A temporary low-resolution volume is generated exclusively for landmark detection.

Characteristics:

```text
Spacing: 0.5 mm isotropic
Padding: 540 × 540 × 540 voxels
Final size: 256 × 256 × 256 voxels
```

This volume reproduces the dimensions and spatial resolution used during landmark-model training and is discarded after landmark prediction.

### High-Resolution Working Space

A second reference space is created specifically for final ear extraction.

Characteristics:

```text
Default spacing: 0.2 mm isotropic
```

Unlike the original preprocessing workflow, the final ear crops are not generated from the low-resolution landmark-detection volume.

When `fast=False`, the entire high-resolution head volume is explicitly generated and rotated prior to ear extraction.

When `fast=True`, each ear field of view is resampled directly from the local reference frame without creating the complete high-resolution head volume, reducing memory usage while preserving the final output space.

---

## Landmark Detection

Anatomical landmark detection is performed using a 3D U-Net model.

The network predicts six anatomical landmarks:

```text
Landmark 8
Landmark 9
Landmark 10
Landmark 11
Landmark 12
Landmark 13
```

Landmarks are predicted as heatmaps and subsequently converted into physical world coordinates within the standardized reference frame.

The coordinate conversion follows the same convention used by the original P2 preprocessing workflow to ensure compatibility with previously trained models and downstream processing scripts.

The most relevant landmarks for this pipeline are:

```text
Landmark 10 → Right eardrum
Landmark 11 → Left eardrum
```

These landmarks are used for both Frankfort-plane alignment and extraction of the final ear-centered volumes.

---

## Frankfort Plane Alignment

To reduce orientation variability across subjects, the head is aligned using the Frankfort plane.

### Standard Mode

The default alignment strategy uses landmarks 10-13 to estimate the Frankfort plane.

The resulting rotation aligns:

- The Frankfort plane with the horizontal plane.
- The left-right anatomical axis with the global x-direction.

### No-Eyes Mode

When eye-based landmarks are unavailable or unreliable, an alternative alignment strategy can be used:

```bash
--no_eyes True
```

In this mode, TotalSegmentator is used to segment:

- Masseter muscles
- Lateral pterygoid muscles

These structures are combined with the eardrum landmarks to estimate a robust anatomical reference plane using SVD-based plane fitting.

### Skip Alignment Mode

For experiments where rotational normalization is not desired:

```bash
--skip_alignment True
```

can be used.

Landmark detection remains active, but no Frankfort-plane alignment is applied.

### Quality Control

Several quality-control checks are performed automatically:

- LM10-LM11 distance validation.
- Landmark outlier detection.
- Muscle outlier detection.
- Muscle containment checks.
- Landmark-to-muscle consistency checks.
- Detection of unusually large rotations (>10°).

Optional visualizations can be generated to facilitate manual review of problematic cases.

---

## Ear Region Cropping

After alignment, separate right- and left-ear volumes are extracted.

For each ear:

1. The corresponding eardrum landmark is identified.
2. A configurable offset is applied to include anatomy medial to the eardrum.
3. A fixed-size cubic field of view is extracted.

Default settings:

```text
Voxel spacing: 0.2 mm isotropic
Crop size: 256 × 256 × 256 voxels
Physical field of view: 51.2 × 51.2 × 51.2 mm
Default offset: [20, 0, 0] mm
```

The crop is centered on:

```text
Landmark 10 → Right ear
Landmark 11 → Left ear
```

The offset helps include additional cochlear and middle-ear anatomy surrounding the tympanic membrane.

To ensure a consistent anatomical orientation across the dataset, left-ear crops are mirrored along the x-axis so that both ears follow the same orientation convention.

The final crop affines are adjusted to remain compatible with the coordinate system used by the historical P3-P4 preprocessing workflow.

---

## Intensity Normalization

The final preprocessing step applies intensity normalization.

Voxel intensities are first clipped to:

```text
[-1000, 2007] HU
```

The clipped values are subsequently normalized to:

```text
[0,1]
```

using:

```text
(I - HUmin) / (HUmax - HUmin)
```

This normalization strategy matches the final intensity-normalization step used by the original tissue-segmentation preprocessing workflow.

---

## Output Files

For each processed patient, the pipeline generates:

```text
patient/
├── patient_right_ear_raw_hu.nii.gz
├── patient_left_ear_raw_hu.nii.gz
├── patient_right_ear_0000.nii.gz
├── patient_left_ear_0000.nii.gz
├── patient_transform_log.json
└── visualizations/
```

### File Description

- `*_raw_hu.nii.gz`: Ear crop stored in Hounsfield Units.
- `*_0000.nii.gz`: Final normalized ear crop in the range [0,1].
- `*_transform_log.json`: Complete record of all transformations applied during preprocessing.
- `visualizations/`: Optional landmark and quality-control figures.

The primary outputs intended for downstream segmentation are:

```text
patient_right_ear_0000.nii.gz
patient_left_ear_0000.nii.gz
```

These files contain normalized voxel intensities and are the recommended inputs for eardrum segmentation models.

---

# Segmentation

*Coming soon.*

This section will describe the automatic eardrum segmentation model, including:

- Model architecture
- Training procedure
- Inference workflow
- Output segmentation masks
- Performance evaluation
- Usage examples

## Mapping Results Back to the Original CT Space

The preprocessing pipeline described above generates ear-centered volumes in a standardized reference frame that is optimized for landmark detection, anatomical alignment, and segmentation.

For downstream applications such as visualization in the original patient CT, surgical planning, anatomical measurements, or integration with other processing pipelines, an additional post-processing step is available to map segmentation results back to the coordinate system of the original CT scan.

### Purpose

During preprocessing, multiple spatial transformations are applied:

- Head ROI extraction.
- Orientation standardization.
- Landmark-based Frankfort-plane alignment.
- Ear-centered cropping.
- Left-ear mirroring.
- Resampling between multiple image resolutions.

As a result, the final segmentation masks, STL models, and landmarks are no longer expressed in the coordinate system of the original CT scan.

The mapping pipeline reconstructs the inverse transformation chain and restores all outputs to their original physical location.

### Inputs

The mapping procedure uses:

- P1 transformation logs.
- P2 transformation logs.
- P3 transformation logs.
- Intermediate NIfTI volumes generated during preprocessing.
- Segmentation masks.
- STL surface models.
- Landmark markup files.

### Transformation Chain

The complete forward preprocessing workflow can be summarized as:

```text
Original CT
    ↓
Head crop
    ↓
Reference-frame standardization
    ↓
Landmark-based alignment
    ↓
Ear cropping
    ↓
Left-ear mirroring (if applicable)
    ↓
Final result space
```

The mapping script computes the inverse of this chain and builds a single transformation matrix:

```text
M : Result voxel space → Original CT voxel space
```

This matrix combines all preprocessing steps into a single voxel-to-voxel transformation.

### Mapping of Segmentation Masks

Segmentation masks are mapped back to the original CT grid using nearest-neighbour interpolation.

The output mask therefore:

- Matches the dimensions of the original CT scan.
- Preserves discrete label values.
- Can be directly overlaid on the original image.

### Mapping of STL Models

STL meshes are mapped by transforming every vertex from result space back into the original scan coordinate system.

For each vertex:

```text
Result world coordinates
        ↓
Result voxel coordinates
        ↓
Original voxel coordinates
        ↓
Original world coordinates
```

The resulting STL occupies the same anatomical location as the corresponding structure within the original CT scan.

### Mapping of Landmark Files

Landmark markup files are mapped similarly to STL vertices.

Each landmark position is transformed from the ear-centered processing space into the physical coordinate system of the original CT scan.

This allows landmarks to be visualized together with the original image and any other anatomical annotations.

### Coordinate-System Conversion

The mapping script supports both:

```text
LPS
RAS
```

coordinate conventions.

When required, coordinates are automatically converted between LPS and RAS systems during transformation.

This ensures compatibility with software packages such as:

- 3D Slicer
- ITK / SimpleITK
- Medical imaging toolkits using LPS conventions

### Left-Ear Mirroring

During preprocessing, left-ear crops are mirrored to match the orientation of right-ear crops.

When mapping results back to the original scan:

- Left-ear results are automatically unmirrored.
- Original anatomical laterality is restored.

This guarantees that mapped results appear on the correct side of the patient's anatomy.

### Validation

A validation mode is provided to verify that the inverse transformation chain is correct.

The validation procedure:

1. Computes the centroid of the processed ear mask.
2. Maps this centroid back to the original CT space.
3. Compares the mapped location with the auditory-canal centroid recorded during P1 preprocessing.

Small distances between both points indicate that the transformation chain has been reconstructed correctly.

Example:

```bash
python map_results_to_original.py \
    --logs_dir Logs/transform_logs \
    --processed_dir Processed-Data \
    --masks_dir Results/masks \
    --validate
```

### Outputs

The mapping pipeline can generate original-space versions of:

```text
Segmentation masks
STL models
Bone STL models
Landmark markup files
```

All outputs are expressed in the coordinate system of the original CT scan and can therefore be used for:

- Visualization in the native CT.
- Anatomical measurements.
- Registration with other image data.
- Surgical planning.
- External modelling and simulation workflows.

### Summary

The preprocessing pipeline operates in a standardized ear-centered reference frame optimized for segmentation performance.

The mapping step reconstructs the full inverse transformation chain and restores masks, STL models, and landmarks to the coordinate system of the original CT scan, preserving anatomical location and compatibility with external workflows.
