# Eardrum Detection and Segmentation

## Table of Contents

## Table of Contents

- #Introduction
  - #eardrum-anatomy
  - #pipeline-overview
- #pre-processing
  - #input-requirements
  - #overview
  - #head-localization-and-cropping
  - #reference-frame-construction
  - #landmark-detection
  - #frankfort-plane-alignment
  - #ear-region-cropping
  - #intensity-normalization
  - #output-files
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

1. Head localization.
2. Anatomical landmark detection.
3. Frankfort plane alignment.
4. High-resolution ear extraction.
5. Intensity normalization.
6. Automatic eardrum segmentation.

Currently, this repository contains the complete preprocessing pipeline. The automatic segmentation model will be incorporated in a future update.

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

The pipeline requires a trained landmark detection model:

```text
best_model_*.pth
```

which predicts six anatomical landmarks used for head alignment and ear localization.

### Software Dependencies

This preprocessing pipeline is adapted from the workflow originally developed for tissue segmentation. Therefore, it requires the same software environments, dependencies, and trained models used by that pipeline.

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
- Anatomically consistent coordinate systems.
- Landmark-based alignment.
- Standardized cropping.

The main difference is that the final ear volumes are generated from a high-resolution representation of the CT scan. Although a temporary low-resolution volume is created for landmark detection, no low-resolution data are used for the final outputs.

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

The centroids of the left and right auditory canals are extracted from the segmentation output. The midpoint between both canals is then calculated and used as the reference point for head localization.

### Head ROI Extraction

A fixed-size head ROI is cropped around the ear midpoint while preserving the native scan resolution.

This step reduces computational cost while retaining all anatomical structures required for subsequent processing stages.

---

## Reference Frame Construction

After head localization, a standardized local reference frame is created.

The cropped volume is:

1. Resampled.
2. Reoriented into a consistent anatomical orientation.
3. Assigned a common origin.

Two separate volumes are then generated.

### Low-Resolution Detection Volume

A temporary low-resolution volume is produced exclusively for landmark detection.

Characteristics:

```text
Spacing: 0.5 mm
Padding: 540 × 540 × 540 voxels
Final size: 256 × 256 × 256 voxels
```

This volume matches the dimensions and resolution used during landmark-model training.

### High-Resolution Working Volume

A second volume is generated specifically for extraction and output generation.

Characteristics:

```text
Default spacing: 0.2 mm isotropic
```

Unlike the original preprocessing workflow, this volume is not downsampled before extraction of the final ear crops.

Consequently, the extracted volumes preserve substantially more anatomical detail around the tympanic membrane and surrounding ear structures.

---

## Landmark Detection

Anatomical landmark detection is performed using a 3D U-Net model.

The network predicts six anatomical landmarks:

- Landmark 8
- Landmark 9
- Landmark 10
- Landmark 11
- Landmark 12
- Landmark 13

Predicted landmark locations are converted into physical world coordinates within the standardized reference frame.

The most relevant landmarks for this pipeline are:

```text
Landmark 10 → Right eardrum
Landmark 11 → Left eardrum
```

These landmarks are subsequently used for head alignment and extraction of the final ear-centered volumes.

---

## Frankfort Plane Alignment

To reduce orientation variability between subjects, the head is aligned using the Frankfort plane.

### Standard Mode

The default alignment strategy uses the predicted anatomical landmarks to estimate the Frankfort plane.

The scan is rotated such that:

- The Frankfort plane becomes horizontal.
- The left-right anatomical axis is standardized.

### No-Eyes Mode

When eye-based landmarks are unavailable or considered unreliable, an alternative alignment strategy can be used.

In this mode, TotalSegmentator is used to segment:

- Masseter muscles
- Lateral pterygoid muscles

These structures are combined with the eardrum landmarks to estimate a robust anatomical reference plane.

### Quality Control

Several quality-control checks are performed automatically, including:

- Left-right eardrum distance consistency.
- Landmark outlier detection.
- Muscle containment checks.
- Landmark-to-muscle consistency checks.
- Detection of unusually large rotations.

Diagnostic figures and visualizations are generated to facilitate review of potentially problematic scans.

---

## Ear Region Cropping

After alignment, separate ear-centered volumes are extracted.

For each ear:

1. The corresponding eardrum landmark is identified.
2. A configurable offset is applied to include additional anatomy medial to the eardrum, such as cochlear structures.
3. A fixed-size cubic ROI is extracted.

Default settings:

```text
Voxel spacing: 0.2 mm
Crop size: 256 × 256 × 256 voxels
Physical field of view: 51.2 mm × 51.2 mm × 51.2 mm
```

To ensure a consistent anatomical orientation across the dataset, the left-ear crop is mirrored so that both ears follow the same coordinate convention.

---

## Intensity Normalization

The final preprocessing step applies intensity normalization.

Voxel intensities are first clipped to:

```text
[-1000, 2007] HU
```

The clipped values are then linearly normalized to:

```text
[0, 1]
```

This normalization strategy is identical to the final normalization step used in the original tissue-segmentation preprocessing workflow.

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

- `*_raw_hu.nii.gz` : Cropped ear volume in original Hounsfield Units.
- `*_0000.nii.gz` : Normalized ear volume in the range [0,1].
- `*_transform_log.json` : Complete log of all transformations applied during processing.
- `visualizations/` : Landmark and quality-control visualizations.

---

# Segmentation

*Coming soon.*

This section will describe the automatic eardrum segmentation model, including:

- Model architecture.
- Training procedure.
- Inference workflow.
- Output segmentation masks.
- Performance evaluation.
- Usage examples.
