# Eardrum Detection and High-Resolution Ear Extraction Pipeline

## Table of Contents

- #introduction
  - #purpose
  - #input-requirements
  - #eardrum-anatomy
- #pre-processing
  - #overview
  - #head-localization-and-cropping
  - #reference-frame-construction
  - #landmark-detection
  - #frankfort-plane-alignment
  - #ear-region-cropping
  - #intensity-normalization
- #output-files
- #usage

---

# Introduction

## Purpose

This repository contains a fully automated high-resolution preprocessing pipeline for detecting the tympanic membrane (eardrum) in head CT scans and extracting standardized ear-centered volumes for downstream segmentation and anatomical analysis tasks.

The pipeline combines and extends the functionality of the original preprocessing workflow:

- `p1_preprocessing.py`
- `p2_preprocessing.py`
- `p3_preprocessing.py`
- `p4_preprocessing.py`

into a single script.

Unlike the original workflow, the final output is generated directly from the high-resolution CT scan. A low-resolution representation is created only internally for landmark detection because the trained landmark detection network expects a fixed input resolution. The final extracted ear volumes remain high-resolution throughout the process (default isotropic spacing of **0.2 mm**), preserving anatomical detail around the eardrum, middle ear cavity, and cochlea.

The pipeline performs:

1. Head localization.
2. Anatomical landmark detection.
3. Frankfort plane alignment.
4. Ear-centered ROI extraction.
5. Intensity normalization.

The resulting volumes can be used as inputs for eardrum segmentation models or other ear anatomy analysis pipelines.

---

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

This pipeline is adapted from the preprocessing workflow developed for the tissue segmentation project. Therefore, it requires the same software environments, dependencies, and trained models used by the tissue segmentation preprocessing pipeline.

Main dependencies include:

- PyTorch
- SimpleITK
- NiBabel
- NumPy
- SciPy
- PyVista
- TotalSegmentator

---

## Eardrum Anatomy

The tympanic membrane, commonly known as the eardrum, is a thin membrane that separates the external auditory canal from the middle ear cavity.

Its primary functions are:

- Converting sound pressure waves into mechanical vibrations.
- Transmitting acoustic energy to the ossicular chain.
- Acting as a barrier between the external and middle ear.

The eardrum is located at the medial end of the external auditory canal and provides a reliable anatomical landmark for defining a standardized ear-centered coordinate system.

In this pipeline, the right and left eardrums are represented by **landmarks 10 and 11**, respectively. These landmarks are detected automatically and are subsequently used for both head alignment and extraction of the final ear-centered volumes.

---

# Pre-processing

## Overview

This preprocessing pipeline is a high-resolution adaptation of the preprocessing workflow originally developed for tissue segmentation.

The same anatomical normalization principles are preserved:

- Head localization.
- Anatomically consistent coordinate systems.
- Landmark-based alignment.
- Standardized cropping.

The main difference is that the final ear volumes are generated from a high-resolution representation of the CT scan. Although a low-resolution copy is temporarily generated for landmark detection, no low-resolution data are used for the final output.

The processing steps are summarized below.

---

## Head Localization and Cropping

The first stage identifies the head and removes unnecessary anatomy.

### Intensity Clipping

CT intensities are clipped to:

```text
[-1000, 2007] HU
```

to reduce the influence of extreme intensity values while preserving relevant anatomical structures.

### Ear Localization

TotalSegmentator is executed using the `head_glands_cavities` task.

The centroids of the:

- Right auditory canal
- Left auditory canal

are extracted from the segmentation output.

The midpoint between both auditory canals is computed and used as the reference point for head cropping.

### Head ROI Extraction

A fixed-size head ROI is extracted around the ear midpoint while preserving the original scan resolution.

This reduces computational requirements while retaining all anatomical structures needed for subsequent landmark detection and alignment.

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

This volume matches the resolution and dimensions used during landmark-model training.

### High-Resolution Working Volume

A second volume is generated specifically for the final outputs.

Characteristics:

```text
Default spacing: 0.2 mm isotropic
```

Unlike the original preprocessing workflow, this volume is not downsampled before ear extraction.

Consequently, the final crops preserve substantially more anatomical detail around the tympanic membrane and surrounding structures.

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

These landmarks define the center of the final ear volumes.

---

## Frankfort Plane Alignment

To reduce orientation variability between subjects, the head is aligned using the Frankfort plane.

### Standard Mode

The default alignment strategy uses the predicted anatomical landmarks to estimate the Frankfort plane.

The scan is rotated such that:

- The Frankfort plane becomes horizontal.
- The left-right anatomical axis is standardized.

### No-Eyes Mode

When eye-based landmarks are not available or are considered unreliable, an alternative alignment strategy can be used.

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
2. A configurable offset is applied to include relevant anatomy medial to the eardrum (e.g., cochlear structures).
3. A fixed-size cubic ROI is extracted.

Default settings:

```text
Voxel spacing: 0.2 mm
Crop size: 256 × 256 × 256 voxels
Physical field of view: 51.2 mm × 51.2 mm × 51.2 mm
```

To ensure a consistent anatomical orientation across the dataset, the left-ear crop is mirrored so that both ears share the same coordinate convention.

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

This normalization procedure is identical to the final normalization step used in the original tissue segmentation preprocessing workflow.

---

# Output Files

For each patient, the pipeline generates:

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

# Usage

## Single Scan

```bash
python p_highres_ear_pipeline.py \
    --input_ct patient__CT.nii.gz \
    --output_dir output \
    --landmark_model best_model.pth
```

## Batch Processing

```bash
python p_highres_ear_pipeline.py \
    --raw_scans_dir raw_scans \
    --output_dir output \
    --landmark_model best_model.pth
```

## Optional Arguments

```bash
--highres_spacing 0.2
--ear_fov_voxels 256
--ear_offset_mm -10 0 0
--no_eyes True
--skip_alignment True
```

The pipeline produces standardized high-resolution ear volumes centered on the tympanic membrane while preserving the anatomical detail required for eardrum detection and segmentation.
