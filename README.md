# Eardrum Detection and Segmentation

## Table of Contents

- [introduction
  - [Eardrum Anatomy](#eardrum-anatomyrview
- #pre-processing
  - [Input-requirements
  - [Overview](#overviewLocalization and Cropping](#head-ference-frame-construction
  - [Landmark Detection](#landmark-ort-plane-alignment
  - [Ear Region Cropping](#ear-region-y-normalization
  - #output-files
- [Segmentation](# Introduction

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

1. Intensity clipping and head localization.
2. Native-resolution head ROI extraction.
3. Reference-frame construction.
4. Anatomical landmark detection.
5. Frankfort plane alignment.
6. High-resolution ear extraction.
7. Intensity normalization.
8. Automatic eardrum segmentation.

The preprocessing pipeline combines the functionality of the original P1-P4 workflow into a single script while preserving compatibility with the existing landmark-detection model.

A temporary low-resolution volume is generated exclusively for landmark detection because the trained model expects a fixed 256 × 256 × 256 input grid. However, this low-resolution representation is never used as the final output.

All final ear crops are extracted at a configurable isotropic resolution (0.2 mm by default) directly from a higher-resolution reference frame, avoiding the coarse intermediate downsampling used in the original preprocessing pipeline.

The pipeline also preserves compatibility with the coordinate system and output space used by the historical P1-P4 workflow, allowing newly generated crops to remain spatially consistent with previously processed datasets.

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
- Anatomically consistent coordinate systems.
- Landmark-based alignment.
- Standardized cropping.

The key difference is that landmark detection is performed on a temporary low-resolution representation, while all final ear volumes are generated from a high-resolution reference frame.

Final outputs are therefore produced without the coarse intermediate downsampling used in the original P1-P4 workflow.

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

Default ROI size:

```text
200 × 270 × 200 mm
```

with an additional superior margin of:

```text
50 mm
```

This step reduces computational cost while retaining all anatomical structures required for subsequent processing stages.

---

## Reference Frame Construction

After head localization, a standardized local reference frame is created.

The cropped head ROI is first resampled to a common local reference frame with:

```text
Spacing: 0.5 mm isotropic
Origin: (0,0,0)
Orientation signs: (-X, -Y, +Z)
```

The volume is automatically flipped when necessary so that all scans share the same orientation convention prior to landmark detection.

Two separate representations are then generated from this common reference frame.

### Landmark Detection Volume

A temporary low-resolution volume is produced exclusively for landmark detection.

Characteristics:

```text
Spacing: 0.5 mm isotropic
Padding: 540 × 540 × 540 voxels
Final size: 256 × 256 × 256 voxels
```

This volume reproduces the dimensions and resolution used during landmark-model training and is discarded once landmark prediction is complete.

### High-Resolution Working Space

A second reference space is created specifically for extraction of the final ear crops.

Characteristics:

```text
Default spacing: 0.2 mm isotropic
```

Unlike the original preprocessing workflow, this representation is not downsampled to the coarse landmark-detection resolution before generating the final outputs.

When `fast=False`, the entire high-resolution head volume is explicitly resampled and rotated prior to ear extraction.

When `fast=True`, each ear field of view is resampled directly from the local reference frame without first creating the full high-resolution head volume, substantially reducing memory requirements.

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

Landmarks are predicted as heatmaps and subsequently converted into physical world coordinates defined within the standardized local reference frame.

The coordinate conversion follows the same convention used by the original P2 preprocessing workflow to maintain compatibility with previously trained models and downstream processing tools.

The most relevant landmarks for this pipeline are:

```text
Landmark 10 → Right eardrum
Landmark 11 → Left eardrum
```

These landmarks are used both for Frankfort-plane alignment and for extraction of the final ear-centered volumes.

---

## Frankfort Plane Alignment

To reduce anatomical orientation variability across subjects, the head is aligned using the Frankfort plane.

### Standard Mode

The default alignment strategy uses landmarks 10–13 to estimate the Frankfort plane.

The resulting rotation aligns:

- The Frankfort plane with the horizontal plane.
- The left-right anatomical axis with the global x-direction.

The same transformation is subsequently applied to the extracted ear volumes.

### No-Eyes Mode

When eye-based landmarks are unavailable or unreliable, an alternative alignment strategy can be used via:

```bash
--no_eyes True
```

In this mode, TotalSegmentator is used to segment:

- Masseter muscles
- Lateral pterygoid muscles

Muscle-derived anatomical landmarks are combined with eardrum landmarks to estimate a robust anatomical reference plane using SVD-based plane fitting.

### Skip Alignment Mode

For experiments where alignment is not desired, Frankfort-plane alignment can be skipped entirely:

```bash
--skip_alignment True
```

In this configuration, landmark detection is still performed, but no rotational normalization is applied.

### Quality Control

Several automated quality-control checks are performed:

- LM10-LM11 distance validation.
- Muscle containment checks.
- Landmark outlier detection.
- Muscle outlier detection.
- Landmark-to-muscle consistency checks.
- Detection of unusually large rotations (>10°).

Potentially problematic scans can be flagged automatically, and optional visualizations can be generated for manual review.

---

## Ear Region Cropping

After Frankfort-plane alignment, separate right- 
