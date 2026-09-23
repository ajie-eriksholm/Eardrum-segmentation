# Eardrum Detection and Segmentation

## Table of Contents

- #introduction
  - #eardrum-anatomy
  - #pipeline-overview
- [pre-processing
  - [Input Requirements](#input-requirementsead-localization-and-cropping
  - #reference-frame-construction
  - #landmark-detection
  - [Frankfort Plane Alignment
  - #ear-region-cropping
  - [Intensity Normalization
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

1. Intensity clipping and head localization.
2. Native-resolution head ROI extraction.
3. Reference-frame construction.
4. Anatomical landmark detection.
5. Frankfort plane alignment.
6. High-resolution ear extraction.
7. Intensity normalization.
8. Automatic eardrum segmentation.

The preprocessing pipeline combines the functionality of the original P1-P4 workflow into a single script while preserving compatibility with the existing landmark-detection model.

A temporary low-resolution volume is generated exclusively for landmark detection because the trained model expects a fixed 256×256×256 input grid. However, this low-resolution representation is never used as the final output.

All final ear crops are extracted at a configurable isotropic resolution (0.2 mm by default) directly from a higher-resolution reference frame, avoiding the coarse intermediate downsampling used in the original preprocessing pipeline.

The pipeline also preserves compatibility with the coordinate system and output space used by the historical P1-P4 workflow, allowing newly generated crops to remain spatially consistent with previously processed datasets.

---

# Pre-processing

## Input Requirements

The pipeline expects CT scans in NIfTI format:

```text
<patient_id>__CT.nii
