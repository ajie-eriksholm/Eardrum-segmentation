# Eardrum Detection and Segmentation Pipeline

## Table of Contents

- [introduction
  - #purpose
  - [Input Requirements](#inputrdrum-anatomy
- #pre-processing
  - #overview
  - [Head Localization-and-cropping
  - [Reference Frame Construction](#- #landmark-detection
  - [Frankfort Plane-alignment
  - #ear-region-cropping
  - [Intensity Normalization
- [output-files
- #usage
``

# 1. Introduction

## 1.1 Purpose

This repository contains a fully automated high-resolution preprocessing pipeline for detecting the tympanic membrane (eardrum) in head CT scans and extracting standardized ear-centered volumes for downstream analysis and segmentation tasks.

The pipeline combines and extends the functionality of the original preprocessing workflow:

- `p1_preprocessing.py`
- `p2_preprocessing.py`
- `p3_preprocessing.py`
- `p4_preprocessing.py`

into a single script.

Unlike the original workflow, the final output is generated directly from the native-resolution CT scan. A temporary low-resolution representation is only created internally for anatomical landmark detection. The final ear volumes are extracted at high isotropic resolution (default: **0.2 mm**) to preserve anatomical detail around the eardrum and middle-ear structures.

The pipeline performs:

1. Head localization.
2. Anatomical landmark detection.
3. Frankfort plane alignment.
4. Ear-centered ROI extraction.
5. Intensity normalization.

The resulting volumes can be used as inputs for eardrum segmentation models or other anatomical analyses.

---

## 1.2 Input Requirements

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
- Voxel spacing finer than:

```text
(1.5, 1.5, 5.5) mm
```

Scans with lower spatial resolution are skipped automatically.

### Required Model

The pipeline requires a trained landmark-detection model:

```text
best_model_*.pth
```

which predicts six anato*ical landmarks used for head align*ent and eardrum localization.

###*Software Dependencies

This pipeli*e is adapted from the preprocessin* workflow originally developed for*the tissue segmentation project an* therefore requires the **same env*ronments, dependencies, and pretra*ned models** used in that pipeline*

Main dependencies include:

- Py*orch
- SimpleITK
- NiBabel
- NumPy*- SciPy
- PyVista
- TotalSegmentat*r

---

## 1.3 Eardrum Anatomy

Th* tympanic membrane, commonly refer*ed to as the eardrum, is a thin me*brane separating the external audi*ory canal from the middle ear cavi*y.

Its main functions are:

- Con*erting sound pressure waves into m*chanical vibrations.
- Transmittin* acoustic energy to the ossicular *hain.
- Protecting the middle ear *rom the external environment.

The*eardrum is located at the medial e*d of the external auditory canal a*d provides a robust anatomical lan*mark for standardizing ear-centere* coordinate systems.

In this pipe*ine, dedicated deep-learning landm*rks corresponding to the right and*left tympanic membranes (**landmar*s 10 and 11**) are detected and us*d to localize the final high-resol*tion ear volumes.

---

# 2. Pre-p*ocessing

## 2.1 Overview

This pr*processing pipeline is a high-reso*ution adaptation of the preprocess*ng workflow originally developed f*r tissue segmentation.

The same a*atomical normalization principles *re used:

- Standardized head loca*ization.
- Anatomically meaningful*coordinate systems.
- Landmark-bas*d orientation correction.
- Consis*ent ear-centered cropping.

The ke* difference is that the final extr*cted ear volumes preserve high-fre*uency anatomical information. Alth*ugh a low-resolution volume is tem*orarily generated for landmark det*ction, the final outputs are obtai*ed directly from a high-resolution*representation of the original CT *can.

---

## 2.2 Head Localizatio* and Cropping

The first stage ide*tifies the head region and removes*unnecessary anatomy.

### Intensit* clipping

CT intensities are clip*ed to:

```text
[-1000, 2007] HU
`*`

to suppress extreme outliers an* standardize the dynamic range.

#*# Ear localization

The `head_glan*s_cavities` task from TotalSegment*tor is used to segment relevant an*tomical structures.

The centroids*of the left and right auditory can*ls are extracted from the segmenta*ion and used to define the midpoin* between both ears.

### Head ROI *xtraction

A fixed-size head volum* is cropped around the ear midpoin* while preserving the original sca* resolution.

This step reduces co*putational cost while ensuring tha* all relevant head structures rema*n available for downstream process*ng.

---

## 2.3 Reference Frame C*nstruction

After head cropping, t*e scan is transformed into a stand*rdized local coordinate system.

T*e cropped volume is:

1. Resampled*to a standardized spacing.
2. Reor*ented into a consistent anatomical*orientation.
3. Assigned a common *ocal origin.

From this reference *rame, two separate volumes are gen*rated.

### Low-resolution detecti*n volume

A temporary low-resoluti*n volume is generated exclusively *or*landmark detection*

Characteristics*

*``*ext**pacing***** mm*Padding:**40 × *40 ×*540
*inal size**256 ×*256 × 256*```

*his representation matches the inp*t resolution used during training *f the landmark detection network.
*### High-resolution working volume*
A separate high-resolution volume*is generated directly from the loc*l reference frame.

Characteristic*:

```text
Default spacing: 0.2 mm*isotropic
```

*o intermediate downsampling is per*ormed after this stage.

This volu*e is used for all subsequent cropp*ng and output generation steps.

-*-

## 2.4 Landmark Detection

Anat*mical landmark detection is perfor*ed using a 3D U-Net model.

The ne*work predicts six landmarks:

- La*dmark 8
- Landmark 9
- Landmark *0*-*Landmark *1*- Landmark**2
- Landmark *3

*he*predicted coordinates*are converted into physical world *oordinates within the standardized*reference frame.

Particular*importance is given to:

*``*ext
Landmark 10 → Right eardrum
La*dmark *1 → Left eardrum
*``

*hese landmarks*are later used*both for Frankfort plane alignment*and for defining*the center of the final ear*crops.

---

*# 2.5 Frankfort Plane Alignment

T**reduce inter*subject orientation variability,*the head is aligned using the*Frankfort plane.

*## Standard*Mode

In*the*default configuration**the Frankfort*plane is estimated*from the detected anatomical*landmarks.

The scan*is*rotated so that:

-*The*Frankfort plane becomes horizontal*
- The left*right anatomical axis is standardi*ed across subjects.

###*No-Eyes Mode

When eye*based*landmarks are unavailable or unrel*able,*an alternative strategy can be use*.

Additional segmentation of:

- *asseter muscles
- L*teral*pterygoid muscles*
is obtained*using*TotalSegmentator*

These anatomical*structures*are combined*with eardrum*landmarks to estimate a*robust anatomical reference plane.*
### Quality Control*
*everal quality*control checks*are performed, including:

-*Left*right*eardrum symmetry.
- Out*ier detection.
- Muscle*containment checks.
* Landmark*consistency checks.
- Excess*ve*rotation detection.

*isualizations*and diagnostic*information are generated to*facilitate quality assessment.

*--

*# 2*6 Ear Region Cropping

*fter alignment* separate*ear-centered volumes are extracted*

For each ear:

1* The*corresponding e*rdrum landmark is identified*
2* A configurable*offset is applied to include addit*onal*anatomical structures such as the *ochlea.
3.*A fixed-size cubic ROI*is extracted.

Default parameters*

```text**pacing* 0*2 mm*Crop*size:*256 ×*256 × *56 voxels
*hysical field*of*view: 51*2 mm ×*51.2*mm × 51.* mm
*``

*o*ensure consistent orientation acro*s the*dataset, the left-ear crop*is mirrored so that both ears foll*w the same anatomical convention.
*---

## *.7 Intensity*Normalization

*he final*stage performs intensity normaliza*ion.

First, voxel values are clip*ed to:

```text
[-1000, 2007] HU*```

*he clipped values are*then linearly normalized to:

```t*xt
[0, 1*
```

*his normalization strategy is iden*ical to the one used in the origin*l tissue segmentation workflow and*ensures compatibility with downstr*am deep-learning models.

---

# 3* Output Files

For each processed*patient, the following files are g*nerated:

```text
patient/
├── pat*ent_right_ear_raw_hu.nii.gz
├── pa*ient_left_ear_raw_hu.nii.gz
├── pa*ient_right_ear_0000.nii.gz
├── pat*ent_left_ear_0000.nii.gz
├── patie*t_transform_log.json
└── visualiza*ions/
```

### File Description

|*File | Description |
|--------|---*---------|
| `*_raw_hu.nii.gz` | E*r crop in original Hounsfield Unit* |
| `*_0000.nii.gz` | Normalized ear crop in the range [0,1] |
| `*_transform_log.json` | Complete processing and transformation log |
| `visualizations/` | Landmark and quality-control visualizations |

---

# 4. Usage

### Single Scan

```bash
python p_highres_ear_pipeline.py \
    --input_ct patient__CT.nii.gz \
    --output_dir output \
    --landmark_model best_model.pth
```

### Batch Processing

```bash
python p_highres_ear_pipeline.py \
    --raw_scans_dir raw_scans \
    --output_dir output \
    --landmark_model best_model.pth
```

### Optional Arguments

```bash
--highres_spacing 0.2
--ear_fov_voxels 256
--ear_offset_mm -10 0 0
--no_eyes True
--skip_alignment True
```

The pipeline produces standardized high-resolution ear volumes centered on the tympanic membrane while preserving anatomical detail required for eardrum detection and segmentation.
