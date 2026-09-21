"""
Pre-processing adapted for eardrum segmentation.

Combines everything that p1_preprocessing.py -> p2_preprocessing.py ->
p3_preprocessing.py -> p4_preprocessing.py do together (head localization,
Frankfort-plane landmark detection + alignment with the full no_eyes/QA
fallback machinery, ear ROI cropping, and HU normalization) but:

  * NEVER downsamples the head to a coarse ~1 mm / 256^3 grid. That grid is
    still built internally (because the trained landmark-detection network
    expects that exact scale), but only as a disposable "detection copy"
    used to locate the landmarks and compute the Frankfort-plane rotation.
  * The volume that is actually cropped/saved is resampled directly from the
    native-resolution scan to a configurable isotropic resolution (default
    0.2 mm) with no intermediate downsampling.
  * Crops a smaller FOV (default 256^3 voxels = 51.2 mm cube at 0.2 mm)
    centered exactly on the right/left eardrum landmarks (landmark 10 / 11),
    instead of the small 90^3 voxel ROI that P3 used at ~1 mm resolution.
  * Ends with the same intensity clipping + [0, 1] normalization P4 applies
    as its final step.

Usage:
    python p_highres_ear_pipeline.py --input_ct path/to/patient__CT.nii.gz \
        --output_dir /path/to/output --landmark_model /path/to/best_model.pth

    # or process every *__CT.nii[.gz] file in a directory:
    python p_highres_ear_pipeline.py --raw_scans_dir /path/to/raw --output_dir /path/to/output
"""

import os
import sys
import warnings
import json
import csv
import argparse
from datetime import datetime

import numpy as np
import nibabel as nib
import SimpleITK as sitk
import torch
import pyvista as pv
from scipy.ndimage import zoom, affine_transform
from scipy.spatial.transform import Rotation as R
from totalsegmentator.python_api import totalsegmentator

pv.set_plot_theme("document")
pv.global_theme.background = 'white'

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# ============================================================================
# Configuration (all overridable via CLI arguments — see parse_arguments())
# ============================================================================

# --- Stage 1: intensity clipping + head localization/crop (native resolution) ---
INTENSITY_CLIP_RANGE = (-1000, 2007)
VOXEL_THRESHOLD = (1.5, 1.5, 5.5)          # skip scans coarser than this
HEAD_ROI_SIZE_MM = np.array([200, 270, 200])
HEAD_ROI_EXTRA_TOP_MM = 50

LABELS_INFO = {
    1: "Left Eye",
    2: "Right Eye",
    16: "Right Auditory Canal",
    17: "Left Auditory Canal",
}

# --- Stage 2: disposable low-res volume used ONLY for landmark detection ---
# (must match the resolution/shape the trained model was trained on)
LOWRES_SPACING = [0.5, 0.5, 0.5]
LOWRES_PAD_SHAPE = (540, 540, 540)
LOWRES_FINAL_SHAPE = (256, 256, 256)
INTERPOLATOR = sitk.sitkBSpline

# --- Final, high-resolution output volume (this is what actually gets cropped) ---
HIGHRES_SPACING = [0.2, 0.2, 0.2]
EAR_FOV_VOXELS = [256, 256, 256]           # 256 * 0.2mm = 51.2mm cube
# Small +X shift so the crop encloses more of the cochlea (medial to the
# eardrum landmark); mirrored automatically for the left ear (see process_scan).
EAR_OFFSET_MM = np.array([100.0, 0.0, 0.0])
MIRROR_AXIS = 0                            # axis used to mirror the left ear onto the right ear's orientation

# --- Landmark detection ---
LANDMARK_IDS = [8, 9, 10, 11, 12, 13]
NUM_LANDMARKS = len(LANDMARK_IDS)

# --- Final normalization (P4 step) ---
MIN_HU = -1000
MAX_HU = 2007

NIFTI_EXTENSIONS = ('.nii.gz', '.nii')

# --- No-eyes / QA fallback (ported from p2_preprocessing.py) ---
HEAD_MUSCLES_LABEL_IDS = {
    'masseter_right': 1,
    'masseter_left': 2,
    'lateral_pterygoid_right': 5,
    'lateral_pterygoid_left': 6,
}
OUTLIER_CENTROID_RATIO_MUSCLE = 1.8
OUTLIER_CENTROID_RATIO_LANDMARK = 2.0
LANDMARK_LABELS_FOR_OUTLIER = ('lm10', 'lm11')
LM10_LM11_MIN_DIST_MM = 80.0
LM10_LM11_MAX_DIST_MM = 200.0
ROTATION_FLAG_THRESHOLD_DEG = 10.0


# ============================================================================
# Generic utilities
# ============================================================================

def strip_nifti_extension(filename):
    for ext in NIFTI_EXTENSIONS:
        if filename.endswith(ext):
            return filename[:-len(ext)]
    return filename


def extract_patient_id_from_raw_filename(filename):
    stem = strip_nifti_extension(filename)
    return stem[:-len('__CT')] if stem.endswith('__CT') else stem


def is_raw_ct_scan(filename):
    return strip_nifti_extension(filename).endswith('__CT')


def is_already_processed(patient_id, output_dir):
    """A patient is considered done once both final normalized ear crops exist."""
    patient_output_dir = os.path.join(output_dir, patient_id)
    final_paths = [
        os.path.join(patient_output_dir, f"{patient_id}_right_ear_0000.nii.gz"),
        os.path.join(patient_output_dir, f"{patient_id}_left_ear_0000.nii.gz"),
    ]
    return all(os.path.isfile(p) and os.path.getsize(p) > 0 for p in final_paths)


def clip_intensity(data, clip_range=INTENSITY_CLIP_RANGE):
    return np.clip(data, clip_range[0], clip_range[1])


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    return path


def voxel_to_world(voxel_coords, affine):
    voxel_h = np.append(voxel_coords, 1.0)
    return (affine @ voxel_h)[:3]


def world_to_voxel(world_coords, affine):
    spacing = np.array([np.abs(affine[i, i]) for i in range(3)])
    origin = affine[:3, 3]
    return (world_coords - origin) / spacing


class _AlignmentSkipped(Exception):
    """Raised internally when skip_alignment=True, to exit the alignment try-block cleanly."""


class _LandmarkOutlierError(Exception):
    """Raised when lm10/lm11 is detected as a plane outlier (no_eyes QA checks)."""


CUDA_WARNING_PATTERNS = [
    r".*cuda capability.*",
    r".*Please install PyTorch with a following CUDA.*",
    r".*is not compatible with the current PyTorch installation.*",
]


def suppress_incompatible_cuda_warnings():
    for pattern in CUDA_WARNING_PATTERNS:
        warnings.filterwarnings("ignore", message=pattern, category=UserWarning)


def get_device():
    with warnings.catch_warnings():
        suppress_incompatible_cuda_warnings()
        if torch.cuda.is_available():
            try:
                major, minor = torch.cuda.get_device_capability()
                compute_capability = float(f"{major}.{minor}")
                if compute_capability >= 7.0:
                    print(f"GPU detected and compatible - using CUDA (compute capability: {compute_capability})")
                    return torch.device('cuda')
                print(f"GPU compute capability {compute_capability} < 7.0 (required by PyTorch 2.x) - using CPU")
                return torch.device('cpu')
            except Exception as e:
                print(f"Error checking GPU compatibility: {e} - using CPU")
                return torch.device('cpu')
        print("No GPU available - using CPU")
        return torch.device('cpu')


def totalsegmentator_device(device):
    return 'gpu' if device.type == 'cuda' else 'cpu'


def run_totalsegmentator(input_path, task, device, output_path):
    """Run TotalSegmentator, hiding unsupported GPUs from its subprocess when falling back to CPU."""
    seg_device = totalsegmentator_device(device)
    original_cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    try:
        if seg_device == 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        with warnings.catch_warnings():
            suppress_incompatible_cuda_warnings()
            seg_img = totalsegmentator(input_path, task=task, device=seg_device)
    finally:
        if original_cuda_visible_devices is None:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible_devices

    if seg_img is None:
        raise ValueError(f"TotalSegmentator task '{task}' failed or returned no output.")
    nib.save(seg_img, output_path)
    return seg_img, output_path


# ============================================================================
# Landmark-detection model (identical architecture to p2_preprocessing.py)
# ============================================================================

class DoubleConv(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            torch.nn.InstanceNorm3d(out_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            torch.nn.InstanceNorm3d(out_ch),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet3D(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_features=16):
        super().__init__()
        f = base_features
        self.enc1 = DoubleConv(in_channels, f)
        self.enc2 = DoubleConv(f, f * 2)
        self.enc3 = DoubleConv(f * 2, f * 4)
        self.enc4 = DoubleConv(f * 4, f * 8)
        self.pool = torch.nn.MaxPool3d(2)
        self.up = torch.nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.dec3 = DoubleConv(f * 8 + f * 4, f * 4)
        self.dec2 = DoubleConv(f * 4 + f * 2, f * 2)
        self.dec1 = DoubleConv(f * 2 + f, f)
        self.out_conv = torch.nn.Conv3d(f, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        return self.out_conv(d1)


# ============================================================================
# Stage 1 (from P1): head localization and native-resolution crop
# ============================================================================

def compute_centroids(segmented_data, affine, labels_info=LABELS_INFO):
    centroids = []
    for label_id, label_name in labels_info.items():
        coords = np.argwhere(segmented_data == label_id)
        if coords.size > 0:
            centroid_voxel = coords.mean(axis=0)
            centroid_mm = nib.affines.apply_affine(affine, centroid_voxel)
            centroids.append((label_id, label_name, centroid_mm))
    return centroids


def crop_to_roi_fixed(image_data, affine, voxel_spacing, midpoint_mm,
                       roi_size_mm=HEAD_ROI_SIZE_MM, extra_top_mm=HEAD_ROI_EXTRA_TOP_MM):
    """Crop a fixed-size ROI around the ear midpoint, at native resolution (no resampling)."""
    inv_affine = np.linalg.inv(affine)
    midpoint_voxel = np.round(nib.affines.apply_affine(inv_affine, midpoint_mm)).astype(int)
    roi_size_voxels = np.round(roi_size_mm / voxel_spacing).astype(int)
    half_size = roi_size_voxels // 2
    roi_start = midpoint_voxel - half_size
    roi_end = midpoint_voxel + half_size
    extra_voxels = int(np.round(extra_top_mm / voxel_spacing[2]))
    roi_end[2] += extra_voxels
    roi_size_voxels[2] += extra_voxels

    roi = np.full(roi_size_voxels, -1000, dtype=image_data.dtype)
    data_shape = np.array(image_data.shape)
    data_start = np.maximum(roi_start, 0)
    data_end = np.minimum(roi_end, data_shape)
    roi_start_in_roi = data_start - roi_start
    roi_end_in_roi = roi_start_in_roi + (data_end - data_start)
    data_slices = tuple(slice(data_start[i], data_end[i]) for i in range(3))
    roi_slices = tuple(slice(roi_start_in_roi[i], roi_end_in_roi[i]) for i in range(3))
    roi[roi_slices] = image_data[data_slices]

    new_origin = nib.affines.apply_affine(affine, roi_start)
    new_affine = affine.copy()
    new_affine[:3, 3] = new_origin
    return roi, new_affine


def resample_sitk_volume(volume_path, new_spacing, interpolator=INTERPOLATOR, default_value=-1000):
    volume = sitk.ReadImage(volume_path, sitk.sitkFloat32)
    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()
    new_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]
    return sitk.Resample(
        volume, new_size, sitk.Transform(), interpolator,
        volume.GetOrigin(), new_spacing, volume.GetDirection(),
        default_value, volume.GetPixelID(),
    )


def reset_origin_and_orient(data, affine, target_signs=np.array([-1, -1, 1])):
    """
    Reset origin to (0,0,0) and standardize orientation signs (LAS-style),
    exactly like P1 steps 6-7. Defines the shared local coordinate frame used
    by BOTH the low-res detection volume and the high-res output volume.
    """
    affine = affine.copy()
    affine[:3, 3] = [0.0, 0.0, 0.0]

    current_signs = np.sign(np.diag(affine[:3, :3]))
    needs_flip = ~np.isclose(current_signs, target_signs)

    if np.any(needs_flip):
        corrected_data = data.copy()
        for axis in range(3):
            if needs_flip[axis]:
                corrected_data = np.flip(corrected_data, axis=axis)
        corrected_affine = np.eye(4)
        for axis in range(3):
            spacing_magnitude = abs(affine[axis, axis])
            corrected_affine[axis, axis] = target_signs[axis] * spacing_magnitude
        return np.ascontiguousarray(corrected_data), corrected_affine

    return data, affine


def build_head_crop(clipped_data, affine, voxel_size, device, patient_output_dir, patient_id):
    """Run head_glands_cavities segmentation, find ear centroids, crop native-res ROI."""
    clipped_path = os.path.join(patient_output_dir, f"{patient_id}_CT_intensity_clipped.nii.gz")
    nib.save(nib.Nifti1Image(clipped_data, affine), clipped_path)

    print("Running head_glands_cavities segmentation...")
    seg_path = os.path.join(patient_output_dir, f"{patient_id}_CT_cavities_segmented.nii.gz")
    _, seg_path = run_totalsegmentator(clipped_path, 'head_glands_cavities', device, seg_path)
    seg_img = nib.load(seg_path)
    seg_data = seg_img.get_fdata()

    centroids = compute_centroids(seg_data, seg_img.affine)
    left_ear = next((c[2] for c in centroids if c[0] == 17), None)
    right_ear = next((c[2] for c in centroids if c[0] == 16), None)
    if left_ear is None or right_ear is None:
        raise RuntimeError(f"Could not find both ear centroids for {patient_id}.")
    midpoint_mm = (left_ear + right_ear) / 2

    print("Cropping to fixed head ROI (native resolution)...")
    cropped_data, cropped_affine = crop_to_roi_fixed(clipped_data, affine, voxel_size, midpoint_mm)

    centroid_info = {name: [float(x) for x in c] for _, name, c in [(lid, n, c) for lid, n, c in centroids]}
    return cropped_data, cropped_affine, midpoint_mm, centroid_info


# ============================================================================
# Stage 2: local reference frame + disposable low-res detection volume
#          + the actual high-resolution working volume
# ============================================================================

def build_reference_frames(cropped_data, cropped_affine, patient_output_dir, patient_id, highres_spacing):
    """
    From the native-resolution head crop, build:
      - `local_data` / `local_affine`: the shared local coordinate frame
        (resampled to LOWRES_SPACING, origin reset, orientation standardized).
        This is the SAME frame both branches below are derived from, so world
        mm coordinates mean the same physical location in both.
      - `lowres_data` / `lowres_affine`: local_data padded to 540^3 and
        downsampled to 256^3 — disposable input for the landmark-detection
        network (matches the resolution it was trained on).
      - `highres_data` / `highres_affine`: local_data resampled directly to
        `highres_spacing` (no padding/downsampling) — this is the volume the
        final ear crops are taken from.
    """
    tmp_dir = os.path.join(patient_output_dir, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    cropped_path = os.path.join(tmp_dir, f"{patient_id}_cropped_native.nii.gz")
    nib.save(nib.Nifti1Image(cropped_data, cropped_affine), cropped_path)

    print(f"Resampling head crop to {LOWRES_SPACING} mm (local reference frame)...")
    resampled_img = resample_sitk_volume(cropped_path, LOWRES_SPACING)
    resampled_img.SetOrigin((0.0, 0.0, 0.0))
    resampled_path = os.path.join(tmp_dir, f"{patient_id}_local_frame.nii.gz")
    sitk.WriteImage(resampled_img, resampled_path)

    local_nib = nib.load(resampled_path)
    local_data, local_affine = reset_origin_and_orient(local_nib.get_fdata(), local_nib.affine)
    local_path = os.path.join(tmp_dir, f"{patient_id}_local_frame_oriented.nii.gz")
    nib.save(nib.Nifti1Image(local_data, local_affine), local_path)

    # --- Disposable low-res volume for the landmark-detection network ---
    print(f"Padding to {LOWRES_PAD_SHAPE} and downsampling to {LOWRES_FINAL_SHAPE} for landmark detection...")
    pad_width = [(0, max(0, LOWRES_PAD_SHAPE[i] - local_data.shape[i])) for i in range(3)]
    padded_data = np.pad(local_data, pad_width, mode='constant', constant_values=-1000)
    zoom_factors = [LOWRES_FINAL_SHAPE[i] / padded_data.shape[i] for i in range(3)]
    lowres_data = zoom(padded_data, zoom_factors, order=1)
    lowres_affine = local_affine.copy()
    for i in range(3):
        lowres_affine[i, i] *= padded_data.shape[i] / LOWRES_FINAL_SHAPE[i]

    # --- High-resolution working volume (this is what gets cropped/saved) ---
    print(f"Resampling local frame directly to {highres_spacing} mm (no downsampling)...")
    highres_img = resample_sitk_volume(local_path, highres_spacing)
    highres_data = sitk.GetArrayFromImage(highres_img).transpose(2, 1, 0)
    highres_affine = local_affine.copy()
    orig_spacing = np.array([np.abs(local_affine[i, i]) for i in range(3)])
    signs = np.sign(np.diag(local_affine[:3, :3]))
    for i in range(3):
        highres_affine[i, i] = signs[i] * highres_spacing[i]

    return local_data, local_affine, lowres_data, lowres_affine, highres_data, highres_affine


# ============================================================================
# Stage 3 (from P2): landmark detection on the low-res volume
# ============================================================================

def detect_landmarks(lowres_data, lowres_affine, model, device):
    """Run the trained U-Net on the disposable low-res volume, returning world-mm landmark
    positions (with the same (x,y,z)->(z,-y,-x) coordinate correction p2 applies)."""
    img_tensor = torch.from_numpy(lowres_data).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    landmark_locations_world = {}
    landmark_voxel_pred = {}
    for i, lm_id in enumerate(LANDMARK_IDS):
        pred_heatmap = probs[i]
        threshold = 0.5
        mask = pred_heatmap >= threshold
        if np.any(mask):
            coords = np.argwhere(mask)
            weights = pred_heatmap[mask]
            centroid = np.average(coords, axis=0, weights=weights)
            voxel_idx = tuple(np.round(centroid).astype(int))
        else:
            voxel_idx = np.unravel_index(np.argmax(pred_heatmap), pred_heatmap.shape)
        landmark_voxel_pred[lm_id] = voxel_idx

        voxel_h = np.append(np.array(voxel_idx)[::-1], 1.0)
        world_coords = lowres_affine @ voxel_h
        x, y, z = world_coords[0], world_coords[1], world_coords[2]
        landmark_locations_world[lm_id] = np.array([z, -y, -x])

    return landmark_locations_world, landmark_voxel_pred


# ============================================================================
# Stage 4 (from P2): Frankfort-plane alignment, incl. full no_eyes/QA fallback
# ============================================================================

def compute_plane_normal(points):
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    normal = np.cross(v1, v2)
    return normal / np.linalg.norm(normal)


def fit_plane_svd(points):
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, Vt = np.linalg.svd(centered)
    normal = Vt[-1]
    normal /= np.linalg.norm(normal)
    return normal, centroid


def align_to_horizontal(scan_points, landmark_positions):
    """Standard-mode Frankfort alignment using landmarks 10-13 (indices 2:6 of the 6-pt array)."""
    orig_normal = compute_plane_normal(landmark_positions[:3])
    target_normal = np.array([0, 0, -1])
    center = landmark_positions.mean(axis=0)
    scan_points_centered = scan_points - center

    rot_to_horizontal, _ = R.align_vectors([target_normal], [orig_normal])
    rotated_landmarks = rot_to_horizontal.apply(landmark_positions - center)

    left_right_vector = rotated_landmarks[3] - rotated_landmarks[2]
    lr_distance = np.linalg.norm(left_right_vector)
    left_right_vector /= lr_distance
    left_right_xy = left_right_vector.copy()
    left_right_xy[2] = 0
    lr_xy_norm = np.linalg.norm(left_right_xy)

    if lr_xy_norm < 0.1:
        final_rotation = rot_to_horizontal
        z_rotation_degrees = 0.0
    else:
        left_right_xy /= lr_xy_norm
        target_lr = np.array([1, 0, 0])
        dot_product = np.dot(left_right_xy, target_lr)
        if dot_product < 0:
            target_lr = -target_lr
            dot_product = -dot_product
        angle_deg = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
        if angle_deg < 1.0:
            final_rotation = rot_to_horizontal
            z_rotation_degrees = 0.0
        else:
            cross = np.cross(left_right_xy, target_lr)
            angle_rad = np.arccos(dot_product) if cross[2] > 0 else -np.arccos(dot_product)
            rot_z = R.from_euler('z', angle_rad)
            final_rotation = rot_z * rot_to_horizontal
            z_rotation_degrees = np.degrees(angle_rad)

    aligned_scan = final_rotation.apply(scan_points_centered) + center
    return aligned_scan, final_rotation, orig_normal, center, abs(z_rotation_degrees)


def align_to_horizontal_multipoint(scan_points, plane_points, right_points, left_points):
    """No-eyes-mode Frankfort alignment using SVD plane fit + muscle-based LR direction."""
    orig_normal, center = fit_plane_svd(plane_points)
    target_normal = np.array([0, 0, -1])
    if np.dot(orig_normal, target_normal) < 0:
        orig_normal = -orig_normal

    scan_points_centered = scan_points - center
    rot_to_horizontal, _ = R.align_vectors([target_normal], [orig_normal])

    right_rotated = rot_to_horizontal.apply(right_points - center)
    left_rotated = rot_to_horizontal.apply(left_points - center)
    right_centroid = right_rotated.mean(axis=0)
    left_centroid = left_rotated.mean(axis=0)

    left_right_vector = right_centroid - left_centroid
    lr_distance = np.linalg.norm(left_right_vector)
    left_right_vector /= lr_distance
    left_right_xy = left_right_vector.copy()
    left_right_xy[2] = 0
    lr_xy_norm = np.linalg.norm(left_right_xy)

    if lr_xy_norm < 0.1:
        final_rotation = rot_to_horizontal
        z_rotation_degrees = 0.0
    else:
        left_right_xy /= lr_xy_norm
        target_lr = np.array([1, 0, 0])
        dot_product = np.dot(left_right_xy, target_lr)
        if dot_product < 0:
            target_lr = -target_lr
            dot_product = -dot_product
        angle_deg = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
        if angle_deg < 1.0:
            final_rotation = rot_to_horizontal
            z_rotation_degrees = 0.0
        else:
            cross = np.cross(left_right_xy, target_lr)
            angle_rad = np.arccos(dot_product) if cross[2] > 0 else -np.arccos(dot_product)
            rot_z = R.from_euler('z', angle_rad)
            final_rotation = rot_z * rot_to_horizontal
            z_rotation_degrees = np.degrees(angle_rad)

    aligned_scan = final_rotation.apply(scan_points_centered) + center
    return aligned_scan, final_rotation, orig_normal, center, abs(z_rotation_degrees)


def check_lm10_lm11_symmetry(lm10, lm11, min_dist=LM10_LM11_MIN_DIST_MM, max_dist=LM10_LM11_MAX_DIST_MM):
    dist = float(np.linalg.norm(lm10 - lm11))
    info = {'distance_mm': dist, 'min_dist_mm': min_dist, 'max_dist_mm': max_dist}
    bad = []
    if dist < min_dist:
        bad.append(f"|LM10 - LM11| = {dist:.1f} mm < {min_dist:.1f} mm (too close)")
    elif dist > max_dist:
        bad.append(f"|LM10 - LM11| = {dist:.1f} mm > {max_dist:.1f} mm (too far)")
    return bad, info


def check_muscle_horizontal_containment(lm10, lm11, muscle_points, muscle_labels, tolerance_mm=10.0):
    x_min, x_max = sorted([lm10[0], lm11[0]])
    x_min -= tolerance_mm
    x_max += tolerance_mm
    bad = []
    info = {'x_min': float(x_min), 'x_max': float(x_max)}
    for lbl, pt in zip(muscle_labels, muscle_points):
        if not (x_min <= pt[0] <= x_max):
            bad.append(lbl)
    return bad, info


def detect_and_remove_outliers(points, point_labels, landmark_labels=LANDMARK_LABELS_FOR_OUTLIER, max_rounds=2):
    current_points = points.copy()
    current_labels = list(point_labels)
    removed_labels = []
    all_round_info = []
    landmark_label_set = set(landmark_labels)

    for _round in range(max_rounds):
        n = len(current_points)
        if n < 4:
            break
        centroid = current_points.mean(axis=0)
        distances = np.linalg.norm(current_points - centroid, axis=1)
        median_distance = float(np.median(distances))
        round_info = {
            'distances': {lbl: float(d) for lbl, d in zip(current_labels, distances)},
            'centroid': [float(c) for c in centroid],
            'median_distance': median_distance,
            'removed': None,
        }
        if median_distance < 1e-6:
            all_round_info.append(round_info)
            break

        excess = np.zeros(n)
        for i, lbl in enumerate(current_labels):
            ratio = OUTLIER_CENTROID_RATIO_LANDMARK if lbl in landmark_label_set else OUTLIER_CENTROID_RATIO_MUSCLE
            excess[i] = distances[i] / (median_distance * ratio)

        if excess.max() <= 1.0:
            all_round_info.append(round_info)
            break

        outlier_idx = int(np.argmax(excess))
        outlier_label = current_labels[outlier_idx]
        round_info['removed'] = outlier_label
        all_round_info.append(round_info)
        removed_labels.append(outlier_label)

        keep_mask = np.ones(n, dtype=bool)
        keep_mask[outlier_idx] = False
        current_points = current_points[keep_mask]
        current_labels = [lbl for i, lbl in enumerate(current_labels) if i != outlier_idx]

        if outlier_label in landmark_label_set:
            break

    return current_points, current_labels, removed_labels, all_round_info


def check_landmarks_further_than_muscles(points, labels, landmark_labels=LANDMARK_LABELS_FOR_OUTLIER):
    if len(points) < 2:
        return [], {'note': 'too few points to evaluate'}
    centroid = points.mean(axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    landmark_label_set = set(landmark_labels)
    muscle_distances = [d for lbl, d in zip(labels, distances) if lbl not in landmark_label_set]
    info = {'centroid': [float(c) for c in centroid], 'distances': {lbl: float(d) for lbl, d in zip(labels, distances)}}
    if not muscle_distances:
        info['note'] = 'no muscle points present after filtering'
        return [], info
    max_muscle_distance = float(max(muscle_distances))
    info['max_muscle_distance'] = max_muscle_distance
    bad = [lbl for lbl, d in zip(labels, distances) if lbl in landmark_label_set and d <= max_muscle_distance]
    return bad, info


def extract_muscle_top_points(segmentation_data, affine):
    unique_labels = np.unique(segmentation_data)
    muscle_top_points = {}
    for muscle_name, label_id in HEAD_MUSCLES_LABEL_IDS.items():
        if label_id not in unique_labels:
            muscle_top_points[muscle_name] = None
            continue
        voxel_coords = np.argwhere(segmentation_data == label_id)
        if voxel_coords.size == 0:
            muscle_top_points[muscle_name] = None
            continue
        voxel_h = np.hstack([voxel_coords[:, [2, 1, 0]], np.ones((len(voxel_coords), 1))])
        world_coords = (affine @ voxel_h.T).T[:, :3]
        world_corrected = np.column_stack([world_coords[:, 2], -world_coords[:, 1], -world_coords[:, 0]])
        top_idx = np.argmax(world_corrected[:, 2])
        muscle_top_points[muscle_name] = world_corrected[top_idx]
    return (muscle_top_points['masseter_right'], muscle_top_points['masseter_left'],
            muscle_top_points['lateral_pterygoid_right'], muscle_top_points['lateral_pterygoid_left'])


def render_landmarks_visualization(scan_data, points, point_labels, output_path,
                                    removed_labels=None, flagged_labels=None, status_text=None, threshold=-300):
    removed_set = set(removed_labels or [])
    flagged_set = set(flagged_labels or [])
    label_display = {
        'lm10': ('LM10', 'yellow'), 'lm11': ('LM11', 'orange'),
        'masseter_right': ('Masseter R', 'cyan'), 'masseter_left': ('Masseter L', 'magenta'),
        'lateral_pterygoid_right': ('Lat.Pteryg. R', 'lime'), 'lateral_pterygoid_left': ('Lat.Pteryg. L', 'red'),
    }
    views = [('Axial (Top-Down)', 'xy', 0), ('Sagittal (Side)', 'yz', 90), ('Coronal (Front)', 'xz', 0)]
    plotter = pv.Plotter(shape=(1, 3), off_screen=True, window_size=[2880, 1080])
    mask_points_vis = np.argwhere(scan_data > threshold).astype(np.float32)

    for idx, (title, view, azimuth) in enumerate(views):
        plotter.subplot(0, idx)
        if mask_points_vis.size:
            plotter.add_mesh(pv.PolyData(mask_points_vis), color='lightgray', opacity=0.2, point_size=2)
        for i, landmark in enumerate(points):
            lbl = point_labels[i]
            name, color = label_display.get(lbl, (lbl, 'gray'))
            is_outlier, is_flagged = lbl in removed_set, lbl in flagged_set
            display_color = 'gray' if (is_outlier or is_flagged) else color
            plotter.add_mesh(pv.Sphere(radius=4, center=landmark), color=display_color, label=name if idx == 2 else None)
            tag = ' (OUTLIER)' if is_outlier else (' (FLAGGED)' if is_flagged else '')
            plotter.add_point_labels([landmark], [f"{name}{tag}"], font_size=16, text_color=display_color,
                                      point_size=1, shape_opacity=0, bold=True)
        if idx == 2:
            plotter.add_legend(bcolor='white', face='rectangle', size=(0.25, 0.35))
        plotter.camera_position = view
        if azimuth != 0:
            plotter.camera.azimuth = azimuth
        plotter.camera.zoom(1.3)
        plotter.add_text(title, position='upper_edge', font_size=14, color='black')
        if status_text and idx == 0:
            plotter.add_text(status_text, position='lower_edge', font_size=12, color='red')
    plotter.screenshot(output_path)
    plotter.close()


def compute_alignment(landmark_locations_world, lowres_data, device, no_eyes, patient_output_dir,
                       patient_id, viz_dir):
    """
    Full port of P2's alignment logic (standard Frankfort-plane mode, or the
    no_eyes SVD multipoint mode with muscle segmentation + outlier/QA checks).
    Returns (rotation, center_mm, diagnostics_dict).
    """
    all_landmarks = np.array([landmark_locations_world[lm_id] for lm_id in LANDMARK_IDS])
    mask_points = np.argwhere(lowres_data > 0).astype(np.float32)
    diagnostics = {'no_eyes': no_eyes}

    if not no_eyes:
        landmarks_for_plane = all_landmarks[2:6]  # landmarks 10-13
        _, rotation, orig_normal, center, z_rot = align_to_horizontal(mask_points, landmarks_for_plane)
        diagnostics['method'] = 'standard (landmarks 10-13)'
    else:
        print("  No-eyes mode: segmenting head muscles for Frankfort plane...")
        muscles_path = os.path.join(patient_output_dir, f"{patient_id}_head_muscles_segmented.nii.gz")
        seg_img, muscles_path = run_totalsegmentator(
            os.path.join(patient_output_dir, "_tmp", f"{patient_id}_local_frame_oriented.nii.gz"),
            'head_muscles', device, muscles_path,
        )
        seg_data = seg_img.get_fdata()
        masseter_r, masseter_l, ptg_r, ptg_l = extract_muscle_top_points(seg_data, seg_img.affine)

        candidates = [('masseter_right', masseter_r), ('masseter_left', masseter_l),
                      ('lateral_pterygoid_right', ptg_r), ('lateral_pterygoid_left', ptg_l)]
        missing = [n for n, p in candidates if p is None]
        present = [(n, p) for n, p in candidates if p is not None]
        right_present = any(n.endswith('_right') for n, _ in present)
        left_present = any(n.endswith('_left') for n, _ in present)
        if len(present) < 2 or not (right_present and left_present):
            raise _LandmarkOutlierError(
                f"Insufficient muscle segmentation for Frankfort plane (missing: {missing}). Scan: {patient_id}."
            )

        point_labels_6pt = ['lm10', 'lm11'] + [n for n, _ in present]
        points_6pt = np.array([landmark_locations_world[10], landmark_locations_world[11]] + [p for _, p in present])

        failure_reasons = []
        flagged_labels = set()

        bad_symmetry, symmetry_info = check_lm10_lm11_symmetry(landmark_locations_world[10], landmark_locations_world[11])
        if bad_symmetry:
            flagged_labels.update(['lm10', 'lm11'])
            failure_reasons.append(f"LM10/LM11 distance check failed: {'; '.join(bad_symmetry)}.")

        muscle_lbls = [n for n, _ in present]
        muscle_pts = np.array([p for _, p in present])
        bad_containment, containment_info = check_muscle_horizontal_containment(
            landmark_locations_world[10], landmark_locations_world[11], muscle_pts, muscle_lbls)
        if bad_containment:
            flagged_labels.update(bad_containment)
            failure_reasons.append(f"Muscle(s) {bad_containment} failed horizontal containment.")

        filtered_points, filtered_labels, removed_labels, _ = (points_6pt, point_labels_6pt, [], [])
        if not failure_reasons:
            filtered_points, filtered_labels, removed_labels, _ = detect_and_remove_outliers(points_6pt, point_labels_6pt)
            removed_landmarks = [lbl for lbl in removed_labels if lbl in ('lm10', 'lm11')]
            if removed_landmarks:
                flagged_labels.update(removed_landmarks)
                failure_reasons.append(f"Landmark(s) {removed_landmarks} detected as centroid outlier(s).")

        lateral_info = {}
        if not failure_reasons:
            bad_lateral, lateral_info = check_landmarks_further_than_muscles(filtered_points, filtered_labels)
            if bad_lateral:
                flagged_labels.update(bad_lateral)
                failure_reasons.append(f"Landmark(s) {bad_lateral} not further from centroid than muscles.")

        try:
            os.makedirs(viz_dir, exist_ok=True)
            vis_path = os.path.join(viz_dir, f"{patient_id}_frankfort_plane_landmarks.png")
            render_landmarks_visualization(
                lowres_data, points_6pt, point_labels_6pt, vis_path,
                removed_labels=removed_labels, flagged_labels=flagged_labels,
                status_text=("SCAN FLAGGED: " + " | ".join(failure_reasons)) if failure_reasons else None,
            )
        except Exception as e:
            print(f"  WARNING: Failed to render Frankfort visualization: {e}")

        if failure_reasons:
            raise _LandmarkOutlierError(f"{' '.join(failure_reasons)} Scan: '{patient_id}'.")

        right_labels = {'masseter_right', 'lateral_pterygoid_right'}
        left_labels = {'masseter_left', 'lateral_pterygoid_left'}
        right_muscle_points = np.array([p for lbl, p in zip(filtered_labels, filtered_points) if lbl in right_labels])
        left_muscle_points = np.array([p for lbl, p in zip(filtered_labels, filtered_points) if lbl in left_labels])
        if len(right_muscle_points) == 0 or len(left_muscle_points) == 0:
            right_muscle_points = np.array([landmark_locations_world[10]])
            left_muscle_points = np.array([landmark_locations_world[11]])

        _, rotation, orig_normal, center, z_rot = align_to_horizontal_multipoint(
            mask_points, filtered_points, right_muscle_points, left_muscle_points)
        diagnostics['method'] = 'no_eyes (SVD multipoint + muscles)'
        diagnostics['removed_outliers'] = removed_labels

    # Fix reflection if present
    rotation_matrix = rotation.as_matrix()
    if np.linalg.det(rotation_matrix) < 0:
        rotation_matrix[:, 2] *= -1
        rotation = R.from_matrix(rotation_matrix)

    diagnostics['z_rotation_angle_degrees'] = float(z_rot)
    diagnostics['flagged_large_rotation'] = bool(abs(z_rot) > ROTATION_FLAG_THRESHOLD_DEG)
    if diagnostics['flagged_large_rotation']:
        print(f"  ⚠️  Large rotation detected ({abs(z_rot):.2f}°)!")

    try:
        os.makedirs(viz_dir, exist_ok=True)
        vis_path = os.path.join(viz_dir, f"{patient_id}_landmarks_visualization.png")
        landmark_names = ['L8', 'L9', 'L10', 'L11', 'L12', 'L13']
        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
        mask_points_vis = np.argwhere(lowres_data > -300).astype(np.float32)
        plotter = pv.Plotter(shape=(1, 3), off_screen=True, window_size=[2880, 1080])
        for idx, (title, view, azimuth) in enumerate(
            [('Axial (Top-Down)', 'xy', 0), ('Sagittal (Side)', 'yz', 90), ('Coronal (Front)', 'xz', 0)]
        ):
            plotter.subplot(0, idx)
            if mask_points_vis.size:
                plotter.add_mesh(pv.PolyData(mask_points_vis), color='lightgray', opacity=0.3, point_size=2)
            for lm, name, color in zip(all_landmarks, landmark_names, colors):
                plotter.add_mesh(pv.Sphere(radius=3, center=lm), color=color, label=name if idx == 2 else None)
                plotter.add_point_labels([lm], [name], font_size=16, text_color=color, point_size=1, shape_opacity=0)
            if idx == 2:
                plotter.add_legend(bcolor='white', face='rectangle', size=(0.2, 0.2))
            plotter.camera_position = view
            if azimuth != 0:
                plotter.camera.azimuth = azimuth
            plotter.camera.zoom(1.3)
            plotter.add_text(title, position='upper_edge', font_size=14, color='black')
        plotter.screenshot(vis_path)
        plotter.close()
    except Exception as e:
        print(f"  WARNING: Failed to render landmark visualization: {e}")

    return rotation, center, diagnostics


def rotate_volume_physical(data, spacing, rotation, center_mm):
    """
    Rotate a volume about `center_mm` (world mm) by `rotation`, correctly
    accounting for `spacing` (mm/voxel). This is the spacing-aware version of
    P2's affine_transform call — P2 fed centers/offsets expressed in mm
    directly into affine_transform's voxel-space `offset`, which only works
    because its low-res grid happens to be ~1.05 mm/voxel. Here spacing can
    be anything (e.g. 0.2 mm), so the mm-based offset is explicitly divided
    by spacing to convert it into voxel units.
    """
    rotation_matrix = rotation.as_matrix()
    inverse_rotation = np.linalg.inv(rotation_matrix)
    spacing = np.asarray(spacing, dtype=float)
    center_voxel = center_mm / spacing
    offset_voxel = center_voxel - inverse_rotation @ center_voxel
    return affine_transform(data, inverse_rotation, offset=offset_voxel, order=1, cval=-1000)


# ============================================================================
# Stage 5 (from P3, adapted): crop a 256^3 FOV centered on each ear landmark
# ============================================================================

def crop_ear_fov(aligned_data, aligned_affine, landmark_world_mm, roi_size_voxels=EAR_FOV_VOXELS,
                  offset_mm=EAR_OFFSET_MM):
    center_world = landmark_world_mm + offset_mm
    center_voxel = np.round(world_to_voxel(center_world, aligned_affine)).astype(int)

    roi_size = np.array(roi_size_voxels)
    half_size = roi_size // 2
    roi_start = center_voxel - half_size
    roi_end = roi_start + roi_size

    roi = np.full(tuple(roi_size), -1000, dtype=aligned_data.dtype)
    data_shape = np.array(aligned_data.shape)
    data_start = np.maximum(roi_start, 0)
    data_end = np.minimum(roi_end, data_shape)
    if np.any(data_end <= data_start):
        raise ValueError(f"Ear ROI does not overlap with scan data (start={roi_start}, end={roi_end}).")

    roi_start_in_roi = data_start - roi_start
    roi_end_in_roi = roi_start_in_roi + (data_end - data_start)
    data_slices = tuple(slice(int(data_start[i]), int(data_end[i])) for i in range(3))
    roi_slices = tuple(slice(int(roi_start_in_roi[i]), int(roi_end_in_roi[i])) for i in range(3))
    roi[roi_slices] = aligned_data[data_slices]

    new_origin = voxel_to_world(roi_start.astype(float), aligned_affine)
    cropped_affine = aligned_affine.copy()
    cropped_affine[:3, 3] = new_origin
    return roi, cropped_affine, roi_start, center_world


# ============================================================================
# Stage 6 (from P4): final HU normalization
# ============================================================================

def normalize_scan(image_data, min_hu=MIN_HU, max_hu=MAX_HU):
    clipped = np.clip(image_data, min_hu, max_hu)
    return ((clipped - min_hu) / (max_hu - min_hu)).astype(np.float32)


# ============================================================================
# Orchestration: process a single raw CT scan end to end
# ============================================================================

def process_scan(scan_path, output_dir, landmark_model_path, model, device,
                  no_eyes=False, skip_alignment=False, highres_spacing=HIGHRES_SPACING,
                  ear_fov_voxels=EAR_FOV_VOXELS, ear_offset_mm=EAR_OFFSET_MM):
    patient_id = extract_patient_id_from_raw_filename(os.path.basename(scan_path))
    patient_output_dir = os.path.join(output_dir, patient_id)
    viz_dir = os.path.join(patient_output_dir, "visualizations")
    os.makedirs(patient_output_dir, exist_ok=True)
    print(f"\n{'='*70}\nProcessing: {patient_id}\n{'='*70}")

    transform_log = {
        "patient_id": patient_id,
        "original_scan_path": scan_path,
        "processing_timestamp": datetime.now().isoformat(),
        "config": {
            "highres_spacing_mm": list(highres_spacing),
            "ear_fov_voxels": list(ear_fov_voxels),
            "ear_offset_mm": [float(x) for x in ear_offset_mm],
            "no_eyes": no_eyes,
            "skip_alignment": skip_alignment,
        },
        "steps": [],
    }

    # --- Stage 1: load, clip, segment head, crop native-res ROI ---
    img = nib.load(scan_path)
    data = img.get_fdata()
    affine = img.affine
    voxel_size = img.header.get_zooms()

    if any(voxel_size[i] > VOXEL_THRESHOLD[i] for i in range(3)):
        raise RuntimeError(f"Scan {patient_id} voxel size {voxel_size} coarser than threshold {VOXEL_THRESHOLD}.")

    print("Clipping intensity...")
    clipped_data = clip_intensity(data)

    cropped_data, cropped_affine, midpoint_mm, centroid_info = build_head_crop(
        clipped_data, affine, voxel_size, device, patient_output_dir, patient_id)
    transform_log["steps"].append({
        "step": 1, "operation": "head_localization_and_crop",
        "midpoint_mm": [float(x) for x in midpoint_mm], "centroids": centroid_info,
        "cropped_dimensions": [int(x) for x in cropped_data.shape],
    })

    # --- Stage 2: shared local frame -> disposable low-res + real high-res volumes ---
    local_data, local_affine, lowres_data, lowres_affine, highres_data, highres_affine = build_reference_frames(
        cropped_data, cropped_affine, patient_output_dir, patient_id, highres_spacing)
    transform_log["steps"].append({
        "step": 2, "operation": "build_reference_frames",
        "lowres_shape": list(lowres_data.shape), "lowres_spacing_mm": LOWRES_SPACING,
        "highres_shape": list(highres_data.shape), "highres_spacing_mm": list(highres_spacing),
    })

    # --- Stage 3: landmark detection on the disposable low-res volume ---
    print("Running landmark detection...")
    landmark_locations_world, landmark_voxel_pred = detect_landmarks(lowres_data, lowres_affine, model, device)
    transform_log["steps"].append({
        "step": 3, "operation": "landmark_detection",
        "landmarks_world_mm": {str(k): [float(x) for x in v] for k, v in landmark_locations_world.items()},
    })

    right_ear_landmark = landmark_locations_world[10]
    left_ear_landmark = landmark_locations_world[11]

    if skip_alignment:
        print("skip_alignment=True — using landmarks as-is (no Frankfort-plane rotation).")
        rotation = R.identity()
        center = np.zeros(3)
        aligned_landmarks = {lm_id: landmark_locations_world[lm_id] for lm_id in LANDMARK_IDS}
        highres_aligned = highres_data
        transform_log["steps"].append({"step": 4, "operation": "frankfort_plane_alignment", "status": "SKIPPED"})
    else:
        print("Computing Frankfort-plane alignment...")
        rotation, center, diagnostics = compute_alignment(
            landmark_locations_world, lowres_data, device, no_eyes, patient_output_dir, patient_id, viz_dir)

        all_landmarks_arr = np.array([landmark_locations_world[lm_id] for lm_id in LANDMARK_IDS])
        aligned_arr = rotation.apply(all_landmarks_arr - center) + center
        aligned_landmarks = {lm_id: aligned_arr[i] for i, lm_id in enumerate(LANDMARK_IDS)}

        print("Rotating high-resolution volume to align Frankfort plane...")
        highres_aligned = rotate_volume_physical(highres_data, highres_spacing, rotation, center)

        aligned_path = os.path.join(patient_output_dir, f"{patient_id}_highres_aligned.nii.gz")
        nib.save(nib.Nifti1Image(highres_aligned.astype(np.float32), highres_affine), aligned_path)

        transform_log["steps"].append({
            "step": 4, "operation": "frankfort_plane_alignment",
            "diagnostics": diagnostics,
            "rotation_center_mm": [float(x) for x in center],
            "rotation_matrix": [[float(x) for x in row] for row in rotation.as_matrix()],
            "aligned_landmarks_world_mm": {str(k): [float(x) for x in v] for k, v in aligned_landmarks.items()},
            "output_file": aligned_path,
        })

    right_ear_aligned = aligned_landmarks[10]
    left_ear_aligned = aligned_landmarks[11]

    # --- Stage 5: crop the ear FOVs, mirror left ear ---
    output_paths = {}
    for lm_id, ear_name, landmark_pos, is_left in [
        (10, 'right_ear', right_ear_aligned, False),
        (11, 'left_ear', left_ear_aligned, True),
    ]:
        print(f"Cropping {ear_name} FOV ({ear_fov_voxels} voxels @ {highres_spacing} mm)...")
        applied_offset = np.array(ear_offset_mm, dtype=float).copy()
        if is_left:
            applied_offset[0] = -applied_offset[0]

        cropped, cropped_affine_ear, roi_start, center_world = crop_ear_fov(
            highres_aligned, highres_affine, landmark_pos, ear_fov_voxels, applied_offset)

        if is_left:
            cropped = np.flip(cropped, axis=MIRROR_AXIS)

        cropped_affine_ear = cropped_affine_ear.copy()
        cropped_affine_ear[:3, 3] = [0.0, 0.0, 0.0]  # origin reset for the saved crop

        raw_path = os.path.join(patient_output_dir, f"{patient_id}_{ear_name}_raw_hu.nii.gz")
        nib.save(nib.Nifti1Image(np.ascontiguousarray(cropped).astype(np.float32), cropped_affine_ear), raw_path)

        normalized = normalize_scan(cropped)
        final_path = os.path.join(patient_output_dir, f"{patient_id}_{ear_name}_0000.nii.gz")
        nib.save(nib.Nifti1Image(np.ascontiguousarray(normalized), cropped_affine_ear), final_path)
        output_paths[ear_name] = final_path

        transform_log["steps"].append({
            "step": 5, "operation": f"crop_and_normalize_{ear_name}",
            "landmark_id": lm_id, "roi_center_world_mm": [float(x) for x in center_world],
            "roi_size_voxels": list(ear_fov_voxels), "mirrored": bool(is_left),
            "raw_hu_output_file": raw_path, "normalized_output_file": final_path,
        })
        print(f"  ✓ Saved {ear_name}: {final_path}")

    save_json(transform_log, os.path.join(patient_output_dir, f"{patient_id}_transform_log.json"))
    print(f"✓ Completed {patient_id}")
    return output_paths


# ============================================================================
# CLI
# ============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Full high-resolution ear-extraction pipeline (P1+P2+P3+P4 in one script)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--input_ct', type=str, default=None,
                        help='Path to a single raw CT scan (<patient>__CT.nii[.gz])')
    parser.add_argument('--raw_scans_dir', type=str, default=None,
                        help='Directory containing multiple raw CT scans to process in batch')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save all outputs')
    parser.add_argument('--landmark_model', type=str,
                        default=os.path.join(os.path.dirname(parent_dir), "best_model_2025-12-05_13-16-35.pth"),
                        help='Path to the trained landmark-detection model checkpoint')
    parser.add_argument('--no_eyes', type=str, default='False', choices=['True', 'False'],
                        help='Use mandible/muscle top points instead of eye landmarks for the Frankfort plane')
    parser.add_argument('--skip_alignment', type=str, default='False', choices=['True', 'False'],
                        help='Skip Frankfort-plane alignment (landmark detection only)')
    parser.add_argument('--highres_spacing', type=float, default=0.2,
                        help='Final isotropic voxel spacing in mm (default: 0.2)')
    parser.add_argument('--ear_fov_voxels', type=int, default=256,
                        help='Final ear crop size in voxels per axis (default: 256)')
    parser.add_argument('--ear_offset_mm', type=float, nargs=3, default=[-10.0, 0.0, 0.0],
                        help='Offset in mm [x y z] from the eardrum landmark for the crop center '
                             '(default shifts +X to include more of the cochlea; mirrored for left ear)')
    parser.add_argument('--overwrite', type=str, default='False', choices=['True', 'False'],
                        help='Reprocess patients even if their final ear crops already exist (default: False, '
                             'i.e. already-processed patients are skipped)')
    return parser.parse_args()


def main():
    args = parse_arguments()
    if not args.input_ct and not args.raw_scans_dir:
        raise ValueError("Provide either --input_ct or --raw_scans_dir.")

    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device()

    print("Loading landmark detection model...")
    model = UNet3D(in_channels=1, out_channels=NUM_LANDMARKS, base_features=16)
    checkpoint = torch.load(args.landmark_model, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    highres_spacing = [args.highres_spacing] * 3
    ear_fov_voxels = [args.ear_fov_voxels] * 3
    ear_offset_mm = np.array(args.ear_offset_mm)
    no_eyes = (args.no_eyes == 'True')
    skip_alignment = (args.skip_alignment == 'True')

    scan_paths = []
    if args.input_ct:
        scan_paths.append(args.input_ct)
    if args.raw_scans_dir:
        for root, _, files in os.walk(args.raw_scans_dir):
            for file in files:
                if is_raw_ct_scan(file):
                    scan_paths.append(os.path.join(root, file))
    scan_paths = sorted(set(scan_paths))
    print(f"Found {len(scan_paths)} scan(s) total.")

    overwrite = (args.overwrite == 'True')
    if not overwrite:
        remaining = []
        for scan_path in scan_paths:
            patient_id = extract_patient_id_from_raw_filename(os.path.basename(scan_path))
            if is_already_processed(patient_id, args.output_dir):
                print(f"Skipping {patient_id} (already processed).")
            else:
                remaining.append(scan_path)
        scan_paths = remaining
    print(f"Processing {len(scan_paths)} scan(s) (overwrite={overwrite}).")

    for scan_path in scan_paths:
        try:
            process_scan(
                scan_path, args.output_dir, args.landmark_model, model, device,
                no_eyes=no_eyes, skip_alignment=skip_alignment,
                highres_spacing=highres_spacing, ear_fov_voxels=ear_fov_voxels,
                ear_offset_mm=ear_offset_mm,
            )
        except Exception as e:
            print(f"FAILED: {scan_path}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
