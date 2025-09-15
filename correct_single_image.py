#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import random
import copy
import torch
import monai
import torchio as tio
import matplotlib.pyplot as plt
from Networks_3D import UNet
from Networks_R3U import R3U_Net
from interp_module_3D import *
from phase_shadow_module import *
from duplication_module_3D import * 
from correct_nifti_artifact_image import *
from preprocess_functions import *
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio
from monai.transforms import RandSpatialCrop
import monai.inferers
import SimpleITK as sitk

# Allowed phases to avoid typos at the CLI
ALLOWED_PHASES = [
    "20EX", "0EX",
    "20IN", "40IN", "60IN", "80IN", "100IN",
    "80EX", "60EX", "40EX"
]

def main():
    parser = argparse.ArgumentParser(
        description="Run correct_NIFTI_image_clean with a case/ID and three phases."
    )

    # Positional args: ID + three phases in the order your function expects
    parser.add_argument("case_id",
                        help="Case/scan ID, e.g., IPF102_SCAN3")
    parser.add_argument("refphase1", choices=ALLOWED_PHASES,
                        help="Reference Phase 1 (e.g., 20EX)")
    parser.add_argument("artifactphase", choices=ALLOWED_PHASES,
                        help="Artifact Phase (e.g., 40EX)")
    parser.add_argument("refphase2", choices=ALLOWED_PHASES,
                        help="Reference Phase 2 (e.g., 60EX)")

    # Optional overrides with sensible defaults from your snippet
    parser.add_argument("--model-dir",
                        default="/Users/carrizales/Pix2Pix/model_checkpoints/3D_Interp_Best_Loss_AttU_Combined_Generator",
                        help="Path to the trained model directory.")
    parser.add_argument("--input-root",
                        default="/Dedicated/OHSU_RT/interpolation_corrected_nifti_images/GDR_Images_for_Correction_Model",
                        help="Root directory containing per-case input folders.")
    parser.add_argument("--output-root",
                        default="/Dedicated/OHSU_RT/interpolation_corrected_nifti_images/GDR_Images_for_Correction_Model/Results",
                        help="Destination directory for outputs.")

    # Flags to toggle the two boolean options (default False to mirror your code)
    parser.add_argument("--add-artificial-interpolation",
                        action="store_true", default=False,
                        help="If set, enables add_artificial_interpolation=True.")
    parser.add_argument("--add-artificial-phase-shadow",
                        action="store_true", default=False,
                        help="If set, enables add_artificial_phase_shadow=True.")

    args = parser.parse_args()

    # Derive input/output dirs based on the case ID and roots
    input_dir = os.path.join(args.input_root.rstrip("/"), args.case_id) + "/"
    output_dir = args.output_root.rstrip("/") + "/"

    # Optional: sanity pings (comment out if noisy)
    # print(f"Model dir: {args.model_dir}")
    # print(f"Input dir: {input_dir}")
    # print(f"Output dir: {output_dir}")
    # print(f"Phases: {args.refphase1}, {args.artifactphase}, {args.refphase2}")
    # print(f"Flags -> add_artificial_interpolation={args.add_artificial_interpolation}, "
    #       f"add_artificial_phase_shadow={args.add_artificial_phase_shadow}")

    # Call your function exactly as before, but with CLI-supplied values
    correct_NIFTI_image_clean(
        args.model_dir,
        input_dir,
        output_dir,
        args.refphase1, args.artifactphase, args.refphase2,
        add_artificial_interpolation=args.add_artificial_interpolation,
        add_artificial_phase_shadow=args.add_artificial_phase_shadow
    )

if __name__ == "__main__":
    main()
