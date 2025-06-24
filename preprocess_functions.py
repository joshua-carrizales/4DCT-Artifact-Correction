import numpy as np
import pandas as pd
import os
import SimpleITK as sitk
import random
import copy
from interp_module_3D import *
from phase_shadow_module import *

def resample_iso(im, spacing, outside_val, interp):
     inputSize = im.GetSize()
     inputSpacing = im.GetSpacing()
     inputOrigin = im.GetOrigin()
     inputDirection = im.GetDirection()

     outputSpacing = [spacing]*im.GetDimension()
     outputSize = np.ceil( (np.array(inputSize) * np.array(inputSpacing)) / outputSpacing)

     outputSizeInt = [int(s) for s in outputSize]
     resampleFilter = sitk.ResampleImageFilter()
     resampleFilter.SetInterpolator(interp)
     resampleFilter.SetOutputDirection(inputDirection)
     resampleFilter.SetOutputOrigin(inputOrigin)
     resampleFilter.SetOutputSpacing(outputSpacing)
     resampleFilter.SetSize(outputSizeInt)
     resampleFilter.SetDefaultPixelValue(outside_val)
     resampleIm = resampleFilter.Execute(im)
     return resampleIm

def resample_ref(im, ref, outside_val, interp):
     resampleFilter = sitk.ResampleImageFilter()
     resampleFilter.SetInterpolator(interp)
     resampleFilter.SetDefaultPixelValue(outside_val)
     resampleFilter.SetReferenceImage(ref)
     resampleIm = resampleFilter.Execute(im)
     return resampleIm

def unpad(x, pad_width):
    slices = []
    for c in pad_width:
        e = None if c[1] == 0 else -c[1]
        slices.append(slice(c[0], e))
    return x[tuple(slices)]

def get_lung_inds(mask):

    img_shape = mask.shape
    inds = np.array(np.where(mask>0))
    min_inds = np.min(inds, axis = 1)
    max_inds = np.max(inds, axis = 1)

    upper_margin = img_shape - max_inds
    print(upper_margin)
    upper_margin[0] = 0
    print(upper_margin)
    return np.array(list(zip(min_inds, upper_margin)))

def crop_img_2_coronal_lung(img,mask):
    ind = np.array(np.where(mask>0))
    new_lung = img[np.min(ind[0]):np.max(ind[0]),np.min(ind[1]):np.max(ind[1]),np.min(ind[2]):np.max(ind[2])]
    return new_lung

def normalise_hu(image, hu_range=[-1024.0,200.0]):
    assert (hu_range[0] < hu_range[1])

    return np.clip(image, hu_range[0], hu_range[1]).astype(np.float32)

def normalise_zero_one(image):
    minimum = -1024.0
    maximum = 200.0
    ret = (image - minimum) / (maximum - minimum)

    # image = image.astype(np.float32)

    # minimum = -1024.0
    # maximum = 200.0

    # if maximum > minimum:
    #     ret = (image - minimum) / (maximum - minimum)
    # else:
    #     ret = image * 0.
    return ret

def normalise_one_one(image):

    ret = normalise_zero_one(image)
    ret *= 2.
    ret -= 1.
    return ret

def resample_img(itk_image, out_spacing=[1.0, 1.0, 1.0], is_label=False):
    assert (out_spacing[0]==out_spacing[1]==out_spacing[2]), "ensure isotropic dimensions"

    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)


def preprocess_4D_scan(images, lungs):
    image_data = []
    lungs_data = []
    for phase in range(len(images)):
        img = sitk.ReadImage(images[phase])
        lung = sitk.ReadImage(lungs[phase])
        
        img_resample = resample_iso(img, 1, -1024, sitk.sitkBSpline)
        lung_resample = resample_ref(lung, img_resample, 0, sitk.sitkNearestNeighbor)

        img_np = sitk.GetArrayFromImage(img_resample)
        img_np = img_np[::-1,:,:]
        lung_np = sitk.GetArrayFromImage(lung_resample)
        lung_np = lung_np[::-1,:,:]

        img_norm = normalise_hu(img_np)
        img_norm = normalise_one_one(img_norm)
        # print(img_norm.shape)

        image_data.append(img_norm)
        lungs_data.append(lung_np)
    all_images = np.stack(image_data)
    all_lungs = np.stack(lungs_data)

    union_lung_mask = np.zeros(all_lungs.shape[1:4])
    for ix in range(len(all_lungs)):
        union_lung_mask = np.logical_or(union_lung_mask, all_lungs[ix])

    ind          = np.array(np.where(union_lung_mask>0))
    min_lung_ind = np.min(ind,axis=1)
    max_lung_ind = np.max(ind,axis=1)

    all_images_crop = []
    all_lungs_crop = []
    for ix in range(len(all_images)):
        img_crop = all_images[ix,min_lung_ind[0]:max_lung_ind[0], min_lung_ind[1]:max_lung_ind[1], min_lung_ind[2]:max_lung_ind[2]]
        lung_crop = all_lungs[ix,min_lung_ind[0]:max_lung_ind[0], min_lung_ind[1]:max_lung_ind[1], min_lung_ind[2]:max_lung_ind[2]]
        all_images_crop.append(img_crop)
        all_lungs_crop.append(lung_crop)
    final_crop_images = np.stack(all_images_crop)
    final_crop_lungs  = np.stack(all_lungs_crop)
    return final_crop_images, final_crop_lungs

def preprocess_4D_scan_for_correction(path_2_4D_scan, ref_phase_1=None, artifact_phase=None, ref_phase_2=None, add_artificial_interpolation=False, add_artificial_phase_shadow=False,
                                      add_artificial_duplication=False):
    image_data = []
    phases = ['0EX','20EX','20IN','40EX','40IN','60EX','60IN','80EX','80IN','100IN']
    if ref_phase_1 == None and ref_phase_2 == None:
        print('No Reference Phases Specified...')
        phases.remove(artifact_phase)
        rand_phase_ind = random.sample(range(9),3)
        random_phase_0 = phases[rand_phase_ind[0]]
        random_phase_1 = phases[rand_phase_ind[1]]
        ref_phase_1 = random_phase_0
        ref_phase_2 = random_phase_1
        if add_artificial_phase_shadow == True:
            PH_phase = phases[rand_phase_ind[2]]

    files_in_4D_scan = os.listdir(path_2_4D_scan)
    wanted_phase_ids = [artifact_phase, ref_phase_1, ref_phase_2]
    new_list = [item for item in phases if item not in wanted_phase_ids]
    PH_phase = random.choice(new_list)
    subject_ID = files_in_4D_scan[0].split('_')
    subject_ID = subject_ID[0] + '_' + subject_ID[1] + '_'
    if add_artificial_phase_shadow == True:
        wanted_phase_names = [subject_ID + wanted_phase_ids[0] + '.nii.gz',subject_ID + wanted_phase_ids[1] + '.nii.gz',subject_ID + wanted_phase_ids[2] + '.nii.gz', subject_ID + PH_phase + '.nii.gz']
    else:
        wanted_phase_names = [subject_ID + wanted_phase_ids[0] + '.nii.gz',subject_ID + wanted_phase_ids[1] + '.nii.gz',subject_ID + wanted_phase_ids[2] + '.nii.gz']
    print('Wanted Phases = ',wanted_phase_names)
    sitk_img = sitk.ReadImage(os.path.join(path_2_4D_scan,wanted_phase_names[0]))
    artifact_img_name = wanted_phase_names[0]
    print('Artifacted Image = ',artifact_img_name)

    for img_name in wanted_phase_names:
        img = sitk.ReadImage(os.path.join(path_2_4D_scan, img_name))
        img_resample = resample_iso(img, 1, -1024, sitk.sitkBSpline)

        img_np = sitk.GetArrayFromImage(img_resample)
        img_np = img_np[::-1,:,:]

        img_norm = normalise_hu(img_np)
        img_norm = normalise_one_one(img_norm)

        image_data.append(img_norm)
        print('Image Pre-processed...')
    original_pp_img = image_data[0]
    if add_artificial_interpolation == True:
        artifact_img, artifact_msk = addInterp3D_big_artifacts(image_data[0])
        image_data[0] = artifact_img
        print('Artificial Interpolation Artifact Added...')
    if add_artificial_phase_shadow == True:
        lung_mask_path = path_2_4D_scan.split('/')
        lung_mask_path[7] = 'IPF_Study_seg_net'
        lung_mask_path[13] = 'seg_net'
        lung_mask_path = '/'.join(lung_mask_path)
        lung_mask_path = os.path.join(lung_mask_path,subject_ID+wanted_phase_ids[0]+'.mask.nii.gz')
        lung_mask = sitk.ReadImage(lung_mask_path)
        lung_mask = resample_iso(lung_mask, 1, -1024, sitk.sitkBSpline)
        lung_mask_np = sitk.GetArrayFromImage(lung_mask)
        lung_mask_np = lung_mask_np[::-1,:,:] > 0

        artifact_img, artifact_msk = create_random_artifact(image_data[0], [image_data[3]], lung_mask_np) # Needs artifact image, additional phase for creating artifact, and lung mask.
        image_data[0] = artifact_img
        print('Artificial Phase Shadow Artifact Added...')

    return image_data[0], image_data[1], image_data[2], sitk_img, artifact_img_name, original_pp_img, artifact_msk

def undo_preprocess(prediction_img, original_image):
    X = (prediction_img + 1)/2
    X = X*(200-(-1024))+(-1024)
    pred = sitk.GetImageFromArray(X)
    pred.SetOrigin(original_image.GetOrigin())
    pred.SetDirection(original_image.GetDirection())
    pred.SetSpacing(original_image.GetSpacing())
    return pred