import numpy as np
import pandas as pd
import random
import time
import copy
import torch
import torchio as tio
import SimpleITK as sitk
import matplotlib.pyplot as plt
from Networks_3D import UNet
from Networks_R3U import R3U_Net, AttU_Net, R3AttU_Net
from interp_module_3D import *
from phase_shadow_module import *
from duplication_module_3D import * 
from preprocess_functions import *
from skimage.metrics       import structural_similarity as ssim
from skimage.metrics       import peak_signal_noise_ratio
from matplotlib.animation import FuncAnimation


def infer_and_reconstruct_volume(model, input_volume, in_loc, aggregator, device):
    
    output_patches  = torch.rand(125, 1, 128, 128, 128)
    
    for i in range(len(in_loc)):
        
        idx         = in_loc[i]
        input_patch = input_volume[:,idx[0]:idx[3], idx[1]:idx[4], idx[2]:idx[5]]
        # input_patch = np.expand_dims(input_patch, 0)
        input_patch = np.expand_dims(input_patch, 0)
        input_patch = torch.from_numpy(input_patch.copy())
        # print(input_patch.shape)
        input_patch = input_patch.float()
        assert not torch.any(torch.isnan(input_patch))
        input_patch = input_patch.to(device)
        
        with torch.no_grad():
            fake = model(input_patch)
            
        output_patches[i, :, :, :, :] = fake.squeeze(0).detach().cpu()          
    
    # print(output_patches.shape)
    aggregator.add_batch(output_patches, in_loc)
    pred = aggregator.get_output_tensor()
    
    return pred

def make_gif_from_numpy_comparison(real,pred,save_path,dpi,time_interval):
    # Create figure and axes
    fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(10,5))
    im0 = ax[0].imshow(real[0], animated=True,cmap='gray')
    im1 = ax[1].imshow(pred[0], animated=True,cmap='gray')
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    ax[0].set_title('Real')
    ax[1].set_title('Model Prediction')
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    fig.set_dpi(dpi)

    # Update function for animation
    def update(i):
        im0.set_array(real[i])
        im1.set_array(pred[i])
        return im0,im1

    # Create animation
    ani = FuncAnimation(fig, update, frames=range(len(real)), interval=time_interval, blit=True)

    # Save as GIF
    ani.save(save_path, writer='pillow')

def visualize_results(save_path, real_img, art_img, pred_img, art_msk):
    art_inds = np.where(art_msk == 1)
    min_art_inds = np.min(art_inds,axis=1)[0]
    max_art_inds = np.max(art_inds,axis=1)[0]
    ssim_pred = ssim(real_img[min_art_inds:max_art_inds,:,:], pred_img[min_art_inds:max_art_inds,:,:], data_range=1-(-1))
    psnr_pred = peak_signal_noise_ratio(real_img[min_art_inds:max_art_inds,:,:], pred_img[min_art_inds:max_art_inds,:,:], data_range=1-(-1))
    ssim_art = ssim(real_img[min_art_inds:max_art_inds,:,:], art_img[:,:,:][min_art_inds:max_art_inds,:,:], data_range=1-(-1))
    psnr_art = peak_signal_noise_ratio(real_img[min_art_inds:max_art_inds,:,:], art_img[:,:,:][min_art_inds:max_art_inds,:,:], data_range=1-(-1))
    cor_slc = np.ceil(real_img.shape[1]/2)
    axi_slc = np.median(art_inds,axis=1)[0]
    sag_slc = np.ceil(real_img.shape[2]/4)
    fig,ax = plt.subplots(nrows=3,ncols=4,figsize=(30,20))
    ax[0,0].imshow(art_img[:,cor_slc,:],cmap='gray')
    ax[0,1].imshow(pred_img[:,cor_slc,:],cmap='gray')
    ax[0,2].imshow(real_img[:,cor_slc,:],cmap='gray')
    d1 = ax[0,3].imshow(np.abs(pred_img[:,cor_slc,:] - real_img[:,cor_slc,:]),cmap='inferno',vmin=0,vmax=1)
    ax[1,0].imshow(art_img[axi_slc,:,:],cmap='gray')
    ax[1,1].imshow(pred_img[axi_slc,:,:],cmap='gray')
    ax[1,2].imshow(real_img[axi_slc,:,:],cmap='gray')
    d2 = ax[1,3].imshow(np.abs(pred_img[axi_slc,:,:] - real_img[axi_slc,:,:]),cmap='inferno',vmin=0,vmax=1)
    ax[2,0].imshow(art_img[:,:,sag_slc],cmap='gray')
    ax[2,1].imshow(pred_img[:,:,sag_slc],cmap='gray')
    ax[2,2].imshow(real_img[:,:,sag_slc],cmap='gray')
    d3 = ax[2,3].imshow(np.abs(pred_img[:,:,sag_slc] - real_img[:,:,sag_slc]),cmap='inferno',vmin=0,vmax=1)
    ax[0,0].set_xticks([])
    ax[0,0].set_yticks([])
    ax[0,1].set_axis_off()
    ax[0,2].set_axis_off()
    ax[0,3].set_axis_off()
    ax[1,0].set_xticks([])
    ax[1,0].set_yticks([])
    ax[1,1].set_axis_off()
    ax[1,2].set_axis_off()
    ax[1,3].set_axis_off()
    ax[2,0].set_xticks([])
    ax[2,0].set_yticks([])
    ax[2,1].set_axis_off()
    ax[2,2].set_axis_off()
    ax[2,3].set_axis_off()
    plt.subplots_adjust(wspace=0.01, hspace=0)
    ax[0,0].set_title('Artifacted Image')
    ax[0,1].set_title('Model Prediction')
    ax[0,2].set_title('Real Image')
    ax[0,3].set_title('Difference')
    ax[0,0].set_ylabel('Coronal')
    ax[1,0].set_ylabel('Axial')
    ax[2,0].set_ylabel('Saggital')
    # fig.colorbar(d1, ax=ax, location='right', anchor=(-0.33,0.5), shrink=0.28)

    # textstr = '\n'.join((
    #     r'$SSIM=%.2f$' % (ssim_art, ),
    #     r'$PSNR=%.2f$' % (psnr_art, )))
    # props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    # ax[0].text(0.73, 0.96, textstr, verticalalignment='top',transform=ax[0].transAxes, fontsize=12,
    #         bbox=props)
    # textstr_pred = '\n'.join((
    #     r'$SSIM=%.2f$' % (ssim_pred, ),
    #     r'$PSNR=%.2f$' % (psnr_pred, )))
    # ax[1].text(0.73, 0.96, textstr_pred, verticalalignment='top',transform=ax[1].transAxes, fontsize=12,
    #         bbox=props)
    fig.set_dpi(300)
    plt.savefig(os.path.join(save_path,'model_full_image_comparison.png'),bbox_inches='tight')

def correct_NIFTI_image(model_path,path_2_4D_scan, save_path, ref_phase_1=None, artifact_phase=None, ref_phase_2=None,
                         add_artificial_interpolation=False, add_artificial_phase_shadow=False,
                         add_artificial_duplication=False):
    artifact_img, ref_img_1, ref_img_2, sitk_img, img_name, og_img, artifact_mask = preprocess_4D_scan_for_correction(path_2_4D_scan,
    ref_phase_1=ref_phase_1,
    artifact_phase=artifact_phase, 
    ref_phase_2=ref_phase_2, 
    add_artificial_interpolation=add_artificial_interpolation,
    add_artificial_phase_shadow=add_artificial_phase_shadow)

    print('Images Pre-processed!!!')
    device = torch.device('cuda:0')
    model = AttU_Net(3,1)
    model.load_state_dict(torch.load(model_path))
    model.eval().to(device)
    print('Model Loaded!!!')

    if artifact_img.shape[0] > 512:
        artifact_img = artifact_img[0:512,:,:]
        ref_img_1 = ref_img_1[0:512,:,:]
        ref_img_2 = ref_img_2[0:512,:,:]
    if artifact_img.shape[1] > 512:
        artifact_img = artifact_img[:,0:512,:]
        ref_img_1 = ref_img_1[:,0:512,:]
        ref_img_2 = ref_img_2[:,0:512,:]
    if artifact_img.shape[2] > 512:
        artifact_img = artifact_img[:,:,0:512]
        ref_img_1 = ref_img_1[:,:,0:512]
        ref_img_2 = ref_img_2[:,:,0:512]

    x_pad = (512-artifact_img.shape[0])/2
    y_pad = (512-artifact_img.shape[1])/2
    z_pad = (512-artifact_img.shape[2])/2

    if x_pad.is_integer():
        extra_top, extra_bottom = int(x_pad), int(x_pad)
    else:
        extra_top, extra_bottom = int(np.floor(x_pad)), int(np.floor(x_pad))+1
    if y_pad.is_integer():
        extra_left, extra_right = int(y_pad), int(y_pad)
    else:
        extra_left, extra_right = int(np.floor(y_pad)), int(np.floor(y_pad))+1
    if z_pad.is_integer():
        extra_front, extra_back = int(z_pad), int(z_pad)
    else:
        extra_front, extra_back = int(np.floor(z_pad)), int(np.floor(z_pad))+1

    art_img = np.pad(artifact_img, ((extra_top, extra_bottom), (extra_left, extra_right), (extra_front, extra_back)),mode='constant', constant_values=-1) 
    ref_img_1 = np.pad(ref_img_1, ((extra_top, extra_bottom), (extra_left, extra_right), (extra_front, extra_back)),mode='constant', constant_values=-1) 
    ref_img_2 = np.pad(ref_img_2, ((extra_top, extra_bottom), (extra_left, extra_right), (extra_front, extra_back)),mode='constant', constant_values=-1) 


    if add_artificial_phase_shadow == True:
        model_input = np.stack((art_img, ref_img_1, ref_img_2)) # PH model was trained in old configuration of reference phases.
    else:
        model_input = np.stack((ref_img_1, art_img, ref_img_2)) # Interp model was trained on this order of phases.
    print('Model Input Shape =',model_input.shape)

    input_subject     = tio.Subject(tlc = tio.ScalarImage(tensor = model_input))
    input_sampler     = tio.GridSampler(subject = input_subject, patch_size = 128, patch_overlap = 32)
    input_locations   = input_sampler.locations
    input_locations   = torch.from_numpy(input_locations.copy())     
    output_aggregator = tio.inference.GridAggregator(input_sampler, overlap_mode = 'average')

    start = time.time()
    pred = infer_and_reconstruct_volume(model,model_input,input_locations,output_aggregator,device)
    end = time.time()
    print(f"Time to apply model to entire 3D image was {np.floor(end-start)} seconds.")

    full_im_pred = np.squeeze(pred.detach().cpu().numpy(),axis=0)
    revert_input = model_input.copy()
    revert_pred = full_im_pred.copy()
    revert_input = revert_input[:,extra_top:512-extra_bottom,extra_left:512-extra_right,extra_front:512-extra_back]
    revert_pred = revert_pred[extra_top:512-extra_bottom,extra_left:512-extra_right,extra_front:512-extra_back]

    # visualize_results(save_path,og_img,revert_input[0,:,:,:],revert_pred,artifact_mask)

    revert_pred = revert_pred[::-1,:,:]
    revert_input = revert_input[1,::-1,:,:]
    sitk_pred_img = undo_preprocess(revert_pred,sitk_img)
    sitk_arti_img = undo_preprocess(revert_input,sitk_img)
    save_name = img_name.split('.')[0] + '_corrected.nii.gz'
    save_name_artifact = img_name.split('.')[0] + '_artifact.nii.gz'
    sitk.WriteImage(sitk_pred_img,os.path.join(save_path,save_name))
    sitk.WriteImage(sitk_arti_img,os.path.join(save_path,save_name_artifact))
    print('Corrected Image Saved as NIFTI File!!!')
    print(os.path.join(save_path,save_name))

def get_eval_metrics_from_NIFTI(model_path,path_2_4D_scan, save_path, artifact_phase, ref_phase_1=None, ref_phase_2=None,
                         add_artificial_interpolation=False, add_artificial_phase_shadow=False,
                         add_artificial_duplication=False):
    artifact_img, ref_img_1, ref_img_2, sitk_img, img_name, og_img, artifact_mask = preprocess_4D_scan_for_correction(path_2_4D_scan,
    artifact_phase=artifact_phase,
    ref_phase_1=ref_phase_1, 
    ref_phase_2=ref_phase_2, 
    add_artificial_interpolation=add_artificial_interpolation)

    print('Images Pre-processed!!!')
    device = torch.device('cuda:0')
    model = AttU_Net(3,1)
    model.load_state_dict(torch.load(model_path))
    model.eval().to(device)
    print('Model Loaded!!!')

    if artifact_img.shape[0] > 512:
        artifact_img = artifact_img[0:512,:,:]
        ref_img_1 = ref_img_1[0:512,:,:]
        ref_img_2 = ref_img_2[0:512,:,:]
    if artifact_img.shape[1] > 512:
        artifact_img = artifact_img[:,0:512,:]
        ref_img_1 = ref_img_1[:,0:512,:]
        ref_img_2 = ref_img_2[:,0:512,:]
    if artifact_img.shape[2] > 512:
        artifact_img = artifact_img[:,:,0:512]
        ref_img_1 = ref_img_1[:,:,0:512]
        ref_img_2 = ref_img_2[:,:,0:512]

    x_pad = (512-artifact_img.shape[0])/2
    y_pad = (512-artifact_img.shape[1])/2
    z_pad = (512-artifact_img.shape[2])/2

    if x_pad.is_integer():
        extra_top, extra_bottom = int(x_pad), int(x_pad)
    else:
        extra_top, extra_bottom = int(np.floor(x_pad)), int(np.floor(x_pad))+1
    if y_pad.is_integer():
        extra_left, extra_right = int(y_pad), int(y_pad)
    else:
        extra_left, extra_right = int(np.floor(y_pad)), int(np.floor(y_pad))+1
    if z_pad.is_integer():
        extra_front, extra_back = int(z_pad), int(z_pad)
    else:
        extra_front, extra_back = int(np.floor(z_pad)), int(np.floor(z_pad))+1

    art_img = np.pad(artifact_img, ((extra_top, extra_bottom), (extra_left, extra_right), (extra_front, extra_back)),mode='constant', constant_values=-1) 
    ref_img_1 = np.pad(ref_img_1, ((extra_top, extra_bottom), (extra_left, extra_right), (extra_front, extra_back)),mode='constant', constant_values=-1) 
    ref_img_2 = np.pad(ref_img_2, ((extra_top, extra_bottom), (extra_left, extra_right), (extra_front, extra_back)),mode='constant', constant_values=-1) 

    if add_artificial_phase_shadow == True:
        model_input = np.stack((art_img, ref_img_1, ref_img_2))
    else:
        model_input = np.stack((ref_img_1, art_img, ref_img_2))
    print('Model Input Shape =',model_input.shape)

    input_subject     = tio.Subject(tlc = tio.ScalarImage(tensor = model_input))
    input_sampler     = tio.GridSampler(subject = input_subject, patch_size = 128, patch_overlap = 32)
    input_locations   = input_sampler.locations
    input_locations   = torch.from_numpy(input_locations.copy())     
    output_aggregator = tio.inference.GridAggregator(input_sampler, overlap_mode = 'average')

    start = time.time()
    pred = infer_and_reconstruct_volume(model,model_input,input_locations,output_aggregator,device)
    end = time.time()
    print(f"Time to apply model to entire 3D image was {np.floor(end-start)} seconds.")

    full_im_pred = np.squeeze(pred.detach().cpu().numpy(),axis=0)
    revert_input = model_input.copy()
    revert_pred  = full_im_pred.copy()
    revert_input = revert_input[:,extra_top:512-extra_bottom,extra_left:512-extra_right,extra_front:512-extra_back]
    revert_pred  = revert_pred[extra_top:512-extra_bottom,extra_left:512-extra_right,extra_front:512-extra_back]

    ind         = np.array(np.where(artifact_mask>0))
    min_art_ind = np.min(ind,axis=1)
    max_art_ind = np.max(ind,axis=1)

    real_crop = og_img[min_art_ind[0]:max_art_ind[0], min_art_ind[1]:max_art_ind[1], min_art_ind[2]:max_art_ind[2]]
    pred_crop = revert_pred[min_art_ind[0]:max_art_ind[0], min_art_ind[1]:max_art_ind[1], min_art_ind[2]:max_art_ind[2]]

    rmse_metric = np.sqrt(np.divide(np.sum(np.power(real_crop-pred_crop,2)),real_crop.size))
    ssim_metric = ssim(real_crop, pred_crop, data_range=1-(-1))
    psnr_metric = peak_signal_noise_ratio(real_crop, pred_crop, data_range=1-(-1))

    return (rmse_metric, ssim_metric, psnr_metric)

def correct_image_and_save_to_NIFTI_for_test_data(model,path_2_4D_scan, save_path, ref_phase_1=None, artifact_phase=None, ref_phase_2=None,
                         add_artificial_interpolation=False, add_artificial_phase_shadow=False,
                         add_artificial_duplication=False):
    artifact_img, ref_img_1, ref_img_2, sitk_img, img_name, og_img, artifact_mask = preprocess_4D_scan_for_correction(path_2_4D_scan,
    ref_phase_1=ref_phase_1,
    artifact_phase=artifact_phase, 
    ref_phase_2=ref_phase_2, 
    add_artificial_interpolation=add_artificial_interpolation,
    add_artificial_phase_shadow=add_artificial_phase_shadow)

    print('Images Pre-processed!!!')
    device = torch.device('cuda:0')

    if artifact_img.shape[0] > 512:
        artifact_img = artifact_img[0:512,:,:]
        ref_img_1 = ref_img_1[0:512,:,:]
        ref_img_2 = ref_img_2[0:512,:,:]
    if artifact_img.shape[1] > 512:
        artifact_img = artifact_img[:,0:512,:]
        ref_img_1 = ref_img_1[:,0:512,:]
        ref_img_2 = ref_img_2[:,0:512,:]
    if artifact_img.shape[2] > 512:
        artifact_img = artifact_img[:,:,0:512]
        ref_img_1 = ref_img_1[:,:,0:512]
        ref_img_2 = ref_img_2[:,:,0:512]

    x_pad = (512-artifact_img.shape[0])/2
    y_pad = (512-artifact_img.shape[1])/2
    z_pad = (512-artifact_img.shape[2])/2

    if x_pad.is_integer():
        extra_top, extra_bottom = int(x_pad), int(x_pad)
    else:
        extra_top, extra_bottom = int(np.floor(x_pad)), int(np.floor(x_pad))+1
    if y_pad.is_integer():
        extra_left, extra_right = int(y_pad), int(y_pad)
    else:
        extra_left, extra_right = int(np.floor(y_pad)), int(np.floor(y_pad))+1
    if z_pad.is_integer():
        extra_front, extra_back = int(z_pad), int(z_pad)
    else:
        extra_front, extra_back = int(np.floor(z_pad)), int(np.floor(z_pad))+1

    art_img = np.pad(artifact_img, ((extra_top, extra_bottom), (extra_left, extra_right), (extra_front, extra_back)),mode='constant', constant_values=-1) 
    ref_img_1 = np.pad(ref_img_1, ((extra_top, extra_bottom), (extra_left, extra_right), (extra_front, extra_back)),mode='constant', constant_values=-1) 
    ref_img_2 = np.pad(ref_img_2, ((extra_top, extra_bottom), (extra_left, extra_right), (extra_front, extra_back)),mode='constant', constant_values=-1) 


    model_input = np.stack((ref_img_1, art_img, ref_img_2))
    print('Model Input Shape =',model_input.shape)

    input_subject     = tio.Subject(tlc = tio.ScalarImage(tensor = model_input))
    input_sampler     = tio.GridSampler(subject = input_subject, patch_size = 128, patch_overlap = 32)
    input_locations   = input_sampler.locations
    input_locations   = torch.from_numpy(input_locations.copy())     
    output_aggregator = tio.inference.GridAggregator(input_sampler, overlap_mode = 'average')

    start = time.time()
    pred = infer_and_reconstruct_volume(model,model_input,input_locations,output_aggregator,device)
    end = time.time()
    print(f"Time to apply model to entire 3D image was {np.floor(end-start)} seconds.")

    full_im_pred = np.squeeze(pred.detach().cpu().numpy(),axis=0)
    revert_input = model_input.copy()
    revert_pred = full_im_pred.copy()
    revert_input = revert_input[:,extra_top:512-extra_bottom,extra_left:512-extra_right,extra_front:512-extra_back]
    revert_pred = revert_pred[extra_top:512-extra_bottom,extra_left:512-extra_right,extra_front:512-extra_back]

    # visualize_results(save_path,og_img,revert_input[0,:,:,:],revert_pred,artifact_mask)

    revert_pred = revert_pred[::-1,:,:]
    revert_input = revert_input[1,::-1,:,:]
    sitk_pred_img = undo_preprocess(revert_pred,sitk_img)
    sitk_arti_img = undo_preprocess(revert_input,sitk_img)
    artifact_mask = sitk.GetImageFromArray(artifact_mask[::-1,:,:])
    artifact_mask.SetOrigin(sitk_img.GetOrigin())
    artifact_mask.SetDirection(sitk_img.GetDirection())
    artifact_mask.SetSpacing(sitk_img.GetSpacing())
    print('Images converted to SITK')
    save_name = img_name.split('.')[0] + '_corrected.nii.gz'
    save_name_artifact = img_name.split('.')[0] + '_artifact.nii.gz'
    save_name_artifact_mask = img_name.split('.')[0] + '_artifact.mask.nii.gz'
    sitk.WriteImage(sitk_pred_img,os.path.join(save_path,save_name))
    print('Corrected Image saved as NIFTI')
    sitk.WriteImage(sitk_arti_img,os.path.join(save_path,save_name_artifact))
    print('Artifact Image saved as NIFTI')
    sitk.WriteImage(artifact_mask,os.path.join(save_path,save_name_artifact_mask))
    print('Artifact Mask saved as NIFTI')
    print(os.path.join(save_path,save_name))