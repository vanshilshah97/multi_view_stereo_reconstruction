import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
import imageio
from tqdm import tqdm
from transforms3d.euler import mat2euler, euler2mat
import pyrender
import trimesh
import cv2
import open3d as o3d
import scipy
# from tqdm import tqdm
# from tqdm.notebook import trange, tqdm


from dataloader import load_middlebury_data
from utils import viz_camera_poses

EPS = 1e-8


def homo_corners(h, w, H):
    corners_bef = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    corners_aft = cv2.perspectiveTransform(corners_bef, H).squeeze(1)
    u_min, v_min = corners_aft.min(axis=0)
    u_max, v_max = corners_aft.max(axis=0)
    return u_min, u_max, v_min, v_max


def rectify_2view(rgb_i, rgb_j, R_irect, R_jrect, K_i, K_j, u_padding=20, v_padding=20):
    """Given the rectify rotation, compute the rectified view and corrected projection matrix

    Parameters
    ----------
    rgb_i,rgb_j : [H,W,3]
    R_irect,R_jrect : [3,3]
        p_rect_left = R_irect @ p_i
        p_rect_right = R_jrect @ p_j
    K_i,K_j : [3,3]
        original camera matrix
    u_padding,v_padding : int, optional
        padding the border to remove the blank space, by default 20

    Returns
    -------
    [H,W,3],[H,W,3],[3,3],[3,3]
        the rectified images
        the corrected camera projection matrix. WE HELP YOU TO COMPUTE K, YOU DON'T NEED TO CHANGE THIS
    """
    # reference: https://stackoverflow.com/questions/18122444/opencv-warpperspective-how-to-know-destination-image-size
    assert rgb_i.shape == rgb_j.shape, "This hw assumes the input images are in same size"
    h, w = rgb_i.shape[:2]

    ui_min, ui_max, vi_min, vi_max = homo_corners(h, w, K_i @ R_irect @ np.linalg.inv(K_i))
    uj_min, uj_max, vj_min, vj_max = homo_corners(h, w, K_j @ R_jrect @ np.linalg.inv(K_j))

    # The distortion on u direction (the world vertical direction) is minor, ignore this
    w_max = int(np.floor(max(ui_max, uj_max))) - u_padding * 2
    h_max = int(np.floor(min(vi_max - vi_min, vj_max - vj_min))) - v_padding * 2

    print(w_max,h_max)

    assert K_i[0, 2] == K_j[0, 2], "This hw assumes original K has same cx"
    K_i_corr, K_j_corr = K_i.copy(), K_j.copy()
    K_i_corr[0, 2] -= u_padding
    K_i_corr[1, 2] -= vi_min + v_padding
    K_j_corr[0, 2] -= u_padding
    K_j_corr[1, 2] -= vj_min + v_padding

    """Student Code Starts"""
    H_i=K_i_corr@(R_irect@np.linalg.inv(K_i))
    H_j=K_j_corr@(R_jrect@np.linalg.inv(K_j))
    rgb_i_rect=cv2.warpPerspective(rgb_i,H_i,(w_max,h_max))
    rgb_j_rect=cv2.warpPerspective(rgb_j,H_j,(w_max,h_max))
    """Student Code Ends"""

    return rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr


def compute_right2left_transformation(R_wi, T_wi, R_wj, T_wj):
    """Compute the transformation that transform the coordinate from j coordinate to i

    Parameters
    ----------
    R_wi, R_wj : [3,3]
    T_wi, T_wj : [3,1]
        p_i = R_wi @ p_w + T_wi
        p_j = R_wj @ p_w + T_wj
    Returns
    -------
    [3,3], [3,1], float
        p_i = R_ji @ p_j + T_ji, B is the baseline
    """

    """Student Code Starts"""
    R_iw=R_wi.T
    T_iw=-R_wi.T@T_wi
    
    R_jw=R_wj.T
    T_jw=-R_wj.T@T_wj

    R_ji=R_wi@R_jw
    T_ji=T_wi+R_wi@T_jw

    B=np.linalg.norm(T_ji)
    """Student Code Ends"""

    return R_ji, T_ji, B


def compute_rectification_R(T_ji):
    """Compute the rectification Rotation

    Parameters
    ----------
    T_ji : [3,1]

    Returns
    -------
    [3,3]
        p_rect = R_irect @ p_i
    """
    # check the direction of epipole, should point to the positive direction of y axis
    # print(e_i)
    e_i = T_ji.squeeze(-1) / (T_ji.squeeze(-1)[1] + EPS)
    print(e_i)
    # print(e_i)
    # print(T_ji.squeeze(-1))
    # print(np.linalg.norm(e_i))
    """Student Code Starts"""
    # R_irect=np.zeros((3,3))
    # R_irect[0,:]=np.cross([0,0,1],T_ji.squeeze(-1))
    # R_irect[0,:]=R_irect[0,:]/np.linalg.norm(R_irect[0,:])
    # R_irect[1,:]=T_ji.squeeze(-1)/np.linalg.norm(T_ji.squeeze(-1))
    # R_irect[2,:]= np.cross(R_irect[0,:],R_irect[1,:])

    R_irect=np.zeros((3,3))
    R_irect[0,:]=np.cross(e_i,[0,0,1])
    R_irect[0,:]=R_irect[0,:]/np.linalg.norm(R_irect[0,:])
    R_irect[1,:]=e_i/np.linalg.norm(e_i)
    R_irect[2,:]= np.cross(R_irect[0,:],R_irect[1,:])
    # print(R_irect@e_i)

    """Student Code Ends"""

    return R_irect


def ssd_kernel(src, dst):
    """Compute SSD Error, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        error score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    """Student Code Starts"""
    ssd=(scipy.spatial.distance_matrix(src[:,:,0],dst[:,:,0]))**2+(scipy.spatial.distance_matrix(src[:,:,1],dst[:,:,1]))**2+(scipy.spatial.distance_matrix(src[:,:,2],dst[:,:,2]))**2
    """Student Code Ends"""

    return ssd  # M,N


def sad_kernel(src, dst):
    """Compute SSD Error, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        error score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    """Student Code Starts"""
    
    """Student Code Ends"""
    sad=(scipy.spatial.distance_matrix(src[:,:,0],dst[:,:,0],p=1))+(scipy.spatial.distance_matrix(src[:,:,1],dst[:,:,1],p=1))+(scipy.spatial.distance_matrix(src[:,:,2],dst[:,:,2],p=1))

    return sad  # M,N


def zncc_kernel(src, dst):
    """Compute negative zncc similarity, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    """Student Code Starts"""
    r_channel_src=src[:,:,0]
    g_channel_src=src[:,:,1]
    b_channel_src=src[:,:,2]

    r_channel_dst=dst[:,:,0]
    g_channel_dst=dst[:,:,1]
    b_channel_dst=dst[:,:,2]

    # print((r_channel_src-np.mean(r_channel_src,axis=1).reshape(-1,1)).shape)

    mean_norm_r_src = (r_channel_src-np.mean(r_channel_src,axis=1).reshape(-1,1))/(np.linalg.norm((r_channel_src-np.mean(r_channel_src,axis=1).reshape(-1,1)),axis=1).reshape(-1,1)/np.sqrt(src.shape[1])+EPS)
    mean_norm_g_src = (g_channel_src-np.mean(g_channel_src,axis=1).reshape(-1,1))/(np.linalg.norm((g_channel_src-np.mean(g_channel_src,axis=1).reshape(-1,1)),axis=1).reshape(-1,1)/np.sqrt(src.shape[1])+EPS)
    mean_norm_b_src = (b_channel_src-np.mean(b_channel_src,axis=1).reshape(-1,1))/(np.linalg.norm((b_channel_src-np.mean(b_channel_src,axis=1).reshape(-1,1)),axis=1).reshape(-1,1)/np.sqrt(src.shape[1])+EPS)

    mean_norm_r_dst = (r_channel_dst-np.mean(r_channel_dst,axis=1).reshape(-1,1))/(np.linalg.norm((r_channel_dst-np.mean(r_channel_dst,axis=1).reshape(-1,1)),axis=1).reshape(-1,1)/np.sqrt(src.shape[1])+EPS)
    mean_norm_g_dst = (g_channel_dst-np.mean(g_channel_dst,axis=1).reshape(-1,1))/(np.linalg.norm((g_channel_dst-np.mean(g_channel_dst,axis=1).reshape(-1,1)),axis=1).reshape(-1,1)/np.sqrt(src.shape[1])+EPS)
    mean_norm_b_dst = (b_channel_dst-np.mean(b_channel_dst,axis=1).reshape(-1,1))/(np.linalg.norm((b_channel_dst-np.mean(b_channel_dst,axis=1).reshape(-1,1)),axis=1).reshape(-1,1)/np.sqrt(src.shape[1])+EPS)

    # print(mean_norm_r_src.shape)
    zncc=np.matmul(mean_norm_r_src,mean_norm_r_dst.T)+np.matmul(mean_norm_g_src,mean_norm_g_dst.T)+np.matmul(mean_norm_b_src,mean_norm_b_dst.T)
    # print(zncc.shape)
    """Student Code Ends"""
    # zncc=np.zeros((src.shape[0],dst.shape[0]))

    return zncc * (-1.0)  # M,N


def image2patch(image, k_size):
    """get patch buffer for each pixel location from an input image; For boundary locations, use zero padding

    Parameters
    ----------
    image : [H,W,3]
    k_size : int, must be odd number; your function should work when k_size = 1

    Returns
    -------
    [H,W,k_size**2,3]
        The patch buffer for each pixel
    """

    """Student Code Starts"""
    if(k_size==1):
        return image[:,:,np.newaxis,:]

    """Student Code Starts"""
    patch_size=np.arange(-k_size//2+1,k_size//2+1)
    patch_buffer=np.zeros((image.shape[0],image.shape[1],k_size*k_size,3))

    for i_ind in range(image.shape[0]):
        for j_ind in range(image.shape[1]):
            i_patch=i_ind+patch_size
            j_patch=j_ind+patch_size
            i_img,j_img=np.meshgrid(i_patch,j_patch,indexing='ij')        
            i_mask=np.logical_and(i_img>=0,i_img<image.shape[0])
            j_mask=np.logical_and(j_img>=0,j_img<image.shape[1])
            final_mask=np.logical_and(i_mask,j_mask)
            new_vector=np.zeros((k_size*k_size,3))
            
            flatten_mask=final_mask.flatten()
            valid_i=i_img.flatten()[flatten_mask]
            valid_j=j_img.flatten()[flatten_mask]
            new_vector[flatten_mask]=image[valid_i,valid_j]
            patch_buffer[i_ind,j_ind]=new_vector

            
        
    return patch_buffer  # H,W,K**2,3


def compute_disparity_map(rgb_i, rgb_j, d0, k_size=5, kernel_func=ssd_kernel):
    """Compute the disparity map from two rectified view

    Parameters
    ----------
    rgb_i,rgb_j : [H,W,3]
    d0 : see the hand out, the bias term of the disparty caused by different K matrix
    k_size : int, optional
        The patch size, by default 3
    kernel_func : function, optional
        the kernel used to compute the patch similarity, by default ssd_kernel

    Returns
    -------
    disp_map: [H,W], dtype=np.float64
        The disparity map, the disparity is defined in the handout as d0 + vL - vR

    lr_consistency_mask: [H,W], dtype=np.float64
        For each pixel, 1.0 if LR consistent, otherwise 0.0
    """

    """Student Code Starts"""
    patches_i = image2patch(rgb_i.astype(float) / 255.0, k_size)  # [h,w,k*k,3]
    patches_j = image2patch(rgb_j.astype(float) / 255.0, k_size)  # [h,w,k*k,3]
    h=rgb_i.shape[0]
    vi_idx, vj_idx = np.arange(h), np.arange(h) 
    disp_candidates = vi_idx[:, None] - vj_idx[None, :] +d0
    valid_disp_mask = disp_candidates > 0.0
    
    disp_map=np.zeros_like(rgb_i[:,:,0],dtype=np.float64)
    lr_consistency_mask=np.zeros_like(rgb_i[:,:,0],dtype=np.float64)
      

    
    for scanline in range(rgb_i.shape[1]):
        buf_i, buf_j = patches_i[:, scanline], patches_j[:, scanline]
        value = kernel_func(buf_i, buf_j)
        _upper = value.max() + 1.0
        value[~valid_disp_mask] = _upper
        best_matched_right_pixel_correspondin_arr=np.argmin(value,axis=1)
        
        # print(best_matched_right_pixel_correspondin_arr)
        # print(disp_candidates[:,best_matched_right_pixel_correspondin_arr].shape)
        # disp_map[:,scanline]=np.take_along_axis(disp_candidates, best_matched_right_pixel_correspondin_arr.reshape(-1,1), axis=1).reshape(-1,)
        disp_map[:,scanline]= np.arange(h) - best_matched_right_pixel_correspondin_arr+d0
        best_matchched_left_pixel_correspondin_arr=np.argmin(value[:,best_matched_right_pixel_correspondin_arr],axis=0)
        # print(best_matchched_left_pixel_correspondin_arr)
        # print(np.arange(h))
        lr_consistency_mask[:,scanline]=best_matchched_left_pixel_correspondin_arr==np.arange(h)
        # break

    """Student Code Ends"""

    return disp_map.astype('float64'), lr_consistency_mask.astype('float64')


def compute_dep_and_pcl(disp_map, B, K):
    """Given disparity map d = d0 + vL - vR, the baseline and the camera matrix K
    compute the depth map and backprojected point cloud

    Parameters
    ----------
    disp_map : [H,W]
        disparity map
    B : float
        baseline
    K : [3,3]
        camera matrix

    Returns
    -------
    [H,W]
        dep_map
    [H,W,3]
        each pixel is the xyz coordinate of the back projected point cloud in camera frame
    """

    """Student Code Starts"""
    dep_map=K[1,1]*B/(disp_map+EPS)
    print(dep_map.max())
    row = np.array([i for i in range(disp_map.shape[0])])
    column = np.array([i for i in range(disp_map.shape[1])])
    rv, cv = np.meshgrid(row, column,indexing='ij')
    
    
    xv_c = (cv-K[0,2])/K[0,0]
    yv_c = (rv-K[1,2])/K[1,1]
    z=dep_map
    xyz_cam=np.zeros((disp_map.shape[0],disp_map.shape[1],3))
    xyz_cam[:,:,0]=z*xv_c
    xyz_cam[:,:,1]=z*yv_c
    xyz_cam[:,:,2]=z


    
    """Student Code Ends"""

    return dep_map, xyz_cam


def postprocess(
    dep_map,
    rgb,
    xyz_cam,
    R_wc,
    T_wc,
    consistency_mask=None,
    hsv_th=45,
    hsv_close_ksize=11,
    z_near=0.45,
    z_far=0.65,
):
    """
    Your goal in this function is: 
    given pcl_cam [N,3], R_wc [3,3] and T_wc [3,1]
    compute the pcl_world with shape[N,3] in the world coordinate
    """

    # extract mask from rgb to remove background
    mask_hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., -1]
    mask_hsv = (mask_hsv > hsv_th).astype(np.uint8) * 255
    # imageio.imsave("./debug_hsv_mask.png", mask_hsv)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hsv_close_ksize, hsv_close_ksize))
    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, morph_kernel).astype(float)
    # imageio.imsave("./debug_hsv_mask_closed.png", mask_hsv)

    # constraint z-near, z-far
    mask_dep = ((dep_map > z_near) * (dep_map < z_far)).astype(float)
    # imageio.imsave("./debug_dep_mask.png", mask_dep)

    mask = np.minimum(mask_dep, mask_hsv)
    if consistency_mask is not None:
        mask = np.minimum(mask, consistency_mask)
    # imageio.imsave("./debug_before_xyz_mask.png", mask)

    # filter xyz point cloud
    pcl_cam = xyz_cam.reshape(-1, 3)[mask.reshape(-1) > 0]
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcl_cam.reshape(-1, 3).copy())
    cl, ind = o3d_pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
    _pcl_mask = np.zeros(pcl_cam.shape[0])
    _pcl_mask[ind] = 1.0
    pcl_mask = np.zeros(xyz_cam.shape[0] * xyz_cam.shape[1])
    pcl_mask[mask.reshape(-1) > 0] = _pcl_mask
    mask_pcl = pcl_mask.reshape(xyz_cam.shape[0], xyz_cam.shape[1])
    # imageio.imsave("./debug_pcl_mask.png", mask_pcl)
    mask = np.minimum(mask, mask_pcl)
    # imageio.imsave("./debug_final_mask.png", mask)

    pcl_cam = xyz_cam.reshape(-1, 3)[mask.reshape(-1) > 0]
    pcl_color = rgb.reshape(-1, 3)[mask.reshape(-1) > 0]

    """Student Code Starts"""
    pcl_world=R_wc.T@pcl_cam.T-R_wc.T@T_wc
    pcl_world=pcl_world.T
    """Student Code Ends"""

    # np.savetxt("./debug_pcl_world.txt", np.concatenate([pcl_world, pcl_color], -1))
    # np.savetxt("./debug_pcl_rect.txt", np.concatenate([pcl_cam, pcl_color], -1))

    return mask, pcl_world, pcl_cam, pcl_color


def two_view(view_i, view_j, k_size=5, kernel_func=ssd_kernel):
    # Full pipeline

    # * 1. rectify the views
    R_wi, T_wi = view_i["R"], view_i["T"][:, None]  # p_i = R_wi @ p_w + T_wi
    R_wj, T_wj = view_j["R"], view_j["T"][:, None]  # p_j = R_wj @ p_w + T_wj

    R_ji, T_ji, B = compute_right2left_transformation(R_wi, T_wi, R_wj, T_wj)
    assert T_ji[1, 0] > 0, "here we assume view i should be on the left, not on the right"

    R_irect = compute_rectification_R(T_ji)

    rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr = rectify_2view(
        view_i["rgb"],
        view_j["rgb"],
        R_irect,
        R_irect @ R_ji,
        view_i["K"],
        view_j["K"],
        u_padding=20,
        v_padding=20,
    )

    # * 2. compute disparity
    assert K_i_corr[1, 1] == K_j_corr[1, 1], "This hw assumes the same focal Y length"
    assert (K_i_corr[0] == K_j_corr[0]).all(), "This hw assumes the same K on X dim"
    assert (
        rgb_i_rect.shape == rgb_j_rect.shape
    ), "This hw makes rectified two views to have the same shape"
    disp_map, consistency_mask = compute_disparity_map(
        rgb_i_rect,
        rgb_j_rect,
        d0=K_j_corr[1, 2] - K_i_corr[1, 2],
        k_size=k_size,
        kernel_func=kernel_func,
    )

    # * 3. compute depth map and filter them
    dep_map, xyz_cam = compute_dep_and_pcl(disp_map, B, K_i_corr)

    mask, pcl_world, pcl_cam, pcl_color = postprocess(
        dep_map,
        rgb_i_rect,
        xyz_cam,
        R_wc=R_irect @ R_wi,
        T_wc=R_irect @ T_wi,
        consistency_mask=consistency_mask,
        z_near=0.5,
        z_far=0.6,
    )

    return pcl_world, pcl_color, disp_map, dep_map


def main():
    DATA = load_middlebury_data("data/templeRing")
    # viz_camera_poses(DATA)
    two_view(DATA[0], DATA[3], 5, zncc_kernel)

    return


if __name__ == "__main__":
    main()
