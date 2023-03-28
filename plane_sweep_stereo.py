import time
from math import floor
import numpy as np
import cv2

from scipy.sparse import csr_matrix
import imageio

EPS = 1e-8


def backproject_corners(K, width, height, depth, Rt):
    """
    Backproject 4 corner points in image plane to the imaginary depth plane

    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        width -- width of the image
        heigh -- height of the image
        depth -- depth value of the imaginary plane
    Output:
        points -- 2 x 2 x 3 array of 3D coordinates of backprojected points
    """

    points = np.array((
        (0, 0, 1),
        (width, 0, 1),
        (0, height, 1),
        (width, height, 1),
    ), dtype=np.float32).reshape(2,2,3)

    # new_points=np.zeros((4,3))
    # print("inside")
    callibrated_coord=np.linalg.inv(K)@points.transpose(2,0,1).reshape(3,-1)
    depth_plane_cord=depth*callibrated_coord
    depth_plane_homo_cord=np.vstack((depth_plane_cord,np.ones((1,4))))
    Rt=np.vstack((Rt,np.array([[0,0,0,1]])))
    world_coord=np.linalg.inv(Rt)@depth_plane_homo_cord
    # print(world_coord)
    world_coord=world_coord[:-1,:]
    points=world_coord.reshape(3,-1,1).transpose(1,0,2).reshape(2,2,3)
    # print(points)
    # Rt=np.vstack((Rt,np.array([0,0,0,1])))
    # print(Rt)
    # for pts in points:
    #     callibrated_coord=np.linalg.inv(K)@pts
    #     depth_plane_cord=depth*callibrated_coord
    #     depth_plane_homo_cord=np.hstack((depth_plane_cord,1))
    #     print(depth_plane_homo_cord)
    #     world_coord=np.linalg.inv(Rt)@depth_plane_homo_cord
    #     print(world_coord)

    """ YOUR CODE HERE
    """

    """ END YOUR CODE
    """
    return points

def project_points(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- height x width x 3 array of 3D points
    Output:
        projections -- height x width x 2 array of 2D projections
    """
    """ YOUR CODE HERE
    """
    ones_arr=np.ones((points.shape[0],points.shape[1]))
    homo_points=np.dstack((points,ones_arr))
    homo_points_camera=np.einsum('ij,abj->abi',Rt,homo_points)
    points=np.einsum('ij,abj->abi',K,homo_points_camera)
    points=points[:,:,:-1]/points[:,:,2][:,:,None]
    # print(points.shape)
    """ END YOUR CODE
    """
    return points

def warp_neighbor_to_ref(backproject_fn, project_fn,depth, neighbor_rgb, K_ref, Rt_ref, K_neighbor, Rt_neighbor):
    """ 
    Warp the neighbor view into the reference view 
    via the homography induced by the imaginary depth plane at the specified depth

    Make use of the functions you've implemented in the same file:
    - backproject_corners
    - project_points

    Also make use of the cv2 functions:
    - cv2.findHomography
    - cv2.warpPerspective

    Input:
        depth -- scalar value of the depth at the imaginary depth plane
        neighbor_rgb -- height x width x 3 array of neighbor rgb image
        K_ref -- 3 x 3 camera intrinsics calibration matrix of reference view
        Rt_ref -- 3 x 4 camera extrinsics calibration matrix of reference view
        K_neighbor -- 3 x 3 camera intrinsics calibration matrix of neighbor view
        Rt_neighbor -- 3 x 4 camera extrinsics calibration matrix of neighbor view
    Output:
        warped_neighbor -- height x width x 3 array of the warped neighbor RGB image
    """

    # Unproject the pixel coordinates from the right camera onto the virtual
    # plane.

    height, width = neighbor_rgb.shape[:2]

    """ YOUR CODE HERE
    """
    Rt_ref=np.vstack((Rt_ref,np.array([[0,0,0,1]])))
    Rt_neighbor=np.vstack((Rt_neighbor,np.array([[0,0,0,1]])))
    transformation_matrix=Rt_neighbor@np.linalg.inv(Rt_ref)
    inv_depth=1/depth
    homography_matrix=K_neighbor@(transformation_matrix[:-1,:-1]+inv_depth*np.matmul(transformation_matrix[:-1,-1].reshape(-1,1),np.array([[0,0,1]])))@np.linalg.inv(K_ref)
    warped_neighbor=cv2.warpPerspective(neighbor_rgb,np.linalg.inv(homography_matrix),(width,height))
    """ END YOUR CODE

    """
    print(transformation_matrix)
    print(transformation_matrix[:-1,-1])
    return warped_neighbor


def zncc_kernel_2D(src, dst):
    """ 
    Compute the cost map between src and dst patchified images via the ZNCC metric
    
    IMPORTANT NOTES:
    - Treat each RGB channel separately but sum the 3 different zncc scores at each pixel

    - When normalizing by the standard deviation, add the provided small epsilon value, 
    EPS which is included in this file, to both sigma_src and sigma_dst to avoid divide-by-zero issues

    Input:
        src -- height x width x K**2 x 3
        dst -- height x width x K**2 x 3
    Output:
        zncc -- height x width array of zncc metric computed at each pixel
    """
    assert src.ndim == 4 and dst.ndim == 4
    assert src.shape[:] == dst.shape[:]

    """ YOUR CODE HERE
    """

    r_channel_src=src[:,:,:,0]
    g_channel_src=src[:,:,:,1]
    b_channel_src=src[:,:,:,2]

    r_channel_dst=dst[:,:,:,0]
    g_channel_dst=dst[:,:,:,1]
    b_channel_dst=dst[:,:,:,2]

    # print((r_channel_src-np.mean(r_channel_src,axis=1).reshape(-1,1)).shape)
    # print(np.mean(r_channel_src,axis=2))
    # print(np.mean(r_channel_src,axis=2).shape)
    mean_norm_r_src = (r_channel_src-np.mean(r_channel_src,axis=2).reshape(src.shape[0],src.shape[1],1))/(np.linalg.norm((r_channel_src-np.mean(r_channel_src,axis=2).reshape(src.shape[0],src.shape[1],1)),axis=2).reshape(src.shape[0],src.shape[1],1)/np.sqrt(src.shape[2])+EPS)
    mean_norm_g_src = (g_channel_src-np.mean(g_channel_src,axis=2).reshape(src.shape[0],src.shape[1],1))/(np.linalg.norm((g_channel_src-np.mean(g_channel_src,axis=2).reshape(src.shape[0],src.shape[1],1)),axis=2).reshape(src.shape[0],src.shape[1],1)/np.sqrt(src.shape[2])+EPS)
    mean_norm_b_src = (b_channel_src-np.mean(b_channel_src,axis=2).reshape(src.shape[0],src.shape[1],1))/(np.linalg.norm((b_channel_src-np.mean(b_channel_src,axis=2).reshape(src.shape[0],src.shape[1],1)),axis=2).reshape(src.shape[0],src.shape[1],1)/np.sqrt(src.shape[2])+EPS)    
    
    mean_norm_r_dst = (r_channel_dst-np.mean(r_channel_dst,axis=2).reshape(src.shape[0],src.shape[1],1))/(np.linalg.norm((r_channel_dst-np.mean(r_channel_dst,axis=2).reshape(src.shape[0],src.shape[1],1)),axis=2).reshape(src.shape[0],src.shape[1],1)/np.sqrt(src.shape[2])+EPS)
    mean_norm_g_dst = (g_channel_dst-np.mean(g_channel_dst,axis=2).reshape(src.shape[0],src.shape[1],1))/(np.linalg.norm((g_channel_dst-np.mean(g_channel_dst,axis=2).reshape(src.shape[0],src.shape[1],1)),axis=2).reshape(src.shape[0],src.shape[1],1)/np.sqrt(src.shape[2])+EPS)
    mean_norm_b_dst = (b_channel_dst-np.mean(b_channel_dst,axis=2).reshape(src.shape[0],src.shape[1],1))/(np.linalg.norm((b_channel_dst-np.mean(b_channel_dst,axis=2).reshape(src.shape[0],src.shape[1],1)),axis=2).reshape(src.shape[0],src.shape[1],1)/np.sqrt(src.shape[2])+EPS)       

    # mean_norm_r_dst = (r_channel_dst-np.mean(r_channel_dst,axis=1).reshape(-1,1))/(np.linalg.norm((r_channel_dst-np.mean(r_channel_dst,axis=1).reshape(-1,1)),axis=1).reshape(-1,1)/np.sqrt(src.shape[1])+EPS)
    # mean_norm_g_dst = (g_channel_dst-np.mean(g_channel_dst,axis=1).reshape(-1,1))/(np.linalg.norm((g_channel_dst-np.mean(g_channel_dst,axis=1).reshape(-1,1)),axis=1).reshape(-1,1)/np.sqrt(src.shape[1])+EPS)
    # mean_norm_b_dst = (b_channel_dst-np.mean(b_channel_dst,axis=1).reshape(-1,1))/(np.linalg.norm((b_channel_dst-np.mean(b_channel_dst,axis=1).reshape(-1,1)),axis=1).reshape(-1,1)/np.sqrt(src.shape[1])+EPS)

    zncc=np.sum(mean_norm_r_src*mean_norm_r_dst+mean_norm_g_src*mean_norm_g_dst+mean_norm_b_src*mean_norm_b_dst,axis=2)

    """ END YOUR CODE
    """

    return zncc  # height x width


def backproject(dep_map, K):
    """ 
    Backproject image points to 3D coordinates wrt the camera frame according to the depth map

    Input:
        K -- camera intrinsics calibration matrix
        dep_map -- height x width array of depth values
    Output:
        points -- height x width x 3 array of 3D coordinates of backprojected points
    """
    _u, _v = np.meshgrid(np.arange(dep_map.shape[1]), np.arange(dep_map.shape[0]))

    """ YOUR CODE HERE
    """
    xv_c = (_u-K[0,2])/K[0,0]
    yv_c = (_v-K[1,2])/K[1,1]
    z=dep_map
    xyz_cam=np.zeros((dep_map.shape[0],dep_map.shape[1],3))
    xyz_cam[:,:,0]=z*xv_c
    xyz_cam[:,:,1]=z*yv_c
    xyz_cam[:,:,2]=z
    
    """ END YOUR CODE
    """
    return xyz_cam

