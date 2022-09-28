import numpy as np
import cv2


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
    ), dtype=np.float32).reshape(2, 2, 3)
    

    """ YOUR CODE HERE
    """
    multiplier = depth * np.linalg.inv(K)                                   #(3x3)
    points = points.reshape(4,3)                                            #(4x3)
    points = np.matmul(multiplier, points.T)                                #(3x4)
    points = np.matmul(np.linalg.inv(Rt[:,:3]), (points - Rt[:,3].reshape(3,1))).T  #(4x3)
    points = points.reshape((2,2,3))
    """ END YOUR CODE
    """
    return points

def project_points(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- points_height x points_width x 3 array of 3D points
    Output:
        projections -- points_height x points_width x 2 array of 2D projections
    """
    """ YOUR CODE HERE
    """
    h = points.shape[0]
    w = points.shape[1]
    points = points.reshape(h*w,3)                                                  #((h*w)x3) =  (Nx3)
    points = points.T                                                               #(3xN)
    ones = np.ones(h*w)
    calibrated_world_points = np.vstack((points, ones))                             #(4xN)
    calibrated_camera_points = np.matmul(K, np.matmul(Rt, calibrated_world_points)) #((3x3) @ (3x4) @ (4xN)) = (3xN)
    calibrated_camera_points = calibrated_camera_points/calibrated_camera_points[2,:]   #NORMALIZE
    calibrated_camera_points = calibrated_camera_points[:2,:]                       #(2xN)
    calibrated_camera_points = calibrated_camera_points.T                           #(Nx2)
    points = calibrated_camera_points.reshape((h,w,2))                              #(h,w,2)
    
    """ END YOUR CODE
    """
    return points

def warp_neighbor_to_ref(backproject_fn, project_fn, depth, neighbor_rgb, K_ref, Rt_ref, K_neighbor, Rt_neighbor):
    """ 
    Warp the neighbor view into the reference view 
    via the homography induced by the imaginary depth plane at the specified depth

    Make use of the functions you've implemented in the same file (which are passed in as arguments):
    - backproject_corners
    - project_points

    Also make use of the cv2 functions:
    - cv2.findHomography
    - cv2.warpPerspective

    Input:
        backproject_fn -- backproject_corners function
        project_fn -- project_points function
        depth -- scalar value of the depth at the imaginary depth plane
        neighbor_rgb -- height x width x 3 array of neighbor rgb image
        K_ref -- 3 x 3 camera intrinsics calibration matrix of reference view
        Rt_ref -- 3 x 4 camera extrinsics calibration matrix of reference view
        K_neighbor -- 3 x 3 camera intrinsics calibration matrix of neighbor view
        Rt_neighbor -- 3 x 4 camera extrinsics calibration matrix of neighbor view
    Output:
        warped_neighbor -- height x width x 3 array of the warped neighbor RGB image
    """

    height, width = neighbor_rgb.shape[:2]

    """ YOUR CODE HERE
    """
    actual_corners = np.array((
        (0, 0),
        (width, 0),
        (0, height),
        (width, height),
    ), dtype=np.float32).reshape(2, 2, 2)
    
    #step 1 : corners in reference view image plane backprojected to imaginary depth plane
    src_corners = backproject_fn(K_ref, width, height, depth, Rt_ref)
    #print(src_corners)
    #print(src_corners.shape)        #(2, 2, 3)
    
    #step 2 : project 3D depth plane corner points to calibrated camera. This calibrated camera is the neighboring one
    dst_corners = project_fn(K_neighbor, Rt_neighbor, src_corners)
    #print(dst_corners)
    #print(dst_corners.shape)        #(2, 2, 2)
    
    #step 3 : reshape the actual and destination corners
    actual_corners = actual_corners.reshape((-1,2))
    dst_corners = dst_corners.reshape((-1,2))

    #step 4 : find homography between the dst_corners and actual_corners
    H, _ = cv2.findHomography(dst_corners, actual_corners)
    
    #step 5 : warp perspective to warp other points of neighbor_rgb
    warped_neighbor = cv2.warpPerspective(neighbor_rgb, H, (width, height))
    #print(warped_neighbor)
      
    """ END YOUR CODE
    """
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
    src_mean = np.mean(src, axis = 2)
    #print(src_mean.shape)      #must be 475x611x3
    dst_mean = np.mean(dst, axis = 2)
    src_std_dev = np.std(src, axis = 2)
    dst_std_dev = np.std(dst, axis = 2)
    
    zncc_total = np.empty((src.shape[0], src.shape[1], src.shape[3])) #475x611x3
    
    for channel in range(src.shape[3]):
        for h in range(src.shape[0]):
            for w in range(src.shape[1]):
                zncc_total[h,w, channel] = np.sum(np.multiply(src[h,w,:,channel]-src_mean[h,w,channel], dst[h,w,:,channel]-dst_mean[h,w,channel]))/(np.multiply(src_std_dev[h,w, channel], dst_std_dev[h,w, channel])+EPS)

    zncc = np.sum(zncc_total, axis = 2)
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
    u0 = K[0,2]
    v0 = K[1,2]
    fX = K[0,0]
    fY = K[1,1]
    
    xyz_cam = np.stack([np.multiply((_u.flatten()-u0), dep_map.flatten())/fX, np.multiply((_v.flatten()-v0), dep_map.flatten())/fY, dep_map.flatten()]).T
    xyz_cam = xyz_cam.reshape((dep_map.shape[0],dep_map.shape[1],3))  
    """ END YOUR CODE
    """
    return xyz_cam

### DELETE BELOW THIS LINE ###
'''
K = np.array([[1.5204e+03, 0.0000e+00, 3.0232e+02],
 [0.0000e+00, 1.5259e+03, 2.4687e+02],
 [0.0000e+00, 0.0000e+00 ,1.0000e+00]])

width = 640
height = 480
depth = 0.5
Rt = np.array(([[-0.12459423,  0.98895929 ,-0.08022345 ,-0.02132782],
 [ 0.28153513, -0.04229297, -0.95861842, -0.05858865],
 [-0.95142748, -0.14202405, -0.27315732,  0.57767114]]))

print(backproject_corners(K, width, height, depth, Rt).shape)
'''