from __future__ import division
import torch
import torch.nn.functional as F

pixel_coords = None
def set_id_grid(depth):
    global pixel_coords
    b,h,w = depth.size()
    i_range = torch.arange(0,h).view(1,h,1).expand(1,h,w).type_as(depth)#(1,H,W)
    j_range = torch.arange(0,w).view(1,1,w).expand(1,h,w).type_as(depth)
    ones = torch.ones(1,h,w).type_as(depth)
    pixel_coords = torch.stack((j_range,i_range,ones),dim=1)#(1,3,H,W)


def check_sizes(input,input_name,expected):
    condition = [input.ndimension() == len(expected)] # ndimension =  len of dim
    for i,size in enumerate(expected):
        if size.isdigit():
            # Python isdigit() 判断数字维度对应
            condition.append(input.size(i)==int(size))
            assert (all(condition)),"wrong size for {}, expected {}, got {}".format(input_name,'x'.join(expected),list(input.size()))

def pixel2cam(depth,intrinsics_inv):
    """
    :param depth: [B,H,W]
    :param intrinsics_inv: [B,3,3]
    :return: Pc=Zc*Puv*K_inv
    """
    global pixel_coords
    b,h,w = depth.size()
    if (pixel_coords is None) or (pixel_coords.size(2)<h):
        set_id_grid(depth)
    current_pixel_coords = pixel_coords[:,:,:h,:w].expand(b,3,h,w).reshape(b,3,-1)#[B,3,H*W]
    cam_coors = (intrinsics_inv @ current_pixel_coords).reshape(b,3,h,w)

    return cam_coors * depth.unsqueeze(1)

def cam2pixel(cam_coords,proj_c2p_rot,proj_c2p_tr,padding_mode):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    X_norm = 2*(X / Z)/(w-1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]
    X_norm = X_norm.reshape(b,h,w)
    Y_norm = Y_norm.reshape(b, h, w)
    grid_x = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(X_norm).expand_as(
        X_norm)  # [bs, H, W]
    grid_y = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(Y_norm).expand_as(
        Y_norm)  # [bs, H, W]
    u = ((X_norm / 2.0 + 0.5) * (w - 1)) - grid_x
    v = ((Y_norm / 2.0 + 0.5) * (h - 1)) - grid_y
    dp_flow = torch.stack([u, v], dim=1)#[B,2,H,W]

    # if padding_mode == 'zeros':
    #     X_mask = ((X_norm > 1)+(X_norm < -1)).detach()
    #     X_norm[X_mask] = 2  # make sure that no point in warped image is a combinaison of im and gray
    #     Y_mask = ((Y_norm > 1)+(Y_norm < -1)).detach()
    #     Y_norm[Y_mask] = 2

    pixel_coords = torch.stack([X_norm, Y_norm], dim=-1)  # [B, H,W, 2]
    return pixel_coords, Z.reshape(b, 1, h, w),dp_flow

def so3_RV(omega):
    def vecMulMat(vec, mat):
        mat_view = mat.view(vec.size()[0], -1)
        out = mat_view * vec
        return out.view(mat_view.size()[0], mat.size()[1], -1)

    threshold_cube = 1e-1
    threshold_square = 1e-1
    """
    (3-tuple)
    omega = torch.zeros(batchSize, 3)

    return batchx3x3 matrix R after exponential mapping, V
    """
    batchSize = omega.size()[0]
    omega_x = omega[:, 2]
    omega_y = omega[:, 1]
    omega_z = omega[:, 0]

    # paramIndex = paramIndex + 3
    omega_skew = torch.zeros(batchSize, 3, 3).type_as(omega)
    """
    0    -oz  oy  0
    oz   0   -ox  0
    -oy  ox   0   0
    0    0    0   0
    """
    omega_skew[:, 1, 0] = omega_z.clone()
    omega_skew[:, 2, 0] = -1 * omega_y

    omega_skew[:, 0, 1] = -1 * omega_z
    omega_skew[:, 2, 1] = omega_x.clone()

    omega_skew[:, 0, 2] = omega_y.clone()
    omega_skew[:, 1, 2] = -1 * omega_x

    omega_skew_sqr = torch.bmm(omega_skew, omega_skew)
    theta_sqr = torch.pow(omega_x, 2) + \
                torch.pow(omega_y, 2) + \
                torch.pow(omega_z, 2)
    theta = torch.pow(theta_sqr, 0.5)
    theta_cube = torch.mul(theta_sqr, theta)  #
    sin_theta = torch.sin(theta)
    sin_theta_div_theta = torch.div(sin_theta, theta)
    sin_theta_div_theta[sin_theta_div_theta != sin_theta_div_theta] = 0  # set nan to zero

    one_minus_cos_theta = torch.ones(theta.size()).cuda() - torch.cos(theta)
    one_minus_cos_div_theta_sqr = torch.div(one_minus_cos_theta, theta_sqr)

    theta_minus_sin_theta = theta - torch.sin(theta)
    theta_minus_sin_div_theta_cube = torch.div(theta_minus_sin_theta, theta_cube)

    sin_theta_div_theta_tensor = torch.ones(omega_skew.size()).cuda()
    one_minus_cos_div_theta_sqr_tensor = torch.ones(omega_skew.size()).cuda()
    theta_minus_sin_div_theta_cube_tensor = torch.ones(omega_skew.size()).cuda()

    # sin_theta_div_theta do not need linear approximation
    sin_theta_div_theta_tensor = sin_theta_div_theta

    for b in range(batchSize):
        if theta_sqr[b] > threshold_square:
            one_minus_cos_div_theta_sqr_tensor[b] = one_minus_cos_div_theta_sqr[b]
        elif theta_sqr[b] < 1e-6:
            one_minus_cos_div_theta_sqr_tensor[b] = 0  # 0.5
        else:  # Taylor expansion
            c = 1.0 / 2.0
            c += theta[b] ** (4 * 1) / 720.0  # np.math.factorial(6)
            c += theta[b] ** (4 * 2) / 3628800.0  # np.math.factorial(6+4)
            c -= theta[b] ** (2) / 24.0  # np.math.factorial(4)
            c -= theta[b] ** (2 + 4) / 40320.0  # np.math.factorial(4+4)
            one_minus_cos_div_theta_sqr_tensor[b] = c

        if theta_cube[b] > threshold_cube:
            theta_minus_sin_div_theta_cube_tensor[b] = theta_minus_sin_div_theta_cube[b]
        elif theta_sqr[b] < 1e-6:
            theta_minus_sin_div_theta_cube_tensor[b] = 0  # 1.0 / 6.0
        else:  # Taylor expansion
            s = 1.0 / 6.0
            s += theta[b] ** (4 * 1) / 5040.0
            s += theta[b] ** (4 * 2) / 39916800.0
            s -= theta[b] ** (2) / 120.0
            s -= theta[b] ** (2 + 4) / 362880.0
            theta_minus_sin_div_theta_cube_tensor[b] = s

    completeTransformation = torch.zeros(batchSize, 3, 3).cuda()

    completeTransformation[:, 0, 0] += 1
    completeTransformation[:, 1, 1] += 1
    completeTransformation[:, 2, 2] += 1

    sin_theta_div_theta_tensor = torch.unsqueeze(sin_theta_div_theta_tensor, dim=1)
    completeTransformation = completeTransformation + \
                             vecMulMat(sin_theta_div_theta_tensor, omega_skew) + \
                             torch.mul(one_minus_cos_div_theta_sqr_tensor, omega_skew_sqr)

    V = torch.zeros(batchSize, 3, 3).cuda()
    V[:, 0, 0] += 1
    V[:, 1, 1] += 1
    V[:, 2, 2] += 1
    V = V + torch.mul(one_minus_cos_div_theta_sqr_tensor, omega_skew) + \
        torch.mul(theta_minus_sin_div_theta_cube_tensor, omega_skew_sqr)
    return completeTransformation, V


def pose_vec2mat(vec):
    """
    :param vec: [B,6],[rho,omega]
    :return:
    """
    rho = vec[:, 0:3]
    omega = vec[:, 3:6]  # torch.Size([batchSize, 3])
    R, V = so3_RV(omega)
    tra = torch.bmm(V, rho.unsqueeze(dim=-1))
    transform_mat = torch.cat([R, tra], dim=2)  # [B, 3, 4]
    return transform_mat

def inverse_warp(img, depth,ref_depth, pose, intrinsics, padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.
    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, 1, H, W]
        ref_depth: the source depth map (where to sample depth) -- [B, 1, H, W]
        pose: 6DoF pose parameters from target to source -- [B,3,4]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        valid_mask: Float array indicating point validity
    """
    check_sizes(img, 'img', 'B3HW')
    check_sizes(depth, 'depth', 'B1HW')
    check_sizes(ref_depth, 'ref_depth', 'B1HW')
    check_sizes(pose, 'pose', 'B34')
    check_sizes(intrinsics, 'intrinsics', 'B33')

    batch_size, _, img_height, img_width = img.size()

    cam_coords = pixel2cam(depth.squeeze(1), intrinsics.inverse())  # [B,3,H,W]

    # pose_mat = pose_vec2mat(pose)  # [B,3,4]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose  # [B, 3, 4]
    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
    src_pixel_coords, computed_depth,dp_flow = cam2pixel(cam_coords, rot, tr, padding_mode)  # [B,H,W,2]
    projected_img = F.grid_sample(img, src_pixel_coords, padding_mode=padding_mode,align_corners=False)


    valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1

    valid_mask = valid_points.unsqueeze(1).float()

    projected_depth = F.grid_sample(ref_depth, src_pixel_coords, padding_mode=padding_mode,align_corners=False).clamp(min=1e-3)

    #dp_flow_rig = get_dp_flow(src_pixel_coords)

    return projected_img, valid_mask, projected_depth, computed_depth, dp_flow

def flow_warp(img,flow,flow_inv=None,ref_depth=None,padding_mode='zeros'):
    """
        Inverse warp a source image to the target image plane.

        Args:
            img: the source image (where to sample pixels) -- [B, 3, H, W]
            flow: flow map of the target image -- [B, 2, H, W]
             padding_mode (str): padding mode for outside grid values
            ``'zeros'`` | ``'border'`` | ``'reflection'``. Default: ``'zeros'``
        Returns:
            Source image warped to the target image plane
        """
    check_sizes(img, 'img', 'BCHW')
    check_sizes(flow, 'flow', 'B2HW')
    bs, _,h,w = flow.size()
    u = flow[:,0,:,:]
    v = flow[:,1,:,:]
    grid_x = torch.arange(0,w).view(1,1,w).expand(1,h,w).type_as(u).expand_as(u) # [bs, H, W]
    grid_y = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(v).expand_as(v)  # [bs, H, W]
    X = grid_x + u
    Y = grid_y + v

    X = 2. * (X / (w - 1.0) - 0.5)
    Y = 2. * (Y / (h - 1.0) - 0.5)
    grid_tf = torch.stack((X, Y), dim=3)

    img_tf = torch.nn.functional.grid_sample(img, grid_tf, padding_mode=padding_mode,align_corners=False)
    valid_points = grid_tf.abs().max(dim=-1)[0] <= 1

    valid_mask = valid_points.unsqueeze(1).float()
    if (flow_inv is not None) and (ref_depth is not None):
        with torch.no_grad():
            # apply backward flow to get occlusion map
            base_grid = torch.stack([grid_x, grid_y], 1) #B2HW
            corr_map = get_corresponding_map(base_grid + flow_inv)  # BHW
        # soft mask, 0 means occlusion, 1 means empty
        occu_mask = corr_map.clamp(min=0., max=1.)
        f_fb_bw_warped = torch.nn.functional.grid_sample(flow_inv, grid_tf, padding_mode=padding_mode,align_corners=False)
        projected_depth = torch.nn.functional.grid_sample(ref_depth, grid_tf, padding_mode=padding_mode,align_corners=False)
        return img_tf,f_fb_bw_warped,projected_depth,valid_mask, occu_mask
    elif (flow_inv is not None):
        with torch.no_grad():
            # apply backward flow to get occlusion map
            base_grid = torch.stack([grid_x, grid_y], 1) #B2HW
            corr_map = get_corresponding_map(base_grid + flow_inv)  # BHW
        # soft mask, 0 means occlusion, 1 means empty
        occu_mask = corr_map.clamp(min=0., max=1.)
        f_fb_bw_warped = torch.nn.functional.grid_sample(flow_inv, grid_tf, padding_mode=padding_mode,align_corners=False)
        return img_tf,f_fb_bw_warped,valid_mask, occu_mask
        #valid_mask = 1 - (img_tf == 0).prod(1, keepdim=True).type_as(img_tf)
    else:
        return img_tf,valid_mask

def get_corresponding_map(data):
    """

    :param data: unnormalized coordinates Bx2xHxW
    :return: Bx1xHxW
    """
    B, _, H, W = data.size()

    # x = data[:, 0, :, :].view(B, -1).clamp(0, W - 1)  # BxN (N=H*W)
    # y = data[:, 1, :, :].view(B, -1).clamp(0, H - 1)

    x = data[:, 0, :, :].view(B, -1)  # BxN (N=H*W)
    y = data[:, 1, :, :].view(B, -1)

    # invalid = (x < 0) | (x > W - 1) | (y < 0) | (y > H - 1)   # BxN
    # invalid = invalid.repeat([1, 4])

    x1 = torch.floor(x)
    x_floor = x1.clamp(0, W - 1)
    y1 = torch.floor(y)
    y_floor = y1.clamp(0, H - 1)
    x0 = x1 + 1
    x_ceil = x0.clamp(0, W - 1)
    y0 = y1 + 1
    y_ceil = y0.clamp(0, H - 1)

    x_ceil_out = x0 != x_ceil
    y_ceil_out = y0 != y_ceil
    x_floor_out = x1 != x_floor
    y_floor_out = y1 != y_floor
    invalid = torch.cat([x_ceil_out | y_ceil_out,
                         x_ceil_out | y_floor_out,
                         x_floor_out | y_ceil_out,
                         x_floor_out | y_floor_out], dim=1)

    # encode coordinates, since the scatter function can only index along one axis
    corresponding_map = torch.zeros(B, H * W).type_as(data)
    indices = torch.cat([x_ceil + y_ceil * W,
                         x_ceil + y_floor * W,
                         x_floor + y_ceil * W,
                         x_floor + y_floor * W], 1).long()  # BxN   (N=4*H*W)
    values = torch.cat([(1 - torch.abs(x - x_ceil)) * (1 - torch.abs(y - y_ceil)),
                        (1 - torch.abs(x - x_ceil)) * (1 - torch.abs(y - y_floor)),
                        (1 - torch.abs(x - x_floor)) * (1 - torch.abs(y - y_ceil)),
                        (1 - torch.abs(x - x_floor)) * (1 - torch.abs(y - y_floor))],
                       1)

    values[invalid] = 0

    corresponding_map.scatter_add_(1, indices, values)
    # decode coordinates
    corresponding_map = corresponding_map.view(B, H, W)

    return corresponding_map.unsqueeze(1)

########################################################################################
def pose2trifocal_tensor(P21,P31):
    C2 = pose_vec2mat(P21)
    C3 = pose_vec2mat(P31)
    b,_,_ = C2.size()
    T1 = (C2[:,:,0].view(b,-1,1) @ C3[:,:,3].view(b,1,-1))-(C2[:,:,3].view(b,-1,1) @ C3[:,:,0].view(b,1,-1))
    T2 = (C2[:,:, 1].view(b,-1, 1) @ C3[:,:, 3].view(b,1, -1)) - (C2[:,:, 3].view(b,-1, 1) @ C3[:,:, 1].view(b,1, -1))
    T3 = (C2[:,:, 2].view(b,-1, 1) @ C3[:,:, 3].view(b,1, -1)) - (C2[:,:, 3].view(b,-1, 1) @ C3[:,:, 2].view(b,1, -1))
    return T1,T2,T3

def pose2img(P12,P13,intrinsics,depth):
    global pixel_coords
    b,_,h,w = depth.size()
    #print(intrinsics.shape)
    # M1 = intrinsics.inverse() @ pixel_coords

    current_pixel_coords = pixel_coords[:, :, :h, :w].expand(b, 3, h, w).reshape(b, 3, -1)  # [B, 3, H*W]
    M1 = (intrinsics.inverse() @ current_pixel_coords).reshape(b, 3, h, w)
    M2,mask2 = cam2pi(P12,depth.squeeze(1),intrinsics)
    M2 = (intrinsics.inverse() @ M2).reshape(b, h, w,3)
    M3,mask3 = cam2pi(P13,depth.squeeze(1),intrinsics)
    M3 = (intrinsics.inverse() @ M3).reshape(b, h, w,3)
   #  M1 = anti_matrix(M1)
   #  M2_anti = anti_matrix(M2)
   #  M3_anti = anti_matrix(M3)
    return M1,M2,M3,mask2,mask3

def  anti_matrix(M):
    b,h,w,_ = M.size()

    A_M = torch.zeros((b,h,w,3,3))
    A_M[:,:,:, 0 ,1] =-M[:,:,:,2]
    A_M[:, :, :, 0, 2] = M[:, :, :, 1]
    A_M[:, :, :, 1, 0] = M[:, :, :, 2]
    A_M[:, :, :, 1, 2] = -M[:, :, :, 0]
    A_M[:, :, :, 2, 0] = -M[:, :, :, 1]
    A_M[:, :, :, 2, 1] = M[:, :, :, 0]
    return A_M

def cam2pi(pose,depth,intrinsics):
    b , h, w = depth.size()
    cam_coords = pixel2cam(depth.squeeze(1), intrinsics.inverse())  # [B,3,H,W]
    pose_mat = pose_vec2mat(pose)  # [B,3,4]
    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose_mat  # [B, 3, 4]
    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
    if rot is not None:
        pcoords = rot @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if tr is not None:
        pcoords = pcoords + tr

    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    X_Norm = X / Z
    Y_Norm = Y / Z
    Z_Norm = Z / Z
    X_norm = 2 * (X_Norm) / (w - 1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2 * (Y_Norm) / (h - 1) - 1  # Idem [B, H*W]
    pixel_coords = torch.stack([X_norm, Y_norm], dim=2).view(b, h, w, 2)  # [B, H*W, 2]
    valid_points = pixel_coords.abs().max(dim=-1)[0] <= 1
    return torch.stack((X_Norm,Y_Norm,Z_Norm),dim=-2),valid_points.view(b, h, w).float()


def flow2img(flow12, flow13, intrinsics):
    global pixel_coords
    b, _, h, w = flow12.size()
    # print(intrinsics.shape)
    # M1 = intrinsics.inverse() @ pixel_coords

    M1 = pixel_coords[:, :, :h, :w].expand(b, 3, h, w).reshape(b, 3, -1)  # [B, 3, H*W]
    M2,mask2 = flow2pi(flow12)
    M3,mask3 = flow2pi(flow13)
    intrinsics_inv = intrinsics.inverse()
    M1 = (intrinsics_inv @ M1).reshape(b, h, w, 3)
    M2 = (intrinsics_inv @ M2).reshape(b, h, w, 3)
    M3 = (intrinsics_inv @ M3).reshape(b, h, w, 3)
    #  M1 = anti_matrix(M1)
    #  M2_anti = anti_matrix(M2)
    #  M3_anti = anti_matrix(M3)
    return M1, M2, M3, mask2, mask3

def flow2pi(flow,padding_mode='zeros'):
    """
        Inverse warp a source image to the target image plane.

        Args:
            img: the source image (where to sample pixels) -- [B, 3, H, W]
            flow: flow map of the target image -- [B, 2, H, W]
             padding_mode (str): padding mode for outside grid values
            ``'zeros'`` | ``'border'`` | ``'reflection'``. Default: ``'zeros'``
        Returns:
            Source image warped to the target image plane
        """
    check_sizes(flow, 'flow', 'B2HW')
    b, _,h,w = flow.size()
    u = flow[:,0,:,:]
    v = flow[:,1,:,:]
    grid_x = torch.arange(0,w).view(1,1,w).expand(1,h,w).type_as(u).expand_as(u) # [bs, H, W]
    grid_y = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(v).expand_as(v)  # [bs, H, W]
    Z = torch.ones(b, h, w).type_as(flow)
    X = grid_x + u
    Y = grid_y + v

    X_n = 2. * (X / (w - 1.0) - 0.5)
    Y_n = 2. * (Y / (h - 1.0) - 0.5)
    grid_tf = torch.stack((X_n, Y_n), dim=3)

    #img_tf = torch.nn.functional.grid_sample(img, grid_tf, padding_mode=padding_mode)
    valid_points = grid_tf.abs().max(dim=-1)[0] <= 1
    valid_mask = valid_points.unsqueeze(1).float()
    #valid_mask = 1 - (img_tf == 0).prod(1, keepdim=True).type_as(img_tf)
    return torch.stack((X.view(b,h*w),Y.view(b,h*w),Z.view(b,h*w)),dim=-2),valid_mask