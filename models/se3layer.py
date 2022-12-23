import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn import Module

'''
   --------------------- Converts SE3s to (R,t) ------------------------------
   SE3ToRt(transform_type, has_pivot) :
   SE3ToRt.forward(input)
   SE3ToRt.backward(grad_output)

   SE3ToRt layer takes in [batchSize x nSE3 x p] values and converts them to [B x N x 3 x 4] or [B x N x 3 x 5] matrix where each transform has:
	"R", a 3x3 affine matrix and "t", a 3x1 translation vector and optionally, a 3x1 pivot vector.
	The layer takes SE3s in different forms and converts each to R" & "t" & "p" based on the type of transform. 
	Parameters are always shaped as: [trans, rot, pivot] where the pivot is optional.
	1) affine 	- Parameterized by 12 values (12-dim vector). "params" need to be 12 values which will be reshaped as a (3x4) matrix.
					  Input parameters are shaped as: [B x k x 12] matrix
	2) se3euler - SE3 transform with an euler angle parameterization for rotation.
					  "params" are 3 translational and 3 rotational (xyz-euler) parameters = [tx,ty,tz,r1,r2,r3].
					  The rotational parameters are converted to the rotation matrix form using the following parameterization:
					  We take the triplet [r1,r2,r3], all in radians and compute a rotation matrix R which is decomposed as:
					  R = Rz(r3) * Ry(r2) * Rx(r1) where Rz,Ry,Rx are rotation matrices around z,y,x axes with angles of r3,r2,r1 respectively.
					  Translational parameters [tx,ty,tz] are the translations along the X,Y,Z axes in m	
					  Input parameters are shaped as: [B x k x 6] matrix	
	3) se3aa 	- SE3 transform with an axis-angle parameterization for rotation. 
					  "params" are 3 translational and 3 rotational (axis-angle) parameters = [tx,ty,tz,r1,r2,r3].
					  The rotational parameters are converted to the rotation matrix form using the following parameterization:
							Rotation angle = || [r1,r2,r3] ||_2 which is the 2-norm of the vector
							Rotation axis  = [r1,r2,r3]/angle   which is the unit vector
					  		These are then converted to a rotation matrix (R) using the Rodriguez's transform
					  Translational parameters [tx,ty,tz] are the translations along the X,Y,Z axes in m	
					  Input parameters are shaped as: [B x k x 6] matrix	
	4) se3quat  - SE3 transform with a quaternion parameterization for the rotation.
					  "params" are 3 translational and 4 rotational (quaternion) parameters = [tx,ty,tz,qx,qy,qz,qw].
					  The rotational parameters are converted to the rotation matrix form using the following parameterization:
							Unit Quaternion = [qx,qy,qz,qw] / || [qx,qy,qz,qw] ||_2  
					  		These are then converted to a rotation matrix (R) using the quaternion to rotation transform
					  Translational parameters [tx,ty,tz] are the translations along the X,Y,Z axes in m	
					  Input parameters are shaped as: [B x k x 7] matrix	
   5) se3spquat - SE3 transform with a stereographic projection of a quaternion as the parameterization for the rotation.
                 "params" are 3 translational and 3 rotational (SP-quaternion) parameters = [tx,ty,tz,sx,sy,sz].
					  The rotational parameters are converted to the rotation matrix form using the following parameterization:
							SP Quaternion -> Quaternion -> Unit Quaternion -> Rotation Matrix 
					  Translational parameters [tx,ty,tz] are the translations along the X,Y,Z axes in m	
					  Input parameters are shaped as: [B x k x 6] matrix	. For more details on this parameterization, check out: 
                 "A Recipe on the Parameterization of Rotation Matrices for Non-Linear Optimization using Quaternions" &
                 https://github.com/FugroRoames/Rotations.jl
	By default, transformtype is set to "affine"
'''

## FWD/BWD pass function
class SE3ToRtFunction(Function):

    ###########################
    #### Forward pass
    @staticmethod
    def forward(ctx, input,transform_type,has_pivot,eps):
        # Check dimensions
        # self.check(input)  # Check size
        batch_size, num_se3, num_params = input.size()
        tot_se3 = batch_size * num_se3
        rot_dim = 4 if (transform_type == 'se3quat') else 9 if (transform_type == 'affine') else 3  # Number of rotation parameters

        # Init memory
        num_cols = 5 if (has_pivot) else 4
        output = input.new().resize_(batch_size, num_se3, 3, num_cols)
        outputv = output.view(tot_se3, 3, num_cols)

        def create_skew_symmetric_matrix(vector):
            # Create the skew symmetric matrix:
            # [0 -z y; z 0 -x; -y x 0]
            N = vector.size(0)
            vec = vector.contiguous().view(N, 3)
            output = vec.new().resize_(N, 3, 3).fill_(0)
            output[:, 0, 1] = -vec[:, 2]
            output[:, 1, 0] = vec[:, 2]
            output[:, 0, 2] = vec[:, 1]
            output[:, 2, 0] = -vec[:, 1]
            output[:, 1, 2] = -vec[:, 0]
            output[:, 2, 1] = vec[:, 0]
            return output
        ####
        # Affine transform: Just reshape things
        if (transform_type == 'affine'):
            output = input.view(batch_size, num_se3, 3, num_cols).contiguous()
            # We are done
            ctx.save_for_backward(input, output,transform_type,has_pivot,eps)
            return output

            # # Create output
            # output[:,:,:,3]   = input[:,:,0:3] # Translation (3x1)
            # output[:,:,:,0:3] = input[:,:,3:3+rot_dim].view(batch_size, num_se3, 3, 3) # Rotation (3x3)
            # #output.narrow(3, 3, 1).copy_(input.narrow(2, 0, 3))
            # #output.narrow(3, 0, 3).copy_(input.narrow(2, 3, rot_dim))
            # if self.has_pivot:
            #     output[:,:,:,4] = input[:,:,3+rot_dim:] # Pivot (3x1)
            #     #output.narrow(3, 4, 1).copy_(input.narrow(2, 3 + rot_dim, 3))  # Pivot (3x1)
            # # Return, we are done
            # return output

        ####
        # Create rotation matrix based on the SE3 types (3x3)
        # Other transform types (se3euler, se3aa, se3quat, se3spquat)
        params = input.view(tot_se3, -1)  # Bk x num_params
        rot_params = params.narrow(1, 3, rot_dim)  # Bk x rot_dim
        if (transform_type == 'se3euler'):  # parameters are [dx, dy, dz, theta1, theta2, theta3]
            # Create rotations about X,Y,Z axes
            # R = Rz(theta3) * Ry(theta2) * Rx(theta1)
            # Last 3 parameters are [theta1, theta2 ,theta3]
            # Rotation about the X-axis by theta
            # From Barfoot's book: http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser15.pdf (6.7)
            def create_rotx(theta):
                N = theta.size(0)
                thetas = theta.squeeze()
                rot = torch.eye(3).view(1, 3, 3).repeat(N, 1, 1).type_as(
                    thetas)  # (DO NOT use expand as it does not allocate new memory)
                rot[:, 1, 1] = torch.cos(thetas)
                rot[:, 2, 2] = rot[:, 1, 1]
                rot[:, 1, 2] = torch.sin(thetas)
                rot[:, 2, 1] = -rot[:, 1, 2]
                return rot

            # Rotation about the Y-axis by theta
            # From Barfoot's book: http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser15.pdf (6.6)
            def create_roty(theta):
                N = theta.size(0)
                thetas = theta.squeeze()
                rot = torch.eye(3).view(1, 3, 3).repeat(N, 1, 1).type_as(
                    thetas)  # (DO NOT use expand as it does not allocate new memory)
                rot[:, 0, 0] = torch.cos(thetas)
                rot[:, 2, 2] = rot[:, 0, 0]
                rot[:, 2, 0] = torch.sin(thetas)
                rot[:, 0, 2] = -rot[:, 2, 0]
                return rot

            # Rotation about the Z-axis by theta
            # From Barfoot's book: http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser15.pdf (6.5)
            def create_rotz(theta):
                N = theta.size(0)
                thetas = theta.squeeze()
                rot = torch.eye(3).view(1, 3, 3).repeat(N, 1, 1).type_as(
                    theta)  # (DO NOT use expand as it does not allocate new memory)
                rot[:, 0, 0] = torch.cos(thetas)
                rot[:, 1, 1] = rot[:, 0, 0]
                rot[:, 0, 1] = torch.sin(thetas)
                rot[:, 1, 0] = -rot[:, 0, 1]
                return rot
            rotx = create_rotx(rot_params.narrow(1, 0, 1))  # Rx(theta1)
            roty = create_roty(rot_params.narrow(1, 1, 1))  # Ry(theta2)
            rotz = create_rotz(rot_params.narrow(1, 2, 1))  # Rz(theta3)

            # Compute Rz(theta3) * Ry(theta2)
            rotzy = torch.bmm(rotz, roty)  # Rzy = R32

            # Compute rotation matrix R3*R2*R1 = R32*R1
            # R = Rz(t3) * Ry(t2) * Rx(t1)
            torch.bmm(rotzy, rotx, out=outputv.narrow(2, 0, 3))  # R = Rzyx

        elif (transform_type == 'se3aa'):
            # Create rot from aa
            # Compute the rotation matrix R from the axis-angle parameters using Rodriguez's formula:
            # (R = I + (sin(theta)/theta) * K + ((1-cos(theta))/theta^2) * K^2)
            # where K is the skew symmetric matrix based on the un-normalized axis & theta is the norm of the input parameters
            # From Wikipedia: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
            def create_rot_from_aa(params,eps):
                # Get the un-normalized axis and angle
                N = params.size(0)
                axis = params.clone().view(N, 3, 1)  # Un-normalized axis
                angle2 = (axis * axis).sum(1).view(N, 1, 1)  # Norm of vector (squared angle)
                angle = torch.sqrt(angle2)  # Angle

                # Compute skew-symmetric matrix "K" from the axis of rotation
                K = create_skew_symmetric_matrix(axis)
                K2 = torch.bmm(K, K)  # K * K

                # Compute sines
                S = torch.sin(angle) / angle
                S.masked_fill_(angle2.lt(eps), 1)  # sin(0)/0 ~= 1

                # Compute cosines
                C = (1 - torch.cos(angle)) / angle2
                C.masked_fill_(angle2.lt(eps), 0)  # (1 - cos(0))/0^2 ~= 0

                # Compute the rotation matrix: R = I + (sin(theta)/theta)*K + ((1-cos(theta))/theta^2) * K^2
                rot = torch.eye(3).view(1, 3, 3).repeat(N, 1, 1).type_as(
                    params)  # R = I (avoid use expand as it does not allocate new memory)
                rot += K * S.expand(N, 3, 3)  # R = I + (sin(theta)/theta)*K
                rot += K2 * C.expand(N, 3, 3)  # R = I + (sin(theta)/theta)*K + ((1-cos(theta))/theta^2)*K^2
                return rot
            outputv.narrow(2, 0, 3).copy_(create_rot_from_aa(rot_params,eps))
        elif (transform_type == 'se3quat'):
            # Compute the unit quaternion
            unitquat = F.normalize(rot_params, p=2, dim=1, eps=1e-12) # self.create_unitquat_from_quat(rot_params)

            # Compute rotation matrix from unit quaternion
            # Compute the rotation matrix R from a set of unit-quaternions (N x 4):
            # From: http://www.tech.plymouth.ac.uk/sme/springerusv/2011/publications_files/Terzakis%20et%20al%202012,%20A%20Recipe%20on%20the%20Parameterization%20of%20Rotation%20Matrices...MIDAS.SME.2012.TR.004.pdf (Eqn 9)
            def create_rot_from_unitquat(unitquat):
                # Init memory
                N = unitquat.size(0)
                rot = unitquat.new().resize_(N, 3, 3)

                # Get quaternion elements. Quat = [qx,qy,qz,qw] with the scalar at the rear
                x, y, z, w = unitquat[:, 0], unitquat[:, 1], unitquat[:, 2], unitquat[:, 3]
                x2, y2, z2, w2 = x * x, y * y, z * z, w * w

                # Row 1
                rot[:, 0, 0] = w2 + x2 - y2 - z2  # rot(0,0) = w^2 + x^2 - y^2 - z^2
                rot[:, 0, 1] = 2 * (x * y - w * z)  # rot(0,1) = 2*x*y - 2*w*z
                rot[:, 0, 2] = 2 * (x * z + w * y)  # rot(0,2) = 2*x*z + 2*w*y

                # Row 2
                rot[:, 1, 0] = 2 * (x * y + w * z)  # rot(1,0) = 2*x*y + 2*w*z
                rot[:, 1, 1] = w2 - x2 + y2 - z2  # rot(1,1) = w^2 - x^2 + y^2 - z^2
                rot[:, 1, 2] = 2 * (y * z - w * x)  # rot(1,2) = 2*y*z - 2*w*x

                # Row 3
                rot[:, 2, 0] = 2 * (x * z - w * y)  # rot(2,0) = 2*x*z - 2*w*y
                rot[:, 2, 1] = 2 * (y * z + w * x)  # rot(2,1) = 2*y*z + 2*w*x
                rot[:, 2, 2] = w2 - x2 - y2 + z2  # rot(2,2) = w^2 - x^2 - y^2 + z^2

                # Return
                return rot
            outputv.narrow(2, 0, 3).copy_(create_rot_from_unitquat(unitquat))
        elif (transform_type == 'se3spquat'):
            def create_unitquat_from_spquat(spquat):
                # Init memory
                N = spquat.size(0)
                unitquat = spquat.new().resize_(N, 4).fill_(0)

                # Compute the unit quaternion (qx, qy, qz, qw)
                x, y, z = spquat[:, 0], spquat[:, 1], spquat[:, 2]
                alpha2 = x * x + y * y + z * z  # x^2 + y^2 + z^2
                unitquat[:, 0] = (2 * x) / (1 + alpha2)  # qx
                unitquat[:, 1] = (2 * y) / (1 + alpha2)  # qy
                unitquat[:, 2] = (2 * z) / (1 + alpha2)  # qz
                unitquat[:, 3] = (1 - alpha2) / (1 + alpha2)  # qw
                return unitquat
            # Compute the unit quaternion
            unitquat = create_unitquat_from_spquat(rot_params)

            # Compute the rotation matrix R from a set of unit-quaternions (N x 4):
            # From: http://www.tech.plymouth.ac.uk/sme/springerusv/2011/publications_files/Terzakis%20et%20al%202012,%20A%20Recipe%20on%20the%20Parameterization%20of%20Rotation%20Matrices...MIDAS.SME.2012.TR.004.pdf (Eqn 9)
            def create_rot_from_unitquat(unitquat):
                # Init memory
                N = unitquat.size(0)
                rot = unitquat.new().resize_(N, 3, 3)

                # Get quaternion elements. Quat = [qx,qy,qz,qw] with the scalar at the rear
                x, y, z, w = unitquat[:, 0], unitquat[:, 1], unitquat[:, 2], unitquat[:, 3]
                x2, y2, z2, w2 = x * x, y * y, z * z, w * w

                # Row 1
                rot[:, 0, 0] = w2 + x2 - y2 - z2  # rot(0,0) = w^2 + x^2 - y^2 - z^2
                rot[:, 0, 1] = 2 * (x * y - w * z)  # rot(0,1) = 2*x*y - 2*w*z
                rot[:, 0, 2] = 2 * (x * z + w * y)  # rot(0,2) = 2*x*z + 2*w*y

                # Row 2
                rot[:, 1, 0] = 2 * (x * y + w * z)  # rot(1,0) = 2*x*y + 2*w*z
                rot[:, 1, 1] = w2 - x2 + y2 - z2  # rot(1,1) = w^2 - x^2 + y^2 - z^2
                rot[:, 1, 2] = 2 * (y * z - w * x)  # rot(1,2) = 2*y*z - 2*w*x

                # Row 3
                rot[:, 2, 0] = 2 * (x * z - w * y)  # rot(2,0) = 2*x*z - 2*w*y
                rot[:, 2, 1] = 2 * (y * z + w * x)  # rot(2,1) = 2*y*z + 2*w*x
                rot[:, 2, 2] = w2 - x2 - y2 + z2  # rot(2,2) = w^2 - x^2 - y^2 + z^2

                # Return
                return rot
            # Compute rotation matrix from unit quaternion
            outputv.narrow(2, 0, 3).copy_(create_rot_from_unitquat(unitquat))

        ####
        # Translation vector (3x1)
        outputv[:,:,3] = params[:,0:3]  # [tx,ty,tz] (B x k x 3)

        ####
        # Pivot vector (3x1)
        if has_pivot:
            outputv.narrow(2, 4, 1).copy_(params.narrow(1, 3 + rot_dim, 3))  # [px, py, pz] (B x k x 3)

        ####
        # Return
        ctx.save_for_backward(input, output)
        ctx.transform_type=transform_type
        ctx.has_pivot=has_pivot
        ctx.eps=eps
        return output

    ###########################
    #### Backward pass
    @staticmethod
    def backward(ctx, grad_output):
        # Check dimensions
        input, output = ctx.saved_tensors
        transform_type = ctx.transform_type
        has_pivot = ctx.has_pivot
        eps = ctx.eps
        def create_skew_symmetric_matrix(vector):
            # Create the skew symmetric matrix:
            # [0 -z y; z 0 -x; -y x 0]
            N = vector.size(0)
            vec = vector.contiguous().view(N, 3)
            output = vec.new().resize_(N, 3, 3).fill_(0)
            output[:, 0, 1] = -vec[:, 2]
            output[:, 1, 0] = vec[:, 2]
            output[:, 0, 2] = vec[:, 1]
            output[:, 2, 0] = -vec[:, 1]
            output[:, 1, 2] = -vec[:, 0]
            output[:, 2, 1] = vec[:, 0]
            return output
        # self.check(input, grad_output)  # Check size
        batch_size, num_se3, num_params = input.size()
        tot_se3 = batch_size * num_se3
        rot_dim = 4 if (transform_type == 'se3quat') else 9 if (transform_type == 'affine') else 3  # Number of rotation parameters

        # Init memory for grad input
        grad_input = input.new().resize_as_(input)
        grad_input_v = grad_input.view(tot_se3, -1)  # View it with (Bk) x num_params

        # Get grad output & input in correct shape
        num_cols = 5 if (has_pivot) else 4
        grad_output_v = grad_output.view(tot_se3, 3, num_cols)  # (Bk) x 3 x num_cols

        ####
        # Affine transform: [trans, rot, pivot]
        if (transform_type == 'affine'):
            grad_input = grad_output.view(batch_size, num_se3, 3*num_cols)
            return grad_input

            # # Create grad_input
            # grad_input_v[:,0:3]         = grad_output_v[:,:,3] # Translation (3x1)
            # grad_input_v[:,3:3+rot_dim] = grad_output_v[:,:,0:3].contiguous().view(batch_size*num_se3, rot_dim) # Rotation (3x3)
            # #grad_input_v.narrow(1, 0, 3).copy_(grad_output_v.narrow(2, 3, 1))  # Translation (3x1)
            # #grad_input_v.narrow(1, 3, rot_dim).copy_(grad_output_v.narrow(2, 0, 3))  # Rotation (3x3)
            # if self.has_pivot:
            #     grad_input_v[:,3+rot_dim:] = grad_output_v[:,:,4] # Pivot (3x1)
            #     #grad_input_v.narrow(1, 3 + rot_dim, 3).copy_(grad_output_v.narrow(2, 4, 1))  # Pivot (3x1)
            # return grad_input

        ####
        # Gradient w.r.t rotation parameters (different based on rotation type)
        # Other transform types (se3euler, se3aa, se3quat, se3spquat)
        params = input.view(tot_se3, -1)  # (Bk) x num_params
        rot_params = params.narrow(1, 3, rot_dim)  # (Bk) x rotdim (Last few parameters are the rotation parameters)
        grad_rot_params = grad_input_v.narrow(1, 3, rot_dim)  # (Bk) x rotdim (Last few parameters are the rotation parameters)
        rot = output.view(tot_se3, 3, num_cols).narrow(2, 0, 3)  # (Bk) x 3 x 3 => 3x3 rotation matrix
        if (transform_type == 'se3euler'):  # parameters are [dx, dy, dz, theta1, theta2, theta3]
            # Create rotations about X,Y,Z axes
            # R = Rz(theta3) * Ry(theta2) * Rx(theta1)
            # Last 3 parameters are [theta1, theta2 ,theta3]
            # Rotation about the X-axis by theta
            # From Barfoot's book: http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser15.pdf (6.7)
            def create_rotx(theta):
                N = theta.size(0)
                thetas = theta.squeeze()
                rot = torch.eye(3).view(1, 3, 3).repeat(N, 1, 1).type_as(
                    thetas)  # (DO NOT use expand as it does not allocate new memory)
                rot[:, 1, 1] = torch.cos(thetas)
                rot[:, 2, 2] = rot[:, 1, 1]
                rot[:, 1, 2] = torch.sin(thetas)
                rot[:, 2, 1] = -rot[:, 1, 2]
                return rot

            # Rotation about the Y-axis by theta
            # From Barfoot's book: http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser15.pdf (6.6)
            def create_roty(theta):
                N = theta.size(0)
                thetas = theta.squeeze()
                rot = torch.eye(3).view(1, 3, 3).repeat(N, 1, 1).type_as(
                    thetas)  # (DO NOT use expand as it does not allocate new memory)
                rot[:, 0, 0] = torch.cos(thetas)
                rot[:, 2, 2] = rot[:, 0, 0]
                rot[:, 2, 0] = torch.sin(thetas)
                rot[:, 0, 2] = -rot[:, 2, 0]
                return rot

            # Rotation about the Z-axis by theta
            # From Barfoot's book: http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser15.pdf (6.5)
            def create_rotz(theta):
                N = theta.size(0)
                thetas = theta.squeeze()
                rot = torch.eye(3).view(1, 3, 3).repeat(N, 1, 1).type_as(
                    theta)  # (DO NOT use expand as it does not allocate new memory)
                rot[:, 0, 0] = torch.cos(thetas)
                rot[:, 1, 1] = rot[:, 0, 0]
                rot[:, 0, 1] = torch.sin(thetas)
                rot[:, 1, 0] = -rot[:, 0, 1]
                return rot
            rotx = create_rotx(rot_params.narrow(1, 0, 1))  # Rx(theta1)
            roty = create_roty(rot_params.narrow(1, 1, 1))  # Ry(theta2)
            rotz = create_rotz(rot_params.narrow(1, 2, 1))  # Rz(theta3)

            # Compute Rz(theta3) * Ry(theta2)
            rotzy = torch.bmm(rotz, roty)  # Rzy = R32

            # Gradient w.r.t Euler angles from Barfoot's book (http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser15.pdf)
            for k in range(3):
                gradr = grad_rot_params[:,k]  # Gradient w.r.t angle (k)
                vec = torch.zeros(1, 3).type_as(gradr)
                vec[0][k] = 1  # Unit vector
                skewsym = create_skew_symmetric_matrix(vec).view(1, 3, 3).expand_as(rot)  # Skew symmetric matrix of unit vector
                if (k == 0):
                    Rv = torch.bmm(torch.bmm(rotzy, skewsym), rotx)  # Eqn 6.61c
                elif (k == 1):
                    Rv = torch.bmm(torch.bmm(rotz, skewsym), torch.bmm(roty, rotx))  # Eqn 6.61b
                else:
                    Rv = torch.bmm(skewsym, rot)
                gradr.copy_((-Rv * grad_output_v.narrow(2, 0, 3)).view(tot_se3,-1).sum(1))
        elif (transform_type == 'se3aa'):
            # Gradient w.r.t rotation parameters (last 3 parameters) from: http://arxiv.org/pdf/1312.0788v1.pdf (Eqn: III.7)
            # Get the unnormalized axis and angle
            axis = params.view(tot_se3, -1, 1).narrow(1, 3,
                                                      rot_dim)  # (Bk) x 3 x 1 => 3 unnormalized AA parameters (note that first 3 parameters are [t])
            angle2 = (axis * axis).sum(1)  # (Bk) x 1 x 1 => Norm of the vector (squared angle)
            nSmall = angle2.lt(eps).sum()  # Num angles less than threshold

            # Compute: v x (Id - R) for all the columns of (Id-R)
            I = torch.eye(3).type_as(rot).repeat(tot_se3, 1, 1).add(-1, rot)  # (Bk) x 3 x 3 => Id - R
            vI = torch.cross(axis.expand_as(I), I, 1)  # (Bk) x 3 x 3 => v x (Id - R)

            # Compute [v * v' + v x (Id - R)] / ||v||^2
            vV = torch.bmm(axis, axis.transpose(1, 2))  # (Bk) x 3 x 3 => v * v'
            vV = (vV + vI) / (angle2.view(tot_se3,1,1).expand_as(vV))  # (Bk) x 3 x 3 => [v * v' + v x (Id - R)] / ||v||^2

            # Iterate over the 3-axis angle parameters to compute their gradients
            # ([v * v' + v x (Id - R)] / ||v||^2 _ k) x (R) .* gradOutput  where "x" is the cross product
            def create_skew_symmetric_matrix(vector):
                # Create the skew symmetric matrix:
                # [0 -z y; z 0 -x; -y x 0]
                N = vector.size(0)
                vec = vector.contiguous().view(N, 3)
                output = vec.new().resize_(N, 3, 3).fill_(0)
                output[:, 0, 1] = -vec[:, 2]
                output[:, 1, 0] = vec[:, 2]
                output[:, 0, 2] = vec[:, 1]
                output[:, 2, 0] = -vec[:, 1]
                output[:, 1, 2] = -vec[:, 0]
                output[:, 2, 1] = vec[:, 0]
                return output
            for k in range(3):
                # Create skew symmetric matrix
                skewsym = create_skew_symmetric_matrix(vV.narrow(2, k, 1))

                # For those AAs with angle^2 < threshold, gradient is different
                # We assume angle = 0 for these AAs and update the skew-symmetric matrix to be one w.r.t identity
                if (nSmall > 0):
                    vec = torch.zeros(1, 3).type_as(skewsym)
                    vec[0][k] = 1  # Unit vector
                    idskewsym = create_skew_symmetric_matrix(vec)
                    for i in range(tot_se3):
                        if (angle2[i].squeeze()[0] < eps):
                            skewsym[i].copy_(idskewsym.squeeze())  # Use the new skew sym matrix (around identity)

                # Compute the gradients now
                out = (torch.bmm(skewsym, rot) * grad_output_v.narrow(2, 0, 3)).sum(2).sum(1)  # [(Bk) x 1 x 1] => (vV x R) .* gradOutput
                grad_rot_params[:,k] = out
        elif (transform_type == 'se3quat'):
            # Compute the unit quaternion
            quat = rot_params
            unitquat = F.normalize(quat, p=2, dim=1, eps=1e-12) #self.create_unitquat_from_quat(quat)

            # We need gradients of the rotation matrix (R) w.r.t the unit-quaternion (q')
            # Compute dR/dq'
            def compute_grad_rot_wrt_unitquat(unitquat):
                # Compute dR/dq' (9x4 matrix)
                N = unitquat.size(0)
                x, y, z, w = unitquat.narrow(1, 0, 1), unitquat.narrow(1, 1, 1), unitquat.narrow(1, 2,
                                                                                                 1), unitquat.narrow(1,
                                                                                                                     3,
                                                                                                                     1)
                dRdqh_w = 2 * torch.cat([w, -z, y, z, w, -x, -y, x, w], 1).view(N, 9, 1)  # Eqn 33, rows first
                dRdqh_x = 2 * torch.cat([x, y, z, y, -x, -w, z, w, -x], 1).view(N, 9, 1)  # Eqn 34, rows first
                dRdqh_y = 2 * torch.cat([-y, x, w, x, y, z, -w, z, -y], 1).view(N, 9, 1)  # Eqn 35, rows first
                dRdqh_z = 2 * torch.cat([-z, -w, x, w, -z, y, x, y, z], 1).view(N, 9, 1)  # Eqn 36, rows first
                dRdqh = torch.cat([dRdqh_x, dRdqh_y, dRdqh_z, dRdqh_w], 2)  # N x 9 x 4
                return dRdqh
            dRdqh = compute_grad_rot_wrt_unitquat(unitquat)

            # Compute dq'/dq = d(q/||q||)/dq = 1/||q|| (I - q'q'^T)
            def compute_grad_unitquat_wrt_quat(unitquat, quat):
                # Compute the quaternion norms
                N = quat.size(0)
                unitquat_v = unitquat.view(-1, 4, 1)
                norm2 = (quat * quat).sum(1)  # Norm-squared
                norm = torch.sqrt(norm2)  # Length of the quaternion

                # Compute gradient dq'/dq
                # TODO: No check for normalization issues currently
                I = torch.eye(4).view(1, 4, 4).expand(N, 4, 4).type_as(quat)
                qQ = torch.bmm(unitquat_v, unitquat_v.transpose(1, 2))  # q'*q'^T
                dqhdq = (I - qQ) / (norm.view(N, 1, 1).expand_as(I))

                # Return
                return dqhdq
            dqhdq = compute_grad_unitquat_wrt_quat(unitquat, quat)

            # Compute dR/dq = dR/dq' * dq'/dq
            dRdq = torch.bmm(dRdqh, dqhdq).view(tot_se3, 3, 3, 4)  # (Bk) x 3 x 3 x 4

            # Scale by grad w.r.t output and sum to get gradient w.r.t quaternion params
            grad_out = grad_output.view(tot_se3, 3, num_cols, 1).narrow(2, 0, 3).expand_as(dRdq)  # (Bk) x 3 x 3 x 4
            grad_rot_params.copy_((dRdq * grad_out).view(tot_se3, -1, 4).sum(1))  # (Bk) x 4
        elif (transform_type == 'se3spquat'):
            def create_unitquat_from_spquat(spquat):
                # Init memory
                N = spquat.size(0)
                unitquat = spquat.new().resize_(N, 4).fill_(0)

                # Compute the unit quaternion (qx, qy, qz, qw)
                x, y, z = spquat[:, 0], spquat[:, 1], spquat[:, 2]
                alpha2 = x * x + y * y + z * z  # x^2 + y^2 + z^2
                unitquat[:, 0] = (2 * x) / (1 + alpha2)  # qx
                unitquat[:, 1] = (2 * y) / (1 + alpha2)  # qy
                unitquat[:, 2] = (2 * z) / (1 + alpha2)  # qz
                unitquat[:, 3] = (1 - alpha2) / (1 + alpha2)  # qw
                return unitquat
            # Compute the unit quaternion
            spquat = rot_params
            unitquat = create_unitquat_from_spquat(spquat)

            # We need gradients of the rotation matrix (R) w.r.t the unit-quaternion (q')
            # Compute dR/dq'
            def compute_grad_rot_wrt_unitquat(unitquat):
                # Compute dR/dq' (9x4 matrix)
                N = unitquat.size(0)
                x, y, z, w = unitquat.narrow(1, 0, 1), unitquat.narrow(1, 1, 1), unitquat.narrow(1, 2,
                                                                                                 1), unitquat.narrow(1,
                                                                                                                     3,
                                                                                                                     1)
                dRdqh_w = 2 * torch.cat([w, -z, y, z, w, -x, -y, x, w], 1).view(N, 9, 1)  # Eqn 33, rows first
                dRdqh_x = 2 * torch.cat([x, y, z, y, -x, -w, z, w, -x], 1).view(N, 9, 1)  # Eqn 34, rows first
                dRdqh_y = 2 * torch.cat([-y, x, w, x, y, z, -w, z, -y], 1).view(N, 9, 1)  # Eqn 35, rows first
                dRdqh_z = 2 * torch.cat([-z, -w, x, w, -z, y, x, y, z], 1).view(N, 9, 1)  # Eqn 36, rows first
                dRdqh = torch.cat([dRdqh_x, dRdqh_y, dRdqh_z, dRdqh_w], 2)  # N x 9 x 4
                return dRdqh
            dRdqh = compute_grad_rot_wrt_unitquat(unitquat)

            # Compute dq'/dq = d(q/||q||)/dq = 1/||q|| (I - q'q'^T)
            def compute_grad_unitquat_wrt_spquat(spquat):
                # Compute scalars
                N = spquat.size(0)
                x, y, z = spquat.narrow(1, 0, 1), spquat.narrow(1, 1, 1), spquat.narrow(1, 2, 1)
                x2, y2, z2 = x * x, y * y, z * z
                s = 1 + x2 + y2 + z2  # 1 + x^2 + y^2 + z^2 = 1 + alpha^2
                s2 = (s * s).expand(N, 4)  # (1 + alpha^2)^2

                # Compute gradient dq'/dspq
                dqhdspq_x = (torch.cat([2 * s - 4 * x2, -4 * x * y, -4 * x * z, -4 * x], 1) / s2).view(N, 4,
                                                                                                       1)  # Eqn 48, order of elements: 2,3,4,1
                dqhdspq_y = (torch.cat([-4 * x * y, 2 * s - 4 * y2, -4 * y * z, -4 * y], 1) / s2).view(N, 4,
                                                                                                       1)  # Eqn 49, order of elements: 2,3,4,1
                dqhdspq_z = (torch.cat([-4 * x * z, -4 * y * z, 2 * s - 4 * z2, -4 * z], 1) / s2).view(N, 4,
                                                                                                       1)  # Eqn 50, order of elements: 2,3,4,1
                dqhdspq = torch.cat([dqhdspq_x, dqhdspq_y, dqhdspq_z], 2)

                # Return
                return dqhdspq
            dqhdspq = compute_grad_unitquat_wrt_spquat(spquat)

            # Compute dR/dq = dR/dq' * dq'/dq
            dRdq = torch.bmm(dRdqh, dqhdspq).view(tot_se3, 3, 3, 3)  # (Bk) x 3 x 3 x 3

            # Scale by grad w.r.t output and sum to get gradient w.r.t quaternion params
            grad_out = grad_output.view(tot_se3, 3, num_cols, 1).narrow(2, 0, 3).expand_as(dRdq)  # (Bk) x 3 x 3 x 3
            grad_rot_params.copy_((dRdq * grad_out).sum(1).sum(2))  # (Bk) x 3

        ####
        # Gradient w.r.t translation vector (3x1)
        grad_input_v[:,0:3] = grad_output_v[:,:,3]  # [tx,ty,tz] (Bk x 3 x 1)

        ####
        # Gradient w.r.t pivot vector (3x1)
        if has_pivot:
            grad_input_v.narrow(1, 3 + rot_dim, 3).copy_(grad_output_v.narrow(2, 4, 1))  # [px, py, pz] (Bk x 3 x 1)

        # Return
        return grad_input,None,None,None


## FWD/BWD pass module
class SE3ToRt(Module):
    def __init__(self, transform_type='se3euler', has_pivot=False):
        super(SE3ToRt, self).__init__()
        self.transform_type = transform_type
        self.has_pivot = has_pivot
        self.eps = 1e-12

    ## Check sizes
    def check(self, input, grad_output=None):
        # Input size
        batch_size, num_se3, num_params = input.size()
        num_pivot = 3 if (self.has_pivot) else 0
        if (self.transform_type == 'affine'):
            assert (num_params == 12 + num_pivot)
        elif (self.transform_type == 'se3euler' or
                      self.transform_type == 'se3aa' or
                      self.transform_type == 'se3spquat'):
            assert (num_params == 6 + num_pivot)
        elif (self.transform_type == 'se3quat'):
            assert (num_params == 7 + num_pivot)
        else:
            print("Unknown transform type input: {0}".format(self.transform_type))
            assert (False);

        # Gradient size
        if grad_output is not None:
            num_cols = 5 if (self.has_pivot) else 4
            assert (grad_output.size() == torch.Size([batch_size, num_se3, 3, num_cols]));
    def forward(self, input):
        input=input.view(-1, 1, 6)
        output = SE3ToRtFunction.apply(input,self.transform_type, self.has_pivot,self.eps)
        return output.squeeze(1)


class SE3_layer(Module):
    def __init__(self):
        super(SE3_layer, self).__init__()

    def so3_RV(self,omega):
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
        omega_skew = torch.zeros(batchSize, 3, 3).to(omega.device)
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

        one_minus_cos_theta = torch.ones(theta.size()).to(omega.device) - torch.cos(theta)
        one_minus_cos_div_theta_sqr = torch.div(one_minus_cos_theta, theta_sqr)

        theta_minus_sin_theta = theta - torch.sin(theta)
        theta_minus_sin_div_theta_cube = torch.div(theta_minus_sin_theta, theta_cube)

        # sin_theta_div_theta_tensor = torch.ones(omega_skew.size()).to(omega.device)
        one_minus_cos_div_theta_sqr_tensor = torch.ones(omega_skew.size()).to(omega.device)
        theta_minus_sin_div_theta_cube_tensor = torch.ones(omega_skew.size()).to(omega.device)

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

        completeTransformation = torch.zeros(batchSize, 3, 3).to(omega.device)

        completeTransformation[:, 0, 0] += 1
        completeTransformation[:, 1, 1] += 1
        completeTransformation[:, 2, 2] += 1

        sin_theta_div_theta_tensor = torch.unsqueeze(sin_theta_div_theta_tensor, dim=1)
        completeTransformation = completeTransformation + \
                                 vecMulMat(sin_theta_div_theta_tensor, omega_skew) + \
                                 torch.mul(one_minus_cos_div_theta_sqr_tensor, omega_skew_sqr)

        V = torch.zeros(batchSize, 3, 3).to(omega.device)
        V[:, 0, 0] += 1
        V[:, 1, 1] += 1
        V[:, 2, 2] += 1
        V = V + torch.mul(one_minus_cos_div_theta_sqr_tensor, omega_skew) + \
            torch.mul(theta_minus_sin_div_theta_cube_tensor, omega_skew_sqr)
        return completeTransformation, V
    def forward(self,vec):
        """
            :param vec: [B,6],[rho,omega]
            :return:
            """
        rho = vec[:, 0:3]
        omega = vec[:, 3:6]  # torch.Size([batchSize, 3])
        R, V = self.so3_RV(omega)
        tra = torch.bmm(V, rho.unsqueeze(dim=-1))
        transform_mat = torch.cat([R, tra], dim=2)  # [B, 3, 4]
        return transform_mat


if __name__ == '__main__':
    input1 = torch.tensor([4.,4.,4.,-1, -2.,-3.],requires_grad=True).view(1,6).unsqueeze(1)

    se3 = SE3ToRt(transform_type='se3euler')
    output = se3(input1)
    print(output)
    g=torch.autograd.grad(output.sum(),[input1])[0]
    print('g1',g)


    def euler2mat(angle):
        """Convert euler angles to rotation matrix.
         Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
        Args:
            angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
        Returns:
            Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
        """
        B = angle.size(0)
        x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

        cosz = torch.cos(z)
        sinz = torch.sin(z)

        zeros = z.detach() * 0
        ones = zeros.detach() + 1
        zmat = torch.stack([cosz, sinz, zeros,
                            -sinz, cosz, zeros,
                            zeros, zeros, ones], dim=1).reshape(B, 3, 3)

        cosy = torch.cos(y)
        siny = torch.sin(y)

        ymat = torch.stack([cosy, zeros, -siny,
                            zeros, ones, zeros,
                            siny, zeros, cosy], dim=1).reshape(B, 3, 3)

        cosx = torch.cos(x)
        sinx = torch.sin(x)

        xmat = torch.stack([ones, zeros, zeros,
                            zeros, cosx, sinx,
                            zeros, -sinx, cosx], dim=1).reshape(B, 3, 3)

        rotMat = zmat @ ymat @ xmat
        return rotMat


    def pose_vec2mat(vec, rotation_mode='euler'):
        """
        Convert 6DoF parameters to transformation matrix.
        Args:s
            vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
        Returns:
            A transformation matrix -- [B, 3, 4]
        """
        translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
        rot = vec[:, 3:]
        rot_mat = euler2mat(rot)  # [B, 3, 3]
        transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
        return transform_mat


    input2 = torch.tensor([4., 4., 4., -1, 2., -3.], requires_grad=True).view(1, 6)
    output2 = pose_vec2mat(input2)
    print(output2)
    g2 = torch.autograd.grad(output2.sum(), [input2])[0]
    print('g2',g2)

    input3 = torch.tensor([4., 4., 4., -1, 2., -3.], requires_grad=True).view(1, 6)
    se3_layer = SE3_layer()
    output3 = se3_layer(input3)
    print(output3)
    g3 = torch.autograd.grad(output3.sum(), [input3])[0]
    print('g3', g3)


# class Func(Function):
#     @staticmethod
#     def forward(ctx, x: torch.Tensor, y: torch.Tensor, z: int):
#         w = x * y * z
#         out = x * y + y * z + w
#         ctx.save_for_backward(x, y, w, out)
#         ctx.z = z  # z is not a tensor
#         return out
#
#     @staticmethod
#     def backward(ctx, grad_out):
#         x, y, w, out = ctx.saved_tensors
#         z = ctx.z
#         gx = grad_out * (y + y * z)
#         gy = grad_out * (x + z + x * z)
#         gz = None
#         return gx, gy, gz
#
# a = torch.tensor(1., requires_grad=True, dtype=torch.double)
# b = torch.tensor(2., requires_grad=True, dtype=torch.double)
# c = 4
# d = Func.apply(a, b, c)
# ga=torch.autograd.grad(d,[a])
# print(ga)
