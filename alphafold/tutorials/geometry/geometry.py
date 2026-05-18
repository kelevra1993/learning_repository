import torch
from torch import nn

import os
import sys
from pathlib import Path

# Dynamically find the 'tutorials' directory (the parent of 05_evoformer)
current_dir = Path(os.getcwd())
tutorials_dir = current_dir.parent

# Add it to the system path
sys.path.append(str(tutorials_dir))

# from tests.structure_module.residue_constants import rigid_group_atom_position_map, chi_angles_mask
from geometry.residue_constants import rigid_group_atom_position_map, chi_angles_mask, chi_angles_chain, atom_local_positions, \
    atom_frame_inds, atom_mask

import os
import numpy as np
from pathlib import Path
import torch


class Stop(Exception):
    def _render_traceback_(self):
        pass


torch.set_printoptions(linewidth=200, threshold=10_000)
np.set_printoptions(linewidth=200, threshold=np.inf)


def print_tensor_list(tensor, round=4):
    print(np.round(tensor.tolist(), round))


def print_shape(name, tensor):
    print(f"Tensor {name} Is Of Shape : {list(tensor.shape)}")


def create_3x3_rotation(ex, ey):
    """
    Creates a rotation matrix by orthonormalizing ex and ey via Gram-Schmidt.
    Supports batched operation.

    Args:
        ex (torch.tensor): X-axes of the new frames, of shape (*, 3).
        ey (torch.tensor): Y-axes of the new frames, of shape (*, 3).

    Returns:
        torch.tensor: Rotation matrices of shape (*, 3, 3).
    """

    R = None

    ##########################################################################
    # TODO: Orthonormalize ex and ey, then compute ez as their crossproduct. # 
    #  Use torch.linalg.vector_norm to compute the norms for normalization.  #
    #  Orthogonalize ey against ex by subtracting the non-orthogonal part,   #
    #  ex * <ex, ey> from ey, after normalizing ex.                          #
    #  The keepdim parameter can be helpful for both operations.             #
    #  Stack the vectors as columns to build the rotation matrix.            #
    #  Make your to broadcast correctly, to allow for any number of          #
    #  leading dimensions.                                                   #
    ##########################################################################

    ex = ex / torch.linalg.vector_norm(ex, dim=-1, keepdim=True)

    ey = ey - ex * torch.sum(ex * ey, dim=-1, keepdim=True)
    ey = ey / torch.linalg.vector_norm(ey, dim=-1, keepdim=True)

    ez = torch.linalg.cross(ex, ey)

    R = torch.stack(tensors=[ex, ey, ez], dim=-1)

    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################

    return R


def quat_from_axis(phi, n):
    """
    Creates a quaternion with scalar cos(phi/2)
    and vector sin(phi/2)*n.

    Args:
        phi (torch.tensor): Angle of shape (*,)
        n (torch.tensor): Unit vector of shape (*, 3).

    Returns:
        torch.tensor: Quaternion of shape (*, 4).
    """

    # q = torch.tensor([torch.cos(phi), torch.sin(phi) * n])
    cos_phi = torch.cos(phi / 2).unsqueeze(dim=-1)
    sin_phi_n = torch.sin(phi / 2).unsqueeze(dim=-1) * n
    q = torch.cat([cos_phi, sin_phi_n], dim=-1)

    # print_shape(name="cos_phi", tensor=cos_phi)
    # print_shape(name="sin_phi_n", tensor=sin_phi_n)
    # print_shape(name="q", tensor=q)
    # raise Stop
    ##########################################################################
    # TODO: Implement the method as described above. You might need to       # 
    #   reshape phi to allow for broadcasting and concatenation.             # 
    ##########################################################################

    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################

    return q


def quat_mul(q1, q2):
    """
    Batched multiplication of two quaternions.

    Args:
        q1 (torch.tensor): Quaternion of shape (*, 4).
        q2 (torch.tensor): Quaternion of shape (*, 4).

    Returns:
        torch.tensor: Quaternion of shape (*, 4).
    """

    a1 = q1[..., 0:1]  # a1 has shape (*, 1)
    v1 = q1[..., 1:]  # v1 has shape (*, 3)
    a2 = q2[..., 0:1]  # a2 has shape (*, 1)
    v2 = q2[..., 1:]  # v2 has shape (*, 3)

    scalar_part = a1 * a2 - torch.sum(v1 * v2, dim=-1, keepdim=True)
    vector_part = a1 * v2 + a2 * v1 + torch.linalg.cross(v1, v2)
    q_out = torch.cat(tensors=[scalar_part, vector_part], dim=-1)
    # print_shape(name="scalar part",tensor=scalar_part)
    # print_shape(name="vector part",tensor=vector_part)
    # print_shape(name="quaternion part",tensor=q_out)
    # raise Stop
    ##########################################################################
    # TODO: Implement batched quaternion multiplication.                     # 
    ##########################################################################

    # # Replace "pass" statement with your code
    # pass

    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################

    return q_out


def conjugate_quat(q):
    """
    Calculates the conjugate of a quaternion, i.e. 
    (a, -v) for q=(a, v).

    Args:
        q (torch.tensor): Quaternion of shape (*, 4).

    Returns:
        torch.tensor: Conjugate quaternion of shape (*, 4).
    """

    q_out = torch.cat(tensors=[q[..., 0:1], - q[..., 1:]], dim=-1)

    ##########################################################################
    # TODO: Implement quaternion conjugation.                                # 
    ##########################################################################

    # Replace "pass" statement with your code
    pass

    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################

    return q_out


def quat_vector_mul(q, v):
    """
    Rotates a vector by a quaternion according to q*v*q', where q' 
    denotes the conjugate. The vector v is promoted to a quaternion 
    by padding a 0 for the scalar aprt.

    Args:
        q (torch.tensor): Quaternion of shape (*, 4).
        v (torch.tensor): Vector of shape (*, 3).

    Returns:
        torch.tensor: Rotated vector of shape (*, 3).
    """
    # batch_shape = v.shape[:-1]
    # print_shape(name="UnPadded V vector", tensor=v)
    # print(v)
    v = torch.nn.functional.pad(v, pad=(1, 0), value=0)
    v_out = quat_mul(q1=quat_mul(q1=q, q2=v), q2=conjugate_quat(q=q))[..., 1:]
    # print_shape(name="Padded V vector", tensor=v)
    # print(v)
    # print_shape(name="Rotated V vector", tensor=v_out)
    # print(v_out)
    # raise Stop
    ##########################################################################
    # TODO: Implement batched quaternion vector multiplication.              # 
    ##########################################################################

    # Replace "pass" statement with your code

    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################

    return v_out


def quat_to_3x3_rotation(q):
    """
    Converts a quaternion to a rotation matrix.

    Args:
        q (torch.tensor): Quaternion of shape (*, 4).
    """

    R = None

    ##########################################################################
    # TODO: Follow these steps to convert a quaternion to a rotation matrix: # 
    #   - Create the vectors [1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0].    #
    #       broadcast them to shape (*, 3) for batched use.                  #
    #   - Rotate these vectors by q and assemble the result into a matrix.   #
    ##########################################################################

    x_axis = quat_vector_mul(q, torch.tensor([1.0, 0.0, 0.0],dtype=torch.float64).broadcast_to((q.shape[:-1]) + (3,)))
    y_axis = quat_vector_mul(q, torch.tensor([0.0, 1.0, 0.0],dtype=torch.float64).broadcast_to((q.shape[:-1]) + (3,)))
    z_axis = quat_vector_mul(q, torch.tensor([0.0, 0.0, 1.0],dtype=torch.float64).broadcast_to((q.shape[:-1]) + (3,)))



    # print_shape(name="x axis", tensor=x_axis)
    R = torch.stack(tensors=[x_axis, y_axis, z_axis], dim=-1)
    # print_shape(name="Rotation Matrix", tensor=R)

    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################

    return R


def assemble_4x4_transform(R, t):
    """
    Assembles a rotation matrix R and a translation t into a 4x4 homogenous transform.

    Args:
        R (torch.tensor): Rotation matrix of shape (*, 3, 3).
        t (torch.tensor): Translation of shape (*, 3).

    Returns:
        torch.tensor: Transform of shape (*, 4, 4).
    """

    T = None
    R_pad = torch.nn.functional.pad(R, (0, 0, 0, 1), value=0)
    t_pad = torch.nn.functional.pad(t, (0, 1), value=1).unsqueeze(dim=-1)
    T = torch.cat(tensors=[R_pad, t_pad], dim=-1)

    ##########################################################################
    # TODO: Implement the method in the following steps:                     # 
    #   - Concatenate R and t along the column axis.                         #
    #   - Build the pad [0,0,0,1] and broadcast it to shape (*, 1, 4)        #
    #   - Concatenate Rt and the pad along the row axis.                     #
    ##########################################################################

    # Replace "pass" statement with your code
    pass

    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################

    return T


def warp_3d_point(T, x):
    """
    Warps a 3D point through a homogenous 4x4 transform. This means promoting the
    point to homogenous coordinates by padding with 1, multiplying it against the
    4x4 matrix T, then cropping the first three coordinates.

    Args:
        T (torch.tensor): Homogenous 4x4 transform of shape (*, 4, 4).
        x (torch.tensor): 3D points of shape (*, 3).

    Returns:
        torch.tensor: Warped points of shape (*, 3).
    """

    x_warped = None
    device = x.device
    dtype = x.dtype

    x_pad = torch.nn.functional.pad(x, (0, 1), value=1).unsqueeze(dim=-1)
    # print(20*'-')
    # print(T.dtype)
    # print(x_pad.dtype)
    # print(20 * '-')
    x_warped = torch.matmul(T, x_pad).squeeze(dim=-1)[..., :-1]
    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################

    return x_warped


def create_4x4_transform(ex, ey, translation):
    """
    Creates a 4x4 transform, where the rotation matrix is constructed from ex and ey
    (after orthogonalizing ey against ex) and with the given translation.

    Args:
        ex (torch.tensor): Vector of shape (*, 3).
        ey (torch.tensor): Vector of shape (*, 3). Orthogonalized against ex before
            used for the creation of the rotation matrix.  
        translation (torch.tensor): Vector of shape (*, 3).

    Returns:
        torch.tensor: Transform of shape (*, 4, 4).
    """

    T = None
    rotation_matrix = create_3x3_rotation(ex, ey)
    T = assemble_4x4_transform(R=rotation_matrix, t=translation)

    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################

    return T


def invert_4x4_transform(T):
    """
    Inverts a 4x4 transform (R, t) according to 
    (R.T, -R.T @ t).

    Args:
        T (torch.tensor): Transform of shape (*, 4, 4).

    Returns:
        torch.tensor: Inverted transform of shape (*, 4, 4).
    """

    rotation_matrix = T[..., :3, :3]
    translation_matrix = T[..., :3, -1]
    inverted_rotation = torch.transpose(rotation_matrix, dim0=-2, dim1=-1)
    inverted_translation = -1 * torch.matmul(inverted_rotation, translation_matrix.unsqueeze(dim=-1)).squeeze(dim=-1)
    # print_shape(name="Inverted Transformation", tensor=inverted_translation)
    # print_shape(name="Rotation Transformation", tensor=rotation_matrix)
    # print_shape(name="Translation Transformation", tensor=translation_matrix)
    inv_T = assemble_4x4_transform(R=inverted_rotation, t=inverted_translation.squeeze(dim=-1))

    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################

    return inv_T


def makeRotX(phi):
    """
    Creates a 4x4 transform for rotation of phi around the X axis.
    phi is given as (cos(phi), sin(phi)).
    The matrix is constructed according to
    [1  0         0        ]
    [0  cos(phi)  -sin(phi)]
    [0  sin(phi)  cos(phi) ]

    Args:
        phi (torch.tensor): Angle of shape (*, 2). The angle is given as
            (cos(phi), sin(phi)).

    Returns:
        torch.tensor: Rotation transform of shape (*, 4, 4).
    """

    batch_shape = phi.shape[:-1]
    device = phi.device
    dtype = phi.dtype
    phi1, phi2 = torch.unbind(phi, dim=-1)
    T = None

    R = torch.zeros(batch_shape + (3, 3), device=device, dtype=dtype)
    R[..., 0, 0] = 1
    R[..., 1, 1] = phi1
    R[..., 2, 1] = phi2
    R[..., 1, 2] = -phi2
    R[..., 2, 2] = phi1

    t = torch.zeros(batch_shape + (3,), device=device, dtype=dtype)
    T = assemble_4x4_transform(R, t)

    # print(first_column[0, 0, :, :])
    # print(second_column[0, 0, :, :])
    # print(third_column[0, 0, :, :])
    # print_shape(name="First Column", tensor=first_column)
    # print_shape(name="Second Column", tensor=second_column)
    # print_shape(name="Third Column", tensor=third_column)
    # print(T[0, 0, :, :])
    # raise Stop
    # first_line=
    # second_line=
    # third_line=
    ##########################################################################
    # TODO: Build the rotation matrix described above. Assemble it together  # 
    #   with a translation of 0 to a 4x4 transformation.                     #
    ##########################################################################
    #
    # # Replace "pass" statement with your code
    # pass

    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################

    return T


### End of general geometry
### Start of AF specific geometry

def revert_key_values(input_dictionary: dict) -> dict:
    """
    todo to be documented
    """
    output_dictionary = {v: k for k, v in input_dictionary.items()}
    return output_dictionary


def calculate_non_chi_transforms():
    """
    Calculates transforms for the following local backbone frames:
    
    backbone_group: Identity
    pre_omega_group: Identity
    phi_group: 
        ex: CA -> N
        ey: (1, 0, 0)
        t:  N
    psi_group:
        ex: CA -> C
        ey: N  -> CA
        t:  C

    Returns:
        torch.tensor: Stacked transforms of shape (20, 4, 4, 4).
            The second dim corresponds to the different frames.
            The last two dims are the shape of the individual transforms.
    """

    non_chi_transforms = []
    ##########################################################################
    # TODO: Build the four non-chi transforms as described above. Stack them # 
    #   to build non_chi_transforms.                                         #
    #   The transforms are built for every amino acid individually. You can  # 
    #   iterate over rigid_group_atom_position_map.values() to get the       #
    #   individual atom -> position maps for each amino acid. You can use    #
    #   enumerate(rigid_group_atom_position_map.values()) to iterate over    #
    #   the amino acid indices and the values jointly.                       #
    ##########################################################################

    for amino_acid, amino_acid_information in rigid_group_atom_position_map.items():
        backbone_transformation = torch.eye(4)
        pre_omega_transformation = torch.eye(4)

        phi_group_ex = amino_acid_information["N"] - amino_acid_information["CA"]
        phi_group_ey = torch.tensor([1, 0, 0])
        phi_group_translation = amino_acid_information["N"]
        phi_group_transformation = create_4x4_transform(
            ex=phi_group_ex,
            ey=phi_group_ey,
            translation=phi_group_translation)

        psi_group_ex = amino_acid_information["C"] - amino_acid_information["CA"]
        psi_group_ey = amino_acid_information["CA"] - amino_acid_information["N"]
        psi_group_translation = amino_acid_information["C"]
        psi_group_transformation = create_4x4_transform(
            ex=psi_group_ex,
            ey=psi_group_ey,
            translation=psi_group_translation)

        amino_acid_non_chi_frames = torch.stack(tensors=[backbone_transformation,
                                                         pre_omega_transformation,
                                                         phi_group_transformation,
                                                         psi_group_transformation], dim=0)
        non_chi_transforms.append(amino_acid_non_chi_frames)

    non_chi_transforms = torch.stack(tensors=non_chi_transforms, dim=0)
    # print_shape(name="None Chi Transforms", tensor=non_chi_transforms)

    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################

    return non_chi_transforms


def calculate_chi_transforms():
    """
    Calculates transforms for the following local side-chain frames:
    chi1: 
        ex: CA   -> #SC0
        ey: CA   -> N
        t:  #SC0
    chi2:
        ex: #SC0 -> #SC1
        ey: #SC0 -> CA
        t:  #SC1
    chi3:
        ex: #SC1 -> #SC2
        ey: #SC1 -> #SC0
        t: #SC2
    chi4:
        ex: #SC2 -> #SC3
        ey: #SC2 -> #SC1
        t: #SC3

    #SC0 - #SC3 denote the names of the side-chain atoms.
    If the chi angles are not present for the amino acid according to
    chi_angles_mask, they are substituted by the Identity transform.
    

    Returns:
        torch.tensor: Stacked transforms of shape (20, 4, 4, 4).
            The second dim corresponds to the different frames.
            The last two dims are the shape of the individual transforms.
    """

    chi_transforms = None

    # Note: For chi2, chi3 and chi4, ey is the inverse of the previous ex.
    # This means, that ey is (-1, 0, 0) in local coordinates for the frame.
    # Also note: For chi2, chi3, and chi4, ex starts at t of the previous transform.
    # This means, that the starting point is 0 in local coordinates.

    ##########################################################################
    # TODO: Construct the chi transforms. You can follow these steps:        # 
    #   - Construct an empty tensor of shape (20, 4, 4, 4).                  #
    #   - Iterate over rigid_group_atom_position_map.items() to get the      #
    #       amino acids names and atom->position maps for each amino acid.   #
    #   - Iterate over range(4) for the different side-chain angles.         #
    #   - If chi_angles_mask is False, set the transform to the Identity.    #
    #   - Select the next side-chain atom from chi_angles_chain.             #
    #   - Build the transforms as described above. You'll need an if-clause  #
    #       to differentiate between chi1 and the other transforms.          #
    ##########################################################################

    chi_transforms = torch.zeros((20, 4, 4, 4))
    for amino_acid_index, (amino_acid, amino_acid_information) in enumerate(rigid_group_atom_position_map.items()):

        side_chain_centers = chi_angles_chain[amino_acid]

        for i in range(4):
            if chi_angles_mask[amino_acid_index][i] == 0:
                chi_transforms[amino_acid_index, i] = torch.eye(4)
                continue

            center_atom = side_chain_centers[i]
            # Side chain matrix to be constructed
            ex = amino_acid_information[center_atom]

            if i == 0:
                ey = amino_acid_information["N"] - amino_acid_information["CA"]
            else:
                ey = torch.tensor([-1, 0, 0])
            transformation = create_4x4_transform(ex=ex,
                                                  ey=ey,
                                                  translation=ex)
            chi_transforms[amino_acid_index, i] = transformation

    # print_shape(name="Chi Transforms", tensor=chi_transforms)

    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################

    return chi_transforms


def precalculate_rigid_transforms():
    """
    Calculates the non-chi transforms backbone_group, pre_omega_group, phi_group and psi_group,
    together with the chi transforms chi1, chi2, chi3, and chi4.

    Returns:
        torch.tensor: Transforms of shape (20, 8, 4, 4).
    """

    rigid_transforms = torch.cat(tensors=[calculate_non_chi_transforms(),
                                          calculate_chi_transforms()], dim=1)
    # print_shape(name="Rigid Transformation", tensor=rigid_transforms)
    ##########################################################################
    # TODO: Concatenate the non-chi transforms and chi transforms.           # 
    ##########################################################################

    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################

    return rigid_transforms


def compute_global_transforms(T, alpha, F):
    """
    Calculates global frames for each frame group of each amino acid
    by applying the global transform T and injecting rotation transforms
    in between the side chain frames. 
    Implements Line 1 - Line 10 of Algorithm 24.

    Args:
        T (torch.tensor): Global backbone transform for each amino acid. Shape (N_res, 4, 4).
        alpha (torch.tensor): Torsion angles for each amino acid. Shape (N_res, 7, 2).
            The angles are in the order (omega, phi, psi, chi1, chi2, chi3, chi4).
            Angles are given as (cos(a), sin(a)).
        F (torch.tensor): Label for each amino acid of shape (N_res,).
            Labels are encoded as 0: Ala, 1: Arg, ..., 19: Val.

    Returns:
        torch.tensor: Global frames for each amino acid of shape (N_res, 8, 4, 4).
    """

    global_transforms = None
    device = T.device
    dtype = T.dtype
    number_residues = F.shape[-1]

    ##########################################################################
    # TODO: Construct the global transforms, according to line 1 - line 10   #
    #   from Algorithm 24. You don't need to support batched use.            #
    #   You can follow these steps:                                          #  
    #   - Normalize alpha, so that its values represent (cos(phi), sin(phi)) #
    #   - Use `torch.unbind` to unbind alpha into omega, phi, psi, chi1,     #
    #       chi2, chi3, and chi4.                                            #
    #   - Compute all_rigid_transforms with precalculate_rigid_transforms.   #
    #   - Send the transforms to the same device that T/alpha/F are on.      #
    #   - Select the correct local_transforms by indexing with F.            #
    #   - The global backbone transform is T, since the local backbone       #
    #       transform is the identity.                                       #
    #   - Iterate over omega, phi, psi, and chi1. Build the global           #
    #       transforms by concatenating T, the local transform, and a        #
    #       rotation matrix. Concatenating 4x4 transforms means multiplying  #
    #       them. You can use the matrix multiplication operator `@`.        #
    #   - Iterate over chi2, chi3, and chi4. Build the global transforms by  #
    #       concatenating the upstream global transform (chi1, chi2, chi3)   #
    #       with the local transform and the rotation.                       #
    #   - Stack the global transforms.                                       #
    ##########################################################################

    normalized_alpha = alpha / torch.linalg.vector_norm(alpha, dim=-1, keepdim=True)
    omega, phi, psi, chi1, chi2, chi3, chi4 = torch.unbind(normalized_alpha, dim=-2)
    initial_rigid_transformations = precalculate_rigid_transforms().to(dtype=dtype, device=device)
    global_transforms = initial_rigid_transformations[F]
    # print_shape(name="Selected Transformation Matrices", tensor=global_transforms)
    # print_shape(name="global_transforms[:, 0, :, :]", tensor=global_transforms[:, 0, :, :])
    # T is the global backbone transformation so it can just be replaced as it is to the correct transformation
    # print("Before Insertion Of Predictions")
    # print_tensor_list(global_transforms[0, 0, :, :])
    # print_tensor_list(global_transforms[0, 1, :, :])
    global_transforms[...,0, :, :] = T
    # print("After Insertion Of Predictions")
    # print_tensor_list(global_transforms[0, 0, :, :])
    # print_tensor_list(global_transforms[0, 1, :, :])

    for index, angle in enumerate([omega, phi, psi, chi1]):
        transformation_index = index + 1
        rotation_matrix = makeRotX(phi=angle)

        global_transforms[...,transformation_index, :, :] = torch.matmul(
            input=global_transforms[..., 0, :, :],
            other=torch.matmul(
                input=global_transforms[...,transformation_index, :, :],
                other=rotation_matrix
            ))

    for index, angle in enumerate([chi2, chi3, chi4]):
        transformation_index = index + 5
        rotation_matrix = makeRotX(phi=angle)
        global_transforms[..., transformation_index, :, :] = torch.matmul(
            input=global_transforms[..., transformation_index - 1, :, :],
            other=torch.matmul(
                input=global_transforms[..., transformation_index, :, :],
                other=rotation_matrix
            ))

    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################

    return global_transforms


def compute_all_atom_coordinates(T, alpha, F):
    """
    Implements Algorithm 24. 

    Args:
        T (torch.tensor): Global backbone transform for each amino acid. Shape (N_res, 4, 4).
        alpha (torch.tensor): Torsion angles for each amino acid. Shape (N_res, 7, 2).
            The angles are in the order (omega, phi, psi, chi1, chi2, chi3, chi4).
            Angles are given as (cos(a), sin(a)).
        F (torch.tensor): Label for each amino acid of shape (N_res,).
            Labels are encoded as 0: Ala, 1: Arg, ..., 19: Val.

    Returns:
        tuple: A tuple consisting of the following values:
            global_positions: Tensor of shape (N_res, 37, 3), containing the global positions
                for each atom for each amino acid.
            all_atom_mask: Boolean tensor of shape (N_res, 37), containing whether or not the atoms
                are present in the amino acids.
    """

    global_positions, all_atom_mask = None, None
    device = T.device
    dtype = T.dtype

    ##########################################################################
    # TODO: Implement Algorithm 24. You can follow these steps:              # 
    #   - build the global frames using compute_global_transforms.           #
    #   - retrieve atom_local_positions, atom_frame_inds, and atom_mask      #
    #       from residue_constants. Map them to the same device used by T,   #
    #       alpha and F using `tensor.to(device=device)`.                    #
    #   - Select the local positions, frame inds and mask using F.           #
    #   - Select the global frames using your selected frame indices.        #
    #       You can use integer indexing or `torch.gather` for this.         #
    #       If using integer indexing, you'll need to index with             #
    #       [0,...,N_res], broadcasted to (N_res, N_atoms) into the first    #
    #       dimension, so that the shape matches the selected frame inds.    #
    #   - Pad the local positions with 1 to promote them to homogenous       #
    #       coordinates. Warp them through the selected global frames by     #  
    #       batched matrix vector multiplication. You can use `torch.einsum` #
    #       to handle the dimensions.                                        #
    #   - Drop the 1 of the resulting homogenous coordinates to select the   #
    #       atom positions.                                                  #
    ##########################################################################

    # Nres,8,4,4
    global_transforms = compute_global_transforms(T=T, alpha=alpha, F=F)
    # print_shape(name="Global Transforms", tensor=global_transforms)

    # Nres, 37, 3
    local_postions = atom_local_positions[F].to(device=device,dtype=dtype)
    # print_shape(name="Local Positions", tensor=local_postions)

    # Nres, 37
    frame_indices = atom_frame_inds[F]
    # print_shape(name="Frame Indices", tensor=frame_indices)

    # Nres, 37
    all_atom_mask = atom_mask[F]
    # print_shape(name="Atom Position Masks", tensor=all_atom_mask)

    # primary_indices = torch.arange(F.shape[0]).unsqueeze(dim=-1)
    # Nres,37,4,4
    # global_frames = global_transforms[primary_indices, frame_indices]
    # todo this has to be thouroughly checked because we did not use torch.gather before
    dim_diff = global_transforms.ndim - frame_indices.ndim
    frame_indices = frame_indices.reshape(frame_indices.shape + (1,) * dim_diff)
    frame_indices = frame_indices.broadcast_to(
        frame_indices.shape[:-dim_diff] + global_transforms.shape[-dim_diff:])
    global_frames = torch.gather(global_transforms, dim=-3, index=frame_indices)
    # print_shape(name="Global Frames", tensor=global_frames)

    # Nres,37,3
    # print("Calling Warp 3D Points")
    # print(f"Global Frames : {global_frames.dtype}")
    # print(f"Local Positions : {local_postions.dtype}")
    global_positions = warp_3d_point(T=global_frames, x=local_postions)
    # print_shape(name="Global Positions", tensor=global_positions)
    # ##########################################################################
    # #               END OF YOUR CODE                                         #
    # ##########################################################################


    return global_positions, all_atom_mask
