import torch
from torch import nn
import numpy as np


def print_shape(name, tensor):
    print(f"Tensor {name} Is Of Shape : {list(tensor.shape)}")


def print_tensor_list(tensor, round=4):
    print(np.round(tensor.tolist(), round))


class RecyclingEmbedder(nn.Module):
    """
    Implements Algorithm 32.
    """

    def __init__(self, c_m, c_z):
        """
        Initializes the RecyclingEmbedder.

        Args:
            c_m (int): Embedding dimension of the MSA representation.
            c_z (int): Embedding dimension of the pair representation.
        """
        super().__init__()
        self.bin_start = 3.25
        self.bin_end = 20.75
        self.bin_count = 15

        ##########################################################################
        # TODO: Initialize the modules layer_norm_m, layer_norm_z and linear.    #
        ##########################################################################
        self.linear = nn.Linear(in_features=self.bin_count, out_features=c_z)
        self.layer_norm_m = nn.LayerNorm(normalized_shape=c_m)
        self.layer_norm_z = nn.LayerNorm(normalized_shape=c_z)
        self.bins = torch.linspace(start=self.bin_start, end=self.bin_end, steps=self.bin_count)
        self.displaced_bins = torch.cat(tensors=(self.bins[1:], torch.tensor([1e8])), dim=-1)

        # print_shape(name="Bins", tensor=self.bins)
        # print_tensor_list(tensor=self.bins)
        #
        # print_shape(name="Displaced Bins", tensor=self.displaced_bins)
        # print_tensor_list(tensor=self.displaced_bins, round=1)
        ##########################################################################
        # END OF YOUR CODE                                                       #
        ##########################################################################

    def forward(self, m_prev, z_prev, x_prev):
        """
        Forward pass for Algorithm 32.

        Args:
            m_prev (torch.tensor): MSA representation of previous iteration, shape (*, N_seq, N_res, c_m).
            z_prev (torch.tensor): Pair representation of previous iteration, shape (*, N_res, N_res, c_z).
            x_prev (torch.tensor): Pseudo-beta positions from the previous iterations of 
                shape (*, N_res, 3). These are the positions of the C-beta atoms from the 
                last prediction (in Angstrom), or of C-alpha for glycin.
            

        Returns:
            tuple: A tuple consisting of m_out of shape (*, N_res, c_m) and z_out 
                of shape (*, N_res, N_res, c_z).
        """

        m_out = None
        z_out = None

        difference_matrix = x_prev.unsqueeze(-2) - x_prev.unsqueeze(-3)
        distance_matrix = torch.linalg.vector_norm(difference_matrix, dim=-1)

        # print_shape(name="X Previous Matrix", tensor=x_prev)
        # print_shape(name="Difference Matrix", tensor=difference_matrix)
        # print_tensor_list(difference_matrix[0])
        # print_shape(name="Distance Matrix", tensor=distance_matrix)
        # print_tensor_list(distance_matrix[0])
        distance_matrix = distance_matrix.unsqueeze(-1)
        # todo this line actually differs from the implementation of alpha fold
        distance_matrix = ((distance_matrix > self.bins) * (distance_matrix < self.displaced_bins)).type(x_prev.dtype)

        # todo this is more in line with the alpha fold paper : but could be better implemented in it's own function.
        # distance_matrix = torch.nn.functional.one_hot(torch.argmax(
        #     (distance_matrix > self.bins) * (distance_matrix < self.displaced_bins).to(
        #         torch.float), dim=-1), num_classes=self.bin_count).to(z_prev.dtype)
        # print_shape(name="Distance Matrix", tensor=distance_matrix)
        # print_shape(name="Z Previous", tensor=z_prev)

        z_out = self.linear(distance_matrix) + self.layer_norm_z(z_prev)
        m_out = self.layer_norm_m(m_prev[..., 0, :, :])

        # print_tensor_list(distance_matrix[0])

        ##########################################################################
        # TODO: Implement Algorithm 32 with the following steps:                 #
        #   * Compute the outer difference of x_prev with itself, by             #
        #      unsqueezing it correctly.                                         #
        #   * Use torch.linalg.vector_norm to compute the norm.                  #
        #   * the result d should have shape (*, N_res, N_res)                   #
        #   * Use torch.linspace to create bin_count values from bin_start to    #
        #      bin_end. These are used as the lower bounds for the bins.         #
        #   * Concatenate the lower bounds (omitting the first) with 1e8 to      #
        #      create the upper bounds for the bins.                             #
        #   * The one-hot encoding of the bins is computed by checking, where    #
        #       d>bins_lower and d<bins_upper. You can compute the logical 'and' #
        #       by multiplying the boolean masks. You will need to unsqueeze d   #
        #       to allow for broadcasting in the comparison with the bins.       #
        #   * Use the linear modules and layer_norms to create the ouptuts.      #
        #                                                                        #
        #   This implementation differs from the supplement: Following the code, #
        #   residues below the smallest bin are not assigned the class of the    #
        #   lowest bin, but to no class at all.                                  #
        ##########################################################################
        #
        # # Replace "pass" statement with your code
        # pass

        ##########################################################################
        # END OF YOUR CODE                                                       #
        ##########################################################################

        return m_out, z_out
