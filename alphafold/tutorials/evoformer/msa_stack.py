import os
import sys
import torch
import numpy as np
from torch import nn
from pathlib import Path

# Dynamically find the 'tutorials' directory (the parent of 05_evoformer)
current_dir = Path(os.getcwd())
tutorials_dir = current_dir.parent

# Add it to the system path
sys.path.append(str(tutorials_dir))
from attention.mha import MultiHeadAttention


def print_tensor_list(tensor, round=4):
    print(np.round(tensor.tolist(), round))


def print_shape(name, tensor):
    print(f"Tensor {name} Is Of Shape : {list(tensor.shape)}")


class MSARowAttentionWithPairBias(nn.Module):
    """
    Implements Algorithm 7.
    """

    def __init__(self, c_m, c_z, c=32, N_head=8):
        """
        Initializes MSARowAttentionWithPairBias.

        Args:
            c_m (int): Embedding dimension of the msa representation.
            c_z (int): Embedding dimension of the pair representation.
            c (int, optional): Embedding dimension for multi-head attention. Defaults to 32.
            N_head (int, optional): Number of heads for multi-head attention. Defaults to 8.
        """
        super().__init__()

        ##########################################################################
        # TODO: Initialize the modules layer_norm_m, layer_norm_z, linear_z      #
        #        and mha for Algorithm 7.                                        #
        #        linear_z is used to embed the pair bias and needs to create     #
        #        one value per head, therefore c_out is N_head.                  #
        ##########################################################################
        # todo to be reviewed to be fully understood
        self.layer_norm_m = nn.LayerNorm(normalized_shape=c_m)
        self.layer_norm_z = nn.LayerNorm(normalized_shape=c_z)

        # todo recheck but this layer does not use bias
        self.linear_z = nn.Linear(in_features=c_z, out_features=N_head, bias=False)

        self.mha = MultiHeadAttention(c_in=c_m,
                                      c=c,
                                      N_head=N_head,
                                      attn_dim=-2,
                                      gated=True,
                                      is_global=False,
                                      use_bias_for_embeddings=False)
        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(self, m, z):
        """
        Implements the forward pass according to Algorithm 7.

        Args:
            m (torch.tensor): MSA representation of shape (*, N_seq, N_res, c_m).
            z (torch.tensor): Pair representation of shape (*, N_res, N_res, c_z).

        Returns:
            torch.tensor: Output tensor of the same shape as m.
        """
        out = None
        ##########################################################################
        # TODO: Implement the forward pass for Algorithm 7. Note that the bias   #
        #        is embedded as (*, z, z, N_head) but needs to have shape        #
        #       (*, N_head, z, z) for MultiHeadAttention.                        #
        ##########################################################################
        normalized_m = self.layer_norm_m(m)
        # Creation of bias
        bias = torch.movedim(self.linear_z(z), source=-1, destination=-3)
        out = self.mha.forward(x=normalized_m, bias=bias, attention_mask=None)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return out


class MSAColumnAttention(nn.Module):
    """
    Implements Algorithm 8.
    """

    def __init__(self, c_m, c=32, N_head=8):
        """
        Initializes MSAColumnAttention.

        Args:
            c_m (int): Embedding dimension of the MSA representation.
            c (int, optional): Embedding dimension for multi-head attention. Defaults to 32.
            N_head (int, optional): Number of heads for multi-head attention. Defaults to 8.
        """
        super().__init__()

        self.layer_norm_m = nn.LayerNorm(normalized_shape=c_m)
        self.mha = MultiHeadAttention(c_in=c_m,
                                      c=c,
                                      N_head=N_head,
                                      attn_dim=-3,
                                      gated=True,
                                      is_global=False,
                                      use_bias_for_embeddings=False)

        ##########################################################################
        # TODO: Initialize the modules layer_norm_m and mha for Algorithm 8.     #
        ##########################################################################

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(self, m):
        """
        Implements the forward pass according to algorithm Algorithm 8.

        Args:
            m (torch.tensor): MSA representation of shape (N_seq, N_res, c_m).

        Returns:
            torch.tensor: Output tensor of the same shape as m.
        """
        normalized_m = self.layer_norm_m(m)
        out = self.mha.forward(x=normalized_m, bias=None, attention_mask=None)

        ##########################################################################
        # TODO: Implement the forward pass for Algorithm 8.                      #
        ##########################################################################

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return out


class MSATransition(nn.Module):
    """
    Implements Algorithm 9.
    """

    def __init__(self, c_m, n=4):
        """
        Initializes MSATransition.

        Args:
            c_m (int): Embedding dimension of the MSA representation.
            n (int, optional): Factor for the number of channels in the intermediate dimension. 
             Defaults to 4.
        """
        super().__init__()

        ##########################################################################
        # TODO: Initialize the modules layer_norm, linear_1, relu and linear_2   #
        #   for Algorithm 9.
        ##########################################################################
        self.layer_norm = nn.LayerNorm(normalized_shape=c_m)
        self.linear_1 = nn.Linear(in_features=c_m, out_features=n * c_m)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(in_features=n * c_m, out_features=c_m)
        self.sequential = nn.Sequential(
            self.layer_norm,
            self.linear_1,
            self.relu,
            self.linear_2,
        )
        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(self, m):
        """
        Implements the forward pass for Algorithm 9.

        Args:
            m (torch.tensor): MSA feat of shape (*, N_seq, N_seq, c_m).

        Returns:
            torch.tensor: Output tensor of the same shape as m.
        """
        out = self.sequential(m)

        ##########################################################################
        # TODO: Implement the forward pass for Algorithm 9.                      #
        ##########################################################################

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return out


class OuterProductMean(nn.Module):
    """
    Implements Algorithm 10.
    """

    def __init__(self, c_m, c_z, c=32):
        """
        Initializes OuterProductMean.

        Args:
            c_m (int): Embedding dimension of the MSA representation.
            c_z (int): Embedding dimension of the pair representation. 
            c (int, optional): Embedding dimension of a and b from Algorithm 10. 
                Defaults to 32.
        """
        super().__init__()
        self.c = c
        self.layer_norm = nn.LayerNorm(normalized_shape=c_m)
        self.linear_1 = nn.Linear(in_features=c_m, out_features=c)
        self.linear_2 = nn.Linear(in_features=c_m, out_features=c)
        self.linear_out = nn.Linear(in_features=c * c, out_features=c_z)

        ##########################################################################
        # TODO: Initialize the modules layer_norm, linear_1, linear_2 and        #
        #   linear_out for Algorithm 10. linear_1 creates the embdding for a,    #
        #   while linear_2 creates the embedding for b.                          #
        ##########################################################################

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(self, m):
        """
        Forward pass for Algorithm 10.

        Args:
            m (torch.tensor): MSA feat of shape (*, N_seq, N_res, c_m).

        Returns:
            torch.tensor: Output tensor of shape (*, N_res, N_res, c_z).
        """
        N_seq = m.shape[-3]
        z = None

        ##########################################################################
        # TODO: Implement the forward pass for Algorithm 10. In contrast to the  #
        #   supplement, the AlphaFold implementation doesn't compute the mean    #
        #   in line 3, but the sum overr all sequences. Instead, the output z    #
        #   is divided by N_seq after line 4. This changes the results, as the   #
        #   biases of the affine linear output layer are affected by the         #
        #   scaling as well. We follow the implementation.                       #
        #                                                                        #
        #   After summation over the sequences, the intermediate o has shape     #
        #   (*, N_res, N_res, c, c) before flattening to (*, N_res, N_res, c*c). #
        #                                                                        #
        #   The outer product and the summation over the sequences in line 4     #
        #   can be computed efficiently using torch.einsum.                      #
        ##########################################################################

        normalized_m = self.layer_norm(m)
        left_matrix = self.linear_1(normalized_m)
        right_matrix = self.linear_2(normalized_m)
        output_matrix = torch.einsum('...sic,...sjd->...ijcd', left_matrix, right_matrix)
        flattened_output_matrix = torch.flatten(input=output_matrix, start_dim=-2, end_dim=-1)
        z = self.linear_out(flattened_output_matrix) / N_seq

        # print_shape(name="Left Matrix", tensor=left_matrix)
        # print_shape(name="Right Matrix", tensor=right_matrix)
        # print_shape(name="Outer Product Matrix", tensor=output_matrix)
        # print_shape(name="Output Matrix", tensor=z)


        return z
