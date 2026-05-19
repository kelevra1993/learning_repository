from torch import nn
import numpy as np
import os
import sys
from pathlib import Path

# Dynamically find the 'tutorials' directory (the parent of 05_evoformer)
current_dir = Path(os.getcwd())
tutorials_dir = current_dir.parent

# Add it to the system path
sys.path.append(str(tutorials_dir))

from attention.mha import MultiHeadAttention
# from evoformer.dropout import DropoutRowwise
from evoformer.msa_stack import MSARowAttentionWithPairBias, MSATransition, OuterProductMean
from evoformer.pair_stack import PairStack


def print_shape(name, tensor):
    print(f"Tensor {name} Is Of Shape : {list(tensor.shape)}")


def print_tensor_list(tensor, round=4):
    print(np.round(tensor.tolist(), round))


class ExtraMsaEmbedder(nn.Module):
    """
    Creates the embeddings of extra_msa_feat for the Extra MSA Stack.
    """

    def __init__(self, f_e, c_e):
        """
        Initializes the ExtraMSAEmbedder.

        Args:
            f_e (int): Initial dimension of the extra_msa_feat.
            c_e (int): Embedding dimension of the extra_msa_feat.
        """
        super().__init__()

        ##########################################################################
        # TODO: Initialize the module self.linear for the extra MSA embedding.   #
        ##########################################################################
        self.linear = nn.Linear(in_features=f_e, out_features=c_e)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(self, batch):
        """
        Passes extra_msa_feat through a linear embedder.

        Args:
            batch (dict): Feature dictionary with the following entries:
                * extra_msa_feat: Extra MSA feature of shape (*, N_extra, N_res, f_e).

        Returns:
            torch.tensor: Output tensor of shape (*, N_extra, N_res, c_e):
        """

        e = batch['extra_msa_feat']
        ##########################################################################
        # TODO: Pass extra_msa_feat through the linear layer defined in init.    #
        ##########################################################################
        out = self.linear(e)
        # # Replace "pass" statement with your code
        # pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return out


class MSAColumnGlobalAttention(nn.Module):
    """
    Implements Algorithm 19.
    """

    def __init__(self, c_m, c_z, c=8, N_head=8):
        """
        Initializes MSAColumnGlobalAttention.

        Args:
            c_m (int): Embedding dimension of the MSA representation.
            c_z (int): Embedding dimension of the pair representation.
            c (int, optional): Embedding dimension for MultiHeadAttention. Defaults to 8.
            N_head (int, optional): Number of heads for MultiHeadAttention. Defaults to 8.
        """

        super().__init__()
        self.layer_norm_m = nn.LayerNorm(normalized_shape=c_m)
        # TODO We still need to properly understand the attention dimension being set to -3
        self.global_attention = MultiHeadAttention(c_in=c_m,
                                                   c=c,
                                                   N_head=N_head,
                                                   attn_dim=-3,
                                                   gated=True,
                                                   is_global=True,
                                                   use_bias_for_embeddings=False)

        ##########################################################################
        # TODO: Initialize the modules layer_norm_m and global_attention.        #
        #   Set the parameters for MultiHeadAttention correctly to use global    #
        #   attention.                                                           #
        ##########################################################################

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(self, m):
        """
        Implements the forward pass for Algorithm 19.

        Args:
            m (torch.tensor): MSA representation of shape (*, N_seq, N_res, c_m).

        Returns:
            torch.tensor: Output tensor of the same shape as m.
        """

        out = None

        ##########################################################################
        # TODO: Implement the forward pass for Algorithm 19.                     #
        ##########################################################################
        m = self.layer_norm_m(m)
        out = self.global_attention(m)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return out


class ExtraMsaBlock(nn.Module):
    """
    Implements one block for Algorithm 18.
    """

    def __init__(self, c_e, c_z):
        """
        Initializes ExtraMSABlock.

        Args:
            c_e (int): Embedding dimension of the extra MSA representation.
            c_z (int): Embedding dimension of the pair representation.
        """
        super().__init__()
        self.msa_att_row = MSARowAttentionWithPairBias(c_m=c_e, c_z=c_z, c=8)
        # todo be careful, here the c_z value does not seem to be used.
        self.msa_att_col = MSAColumnGlobalAttention(c_m=c_e,c_z=c_z)
        self.msa_transition = MSATransition(c_m=c_e)
        self.outer_product_mean = OuterProductMean(c_m=c_e, c_z=c_z)

        self.core = PairStack(c_z)

        ##########################################################################
        # TODO: Initialize the modules msa_att_row, msa_att_col, msa_transition, #
        #   outer_product_mean, core (the PairStack), and (optionally for        #
        #   inference) dropout_rowwise. Your implementation should be looking    #
        #   very similar to the one for the EvoformerBlock, but using global     #
        #   column attention.                                                    #
        ##########################################################################

        # # Replace "pass" statement with your code
        # pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(self, e, z):
        """
        Forward pass for Algorithm 18.

        Args:
            e (torch.tensor): Extra MSA representation of shape (*, N_extra, N_res, c_e).
            z (torch.tensor): Pair representation of shape (*, N_res, N_res, c_z).

        Returns:
            tuple: Tuple consisting of the transformed features e and z.
        """
        out_e = e + self.msa_att_row(m=e, z=z)
        out_e += self.msa_att_col(out_e)
        out_e += self.msa_transition(out_e)

        out_z = z + self.outer_product_mean(out_e)
        out_z = self.core(out_z)
        ##########################################################################
        # TODO: Implement one block of Algorithm 18. This should look very       #
        #   similar to your implementation in EvoformerBlock.                    #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return out_e, out_z


class ExtraMsaStack(nn.Module):
    """
    Implements Algorithm 18.
    """

    def __init__(self, c_e, c_z, num_blocks):
        """
        Initializes the ExtraMSAStack.

        Args:
            c_e (int): Embedding dimension of the extra MSA representation.
            c_z (int): Embedding dimension of the pair representation.
            num_blocks (int): Number of blocks in the ExtraMSAStack.
        """
        super().__init__()
        self.blocks = nn.ModuleList([ExtraMsaBlock(c_e=c_e, c_z=c_z) for i in range(num_blocks)])
        ##########################################################################
        # TODO: Initialize self.blocks as a ModuleList of ExtraMSABlocks.        #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(self, e, z):
        """
        Implements the forward pass for Algorithm 18.

        Args:
            e (torch.tensor): Extra MSA representation of shape (*, N_extra, N_res, c_e).
            z (torch.tensor): Pair representation of shape (*, N_res, N_res, c_z).

        Returns:
            torch.tensor: Output tensor of the same shape as z.
        """

        ##########################################################################
        # TODO: Implement the forward pass for Algorithm 18.                     #
        ##########################################################################
        for block in self.blocks:
            e, z = block(e, z)
        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return z
