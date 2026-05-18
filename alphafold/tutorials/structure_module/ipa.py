import torch
import math
from torch import nn

import os
import sys
from pathlib import Path

# Dynamically find the 'tutorials' directory (the parent of 05_evoformer)
current_dir = Path(os.getcwd())
tutorials_dir = current_dir.parent

# Add it to the system path
sys.path.append(str(tutorials_dir))

from geometry.geometry import invert_4x4_transform, warp_3d_point
from tensor_utilities.utilities import unsqueeze_tensor

import numpy as np


class Stop(Exception):
    def _render_traceback_(self):
        pass


torch.set_printoptions(linewidth=200, threshold=10_000)
np.set_printoptions(linewidth=200, threshold=np.inf)


def print_tensor_list(tensor, round=4):
    print(np.round(tensor.tolist(), round))


def print_shape(name, tensor):
    print(f"Tensor {name} Is Of Shape : {list(tensor.shape)}")


class InvariantPointAttention(nn.Module):
    """
    Implements invariant point attention, according to Algorithm 22.
    """

    def __init__(self, c_s, c_z, n_query_points=4, n_point_values=8, N_head=12, c=16):
        """
        Initializes the invariant point attention module. 

        Args:
            c_s (int): Number of channels for the single representation.
            c_z (int): Number of channels for the pair representation.
            n_query_points (int, optional): Number of query points for point attention. 
                Used for the embedding of q_points and k_points. Defaults to 4.
            n_point_values (int, optional): Number of value points for point attention. 
                Used for the embedding of v_points. Defaults to 8.
            n_head (int, optional): Number of heads for multi-head attention. Defaults to 12.
            c (int, optional): Embedding dimension for each individual head. Defaults to 16.
        """
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.n_query_points = n_query_points
        self.n_point_values = n_point_values
        self.N_head = N_head
        self.c = c

        # todo : here alpha fold implementation did not use bias but we will use Bias
        self.linear_k = nn.Linear(in_features=c_s, out_features=c * N_head, bias=True)
        self.linear_q = nn.Linear(in_features=c_s, out_features=c * N_head, bias=True)
        self.linear_v = nn.Linear(in_features=c_s, out_features=c * N_head, bias=True)

        self.linear_k_points = nn.Linear(in_features=c_s, out_features=N_head * n_query_points * 3, bias=True)
        self.linear_q_points = nn.Linear(in_features=c_s, out_features=N_head * n_query_points * 3, bias=True)
        self.linear_v_points = nn.Linear(in_features=c_s, out_features=N_head * n_point_values * 3, bias=True)

        self.linear_b = nn.Linear(in_features=c_z, out_features=N_head, bias=True)

        self.linear_out = nn.Linear(in_features=N_head * (c + c_z + n_point_values * 4), out_features=c_s, bias=True)

        self.head_weights = nn.Parameter(torch.zeros((N_head,)))
        self.softplus = nn.Softplus()

        ##########################################################################
        # TODO: Initialize the layers linear_q, linear_k, linear_v,              #
        #   linear_q_points, linear_k_points, linear_v_points, linear_b, and     # 
        #   linear_out. The embeddings for q, k and v are similar to             #
        #   MultiHeadAttention, except that they use bias (this clashes with the #
        #   supplement, but follows the official implementation).                #
        #   The point embeddings need to create three values per head and point. #
        #   They also use bias.                                                  #
        #   The embedding for the bias computes one bias value per head.         #
        #   For the input dimension of linear_out, count the channels of the     #
        #   various outputs in line 11 from the algorithm. If you have trouble   #
        #   with this, you can look below at the output description of           #
        #   `compute_outputs`. The output dimension of linear_out is c_s.        #
        #                                                                        #
        #   For the weight per head, gamma, initialize head_weights to a         #
        #   zero-tensor wrapped in nn.Parameter. Also, initialize nn.Softplus    #
        #   for the computation of gamma.                                        #
        ##########################################################################

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def prepare_qkv(self, s):
        """
        Creates the standard attention embeddings q, k, and v, as well as the point 
        embeddings qp, kp, and vp, for invariant point attention.

        Args:
            s (torch.tensor): Single representation of shape (*, N_res, c_s).

        Returns:
            tuple: A tuple consisting of the following embeddings:
                q: Tensor of shape (*, N_head, N_res, c)  
                k: Tensor of shape (*, N_head, N_res, c)
                v: Tensor of shape (*, N_head, N_res, c)
                qp: Tensor of shape (*, N_head, N_query_points, N_res, 3)
                kp: Tensor of shape (*, N_head, N_query_points, N_res, 3)
                vp: Tensor of shape (*, N_head, N_point_values, N_res, 3)
        """
        c = self.c
        n_head = self.N_head
        n_qp = self.n_query_points
        n_pv = self.n_point_values

        layers = [self.linear_q, self.linear_k, self.linear_v, self.linear_q_points, self.linear_k_points,
                  self.linear_v_points]
        embeddings = [layer(s) for layer in layers]

        # TODO Decide on which implementation makes sense.
        # Solution proprosed by Kilian Mandon
        shape_adds = [(n_head, c), (n_head, c), (n_head, c), (3, n_head, n_qp), (3, n_head, n_qp), (3, n_head, n_pv)]
        out_shapes = [out.shape[:-1] + shape_add for out, shape_add in zip(embeddings, shape_adds)]
        embeddings = [out.view(out_shape) for out, out_shape in zip(embeddings, out_shapes)]
        for i in range(3):
            embeddings[i] = embeddings[i].movedim(-3, -2)
        for i in range(3, 6):
            embeddings[i] = embeddings[i].movedim(-3, -1).movedim(-4, -2)

        # TODO Decide on which implementation makes sense.
        # # Personal implementation
        # for i in range(3):
        #     embedding_chunks = torch.split(embeddings[i], split_size_or_sections=self.c, dim=-1)
        #     embeddings[i] = torch.stack(embedding_chunks, dim=-3)
        #     print_shape(name="After Reshape Tensor Points", tensor=embeddings[i])
        #
        # points = [n_qp,n_qp,n_pv]
        # for i in range(3,6):
        #     embedding_chunks = torch.split(embeddings[i], split_size_or_sections=points[i%3]*3, dim=-1)
        #     embeddings[i] = torch.stack(embedding_chunks, dim=-3)
        #     embedding_chunks = torch.split(embeddings[i], split_size_or_sections=3, dim=-1)
        #     embeddings[i] = torch.stack(embedding_chunks, dim=-3)

        ##########################################################################
        # TODO: Implement the embedding preparation in the following steps:      #
        #   - Pass s through all of the embedding layers.                        # 
        #   - Reshape the feature dimension of the embeddings so that q, k and v #
        #     have shape (*, N_head, c), qp and kp have shape                    #
        #     (*, 3, N_head, n_qp) and vp has shape (*, 3, N_head, n_pv).        #
        #   - Move the dimensions to match the shapes in the method description. # 
        ##########################################################################

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return embeddings

    def compute_attention_scores(self, q, k, qp, kp, z, T):
        """
        Computes the attention scores for invariant point attention, 
        according to line 7 from Algorithm 22.

        Args:
            q (torch.tensor): Query embeddings of shape (*, N_head, N_res, c).
            k (torch.tensor): Key embeddings of shape (*, N_head, N_res, c).
            qp (torch.tensor): Query point embeddings of shape (*, N_head, N_query_points, N_res, 3).
            kp (torch.tensor): Key point embeddings of shape (*, N_head, N_query_points, N_res, 3).
            T (torch.tensor): Backbone transforms of shape (*, N_res, 4, 4).

        Returns:
            torch.tensor: Attention scores of shape (*, N_head, N_res, N_res).
        """

        att_scores = None
        # print_shape(name="Query", tensor=q)
        # print_shape(name="Key", tensor=k)
        # print_shape(name="Query Point", tensor=qp)
        # print_shape(name="Key Point", tensor=kp)
        # print_shape(name="(Single Representation) Z Bias Input", tensor=z)
        # print(20 * "=")

        wc = math.sqrt(2 / (9 * self.n_query_points))
        wl = math.sqrt(1 / 3)
        # gamma = self.softplus(self.head_weights).view((-1, 1, 1))

        # Classic attention
        scaler = np.sqrt(1 / self.c)
        attention = torch.matmul(q, torch.transpose(k, dim0=-1, dim1=-2)) * scaler

        # Add bias
        bias = self.linear_b(z)
        bias = bias.movedim(source=-1, destination=-3)

        # This seems a little bit akward that it would be here. But it should be properly documented
        # Especially unsqueezed as it is.
        T = T.unsqueeze(dim=-4).unsqueeze(dim=-4)

        scaled_head_weights = unsqueeze_tensor(input=wc * self.softplus(self.head_weights) / 2,
                                               number=2,
                                               direction="right")

        # personal implementation that might be wrong commented out !!!
        # global_query = warp_3d_point(T=T, x=qp)
        # global_key = warp_3d_point(T=T, x=kp)
        # key_query_distance_squared = torch.linalg.vector_norm(global_key - global_query, dim=-1, keepdim=True)**2
        # proposal that might need to be checked to be properly understood especially the outer product

        # Despite is seeming a little bit wierd, this is what ultimately allows us to get to our final Nres, Nres, Matrix
        # When we will be doing the difference :: This has to be thouroughly reviewed
        global_query = warp_3d_point(T=T, x=qp).unsqueeze(-2)
        global_key = warp_3d_point(T=T, x=kp).unsqueeze(-3)
        key_query_distance_squared = torch.linalg.vector_norm((global_query - global_key), dim=-1) ** 2
        sum_key_query_distances_squared = torch.sum(key_query_distance_squared, dim=-3)

        attention_scores = torch.softmax(
            wl * (attention + bias - scaled_head_weights * sum_key_query_distances_squared), dim=-1)


        return attention_scores

    def compute_outputs(self, att_scores, z, v, vp, T):
        """
        Computes the different output vectors for the IPA attention mechanism:
        The pair output, the standard attention output, and the point attention output,
        as well as the norm of the point attention output.

        Args:
            att_scores (torch.tensor): Attention scores of shape (*, N_head, N_res, N_res).
            z (torch.tensor): Pair representation of shape (*, N_res, N_res, c_z).
            v (torch.tensor): Value vectors of shape (*, N_head, N_res, c).
            vp (torch.tensor): Value points of shape (*, N_head, N_point_values, N_res, 3).
            T (torch.tensor): Backbone transforms of shape (*, N_res, 4, 4).

        Returns:
            tuple: A tuple consisting of the following outputs:
                - output from the value vectors of shape (*, N_res, N_head*c).
                - output from the value points of shape (*, N_res, N_head*3*N_point_values).
                - norm of the output vectors from the value points of shape (*, N_res, N_head*N_point_values)
                - output from the pair representation of shape (*, N_res, N_head*c_z).
        """

        # Value Applied attention
        v_out = torch.einsum('...hij,...hjc->...hic', att_scores, v)
        v_out = v_out.movedim(source=-3, destination=-2).flatten(start_dim=-2)

        # Pairwise Applied Attention
        pairwise_out = torch.einsum('...hij,...ijc->...hic', att_scores, z)
        pairwise_out = pairwise_out.movedim(source=-3, destination=-2).flatten(start_dim=-2, end_dim=-1)

        # Value Point Applied Attention
        T = T.unsqueeze(dim=-4).unsqueeze(dim=-4)
        global_value_point = warp_3d_point(T=T, x=vp)
        scaled_global_value_point = torch.einsum('...Bij,...BNjk->...BNik', att_scores, global_value_point)
        vp_out = warp_3d_point(T=invert_4x4_transform(T=T), x=scaled_global_value_point)
        vp_out = torch.einsum('...hpic->...ichp', vp_out)
        vp_out_norm = torch.linalg.vector_norm(vp_out, dim=-3, keepdim=True)

        # print_shape(name="Normed Global Value Point Output", tensor=vp_out_norm)
        vp_out = vp_out.flatten(start_dim=-3)
        vp_out_norm = vp_out_norm.flatten(start_dim=-3)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return v_out, vp_out, vp_out_norm, pairwise_out

    def forward(self, s, z, T):
        """
        Implements the forward pass for InvariantPointAttention, as specified in Algorithm 22.

        Args:
            s (torch.tensor): Single representation of shape (*, N_res, c_s).
            z (torch.tensor): Pair representation of shape (*, N_res, N_res, c_z).
            T (torch.tensor): Backbone transforms of shape (*, N_res, 4, 4).

        Returns:
            torch.tensor: Output tensor of shape (*, N_res, c_s).
        """

        q, k, v, qp, kp, vp = self.prepare_qkv(s=s)

        att_scores = self.compute_attention_scores(q=q, k=k, qp=qp, kp=kp, z=z, T=T)

        v_out, vp_out, vp_out_norm, pairwise_out = self.compute_outputs(att_scores=att_scores, z=z, v=v, vp=vp, T=T)

        # print_shape(name="V OUT", tensor=v_out)
        # print_shape(name="VP OUT", tensor=vp_out)
        # print_shape(name="VP OUT NORM", tensor=vp_out_norm)
        # print_shape(name="PAIRWISE OUT", tensor=pairwise_out)
        out = self.linear_out(torch.cat(tensors=(v_out, vp_out, vp_out_norm, pairwise_out), dim=-1))
        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return out
