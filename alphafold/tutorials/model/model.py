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
for folder in ["attention", "feature_embedding", "geometry", "model", "tensor_introduction", "evoformer",
               "feature_extraction", "machine_learning", "structure_module", "tensor_utilities"]:
    sys.path.append(str(tutorials_dir / folder))

import numpy as np
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


from feature_embedding.input_embedder import InputEmbedder
from feature_embedding.recycling_embedder import RecyclingEmbedder
from feature_embedding.extra_msa_stack import ExtraMsaStack, ExtraMsaEmbedder
from evoformer.evoformer import EvoformerStack
from structure_module.structure_module import StructureModule


class Model(nn.Module):
    """
    Implements the Alphafold model according to Algorithm 2.
    """

    def __init__(self, c_m=256, c_z=128, c_e=64, f_e=25, tf_dim=21, c_s=384, num_blocks_extra_msa=4,
                 num_blocks_evoformer=48):
        """
        Initializes the Alphafold model.

        Args:
            c_m (int, optional): Number of channels for the MSA representation. Defaults to 256.
            c_z (int, optional): Number of channels for the pair representation. Defaults to 128.
            c_e (int, optional): Number of channels for the extra MSA representation. Defaults to 64.
            f_e (int, optional): Number of channels of the extra MSA feature. Defaults to 25.
            tf_dim (int, optional): Number of channels of the target feature. Defaults to 22.
            c_s (int, optional): Number of channels for the single representation. Defaults to 384.
            num_blocks_extra_msa (int, optional): Number of blocks for the extra MSA stack. Defaults to 4.
            num_blocks_evoformer (int, optional): Number of blocks for the Evoformer. Defaults to 48.
        """
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_e = c_e
        self.c_s = c_s

        self.input_embedder = InputEmbedder(c_m=c_m, c_z=c_z, tf_dim=tf_dim)
        self.extra_msa_embedder = ExtraMsaEmbedder(f_e=f_e, c_e=c_e)
        self.recycling_embedder = RecyclingEmbedder(c_m=c_m, c_z=c_z)
        self.extra_msa_stack = ExtraMsaStack(c_e=c_e, c_z=c_z, num_blocks=num_blocks_extra_msa)
        self.evoformer = EvoformerStack(c_m=c_m, c_z=c_z, num_blocks=num_blocks_evoformer)
        self.structure_module = StructureModule(c_s=c_s, c_z=c_z)

        ##########################################################################
        # TODO: Initialize the modules input_embedder, extra_msa_embedder,       #
        #   recycling_embedder, extra_msa_stack, evoformer and structure_module. #
        ##########################################################################

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(self, batch):
        """
        Forward pass for the Alphafold model.

        Args:
            batch (dict): A dictionary containing the following features:
                * msa_feat:  Tensor of shape (*, N_seq, N_res, msa_feat_dim, N_cycle).
                * extra_msa_feat: Tensor of shape (*, N_extra, N_res, f_e, N_cycle).
                * target_feat: Tensor of shape (*, N_res, tf_dim, N_cycle). One-hot encoding of the target sequence.
                * residue_index: Tensor of shape (*, N_res, N_cycle). The index of each residue, which is [0,...,N_res-1].

        Returns:
            dict: A dictionary with the following entries:
                * final_positions: Heavy-atom positions in Angstrom of shape (*, N_res, 37, 3, N_cycle).
                * position_mask: Boolean tensor of shape (*, N_res, 37, N_cycle), masking atoms that
                    aren't present in the amino acids.
                * angles: Torsion angles of shape (*, N_layers, N_res, n_torsion_angles, 2, N_cycle) for 
                    every iteration of the Structure Module in every cycle.
                * frames: Backbone frames of shape (*, N_layers, N_res, 4, 4, N_cycle) for every iteration
                    of the Structure Module in every cycle.
        """
        N_cycle = batch['msa_feat'].shape[-1]
        N_seq, N_res = batch['msa_feat'].shape[-4:-2]
        batch_shape = batch['msa_feat'].shape[:-4]
        device = batch['msa_feat'].device
        dtype = batch['msa_feat'].dtype

        print_shape(name="Feature > MSA Feat", tensor=batch["msa_feat"])
        print_shape(name="Feature > Extra MSA Feat", tensor=batch["extra_msa_feat"])
        print_shape(name="Feature > Target Feat", tensor=batch["target_feat"])
        print_shape(name="Feature > Residue Index", tensor=batch["residue_index"])

        c_m = self.c_m
        c_z = self.c_z

        outputs = {}

        # todo will have to put dtype in every single class as an input and by default dtype=torch.float64
        # Todo Remind Yourself that N_seq is for the msa sequences and does not represent the batch
        # Todo condsider calling .forward explicitly to explicitly hand out arguments

        # Initialisation of first tensors
        prev_m = torch.zeros((batch_shape + (N_seq, N_res, c_m)), dtype=torch.float64)
        prev_z = torch.zeros((batch_shape + (N_res, N_res, c_z)), dtype=torch.float64)
        prev_pseudo_beta_x = torch.zeros((batch_shape + (N_res, 3)), dtype=torch.float64)

        for cycle in range(N_cycle):
            print(20*'-')
            print(f'Starting iteration {cycle}')

            current_cycle_input_batch = {key: value[..., cycle] for key, value in batch.items()}

            msa_tensor, pair_rep_tensor = self.input_embedder(current_cycle_input_batch)

            recycled_msa, recycled_pair_rep = self.recycling_embedder(prev_m, prev_z, prev_pseudo_beta_x)

            # todo review this
            msa_tensor[..., 0, :, :] += recycled_msa

            pair_rep_tensor += recycled_pair_rep

            extra_msa_embedding = self.extra_msa_embedder(current_cycle_input_batch)

            # The pair representation is updated by the extra msa embedding before being fed to the evoformer stack
            pair_rep_tensor = self.extra_msa_stack(extra_msa_embedding, pair_rep_tensor)

            # Pass through the evorformer block
            msa_tensor, pair_rep_tensor, single_representation_tensor = self.evoformer(msa_tensor, pair_rep_tensor)

            # TODO Rename F not clear enough
            F = torch.argmax(current_cycle_input_batch["target_feat"], dim=-1)
            structure_module_output = self.structure_module(single_representation_tensor, pair_rep_tensor, F)

            # TODO review this for prev_m here [..., 0, :] ?
            prev_m = msa_tensor
            prev_z = pair_rep_tensor
            prev_pseudo_beta_x = structure_module_output['pseudo_beta_positions']

            for key, value in structure_module_output.items():
                if key in outputs:
                    outputs[key].append(value)
                else:
                    outputs[key] = [value]
            print(20 * '-')
        outputs = {
            key: torch.stack(value, dim=-1) for key, value in outputs.items()
        }

        return outputs
        #
        # ##########################################################################
        # # TODO: Implement the forward pass of Algorithm 2:                       #
        # #   - DONE : Create the initial prev_m, prev_z, and prev_pseudo_beta_x          #
        # #       as zeros of shape (*, N_seq, N_res, c_m), (*, N_res, N_res, c_z) #
        # #       and (*, N_res, 3).                                               #
        # #   - Loop for N_cycle times. At the start of the loop, you can print    #
        # #       'Starting iteration {i}'. Select current_batch from batch by     #
        # #       selecting the i-th element for each tensor in batch.             #
        # #       Implement the main loop according to Algorithm 2. The labels F   #
        # #       for the Structure Module can be computed from target_feat via    #
        # #       `argmax`, as target_feat is one-hot encoded.                     #
        # #       At the end of each loop, append the outputs from the structure   #
        # #       module to outputs (if they are already present) or create a new  #
        # #       list for them in outputs (in the first iteration).               #
        # #   - Stack the outputs along the last dimension.                        #
        # ##########################################################################
        #
        # # Replace "pass" statement with your code
        # pass
        #
        # ##########################################################################
        # #               END OF YOUR CODE                                         #
        # ##########################################################################
        #
        # return outputs
