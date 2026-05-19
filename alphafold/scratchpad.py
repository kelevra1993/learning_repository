import asyncio
import time


async def download_A():
    print("Starting A...")
    # Using a normal sleep instead of 'await asyncio.sleep()'
    # This BLOCKS the single thread. The worker cannot leave.
    await time.sleep(3)
    print("Finished A!")


async def download_B():
    print("Starting B...")
    await time.sleep(3)
    print("Finished B!")


async def main():
    start_time = time.time()

    # We tell it to gather both...
    await asyncio.gather(download_A(), download_B())

    print(f"Total time: {time.time() - start_time:.1f}s")


asyncio.run(main())


exit()
import torch
import numpy as np
import math
import torchvision
from fontTools.misc.intTools import bit_count
from torchvision import transforms
from matplotlib import pyplot as plt


def print_tensor_list(tensor, round=4):
    print(np.round(tensor.tolist(), round))


def print_shape(name, tensor):
    print(f"Tensor {name} Is Of Shape : {list(tensor.shape)}")


torch.set_printoptions(linewidth=200, threshold=10_000)
np.set_printoptions(linewidth=200, threshold=10000)

elements = (torch.randn((100, 100, 100)) * 3) - 0.5

train_labels = torch.tensor([[0.90, 9.0000, 2.0000, 1.0000, 3.0000, 1.0000],
                             [4.0000, 3.0000, 5.0000, 3.0000, 6.0000, 1.0000],
                             [7.0000, 2.0000, 8.0000, 6.0000, 9.0000, 4.0000],
                             [0.0000, 9.0000, 1.0000, 1.0000, 2.0000, 4.0000],
                             [3.0000, 2.0000, 7.0000, 3.0000, 8.0000, 6.0000],
                             [9.0000, 0.0000, 5.0000, 6.0000, 0.0000, 7.0000]])
#
# data = torch.arange(210)
# reshaped_data = data.reshape((3, 7, 5, 2))
# multiplier = torch.arange(7).reshape((7,))
# multiplier = multiplier.unsqueeze(dim=-1).unsqueeze(dim=-1)
# output = reshaped_data * multiplier
random_amino_acids = [np.random.random_integers(low=0, high=19) for i in range(20)]
print(*random_amino_acids,sep=",")
# indexes = [2,3,3]
# print(train_labels[indexes,:4])
# a = ["what", "the", "other", "for", "us", "and", "weather", "mouse", "the"]
# print(a.index("the"))

# t = [1, 2, 2, 2, 3]
# t[0,3,4]
# train_labels = torch.reshape(train_labels, shape=(6, 6))
# bins = torch.linspace(start=1, end=10, steps=11)
# displaced_bins = torch.cat(tensors=(bins[1:], torch.tensor([1e8])), dim=-1)
# print(bins)
# print(displaced_bins)
# print(train_labels)

# print(train_labels.unsqueeze(-1) > bins)
# print(train_labels.unsqueeze(-1) < displaced_bins)

# train_labels = train_labels.unsqueeze(-1)
# one_hot = torch.argmax(((train_labels > bins) * (train_labels <= displaced_bins)).to(torch.float), dim=-1)
# print(one_hot)
#
# distance_matrix = ((train_labels > bins) * (train_labels < displaced_bins))
# print(distance_matrix)

# # relu = torch.nn.ReLU()(train_labels)
# # print(relu)
#
# import torch.nn.functional as F
#
# # Example tensor with shape (Batch, Channels, Length) e.g., (2, 3, 4)
# tensor = torch.randn(2, 3, 4)
# print(tensor.numpy())
# tensor = torch.sigmoid(tensor)
# print(tensor.numpy())

# # We want to add 0 zeros to the front, and 2 zeros to the end of the last dimension
# pad_size = (0, 2)
#
# # Apply padding
# padded_tensor = F.pad(tensor, pad_size,value=10)
#
# print(tensor.numpy())
# print(f"Original shape: {tensor.shape}")
# print(f"Padded shape:   {padded_tensor.shape}")
# print(padded_tensor.numpy())

# labels = torch.tensor([7, 2, 7, 1, 0])
# print(torch.bincount(labels))
# one_hot = torch.nn.functional.one_hot(labels, num_classes=10)
# print(one_hot)
#
# q = torch.arange(3000)
# q = q.reshape((3, 10, 100))
# chunk_size = 20
# attention_dimension = -1
# print(q.shape)
# q_chunks = torch.split(q, split_size_or_sections=chunk_size, dim=attention_dimension)
# q = torch.stack(q_chunks, dim=attention_dimension - 2)
# print(f"The Shape Of Q is {q.shape}")
#

#
# # Testing for broadcasting
# hidden_dimension_size = 15
# q = torch.arange(1800)
# q = q.reshape((3, 10, 5, 12))

# matrix_before_broad_cast = q.reshape((3, 10, 1, 5, 12))
# matrix_after_broad_cast = matrix_before_broad_cast.broadcast_to((3, 10, hidden_dimension_size, 5, 12))
# print(q.shape)
# print(matrix_after_broad_cast.shape)
#
# for i in [0, 1, 2, 3, 4]:
#     print(f"At Index {i} we get this matrix")
#     print(20 * '-')
#     for j in [0, 2, 6]:
#         print(matrix_after_broad_cast[0, i, j, :].numpy())
#         print(20 * '-')
#     print(50 * "-")

# torch.flatten()
# # Randomisation
# q = torch.randn(30)
# q = q.reshape((5, 6))
# print(q.numpy())
# offset = -10 * (q < 0).to(torch.int)
# print(offset.numpy())
