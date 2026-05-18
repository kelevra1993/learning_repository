import math
import torch

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


n_layer = 2
N = 3
c_m = 4
c_z = 5
c = 6
N_head = 7
N_seq = 8
N_extra = 9
N_res = 10
c_s = 11
n_qp = 12
n_pv = 13
n_torsion_angles = 7

feature_shapes = {
    'm': (N_seq, N_res, c_m),
    's': (N_res, c_s),
    'z': (N_res, N_res, c_z),
    'residue_index': (N_res,),
    'x': (N_res, 3),
    'q': (N_head, N_res, c),
    'k': (N_head, N_res, c),
    'v': (N_head, N_res, c),
    'qp': (N_head, n_qp, N_res, 3),
    'kp': (N_head, n_qp, N_res, 3),
    'vp': (N_head, n_pv, N_res, 3),
    'T': (N_res, 4, 4),
    'att_scores': (N_head, N_res, N_res),
    'a': (N_res, c),
    's_initial': (N_res, c_s),
    'alpha': (N_res, n_torsion_angles, 2),
    'F': (N_res,),
}

batched_feature_shapes = {
    key: (N,) + value
    for key, value in feature_shapes.items()
}

test_inputs = {
    key: torch.linspace(-2-i/5, 2+i/5, math.prod(shape)).reshape(shape).double()
    for i, (key, shape) in enumerate(feature_shapes.items())
}

test_inputs['F'] = torch.arange(test_inputs['F'].numel()).reshape(test_inputs['F'].shape) % 20


def output_differences(out, expected_out):
    # Convert to numpy once
    a = out.numpy()
    b = expected_out.numpy()

    # 1. Get a boolean mask where elements are NOT close
    # Note the '~' which inverts the boolean array (True becomes False, etc.)
    mismatch_mask = ~np.isclose(a, b, rtol=1e-5, atol=0)


    # 2. Get the indices of the mismatches
    # np.argwhere returns an array of N-dimensional coordinates, which is very easy to read
    bad_indices = np.argwhere(mismatch_mask)

    # 3. (Optional) Print the actual vs expected values at those specific indices
    if len(bad_indices) > 0:
        print(f" ❌ Found {len(bad_indices)} problematic indices. ❌")
        print(" ❌ Indices:\n", bad_indices)

        for element in bad_indices[:10]:
            print("------------------------------")
            print(f"For Index {element}")
            print(f" - We Got  :: {a[tuple(element)]}")
            print(f" - Expect  :: {b[tuple(element)]}")
            print(f"Difference :: {np.abs(a[tuple(element)] - b[tuple(element)])}")
            print("------------------------------")

        print("\n")


def test_module_shape(module, test_name, control_folder):
    param_shapes = {name: param.shape for name, param in module.named_parameters()}

    shapes_path = f'{control_folder}/{test_name}_param_shapes.pt'
    expected_shapes = torch.load(shapes_path, weights_only=True)

    keys_in_dict1_only = set(param_shapes.keys()) - set(expected_shapes.keys())
    keys_in_dict2_only = set(expected_shapes.keys()) - set(param_shapes.keys())

    if keys_in_dict1_only or keys_in_dict2_only:
        error_message = "Parameter key set mismatch:\n"
        if keys_in_dict1_only:
            error_message += f"Unexpected keys: {keys_in_dict1_only}\n"
        if keys_in_dict2_only:
            error_message += f"Missing keys: {keys_in_dict2_only}\n"
        raise AssertionError(error_message)

    for name, param_shape in param_shapes.items():
        assert param_shape == expected_shapes[name], f'Invalid shape for parameter {name}.'


def controlled_execution(module, inp, method):
    was_training = module.training
    module.eval()
    module.double()
    with torch.no_grad():
        original_params = [param.clone() for param in module.parameters()]
        for param in module.parameters():
            param.copy_(torch.linspace(-1, 1, param.numel()).reshape(param.shape))

        out = method(*inp)

        for orig_param, param in zip(original_params, module.parameters()):
            param.copy_(orig_param)

    if was_training:
        module.train()

    return out


def controlled_forward(module, inp):
    return controlled_execution(module, inp, lambda *x: module(*x))


def test_module_method(module, test_name, input_names, output_names, control_folder, method, include_batched=True):
    if not isinstance(input_names, (tuple, list)):
        input_names = (input_names,)

    if not isinstance(output_names, (tuple, list)):
        output_names = (output_names,)

    non_batched_inps = tuple(test_inputs[name] for name in input_names)

    if include_batched:
        batched_inps = []
        for name in input_names:
            inp = test_inputs[name]
            if isinstance(inp, dict):
                batched_inp = {
                    key: value.broadcast_to(batched_feature_shapes[key])
                    for key, value in inp.items()
                }
            else:
                batched_inp = inp.broadcast_to(batched_feature_shapes[name])
            batched_inps.append(batched_inp)

        batched_inp = tuple(batched_inp)

    with torch.no_grad():
        non_batched_out = controlled_execution(module, non_batched_inps, method)
        if include_batched:
            batched_out = controlled_execution(module, batched_inps, method)

    if not isinstance(non_batched_out, (tuple, list)):
        non_batched_out = (non_batched_out,)
        if include_batched:
            batched_out = (batched_out,)

    out_file_names = [f'{control_folder}/{test_name}_{out_name}.pt' for out_name in output_names]
    for out, out_file_name, out_name in zip(non_batched_out, out_file_names, output_names):
        expected_out = torch.load(out_file_name, weights_only=True)
        # out = out.to(torch.float32)
        # expected_out = expected_out.to(torch.float32)
        # print(40 * "=")
        # print(f"Tensor Name Of Interest : {out_name}")
        # print_shape(name="Our Output", tensor=out)
        #
        # print_shape(name="Expected Output", tensor=expected_out)
        # print(f"Type For Output : {out.dtype}")
        # print(f"Type For Expected Output : {expected_out.dtype}")
        # print(40 * "=")

        output_differences(out, expected_out)
        assert torch.allclose(out, expected_out), f'Problem with output {out_name} in test {test_name} in non-batched check.'
        print(f"✅ ALL GOOD FOR {out_name} Tensor ✅")

    if include_batched:
        for out, out_file_name, out_name in zip(batched_out, out_file_names, output_names):

            expected_out = torch.load(out_file_name, weights_only=True)
            expected_out_batch_shape = (N,) + expected_out.shape
            expected_out = expected_out.unsqueeze(0).broadcast_to(expected_out_batch_shape)

            max_abs_err = (out.double() - expected_out.double()).abs().max()
            max_rel_err = ((out.double() - expected_out.double()).abs() / (
                        torch.maximum(out.double().abs(), expected_out.double().abs()) + 1e-8)).max()
            # # print(f'Max batched error: abs {max_abs_err}; rel {max_rel_err}')
            # print(60 * "=")
            # print(f"Tensor Name Of Interest : {out_name}")
            # print_shape(name="Our Output", tensor=out)
            # print_shape(name="Expected Output", tensor=expected_out)
            # print(60 * "=")
            # # if expected_out.ndim == 3 and out_name == "pseudo_beta_positions":
            # #     # print_tensor_list(expected_out[:10,:4, :])
            # #     print_tensor_list(out)
            # #     print(60 * "-")
            # #     print_tensor_list(expected_out)
            # #     print(60 * "-")
            # #     print_tensor_list(out.to(torch.int) - expected_out.to(torch.int), round=6)
            # #     print_tensor_list(torch.max(out.to(torch.int) - expected_out.to(torch.int)))
            # #     print_tensor_list(torch.min(out.to(torch.int) - expected_out.to(torch.int)))
            # output_differences(out, expected_out)
            assert torch.allclose(out,
                                  expected_out), f'Problem with output {out_name} in test {test_name} in batched check. Maximum absolute error: {max_abs_err}. Maximum relative error: {max_rel_err}.'
            print(f"✅ ALL GOOD FOR {out_name} Tensor ✅")

def test_module_forward(module, test_name, input_names, output_names, control_folder):
    test_module_method(module, test_name, input_names, output_names, control_folder, lambda *x: module(*x))
