# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random, skip_for_grayskull


def convert_scale_to_ttnn_tensor(device, scale, dim):
    scale_tensor = torch.tensor([scale])
    for i in range(1, dim):
        scale_tensor = torch.unsqueeze(scale_tensor, 0)
    return ttnn.from_torch(scale_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


# Quantize to the [q_min, q_max] range (scale-then-shift)
#   Quant:   q = t / s + z
#   Dequant: t = (q - z) * s
def calculate_scale_zero_point(torch_input_tensor, q_min, q_max):
    i_min = torch.min(torch_input_tensor).item()
    i_max = torch.max(torch_input_tensor).item()

    scale = (i_max - i_min) / (q_max - q_min)
    zero_point = q_min - int(i_min / scale)

    return (scale, zero_point)


# PCC can't catch the case that the output tensor is all zeros (or any other constant)
# Torch.allclose is too sensitive to outliers (e.g. quant-dequant of 3e-5 when most other values are around 1e-1)
# Instead, we assert that over 95% of the elements in the tensors are close enough
def get_match_ratio(golden, other):
    deduced_atol = 0.05 * torch.max(torch.abs(golden)).item()
    return torch.count_nonzero(torch.isclose(golden, other, rtol=0.05, atol=deduced_atol)) / torch.numel(golden)


min_match_ratio = 0.95


@pytest.mark.parametrize("n", [16, 31, 63, 128, 65536])
@pytest.mark.parametrize("scale_tensor_dim", [0, 1])
def test_quantize_1d(device, n, scale_tensor_dim):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand(n, dtype=torch.float32)
    scale, zero_point = calculate_scale_zero_point(torch_input_tensor, -128, 127)

    torch_quantized_tensor = torch.quantize_per_tensor(torch_input_tensor, scale, zero_point, dtype=torch.qint32)

    if scale_tensor_dim > 0:
        scale = convert_scale_to_ttnn_tensor(device, scale, scale_tensor_dim)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    quantized_tensor = ttnn.quantize(input_tensor, scale, zero_point)
    output_tensor = ttnn.to_torch(quantized_tensor)

    assert_with_pcc(torch_quantized_tensor.int_repr(), output_tensor)
    assert get_match_ratio(torch_quantized_tensor.int_repr(), output_tensor) > min_match_ratio


@pytest.mark.parametrize("n", [16, 31, 63, 128, 65536])
@pytest.mark.parametrize("scale_tensor_dim", [0, 1])
def test_dequantize_1d(device, n, scale_tensor_dim):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand(n, dtype=torch.float32)
    scale, zero_point = calculate_scale_zero_point(torch_input_tensor, -128, 127)

    torch_quantized_tensor = torch.quantize_per_tensor(torch_input_tensor, scale, zero_point, dtype=torch.qint32)
    torch_dequantized_tensor = torch.dequantize(torch_quantized_tensor)

    if scale_tensor_dim > 0:
        scale = convert_scale_to_ttnn_tensor(device, scale, scale_tensor_dim)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    quantized_tensor = ttnn.quantize(input_tensor, scale, zero_point)
    dequantized_tensor = ttnn.dequantize(quantized_tensor, scale, zero_point)
    output_tensor = ttnn.to_torch(dequantized_tensor)

    assert_with_pcc(torch_input_tensor, output_tensor)
    assert_with_pcc(torch_dequantized_tensor, output_tensor)
    assert get_match_ratio(torch_input_tensor.to(torch.bfloat16), output_tensor) > min_match_ratio
    assert get_match_ratio(torch_dequantized_tensor.to(torch.bfloat16), output_tensor) > min_match_ratio


@pytest.mark.parametrize("n", [16, 41, 37, 128, 65536])
def test_requantize_1d(device, n):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand(n, dtype=torch.float32)
    scale, zero_point = calculate_scale_zero_point(torch_input_tensor, -128, 127)
    scale_new, zero_point_new = calculate_scale_zero_point(torch_input_tensor, -37, 73)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    quantized_tensor = ttnn.quantize(input_tensor, scale, zero_point)
    requantized_tensor = ttnn.requantize(quantized_tensor, scale, zero_point, scale_new, zero_point_new)
    dequantized_tensor = ttnn.dequantize(requantized_tensor, scale_new, zero_point_new)
    output_tensor = ttnn.to_torch(dequantized_tensor)

    assert_with_pcc(torch_input_tensor, output_tensor)
    assert get_match_ratio(torch_input_tensor.to(torch.bfloat16), output_tensor) > min_match_ratio


@pytest.mark.parametrize("h", [16, 41, 37, 128])
@pytest.mark.parametrize("w", [16, 31, 63, 128])
@pytest.mark.parametrize("scale_tensor_dim", [0, 1, 2])
def test_quantize_2d(device, h, w, scale_tensor_dim):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand(h, w, dtype=torch.float32)
    scale, zero_point = calculate_scale_zero_point(torch_input_tensor, -128, 127)

    torch_quantized_tensor = torch.quantize_per_tensor(torch_input_tensor, scale, zero_point, dtype=torch.qint32)

    if scale_tensor_dim > 0:
        scale = convert_scale_to_ttnn_tensor(device, scale, scale_tensor_dim)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    quantized_tensor = ttnn.quantize(input_tensor, scale, zero_point)
    output_tensor = ttnn.to_torch(quantized_tensor)

    assert_with_pcc(torch_quantized_tensor.int_repr(), output_tensor)
    assert get_match_ratio(torch_quantized_tensor.int_repr(), output_tensor) > min_match_ratio


@pytest.mark.parametrize("h", [16, 41, 37, 128])
@pytest.mark.parametrize("w", [16, 31, 63, 128])
@pytest.mark.parametrize("scale_tensor_dim", [0, 1, 2])
def test_dequantize_2d(device, h, w, scale_tensor_dim):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand(h, w, dtype=torch.float32)
    scale, zero_point = calculate_scale_zero_point(torch_input_tensor, -128, 127)

    torch_quantized_tensor = torch.quantize_per_tensor(torch_input_tensor, scale, zero_point, dtype=torch.qint32)
    torch_dequantized_tensor = torch.dequantize(torch_quantized_tensor)

    if scale_tensor_dim > 0:
        scale = convert_scale_to_ttnn_tensor(device, scale, scale_tensor_dim)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    quantized_tensor = ttnn.quantize(input_tensor, scale, zero_point)
    dequantized_tensor = ttnn.dequantize(quantized_tensor, scale, zero_point)
    output_tensor = ttnn.to_torch(dequantized_tensor)

    assert_with_pcc(torch_input_tensor, output_tensor)
    assert_with_pcc(torch_dequantized_tensor, output_tensor)
    assert get_match_ratio(torch_input_tensor.to(torch.bfloat16), output_tensor) > min_match_ratio
    assert get_match_ratio(torch_dequantized_tensor.to(torch.bfloat16), output_tensor) > min_match_ratio


@pytest.mark.parametrize("h", [16, 41, 37, 128])
@pytest.mark.parametrize("w", [16, 31, 63, 128])
def test_requantize_2d(device, h, w):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand(h, w, dtype=torch.float32)
    scale, zero_point = calculate_scale_zero_point(torch_input_tensor, -128, 127)
    scale_new, zero_point_new = calculate_scale_zero_point(torch_input_tensor, -37, 73)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    quantized_tensor = ttnn.quantize(input_tensor, scale, zero_point)
    requantized_tensor = ttnn.requantize(quantized_tensor, scale, zero_point, scale_new, zero_point_new)
    dequantized_tensor = ttnn.dequantize(requantized_tensor, scale_new, zero_point_new)
    output_tensor = ttnn.to_torch(dequantized_tensor)

    assert_with_pcc(torch_input_tensor, output_tensor)
    assert get_match_ratio(torch_input_tensor.to(torch.bfloat16), output_tensor) > min_match_ratio


@pytest.mark.parametrize("x0", [128])
@pytest.mark.parametrize("x1", [17])
@pytest.mark.parametrize("x2", [3])
@pytest.mark.parametrize("x3", [64])
@pytest.mark.parametrize("scale_tensor_dim", [0, 1, 2, 3, 4])
def test_quantize_4d(device, x0, x1, x2, x3, scale_tensor_dim):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand(x0, x1, x2, x3, dtype=torch.float32)
    scale, zero_point = calculate_scale_zero_point(torch_input_tensor, -128, 127)

    torch_quantized_tensor = torch.quantize_per_tensor(torch_input_tensor, scale, zero_point, dtype=torch.qint32)

    if scale_tensor_dim > 0:
        scale = convert_scale_to_ttnn_tensor(device, scale, scale_tensor_dim)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    quantized_tensor = ttnn.quantize(input_tensor, scale, zero_point)
    output_tensor = ttnn.to_torch(quantized_tensor)

    assert_with_pcc(torch_quantized_tensor.int_repr(), output_tensor)
    assert get_match_ratio(torch_quantized_tensor.int_repr(), output_tensor) > min_match_ratio


@pytest.mark.parametrize("x0", [128])
@pytest.mark.parametrize("x1", [17])
@pytest.mark.parametrize("x2", [3])
@pytest.mark.parametrize("x3", [64])
@pytest.mark.parametrize("scale_tensor_dim", [0, 1, 2, 3, 4])
def test_dequantize_4d(device, x0, x1, x2, x3, scale_tensor_dim):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand(x0, x1, x2, x3, dtype=torch.float32)
    scale, zero_point = calculate_scale_zero_point(torch_input_tensor, -128, 127)

    torch_quantized_tensor = torch.quantize_per_tensor(torch_input_tensor, scale, zero_point, dtype=torch.qint32)
    torch_dequantized_tensor = torch.dequantize(torch_quantized_tensor)

    if scale_tensor_dim > 0:
        scale = convert_scale_to_ttnn_tensor(device, scale, scale_tensor_dim)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    quantized_tensor = ttnn.quantize(input_tensor, scale, zero_point)
    dequantized_tensor = ttnn.dequantize(quantized_tensor, scale, zero_point)
    output_tensor = ttnn.to_torch(dequantized_tensor)

    assert_with_pcc(torch_input_tensor, output_tensor)
    assert_with_pcc(torch_dequantized_tensor, output_tensor)
    assert get_match_ratio(torch_input_tensor.to(torch.bfloat16), output_tensor) > min_match_ratio
    assert get_match_ratio(torch_dequantized_tensor.to(torch.bfloat16), output_tensor) > min_match_ratio


@pytest.mark.parametrize("x0", [128])
@pytest.mark.parametrize("x1", [17])
@pytest.mark.parametrize("x2", [3])
@pytest.mark.parametrize("x3", [64])
def test_requantize_4d(device, x0, x1, x2, x3):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand(x0, x1, x2, x3, dtype=torch.float32)
    scale, zero_point = calculate_scale_zero_point(torch_input_tensor, -128, 127)
    scale_new, zero_point_new = calculate_scale_zero_point(torch_input_tensor, -37, 73)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    quantized_tensor = ttnn.quantize(input_tensor, scale, zero_point)
    requantized_tensor = ttnn.requantize(quantized_tensor, scale, zero_point, scale_new, zero_point_new)
    dequantized_tensor = ttnn.dequantize(requantized_tensor, scale_new, zero_point_new)
    output_tensor = ttnn.to_torch(dequantized_tensor)

    assert_with_pcc(torch_input_tensor, output_tensor)
    assert get_match_ratio(torch_input_tensor.to(torch.bfloat16), output_tensor) > min_match_ratio


# TODO: scale-to-tensor this!
@pytest.mark.parametrize("dim", [1, 2, 3, 4])
def test_quantization_program_cache(device, use_program_cache, dim):
    torch.manual_seed(0)

    num_program_cache_entries_list = []

    for i in range(3):
        # Each iteration gets completely different input tensors, quant ranges, etc.
        torch_input_tensor = torch.rand([30 + i] * dim, dtype=torch.float32)

        scale, zero_point = calculate_scale_zero_point(torch_input_tensor, -120 + i, 121 - i)
        scale_new, zero_point_new = calculate_scale_zero_point(torch_input_tensor, -50 - i, 42 + i)

        input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        quantized_tensor = ttnn.quantize(input_tensor, scale, zero_point)
        requantized_tensor = ttnn.requantize(quantized_tensor, scale, zero_point, scale_new, zero_point_new)
        dequantized_tensor = ttnn.dequantize(requantized_tensor, scale_new, zero_point_new)
        output_tensor = ttnn.to_torch(dequantized_tensor)

        assert_with_pcc(torch_input_tensor, output_tensor)
        assert get_match_ratio(torch_input_tensor.to(torch.bfloat16), output_tensor) > min_match_ratio

        num_program_cache_entries_list.append(device.num_program_cache_entries())

    assert num_program_cache_entries_list[0] > 0
    assert num_program_cache_entries_list[0] == num_program_cache_entries_list[1]
    assert num_program_cache_entries_list[0] == num_program_cache_entries_list[2]
