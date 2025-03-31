// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "quantization.hpp"
#include "ttnn/operations/eltwise/binary_ng/binary_ng.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"

namespace ttnn::operations::quantization {

Tensor QuantOp::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const float scale,
    const int32_t zero_point,
    const std::optional<int32_t> axis,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    const ttnn::DataType a_dtype = input_tensor.get_dtype();
    const bool typecast_a = needs_typecast_to_bfloat16(a_dtype);
    Tensor input_a = typecast_a ? typecast_to(DataType::BFLOAT16, input_tensor) : input_tensor;

    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations{};
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations{};

    std::array<const ttnn::operations::unary::UnaryWithParam, 1> post_activations{
        ttnn::operations::unary::UnaryWithParam{unary::UnaryOpType::ZERO_POINT, static_cast<float>(zero_point)}};

    std::cout << "++ QT scale: " << scale << std::endl;

    // LLK quant kernel expects the reciprocal of the actual scale to avoid doing div on the device
    return ttnn::prim::binary_ng(
        queue_id,
        input_a,
        1.0f / scale,
        binary_ng::BinaryOpType::QUANT,
        output_dtype.value_or(DataType::INT32),
        memory_config,
        optional_output_tensor,
        lhs_activations,
        rhs_activations,
        post_activations);
}

Tensor QuantOp::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const Tensor& scale,
    const int32_t zero_point,
    const std::optional<int32_t> axis,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    TT_FATAL(scale.logical_shape().volume() == 1u, "Scale and zero-point must have matching shapes");

    const ttnn::DataType a_dtype = input_tensor.get_dtype();
    const ttnn::DataType b_dtype = scale.get_dtype();
    const bool typecast_a = needs_typecast_to_bfloat16(a_dtype);
    const bool typecast_b = needs_typecast_to_bfloat16(b_dtype);
    Tensor input_a = typecast_a ? typecast_to(DataType::BFLOAT16, input_tensor) : input_tensor;
    Tensor input_b = typecast_a ? typecast_to(DataType::BFLOAT16, scale) : scale;

    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations{};
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations{};

    std::array<const ttnn::operations::unary::UnaryWithParam, 1> post_activations{
        ttnn::operations::unary::UnaryWithParam{unary::UnaryOpType::ZERO_POINT, static_cast<float>(zero_point)}};

    Tensor inv = ttnn::reciprocal(input_b);

    // inv.print();
    inv.print_info("++ QT scale:");

    // LLK quant kernel expects the reciprocal of the actual scale to avoid doing div on the device
    return ttnn::prim::binary_ng(
        queue_id,
        input_a,
        inv,
        binary_ng::BinaryOpType::QUANT,
        output_dtype.value_or(DataType::INT32),
        memory_config,
        optional_output_tensor,
        lhs_activations,
        rhs_activations,
        post_activations);
}

Tensor RequantOp::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const float in_scale,
    const int32_t in_zero_point,
    const float out_scale,
    const int32_t out_zero_point,
    const std::optional<int32_t> axis,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    const ttnn::DataType a_dtype = input_tensor.get_dtype();
    const bool typecast_a = needs_typecast_to_bfloat16(a_dtype);
    Tensor input_a = typecast_a ? typecast_to(DataType::BFLOAT16, input_tensor) : input_tensor;

    // Expansion of q' = [(q - z_in) * s_in] / s_out + z_out
    const float scale = out_scale / in_scale;
    const int32_t zero_point = out_zero_point - in_zero_point / scale;

    std::cout << "++ Requant: new scale " << scale << " new 1/scale " << (1.0f / scale) << " new zero-point "
              << zero_point << std::endl;

    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations{};
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations{};

    std::array<const ttnn::operations::unary::UnaryWithParam, 1> post_activations{
        ttnn::operations::unary::UnaryWithParam{unary::UnaryOpType::ZERO_POINT, static_cast<float>(zero_point)}};

    return ttnn::prim::binary_ng(
        queue_id,
        input_a,
        1.0f / scale,
        binary_ng::BinaryOpType::REQUANT,
        output_dtype.value_or(DataType::INT32),
        memory_config,
        optional_output_tensor,
        lhs_activations,
        rhs_activations,
        post_activations);
}

Tensor DequantOp::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const float scale,
    const int32_t zero_point,
    const std::optional<int32_t> axis,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    const ttnn::DataType a_dtype = input_tensor.get_dtype();
    const bool typecast_a = needs_typecast_to_bfloat16(a_dtype);
    Tensor input_a = typecast_a ? typecast_to(DataType::BFLOAT16, input_tensor) : input_tensor;

    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations{};
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations{};

    // LLK dequant kernel does addition, so we need to negate zero_point
    std::array<const ttnn::operations::unary::UnaryWithParam, 1> post_activations{
        ttnn::operations::unary::UnaryWithParam{unary::UnaryOpType::ZERO_POINT, static_cast<float>(-zero_point)}};

    std::cout << "++ DQ scale: " << scale << std::endl;

    return ttnn::prim::binary_ng(
        queue_id,
        input_a,
        scale,
        binary_ng::BinaryOpType::DEQUANT,
        output_dtype.value_or(DataType::BFLOAT16),
        memory_config,
        optional_output_tensor,
        lhs_activations,
        rhs_activations,
        post_activations);
}

Tensor DequantOp::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const Tensor& scale,
    const int32_t zero_point,
    const std::optional<int32_t> axis,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    TT_FATAL(scale.logical_shape().volume() == 1u, "Scale and zero-point must have matching shapes");

    const ttnn::DataType a_dtype = input_tensor.get_dtype();
    const ttnn::DataType b_dtype = scale.get_dtype();
    const bool typecast_a = needs_typecast_to_bfloat16(a_dtype);
    const bool typecast_b = needs_typecast_to_bfloat16(b_dtype);
    Tensor input_a = typecast_a ? typecast_to(DataType::BFLOAT16, input_tensor) : input_tensor;
    Tensor input_b = typecast_b ? typecast_to(DataType::BFLOAT16, scale) : scale;

    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations{};
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations{};

    // LLK dequant kernel does addition, so we need to negate zero_point
    std::array<const ttnn::operations::unary::UnaryWithParam, 1> post_activations{
        ttnn::operations::unary::UnaryWithParam{unary::UnaryOpType::ZERO_POINT, static_cast<float>(-zero_point)}};

    // input_b.print();
    input_b.print_info("++ DQ scale:");

    return ttnn::prim::binary_ng(
        queue_id,
        input_a,
        input_b,
        binary_ng::BinaryOpType::DEQUANT,
        output_dtype.value_or(DataType::BFLOAT16),
        memory_config,
        optional_output_tensor,
        lhs_activations,
        rhs_activations,
        post_activations);
}

}  // namespace ttnn::operations::quantization
