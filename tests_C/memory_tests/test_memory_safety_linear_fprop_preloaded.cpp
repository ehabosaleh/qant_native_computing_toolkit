/**
 * This file is a standalone executable, to be used for the memory leak check
 * with valgrind.
 */

#include <xtensor/xrandom.hpp>
#include <xtensor/xtensor.hpp>

#include "dlpack_utils.h"
#include "qant_native_computing_toolkit.h"

typedef xt::xarray<std::bfloat16_t> Tensor;

namespace q_generic = ::qant_native_computing_toolkit::generic;
namespace q_native = ::qant_native_computing_toolkit::native;

int main()
{
    auto npu_id = 0;
    auto n_channels_out = 4;
    auto n_channels_in = 3;
    auto n_batches = 3;

    Tensor weights = xt::random::rand({n_channels_out, n_channels_in}, -1.0, 1.0);
    Tensor features = xt::random::rand({n_batches, n_channels_in}, -1.0, 1.0);
    auto weights_dltensor = xarray_to_dltensor(weights);
    auto features_dltensor = xarray_to_dltensor(features);

    auto weights_handle = q_generic::npu_allocate(npu_id, n_channels_out, n_channels_in, q_generic::QantDataRole::Weights);
    auto features_handle = q_generic::npu_allocate(npu_id, n_batches, n_channels_in, q_generic::QantDataRole::Features);
    auto result_handle = q_generic::npu_allocate(npu_id, n_batches, n_channels_out, q_generic::QantDataRole::Features);

    q_generic::npu_upload_into(weights_handle, &weights_dltensor);
    q_generic::npu_upload_into(features_handle, &features_dltensor);

    q_native::linear_fprop_preloaded(features_handle, weights_handle, result_handle);

    weights_dltensor.deleter(&weights_dltensor);
    features_dltensor.deleter(&features_dltensor);
    q_generic::npu_free(weights_handle);
    q_generic::npu_free(features_handle);
    q_generic::npu_free(result_handle);
    q_generic::release_npu(npu_id);

    return 0;
}
