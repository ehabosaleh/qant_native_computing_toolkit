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
    Tensor us = xt::cast<std::bfloat16_t>(xt::random::rand<float>({16}, -1.0, 1.0));
    Tensor vs = xt::cast<std::bfloat16_t>(xt::random::rand<float>({16}, -1.0, 1.0));

    auto us_dltensor = xarray_to_dltensor(us);
    auto vs_dltensor = xarray_to_dltensor(vs);

    auto qant_res_dltensor = q_native::mul_npu(npu_id, &us_dltensor, &vs_dltensor);
    q_generic::release_npu(npu_id);

    us_dltensor.deleter(&us_dltensor);
    vs_dltensor.deleter(&vs_dltensor);

    // This calls the deleter of the tensor that was created by the rust library
    qant_res_dltensor->deleter(qant_res_dltensor);

    return 0;
}
