#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include "ater_hip_common.h"

struct __attribute__((packed)) KernelArgs
{
    void *ptr_O;
    p2 _p0;
    void *ptr_In;
    p2 _p1;
    void *ptr_Weight;
    p2 _p2;
    void *ptr_Bias;
    p2 _p3;
    float epsilon;
    p3 _p4;
    unsigned int M;
    p3 _p5;
    unsigned int N;
    p3 _p6;
    void *ptr_OutResidual;
    p2 _p7;
    void *ptr_InResidual;
    p2 _p8;
    void *ptr_OutYScale;
    p2 _p9;
    void *ptr_XScale;
    p2 _p10;
};

void layernorm2d_with_add_asm(torch::Tensor &out,          // [m ,n]
                              torch::Tensor &input,        // [m ,n]
                              torch::Tensor &residual_in,  // [m ,n]
                              torch::Tensor &residual_out, // [m ,n]
                              torch::Tensor &weight,       // [1 ,n]
                              torch::Tensor &bias,         // [1 ,n]
                              float epsilon)
{
    auto dtype = input.dtype();
    TORCH_CHECK(dtype == torch::kBFloat16,
                __func__, " for now only support bf16 data type");
    TORCH_CHECK(input.is_contiguous(),
                __func__, " for now only support input.is_contiguous()");

    KernelArgs args;
    int n = input.size(-1);
    int m = input.numel() / n;

    size_t arg_size = sizeof(args);
    args.ptr_O = out.data_ptr();
    args.ptr_In = input.data_ptr();
    args.ptr_Weight = weight.data_ptr();
    args.ptr_Bias = bias.data_ptr();
    args.epsilon = epsilon;
    args.M = m;
    args.N = n;
    args.ptr_OutResidual = residual_out.data_ptr();
    args.ptr_InResidual = residual_in.data_ptr();

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int sub_M = 2;
    static AterAsmKernel impl("layer_norm_kernel_func", "layer_norm.co");

    impl.launch_kernel({&args,
                        &arg_size,
                        ((m + sub_M - 1) / sub_M), // gdx
                        1,                         // gdy
                        1,                         // gdz
                        256,                       // bdx: 4 wv64
                        1,                         // bdy
                        1,                         // bdz
                        stream});
}