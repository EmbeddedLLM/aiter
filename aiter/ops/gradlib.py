# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from typing import Optional
from functools import lru_cache
import os

from torch._C import NoneType
from ..jit.core import compile_ops, AITER_CONFIG_HIPB_MM_TUNED_SOLUTIONS_FILE, AITER_ROOT_DIR
from ..jit.utils.chip_info import get_cu_num


@compile_ops("module_hipbsolgemm")
def hipb_create_extension() -> None: ...


@compile_ops("module_hipbsolgemm")
def hipb_destroy_extension() -> None: ...


def gen_hipb_mm_fake_tensor(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    solution_index: int,
    bias: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    scaleA: Optional[torch.Tensor] = None,
    scaleB: Optional[torch.Tensor] = None,
    scaleOut: Optional[torch.Tensor] = None,
    bpreshuffle: Optional[bool] = None,
):
    mat1_sizes = mat1.size()
    mat2_sizes = mat2.size()
    in_dtype = mat1.dtype
    out_dtype = out_dtype if out_dtype is not None else in_dtype
    result = torch.empty(
        (mat1_sizes[0], mat2_sizes[1]), dtype=out_dtype, device=mat1.device
    )

    return result


@compile_ops("module_hipbsolgemm", gen_fake=gen_hipb_mm_fake_tensor)
def hipb_mm(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    solution_index: int,
    bias: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    scaleA: Optional[torch.Tensor] = None,
    scaleB: Optional[torch.Tensor] = None,
    scaleOut: Optional[torch.Tensor] = None,
    bpreshuffle: Optional[bool] = None,
) -> torch.Tensor: ...


@compile_ops("module_hipbsolgemm", gen_fake=gen_hipb_mm_fake_tensor)
def _hipb_mm_tuned_internal(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    scaleA: Optional[torch.Tensor] = None,
    scaleB: Optional[torch.Tensor] = None,
    scaleOut: Optional[torch.Tensor] = None,
    bpreshuffle: Optional[bool] = None,
) -> torch.Tensor: ...


@compile_ops("module_hipbsolgemm")
def _hipb_load_solution_cache_internal(csv_path: str, cu_num: int) -> None: ...


@compile_ops("module_hipbsolgemm")
def hipb_findallsols(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    scaleA: Optional[torch.Tensor] = None,
    scaleB: Optional[torch.Tensor] = None,
    scaleC: Optional[torch.Tensor] = None,
    bpreshuffle: bool = False,
) -> list[int]: ...


_hipb_solution_cache_loaded = False


@lru_cache(maxsize=1)
def _ensure_hipb_solution_cache_loaded():
    """Ensure the C++ solution cache is loaded once"""
    global _hipb_solution_cache_loaded
    if not _hipb_solution_cache_loaded:
        sol_file = AITER_CONFIG_HIPB_MM_TUNED_SOLUTIONS_FILE
        cu_num = get_cu_num()
        if os.path.exists(sol_file):
            _hipb_load_solution_cache_internal(sol_file, cu_num)
        _hipb_solution_cache_loaded = True


def hipb_mm_tuned_cpp(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    scaleA: Optional[torch.Tensor] = None,
    scaleB: Optional[torch.Tensor] = None,
    scaleOut: Optional[torch.Tensor] = None,
    bpreshuffle: Optional[bool] = None,
) -> torch.Tensor:
    """
    Optimized tuned GEMM that automatically selects the best solution.
    Uses C++ solution cache for minimal overhead.
    """
    # Ensure C++ cache is loaded on first call
    _ensure_hipb_solution_cache_loaded()

    # Optional: capture running shapes if enabled (for tuning)
    capture_running_shapes, tune_file = _hipb_mm_tuned_capture_running_shapes()
    if capture_running_shapes:
        m = mat1.size(0)
        n = mat2.size(1)
        k = mat1.size(1)
        _in_dtype = str(mat1.dtype)
        _out_dtype = str(out_dtype) if out_dtype is not None else _in_dtype
        _bias = True if bias is not None else False

        def _determine_scale_type(scale: Optional[torch.Tensor], mat_dim: int, dim1: int, dim2: int) -> str:
            if scale is None:
                return "no_scale"
            if scale.shape[dim1] == mat_dim and scale.shape[dim2] == 1:
                return "rowwise"
            elif scale.shape[dim1] == 1 and scale.shape[dim2] == 1:
                return "scalar"
            else:
                return "no_scale"

        scale_a_type = _determine_scale_type(scaleA, m, 0, 1)
        scale_b_type = _determine_scale_type(scaleB, n, 1, 0)
        _bpreshuffle = bpreshuffle if bpreshuffle is not None else False
        _save_tuning_config(tune_file, m, n, k, _in_dtype, _out_dtype, _bias, scale_a_type, scale_b_type, _bpreshuffle)

    # Call the C++ implementation which does fast lookup
    return _hipb_mm_tuned_internal(mat1, mat2, bias, out_dtype, scaleA, scaleB, scaleOut, bpreshuffle)


_unique_shapes_cache = None

_hipb_mm_tuned_sols_cache = None


@lru_cache(maxsize=1)
def get_hipb_mm_tuned_sols() -> dict:
    global _hipb_mm_tuned_sols_cache
    if _hipb_mm_tuned_sols_cache is None:
        import pandas as pd
        sol_file = AITER_CONFIG_HIPB_MM_TUNED_SOLUTIONS_FILE
        cu_num = get_cu_num()
        if not os.path.exists(sol_file):
            return {}
        df = pd.read_csv(sol_file, keep_default_na=False)
        _hipb_mm_tuned_sols_cache =  df[df["cu_num"] == cu_num].set_index(["M", "N", "K", "in_dtype", "out_dtype", "bias", "scale_A_type", "scale_B_type", "B_preshuffled"])["best_solution_idx"].to_dict()
        # print(_hipb_mm_tuned_sols_cache)
        # return _hipb_mm_tuned_sols_cache
    # return df.set_index(["cu_num", "M", "N", "K", "in_dtype", "out_dtype", "bias", "scale_A_type", "scale_B_type", "B_preshuffled"]).to_dict("index")
    # print(f"!!!!!!!!!!!!!!!!!!!!!!!!!{_hipb_mm_tuned_sols_cache}")
    return _hipb_mm_tuned_sols_cache


def _determine_scale_type(scale: Optional[torch.Tensor], mat_mn: int, scale_mn_dim: int, scale_k_dim: int) -> str:
    if scale is None:
        return "no_scale"
    if scale.shape[scale_mn_dim] == mat_mn and scale.shape[scale_k_dim] == 1:
        return "rowwise"
    elif scale.shape[scale_mn_dim] == 1 and scale.shape[scale_k_dim] == 1:
        return "scalar"
    else:
        return "no_scale"


def hipb_mm_tuned_python(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    scaleA: Optional[torch.Tensor] = None,
    scaleB: Optional[torch.Tensor] = None,
    scaleOut: Optional[torch.Tensor] = None,
    bpreshuffle: Optional[bool] = None,
) -> torch.Tensor:
    tuned_sols = get_hipb_mm_tuned_sols()
    m = mat1.size(0)
    n = mat2.size(1)
    k = mat1.size(1)
    _in_dtype = str(mat1.dtype)
    _out_dtype = str(out_dtype) if out_dtype is not None else _in_dtype
    _bias = True if bias is not None else False

    scale_a_type = _determine_scale_type(scaleA, m, 0, 1)
    scale_b_type = _determine_scale_type(scaleB, n, 1, 0)
    _bpreshuffle = bpreshuffle if bpreshuffle is not None else False
    
    capture_running_shapes, tune_file = _hipb_mm_tuned_capture_running_shapes()
    if capture_running_shapes:
        _save_tuning_config(tune_file, m, n, k, _in_dtype, _out_dtype, _bias, scale_a_type, scale_b_type, _bpreshuffle)
        
        # global _unique_shapes_cache
        # in_type_str = "fp8" if mat1.dtype == torch.float8_e4m3fnuz else "bf16"
        # scale_a_str = "rowwise" if scaleA is not None else "no_scale"
        # scale_b_str = "rowwise" if scaleB is not None else "no_scale"
        # cond = f"{m},{n},{k},{in_type_str},bf16,{_bias},{scale_a_str},{scale_b_str},{_bpreshuffle}"
        # if cond not in _unique_shapes_cache:
        #     _unique_shapes_cache.add(cond)
        #     print(f"!!!!!!!!!!!!!!!!!!!!!!!!!{cond}")

    # def _next_power_of_2(x: int) -> int:
    #     return 1 << (x - 1).bit_length()

    # def _next_multiple_of(x: int, y: int) -> int:
    #     return ((x + y - 1) // y) * y

    # if m < 512:
    #     m = _next_multiple_of(m, 8)
    # else:
    #     m = _next_power_of_2(m)

    key = (m, n, k, _in_dtype, _out_dtype, _bias, scale_a_type, scale_b_type, _bpreshuffle)
    if key in tuned_sols:
        best_solution_idx = tuned_sols[key]
        print(f"Using tuned solution {best_solution_idx} for key {key}")
    else:
        best_solution_idx = -1
        # print(f"No tuned solution found for key {key}")
    # best_solution_idx = tuned_sols.get(key, -1)

    return hipb_mm(mat1, mat2, best_solution_idx, bias, out_dtype, scaleA, scaleB, scaleOut, _bpreshuffle)


@lru_cache(maxsize=1)
def _hipb_mm_tuned_capture_running_shapes() -> bool:
    tune_file = os.environ.get("AITER_HIPB_MM_ONLINE_TUNE_FILE", f"{AITER_ROOT_DIR}/aiter/configs/hipb_mm_shapes_to_tune.csv")
    if not os.path.exists(tune_file):
        try:
            with open(tune_file, "w") as f:
                f.write("M,N,K,in_dtype,out_dtype,bias,scale_A_type,scale_B_type,B_preshuffled\n")
        except FileExistsError:
            pass
        except Exception as e:
            print(f"Error creating tune file: {e}")
            return False, None
    global _unique_shapes_cache
    if _unique_shapes_cache is None:
        _unique_shapes_cache = set()
    return os.environ.get("AITER_HIPB_MM_ONLINE_TUNE", "0") == "1", tune_file


def _save_tuning_config(tuned_file: str, m: int, n: int, k: int, in_dtype: str, out_dtype: str, bias: bool, scale_a_type: str, scale_b_type: str, bpreshuffle: bool) -> int:
    config = f"{m},{n},{k},{in_dtype},{out_dtype},{bias},{scale_a_type},{scale_b_type},{bpreshuffle}"
    global _unique_shapes_cache
    if config not in _unique_shapes_cache:
        _unique_shapes_cache.add(config)
        with open(tuned_file, "a") as f:
            f.write(f"{config}\n")
    

@compile_ops("module_hipbsolgemm")
def getHipblasltKernelName() -> None: ...


@compile_ops("module_rocsolgemm")
def rocb_create_extension() -> None: ...


@compile_ops("module_rocsolgemm")
def rocb_destroy_extension() -> None: ...


def gen_rocb_mm_fake_tensor(
    arg0: torch.Tensor, arg1: torch.Tensor, arg2: int
) -> torch.Tensor:
    mat1_sizes = arg0.size()
    mat2_sizes = arg0.size()
    in_dtype = arg0.dtype
    result = torch.empty(
        (mat1_sizes[0], mat2_sizes[1]), dtype=in_dtype, device=arg0.device
    )

    return result


@compile_ops("module_rocsolgemm", gen_fake=gen_rocb_mm_fake_tensor)
def rocb_mm(arg0: torch.Tensor, arg1: torch.Tensor, arg2: int) -> torch.Tensor: ...


@compile_ops("module_rocsolgemm")
def rocb_findallsols(arg0: torch.Tensor, arg1: torch.Tensor) -> list[int]: ...


hipb_mm_tuned = hipb_mm_tuned_cpp
