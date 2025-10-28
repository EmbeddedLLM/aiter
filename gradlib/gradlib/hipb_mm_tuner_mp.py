"""
* Copyright (C) Advanced Micro Devices, Inc. All rights reserved.
* Copyright (C) 2024-2025, The vLLM team.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import time
import multiprocessing as mp
from functools import partial

import torch
import aiter
from aiter import dtypes
import pandas as pd

from HipbMmTuner import HipbMmTuner
from aiter.jit.core import AITER_CONFIG_GEMM_A8W8_FILE, AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE, AITER_CONFIG_HIPB_MM_TUNED_SOLUTIONS_FILE

DEFAULT_HIPB_MM_OUTPUT = AITER_CONFIG_HIPB_MM_TUNED_SOLUTIONS_FILE


def generate_mk_sets(model_dir, tp=1):
    """Generate M, K sets from model config for hipb_mm tuning.

    Returns:
        tuple: (list of (m, k) tuples, hidden_size, dtype)
    """
    with open(f"{model_dir}/config.json") as f:
        data = json.load(f)
        hidden_size = data["hidden_size"]
        intermediate_size = data["intermediate_size"]
        total_num_heads = data["num_attention_heads"]
        total_num_kv_heads = data["num_key_value_heads"]
        dtype = get_dtype(data["torch_dtype"])
        head_dim = hidden_size // total_num_heads
    return (
        [
            # QKV projection: (num_heads + 2*num_kv_heads) * head_dim
            (
                (total_num_heads + (2 * total_num_kv_heads)) * head_dim // tp,
                hidden_size,
            ),
            # Output projection
            (hidden_size, hidden_size // tp),
            # MLP gate + up projection (2x intermediate for SwiGLU)
            (intermediate_size * 2 // tp, hidden_size),
            # MLP down projection
            (hidden_size, intermediate_size // tp),
        ],
        hidden_size,
        dtype,
    )


dtype_map = {
    "f32": dtypes.fp32,
    "float32": dtypes.fp32,
    "f16": dtypes.fp16,
    "float16": dtypes.fp16,
    "bf16": dtypes.bf16,
    "bfloat16": dtypes.bf16,
    "fp8": dtypes.fp8,
}


def get_dtype(dtype_str: str):
    """Convert dtype string to torch/aiter dtype."""
    if dtype_str is None:
        return None
    if dtype_str.startswith("torch"):
        return getattr(torch, dtype_str.split(".")[1])
    if dtype_str in dtype_map:
        return dtype_map[dtype_str]
    else:
        print(f">>> Warning! Invalid dtype {dtype_str}, using default dtype fp8")
    return dtypes.fp8


def list_of_ints(arg):
    """Parse comma-separated integers."""
    return list(map(int, arg.split(",")))


def worker_process(gpu_id, problem_chunk, output_file, err_ratio, warmup_iters, test_iters, fast_mode):
    """
    Worker process that tunes a chunk of problems on a specific GPU.

    Args:
        gpu_id: GPU device ID to use
        problem_chunk: DataFrame containing problems to tune
        output_file: Path to output CSV file
        err_ratio: Maximum acceptable error ratio
        warmup_iters: Number of warmup iterations
        test_iters: Number of test iterations
        fast_mode: Whether to use fast mode
    """
    try:
        # Set CUDA device for this worker
        # Don't use CUDA_VISIBLE_DEVICES as it can cause issues with multiprocessing
        # Instead, just set the device directly
        torch.cuda.set_device(gpu_id)

        # Initialize hipBLASLt extension
        aiter.hipb_create_extension()

        print(f"[GPU {gpu_id}] Worker started with {len(problem_chunk)} problems")
        print(f"[GPU {gpu_id}] CUDA device: {torch.cuda.current_device()}")
        print(f"[GPU {gpu_id}] Device name: {torch.cuda.get_device_name()}")

        # Create tuner for this worker
        # Note: Each worker creates its own tuner instance but shares the same output CSV
        tuner = HipbMmTuner(
            output_csv=output_file,
            err_ratio=err_ratio,
            warmup_iters=warmup_iters,
            test_iters=test_iters,
            fast_mode=fast_mode,
            rank_id=gpu_id,
        )

        # Set the problems for this worker
        tuner.problems_df = problem_chunk.reset_index(drop=True)

        # Start tuning
        start_time = time.time()
        tuner.tune_all()
        elapsed = time.time() - start_time

        print(f"[GPU {gpu_id}] Completed tuning in {elapsed:.2f}s ({elapsed/60:.2f}min)")
        return gpu_id, len(problem_chunk), elapsed

    except Exception as e:
        print(f"[GPU {gpu_id}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return gpu_id, 0, -1


def filter_already_tuned_problems(problems_df, output_file, cu_num):
    """
    Filter out problems that have already been tuned for the current GPU.

    Args:
        problems_df: DataFrame containing all problems
        output_file: Path to output CSV file with existing results
        cu_num: Current GPU CU count

    Returns:
        tuple: (filtered_df, skipped_count)
    """
    from pathlib import Path

    if not Path(output_file).exists():
        print("No existing results file - all problems need tuning")
        return problems_df, 0

    try:
        results_df = pd.read_csv(output_file)
        if len(results_df) == 0:
            print("Existing results file is empty - all problems need tuning")
            return problems_df, 0

        # Filter to current GPU
        results_for_gpu = results_df[results_df["cu_num"] == cu_num]
        if len(results_for_gpu) == 0:
            print(f"No existing results for current GPU (CU={cu_num}) - all problems need tuning")
            return problems_df, 0

        print(f"\nFound {len(results_for_gpu)} existing results for current GPU (CU={cu_num})")

        # Check each problem
        problems_to_tune = []
        skipped_count = 0

        # Normalize scale types - pandas reads "None" as NaN
        def normalize_scale(val):
            if pd.isna(val) or str(val).lower() in ['none', 'nan']:
                return 'None'
            return str(val)

        for idx, problem_row in problems_df.iterrows():
            # Normalize boolean values
            bias_val = bool(problem_row["bias"])
            preshuffled_val = bool(
                problem_row["B_preshuffled"] == 'Y' or
                problem_row["B_preshuffled"] == True or
                problem_row["B_preshuffled"] == 'True'
            )

            # Normalize scale types for comparison
            problem_scale_a = normalize_scale(problem_row["scale_A_type"])
            problem_scale_b = normalize_scale(problem_row["scale_B_type"])

            # Check if exact match exists
            mask = (
                (results_for_gpu["M"] == int(problem_row["M"])) &
                (results_for_gpu["N"] == int(problem_row["N"])) &
                (results_for_gpu["K"] == int(problem_row["K"])) &
                (results_for_gpu["in_dtype"] == str(problem_row["in_dtype"])) &
                (results_for_gpu["out_dtype"] == str(problem_row["out_dtype"])) &
                (results_for_gpu["bias"] == bias_val) &
                (results_for_gpu["scale_A_type"].apply(normalize_scale) == problem_scale_a) &
                (results_for_gpu["scale_B_type"].apply(normalize_scale) == problem_scale_b) &
                (results_for_gpu["B_preshuffled"] == preshuffled_val)
            )

            if not mask.any():
                problems_to_tune.append(problem_row)
            else:
                skipped_count += 1

        if len(problems_to_tune) == 0:
            print(f"All {len(problems_df)} problems already tuned for this GPU!")
            return pd.DataFrame(), skipped_count

        filtered_df = pd.DataFrame(problems_to_tune).reset_index(drop=True)
        print(f"Filtered: {len(filtered_df)} problems need tuning, {skipped_count} already tuned")

        return filtered_df, skipped_count

    except Exception as e:
        print(f"Warning: Could not filter problems: {e}")
        print("Proceeding with all problems")
        return problems_df, 0


def split_problems_across_gpus(problems_df, num_gpus):
    """
    Split problems across GPUs in an interleaved manner.

    This ensures that if problems are grouped by difficulty in the input,
    each GPU gets a mix of easy and hard problems.

    Example with 10 problems and 3 GPUs:
        GPU 0: Problems [0, 3, 6, 9]    (indices 0, 3, 6, 9)
        GPU 1: Problems [1, 4, 7]       (indices 1, 4, 7)
        GPU 2: Problems [2, 5, 8]       (indices 2, 5, 8)

    Args:
        problems_df: DataFrame containing all problems
        num_gpus: Number of GPUs to use

    Returns:
        list: List of DataFrames, one per GPU
    """
    total_problems = len(problems_df)

    # Create chunks using interleaved distribution
    chunks = [[] for _ in range(num_gpus)]

    for idx in range(total_problems):
        gpu_id = idx % num_gpus
        chunks[gpu_id].append(idx)

    # Convert to DataFrames and print distribution
    result_chunks = []
    print("\nInterleaved problem distribution:")
    for gpu_id in range(num_gpus):
        if len(chunks[gpu_id]) > 0:
            chunk_df = problems_df.iloc[chunks[gpu_id]].copy().reset_index(drop=True)
            result_chunks.append(chunk_df)

            # Show which problem indices this GPU will process
            indices_str = str(chunks[gpu_id][:10])  # Show first 10
            if len(chunks[gpu_id]) > 10:
                indices_str = indices_str[:-1] + f", ... +{len(chunks[gpu_id])-10} more]"

            print(f"  GPU {gpu_id}: {len(chunk_df)} problems - indices {indices_str}")
        else:
            # Empty chunk (more GPUs than problems)
            result_chunks.append(pd.DataFrame())
            print(f"  GPU {gpu_id}: 0 problems (idle)")

    return result_chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune hipb_mm (hipBLASLt) solutions for FP8 GEMM operations with GPU multiprocessing support"
    )

    # Multiprocessing options
    parser.add_argument(
        "--mp",
        type=int,
        default=1,
        help="Number of GPUs to use for parallel tuning (default: 1)",
    )

    # Input source options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--model_dir",
        type=str,
        default=os.getenv("HIPB_TUNE_MODEL", ""),
        help="Model directory to extract GEMM shapes from config.json",
    )
    input_group.add_argument(
        "--input_file",
        type=str,
        default=os.getenv("HIPB_TUNE_INPUT", None),
        help="CSV file with problem configurations (comma-separated: M,N,K,in_dtype,out_dtype,bias,scale_A_type,scale_B_type,B_preshuffled)",
    )

    # Output options
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output CSV file for tuned solutions (default: aiter/configs/hipb_mm_tuned_solutions.csv)",
    )

    # Tensor parallelism
    parser.add_argument(
        "--tp",
        type=int,
        default=int(os.getenv("HIPB_TUNE_TP", "1")),
        help="Tensor parallelism factor for model shapes",
    )

    # Batch size and N sets
    parser.add_argument(
        "--batch_size",
        type=int,
        default=int(os.getenv("HIPB_TUNE_BATCH_SIZE", "1")),
        help="Batch size to tune for",
    )
    parser.add_argument(
        "--nsets",
        type=list_of_ints,
        default=[16, 32, 64, 128, 256, 512, 1024, 2048, 3072, 4096, 8192],
        help="N (sequence length) sizes to tune for: 16,128,2048 (Note: rowwise scaling requires N>=16)",
    )

    # Data types
    parser.add_argument(
        "--in_dtype",
        type=str,
        default="fp8",
        choices=["fp8", "bf16", "bfloat16"],
        help="Input data type (fp8 recommended for quantized models)",
    )
    parser.add_argument(
        "--out_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "bfloat16"],
        help="Output data type (bf16 required for rowwise scaling)",
    )

    # Scaling configuration
    parser.add_argument(
        "--scale_a_type",
        type=str,
        default="rowwise",
        choices=["None", "scalar", "rowwise"],
        help="Scale type for input matrix A (rowwise recommended for FP8)",
    )
    parser.add_argument(
        "--scale_b_type",
        type=str,
        default="rowwise",
        choices=["None", "scalar", "rowwise"],
        help="Scale type for weight matrix B (rowwise recommended for FP8)",
    )

    # Bias options
    parser.add_argument(
        "--bias",
        action="store_true",
        default=False,
        help="Include bias in tuning",
    )
    parser.add_argument(
        "--all_bias",
        action="store_true",
        help="Tune for both bias and non-bias cases",
    )

    # Preshuffled weight layout
    parser.add_argument(
        "--bpreshuffle",
        action="store_true",
        default=False,
        help="Tune for preshuffled (swizzled) weight layout",
    )

    # Tuning parameters
    parser.add_argument(
        "--err_ratio",
        type=float,
        default=0.05,
        help="Maximum acceptable error ratio (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=20,
        help="Number of warmup iterations per solution",
    )
    parser.add_argument(
        "--test_iters",
        type=int,
        default=20,
        help="Number of timed iterations per solution",
    )
    parser.add_argument(
        "--fast_mode",
        action="store_true",
        help="Use fast mode (fewer iterations for quick testing)",
    )

    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = DEFAULT_HIPB_MM_OUTPUT

    print("=" * 80)
    print("hipb_mm Tuner - GPU Multiprocessing Mode")
    print("=" * 80)
    print(f"Number of GPUs: {args.mp}")
    print(f"Output file: {args.output_file}")
    print(f"Input dtype: {args.in_dtype}, Output dtype: {args.out_dtype}")
    print(f"Scale A: {args.scale_a_type}, Scale B: {args.scale_b_type}")
    print(f"Preshuffled: {args.bpreshuffle}")
    print(f"Error ratio threshold: {args.err_ratio}")
    print("=" * 80)

    # Check available GPUs
    num_gpus_available = torch.cuda.device_count()
    if args.mp > num_gpus_available:
        print(f"ERROR: Requested {args.mp} GPUs but only {num_gpus_available} available")
        sys.exit(1)

    print(f"Available GPUs: {num_gpus_available}")
    for i in range(num_gpus_available):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Convert dtype strings
    in_dtype = get_dtype(args.in_dtype)
    out_dtype = get_dtype(args.out_dtype)

    # Create a temporary tuner just to generate/load the problem list
    temp_tuner = HipbMmTuner(
        output_csv=None,  # Don't write anything yet
        err_ratio=args.err_ratio,
        warmup_iters=args.warmup_iters,
        test_iters=args.test_iters,
        fast_mode=args.fast_mode,
    )

    # Load or generate problems (same logic as original script)
    if args.input_file:
        # Load from CSV file
        print(f"\n>>> Loading problems from {args.input_file}")
        if not Path(args.input_file).is_file():
            print(f">>> ERROR: {args.input_file} does not exist. Exiting.")
            exit(1)

        shapes = pd.read_csv(args.input_file).fillna("")

        # Add each problem from the CSV
        for i in range(len(shapes)):
            row = shapes.iloc[i]

            # Handle different column naming conventions
            m = int(row.get("M", row.get("m", 0)))
            n = int(row.get("N", row.get("n", 0)))
            k = int(row.get("K", row.get("k", 0)))

            in_dtype_str = row.get("in_dtype", row.get("indtype", args.in_dtype))
            out_dtype_str = row.get("out_dtype", row.get("outdtype", args.out_dtype))

            bias = row.get("bias", args.bias)
            if isinstance(bias, str):
                bias = bias.lower() in ['true', 'yes', '1']

            scale_a = row.get("scale_A_type", row.get("scale_a_type", args.scale_a_type))
            scale_b = row.get("scale_B_type", row.get("scale_b_type", args.scale_b_type))

            # Normalize scale types: "None" string or empty string -> "None"
            if isinstance(scale_a, str):
                scale_a = scale_a.strip()
                if scale_a == "" or scale_a.lower() == "none" or scale_a.lower() == "no_scale":
                    scale_a = "no_scale"
            if isinstance(scale_b, str):
                scale_b = scale_b.strip()
                if scale_b == "" or scale_b.lower() == "none" or scale_b.lower() == "no_scale":
                    scale_b = "no_scale"

            b_preshuffled = row.get("B_preshuffled", row.get("bpreshuffle", args.bpreshuffle))
            if isinstance(b_preshuffled, str):
                b_preshuffled = b_preshuffled.upper() in ['Y', 'YES', 'TRUE', '1']

            # Add problem (handle all_bias)
            for bias_val in [True, False] if args.all_bias else [bias]:
                temp_tuner.add_problem(
                    m=m, n=n, k=k,
                    in_dtype=get_dtype(in_dtype_str),
                    out_dtype=get_dtype(out_dtype_str),
                    bias=bias_val,
                    scale_a_type=scale_a,
                    scale_b_type=scale_b,
                    b_preshuffled=b_preshuffled,
                )

        print(f">>> Loaded {len(shapes)} problem configurations")

    else:
        # Generate from model or use default shapes
        if not args.model_dir:
            print(">>> Warning! NO MODEL SPECIFIED. Tuning for LLaMA2 13B TP1 shapes")
            # LLaMA2 13B default sizes
            mksets = [(15360, 5120), (5120, 5120), (27648, 5120), (5120, 13824)]
            hidden_size = 5120

            # Add logits GEMM (vocab projection) - skip if rowwise scaling and N < 16
            logits_n = 1 * args.batch_size
            is_rowwise = (args.scale_a_type == "rowwise" or args.scale_b_type == "rowwise")
            if not is_rowwise or logits_n >= 16:
                temp_tuner.add_problem(
                    m=32000,
                    n=logits_n,
                    k=hidden_size,
                    in_dtype=in_dtype,
                    out_dtype=out_dtype,
                    bias=args.bias,
                    scale_a_type=args.scale_a_type,
                    scale_b_type=args.scale_b_type,
                    b_preshuffled=args.bpreshuffle,
                )
            else:
                print(f">>> Skipping logits GEMM (N={logits_n}) - rowwise scaling requires N >= 16")
        else:
            print(f">>> Extracting shapes from model: {args.model_dir}")
            mksets, hidden_size, model_dtype = generate_mk_sets(args.model_dir, args.tp)

            # Add logits GEMM (skip if rowwise scaling and N < 16)
            logits_n = 1 * args.batch_size
            is_rowwise = (args.scale_a_type == "rowwise" or args.scale_b_type == "rowwise")
            if not is_rowwise or logits_n >= 16:
                temp_tuner.add_problem(
                    m=32000 // args.tp,  # TODO: Handle vocab_size from config
                    n=logits_n,
                    k=hidden_size,
                    in_dtype=in_dtype,
                    out_dtype=out_dtype,
                    bias=args.bias,
                    scale_a_type=args.scale_a_type,
                    scale_b_type=args.scale_b_type,
                    b_preshuffled=args.bpreshuffle,
                )
            else:
                print(f">>> Skipping logits GEMM (N={logits_n}) - rowwise scaling requires N >= 16")

        # Add all combinations of (M, K) from model and N from nsets
        nsets = [i * args.batch_size for i in args.nsets]

        print(f">>> Generating problems for N={nsets}")
        print(f">>> Model shapes (M, K): {mksets}")

        for n in sorted(nsets):
            for m, k in mksets:
                for bias_val in [True, False] if args.all_bias else [args.bias]:
                    temp_tuner.add_problem(
                        m=m, n=n, k=k,
                        in_dtype=in_dtype,
                        out_dtype=out_dtype,
                        bias=bias_val,
                        scale_a_type=args.scale_a_type,
                        scale_b_type=args.scale_b_type,
                        b_preshuffled=args.bpreshuffle,
                    )

        total_problems = len(temp_tuner.problems_df) if temp_tuner.problems_df is not None else 0
        print(f">>> Generated {total_problems} problem configurations")

    # Get the complete problem list
    all_problems = temp_tuner.problems_df
    if all_problems is None or len(all_problems) == 0:
        print("ERROR: No problems to tune!")
        sys.exit(1)

    print(f"\n>>> Total problems in input: {len(all_problems)}")

    # Filter out already-tuned problems before distribution
    # Get GPU info to check for existing results
    gpu = torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(gpu)
    cu_num = device_properties.multi_processor_count

    print(f">>> Current GPU CU count: {cu_num}")
    print(f">>> Checking for already-tuned problems...")

    problems_to_tune, skipped_count = filter_already_tuned_problems(
        all_problems, args.output_file, cu_num
    )

    if len(problems_to_tune) == 0:
        print("\n" + "=" * 80)
        print("All problems already tuned! Nothing to do.")
        print("=" * 80)
        sys.exit(0)

    print(f"\n>>> Problems to tune: {len(problems_to_tune)}")
    print(f">>> Already tuned (skipped): {skipped_count}")

    # Split problems across GPUs
    if args.mp == 1:
        # Single GPU mode - no multiprocessing
        print("\n>>> Running in single-GPU mode (no multiprocessing)")
        print("=" * 80)

        # Initialize hipBLASLt for single GPU
        aiter.hipb_create_extension()

        tuner = HipbMmTuner(
            output_csv=args.output_file,
            err_ratio=args.err_ratio,
            warmup_iters=args.warmup_iters,
            test_iters=args.test_iters,
            fast_mode=args.fast_mode,
        )
        tuner.problems_df = problems_to_tune

        start_time = time.time()
        tuner.tune_all()
        elapsed = time.time() - start_time

        print("\n" + "=" * 80)
        print(f"Tuning completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        print("=" * 80)
    else:
        # Multi-GPU mode
        print(f"\n>>> Distributing {len(problems_to_tune)} problems across {args.mp} GPUs (interleaved)")
        print("=" * 80)

        problem_chunks = split_problems_across_gpus(problems_to_tune, args.mp)

        print("\n>>> Starting multiprocessing tuning...")
        print("=" * 80)

        # Create process pool
        # Use 'spawn' method to ensure clean CUDA initialization per process
        mp_ctx = mp.get_context('spawn')

        start_time = time.time()

        # Create worker function with fixed parameters
        worker_func = partial(
            worker_process,
            output_file=args.output_file,
            err_ratio=args.err_ratio,
            warmup_iters=args.warmup_iters,
            test_iters=args.test_iters,
            fast_mode=args.fast_mode
        )

        # Launch workers
        with mp_ctx.Pool(processes=args.mp) as pool:
            # Map GPU IDs and problem chunks to workers
            results = pool.starmap(
                worker_func,
                [(gpu_id, chunk) for gpu_id, chunk in enumerate(problem_chunks)]
            )

        elapsed = time.time() - start_time

        # Print summary
        print("\n" + "=" * 80)
        print("Multi-GPU Tuning Summary")
        print("=" * 80)

        total_tuned = 0
        for gpu_id, num_problems, worker_time in results:
            if worker_time > 0:
                print(f"GPU {gpu_id}: Tuned {num_problems} problems in {worker_time:.2f}s ({worker_time/60:.2f}min)")
                total_tuned += num_problems
            else:
                print(f"GPU {gpu_id}: FAILED")

        print(f"\nTotal: {total_tuned} problems tuned")
        print(f"Wall time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        print(f"Speedup: {total_tuned / elapsed:.2f} problems/second")
        print("=" * 80)
