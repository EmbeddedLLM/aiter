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

import os
import pandas as pd
from pathlib import Path
import torch
import torch.nn.functional as F
from functools import lru_cache

import aiter
from aiter import dtypes, hipb_create_extension
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.mp_tuner import mp_tuner
from aiter.jit.utils.chip_info import get_cu_num
from aiter.test_common import checkAllclose

# Initialize hipBLASLt extension
hipb_create_extension()


class HipbMmProblem:
    """Represents a single hipb_mm problem to tune"""

    def __init__(
        self,
        m, n, k,
        in_dtype,
        out_dtype,
        bias=False,
        scale_a_type="no_scale",  # "no_scale", "scalar", "rowwise"
        scale_b_type="no_scale",  # "no_scale", "scalar", "rowwise"
        b_preshuffled=False,
        mp=1,
        err_ratio=0.05,
        warmup_iters=20,
        test_iters=20,
        rank_id=None,
    ):
        self.m = m
        self.n = n
        self.k = k
        self.in_dtype = self._parse_dtype(in_dtype)
        self.out_dtype = self._parse_dtype(out_dtype)
        self.bias = bias
        self.scale_a_type = scale_a_type
        self.scale_b_type = scale_b_type
        self.b_preshuffled = b_preshuffled
        self.mp = mp
        self.err_ratio = err_ratio
        self.warmup_iters = warmup_iters
        self.test_iters = test_iters
        self.rank_id = rank_id

        # Prefix for logging
        self.log_prefix = f"[Rank {rank_id}] " if rank_id is not None else ""

        # Results
        self.best_solidx = None
        self.best_time_ms = None
        self.best_err_ratio = None
        self.solutions = []

        # CUDA events for timing
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def _parse_dtype(self, dtype_str):
        """Parse dtype string to torch dtype"""
        if isinstance(dtype_str, torch.dtype):
            return dtype_str
        dtype_map = {
            "fp8": dtypes.fp8,
            "bf16": dtypes.bf16,
            "bfloat16": dtypes.bf16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "torch.float32": torch.float32,
            "torch.float16": torch.float16,
            "torch.bfloat16": dtypes.bf16,
            "torch.float8_e4m3fnuz": dtypes.fp8,
        }
        return dtype_map.get(str(dtype_str).lower(), dtype_str)

    def generate_test_data(self, device="cuda"):
        """Generate test input tensors and scales"""
        torch.manual_seed(42)

        # Generate input and weight
        x = torch.randn((self.m, self.k), dtype=self.out_dtype, device=device)
        weight = torch.randn((self.n, self.k), dtype=self.out_dtype, device=device)

        # Quantize if FP8
        if self.in_dtype == dtypes.fp8:
            x_q, x_scale_rowwise = aiter.pertoken_quant(x, quant_dtype=self.in_dtype)
            weight_q, w_scale_rowwise = aiter.pertoken_quant(weight, quant_dtype=self.in_dtype)
        else:
            x_q = x.to(self.in_dtype)
            weight_q = weight.to(self.in_dtype)
            x_scale_rowwise = None
            w_scale_rowwise = None

        # Generate scales based on scale type
        scale_a = self._generate_scale(
            self.scale_a_type, self.m, x_scale_rowwise, device
        )
        scale_b = self._generate_scale(
            self.scale_b_type, self.n, w_scale_rowwise, device
        )

        # Shuffle weight if requested
        if self.b_preshuffled:
            weight_shuffled = shuffle_weight(weight_q, layout=(16, 16))
        else:
            weight_shuffled = weight_q

        # Generate bias
        bias_tensor = None
        if self.bias:
            bias_tensor = torch.randn(self.n, dtype=self.out_dtype, device=device)

        return x_q, weight_shuffled, scale_a, scale_b, bias_tensor, x_scale_rowwise, w_scale_rowwise, weight_q

    def _generate_scale(self, scale_type, dim, rowwise_scale, device):
        """Generate scale tensor based on type"""
        if scale_type is None:
            return None
        elif scale_type == "no_scale":
            return None
        elif scale_type == "scalar":
            if rowwise_scale is not None:
                # Use average of rowwise scales
                return rowwise_scale.mean().view(1, 1)
            else:
                return torch.tensor([[0.5]], dtype=torch.float32, device=device)
        elif scale_type == "rowwise":
            if rowwise_scale is not None:
                return rowwise_scale
            else:
                return torch.randn((dim, 1), dtype=torch.float32, device=device).abs()
        else:
            raise ValueError(f"Unknown scale type: {scale_type}")

    def compute_reference(self, x_q, weight_q, scale_a, scale_b, bias, x_scale_full, w_scale_full):
        """Compute reference output using PyTorch"""
        # Dequantize using appropriate scales
        if scale_a is not None:
            x_fp = x_q.to(torch.float32) * scale_a
        elif x_scale_full is not None:
            x_fp = x_q.to(torch.float32) * x_scale_full
        else:
            x_fp = x_q.to(torch.float32)

        if scale_b is not None:
            # For scale_b, need to handle shape properly
            if scale_b.numel() == 1:
                weight_fp = weight_q.to(torch.float32) * scale_b
            else:
                # Rowwise scale for weight: (n, 1)
                weight_fp = weight_q.to(torch.float32) * scale_b
        elif w_scale_full is not None:
            weight_fp = weight_q.to(torch.float32) * w_scale_full
        else:
            weight_fp = weight_q.to(torch.float32)

        # Compute reference
        ref = F.linear(x_fp, weight_fp, bias)
        return ref.to(self.out_dtype)

    def find_all_solutions(self):
        """Find all available hipBLASLt solutions for this problem"""
        x_q, weight_shuffled, scale_a, scale_b, bias, _, _, _ = self.generate_test_data()

        try:
            # Prepare scale_b for hipb_findallsols (needs transpose for rowwise)
            scale_b_for_find = scale_b
            if scale_b is not None and scale_b.numel() > 1:
                scale_b_for_find = scale_b.t()  # (1, n)

            sols = aiter.hipb_findallsols(
                x_q,
                weight_shuffled.t(),
                bias=bias,
                out_dtype=self.out_dtype,
                scaleA=scale_a,
                scaleB=scale_b_for_find,
                bpreshuffle=self.b_preshuffled,
            )

            # Synchronize and clean up after finding solutions
            torch.cuda.synchronize()

            # Workaround for hipBLASLt 1.1.0 bug: getAllAlgos() leaves GPU in corrupted state
            # Perform a dummy operation to catch any async HIP errors from findallsols
            try:
                _ = torch.zeros(1, device='cuda')
                torch.cuda.synchronize()
            except Exception as async_error:
                # hipBLASLt getAllAlgos() failed - this is a known issue
                # Try to recover by recreating the extension
                print(f"{self.log_prefix}Warning: hipBLASLt getAllAlgos() corrupted GPU state: {async_error}")
                print(f"{self.log_prefix}Attempting to recover by reinitializing hipBLASLt...")

                try:
                    # Destroy and recreate the extension
                    aiter.hipb_destroy_extension()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    aiter.hipb_create_extension()
                    torch.cuda.synchronize()

                    # Verify recovery with a simple operation
                    _ = torch.zeros(1, device='cuda')
                    torch.cuda.synchronize()
                    print(f"{self.log_prefix}Successfully recovered from hipBLASLt error")
                except Exception as recovery_error:
                    print(f"{self.log_prefix}Failed to recover: {recovery_error}")
                    import traceback
                    traceback.print_exc()
                    return []

            torch.cuda.empty_cache()

            print(
                f"{self.log_prefix}Problem (M={self.m}, N={self.n}, K={self.k}, "
                f"in_dtype={self.in_dtype}, out_dtype={self.out_dtype}, "
                f"bias={self.bias}, scale_a={self.scale_a_type}, "
                f"scale_b={self.scale_b_type}, shuffled={self.b_preshuffled}): "
                f"Found {len(sols)} solutions",
                flush=True,
            )
            self.solutions = sols
            return sols
        except Exception as e:
            print(f"{self.log_prefix}Error finding solutions: {e}")
            import traceback
            traceback.print_exc()
            return []

    def test_solution(self, solution_idx, num_warmup=None, num_iters=None, err_ratio=0.05):
        """Test a specific solution and return timing and error rate"""
        if num_warmup is None:
            num_warmup = self.warmup_iters
        if num_iters is None:
            num_iters = self.test_iters

        x_q, weight_shuffled, scale_a, scale_b, bias, x_scale_full, w_scale_full, weight_q = \
            self.generate_test_data()

        # Prepare scale_b for hipb_mm (needs transpose for rowwise)
        scale_b_for_mm = scale_b
        if scale_b is not None and scale_b.numel() > 1:
            scale_b_for_mm = scale_b.t()  # (1, n)
            
        weight_shuffled_t = weight_shuffled.t()

        try:
            # Validate result to do early stopping
            val_result = aiter.hipb_mm(
                x_q,
                weight_shuffled_t,
                solution_index=solution_idx,
                bias=bias,
                out_dtype=self.out_dtype,
                scaleA=scale_a,
                scaleB=scale_b_for_mm,
                bpreshuffle=self.b_preshuffled,
            )
            ref = self.compute_reference(
                x_q, weight_q, scale_a, scale_b, bias, x_scale_full, w_scale_full
            )
            val_err_ratio = checkAllclose(
                ref,
                val_result,
                atol=1e-2,
                rtol=1e-2,
                tol_err_ratio=err_ratio,
                printLog=False,
                msg=f"Solution {solution_idx}: ",
            )
            if val_err_ratio > err_ratio:
                return float('inf'), 1.0
            
            # Warmup
            for _ in range(num_warmup):
                _ = aiter.hipb_mm(
                    x_q,
                    weight_shuffled_t,
                    solution_index=solution_idx,
                    bias=bias,
                    out_dtype=self.out_dtype,
                    scaleA=scale_a,
                    scaleB=scale_b_for_mm,
                    bpreshuffle=self.b_preshuffled,
                )

            torch.cuda.synchronize()

            # Timed runs
            self.start.record()
            for _ in range(num_iters):
                result = aiter.hipb_mm(
                    x_q,
                    weight_shuffled_t,
                    solution_index=solution_idx,
                    bias=bias,
                    out_dtype=self.out_dtype,
                    scaleA=scale_a,
                    scaleB=scale_b_for_mm,
                    bpreshuffle=self.b_preshuffled,
                )
            self.end.record()
            torch.cuda.synchronize()

            elapsed_ms = self.start.elapsed_time(self.end) / num_iters

            return elapsed_ms, val_err_ratio

        except Exception as e:
            print(f"Error testing solution {solution_idx}: {e}")
            return float('inf'), 1.0

    def find_best_solution(self, fast_mode=False):
        """Find the best solution through tuning"""
        # Find all solutions
        solutions = self.find_all_solutions()

        if not solutions:
            print(f"{self.log_prefix}No solutions found!")
            self.best_solidx = -1
            self.best_time_ms = float('inf')
            self.best_err_ratio = 1.0
            return

        # Test settings
        if fast_mode:
            warmup = 5
            iters = 10
        else:
            warmup = self.warmup_iters
            iters = self.test_iters

        # Test each solution
        results = []
        for sol_idx in solutions:
            time_ms, err_ratio = self.test_solution(sol_idx, warmup, iters, self.err_ratio)

            # Only consider solutions with acceptable error
            if err_ratio <= self.err_ratio:
                results.append((sol_idx, time_ms, err_ratio))
            #     print(f"  Solution {sol_idx}: {time_ms:.4f} ms, err_ratio={err_ratio:.6f}")
            # else:
            #     print(f"  Solution {sol_idx}: FAILED (err_ratio={err_ratio:.6f} > {self.err_ratio})")
            
        default_time_ms, default_err_ratio = self.test_solution(-1, warmup, iters, self.err_ratio)

        if not results:
            print("No valid solutions found (all failed accuracy check)!")
            self.best_solidx = -1
            self.best_time_ms = float('inf')
            self.best_err_ratio = 1.0
            return

        # Find best solution
        results.sort(key=lambda x: x[1])  # Sort by time
        self.best_solidx, self.best_time_ms, self.best_err_ratio = results[0]
        if default_time_ms < self.best_time_ms:
            self.best_solidx = -1
            self.best_time_ms = default_time_ms
            self.best_err_ratio = default_err_ratio

        print(
            f"{self.log_prefix}Best solution: idx={self.best_solidx}, "
            f"time={self.best_time_ms:.4f} ms, "
            f"err_ratio={self.best_err_ratio:.6f}, "
            f"default_time_ms={default_time_ms:.4f} ms, "
            f"default_err_ratio={default_err_ratio:.6f}, "
            f"speedup={default_time_ms / self.best_time_ms:.2f}x"
        )

        return self.best_solidx, self.best_time_ms, self.best_err_ratio


class HipbMmTuner:
    """Tuner for hipb_mm across multiple problem configurations"""

    def __init__(
        self,
        input_csv=None,
        output_csv=None,
        mp=1,
        err_ratio=0.05,
        warmup_iters=20,
        test_iters=20,
        fast_mode=False,
        rank_id=None,
    ):
        """
        Initialize HipbMmTuner

        Args:
            input_csv: Path to input CSV with problem definitions
            output_csv: Path to output CSV for tuning results
            mp: Number of parallel processes (currently 1 recommended)
            err_ratio: Maximum acceptable error ratio (default 0.05 = 5%)
            warmup_iters: Number of warmup iterations
            test_iters: Number of timed iterations
            fast_mode: If True, use fewer iterations for faster tuning
            rank_id: GPU rank ID for multiprocessing logging (None for single process)
        """
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.mp = mp
        self.err_ratio = err_ratio
        self.warmup_iters = warmup_iters if not fast_mode else 5
        self.test_iters = test_iters if not fast_mode else 10
        self.fast_mode = fast_mode
        self.rank_id = rank_id

        # Prefix for logging
        self.log_prefix = f"[Rank {rank_id}] " if rank_id is not None else ""

        # Get GPU info
        gpu = torch.cuda.current_device()
        device_properties = torch.cuda.get_device_properties(gpu)
        self.cu_num = device_properties.multi_processor_count

        # Load problems if input CSV exists
        if input_csv and Path(input_csv).exists():
            self.problems_df = pd.read_csv(input_csv)
            print(f"{self.log_prefix}Loaded {len(self.problems_df)} problems from {input_csv}")
        else:
            self.problems_df = None
            print(f"{self.log_prefix}No input CSV provided or file not found")

        # Initialize output CSV
        if output_csv:
            output_path = Path(output_csv)
            if output_path.exists():
                try:
                    self.results_df = pd.read_csv(output_csv)
                    # Count results for current GPU
                    current_gpu_results = len(self.results_df[self.results_df["cu_num"] == self.cu_num])
                    total_results = len(self.results_df)
                    unique_gpus = self.results_df["cu_num"].nunique()

                    print(f"Loaded existing results from {output_csv}")
                    print(f"  Total entries: {total_results}")
                    print(f"  Unique GPUs: {unique_gpus}")
                    print(f"  Entries for current GPU (CU={self.cu_num}): {current_gpu_results}")

                    if current_gpu_results > 0:
                        print(f"  → Will skip {current_gpu_results} already-tuned problems for this GPU")
                except Exception as e:
                    print(f"Warning: Could not load existing results: {e}")
                    self.results_df = None
            else:
                # Create new results file with header
                self.results_df = pd.DataFrame(columns=[
                    "M", "N", "K",
                    "in_dtype", "out_dtype",
                    "bias", "scale_A_type", "scale_B_type", "B_preshuffled",
                    "cu_num",
                    "best_solution_idx",
                    "best_time_ms",
                    "err_ratio",
                    "tflops",
                    "bw_gb_s",
                ])
                self.results_df.to_csv(output_csv, index=False)
                print(f"Created new output file: {output_csv}")

    def add_problem(
        self, m, n, k,
        in_dtype, out_dtype,
        bias=False,
        scale_a_type="no_scale",
        scale_b_type="no_scale",
        b_preshuffled=False,
    ):
        """Add a single problem to tune"""
        problem_dict = {
            "M": m,
            "N": n,
            "K": k,
            "in_dtype": in_dtype,
            "out_dtype": out_dtype,
            "bias": bias,
            "scale_A_type": scale_a_type,
            "scale_B_type": scale_b_type,
            "B_preshuffled": b_preshuffled,
        }

        if self.problems_df is None:
            self.problems_df = pd.DataFrame([problem_dict])
        else:
            self.problems_df = pd.concat(
                [self.problems_df, pd.DataFrame([problem_dict])],
                ignore_index=True
            )

    def calculate_performance(self, m, n, k, time_ms, in_dtype, out_dtype):
        """Calculate TFLOPS and bandwidth"""
        if time_ms <= 0 or time_ms == float('inf'):
            return 0.0, 0.0

        # FLOPS calculation
        flops = 2 * m * n * k  # Multiply-add = 2 ops
        tflops = flops / (time_ms * 1e9)  # Convert ms to seconds and to TFLOPS

        # Bandwidth calculation
        dtype_map = {
            dtypes.fp8: 1,
            dtypes.bf16: 2,
            dtypes.fp16: 2,
            dtypes.fp32: 4,
        }
        in_bytes = dtype_map.get(in_dtype, 2)
        out_bytes = dtype_map.get(out_dtype, 2)

        # Total data movement: m*k (input) + n*k (weight) + m*n (output)
        total_bytes = (m * k * in_bytes) + (n * k * in_bytes) + (m * n * out_bytes)
        bw_gb_s = (total_bytes / (time_ms * 1e-3)) / 1e9  # GB/s

        return round(tflops, 3), round(bw_gb_s, 2)

    def is_already_tuned(self, problem_row):
        """
        Check if this problem configuration was already tuned for the current GPU.

        A problem is considered already tuned if there exists an entry with:
        - Same M, N, K dimensions
        - Same dtypes (in_dtype, out_dtype)
        - Same configuration (bias, scale types, shuffle mode)
        - Same CU count (GPU-specific)

        Returns:
            bool: True if already tuned, False otherwise
        """
        if self.results_df is None or len(self.results_df) == 0:
            return False

        # Normalize boolean values for comparison
        bias_val = bool(problem_row["bias"])
        preshuffled_val = bool(
            problem_row["B_preshuffled"] == 'Y' or
            problem_row["B_preshuffled"] == True or
            problem_row["B_preshuffled"] == 'True'
        )

        def normalize_scale(val):
            if pd.isna(val) or str(val).lower() in ['none', 'nan', 'no_scale']:
                return 'no_scale'
            return str(val)

        problem_scale_a = normalize_scale(problem_row["scale_A_type"])
        problem_scale_b = normalize_scale(problem_row["scale_B_type"])

        # Check if exact match exists with same CU count
        mask = (
            (self.results_df["M"] == int(problem_row["M"])) &
            (self.results_df["N"] == int(problem_row["N"])) &
            (self.results_df["K"] == int(problem_row["K"])) &
            (self.results_df["in_dtype"] == str(problem_row["in_dtype"])) &
            (self.results_df["out_dtype"] == str(problem_row["out_dtype"])) &
            (self.results_df["bias"] == bias_val) &
            (self.results_df["scale_A_type"].apply(normalize_scale) == problem_scale_a) &
            (self.results_df["scale_B_type"].apply(normalize_scale) == problem_scale_b) &
            (self.results_df["B_preshuffled"] == preshuffled_val) &
            (self.results_df["cu_num"] == self.cu_num)
        )

        if mask.any():
            # Found existing entry - print info
            existing = self.results_df[mask].iloc[0]
            print(f"  Found existing result (CU={self.cu_num}, "
                  f"solution_idx={existing['best_solution_idx']}, "
                  f"time={existing['best_time_ms']:.4f}ms)")
            return True

        return False

    def tune_all(self):
        """Tune all problems in the input CSV"""
        if self.problems_df is None or len(self.problems_df) == 0:
            print("No problems to tune!")
            return

        # Reload results at start to catch any new entries
        if self.output_csv and Path(self.output_csv).exists():
            try:
                self.results_df = pd.read_csv(self.output_csv)
            except Exception as e:
                print(f"Warning: Could not reload results: {e}")

        print(f"\n{self.log_prefix}{'='*80}")
        print(f"{self.log_prefix}Starting HipbMm Tuning")
        print(f"{self.log_prefix}Total problems: {len(self.problems_df)}")
        print(f"{self.log_prefix}Current GPU CU count: {self.cu_num}")
        print(f"{self.log_prefix}Error ratio threshold: {self.err_ratio}")
        print(f"{self.log_prefix}Fast mode: {self.fast_mode}")
        print(f"{self.log_prefix}{'='*80}\n")

        # Count how many will be skipped
        skipped_count = 0
        tuned_count = 0

        for idx, row in self.problems_df.iterrows():
            print(f"\n{self.log_prefix}{'='*80}")
            print(f"{self.log_prefix}Problem {idx+1}/{len(self.problems_df)}: "
                  f"M={row['M']}, N={row['N']}, K={row['K']}, "
                  f"dtype={row['in_dtype']}->{row['out_dtype']}, "
                  f"bias={row['bias']}, "
                  f"scales={row['scale_A_type']}/{row['scale_B_type']}, "
                  f"shuffled={row['B_preshuffled']}")
            print(f"{self.log_prefix}{'='*80}")

            # Check if already tuned
            if self.is_already_tuned(row):
                print(f"→ Skipping - already tuned for this GPU (CU={self.cu_num})")
                skipped_count += 1
                continue

            # Create problem
            problem = HipbMmProblem(
                m=int(row["M"]),
                n=int(row["N"]),
                k=int(row["K"]),
                in_dtype=row["in_dtype"],
                out_dtype=row["out_dtype"],
                bias=bool(row["bias"]),
                scale_a_type=row["scale_A_type"],
                scale_b_type=row["scale_B_type"],
                b_preshuffled=bool(row["B_preshuffled"] == 'Y' or row["B_preshuffled"] == True),
                mp=self.mp,
                err_ratio=self.err_ratio,
                warmup_iters=self.warmup_iters,
                test_iters=self.test_iters,
                rank_id=self.rank_id,
            )

            # Find best solution
            try:
                problem.find_best_solution(fast_mode=self.fast_mode)
                tuned_count += 1
            except Exception as e:
                print(f"{self.log_prefix}Error tuning problem: {e}")
                continue

            # Calculate performance metrics
            tflops, bw = self.calculate_performance(
                problem.m, problem.n, problem.k,
                problem.best_time_ms,
                problem.in_dtype,
                problem.out_dtype
            )

            # Save result
            result_row = {
                "M": problem.m,
                "N": problem.n,
                "K": problem.k,
                "in_dtype": str(problem.in_dtype),
                "out_dtype": str(problem.out_dtype),
                "bias": problem.bias,
                "scale_A_type": problem.scale_a_type,
                "scale_B_type": problem.scale_b_type,
                "B_preshuffled": problem.b_preshuffled,
                "cu_num": self.cu_num,
                "best_solution_idx": problem.best_solidx,
                "best_time_ms": round(problem.best_time_ms, 4) if problem.best_time_ms != float('inf') else -1,
                "err_ratio": round(problem.best_err_ratio, 6),
                "tflops": tflops,
                "bw_gb_s": bw,
            }

            # Append to results
            if self.output_csv:
                result_df = pd.DataFrame([result_row])
                result_df.to_csv(
                    self.output_csv,
                    mode='a',
                    header=False,
                    index=False
                )

                # Reload results_df so next iteration sees this new result
                try:
                    self.results_df = pd.read_csv(self.output_csv)
                except Exception as e:
                    print(f"{self.log_prefix}Warning: Could not reload results after save: {e}")

            # Clear cache
            del problem
            torch.cuda.empty_cache()

        print(f"\n{'='*80}")
        print("Tuning Complete!")
        print(f"{'='*80}")
        print(f"Total problems in input: {len(self.problems_df)}")
        print(f"Skipped (already tuned): {skipped_count}")
        print(f"Newly tuned: {tuned_count}")
        print(f"Current GPU CU count: {self.cu_num}")
        if self.output_csv:
            print(f"Results saved to: {self.output_csv}")
        print(f"{'='*80}\n")

        # Print summary for current GPU
        if self.output_csv:
            final_df = pd.read_csv(self.output_csv)
            current_gpu_df = final_df[final_df["cu_num"] == self.cu_num]

            print(f"\nResults for Current GPU (CU={self.cu_num}):")
            print(f"{'='*80}")
            if len(current_gpu_df) > 0:
                print(current_gpu_df.to_string(index=False))
            else:
                print("No results for current GPU")
            print(f"\n{'='*80}")

            # Show all GPUs summary
            if len(final_df["cu_num"].unique()) > 1:
                print(f"\nAll GPUs Summary:")
                print(f"{'='*80}")
                for cu in sorted(final_df["cu_num"].unique()):
                    count = len(final_df[final_df["cu_num"] == cu])
                    print(f"  CU={cu}: {count} results")
                print(f"{'='*80}\n")


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tune hipb_mm for various problem sizes")
    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Path to input CSV with problem definitions"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        required=True,
        help="Path to output CSV for tuning results"
    )
    parser.add_argument(
        "--mp",
        type=int,
        default=1,
        help="Number of parallel processes (default: 1)"
    )
    parser.add_argument(
        "--err-ratio",
        type=float,
        default=0.05,
        help="Maximum acceptable error ratio (default: 0.05)"
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=20,
        help="Number of warmup iterations (default: 20)"
    )
    parser.add_argument(
        "--test-iters",
        type=int,
        default=20,
        help="Number of test iterations (default: 20)"
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Use fast mode with fewer iterations"
    )

    args = parser.parse_args()

    tuner = HipbMmTuner(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        mp=args.mp,
        err_ratio=args.err_ratio,
        warmup_iters=args.warmup_iters,
        test_iters=args.test_iters,
        fast_mode=args.fast_mode,
    )

    tuner.tune_all()
