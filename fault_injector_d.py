import torch
import torch.nn as nn
import numpy as np
import numba
from numba import cuda
import cupy as cp
import random
import math


@cuda.jit(device=True)
def Bf16BitflipFromFp32_gpu(data: np.float32, pos: int) -> np.float32:
    """
    Perform bitflip on a specific bit in bfloat16 representation (upper 16 bits of fp32), then convert back to fp32.
    pos ∈ [0, 15].
    """
    if not (0 <= pos < 16):
        return data

    local_array = cuda.local.array(shape=1, dtype=numba.float32)
    local_array[0] = data
    bits_view = local_array.view(numba.int32)
    bits = bits_view[0]


    upper16 = (bits >> 16) & 0xFFFF
    upper16 ^= (1 << pos)
    new_bits = (upper16 << 16)

    bits_view[0] = np.int32(new_bits)
    result = local_array[0]

    if math.isinf(result) or math.isnan(result):
        REPLACEMENT_VALUE = np.float32(3.4e38)
        return -REPLACEMENT_VALUE if data < 0 else REPLACEMENT_VALUE
    return result


@cuda.jit(device=True)
def Bf16StuckAtFromFp32_gpu(data: np.float32, pos: int, stuck_val: int) -> np.float32:
    """
    Force a specific bit in bfloat16 representation (upper 16 bits of fp32) to 0 or 1, then convert back to fp32.
    pos ∈ [0, 15]; stuck_val ∈ {0,1}
    """
    if not (0 <= pos < 16):
        return data

    local_array = cuda.local.array(shape=1, dtype=numba.float32)
    local_array[0] = data
    bits_view = local_array.view(numba.int32)
    bits = bits_view[0]

    upper16 = (bits >> 16) & 0xFFFF
    mask16 = (1 << pos)

    if stuck_val == 1:
        upper16 = upper16 | mask16
    else:
        upper16 = upper16 & (~mask16 & 0xFFFF)

    new_bits = (upper16 << 16)
    bits_view[0] = np.int32(new_bits)
    result = local_array[0]

    if math.isinf(result) or math.isnan(result):
        REPLACEMENT_VALUE = np.float32(3.4e38)
        return -REPLACEMENT_VALUE if data < 0 else REPLACEMENT_VALUE
    return result


# ===================================================================
# 2. Global Fault Injection Kernel
# ===================================================================
@cuda.jit
def global_matmul_fault_kernel(
    X, W, Y, sa_rows, sa_cols,
    fault_pe_row, fault_pe_col, num_faults,
    fault_type_is_input, fault_type_is_weight,
    fault_injpos, fault_stuckval
):
    """
    Compute the entire Y = X @ W matrix at once with internal fault injection.
    Each thread is responsible for computing one element (i, j) of output matrix Y.
    """
    i, j = cuda.grid(2)

    if i >= Y.shape[0] or j >= Y.shape[1]:
        return

    accumulator = np.float32(0.0)
    sa_j = j % sa_cols

    for k in range(X.shape[1]):
        input_val = X[i, k]
        weight_val = W[k, j]
        sa_k = k % sa_rows

        is_faulty_pe = False
        for fault_idx in range(num_faults):
            if sa_k == fault_pe_row[fault_idx] and sa_j == fault_pe_col[fault_idx]:
                is_faulty_pe = True
                break

        # Inject weight fault
        if fault_type_is_weight and is_faulty_pe:
            if fault_stuckval == -1:
                weight_val = Bf16BitflipFromFp32_gpu(weight_val, fault_injpos)
            else:
                weight_val = Bf16StuckAtFromFp32_gpu(weight_val, fault_injpos, fault_stuckval)

        # Inject input fault
        is_faulty_input_path = False
        if fault_type_is_input:
            for fault_idx in range(num_faults):
                if sa_k == fault_pe_row[fault_idx] and sa_j >= fault_pe_col[fault_idx]:
                    is_faulty_input_path = True
                    break

        if is_faulty_input_path:
            input_val = Bf16StuckAtFromFp32_gpu(input_val, fault_injpos, fault_stuckval)

        accumulator += input_val * weight_val

    Y[i, j] = accumulator


# ===================================================================
# 3. GPU Fault Injection Class
# ===================================================================
class SA_FaultInjector_d:
    def __init__(self, sa_rows=256, sa_cols=256, fault_type='input_bitflip'):
        self.sa_rows = sa_rows
        self.sa_cols = sa_cols
        self.fault_type_str = fault_type
        self.parse_fault_type()

        self.fault_pe_row = []
        self.fault_pe_col = []
        self.enabled = True

    def print_config(self):
        print(f"SA Fault Injector Configuration (GPU Accelerated):")
        print(f"  SA Size: {self.sa_rows} rows x {self.sa_cols} cols")
        print(f"  Fault Type: {self.fault_type_str}")
        print(f"  Fault Config: {self.fault_config}")
        print(f"  Fault PE Position: {list(zip(self.fault_pe_row, self.fault_pe_col))}")
        print(f"  Enabled: {self.enabled}")

    def parse_fault_type(self):
        parts = self.fault_type_str.lower().split('_')
        self.fault_config = {
            'stationary': "WS",
            'mode': parts[0],
            'type': parts[1],
            'val': None,
            'stuck': -1
        }
        if self.fault_config['type'] == 'bitflip':
            self.fault_config['val'] = int(parts[2])
        elif self.fault_config['type'] == 'stuck':
            self.fault_config['stuck'] = 0 if '0' in parts[3] else 1

    def set_fault_config_injpos(self, val):
        self.fault_config['val'] = val

    def reset_fault_pe(self):
        self.fault_pe_row = []
        self.fault_pe_col = []

    def init_fault_position(self):
        self.reset_fault_pe()
        self.fault_pe_row.append(random.randint(0, self.sa_rows - 1))
        self.fault_pe_col.append(random.randint(0, self.sa_cols - 1))
        # print(f"[INFO]Fault position for this run: PE({self.fault_pe_row[0]}, {self.fault_pe_col[0]})")

    def init_multi_fault_positions(self, num_faults: int):
        """Randomly initialize multiple unique fault PE positions (within entire SA array, without replacement sampling).

        Use sampling without replacement to ensure positions are unique, and throw error if num_faults exceeds available PE count.
        """
        self.reset_fault_pe()
        max_positions = self.sa_rows * self.sa_cols
        if num_faults <= 0:
            return
        if num_faults > max_positions:
            raise ValueError(f"num_faults ({num_faults}) exceeds total PE count ({max_positions})")

        # Sample without replacement from 0..max_positions-1, then map to (row, col)
        flat_indices = random.sample(range(max_positions), num_faults)
        for idx in flat_indices:
            row = idx // self.sa_cols
            col = idx % self.sa_cols
            self.fault_pe_row.append(row)
            self.fault_pe_col.append(col)

    def set_fault_position(self, row: int, col: int):
        self.reset_fault_pe()
        if not (0 <= row < self.sa_rows and 0 <= col < self.sa_cols):
            raise ValueError("Specified fault PE position exceeds SA physical range")
        self.fault_pe_row.append(row)
        self.fault_pe_col.append(col)
        # print(f"[INFO]Fault position manually set to: PE({self.fault_pe_row[0]}, {self.fault_pe_col[0]})")

    def set_multi_fault_positions(self, positions: list):
        self.reset_fault_pe()
        for row, col in positions:
            if not (0 <= row < self.sa_rows and 0 <= col < self.sa_cols):
                raise ValueError(f"Specified fault PE position ({row}, {col}) exceeds SA physical range")
            self.fault_pe_row.append(row)
            self.fault_pe_col.append(col)
        # print(f"[INFO]Multiple fault positions manually set to: {list(zip(self.fault_pe_row, self.fault_pe_col))}")

    def hook_fn(self, module, input_tuple, output_tuple):
        if not self.enabled or not self.fault_pe_row:
            return output_tuple

        X_torch = input_tuple[0].contiguous()
        W_torch = module.weight.T.contiguous()

        device = W_torch.device
        device_index = 0 if device.index is None else device.index
        with cp.cuda.Device(device_index):
            original_shape = X_torch.shape
            if X_torch.dim() > 2:
                X_torch_reshaped = X_torch.view(-1, original_shape[-1])
            else:
                X_torch_reshaped = X_torch

            X_cp = cp.asarray(X_torch_reshaped, dtype=cp.float32)
            W_cp = cp.asarray(W_torch, dtype=cp.float32)

            Y_faulty_cp = self.simulate_systolic_array_gpu(
                X=X_cp, W=W_cp,
                sa_rows=self.sa_rows, sa_cols=self.sa_cols,
                fault_pe_row=self.fault_pe_row, fault_pe_col=self.fault_pe_col,
                fault_type=self.fault_type_str,
                fault_injpos=self.fault_config['val'],
                fault_stuckval=self.fault_config['stuck']
            )
            Y_faulty_cp = cp.nan_to_num(Y_faulty_cp)
            Y_faulty_cp = cp.clip(Y_faulty_cp, -1e3, 1e3)
            Y_faulty_torch = torch.from_dlpack(Y_faulty_cp)

        if X_torch.dim() > 2:
            output_shape = original_shape[:-1] + (W_torch.shape[1],)
            Y_faulty_torch = Y_faulty_torch.view(output_shape)

        return Y_faulty_torch

    # ===================================================================
    # 4. GPU Matrix Multiplication Simulation Function
    # ===================================================================
    @staticmethod
    def simulate_systolic_array_gpu(
        X: cp.ndarray, W: cp.ndarray,
        sa_rows: int, sa_cols: int,
        fault_pe_row: list, fault_pe_col: list,
        fault_type: str, fault_injpos: int, fault_stuckval: int
    ):
        M, K = X.shape
        _, N = W.shape

        Y_final = cp.zeros((M, N), dtype=cp.float32)
        fault_pe_row_cp = cp.array(fault_pe_row, dtype=cp.int32)
        fault_pe_col_cp = cp.array(fault_pe_col, dtype=cp.int32)

        fault_type_is_input = 'input' in fault_type
        fault_type_is_weight = 'weight' in fault_type
        num_faults = len(fault_pe_row)

        threads_per_block = (16, 16)
        blocks_per_grid_x = math.ceil(M / threads_per_block[0])
        blocks_per_grid_y = math.ceil(N / threads_per_block[1])
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        global_matmul_fault_kernel[blocks_per_grid, threads_per_block](
            X, W, Y_final, sa_rows, sa_cols,
            fault_pe_row_cp, fault_pe_col_cp, num_faults,
            fault_type_is_input, fault_type_is_weight,
            fault_injpos, fault_stuckval
        )

        return Y_final