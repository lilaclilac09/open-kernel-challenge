# okc: problem=muse_bf16_proj
"""BF16 projections v4：把 dispatch 请回来。

相对 v3（mm-only + ROUTER_FP32=8 + contiguous）的测量结论：
- v3 = 1.022× / 4727 µs；此前最好 check = 1.050× / 4599 µs。
- 最大回退：verify1 router m=8 —— FP32 单 kernel ~26µs，hi/lo ~14µs（~1.81×）。
  题面「m<=8 走 FP32」是 baseline 规则，不是最优；实测阈值保持 7。
- 次大回退：verify32 qkv_gate m=256 —— mm-only 0.94×，dispatch 明显更好。

本版相对已验证赢面（pretranspose + contiguous wt + hybrid router m<=7）：
只加一处：x = data["x"].contiguous()。harness 已连续则 no-op。
"""

from typing import Any, TypedDict

import torch
import torch.nn.functional as F
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor  # bfloat16 [8192, 1024]
    pooler: torch.Tensor  # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor  # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor
    config: Config


# Measured: m==8 FP32 ~26µs vs hi/lo ~14µs. Keep 7.
ROUTER_FP32_MAX_TOKENS = 7


def prepare(weights: Weights) -> dict[str, Any]:
    state: dict[str, Any] = {name: w.contiguous() for name, w in weights.items()}
    for name in ("qkv_gate", "o", "pooler", "unpooler"):
        state[f"{name}_t"] = state[name].t().contiguous()
    ut = state["unpooler_t"]
    state["unpooler_t_1024"] = ut[:, :1024].contiguous()
    state["unpooler_t_8192"] = ut[:, :8192].contiguous()
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    return state


def _w_t(state: dict[str, Any], layer: str, n: int) -> torch.Tensor:
    if layer == "unpooler":
        if n == 1024:
            return state["unpooler_t_1024"]
        if n == 8192:
            return state["unpooler_t_8192"]
        return state["unpooler_t"][:, :n]
    wt = state[f"{layer}_t"]
    if n == wt.shape[1]:
        return wt
    return wt[:, :n]


def kernel_fn(data: Inputs, state: dict[str, Any]) -> torch.Tensor:
    x = data["x"].contiguous()
    cfg = data["config"]
    layer = cfg["layer"]
    n = cfg["n"]
    m = cfg["m"]

    if layer != "router":
        # torch.mm on contiguous [K,n] wins mid sizes (esp. pooler);
        # dispatch wins large-m qkv (m=256).
        if m >= 192:
            return _bf16_gemm_dispatch_impl(x, state[layer][:n], None)
        wt = _w_t(state, layer, n)
        if wt.is_contiguous() and wt.shape[1] == n:
            return torch.mm(x, wt)
        return _bf16_gemm_dispatch_impl(x, state[layer][:n], None)

    if m <= ROUTER_FP32_MAX_TOKENS:
        return F.linear(x.float(), state["router"])
    acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
    acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
    return acc
