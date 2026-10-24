# okc: problem=muse_bf16_proj
"""BF16 projections v4：把大 m 的 dispatch 请回来。

相对此前最好 check（~1.042x，4599 µs）只改一处，其余原样：
- x 先 .contiguous() 兜底：harness 给的 x 若已连续就是 no-op；
  若不连续，这一行能避免 torch.mm / dispatch 走慢路径。

相对 v3（1.022x，4727 µs）改回两处——v3 的实测结论就是这两处不能动：
1. 非 router 恢复 m >= 192 走 SGLang dispatch，m < 192 仍走 torch.mm。
   v3 全走 torch.mm 反而慢了约 128 µs，说明 dispatch 在大 m 有真本事。
2. router FP32 阈值 8 -> 7。m = 8 那档实测：FP32 约 26 µs，
   hi+lo 约 14 µs，hi+lo 明显更好，所以 m = 8 必须走 hi+lo。

其余结构和最好版一致：预转置 + contiguous 权重、unpooler 预切片、
router hi+lo 预转置。纯 Triton 版用户已实测更慢，本版不碰。
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
        if m >= 192:
            return _bf16_gemm_dispatch_impl(x, state[layer][:n], None)
        return torch.mm(x, _w_t(state, layer, n))

    if m <= ROUTER_FP32_MAX_TOKENS:
        return F.linear(x.float(), state["router"])
    acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
    acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
    return acc
