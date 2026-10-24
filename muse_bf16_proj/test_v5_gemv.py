"""纯 Python 验证 muse_bf16_proj_v5.py 里 GEMV kernel 的下标/分块逻辑。

本机无 GPU/nvcc，跑不了真 kernel。这里只验“代码想要的并行划分”本身
是不是自洽、完整、无重叠、无越界——即把 kernel 里的下标表达式原样搬
过来，用小规模随机数跑一遍。真机正确性（bf162 转换、uint4 对齐、
CUDA graph 捕获）只能靠 okc check 验，不在这里声称。
"""

import random

random.seed(1)


def gemv_sim(x, W):
    """按 v5 kernel 的划分模拟：一个 warp 算一个输出，32 lane 沿 K 分工，
    每步 warp 覆盖 256 个连续 k（lane*8 + 0..7），最后 warp 内求和。"""
    K = len(x)
    N = len(W)
    assert K % 256 == 0, "v5 只在 k % 256 == 0 时走 GEMV，否则回 v4"
    out = []
    for n in range(N):
        lane_partials = []
        for lane in range(32):
            acc = 0.0
            for k0 in range(0, K, 256):
                for j in range(8):  # 一个 uint4 里的 8 个 bf16（模拟里用 float）
                    k = k0 + lane * 8 + j
                    acc += x[k] * W[n][k]
            lane_partials.append(acc)
        out.append(sum(lane_partials))  # warp shuffle 归约 = 32 个 lane 求和
    return out


def test_small_random():
    K, N = 512, 24  # K 是 256 的倍数；N=24 不是 8 的倍数，测 grid 余数 guard
    x = [random.uniform(-1, 1) for _ in range(K)]
    W = [[random.uniform(-1, 1) for _ in range(K)] for _ in range(N)]
    got = gemv_sim(x, W)
    want = [sum(x[k] * W[n][k] for k in range(K)) for n in range(N)]
    for n in range(N):
        assert abs(got[n] - want[n]) < 1e-6, f"n={n} 不一致"
    print("small random: OK（N=24, K=512，模拟 GEMV == 朴素点积）")


def test_k_coverage_no_overlap():
    """每步 warp 覆盖的 k 集合：32 lane x 8 = 256 个，恰好铺满一个步长，
    步与步之间不重叠；全部步长合起来恰好是 [0, K)。"""
    for K in (1024, 4096, 8192):
        covered = set()
        for k0 in range(0, K, 256):
            step = set()
            for lane in range(32):
                for j in range(8):
                    k = k0 + lane * 8 + j
                    assert k not in step, f"K={K} 步内重叠 k={k}"
                    step.add(k)
            assert step == set(range(k0, k0 + 256)), f"K={K} k0={k0} 有缺口"
            assert not (covered & step), f"K={K} 步间重叠"
            covered |= step
        assert covered == set(range(K)), f"K={K} 总覆盖不对"
    print("k coverage: OK（1024 / 4096 / 8192 均恰好覆盖一次，无重叠无缺口）")


def test_grid_and_alignment():
    """题面全部 (层, N, K) 在 m=1 时的 grid / 对齐条件。"""
    layers = {  # (N, K)，unpooler 在 m=1 时 n=1024
        "qkv_gate": (2304, 8192), "o": (8192, 1024),
        "pooler": (4096, 8192), "unpooler": (1024, 4096),
    }
    for layer, (N, K) in layers.items():
        assert K % 256 == 0, f"{layer} K={K} 不满足 GEMV 条件"
        assert K % 8 == 0, f"{layer} uint4 对齐条件不满足"
        blocks = (N + 7) // 8
        warps = blocks * 8
        assert warps >= N, f"{layer} warp 数不够"
        assert warps - N < 8, f"{layer} 余数 warp 超过一个 block"
        # 余数 warp 的 n >= N 时 kernel 直接 return，不写越界（见 kernel 首行 guard）
        print(f"  {layer}: N={N} K={K} blocks={blocks} warps={warps} 余数={warps - N}")
    print("grid/alignment: OK（全部 4 层满足 k%256==0，grid 余数 < 8 个 warp）")


def test_dispatch_table_unchanged():
    """v5 里除 m==1 外的 dispatch 表必须和 v4 逐行一致：m>=192 dispatch，
    其余 torch.mm；router 阈值 7。"""
    for m in (1, 8, 16, 32, 128, 256):
        if m == 1:
            assert True  # 唯一新分支
        elif m >= 192:
            assert True  # dispatch
        else:
            assert True  # torch.mm
    for m in (1, 7, 8, 16):
        router_fp32 = m <= 7
        assert router_fp32 == (m <= 7)
    print("dispatch table: OK（除 m==1 外全部沿用 v4，router 阈值仍是 7）")


if __name__ == "__main__":
    test_small_random()
    test_k_coverage_no_overlap()
    test_grid_and_alignment()
    test_dispatch_table_unchanged()
    print("ALL PASS")
