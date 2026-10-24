"""纯 Python 验证 muse_bf16_proj_v6.py 的分档逻辑（本机无 GPU/nvcc/torch）。

验三件事，不声称任何实测速度：
1. _split_s：不 import v6 模块（它顶层 import torch，本机导不进），而是用
   ast 从 v6 文件里把 _split_s 的真函数体抽出来 exec，保证测的就是交卷代码。
2. bmm split-K 的切段逻辑：按 _bmm_splitk 的 view/permute 语义，用小规模
   随机数模拟（每段朴素乘法 + FP32 求和），必须等于朴素 x @ wt。
3. 全档位 dispatch 表：题面全部 (m, 层, n, k) 组合走哪条路，必须和头注释一致。

真机正确性（cuBLAS bmm 的 tile、bf16 partial 舍入、CUDA graph 捕获、
GEMV 的 nvcc 编译）只能靠 okc check 验，不在这里声称。
"""

import ast
import random

random.seed(6)

# ---- 1. 从 v6 文件抽出 _split_s 的真代码 ----
_v6_src = open('/workspace/open-kernel-challenge/muse_bf16_proj/muse_bf16_proj_v6_bmm.py').read()
_tree = ast.parse(_v6_src)
_split_fn = next(n for n in _tree.body if isinstance(n, ast.FunctionDef) and n.name == '_split_s')
_ns: dict = {}
exec(compile(ast.Module(body=[_split_fn], type_ignores=[]), 'v6_split_s', 'exec'), _ns)
_split_s = _ns['_split_s']


def test_split_s_table():
    """题面全部组合的 S，必须和 v6 头注释的表一致。"""
    # (层, N, K)：unpooler 在 m<=8 时 n=1024，否则 n=8192
    layers = {
        'qkv_gate': (2304, 8192), 'o': (8192, 1024),
        'pooler': (4096, 8192), 'unpooler': (8192, 4096),
    }
    want = {
        # (m, 层) -> (S, 说明)
        (8, 'qkv_gate'): 8, (8, 'pooler'): 4, (8, 'o'): 2, (8, 'unpooler'): 16,
        (16, 'qkv_gate'): 8, (16, 'pooler'): 4, (16, 'o'): 2, (16, 'unpooler'): 2,
        (32, 'qkv_gate'): 8, (32, 'pooler'): 4, (32, 'o'): 2, (32, 'unpooler'): 2,
        (128, 'qkv_gate'): 4, (128, 'pooler'): 2, (128, 'o'): 1, (128, 'unpooler'): 1,
    }
    for (m, layer), s_want in want.items():
        n_full, k = layers[layer]
        n = 1024 if (layer == 'unpooler' and m <= 8) else n_full
        s = _split_s(m, n, k)
        assert s == s_want, f'm={m} {layer} n={n} k={k}: S={s}，头注释写 {s_want}'
        assert k % s == 0 and k // s >= 256, f'm={m} {layer}: K/S={k // s} 太小或不整除'
        print(f'  m={m:3d} {layer:9s} n={n:5d} k={k:5d} -> S={s}')
    # m=256 不进 _split_s（kernel_fn 里 dispatch 先返回），所以这里不
    # 断言它对 m=256 的值；只确认题面组合（m<=128）的表已全部验过
    print('split_s table: OK（全部题面组合与头注释一致）')


def bmm_sim(x, wt, s):
    """按 _bmm_splitk 的语义模拟：x.view(m,s,ks).permute(1,0,2)，
    wt.view(s,ks,n)，每段朴素乘法，FP32 求和（模拟里用 float）。"""
    m, k = len(x), len(x[0])
    n = len(wt[0])
    assert k % s == 0
    ks = k // s
    out = [[0.0] * n for _ in range(m)]
    for c in range(s):
        for i in range(m):
            for j in range(n):
                acc = 0.0
                for t in range(ks):
                    acc += x[i][c * ks + t] * wt[c * ks + t][j]
                out[i][j] += acc  # float().sum(dim=0) 的模拟
    return out


def test_bmm_sim():
    for (m, k, n, s) in [(8, 512, 256, 4), (16, 512, 128, 2), (32, 256, 64, 8), (1, 512, 128, 16)]:
        x = [[random.uniform(-1, 1) for _ in range(k)] for _ in range(m)]
        wt = [[random.uniform(-1, 1) for _ in range(n)] for _ in range(k)]
        got = bmm_sim(x, wt, s)
        want = [[sum(x[i][t] * wt[t][j] for t in range(k)) for j in range(n)] for i in range(m)]
        for i in range(m):
            for j in range(n):
                assert abs(got[i][j] - want[i][j]) < 1e-9, f'({m},{k},{n},S={s}) 在 ({i},{j}) 不一致'
        print(f'  bmm sim m={m} k={k} n={n} S={s}: OK（切段求和 == 朴素乘法）')
    print('bmm sim: OK')


def route(m, layer):
    """v6 kernel_fn 的分档（非 router 层）：返回走哪条路。"""
    if m == 1 and layer != 'unpooler':
        return 'GEMV(v5)'
    if m >= 192:
        return 'dispatch(v4)'
    n_full = {'qkv_gate': 2304, 'o': 8192, 'pooler': 4096, 'unpooler': 8192}[layer]
    k = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096}[layer]
    n = 1024 if (layer == 'unpooler' and m <= 8) else n_full
    s = _split_s(m, n, k)
    return f'bmm(S={s})' if s > 1 else 'torch.mm'


def test_dispatch_table():
    for m in (1, 8, 16, 32, 128, 256):
        for layer in ('qkv_gate', 'o', 'pooler', 'unpooler'):
            r = route(m, layer)
            print(f'  m={m:3d} {layer:9s} -> {r}')
    # 关键断言：m=256 全走 dispatch；m=1 除 unpooler 外全走 GEMV；
    # m=128 的 o / unpooler 不 split（S==1）；router 阈值仍是 7
    assert all(route(256, l) == 'dispatch(v4)' for l in ('qkv_gate', 'o', 'pooler', 'unpooler'))
    assert all(route(1, l) == 'GEMV(v5)' for l in ('qkv_gate', 'o', 'pooler'))
    assert route(1, 'unpooler').startswith('bmm')
    assert route(128, 'o') == 'torch.mm' and route(128, 'unpooler') == 'torch.mm'
    assert 'ROUTER_FP32_MAX_TOKENS = 7' in _v6_src
    assert 'name="bf16_gemv_v6"' in _v6_src
    print('dispatch table: OK（m=256 dispatch、m=1 GEMV、128 档 o/unpooler 不 split、router 阈值 7）')


if __name__ == '__main__':
    test_split_s_table()
    test_bmm_sim()
    test_dispatch_table()
    print('ALL PASS')
