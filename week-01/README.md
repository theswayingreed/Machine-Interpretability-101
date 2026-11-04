# Goal of the Week
本周（第 1 周，10/27–11/8）的目标是**自己写出 GPT-2 small，并和 HF 对齐**，门槛检查在 **11/8**。下面是「第 1 周计划」标签页里的最新日程：

| 日期 | 时长 | 内容 | 产出 | 状态 |
|---|---|---|---|---|
| 周二 10/27 | 约 7h | 搭好环境；ARENA 读到约 1/3；弄懂 logits、残差流、W_U、点积与对偶 | 环境自检通过 | ✅ |
| 周三 10/28 | 约 7h | 读完「输入与输出」；LayerNorm、Embed、PosEmbed、apply_causal_mask、Attention | 5 个练习通过 | ✅ |
| 周四–周六 10/29–31 | 轻量 | 3Blue1Brown；ARENA 里 MLP、Unembed、采样的讲解 | 笔记 | ✅ |
| 周日 11/1 | 0 | （休息，内容顺延） | — | — |
| 周一 11/2 | 2–3h | MLP（gelu_new）、TransformerBlock | 2 个练习通过 | ❓ |
| **周二 11/3（今天）** | 7h | Unembed、DemoTransformer，载入官方权重并核对 loss；开始 greedy 采样 | 能前向的 GPT-2 small | ⏳ |
| 周三 11/4 | 7h | temperature、top-k；去掉 √d_head 的实验 | 能采样的 GPT-2 small | |
| 周四 11/5 | 7h | `test_equiv.py`，调到两条对齐标准都通过 | `week-01/gpt2/` | |
| 周五 11/6 | 7h | 复习总结（先凭记忆写）；练习 0-P2；默写 attention 和 LN | `summary.md`、草图 | |
| 周六 11/7 | 7h | **0-P1 闭卷重写（6h）** | 0-P1 交付物 | |
| 周日 11/8 | 3h | **概念题、0-P2**；200 字周报；周检查；错题补进总结 | 测验结果、周报 | |

**门槛（11/8）：** `log_softmax` 和 HF 的误差小于 1e-4（5 段文本），3 个 prompt 的 greedy 生成完全一致；概念题答对 80% 以上；0-P1、0-P2 通过。如果周四对齐没调通，可以按超时规则占用第 2 周的周一、周二。

昨天的 `MLP` 和 `TransformerBlock` 写完、测试通过了吗？如果还没做，今天就先做这两个，再做 `Unembed` 和 `DemoTransformer`。

# 2026.11.03 Day 6 总结
今天主要把 Demo Transformer 手动重新完整写了一遍，然后载入了官方权重，进行了 greedy 采样，还算了 loss。

关于 Attention：
- $Q$: ln(residual_stream) 通过 $W_Q$ 做线性变化，从 d_model 映射到 $d_{head}$ 的向量空间里。i.e.  $(s, d_{model})$ -> $(s, d_{head})$. 每一行 $i$ 是对应 token $x_i$ 在**问题空间**里的表示 $q_i$。
- $K$: 相应的，$K$ 的每一行 $i$ 是对应 token $x_i$ 在 Key 空间里的表示 $k_i$。
- $QK^T$: 对于问题 $q_i$，我们都会去算每个 key $k_j$ 能提供的答案强度，经过 causal mask 之后，为了数学上的目的，需要做 $P = Softmax(mask(QK^T/\sqrt{d_{head}}))$。
- $P_{i,j}$：代表对于词元 $x_i$, 另一个词元 $x_j$ 能重要程度。这时，对于 $x_2$, 我们会去看的的值是：$p_{2,0}, p_{2,1}, p_{2,2}$。这是最终我们计算 $x_2$ 位置的 residual stream 的权重。$z_2 = p_{2,0}*v_0+ p_{2,1}*v_1 + p_{2,2}*v_2$。然后将 $z_2$ 映射回 residual stream 的空间 $x_2 \leftarrow \sum_{h=1}^{12}{z_2^{(h)} W_O^{(h)}+b_O} + x_2$

# 2026.11.02 Day 5

MLP - Multi Layer Perceptron

$f(x^TW^{in})W^{out}$
- $x^T: (1, d_{model})$, which is $(1, 768)$
- $W^{in}: (d_{model}, d_{mlp})$, which is $(768, 3072)$
- $W^{out}: (d_{mlp}, d_{model})$, which is $(3072, 768)$

MLP 为什么重要呢？

MLP 是每一层里唯一的强非线性计算，也是大部分参数所在的地方。每层 MLP 有 2 × 768 × 3072 ≈ 4.7M 个参数，attention 只有 4 × 768 × 768 ≈ 2.4M 个，MLP 大约占了一个 block 的三分之二。研究发现，很多事实性知识主要存放在中间几层的 MLP 里（比如 ROME，Meng et al., 2022）。

MLP 在做什么？
Attention 把上下文中有用的信息搬到当前位置；MLP 只在这个位置内部，根据已有的信息进行加工。一种常用的解释是键值存储（key-value memory）（Geva et al., 2021）：
1. $W^{in}: (d_{model}, d_{mlp})$ 的每一列$k_i$都对应的是 $d_{model}$ 维度里存储的一个**知识检测器（key）**，
2. $x^TW^{in}+b^{in}$ 得到的，是当前 residual stream 的信息在 $k_i$ 维度的知识强度，对其进行激活 $a = gelu(x^TW^{in}+b^{in})$，得到知识强度的权重。
3. $W^{out}: (d_{mlp}, d_{model})$，每一行对应的是 $k_i$ 对应的知识维度，知识的值本身 $v_i$。

因而，对于输入的包含上下文信息的 $x_i$，我们先算出知识强度的权重 $a: (1, d_{mlp})$。随后对知识值本身 $W^{out}: (d_{mlp}, d_{model})$ 进行加权求和，再加回残差流。

要注意，单个神经元的 kᵢ、vᵢ 常常不对应单一概念（polysemantic），所以不能把一个神经元直接读成一条知识。

# 2026.10.27 Day 1 Summary
1. 注册云 GPU 平台，预充值并关掉自动充值（11 月上限 US$60）；
2. 建公开的 GitHub 学习仓库；
3. 建 env-arena，生成 lock 文件并提交；
4. 在云端 notebook 里成功加载 GPT-2 small；
5. 读 ARENA「Transformer from Scratch」的开头。

| 项目 | 状态 |
|---|---|
| 云 GPU（Vast） | ✅ 已经在用。确认一下 autobilling 是关着的 |
| 公开仓库 | ✅ |
| env-arena + TF32 处理 | ✅ 自检通过（误差 2.6e-7） |
| lock 文件 + setup 脚本提交 | ✅ 还要 push，按 lock 安装那条路径也还没测 |
| notebook 里 kernel 的 TF32 检查 | ✅ |
| 新开实例自动恢复环境 | ✅ |
| notebook 里加载 GPT-2 small | ✅ |
| 空闲自动关机 | ✅ |
| 开始读 ARENA | ✅ |


- 云 GPU
    - Vast.ai. Cost $0.18/hr.
    - Image: https://hub.docker.com/r/nvcr.io/nvidia/pytorch/
    - [Onstart Script](#Vast-Onstart-Script).
    - vRAM >= 24GB 
    - Disk >= 40GB
- Put $50 in RunProd, will need to use the money later in the progress.
- Created a public repo to store the environment setup to restore the same env everytime recreating an instance on vast.ai.

# References
## Vast Onstart Script

```sh
env >> /etc/environment
mkdir -p ${DATA_DIRECTORY:-/workspace}
cd /root || exit 1
[ -d Machine-Interpretability-101 ] || git clone https://github.com/theswayingreed/Machine-Interpretability-101.git
git -C Machine-Interpretability-101 pull --ff-only || true
bash Machine-Interpretability-101/setup/vast_setup.sh
```
