# Goal of the Week

**本周目标**：自己写出 GPT-2 small，载入官方权重，结果和 HF 版对得上。还要把环境、云 GPU 和公开仓库都建好，以后每周都在这套东西上干活。

**过关标准（11/8 周日检查，按文档严格执行）**
1. **对齐测试**：用 `test_equiv.py` 和 HF `GPT2LMHeadModel` 比 `log_softmax`。取 5 段文本，每段至少 64 token，最大绝对误差要小于 1e-4。两边要在同一设备上、用 fp32、关闭 TF32、激活函数用 `gelu_new`。另外 3 个 prompt 各 greedy 生成 30 token，结果要和 HF 完全一致。别直接比 raw logits。
2. **阶段 0 测验**：
   - 概念题 7 道，闭卷 60 分钟，要答对 80% 以上；
   - **0-P1**：6 小时从空白文件重写，只能查 PyTorch 文档，不能用 AI；
   - **0-P2**：残差流手绘图加 10 分钟口述录音。
3. **超时规则**：最多可以占用第 2 周的周一、周二。这样 Mathematical Framework 前半要挪到周三、周四读，ARENA「mech interp 入门」只做前两节。

**按天安排（约 45 小时，每天另留 30 分钟写学习日志）**

| 日期 | 时长 | 内容 |
|---|---|---|
| 周一 11/2 | 7h | 注册 RunPod 并预充值（11 月上限 US$60，关掉自动充值）；建 network volume；开一个公开 GitHub 学习仓库；建 env-arena，生成 lock 文件并提交；在云端 notebook 里成功加载 GPT-2 small。剩下的时间开始读 ARENA「从零搭建 Transformer」的开头部分 |
| 周二 11/3 | 7h | ARENA：LayerNorm、Embed、PosEmbed、Attention（含 causal mask）、MLP |
| 周三 11/4 | 7h | 拼成 TransformerBlock 和完整模型，载入权重；写采样函数（greedy、temperature、top-k） |
| 周四 11/5 | 7h | 写 `test_equiv.py` 对齐 HF，调到通过两条标准。卡在线代上就查 3Blue1Brown |
| 周五 11/6 | 7h | 不看代码，凭记忆复述每个组件的张量形状；把最没把握的部分（attention、LN）默写一遍；练习画 0-P2 的残差流图 |
| 周六 11/7 | 7h | **0-P1 闭卷重写，限时 6 小时**，最后跑 `test_equiv.py` 验收 |
| 周日 11/8 | 3h | 概念题（60 分钟）；0-P2 画图和录音（1 小时）；写 200 字周报；和我做周检查 |

- [x] 环境设置
  - [x] 在公开仓库里建好 env-arena，并提交 lock 文件
  - [x] 在 Vast.ai 租好 GPU (vRam 24GB, Disk >= 40GB)
- [ ] ARENA (Transofrmer from Scratch).
  - [ ] 自己写出 GPT-2 Small
  - [ ] 载入官方权重
  - [ ] 能采样 (greedy, temperature, top-k)
- [ ] model.py + test_equiv.py: 自己的实现和 HF 版 GPT-2 对齐
- [ ] 阶段 0 测验

# Day 1 Summary
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
| 空闲自动关机 | ⏳ 还没做 |
| 开始读 ARENA | ⏳ |


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
