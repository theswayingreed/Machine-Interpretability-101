#!/bin/bash
# =============================================================================
# setup/vast_setup.sh — 在新的 Vast 实例上重建 env-arena
#
# 由 Vast 模板的 On-start Script 自动调用，也可以手动运行：
#   bash /root/Machine-Interpretability-101/setup/vast_setup.sh
# 可以重复运行（幂等）。进度看：tail -f /root/setup.log
#
# 机器要求：Max CUDA >= 13.0（驱动 >= 580）；硬盘 >= 40 GB
# =============================================================================
set -euo pipefail
exec > >(tee -a /root/setup.log) 2>&1
echo "=== setup start $(date) ==="

# ---------- 配置 ----------
REPO_DIR=/root/Machine-Interpretability-101
REPO_SSH=git@github.com:theswayingreed/Machine-Interpretability-101.git
ARENA_DIR=/root/ARENA_materials
ARENA_COMMIT="082f95b8958e316404ead5c0e428082bccc5becb"   # 2026-10-27 的 ARENA_materials main
VENV="$ARENA_DIR/.venv"
LOCK="$REPO_DIR/envs/env-arena.lock"
KERNEL_JSON=/root/.local/share/jupyter/kernels/arena-env/kernel.json
GIT_NAME="theswayingreed"     # 填你的 GitHub 显示名，留空则跳过
GIT_EMAIL="corealitylabs@gmail.com"    # 填你的 commit 邮箱，留空则跳过

add_line() {  # add_line <文件> <行>：行不存在时才追加
  grep -qxF "$2" "$1" 2>/dev/null || echo "$2" >> "$1"
}

# ---------- 1. 环境变量：去掉 NGC 镜像的 TF32 强制和 pip 约束；uv 加进 PATH ----------
# TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 会让 cuBLAS 的 fp32 matmul 强制走 TF32，
# Python 里的 allow_tf32 / set_float32_matmul_precision 都关不掉它。
touch /etc/environment
sed -i '/^TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=/d; /^PIP_CONSTRAINT=/d' /etc/environment
add_line /root/.bashrc 'unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE PIP_CONSTRAINT'
add_line /root/.bashrc 'export PATH="/root/.local/bin:$PATH"'
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE PIP_CONSTRAINT
export PATH="/root/.local/bin:$PATH"

# ---------- 2. Git：fetch 走 HTTPS，push 走 SSH（配合 ssh -A agent forwarding） ----------
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" remote set-url --push origin "$REPO_SSH"
fi
mkdir -p /root/.ssh && chmod 700 /root/.ssh
if ! grep -q '^github.com ' /root/.ssh/known_hosts 2>/dev/null; then
  ssh-keyscan -t ed25519 github.com >> /root/.ssh/known_hosts 2>/dev/null || true
fi
if [ -n "$GIT_NAME" ];  then git config --global user.name  "$GIT_NAME";  fi
if [ -n "$GIT_EMAIL" ]; then git config --global user.email "$GIT_EMAIL"; fi

# ---------- 3. ARENA 材料，固定到指定 commit ----------
if [ ! -d "$ARENA_DIR/.git" ]; then
  git clone -q --depth 1 https://github.com/ARENA-education/ARENA_materials.git "$ARENA_DIR"
fi
git -C "$ARENA_DIR" fetch -q --depth 1 origin "$ARENA_COMMIT"
git -C "$ARENA_DIR" checkout -q "$ARENA_COMMIT"
echo "ARENA at $(git -C "$ARENA_DIR" rev-parse HEAD)"

# ---------- 4. Python 环境 ----------
if [ -f "$LOCK" ]; then
  echo "=== 按 lock 文件安装：$LOCK ==="
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  [ -x "$VENV/bin/python" ] || uv venv --python 3.13 --seed "$VENV"
  uv pip sync --python "$VENV/bin/python" "$LOCK"
  "$VENV/bin/python" -m ipykernel install --user --name arena-env --display-name arena-env
  add_line /root/.bashrc "source $VENV/bin/activate"
else
  echo "=== 没找到 lock 文件，改用 ARENA 自带的 install_uv.sh ==="
  bash "$ARENA_DIR/install_uv.sh" --platform vastai
  echo "!!! 装完后请生成 lock：uv pip freeze --python $VENV/bin/python > $LOCK"
fi

# ---------- 5. Jupyter kernel 也关掉 TF32 强制 ----------
if [ ! -f "$KERNEL_JSON" ]; then
  echo "!!! 找不到 $KERNEL_JSON，kernel 没注册成功"; exit 1
fi
python3 - "$KERNEL_JSON" <<'EOF'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
k = json.loads(p.read_text())
k.setdefault("env", {})["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
p.write_text(json.dumps(k, indent=1))
print("kernel env:", k["env"])
EOF

# ---------- 6. 自检 ----------
"$VENV/bin/python" - <<'EOF'
import os, sys, torch
print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda", torch.version.cuda)
assert torch.cuda.is_available(), "GPU 不可用"
assert os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") in (None, "0"), "TF32 override 还在"
a = torch.randn(1024, 1024, dtype=torch.float64)
b = torch.randn(1024, 1024, dtype=torch.float64)
ref = a @ b
gpu = (a.float().cuda() @ b.float().cuda()).double().cpu()
rel = ((gpu - ref).norm() / ref.norm()).item()
print(f"fp32 GPU matmul vs float64 CPU, rel err = {rel:.2e}   (TF32 关闭时约 1e-7)")
assert rel < 1e-5, "误差过大：TF32 可能还开着"
EOF
df -h /
echo "=== setup done $(date) ==="