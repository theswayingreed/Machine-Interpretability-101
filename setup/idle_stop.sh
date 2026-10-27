#!/bin/bash
# =============================================================================
# setup/idle_stop.sh — GPU 闲置时自动 stop 本实例（安全网，不代替每天手动销毁）
#
# 规则（每分钟检查一次）：
#   - 没有任何进程占用显存，连续 NO_MEM_LIMIT 分钟（默认 30）→ stop
#     （notebook kernel 里加载了模型就会占显存，所以读 ARENA、想问题时不会被误关）
#   - GPU 利用率为 0，连续 NO_UTIL_LIMIT 分钟（默认 120）→ stop
#     （防止 kernel 开着、人走了一整晚）
#
# 测试：DRY_RUN=1 NO_MEM_LIMIT=2 bash idle_stop.sh   # 只写日志，不真的 stop
# 日志：/root/idle_stop.log
# =============================================================================
NO_MEM_LIMIT=${NO_MEM_LIMIT:-30}
NO_UTIL_LIMIT=${NO_UTIL_LIMIT:-120}
MEM_IDLE_MIB=${MEM_IDLE_MIB:-100}     # 显存占用低于这个值（MiB）视为没人在用
DRY_RUN=${DRY_RUN:-0}
LOG=/root/idle_stop.log
VASTAI=/root/.local/bin/vastai

log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
log "start: NO_MEM_LIMIT=${NO_MEM_LIMIT}m NO_UTIL_LIMIT=${NO_UTIL_LIMIT}m DRY_RUN=$DRY_RUN instance=${CONTAINER_ID:-?}"

no_mem=0
no_util=0
while true; do
  sleep 60
  read -r util mem < <(nvidia-smi --query-gpu=utilization.gpu,memory.used \
                         --format=csv,noheader,nounits | head -1 | tr -d ',')
  if [ "${mem:-0}" -lt "$MEM_IDLE_MIB" ]; then no_mem=$((no_mem + 1)); else no_mem=0; fi
  if [ "${util:-0}" -eq 0 ];              then no_util=$((no_util + 1)); else no_util=0; fi

  if [ "$no_mem" -ge "$NO_MEM_LIMIT" ] || [ "$no_util" -ge "$NO_UTIL_LIMIT" ]; then
    log "idle: no_mem=${no_mem}m no_util=${no_util}m (util=${util}% mem=${mem}MiB) -> stop"
    if [ "$DRY_RUN" = "1" ]; then
      log "DRY_RUN: 跳过 stop"; no_mem=0; no_util=0
    else
      "$VASTAI" stop instance "$CONTAINER_ID" ${CONTAINER_API_KEY:+--api-key "$CONTAINER_API_KEY"} >> "$LOG" 2>&1
      sleep 600   # 给平台时间执行 stop；如果失败，10 分钟后重试
    fi
  fi
done
