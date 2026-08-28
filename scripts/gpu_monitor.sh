#!/usr/bin/env bash
# 每 10 秒把 GPU 功率/温度/占用/显存追加到 CSV，远程 tail -f 即可实时看。
# 用法:  nohup bash scripts/gpu_monitor.sh runs/gpu_monitor.csv > /dev/null 2>&1 &
set -u
OUT="${1:-runs/gpu_monitor.csv}"
mkdir -p "$(dirname "$OUT")"
if [ ! -f "$OUT" ]; then
  echo "time,power_w,temp_c,util_pct,mem_used_mb,mem_total_mb" > "$OUT"
fi
echo "logging to $OUT (Ctrl-C 停止)"
while true; do
  ts=$(date '+%F %T')
  line=$(nvidia-smi --query-gpu=power.draw,temperature.gpu,utilization.gpu,memory.used,memory.total \
         --format=csv,noheader,nounits 2>/dev/null | sed 's/, /,/g')
  [ -n "$line" ] && echo "$ts,$line" >> "$OUT"
  sleep 10
done
