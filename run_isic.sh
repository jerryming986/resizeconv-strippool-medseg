#!/usr/bin/env bash
# ISIC2017 / ISIC2018 皮肤病灶二分类分割：训练 + 测试(best/ema/last 三种权重)。
# 复用与 Synapse/ACDC 完全相同的 train.py / test.py / trainer.py，只是 --dataset 不同。
# 数据为 MALUNet/EGE-UNet 的 7:3 布局：<DATA_ROOT>/<isic2017|isic2018>/{train,val}/{images,masks}。
#
# 用法：
#   DATASET=ISIC2017 bash run_isic.sh                              # 默认 = 论文 headline(resize 头 + strip-pooling)
#   DATASET=ISIC2018 bash run_isic.sh
#   DATA_ROOT=/root/autodl-tmp/dataset/ISIC DATASET=ISIC2017 bash run_isic.sh
#   # 消融：关 strip-pooling 只留 resize 头
#   DATASET=ISIC2017 USE_SP=0 bash run_isic.sh
set -u

DATASET="${DATASET:-ISIC2017}"
case "$DATASET" in
  ISIC2017) SUB=isic2017;;
  ISIC2018) SUB=isic2018;;
  *) echo "DATASET 必须是 ISIC2017 或 ISIC2018"; exit 1;;
esac

# 含 isic2017/ 与 isic2018/ 的父目录（服务器上改成你的实际路径）
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/dataset/ISIC}"
DATA="${DATA_ROOT}/${SUB}"

EPOCHS="${EPOCHS:-150}"
REPEAT="${REPEAT:-3}"       # 同 seed 重复跑次数 -> 各进 _run1/_run2/_run3 独立目录
BATCH="${BATCH:-24}"        # 256² 显存吃紧就设 BATCH=16
LR="${LR:-3e-4}"           # AdamW
IMG="${IMG:-256}"          # 数据本身即 256×256
NCLASS=2                    # 背景 + 病灶
SEED="${SEED:-1234}"
CFG=configs/swin_tiny_patch4_window7_224_lite.yaml
OUT_ROOT="${OUT_ROOT:-./model_out}"
PY="${PY:-python}"          # 需要时换绝对路径，如 /root/miniconda3/envs/swinunet/bin/python

# —— 架构开关(train/test 必须一致，决定严格加载)；默认 = 论文两创新点 ——
export HEAD=${HEAD:-resize}        # resize 头(论文) / convtr(基线)
export USE_BIM=${USE_BIM:-0}
export BIM_STAGES=${BIM_STAGES:-0,1,2,3}
export BIM_DECODER=${BIM_DECODER:-1}
export USE_AG=${USE_AG:-0}
export USE_SP=${USE_SP:-1}          # strip-pooling(论文) 默认开
export SP_MODE=${SP_MODE:-add}
export USE_EMA=${USE_EMA:-1}
export EMA_DECAY=${EMA_DECAY:-0.9998}

# 实验名随开关自动区分
if [ -z "${EXP:-}" ]; then
  if [ "$HEAD" = "resize" ]; then EXP="${DATASET}_resize"; else EXP="${DATASET}_convtr"; fi
  [ "$USE_SP"  != "0" ] && EXP="${EXP}_sp"
  [ "$USE_BIM" != "0" ] && EXP="${EXP}_bim"
  [ "$USE_AG"  != "0" ] && EXP="${EXP}_ag"
fi
LAST=epoch_$((EPOCHS-1)).pth
echo "==== $DATASET | DATA=$DATA | HEAD=$HEAD USE_SP=$USE_SP USE_BIM=$USE_BIM USE_AG=$USE_AG | EPOCHS=$EPOCHS REPEAT=$REPEAT BATCH=$BATCH LR=$LR | EXP=$EXP seed=$SEED ===="
if [ ! -d "$DATA/train/images" ]; then
  echo "找不到 $DATA/train/images —— 请用 DATA_ROOT=... 指向含 ${SUB}/ 的父目录"; exit 1
fi

# 同 seed 重复 REPEAT 次，各进独立目录 ${EXP}_seed${SEED}_run{1..REPEAT}（互不覆盖，可断点续跑）
for r in $(seq 1 "$REPEAT"); do
  OUT="${OUT_ROOT}/${EXP}_seed${SEED}_run${r}"
  mkdir -p "$OUT"
  echo "=================== run ${r}/${REPEAT}  ->  ${OUT} ==================="

  # 1) 训练（last-epoch 已存在则跳过，可断点续跑）
  if [ -f "${OUT}/${LAST}" ]; then
    echo "[skip train] ${OUT}/${LAST} 已存在"
  else
    $PY train.py --dataset "$DATASET" --cfg "$CFG" --root_path "$DATA" \
      --max_epochs "$EPOCHS" --output_dir "$OUT" \
      --img_size "$IMG" --base_lr "$LR" --batch_size "$BATCH" \
      --n_class "$NCLASS" --seed "$SEED" 2>&1 | tee "${OUT}/train.log"
  fi

  # 2) 测试 best_model / ema / last（best_model 由 val-DSC 选出，是主报数）
  for CK in best_model.pth ema.pth "$LAST"; do
    [ -f "${OUT}/${CK}" ] || { echo "  !! ${OUT}/${CK} 不存在，跳过"; continue; }
    echo "[test] run ${r} ckpt ${CK}"
    CKPT="$CK" $PY test.py --dataset "$DATASET" --cfg "$CFG" --root_path "$DATA" \
      --max_epochs "$EPOCHS" --output_dir "$OUT" \
      --img_size "$IMG" --base_lr "$LR" --batch_size "$BATCH" \
      --n_class "$NCLASS" --seed "$SEED" 2>&1 | tee "${OUT}/test_${CK}.log"
  done
done

# 3) 汇总：逐 run 的 best_model 结果 + 跨 run 的 mean±std（DSC/IoU）
echo ""
echo "==== 逐 run 结果(best_model.pth: DSC/IoU/Acc/Sen/Spe) ===="
DSCS=""; IOUS=""
for r in $(seq 1 "$REPEAT"); do
  L="${OUT_ROOT}/${EXP}_seed${SEED}_run${r}/test_best_model.pth.log"
  [ -f "$L" ] || { printf "run%-2s (无 best_model 日志)\n" "$r"; continue; }
  line=$(grep 'Testing performance' "$L" | tail -1)
  printf "run%-2s %s\n" "$r" "$line"
  d=$(echo "$line" | sed -n 's/.*performance: DSC : \([0-9.]*\).*/\1/p')
  i=$(echo "$line" | sed -n 's/.* IoU : \([0-9.]*\).*/\1/p')
  [ -n "$d" ] && DSCS="$DSCS $d"
  [ -n "$i" ] && IOUS="$IOUS $i"
done
echo "---- 跨 run 汇总 ----"
echo "$DSCS" | awk '{n=NF; if(n==0){print "  DSC: 无数据"; exit} s=0;for(k=1;k<=n;k++)s+=$k; m=s/n; v=0;for(k=1;k<=n;k++)v+=($k-m)^2; sd=(n>1)?sqrt(v/(n-1)):0; printf "  DSC mean±std = %.4f ± %.4f (n=%d)\n", m, sd, n}'
echo "$IOUS" | awk '{n=NF; if(n==0){print "  IoU: 无数据"; exit} s=0;for(k=1;k<=n;k++)s+=$k; m=s/n; v=0;for(k=1;k<=n;k++)v+=($k-m)^2; sd=(n>1)?sqrt(v/(n-1)):0; printf "  IoU mean±std = %.4f ± %.4f (n=%d)\n", m, sd, n}'
