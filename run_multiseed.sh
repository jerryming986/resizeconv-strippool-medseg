#!/usr/bin/env bash
# 多 seed 训练 + EMA / last-epoch 自动对比，跑完输出 mean±std 表。
#
# 设计：
#   - 每个 seed 一个独立 output_dir；训练已完成(last-epoch 存在)则自动跳过，可断点续跑。
#   - 每个 seed 训练后，分别测 ema.pth 与 epoch_{N}.pth，解析总分写入 CSV。
#   - 最后调用 aggregate_results.py 出汇总表（也可随时单独再跑该 py 刷新表）。
#
# 用法：
#   bash run_multiseed.sh
#   建议后台跑：nohup bash run_multiseed.sh > multiseed.out 2>&1 &   （或放进 tmux）
#
# 做"无 BIM"对照：复制一份，把 EXP 改名、USE_BIM 改 0 即可（其余不变）。
set -u

# ==================== 配置（按需修改） ====================
# seed 列表：空格或逗号分隔都行（容错）；也可用环境变量覆盖，如 SEEDS="1234" 或 SEEDS=1234,42
SEEDS_RAW="${SEEDS:-1234 42 2024}"
IFS=', ' read -r -a SEEDS <<< "$SEEDS_RAW"

# 同一 seed 重复跑几次（验证同 seed 噪声带）；>1 时各次进独立 _runN 目录互不覆盖。
# 例：测 AG 噪声 -> SEEDS=1234 REPEAT=3 USE_BIM=0 USE_AG=1 bash run_multiseed.sh
REPEAT="${REPEAT:-1}"

EPOCHS=150
BATCH=24
LR=3e-4
IMG=224
NCLASS=9
CFG=configs/swin_tiny_patch4_window7_224_lite.yaml
DATA=/root/autodl-tmp/dataset/project_TransUNet/data/Synapse
OUT_ROOT=./model_out

# —— 实验臂：用环境变量切换，无需改文件 ——
#   无 BIM:           USE_BIM=0 bash run_multiseed.sh                （默认）
#   无 BIM + 注意力门: USE_BIM=0 USE_AG=1 bash run_multiseed.sh
#   有 BIM:           USE_BIM=1 bash run_multiseed.sh
# 架构相关 env（train/test 必须一致，决定模型结构 -> 严格加载）；?=保留外部已设的值
export HEAD=${HEAD:-resize}
export USE_BIM=${USE_BIM:-0}
export BIM_STAGES=${BIM_STAGES:-0,1,2,3}
export BIM_DECODER=${BIM_DECODER:-1}
export USE_AG=${USE_AG:-0}
export USE_SP=${USE_SP:-0}
export SP_MODE=${SP_MODE:-add}        # add=加法上下文(原) / gate=乘法门(治 HD95 不稳)
# EMA（仅训练用）
export USE_EMA=${USE_EMA:-1}
export EMA_DECAY=${EMA_DECAY:-0.9998}

# 实验名随臂自动区分，避免不同臂目录/CSV 撞车（可用 EXP=... 覆盖）
if [ "$USE_BIM" = "0" ]; then EXP=${EXP:-e1_resize_nobim}; else EXP=${EXP:-e1_resize_bim}; fi
[ "$USE_AG" != "0" ] && EXP="${EXP}_ag"
if [ "$USE_SP" != "0" ]; then
  if [ "$SP_MODE" = "gate" ]; then EXP="${EXP}_spgate"; else EXP="${EXP}_sp"; fi
fi

PY=python                            # 需要时换成绝对路径，如 /root/miniconda3/envs/swinunet/bin/python
echo "实验臂: USE_BIM=$USE_BIM  USE_AG=$USE_AG  USE_SP=$USE_SP  SP_MODE=$SP_MODE  HEAD=$HEAD  EXP=$EXP  SEEDS=(${SEEDS[*]})  REPEAT=$REPEAT"
# =========================================================

LAST=epoch_$((EPOCHS-1)).pth
RESULTS=${OUT_ROOT}/${EXP}_results.csv
mkdir -p "$OUT_ROOT"
echo "seed,run,ckpt,dsc,hd95" > "$RESULTS"

record () {   # $1=seed  $2=run  $3=ckpt_name  $4=logfile
  local line dsc hd95
  line=$(grep "Testing performance" "$4" | tail -1)
  dsc=$(echo "$line"  | sed -n 's/.*mean_dice : \([0-9.]*\).*/\1/p')
  hd95=$(echo "$line" | sed -n 's/.*mean_hd95 : \([0-9.]*\).*/\1/p')
  if [ -n "$dsc" ] && [ -n "$hd95" ]; then
    echo "$1,$2,$3,$dsc,$hd95" >> "$RESULTS"
    echo "  -> seed $1 run $2 [$3]: DSC=$dsc HD95=$hd95"
  else
    echo "  !! seed $1 run $2 [$3]: 解析失败，请检查 $4"
  fi
}

for S in "${SEEDS[@]}"; do
  for r in $(seq 1 "$REPEAT"); do
    # REPEAT=1 不加后缀（向后兼容旧目录名）；>1 时各次进独立 _runN 目录，互不覆盖
    if [ "$REPEAT" = "1" ]; then OUT=${OUT_ROOT}/${EXP}_seed${S}; else OUT=${OUT_ROOT}/${EXP}_seed${S}_run${r}; fi
    mkdir -p "$OUT"
    echo "=================== seed ${S}  run ${r}/${REPEAT}  ->  ${OUT} ==================="

    # 1) 训练（last-epoch 已存在则跳过，可断点续跑）
    if [ -f "${OUT}/${LAST}" ]; then
      echo "[skip train] ${OUT}/${LAST} 已存在"
    else
      echo "[train] seed ${S} run ${r} ..."
      $PY train.py --dataset Synapse --cfg "$CFG" --root_path "$DATA" \
        --max_epochs "$EPOCHS" --output_dir "$OUT" \
        --img_size "$IMG" --base_lr "$LR" --batch_size "$BATCH" \
        --n_class "$NCLASS" --seed "$S" 2>&1 | tee "${OUT}/train.log"
    fi

    # 2) 测试 EMA 与 last-epoch
    for CK in ema.pth "$LAST"; do
      if [ ! -f "${OUT}/${CK}" ]; then
        echo "  !! ${OUT}/${CK} 不存在，跳过（EMA 需用新 trainer.py 训练才会生成）"
        continue
      fi
      LOG=${OUT}/test_${CK}.log
      echo "[test] seed ${S} run ${r}  ckpt ${CK} ..."
      CKPT="$CK" $PY test.py --dataset Synapse --cfg "$CFG" --root_path "$DATA" \
        --max_epochs "$EPOCHS" --output_dir "$OUT" \
        --img_size "$IMG" --base_lr "$LR" --batch_size "$BATCH" \
        --n_class "$NCLASS" --split_name test_vol --seed "$S" 2>&1 | tee "$LOG"
      record "$S" "$r" "$CK" "$LOG"
    done
  done
done

# 3) 汇总
echo ""
$PY aggregate_results.py "$RESULTS"
echo "结果明细已存: $RESULTS    （随时可 '$PY aggregate_results.py $RESULTS' 刷新表）"
