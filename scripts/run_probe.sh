CKPT=runs/cifar10_fedx_s42/global_round_005.pt
python -m src.eval.linear_probe \
  --dataset cifar10 \
  --checkpoint ${CKPT} \
  --epochs 100 \
  --batch_size 256 \
  --seed 42 \
  --device auto \
  --wandb_project FedX_CKD