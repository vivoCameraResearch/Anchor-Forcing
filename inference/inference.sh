torchrun \
  --nproc_per_node=1 \
  --master_port=1234 \
  inference/inference.py \
  --config_path configs/inference.yaml
