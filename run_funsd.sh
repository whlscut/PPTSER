#!/usr/share/bash

SHOT=1
SPLIT=0
SEED=89
OUTDIR=outputs/FUNSD/${SHOT}_shot_split_${SPLIT}_seed_${SEED}

python -m torch.distributed.launch \
	--nproc_per_node=1 --master_port 7000 examples/run_funsd.py \
	--dataset_name funsd \
	--data_dir /path/to/FUNSD/split/${SHOT}_shot_split_${SPLIT} \
	--do_train --do_eval \
	--model_name_or_path /path/to/layoutlmv3-base \
	--output_dir ${OUTDIR} \
	--segment_level_layout 1 --visual_embed 1 --input_size 224 \
	--max_steps 2000 --evaluation_strategy steps --eval_steps 100 \
	--learning_rate 1e-5 \
	--per_device_train_batch_size 2 \
	--per_device_eval_batch_size 2 \
	--gradient_accumulation_steps 1 \
	--overwrite_cache True \
	--dataloader_num_workers 0 \
	--logging_steps 10 \
	--load_best_model_at_end \
	--save_strategy steps --save_steps 100 \
	--metric_for_best_model f1 \
	--seed ${SEED}
