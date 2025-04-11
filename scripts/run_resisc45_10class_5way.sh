# bash scripts/run_resisc45_10class_5way.sh 1 2 4 8 16

shots=($1)
for shot in "${shots[@]}"; do
    for seed in 1 2 3; do
        CUDA_VISIBLE_DEVICES=0 python mplnet/train.py \
            --seed $seed\
            --clip_name clip_resnet50 \
            --exp_name nwpu_resisc45_10class_5way_shot_${shot} \
            --save_dir ./results \
            --dataset_name nwpu_resisc45_10class_5way \
            --max_epoch 50\
            --batch_size 5 \
            --eval_ood False \
            --eval_domains False \
            --eval_freq 50 \
            --checkpointing_segments 0 \
            --use_local_features True \
            --lr_init 0.002 \
            --warmup_epoch 1 \
            --num_shots ${shot} \
            --parallel_text_encoder True \
            --parallel_vision_encoder True \
            \
            --learn_global_prompts True \
            --n_global_prompts 4 \
            --learn_regional_prompts True \
            --n_regional_prompts 4 \
            --learn_local_prompts True \
            --n_local_prompts 4 \

    done
done