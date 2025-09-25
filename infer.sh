task=low_contrast_color

python infer.py \
        --unified_checkpoint_path pre-trained/unified_checkpoint.pt \
        --distortion_type $task \
        --img_path data/real_demo/low_contrast_color.png \
        --save_root temp_results \
        --num_inference_steps 20 \
        --seed 42