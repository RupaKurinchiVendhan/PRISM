task=low_haze_snow

python infer.py \
        --prism_checkpoint_path pre-trained/prism_model.pt \
        --distortion_type $task \        
        --img_path data/demo/cdd.png \
        --save_root temp_results \
        --num_inference_steps 20 \
        --seed 42

python infer.py \
        --prism_checkpoint_path pre-trained/prism_model.pt \
        --distortion_type blur_clouds_defocus \     
        --num_inference_steps 20 \
        --save_root temp_results \
        --seed 42
        --img_path /data/vision/beery/scratch/rupa/image-restoration/OneRestore/data_v4/train/blur_clouds_defocus/000004.png \

# python infer.py         --prism_checkpoint_path pre-trained/prism_model.pt         --distortion_type blur_clouds_defocus         --img_path /data/vision/beery/scratch/rupa/image-restoration/OneRestore/data_v4/train/blur_clouds_defocus/000004.png         --save_root temp_results         --num_inference_steps 20         --seed 42