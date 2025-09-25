import os
import math
import logging
import argparse
from pathlib import Path
from tqdm.auto import tqdm

import torch
import torch.utils.checkpoint
import torch.nn.functional as F

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, save

import diffusers
from diffusers.optimization import get_scheduler
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel

import transformers
from transformers import CLIPVisionModel
from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

from val import log_validation_prism
from utils import save_random_states
from clip_loader import load_clip_model, get_clip_model_path
from dataset.dataset import make_train_dataset, collate_fn
from modules import PRISM
from utils import  get_latest_checkpoint, save_args, code_backup




logger = get_logger(__name__)

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a Diff-Plugin training script.")
    parser.add_argument('--project_path', type=str, required=True, default=None)
    parser.add_argument('--data_root', type=str, required=True, default=None)    
    parser.add_argument("--pretrained_model_name_or_path",type=str,default="CompVis/stable-diffusion-v1-4",required=False,)
    parser.add_argument("--clip_path",type=str,default="auto",required=False,help="Path to CLIP model - use 'auto' for automatic selection")
    parser.add_argument("--output_dir",type=str,default="./results/test",)
    parser.add_argument("--cache_dir",type=str,default="./cache",)
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--resolution",type=int,default=512,)
    parser.add_argument("--revision",type=str,default=None,required=False,)
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader.")
   
    parser.add_argument("--num_train_epochs", type=int, default=1) 
    parser.add_argument("--max_train_steps",type=int,default=None,)
    parser.add_argument("--checkpointing_steps",type=int,default=10,)
    parser.add_argument("--checkpoints_total_limit",type=int,default=None,)
    parser.add_argument("--resume_from_checkpoint",type=str,default=None,)
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1,)
    
    parser.add_argument("--learning_rate",type=float,default=1e-5,)
    parser.add_argument("--scale_lr",action="store_true",default=True,)
    parser.add_argument("--lr_scheduler",type=str,default="constant",)
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles",type=int,default=1,)
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers",type=int,default=4,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument("--logging_dir",type=str,default="logs",)
    parser.add_argument("--allow_tf32",action="store_true",)
    parser.add_argument("--report_to",type=str,default="tensorboard",)
    parser.add_argument("--mixed_precision",type=str,default=None,choices=["no", "fp16", "bf16"],)
    parser.add_argument("--set_grads_to_none",action="store_true",)

    parser.add_argument("--train_data_file_list",type=str,default='data/train/derain.csv',)
    parser.add_argument("--validation_image",type=str,default=["/scratch/yuhliu9/Dataset/DeRain/mixtest/R100L/input/rain-002.png"],nargs="+",)
    parser.add_argument("--validation_steps",type=int,default=10,)
    parser.add_argument("--num_inference_steps",type=int,default=20,help=("diffusion steps for inference process"),)
    parser.add_argument("--tracker_project_name",type=str,default="diff-plugin", help="the name of dataset/task, e.g., derain, desnow")
    parser.add_argument("--used_clip_vision_layers",type=int,default=24,)
    parser.add_argument("--used_clip_vision_global",action="store_true",default=False,)

    parser.add_argument("--down_block_types", type=str, nargs="+", default="CrossAttnDownBlock2D",)
    parser.add_argument("--block_out_channels", type=int, nargs="+", default=320)
    parser.add_argument("--load_weights_from_unet", action="store_true", default=False, help='when change plugin position, this will be false')

    parser.add_argument("--num_cross_proj_layers", type=int, default=2, help='the number of projection layers for cross-att')
    parser.add_argument("--clip_v_dim", type=int, default=1024, choices=[768,1024], help='the dim of last layer of the pre-trained clip-v')
    parser.add_argument("--use_data_aug", action="store_true", default=False, help="use data augmentation or not")
    

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        save_args(args)
        code_backup(args)
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(os.path.join(args.output_dir, "visuals"), exist_ok=True)


    # Initialize PRISM model
    # Convert down_block_types and block_out_channels to lists if needed
    if type(args.down_block_types) != list:
        args.down_block_types = [args.down_block_types]
    if type(args.block_out_channels) != list:
        args.block_out_channels = [args.block_out_channels]
    
    prism_model = PRISM(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        clip_path=args.clip_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_cross_proj_layers=args.num_cross_proj_layers,
        clip_v_dim=args.clip_v_dim,
        used_clip_vision_layers=args.used_clip_vision_layers,
        used_clip_vision_global=args.used_clip_vision_global,
        down_block_types=args.down_block_types,
        block_out_channels=args.block_out_channels,
        load_weights_from_unet=args.load_weights_from_unet,
    )
    
    # Set training mode
    prism_model.set_training_mode(True)

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32.")
    if prism_model.img_net.dtype != torch.float32:
        raise ValueError(f"Image conditioning model loaded as datatype {prism_model.img_net.dtype}. {low_precision_error_string}")

    # Enable TF32 for faster training on Ampere GPUs,
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes)
        logger.info('----------The true learning rate is {} ----------'.format(args.learning_rate))


    # Optimizer creation
    optimizer_class = torch.optim.AdamW
    trainable_params = prism_model.get_trainable_parameters()
    params_to_optimize = [{'params': trainable_params, 'lr': args.learning_rate}]
    assert len(trainable_params) > 0, "No trainable parameters found. Make sure to have at least one of the models enabled."
    optimizer = optimizer_class(params_to_optimize, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,)

    # dataset and dataloader
    train_dataset = make_train_dataset(args, accelerator)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.train_batch_size, num_workers=args.dataloader_num_workers,)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    """
    When the gradient_accumulation_steps option is used, the max_train_steps will be automatically calculated 
    according to the number of epochs and the length of the training dataset
    """
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    # scheduler can be obtained from diffusers.optimization

    # Prepare everything with our `accelerator`.
    optimizer, train_dataloader, lr_scheduler, prism_model = accelerator.prepare(
        optimizer, train_dataloader, lr_scheduler, prism_model
    )


    # For mixed precision training we cast the weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move PRISM components to device with appropriate dtype
    prism_model.to(accelerator.device, dtype=weight_dtype)


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_image")
        tracker_config.pop("down_block_types")
        tracker_config.pop("block_out_channels") 

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            path = get_latest_checkpoint(args.output_dir)

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            if os.path.exists(args.resume_from_checkpoint):
                accelerator.load_state(args.resume_from_checkpoint)
                print('load_state successfully-----------from: ', args.resume_from_checkpoint)
            else:
                # copy the checkpoint to the output_dir and do necessary changes
                accelerator.load_state(os.path.join(args.output_dir, path))
                print('load_state successfully-----------from: ', os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])  # for example, checkpoint-1000

            initial_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    logger.info("***** -------------Note the code for accelerator.accumulate----------------- *****")
    logger.info("***** -------------Note the code for accelerator.accumulate----------------- *****")
    logger.info("***** -------------Note the code for accelerator.accumulate----------------- *****")

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(prism_model):
                # Generate noise and timesteps
                latents = prism_model.encode_image_to_latent(batch["pixel_values"].to(dtype=weight_dtype))
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, prism_model.noise_scheduler_training.config.num_train_timesteps, 
                    (bsz,), device=latents.device
                ).long()

                # Forward through PRISM model
                model_pred, target = prism_model.forward_training(
                    pixel_values=batch["pixel_values"],
                    conditioning_pixel_values=batch["conditioning_pixel_values"],
                    timesteps=timesteps,
                    noise=noise,
                    weight_dtype=weight_dtype
                )
                
                # Compute loss
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(prism_model.get_trainable_parameters(), args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0 or global_step == 1:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        save_random_states(logger, save_path)
                        
                        if global_step != 1:
                            save(optimizer.state_dict(), os.path.join(save_path, 'optimizer.bin'))
                            logger.info(f"Optimizer state saved in {os.path.join(save_path, 'optimizer.bin')}")
                            save(lr_scheduler.state_dict(), os.path.join(save_path, 'scheduler.bin'))
                            logger.info(f"Scheduler state saved in {os.path.join(save_path, 'scheduler.bin')}")
        
                            # Save PRISM model
                            unwrapped_prism = accelerator.unwrap_model(prism_model)
                            unwrapped_prism.save_pretrained(save_path)
                            logger.info(f"Saved PRISM model to {save_path}")

                        logger.info(f"Saved state to {save_path}")

                    if global_step % args.validation_steps == 0:
                        # Use PRISM validation
                        log_validation_prism(logger, prism_model, args, accelerator, global_step)

                        prism_model.set_training_mode(True)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break



    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()

    main(args)
