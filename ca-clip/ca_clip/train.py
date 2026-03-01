"""
CA-CLIP Training Script
Trains a Compositional-Aware CLIP model with Jaccard-weighted contrastive learning.
"""
import argparse
import logging
import math
import os
import random
import sys
import copy

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Package imports
from . import config as option
from .models import create_model
from .data import create_dataloader, create_dataset
from .data.data_sampler import DistIterSampler
from .data.util import bgr2ycbcr
from . import utils as util

# OpenCLIP imports
from . import open_clip
from .open_clip.ca_clip_loss import CAClipLoss

# Wandb for logging
import wandb


def init_dist(backend="nccl", **kwargs):
    """Initialization for distributed training"""
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
    rank = int(os.environ["RANK"])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main():
    #### Setup options
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, help="Path to option YAML file.")
    parser.add_argument(
        "--launcher", choices=["none", "pytorch"], default="none", help="job launcher"
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)
    opt = option.dict_to_nonedict(opt)

    seed = opt["train"]["manual_seed"]

    #### Distributed training settings
    if args.launcher == "none":
        opt["dist"] = False
        rank = -1
        print("Disabled distributed training.")
    else:
        opt["dist"] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    torch.backends.cudnn.benchmark = True

    #### Loading resume state if exists
    if opt["path"].get("resume_state", None):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt["path"]["resume_state"],
            map_location=lambda storage, loc: storage.cuda(device_id),
        )
        option.check_resume(opt, resume_state["iter"])
    else:
        resume_state = None

    #### Create directories and loggers
    if rank <= 0:
        if resume_state is None:
            util.mkdir_and_rename(opt["path"]["experiments_root"])
            util.mkdirs(
                (
                    path
                    for key, path in opt["path"].items()
                    if not key == "experiments_root"
                    and "pretrain_model" not in key
                    and "resume" not in key
                )
            )
            os.system("rm ./log")
            os.symlink(os.path.join(opt["path"]["experiments_root"], ".."), "./log")

        # Setup loggers
        util.setup_logger(
            "base",
            opt["path"]["log"],
            "train_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        util.setup_logger(
            "val",
            opt["path"]["log"],
            "val_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        logger = logging.getLogger("base")
        logger.info(option.dict2str(opt))
        
        # Weights & Biases logger
        if opt["use_wandb"] and "debug" not in opt["name"]:
            wandb.init(
                project=opt.get("wandb_project", "ca-clip"),
                name=opt["name"],
                config=opt,
                dir="log/{}/".format(opt["name"]),
                resume="allow" if opt["path"].get("resume_state") else None
            )
    else:
        util.setup_logger(
            "base", opt["path"]["log"], "train", level=logging.INFO, screen=False
        )
        logger = logging.getLogger("base")

    #### Create train and val dataloaders
    train_loader = None
    val_loader = None
    dataset_ratio = 200
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
            total_iters = int(opt["train"]["niter"])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt["dist"]:
                train_sampler = DistIterSampler(
                    train_set, world_size, rank, dataset_ratio
                )
                total_epochs = int(
                    math.ceil(total_iters / (train_size * dataset_ratio))
                )
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info(
                    "Number of train images: {:,d}, iters: {:,d}".format(
                        len(train_set), train_size
                    )
                )
                logger.info(
                    "Total epochs needed: {:d} for iters {:,d}".format(
                        total_epochs, total_iters
                    )
                )
        elif phase == "val":
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info(
                    "Number of val images in [{:s}]: {:d}".format(
                        dataset_opt["name"], len(val_set)
                    )
                )
        else:
            raise NotImplementedError("Phase [{:s}] is not recognized.".format(phase))
    assert train_loader is not None
    # val_loader is optional
    if val_loader is None:
        if rank <= 0:
            logger.info("No validation dataset provided. Skipping validation during training.")

    #### Create model (for image restoration)
    model = create_model(opt)
    device = model.device

    #### Load CLIP model
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k'
    )
    if rank <= 0:
        logger.info("Loaded pretrained CLIP ViT-B-32 (will use for CA-CLIP without degradation control)")
    
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    clip_model = clip_model.to(device)

    #### Initialize CA-CLIP loss
    ca_clip_enabled = opt.get('ca_clip', {}).get('enabled', False)
    if ca_clip_enabled:
        ca_loss_fn = CAClipLoss(
            temperature=opt['ca_clip'].get('temperature', 0.1),
            local_loss=False,
            gather_with_grad=False,
            rank=rank if opt["dist"] else 0,
            world_size=world_size if opt["dist"] else 1,
            use_horovod=False
        ).to(device)
        ca_weight = opt['ca_clip'].get('ca_weight', 1.0)
        degra_weight = opt['ca_clip'].get('degra_weight', 0.5)
        
        if rank <= 0:
            logger.info("="*80)
            logger.info("CA-CLIP Configuration:")
            logger.info(f"  Temperature: {opt['ca_clip'].get('temperature', 0.1)}")
            logger.info(f"  CA Weight: {ca_weight}")
            logger.info(f"  Degradation Weight: {degra_weight}")
            logger.info(f"  Num Variants: {opt['ca_clip'].get('num_variants', 256)}")
            logger.info(f"  Compound Prob: {opt['ca_clip'].get('compound_prob', 0.3)}")
            logger.info("="*80)
    else:
        ca_loss_fn = None
        ca_weight = 0.0
        degra_weight = 1.0
        if rank <= 0:
            logger.info("CA-CLIP disabled. Using standard training.")

    #### Resume training
    if resume_state:
        logger.info(
            "Resuming training from epoch: {}, iter: {}.".format(
                resume_state["epoch"], resume_state["iter"]
            )
        )
        start_epoch = resume_state["epoch"]
        current_step = resume_state["iter"]
        model.resume_training(resume_state)
    else:
        current_step = 0
        start_epoch = 0

    #### Setup SDE for image restoration
    sde = util.IRSDE(
        max_sigma=opt["sde"]["max_sigma"],
        T=opt["sde"]["T"],
        schedule=opt["sde"]["schedule"],
        eps=opt["sde"]["eps"],
        device=device
    )
    sde.set_model(model.model)

    # Scale is only needed for SR tasks
    scale = opt.get('degradation', {}).get('scale', 1)

    #### Training
    logger.info(
        "Start training from epoch: {:d}, iter: {:d}".format(start_epoch, current_step)
    )

    best_psnr = 0.0
    best_iter = 0
    error = mp.Value('b', False)

    os.makedirs('image', exist_ok=True)

    # Track CA loss statistics
    ca_loss_accumulator = 0.0
    ca_loss_count = 0

    for epoch in range(start_epoch, total_epochs + 1):
        if opt["dist"]:
            train_sampler.set_epoch(epoch)
            
        for batch_idx, train_data in enumerate(train_loader):
            current_step += 1

            if current_step > total_iters:
                break

            # Check if we're using PreDegradedMultiVariant dataset (CA-CLIP mode)
            if 'variants' in train_data:
                # ========== CA-CLIP Training Mode ==========
                clean_img = train_data["clean"]  # [B, 3, H, W]
                variants = train_data["variants"]  # [B*m, 3, H, W]
                clean_clip = train_data["clean_clip"].to(device)  # [B, 3, 224, 224]
                variants_clip = train_data["variants_clip"].to(device)  # [B*m, 3, 224, 224]
                deg_labels = train_data["deg_labels"]  # List[str] of length B*m
                num_variants = train_data["num_variants"]  # m
                
                batch_size = clean_img.shape[0]
                
                # For restoration training, use first variant of each clean image
                LQ = variants.view(batch_size, num_variants, *variants.shape[1:])[:, 0].to(device)
                GT = clean_img.to(device)
                
                # Compute CLIP features for CA loss
                with torch.no_grad(), torch.cuda.amp.autocast():
                    # Standard CLIP - no degradation control
                    variant_features = clip_model.encode_image(variants_clip)
                    variant_features = variant_features.float()
                    variant_degra = None
                    
                    clean_features = clip_model.encode_image(clean_clip)
                    clean_features = clean_features.float()
                    clean_degra = None
                
                # Compute CA contrastive loss
                ca_loss = torch.tensor(0.0, device=device)
                if ca_clip_enabled and ca_loss_fn is not None:
                    ca_loss_dict = ca_loss_fn(
                        distorted_features=variant_features,
                        clean_features=clean_features,
                        degradation_labels=deg_labels,
                        num_variants=num_variants,
                        output_dict=True
                    )
                    ca_loss = ca_loss_dict["ca_contrastive_loss"] * ca_weight
                    
                    # Accumulate for logging
                    ca_loss_accumulator += ca_loss.item()
                    ca_loss_count += 1
                
                # Use features for restoration guidance
                image_context = clean_features
                degra_context = clean_features  # Fallback to image features
                
            else:
                # ========== Standard Training Mode ==========
                LQ, GT, deg_type = train_data["LQ"], train_data["GT"], train_data["type"]
                deg_token = tokenizer(deg_type).to(device)
                img4clip = train_data["LQ_clip"].to(device)
                
                with torch.no_grad(), torch.cuda.amp.autocast():
                    image_context = clip_model.encode_image(img4clip)
                    image_context = image_context.float()
                    degra_context = image_context  # Use same features
                
                ca_loss = torch.tensor(0.0, device=device)
                ca_loss_dict = {}

            # Generate random states for SDE
            timesteps, states = sde.generate_random_states(x0=GT, mu=LQ)

            # Feed data to restoration model
            model.feed_data(
                states, LQ, GT,
                text_context=degra_context,
                image_context=image_context
            )
            
            # Optimize restoration model
            model.optimize_parameters(current_step, timesteps, sde)
            
            # Add CA loss to model's loss dict for logging
            if ca_clip_enabled and ca_loss.item() > 0:
                model.log_dict['ca_loss'] = ca_loss.item()
                for k, v in ca_loss_dict.items():
                    if k != "ca_contrastive_loss" and torch.is_tensor(v):
                        model.log_dict[f'ca_{k}'] = v.item()
            
            model.update_learning_rate(
                current_step, warmup_iter=opt["train"]["warmup_iter"]
            )

            # ========== Logging ==========
            if current_step % opt["logger"]["print_freq"] == 0:
                logs = model.get_current_log()
                message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                    epoch, current_step, model.get_current_learning_rate()
                )
                for k, v in logs.items():
                    message += "{:s}: {:.4e} ".format(k, v)
                    # Wandb logger
                    if opt["use_wandb"] and "debug" not in opt["name"]:
                        if rank <= 0:
                            wandb.log({k: v}, step=current_step)
                
                # Log CA-specific metrics
                if ca_clip_enabled and ca_loss_count > 0:
                    avg_ca_loss = ca_loss_accumulator / ca_loss_count
                    message += "ca_loss_avg: {:.4e} ".format(avg_ca_loss)
                    if opt["use_wandb"] and "debug" not in opt["name"]:
                        if rank <= 0:
                            wandb.log({"ca_loss_avg": avg_ca_loss}, step=current_step)
                    ca_loss_accumulator = 0.0
                    ca_loss_count = 0
                
                if rank <= 0:
                    logger.info(message)

            # ========== Validation ==========
            if current_step % opt["train"]["val_freq"] == 0 and rank <= 0 and val_loader is not None:
                torch.cuda.empty_cache()
                avg_psnr = 0.0
                idx = 0
                
                for _, val_data in enumerate(val_loader):
                    LQ, GT, deg_type = val_data["LQ"], val_data["GT"], val_data["type"]
                    deg_token = tokenizer(deg_type).to(device)
                    img4clip = val_data["LQ_clip"].to(device)
                    
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        image_context, degra_context = clip_model.encode_image(
                            img4clip, control=True
                        )
                        image_context = image_context.float()
                        degra_context = degra_context.float()

                    noisy_state = sde.noise_state(LQ)

                    # Validate
                    model.feed_data(
                        noisy_state, LQ, GT,
                        text_context=degra_context,
                        image_context=image_context
                    )
                    model.test(sde)
                    visuals = model.get_current_visuals()

                    output = util.tensor2img(visuals["Output"].squeeze())
                    gt_img = util.tensor2img(GT.squeeze())
                    lq_img = util.tensor2img(LQ.squeeze())

                    util.save_img(output, f'image/{idx}_{deg_type[0]}_SR.png')
                    util.save_img(gt_img, f'image/{idx}_{deg_type[0]}_GT.png')
                    util.save_img(lq_img, f'image/{idx}_{deg_type[0]}_LQ.png')

                    # Calculate PSNR
                    avg_psnr += util.calculate_psnr(output, gt_img)
                    idx += 1

                    if idx > 99:
                        break

                avg_psnr = avg_psnr / idx

                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    best_iter = current_step

                # Log
                logger.info(
                    "# Validation # PSNR: {:.6f}, Best PSNR: {:.6f}| Iter: {}".format(
                        avg_psnr, best_psnr, best_iter
                    )
                )
                logger_val = logging.getLogger("val")
                logger_val.info(
                    "<epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}".format(
                        epoch, current_step, avg_psnr
                    )
                )
                print(
                    "<epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}".format(
                        epoch, current_step, avg_psnr
                    )
                )
                # Wandb logger
                if opt["use_wandb"] and "debug" not in opt["name"]:
                    wandb.log({"psnr": avg_psnr}, step=current_step)

            if error.value:
                sys.exit(0)
                
            # ========== Save Models ==========
            if current_step % opt["logger"]["save_checkpoint_freq"] == 0:
                if rank <= 0:
                    logger.info("Saving models and training states.")
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info("Saving the final model.")
        model.save("latest")
        logger.info("End of CA-CLIP training.")
        
    # Close wandb
    if opt["use_wandb"] and "debug" not in opt["name"]:
        if rank <= 0:
            wandb.finish()


if __name__ == "__main__":
    main()
