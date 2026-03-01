"""create dataset and dataloader"""
import logging

import torch
import torch.utils.data


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt["phase"]
    
    # Check if we need custom collate_fn for PreDegradedMultiVariant dataset
    collate_fn = None
    if dataset_opt.get("mode") == "PreDegradedMultiVariant":
        from .predegraded_multivariant_dataset import collate_fn_predegraded_multivariant
        collate_fn = collate_fn_predegraded_multivariant
    
    if phase == "train":
        if opt["dist"]:
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt["n_workers"]
            assert dataset_opt["batch_size"] % world_size == 0
            batch_size = dataset_opt["batch_size"] // world_size
            shuffle = False
        else:
            num_workers = dataset_opt["n_workers"] * len(opt["gpu_ids"])
            batch_size = dataset_opt["batch_size"]
            shuffle = True
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True,
            pin_memory=False,
            collate_fn=collate_fn,
        )
    else:
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=(phase=="val"),
            collate_fn=collate_fn
        )


def create_dataset(dataset_opt):
    mode = dataset_opt["mode"]
    if mode == "PreDegradedMultiVariant":  # CA-CLIP with pre-degraded images
        from .predegraded_multivariant_dataset import PreDegradedMultiVariantDataset as D
        dataset = D(dataset_opt)
    else:
        raise NotImplementedError("Dataset [{:s}] is not recognized.".format(mode))

    logger = logging.getLogger("base")
    logger.info(
        "Dataset [{:s} - {:s}] is created.".format(
            dataset.__class__.__name__, dataset_opt["name"]
        )
    )
    return dataset
