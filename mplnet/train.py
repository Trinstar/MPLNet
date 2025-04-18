from typing import Type, Dict, Tuple, Optional
from collections import defaultdict
import os
import math
import argparse

import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR
from clip.clip import _transform
from timm.utils import accuracy

import mplnet.lib as lib
import mplnet.vlprompt.tools as vlp_tools
import mplnet.datasets.tools as dts_tools
from mplnet.datasets import return_train_val_datasets, return_ood_loaders, return_domains_loaders
from mplnet.vlprompt import MPLNet
from mplnet.vlprompt.tools import TotalLoss

NoneType = Type[None]


def train_one_epoch(
    model: MPLNet,
    train_loader: DataLoader,
    loss_fn: TotalLoss,
    optimizer: Optimizer,
    lr_scheduler: _LRScheduler,
    epoch: int,
    fp16_scaler: GradScaler,
    args: argparse.Namespace,
) -> lib.DictAverage:
    meter = lib.DictAverage()
    progress = lib.ProgressMeter(len(train_loader), meter, prefix=f"Epoch: [{epoch}]")

    class_names = train_loader.dataset.all_names

    if not args.learn_global_prompts and not args.learn_regional_prompts and not args.learn_local_prompts:
        with torch.no_grad(), autocast(enabled=args.use_fp16):
            text_features, regional_text_features, local_text_features = model.encode_text(class_names)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            regional_text_features /= regional_text_features.norm(dim=-1, keepdim=True)
            local_text_features /= local_text_features.norm(dim=-1, keepdim=True)
    else:
        text_features = regional_text_features = local_text_features = None

    model.train()
    optimizer.zero_grad()
    track_loader = lib.track(train_loader, f"Epoch {epoch} / {args.max_epoch}")
    
    for i, batch in enumerate(track_loader):
        images = batch["image"].cuda(non_blocking=True)
        targets = batch["target"].cuda(non_blocking=True)
        with autocast(enabled=args.use_fp16):
            global_logits, regional_logits, local_logits, global_prompts_features, regional_prompts_features, local_prompts_features, regional_attention, local_attention = model(images, class_names, text_features, local_text_features)
            
            loss = loss_fn(
                global_logits,
                regional_logits, 
                local_logits,
                targets,
                model.logit_scale.exp(), 
                att_scale = model.att_scale.exp(), 
                rprompts_features = regional_prompts_features, 
                gprompts_features = global_prompts_features,
                lprompts_features = local_prompts_features,
                regional_attention = regional_attention,
                local_attention = local_attention,
                num_shots = args.num_shots
            )

        fp16_scaler.scale(loss).backward()
        track_loader.set_postfix({"gpu": torch.cuda.max_memory_allocated() / 1024**3})
        fp16_scaler.step(optimizer)
        fp16_scaler.update()
        optimizer.zero_grad()

        multi_probs, global_probs, regional_probs, local_probs, l12, l13, l23 = model.create_prediction_scores(global_logits, regional_logits, local_logits, regional_attention, local_attention)

        topk = accuracy(multi_probs, targets, topk=(1,))
        global_topk = accuracy(global_probs, targets, topk=(1,))

        meter.update(
            {
                "loss": loss.detach().item(),
                "top1_multi": topk[0],
                "top1_global": global_topk[0],
            },
            images.size(0),
        )

        if regional_probs is not None:
            regional_topk = accuracy(regional_probs, targets, topk=(1,))
            meter.update(
                {
                    "top1_regional": regional_topk[0],
                },
                images.size(0),
            )

        if local_probs is not None:
            local_topk = accuracy(local_probs, targets, topk=(1,))
            meter.update(
                {
                    "top1_local": local_topk[0],
                },
                images.size(0),
            )

    progress.display_summary()
    lr_scheduler.step()
    return meter


@torch.no_grad()
def evaluate(
    model: MPLNet,
    val_loader: DataLoader,
    args: argparse.Namespace,
    return_scores: bool = False,
) -> Tuple[lib.DictAverage, np.ndarray]:
    meter = lib.DictAverage()

    class_names = val_loader.dataset.all_names

    with autocast(enabled=args.use_fp16):
        text_features, regional_text_features, local_text_features = model.encode_text(class_names)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        regional_text_features /= regional_text_features.norm(dim=-1, keepdim=True)
        local_text_features /= local_text_features.norm(dim=-1, keepdim=True)

    mode = model.training
    model.eval()
    test_scores = np.zeros(len(val_loader.dataset))
    dataset_name = val_loader.dataset.__class__.__name__[:-7]


    for batch in lib.track(val_loader, f"Evaluating on {dataset_name}"):
        images = batch["image"].cuda(non_blocking=True)
        targets = batch["target"].cuda(non_blocking=True)

        with autocast(enabled=args.use_fp16):
            global_logits, regional_logits, local_logits, _, _, _, regional_attention, local_attention = model(images, text_features=text_features, regional_text_features=regional_text_features, local_text_features=local_text_features)

            if return_scores:
                test_scores[batch["index"].numpy()] = model.compute_scores(global_logits, regional_logits)

        multi_probs, global_probs, regional_probs, local_probs, l12, l13, l23 = model.create_prediction_scores(global_logits, regional_logits, local_logits, regional_attention, local_attention)
        global_topk = accuracy(global_probs, targets, topk=(1,))

        if regional_probs is not None:
            regional_topk = accuracy(regional_probs, targets, topk=(1,))

            topk = accuracy(multi_probs, targets, topk=(1,))

            local_topk = accuracy(local_probs, targets, topk=(1,))
            l12_topk = accuracy(l12, targets, topk=(1,))
            l13_topk = accuracy(l13, targets, topk=(1,))
            l23_topk = accuracy(l23, targets, topk=(1,))

            logs = {
                "top1_multi": topk[0],
                "top1_global": global_topk[0],
                "top1_regional": regional_topk[0],
                "top1_local": local_topk[0],
                "l12" : l12_topk[0],
                "l13" : l13_topk[0],
                "l23" : l23_topk[0],
            }

        else:
            logs = {
                "top1_multi": global_topk[0],
                "top1_global": global_topk[0],
            }

        meter.update(logs, images.size(0))

    model.train(mode)
    return meter, test_scores


@torch.no_grad()
def evaluate_ood(
    model:MPLNet,
    val_loader: DataLoader,
    ood_loaders: Dict[str, DataLoader],
    args: argparse.Namespace,
    test_scores: Optional[np.ndarray] = None,
) -> lib.DictAverage:
    metrics = defaultdict(dict)

    class_names = val_loader.dataset.all_names

    with autocast(enabled=args.use_fp16):
        text_features, regional_text_features, local_text_features = model.encode_text(class_names)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        regional_text_features /= regional_text_features.norm(dim=-1, keepdim=True)
        local_text_features /= local_text_features.norm(dim=-1, keepdim=True)

    mode = model.training
    model.eval()
    if test_scores is None:
        test_scores = np.zeros(len(val_loader.dataset))
        for batch in lib.track(val_loader, "Computing ood scores for Test"):
            images = batch["image"].cuda(non_blocking=True)
            with autocast(enabled=args.use_fp16):
                global_logits, regional_logits, local_logits, _, _, _, _ , _ = model(images, text_features=text_features, local_text_features=local_text_features)
                test_scores[batch["index"].numpy()] = model.compute_scores(global_logits, regional_logits, local_logits)

    for ood_name, ood_loader in ood_loaders.items():
        ood_scores = np.zeros(len(ood_loader.dataset))
        for batch in lib.track(ood_loader, f"Computing ood scores for {ood_name}"):
            images = batch["image"].cuda(non_blocking=True)
            with autocast(args.use_fp16):
                global_logits, regional_logits, local_logits, _, _, _, _, _ = model(images, text_features=text_features, local_text_features=local_text_features)
                ood_scores[batch["index"].numpy()] = model.compute_scores(global_logits, regional_logits, local_logits)

        metrics[ood_name]["fpr95"] = lib.get_fpr(test_scores, ood_scores)
        metrics[ood_name]["auroc"] = lib.get_auroc(test_scores, ood_scores)

    model.train(mode)
    return metrics


if __name__ == "__main__":
    clip_model_names = [
        "clip_vit_b32",
        "clip_vit_b16",
        "clip_resnet50",
        "clip_resnet101",
    ]

    parser = argparse.ArgumentParser(" Multi-grained Prompt Learning with Vision-Language Model for Remote Sensing Image Scene Classification")
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument("--data_dir", default="/root/autodl-tmp", type=str)
    parser.add_argument("--save_dir", default="./results/", type=str)
    parser.add_argument("--checkpoint_path", default=None, type=str)
    parser.add_argument("--dataset_name", default="nwpu_resisc45", type=str)
    parser.add_argument("--eval_only", default=False, type=lib.boolean_flags)
    parser.add_argument("--eval_ood", default=False, type=lib.boolean_flags)
    parser.add_argument("--eval_domains", default=False, type=lib.boolean_flags)

    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--num_shots", default=1, type=int, help="Number of shots by class. -1 means the whole dataset")
    parser.add_argument("--use_local_features", default=False, type=lib.boolean_flags)
    parser.add_argument("--use_global_loss", default=True, type=lib.boolean_flags)
    parser.add_argument("--use_local_loss", default=True, type=lib.boolean_flags)
    parser.add_argument("--learn_local_proj", default=True, type=lib.boolean_flags)
    parser.add_argument("--learn_global_prompts", default=True, type=lib.boolean_flags)
    parser.add_argument("--learn_regional_prompts", default=True, type=lib.boolean_flags)
    parser.add_argument("--learn_local_prompts", default=True, type=lib.boolean_flags)
    parser.add_argument("--n_global_prompts", default=1, type=int)
    parser.add_argument("--n_regional_prompts", default=1, type=int)
    parser.add_argument("--n_local_prompts", default=1, type=int)
    parser.add_argument("--global_dropout_p", default=0.75, type=lib.float_range(0.0, 1.0))

    parser.add_argument("--prompts_batch_size", default=math.inf, type=int)

    parser.add_argument("--parallel_text_encoder", default=False, type=lib.boolean_flags)
    parser.add_argument("--parallel_vision_encoder", default=False, type=lib.boolean_flags)

    parser.add_argument("--ood_method", default="GL-MCM", type=str)
    parser.add_argument("--ood_temp_scale", default=1.0, type=float)

    parser.add_argument("--clip_name", required=True, choices=clip_model_names, type=str)

    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--inference_batch_size", default=256, type=int)
    parser.add_argument("--max_epoch", default=50, type=int)
    parser.add_argument("--optimizer", default="sgd", type=str)
    parser.add_argument("--lr_init", default=0.002, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=1e-2, type=float)
    parser.add_argument("--warmup_epoch", default=0, type=int)
    parser.add_argument("--cons_lr", default=1e-5, type=float)

    parser.add_argument("--use_fp16", default=True, type=lib.boolean_flags)
    parser.add_argument("--persistent_workers", default=False, type=lib.boolean_flags)
    parser.add_argument("--checkpointing_segments", default=4, type=int, help="Number of segments used for gradient checkpointing for the text encoder.")

    parser.add_argument("--eval_freq", default=5, type=int)
    parser.add_argument("--save_freq", default=5, type=int)
    parser.add_argument("--print_freq", default=20, type=int)

    args = parser.parse_args()

    lib.setup_logger()
    lib.random_seed(args.seed)

    if args.exp_name is not None:
        lib.LOGGER.info(f"Running shots {args.num_shots}")
        lib.LOGGER.info(f"Running experiment {args.exp_name}")
        args.save_dir = os.path.join(args.save_dir, args.exp_name)


    # seting-up transforms
    train_transform = dts_tools.get_train_transform()
    val_transform = _transform(224)

    # Setting-up dataset train
    train_dataset, val_dataset, template = return_train_val_datasets(args.dataset_name, args.data_dir, train_transform, val_transform, seed=args.seed, num_shots=args.num_shots)
    # The initial value of the learnable prompts is 'a photo of a'
    template = "A photo of a {}" if (args.learn_global_prompts or args.learn_regional_prompts or args.learn_local_prompts) else template     

    lib.LOGGER.info("Using dataset: " + train_dataset.level)
    lib.LOGGER.info("Number of classes: " + train_dataset.num_classes)
    
    if args.dataset_name != "nwpu_resisc45_10class_5way":
        train_dataset = dts_tools.create_few_shots_dataset(train_dataset, args.num_shots, seed=args.seed)
    
    lib.LOGGER.info("Using template: " + template.format("<class_name>"))

    # Setting-up dataloaders
    train_loader = dts_tools.get_train_loader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=10,
        persistent_workers=args.persistent_workers,
    )
    val_loader = dts_tools.get_eval_loader(val_dataset, batch_size=args.inference_batch_size)

    if args.eval_ood:
        ood_loaders = return_ood_loaders(args.data_dir, val_transform)

    if args.eval_domains:
        domains_loaders = return_domains_loaders(args.data_dir, val_transform)

    # Setting-up model
    model =MPLNet(
        clip_name=args.clip_name,
        use_local_features=args.use_local_features,
        checkpointing_segments=args.checkpointing_segments,
        template=template,
        learn_local_proj=args.learn_local_proj,
        learn_local_prompts=args.learn_local_prompts,
        learn_global_prompts=args.learn_global_prompts,
        learn_regional_prompts=args.learn_regional_prompts,
        class_names=train_dataset.all_names,
        n_global_prompts=args.n_global_prompts,
        n_regional_prompts = args.n_regional_prompts,
        n_local_prompts=args.n_local_prompts,
        prompts_batch_size=args.prompts_batch_size,
        ood_method=args.ood_method,
        ood_temp_scale=args.ood_temp_scale,
        parallel_text_encoder=args.parallel_text_encoder,
        parallel_vision_encoder=args.parallel_vision_encoder,
        num_shots=args.num_shots
    )

    model.initialize_prompt()
    lib.LOGGER.info("Model and prompts initialized successfully")
    # eventually load pre-trained prompts
    lib.load_checkpoint(model, args.checkpoint_path)
    
    model.freeze_clip()
    model = model.cuda()
    
    # setting-up loss
    loss_fn = TotalLoss(
        global_dropout_p=args.global_dropout_p,
    )
    
    # Setting-up optimizer
    optimizer = vlp_tools.get_optimizer(args.optimizer, model, args.lr_init, args.weight_decay, args.momentum)
    
    # Setting-up scheduler
    lr_scheduler = CosineAnnealingLR(optimizer, args.max_epoch)
    if args.warmup_epoch > 0:
        lr_scheduler = vlp_tools.ConstantWarmupScheduler(optimizer, lr_scheduler, args.warmup_epoch, args.cons_lr)

    # Setting-up GradScaler for amp
    fp16_scaler = GradScaler(enabled=args.use_fp16)
    
    # Training loop
    for epoch in range(args.max_epoch):
        if not args.eval_only:
            assert args.learn_regional_prompts or args.learn_local_prompts or args.learn_global_prompts, "At least one of learn_local_prompts or learn_global_prompts or learn_regional_prompt must be True"
            train_meter = train_one_epoch(
                model=model,
                train_loader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                epoch=epoch,
                fp16_scaler=fp16_scaler,
                args=args,
            )

            lib.save_checkpoint(args.save_dir, epoch, model, optimizer, lr_scheduler, fp16_scaler, train_meter, args)

        if ((epoch % args.eval_freq == 0) and (epoch > 0)) or (epoch + 1 == args.max_epoch) or args.eval_only:
            lib.LOGGER.info("Evaluation")
            val_meter, test_scores = evaluate(model, val_loader, args, return_scores=args.eval_ood and (args.eval_only or (epoch + 1 == args.max_epoch)))
            lib.LOGGER.info("Evaluation metrics: " + " ".join([" *"] + val_meter.summary()))
            lib.LOGGER.info("seed: " + str(args.seed) + "  num_shots: " + str(args.num_shots))
            lib.LOGGER.info("-------------------------------------------------------end------------------------------------------------------------")

            if args.eval_ood and (args.eval_only or (epoch + 1 == args.max_epoch)):
                ood_metrics = evaluate_ood(model, val_loader, ood_loaders, args, test_scores=test_scores)
                lib.LOGGER.info(f"OOD Evaluation metrics with temperature scale {args.ood_temp_scale} (FPR95 / AUROC): ")
                lib.log_ood_metrics(ood_metrics)

            if args.eval_domains and (args.eval_only or (epoch + 1 == args.max_epoch)):
                metrics = {}
                for domain_name, domain_loader in domains_loaders.items():
                    metrics[domain_name], _ = evaluate(model, domain_loader, args)
                    lib.LOGGER.info(f"Evaluation metrics for {domain_name}: " + " ".join([" *"] + metrics[domain_name].summary()))
                avg_top1 = np.mean([metrics[domain_name].avg["top1"] for domain_name in domains_loaders.keys()])
                lib.LOGGER.info(f"Average evaluation metrics for domains: * top1: {avg_top1: .3f}")

            if args.eval_only:
                break
