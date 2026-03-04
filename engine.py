import math
import os
import sys
from typing import Iterable
import time
import torch
import util.misc as utils
from torch.nn import functional as F
from models.segmentation import loss_masks
from PIL import Image


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    lr_scheduler=None, args=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    step=0

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        step+=1
        model.train()
        samples = samples.to(device)
        captions = [t["caption"] for t in targets]
        cls = [t["cls"] for t in targets]
        init_frames_mask =[targets[idx]["masks"][0] for idx in range(len(targets))] # batch_size of ( H x W ) 
        batch_anchor = []
        for idx in range(len(init_frames_mask)):
            anchor = {}
            key = (0, cls[idx])
            anchor[key] = init_frames_mask[idx]
            batch_anchor.append(anchor)
        outputs = model(samples, captions, cls, targets)
        
        # saving mask during training to check what's being learned
        save_mask = outputs["masks"][0].squeeze().detach().cpu().numpy()
        save_mask = save_mask.reshape(save_mask.shape[0], save_mask.shape[1]).astype('uint8') # np
        save_mask = save_mask > 0.5
        save_mask = Image.fromarray(save_mask)
        save_mask.save("output/viz_check.png")        
        ## 
        losses = {}
        seg_loss = loss_masks(torch.cat(outputs["masks"]), targets, num_frames=samples.tensors.shape[1])
        losses.update(seg_loss)

        loss_dict = losses
        losses = sum(loss_dict[k] for k in loss_dict.keys())
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v for k, v in loss_dict_reduced.items()}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()

        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)

        optimizer.step()
        lr_scheduler.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    