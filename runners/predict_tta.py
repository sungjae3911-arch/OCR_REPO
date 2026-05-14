"""
TTA (Test Time Augmentation) predict runner.
Runs inference at two scales (1024 + 1536), merges boxes with NMS.
"""
import os
import sys
import json
import torch
import numpy as np
import albumentations as A
import lightning.pytorch as pl
import hydra

from datetime import datetime
from pathlib import Path
from collections import OrderedDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from shapely.geometry import Polygon

sys.path.append(os.getcwd())

from ocr.lightning_modules import get_pl_modules_by_cfg
from ocr.lightning_modules.ocr_pl import OCRPLModule
from ocr.datasets.base import OCRDataset
from ocr.datasets.transforms import DBTransforms
from ocr.datasets.db_collate_fn import DBCollateFN

CONFIG_DIR = os.environ.get('OP_CONFIG_DIR') or '../configs'


def run_inference(model, dataloader, device):
    results = {}
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Inference'):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            pred = model.model(return_loss=False, **batch)
            boxes_batch, scores_batch = model.model.get_polygons_from_maps(batch, pred)
            for i, fname in enumerate(batch['image_filename']):
                results[fname] = {
                    'boxes': boxes_batch[i],
                    'scores': scores_batch[i],
                }
    return results


def polygon_iou(pts1, pts2):
    try:
        p1 = Polygon(pts1).buffer(0)
        p2 = Polygon(pts2).buffer(0)
        if not p1.is_valid or not p2.is_valid or p1.area == 0 or p2.area == 0:
            return 0.0
        return p1.intersection(p2).area / (p1.union(p2).area + 1e-6)
    except Exception:
        return 0.0


def nms_polygons(boxes, scores, iou_thresh=0.5):
    if not boxes:
        return []
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    removed = set()
    kept = []
    for i in order:
        if i in removed:
            continue
        kept.append(boxes[i])
        for j in order:
            if j == i or j in removed:
                continue
            if polygon_iou(boxes[i], boxes[j]) >= iou_thresh:
                removed.add(j)
    return kept


def make_dataloader(image_dir, scale):
    transform = DBTransforms(
        transforms=[
            A.LongestMaxSize(max_size=scale, p=1.0),
            A.PadIfNeeded(min_width=scale, min_height=scale, border_mode=0, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
        keypoint_params=None,
    )
    dataset = OCRDataset(
        image_path=image_dir,
        annotation_path=None,
        transform=transform,
    )
    collate_fn = DBCollateFN()
    collate_fn.inference_mode = True
    return DataLoader(dataset, batch_size=1, shuffle=False,
                      collate_fn=collate_fn, num_workers=4)


@hydra.main(config_path=CONFIG_DIR, config_name='predict', version_base='1.2')
def predict_tta(config):
    pl.seed_everything(config.get('seed', 42), workers=True)

    ckpt_path = config.get('checkpoint_path')
    assert ckpt_path, "checkpoint_path must be provided"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_module, data_module = get_pl_modules_by_cfg(config)
    pl_model = OCRPLModule.load_from_checkpoint(
        ckpt_path,
        model=model_module.model,
        dataset=model_module.dataset,
        config=model_module.config,
    )
    pl_model = pl_model.to(device)

    image_dir = '/root/ocr/data/datasets/images/test'

    # Scale 1: 1024
    print('\n[TTA] Scale 1: 1024')
    loader_1024 = make_dataloader(image_dir, scale=1024)
    results_1024 = run_inference(pl_model, loader_1024, device)

    # Scale 2: 1536
    print('\n[TTA] Scale 2: 1536')
    loader_1536 = make_dataloader(image_dir, scale=1536)
    results_1536 = run_inference(pl_model, loader_1536, device)

    # Merge with NMS
    print('\n[TTA] Merging predictions...')
    submission = OrderedDict(images=OrderedDict())

    all_fnames = sorted(set(list(results_1024.keys()) + list(results_1536.keys())))
    for fname in all_fnames:
        boxes1 = results_1024.get(fname, {}).get('boxes', [])
        scores1 = results_1024.get(fname, {}).get('scores', [])
        boxes2 = results_1536.get(fname, {}).get('boxes', [])
        scores2 = results_1536.get(fname, {}).get('scores', [])

        all_boxes = boxes1 + boxes2
        all_scores = list(scores1) + list(scores2)

        kept_boxes = nms_polygons(all_boxes, all_scores, iou_thresh=0.5)

        words = OrderedDict()
        for i, box in enumerate(kept_boxes):
            words[f'{i + 1:04d}'] = OrderedDict(points=box)
        submission['images'][fname] = OrderedDict(words=words)

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(config.submission_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{timestamp}_tta.json'

    with out_path.open('w') as fp:
        if config.minified_json:
            json.dump(submission, fp, indent=None, separators=(',', ':'))
        else:
            json.dump(submission, fp, indent=4)

    print(f'\n[TTA] Saved: {out_path}')
    total_boxes = sum(len(v['words']) for v in submission['images'].values())
    print(f'[TTA] Total images: {len(submission["images"])}, Total boxes: {total_boxes}')


if __name__ == '__main__':
    predict_tta()
