"""
Ensemble predict: v4 (high precision) + v12 (high recall)
두 체크포인트의 박스를 NMS로 합쳐서 제출 파일 생성
"""
import os
import sys
import json
import torch
import lightning.pytorch as pl
import hydra

from datetime import datetime
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm
from shapely.geometry import Polygon

sys.path.append(os.getcwd())

from ocr.lightning_modules import get_pl_modules_by_cfg
from ocr.lightning_modules.ocr_pl import OCRPLModule

CONFIG_DIR = os.environ.get('OP_CONFIG_DIR') or '../configs'


def run_inference(pl_model, dataloader, device):
    results = {}
    pl_model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Inference'):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            pred = pl_model.model(return_loss=False, **batch)
            boxes_batch, scores_batch = pl_model.model.get_polygons_from_maps(batch, pred)
            for i, fname in enumerate(batch['image_filename']):
                results[fname] = {
                    'boxes': boxes_batch[i],
                    'scores': list(scores_batch[i]),
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


@hydra.main(config_path=CONFIG_DIR, config_name='predict', version_base='1.2')
def predict_ensemble(config):
    pl.seed_everything(config.get('seed', 42), workers=True)

    ckpt_v4  = 'outputs/ocr_training/checkpoints/epoch=9-step=8180.ckpt'
    ckpt_v12 = 'outputs/ocr_training/checkpoints/epoch=8-step=7362-v5.ckpt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_module, data_module = get_pl_modules_by_cfg(config)
    data_module.setup('predict')
    predict_loader = data_module.predict_dataloader()

    # v4 inference
    print('\n[Ensemble] Model 1: v4')
    pl_model = OCRPLModule.load_from_checkpoint(
        ckpt_v4,
        model=model_module.model,
        dataset=model_module.dataset,
        config=model_module.config,
    ).to(device)
    results_v4 = run_inference(pl_model, predict_loader, device)

    # v12 inference
    print('\n[Ensemble] Model 2: v12')
    pl_model = OCRPLModule.load_from_checkpoint(
        ckpt_v12,
        model=model_module.model,
        dataset=model_module.dataset,
        config=model_module.config,
    ).to(device)
    results_v12 = run_inference(pl_model, predict_loader, device)

    # NMS merge
    print('\n[Ensemble] Merging with NMS...')
    submission = OrderedDict(images=OrderedDict())

    for fname in sorted(results_v4.keys()):
        boxes_v4   = results_v4[fname]['boxes']
        scores_v4  = results_v4[fname]['scores']
        boxes_v12  = results_v12.get(fname, {}).get('boxes', [])
        scores_v12 = results_v12.get(fname, {}).get('scores', [])

        all_boxes  = boxes_v4 + boxes_v12
        all_scores = scores_v4 + scores_v12

        kept = nms_polygons(all_boxes, all_scores, iou_thresh=0.5)

        words = OrderedDict()
        for i, box in enumerate(kept):
            words[f'{i + 1:04d}'] = OrderedDict(points=box)
        submission['images'][fname] = OrderedDict(words=words)

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(config.submission_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{timestamp}_ensemble_v4_v12.json'

    with out_path.open('w') as fp:
        json.dump(submission, fp, indent=4)

    total = sum(len(v['words']) for v in submission['images'].values())
    print(f'\n[Ensemble] Saved: {out_path}')
    print(f'[Ensemble] Total images: {len(submission["images"])}, Total boxes: {total}')


if __name__ == '__main__':
    predict_ensemble()
