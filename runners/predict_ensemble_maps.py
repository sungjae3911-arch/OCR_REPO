"""
Probability map averaging ensemble: v4 + v12
두 모델의 prob_maps를 평균낸 뒤 postprocess → 제출 파일 생성
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

sys.path.append(os.getcwd())

from ocr.lightning_modules import get_pl_modules_by_cfg
from ocr.lightning_modules.ocr_pl import OCRPLModule

CONFIG_DIR = os.environ.get('OP_CONFIG_DIR') or '../configs'

CKPT_V4  = 'outputs/ocr_training/checkpoints/epoch=9-step=8180.ckpt'
CKPT_V12 = 'outputs/ocr_training/checkpoints/epoch=8-step=7362-v5.ckpt'


@hydra.main(config_path=CONFIG_DIR, config_name='predict', version_base='1.2')
def predict_ensemble_maps(config):
    pl.seed_everything(config.get('seed', 42), workers=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_module, data_module = get_pl_modules_by_cfg(config)
    data_module.setup('predict')
    predict_loader = data_module.predict_dataloader()

    # 두 모델 로드
    print('[Ensemble] Loading v4...')
    model_v4 = OCRPLModule.load_from_checkpoint(
        CKPT_V4,
        model=model_module.model,
        dataset=model_module.dataset,
        config=model_module.config,
    ).to(device).eval()

    print('[Ensemble] Loading v12...')
    model_v12 = OCRPLModule.load_from_checkpoint(
        CKPT_V12,
        model=model_module.model,
        dataset=model_module.dataset,
        config=model_module.config,
    ).to(device).eval()

    submission = OrderedDict(images=OrderedDict())

    print('[Ensemble] Running inference with prob map averaging...')
    with torch.no_grad():
        for batch in tqdm(predict_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            pred_v4  = model_v4.model(return_loss=False, **batch)
            pred_v12 = model_v12.model(return_loss=False, **batch)

            # prob_maps 평균
            avg_prob_maps = 0.3 * pred_v4['prob_maps'] + 0.7 * pred_v12['prob_maps']
            avg_pred = {'prob_maps': avg_prob_maps}

            boxes_batch, _ = model_v12.model.get_polygons_from_maps(batch, avg_pred)

            for i, fname in enumerate(batch['image_filename']):
                words = OrderedDict()
                for j, box in enumerate(boxes_batch[i]):
                    words[f'{j + 1:04d}'] = OrderedDict(points=box)
                submission['images'][fname] = OrderedDict(words=words)

    # 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(config.submission_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{timestamp}_ensemble_maps_v4x03_v12x07.json'

    with out_path.open('w') as fp:
        json.dump(submission, fp, indent=4)

    total = sum(len(v['words']) for v in submission['images'].values())
    print(f'\n[Ensemble] Saved: {out_path}')
    print(f'[Ensemble] Total boxes: {total}')


if __name__ == '__main__':
    predict_ensemble_maps()
