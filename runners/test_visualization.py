import os
import sys
import json
import cv2
import numpy as np
import hydra
import matplotlib.pyplot as plt
import koreanize_matplotlib
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.getcwd())

CONFIG_DIR = os.environ.get('OP_CONFIG_DIR') or '../configs'


@hydra.main(config_path=CONFIG_DIR, config_name='test', version_base='1.2')
def visualize(config):
    import torch
    import lightning.pytorch as pl
    from ocr.lightning_modules import get_pl_modules_by_cfg
    from ocr.lightning_modules.ocr_pl import OCRPLModule

    pl.seed_everything(config.get('seed', 42))
    model_module, data_module = get_pl_modules_by_cfg(config)

    ckpt_path = config.get('checkpoint_path')
    assert ckpt_path, "checkpoint_path를 지정하세요"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pl_model = OCRPLModule.load_from_checkpoint(
        ckpt_path,
        model=model_module.model,
        dataset=model_module.dataset,
        config=model_module.config,
    )
    pl_model.eval().to(device)

    # predict dataloader 사용 (test 이미지, GT 없음)
    data_module.setup('predict')
    predict_loader = data_module.predict_dataloader()

    results = {}
    with torch.no_grad():
        for batch in tqdm(predict_loader, desc='Inference'):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            pred = pl_model.model(return_loss=False, **batch)
            boxes_batch, _ = pl_model.model.get_polygons_from_maps(batch, pred)

            for i, fname in enumerate(batch['image_filename']):
                results[fname] = boxes_batch[i]

    test_img_dir = Path('/root/ocr/data/datasets/images/test')
    out_dir = Path('outputs/test_visualization')
    out_dir.mkdir(parents=True, exist_ok=True)

    # 검출 수 기준 분류
    n_boxes = {fname: len(boxes) for fname, boxes in results.items()}
    sorted_by_boxes = sorted(n_boxes.items(), key=lambda x: x[1])

    least_detected = [f for f, _ in sorted_by_boxes[:10]]   # 적게 검출된 10개
    most_detected  = [f for f, _ in sorted_by_boxes[-10:]]  # 많이 검출된 10개
    median_idx = len(sorted_by_boxes) // 2
    median_detected = [f for f, _ in sorted_by_boxes[median_idx-5:median_idx+5]]  # 중간 10개

    def draw_pred(img_path, boxes):
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for box in boxes:
            pts = np.array(box).reshape(-1, 2).astype(np.int32)
            cv2.polylines(img, [pts], True, (0, 200, 0), 2)
        h, w = img.shape[:2]
        scale = 400 / max(h, w)
        return cv2.resize(img, (int(w * scale), int(h * scale)))

    def save_grid(fnames, title, save_path, results):
        fig, axes = plt.subplots(2, 5, figsize=(25, 10))
        axes = axes.flatten()
        for idx, fname in enumerate(fnames):
            vis = draw_pred(test_img_dir / fname, results[fname])
            if vis is None:
                continue
            axes[idx].imshow(vis)
            axes[idx].set_title(f'{fname[:30]}\n검출 수: {len(results[fname])}', fontsize=7)
            axes[idx].axis('off')
        plt.suptitle(title, fontsize=13)
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f'저장: {save_path}')

    save_grid(least_detected,  'Test 검출 적은 10개 (미검출 의심)', out_dir / 'least_detected.png', results)
    save_grid(most_detected,   'Test 검출 많은 10개 (오검출 의심)', out_dir / 'most_detected.png', results)
    save_grid(median_detected, 'Test 중간 검출 10개',              out_dir / 'median_detected.png', results)

    # 통계 출력
    box_counts = list(n_boxes.values())
    print(f'\n=== Test 검출 통계 ===')
    print(f'평균 검출 수: {np.mean(box_counts):.1f}')
    print(f'최소: {np.min(box_counts)}, 최대: {np.max(box_counts)}, 중앙값: {np.median(box_counts):.0f}')
    print(f'검출 수 < 10: {sum(1 for c in box_counts if c < 10)}개 이미지')
    print(f'검출 수 > 200: {sum(1 for c in box_counts if c > 200)}개 이미지')


if __name__ == '__main__':
    visualize()
