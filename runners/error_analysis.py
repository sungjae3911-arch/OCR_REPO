import os
import sys
import json
import cv2
import numpy as np
import hydra
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import koreanize_matplotlib
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

sys.path.append(os.getcwd())
from ocr.lightning_modules import get_pl_modules_by_cfg  # noqa: E402

CONFIG_DIR = os.environ.get('OP_CONFIG_DIR') or '../configs'


def box_iou(box1, box2):
    """두 quad box의 IoU 계산 (bounding rect 기반)"""
    def to_rect(box):
        pts = np.array(box).reshape(-1, 2)
        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        x2, y2 = pts[:, 0].max(), pts[:, 1].max()
        return x1, y1, x2, y2

    ax1, ay1, ax2, ay2 = to_rect(box1)
    bx1, by1, bx2, by2 = to_rect(box2)

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union


def match_boxes(pred_boxes, gt_boxes, iou_thresh=0.5):
    """예측 박스와 GT 박스 매칭 → TP, FP, FN 반환"""
    matched_gt = set()
    matched_pred = set()

    for pi, pb in enumerate(pred_boxes):
        for gi, gb in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            if box_iou(pb, gb) >= iou_thresh:
                matched_gt.add(gi)
                matched_pred.add(pi)
                break

    tp = len(matched_pred)
    fp_boxes = [pb for pi, pb in enumerate(pred_boxes) if pi not in matched_pred]
    fn_boxes = [gb for gi, gb in enumerate(gt_boxes) if gi not in matched_gt]
    tp_boxes = [pb for pi, pb in enumerate(pred_boxes) if pi in matched_pred]

    return tp, fp_boxes, fn_boxes, tp_boxes


def draw_boxes_on_image(img, tp_boxes, fp_boxes, fn_boxes):
    """TP(초록), FP(빨강), FN(파랑) 박스 시각화"""
    vis = img.copy()
    for box in tp_boxes:
        pts = np.array(box).reshape(-1, 2).astype(np.int32)
        cv2.polylines(vis, [pts], True, (0, 200, 0), 2)
    for box in fp_boxes:
        pts = np.array(box).reshape(-1, 2).astype(np.int32)
        cv2.polylines(vis, [pts], True, (0, 0, 255), 2)
    for box in fn_boxes:
        pts = np.array(box).reshape(-1, 2).astype(np.int32)
        cv2.polylines(vis, [pts], True, (255, 0, 0), 2)
    return vis


@hydra.main(config_path=CONFIG_DIR, config_name='test', version_base='1.2')
def analyze(config):
    import lightning.pytorch as pl
    pl.seed_everything(config.get('seed', 42))

    model_module, data_module = get_pl_modules_by_cfg(config)

    # checkpoint 로드
    ckpt_path = config.get('checkpoint_path')
    assert ckpt_path, "checkpoint_path를 지정하세요"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from ocr.lightning_modules.ocr_pl import OCRPLModule
    pl_model = OCRPLModule.load_from_checkpoint(
        ckpt_path,
        model=model_module.model,
        dataset=model_module.dataset,
        config=model_module.config,
    )
    pl_model.eval().to(device)

    # val GT 로드
    with open('/root/ocr/data/datasets/jsons/val.json') as f:
        val_ann = json.load(f)

    val_img_dir = Path('/root/ocr/data/datasets/images/val')
    data_module.setup('test')
    test_loader = data_module.test_dataloader()

    # 이미지별 결과 수집
    results = {}
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Inference'):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            pred = pl_model.model(return_loss=False, **batch)
            boxes_batch, _ = pl_model.model.get_polygons_from_maps(batch, pred)

            for i, fname in enumerate(batch['image_filename']):
                pred_boxes = [b for b in boxes_batch[i]]

                if fname not in val_ann['images']:
                    continue
                gt_words = val_ann['images'][fname]['words']
                gt_boxes = [w['points'] for w in gt_words.values() if w['points']]

                tp, fp_boxes, fn_boxes, tp_boxes = match_boxes(pred_boxes, gt_boxes)
                precision = tp / (tp + len(fp_boxes) + 1e-6)
                recall = tp / (tp + len(fn_boxes) + 1e-6)
                hmean = 2 * precision * recall / (precision + recall + 1e-6)

                results[fname] = dict(
                    hmean=hmean, precision=precision, recall=recall,
                    fp_boxes=fp_boxes, fn_boxes=fn_boxes, tp_boxes=tp_boxes,
                    n_gt=len(gt_boxes), n_pred=len(pred_boxes)
                )

    # 최악 20개 시각화
    worst = sorted(results.items(), key=lambda x: x[1]['hmean'])[:20]

    out_dir = Path('outputs/error_analysis')
    out_dir.mkdir(parents=True, exist_ok=True)

    # 통계 출력
    hmeans = [v['hmean'] for v in results.values()]
    print(f'\n=== Val 전체 통계 ===')
    print(f'평균 H-Mean: {np.mean(hmeans):.4f}')
    print(f'H-Mean < 0.5 이미지: {sum(1 for h in hmeans if h < 0.5)}개')
    print(f'H-Mean < 0.8 이미지: {sum(1 for h in hmeans if h < 0.8)}개')
    print(f'H-Mean < 0.9 이미지: {sum(1 for h in hmeans if h < 0.9)}개')

    # FP/FN 패턴 분석
    total_fp = sum(len(v['fp_boxes']) for v in results.values())
    total_fn = sum(len(v['fn_boxes']) for v in results.values())
    total_tp = sum(len(v['tp_boxes']) for v in results.values())
    print(f'\n=== FP/FN 분석 ===')
    print(f'총 TP: {total_tp}, FP: {total_fp}, FN: {total_fn}')
    print(f'FP 크기 분포 (오검출):')
    fp_areas = []
    for v in results.values():
        for box in v['fp_boxes']:
            pts = np.array(box).reshape(-1, 2)
            w = pts[:, 0].max() - pts[:, 0].min()
            h = pts[:, 1].max() - pts[:, 1].min()
            fp_areas.append(w * h)
    fn_areas = []
    for v in results.values():
        for box in v['fn_boxes']:
            pts = np.array(box).reshape(-1, 2)
            w = pts[:, 0].max() - pts[:, 0].min()
            h = pts[:, 1].max() - pts[:, 1].min()
            fn_areas.append(w * h)
    if fp_areas:
        print(f'  FP 평균 크기: {np.mean(fp_areas):.0f}px², 중앙값: {np.median(fp_areas):.0f}px²')
    if fn_areas:
        print(f'  FN 평균 크기: {np.mean(fn_areas):.0f}px², 중앙값: {np.median(fn_areas):.0f}px²')

    # 최악 20개 시각화
    fig, axes = plt.subplots(4, 5, figsize=(25, 20))
    axes = axes.flatten()

    for idx, (fname, info) in enumerate(worst):
        img_path = val_img_dir / fname
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        vis = draw_boxes_on_image(img, info['tp_boxes'], info['fp_boxes'], info['fn_boxes'])

        # 이미지 축소 (시각화용)
        h, w = vis.shape[:2]
        scale = 400 / max(h, w)
        vis = cv2.resize(vis, (int(w * scale), int(h * scale)))

        axes[idx].imshow(vis)
        axes[idx].set_title(
            f'{fname[:30]}\nH={info["hmean"]:.3f} P={info["precision"]:.3f} R={info["recall"]:.3f}\n'
            f'GT:{info["n_gt"]} Pred:{info["n_pred"]} FP:{len(info["fp_boxes"])} FN:{len(info["fn_boxes"])}',
            fontsize=7
        )
        axes[idx].axis('off')

    plt.suptitle('Worst 20 Cases  |  Green=TP  Red=FP  Blue=FN', fontsize=14)
    plt.tight_layout()
    save_path = out_dir / 'worst20_analysis.png'
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f'\n시각화 저장: {save_path}')

    # FP 크기별 분류
    small_fp = sum(1 for a in fp_areas if a < 500)
    medium_fp = sum(1 for a in fp_areas if 500 <= a < 5000)
    large_fp = sum(1 for a in fp_areas if a >= 5000)
    small_fn = sum(1 for a in fn_areas if a < 500)
    medium_fn = sum(1 for a in fn_areas if 500 <= a < 5000)
    large_fn = sum(1 for a in fn_areas if a >= 5000)
    print(f'\n=== 크기별 분류 ===')
    print(f'FP - 소(<500px²):{small_fp} 중(500~5000):{medium_fp} 대(>5000):{large_fp}')
    print(f'FN - 소(<500px²):{small_fn} 중(500~5000):{medium_fn} 대(>5000):{large_fn}')


if __name__ == '__main__':
    analyze()
