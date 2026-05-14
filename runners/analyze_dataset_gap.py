"""
Train vs Test 이미지 분포 비교 분석
→ test 이미지 특성을 train augmentation으로 반영하기 위한 근거 수집
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import koreanize_matplotlib
from pathlib import Path
from tqdm import tqdm

TRAIN_DIR = Path('/root/ocr/data/datasets/images/train')
TEST_DIR = Path('/root/ocr/data/datasets/images/test')
OUT_DIR = Path('/root/ocr/baseline_code/outputs/dataset_analysis')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_blur(gray):
    """Laplacian variance → 낮을수록 흐림"""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def compute_rotation_angle(gray):
    """이미지 내 텍스트 라인의 기울기 추정 (HoughLines 기반)"""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    if lines is None or len(lines) < 3:
        return 0.0
    angles = []
    for line in lines[:30]:
        theta = line[0][1]
        angle = np.degrees(theta) - 90
        if abs(angle) < 45:
            angles.append(angle)
    return float(np.median(angles)) if angles else 0.0


def analyze_images(img_dir, label, max_imgs=None):
    paths = sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png'))
    if max_imgs:
        paths = paths[:max_imgs]

    stats = dict(brightness=[], contrast=[], blur=[], aspect_ratio=[],
                 short_side=[], long_side=[], rotation=[])

    for p in tqdm(paths, desc=label):
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        stats['brightness'].append(float(np.mean(gray)))
        stats['contrast'].append(float(np.std(gray)))
        stats['blur'].append(compute_blur(gray))
        stats['aspect_ratio'].append(max(h, w) / min(h, w))
        stats['short_side'].append(min(h, w))
        stats['long_side'].append(max(h, w))
        stats['rotation'].append(compute_rotation_angle(gray))

    return stats


def print_stats(label, stats):
    print(f'\n=== {label} ===')
    for k, v in stats.items():
        arr = np.array(v)
        print(f'  {k:15s}: mean={arr.mean():.1f}  std={arr.std():.1f}  '
              f'min={arr.min():.1f}  median={np.median(arr):.1f}  max={arr.max():.1f}')


def plot_comparison(train_stats, test_stats):
    keys = ['brightness', 'contrast', 'blur', 'aspect_ratio', 'short_side', 'rotation']
    titles = ['밝기 (Brightness)', '대비 (Contrast)', '선명도 (Blur↑=선명)',
              'Aspect Ratio', '단변 길이 (px)', '회전각 (°)']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, (k, title) in enumerate(zip(keys, titles)):
        tr = np.array(train_stats[k])
        te = np.array(test_stats[k])

        # clip blur for readability
        if k == 'blur':
            tr = np.clip(tr, 0, np.percentile(tr, 95))
            te = np.clip(te, 0, np.percentile(te, 95))

        vmin = min(tr.min(), te.min())
        vmax = max(np.percentile(tr, 99), np.percentile(te, 99))
        bins = np.linspace(vmin, vmax, 40)

        axes[i].hist(tr, bins=bins, alpha=0.6, label=f'Train (n={len(tr)})', color='steelblue', density=True)
        axes[i].hist(te, bins=bins, alpha=0.6, label=f'Test  (n={len(te)})',  color='tomato',    density=True)
        axes[i].set_title(title, fontsize=13)
        axes[i].legend(fontsize=9)

        # 평균선
        axes[i].axvline(tr.mean(), color='steelblue', linestyle='--', linewidth=1.5)
        axes[i].axvline(te.mean(), color='tomato',    linestyle='--', linewidth=1.5)

        # 통계 텍스트
        axes[i].text(0.02, 0.97,
            f'Train: μ={tr.mean():.1f} σ={tr.std():.1f}\n'
            f'Test:  μ={te.mean():.1f} σ={te.std():.1f}',
            transform=axes[i].transAxes, va='top', fontsize=8,
            bbox=dict(boxstyle='round', fc='white', alpha=0.7))

    plt.suptitle('Train vs Test 이미지 분포 비교', fontsize=15)
    plt.tight_layout()
    save_path = OUT_DIR / 'train_vs_test_distribution.png'
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    print(f'\n시각화 저장: {save_path}')


def suggest_augmentations(train_stats, test_stats):
    print('\n' + '='*60)
    print('  Augmentation 제안 (Test 분포 기준)')
    print('='*60)

    tr_b = np.mean(train_stats['brightness'])
    te_b = np.mean(test_stats['brightness'])
    tr_c = np.mean(train_stats['contrast'])
    te_c = np.mean(test_stats['contrast'])
    tr_blur = np.mean(train_stats['blur'])
    te_blur = np.mean(test_stats['blur'])
    tr_rot = np.array(train_stats['rotation'])
    te_rot = np.array(test_stats['rotation'])
    tr_short = np.mean(train_stats['short_side'])
    te_short = np.mean(test_stats['short_side'])

    diff_b = te_b - tr_b
    diff_c = te_c - tr_c

    print(f'\n[밝기] Train:{tr_b:.1f} → Test:{te_b:.1f}  (차이:{diff_b:+.1f})')
    if abs(diff_b) > 10 or abs(diff_c) > 10:
        limit = abs(diff_b) / 255 * 1.5
        print(f'  → RandomBrightnessContrast brightness_limit={limit:.2f} contrast_limit={abs(diff_c)/255*1.5:.2f}')
    else:
        print('  → 유사 (augmentation 불필요)')

    print(f'\n[선명도] Train:{tr_blur:.0f} → Test:{te_blur:.0f}')
    if te_blur < tr_blur * 0.7:
        print('  → Test가 더 흐림 → GaussianBlur / MotionBlur 강화 권장')
        ratio = tr_blur / max(te_blur, 1)
        print(f'     blur_limit 상한 {min(int(ratio*3)+1, 9)}~{min(int(ratio*5)+1, 15)} 고려')
    elif te_blur > tr_blur * 1.3:
        print('  → Test가 더 선명 → blur 약하게 or 제거 권장')
    else:
        print('  → 유사')

    print(f'\n[회전] Train p50={np.percentile(abs(tr_rot),50):.1f}° p90={np.percentile(abs(tr_rot),90):.1f}°')
    print(f'       Test  p50={np.percentile(abs(te_rot),50):.1f}° p90={np.percentile(abs(te_rot),90):.1f}°')
    te_rot90 = np.percentile(abs(te_rot), 90)
    print(f'  → Rotate limit={int(te_rot90)+5}° (test 90th percentile + 여유)')

    print(f'\n[이미지 크기] Train short:{tr_short:.0f}px → Test short:{te_short:.0f}px')
    ratio = te_short / tr_short
    if ratio < 0.8:
        print(f'  → Test 이미지가 더 작음 → RandomScale 축소 강화 권장')
    elif ratio > 1.2:
        print(f'  → Test 이미지가 더 큼 → LongestMaxSize 1280으로 늘리기 고려')
    else:
        print('  → 유사')

    # 회전 분포 상세
    buckets = [0, 2, 5, 10, 20, 45]
    print('\n[회전각 분포 상세]')
    print(f'  {"범위":15s} {"Train":>8s} {"Test":>8s}')
    for lo, hi in zip(buckets[:-1], buckets[1:]):
        tr_cnt = np.sum((abs(tr_rot) >= lo) & (abs(tr_rot) < hi))
        te_cnt = np.sum((abs(te_rot) >= lo) & (abs(te_rot) < hi))
        print(f'  {lo:2d}° ~ {hi:2d}°       {tr_cnt/len(tr_rot)*100:6.1f}%  {te_cnt/len(te_rot)*100:6.1f}%')


if __name__ == '__main__':
    print('Train 이미지 분석 중...')
    train_stats = analyze_images(TRAIN_DIR, 'Train')
    print('Test 이미지 분석 중...')
    test_stats = analyze_images(TEST_DIR, 'Test')

    print_stats('Train', train_stats)
    print_stats('Test', test_stats)

    plot_comparison(train_stats, test_stats)
    suggest_augmentations(train_stats, test_stats)
