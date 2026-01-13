import cv2
import numpy as np
from pathlib import Path

#
def load_img(p: Path):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)

def process(p: Path):
    img = load_img(p)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(cnts, key=cv2.contourArea) if cnts else None

tpl_dir  = Path(r'D:\total\opencv抠图(1)\模板文件夹')
img_dir  = Path(r'D:\total\opencv抠图(1)\图片文件夹')

#载入模板轮廓
templates = {}
for p in tpl_dir.glob('*'):
    if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp'):
        cnt = process(p)
        if cnt is not None:
            templates[p.stem] = cnt

# 批量匹配
for img_path in img_dir.glob('*'):
    cnt_img = process(img_path)

    # 取最相似的一个
    best_k, best_s = min(
        ((k, cv2.matchShapes(cnt_img, cnt_tpl, cv2.CONTOURS_MATCH_I1, 0.0))
         for k, cnt_tpl in templates.items()),
        key=lambda x: x[1]
    )
    print(f'{img_path.name} -> {best_k}  分数:{best_s:.4f}')