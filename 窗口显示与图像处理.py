import sys
import cv2
import numpy as np
import os
from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel,
                               QFileDialog, QVBoxLayout, QWidget)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt

# ================= 参数区 =================
bg_folder_path = r"D:\total\opencv抠图(1)\背景"
gaussian_kernel = (33, 33)
threshold_value = 35
morph_kernel_size = (21, 21)
morph_iterations = 2
min_contour_area = 200
# =====================================================

def cv2_imread_chinese(path):
    try:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"[Error] 读取失败：{path}  {e}")
        return None


def get_all_png_paths(folder_path):
    png_ext = ('.png', '.PNG')
    paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
             if f.endswith(png_ext)]
    return sorted(paths)


def generate_optimal_mask(img, bg_paths):
    """完全复用 1.py 的多背景平均逻辑"""
    h, w = img.shape[:2]
    diff_accumulator = np.zeros((h, w), dtype=np.float32)

    for bg_path in bg_paths:
        bg = cv2_imread_chinese(bg_path)
        if bg is None:
            print(f"[Warn] 背景图读取失败，已跳过：{os.path.basename(bg_path)}")
            continue
        bg = cv2.resize(bg, (w, h))

        diff = cv2.absdiff(img, bg)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, gaussian_kernel, 0)
        diff_accumulator += blur.astype(np.float32)

    if len(bg_paths) == 0:
        raise RuntimeError("背景文件夹里没找到任何 PNG！")
    avg_diff = (diff_accumulator / len(bg_paths)).astype(np.uint8)
    _, th = cv2.threshold(avg_diff, threshold_value, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_size)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)
    return th


# --------------- UI 部分---------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("单图处理：二值化 + 轮廓面积")
        self.resize(600, 450)

        self.btn_open = QPushButton("1. 选一张图")
        self.btn_open.clicked.connect(self.slot_open)

        self.label_src = QLabel("图片图片图片图片图片图片图片图片图片图片图片图片图片图片图片图片图片图片图片图片图片图片图片图片")
        self.label_src.setScaledContents(True)

        layout = QVBoxLayout()
        layout.addWidget(self.btn_open)
        layout.addWidget(self.label_src)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        # 子窗口占位
        self.win_bin = None
        self.win_cont = None

        # 预载背景模板
        self.bg_paths = get_all_png_paths(bg_folder_path)
        if not self.bg_paths:
            print("[Error] 背景文件夹里没有 PNG！请先检查路径：", bg_folder_path)
        else:
            print(f"[Info] 成功加载 {len(self.bg_paths)} 张背景模板")

    # ---------- 槽：选图 ----------
    def slot_open(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选一张图", "", "Images (*.png *.jpg *.bmp)")
        if not file_path:
            return

        img = cv2_imread_chinese(file_path)
        if img is None:
            return
        self.show_cv_on_label(img, self.label_src)

        # 1. 生成二值 mask
        mask = generate_optimal_mask(img, self.bg_paths)   # 沿用 1.py 完整逻辑
        # 2. 轮廓 + 面积筛选
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cont_img = img.copy()
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < min_contour_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(cont_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cx, cy = x + w // 2, y + h // 2
            cv2.putText(cont_img, f"{area:.0f}", (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 3. 弹出两个窗口
        self.win_bin = self.new_cv_window("二值化", cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
        self.win_cont = self.new_cv_window("轮廓及面积", cont_img)

    # ---------- 工具：cv -> QLabel ----------
    def show_cv_on_label(self, cv_bgr, qlabel):
        rgb = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, w * ch, QImage.Format_RGB888)
        qlabel.setPixmap(QPixmap.fromImage(qimg))

    # ---------- 工具：新建独立窗口 ----------
    def new_cv_window(self, title, cv_bgr):
        win = QWidget()
        win.setWindowTitle(title)
        win.resize(500, 400)
        label = QLabel(win)
        label.setAlignment(Qt.AlignCenter)
        self.show_cv_on_label(cv_bgr, label)
        lay = QVBoxLayout()
        lay.addWidget(label)
        win.setLayout(lay)
        win.show()
        return win


# ---------------- 入口 ----------------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec())