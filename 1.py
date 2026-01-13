import cv2
import numpy as np
import os

#配置参数
bg_folder_path = r"D:\total\opencv抠图(1)\背景"
target_folder_path = r"D:\total\opencv抠图(1)\物体"
output_dir = '抠图与二值化结果'
rgba_output_subdir = os.path.join(output_dir, 'RGBA抠图')
binary_output_subdir = os.path.join(output_dir, '二值化图')

gaussian_kernel = (33, 33)
threshold_value = 35
morph_kernel_size = (21, 21)
morph_iterations = 2
min_contour_area = 200

# 新增：支持中文路径的图片读取
def cv2_imread_chinese(path):
    """
    支持中文路径/特殊字符路径的图片读取，替代cv2.imread()
    :param path: 图片完整路径
    :return: BGR格式图片（None表示读取失败）
    """
    try:
        img_np = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"读取图片失败：{path}，错误信息：{e}")
        return None

# 工具函数1：仅获取png格式图片路径
def get_all_png_paths(folder_path):
    png_extensions = ('.png', '.PNG')
    png_paths = []
    for filename in os.listdir(folder_path):
        if filename.endswith(png_extensions):
            full_png_path = os.path.join(folder_path, filename)
            png_paths.append(full_png_path)
    png_paths.sort()
    return png_paths

# 工具函数2：多背景图生成优化掩码
def generate_optimal_mask(img, bg_paths):
    h, w = img.shape[:2]
    diff_accumulator = np.zeros((h, w), dtype=np.float32)

    for bg_path in bg_paths:
        # 使用中文路径读取函数
        bg = cv2_imread_chinese(bg_path)
        if bg is None:
            print(f"警告：无法读取png背景图 {os.path.basename(bg_path)}，已跳过该图")
            continue
        bg = cv2.resize(bg, (w, h))

        diff = cv2.absdiff(img, bg)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, gaussian_kernel, 0)
        diff_accumulator += blur.astype(np.float32)

    avg_diff = (diff_accumulator / len(bg_paths)).astype(np.uint8)
    _, th = cv2.threshold(avg_diff, threshold_value, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_size)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)
    return th

#主函数：批量处理png目标图
#主函数：批量处理png目标图
def batch_process_target_imgs():
    bg_paths = get_all_png_paths(bg_folder_path)
    target_img_paths = get_all_png_paths(target_folder_path)

    if not bg_paths:
        print(f"错误：在背景文件夹 {bg_folder_path} 中未找到任何png格式图片！")
        return
    if not target_img_paths:
        print(f"错误：在目标文件夹 {target_folder_path} 中未找到任何png格式图片！")
        return
    print(f"成功读取 {len(bg_paths)} 张png背景图")
    print(f"成功读取 {len(target_img_paths)} 张png目标图")

    os.makedirs(rgba_output_subdir, exist_ok=True)
    os.makedirs(binary_output_subdir, exist_ok=True)
    print(f"\n输出文件夹已创建：{output_dir}")
    print(f"RGBA抠图将保存至：{rgba_output_subdir}")
    print(f"二值化图将保存至：{binary_output_subdir}")

    for idx, target_path in enumerate(target_img_paths):
        if not os.path.exists(target_path):
            print(f"\n错误：目标文件不存在 -> {target_path}")
            continue
        img = cv2_imread_chinese(target_path)
        if img is None:
            print(f"\n错误：无法读取png目标图 {os.path.basename(target_path)}，已跳过该图")
            continue
        h, w = img.shape[:2]
        print(f"\n正在处理第 {idx+1}/{len(target_img_paths)} 张png目标图：{os.path.basename(target_path)}")

        optimal_mask = generate_optimal_mask(img, bg_paths)
        cnts, _ = cv2.findContours(optimal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            print(f"警告：{os.path.basename(target_path)} 中未检测到前景物体，已跳过")
            continue

        # ---------- 计算面积 ----------
        areas = []
        object_mask = np.zeros((h, w), dtype=np.uint8)
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < min_contour_area:
                continue
            areas.append(area)
            cv2.drawContours(object_mask, [cnt], -1, 255, -1)

        if not areas:          # 可能所有轮廓都被面积阈值过滤掉
            print(f"警告：{os.path.basename(target_path)} 中有效前景物体面积为0，已跳过")
            continue

        total_area = sum(areas)
        print(f"  - 检测到 {len(areas)} 个有效轮廓，总面积 = {total_area:.2f} 像素")

        b, g, r = cv2.split(img)
        rgba_img = cv2.merge([b, g, r, object_mask])
        binary_img = object_mask.copy()

        img_filename = os.path.basename(target_path)
        img_name, img_ext = os.path.splitext(img_filename)

        # 保存RGBA抠图
        rgba_save_path = os.path.join(rgba_output_subdir, f"{img_name}_rgba.png")
        cv2.imencode('.png', rgba_img)[1].tofile(rgba_save_path)
        print(f"  - RGBA抠图已保存：{os.path.basename(rgba_save_path)}")

        # 保存二值化图
        binary_save_path = os.path.join(binary_output_subdir, f"{img_name}_binary.png")
        cv2.imencode('.png', binary_img)[1].tofile(binary_save_path)
        print(f"  - 二值化图已保存：{os.path.basename(binary_save_path)}")

        # ---------- 画面积到调试图 ----------
        debug_img = img.copy()
        y_offset = 20                                        # 文字起始y坐标
        for i, cnt in enumerate(cnts):
            area = cv2.contourArea(cnt)
            if area < min_contour_area:
                continue
            # 外接矩形中心作为文字位置
            x, y, ww, hh = cv2.boundingRect(cnt)
            cx, cy = x + ww // 2, y + hh // 2
            # 画矩形 & 写面积
            cv2.rectangle(debug_img, (x, y), (x + ww, y + hh), (0, 0, 255), 2)
            text = f"{area:.0f}"
            # 简单背景条，保证中文路径下也能看清
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(debug_img, (cx - tw // 2 - 2, cy - th - 2),
                          (cx + tw // 2 + 2, cy + 2), (0, 0, 0), -1)
            cv2.putText(debug_img, text, (cx - tw // 2, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 在左上角再写一次总面积
        cv2.putText(debug_img, f"TotalArea={total_area:.0f}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        debug_save_path = os.path.join(output_dir, f"{img_name}_debug.jpg")
        cv2.imencode('.jpg', debug_img)[1].tofile(debug_save_path)
        print(f"  - 调试图已保存：{os.path.basename(debug_save_path)}")

    print(f"\n所有可处理的png目标图已完成！结果存放于：{output_dir}")

#执行批量处理
if __name__ == '__main__':
    batch_process_target_imgs()
