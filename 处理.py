import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# 读取带有中文路径的图片
def read_image_with_chinese_path(image_path):
    with open(image_path, 'rb') as file:
        content = file.read()
    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

# 加载文件夹中的所有背景图
def load_background_images(folder_path):
    background_images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = read_image_with_chinese_path(img_path)
            if img is not None:
                background_images.append(img)
    return background_images

# 计算背景模型
def compute_background_model(background_images):
    if not background_images:
        print("Error: No background images found.")
        return None

    first_image = background_images[0]
    for img in background_images:
        if img.shape != first_image.shape:
            print("Error: All background images must have the same dimensions.")
            return None

    background_model = np.median(background_images, axis=0).astype(np.uint8)
    return background_model

# 使用背景模型去除背景
def remove_background_with_model(img, background_model):
    if img.shape != background_model.shape:
        print("Error: Image and background model must have the same dimensions.")
        return None

    diff = cv2.absdiff(img, background_model)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    return mask

# 二值化处理
def binarize_image(mask):
    _, binary = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    return binary

# 找到并筛选轮廓
def find_and_filter_contours(binary, min_area, max_area):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            filtered_contours.append(contour)
    return filtered_contours

# 定义形态学操作函数
def morphological_operation(img, kernel_size=3):
    # 创建一个核，这里使用矩形核
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    # 进行闭运算
    closed_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return closed_img

# 在图中显示轮廓和面积
def draw_contours_and_show(img, contours):
    for contour in contours:
        area = cv2.contourArea(contour)
        cv2.drawContours(img, [contour],-1, (0, 255, 0), 2)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(img, f"{area:.2f}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # 使用 matplotlib 显示图像
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Detected Objects")
    plt.axis("off")
    plt.show()

# 主函数
def process_image_with_background_model(image_path, background_folder_path, min_area=100, max_area=10000):
    # 加载所有背景图
    background_images = load_background_images(background_folder_path)
    if not background_images:
        print("Error: Unable to load background images.")
        return

    # 计算背景模型
    background_model = compute_background_model(background_images)
    if background_model is None:
        return

    # 读取目标图片
    img = read_image_with_chinese_path(image_path)
    if img is None:
        print("Error: Unable to load image.")
        return

    # 使用背景模型去除背景
    mask = remove_background_with_model(img, background_model)
    if mask is None:
        return

    # 二值化
    binary = binarize_image(mask)

    # 应用形态学闭运算
    closed_img = morphological_operation(binary)

    # 找到并筛选轮廓
    contours, _ = cv2.findContours(closed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            filtered_contours.append(contour)

    # 显示轮廓和面积
    draw_contours_and_show(img, filtered_contours)

if __name__ == "__main__":
    process_image_with_background_model(
        "much.png",  # 替换为你的目标图片路径
        r"D:\\total\\opencv抠图(1)\\背景",  # 替换为你的背景图文件夹路径
        min_area=110,  # 最小面积阈值
        max_area=50000  # 最大面积阈值
    )