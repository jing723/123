# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 读取图像并检查中文路径
# image = cv2.imdecode(np.fromfile('test1.png', dtype=np.uint8), cv2.IMREAD_COLOR)
#
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # Canny边缘检测
# edges = cv2.Canny(gray, 50, 150)
#
# # plt.imshow(edges)
# # plt.show()
#
# # 查找轮廓
# contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# h, w = image.shape[:2]
# blank1 = np.zeros((h, w), dtype=np.uint8)
# blank2 = np.zeros((h, w), dtype=np.uint8)
# blank3 = np.zeros((h, w), dtype=np.uint8)
#
# # 选择前两个轮廓作为示例
# contour1 = contours[0]
# contour2 = contours[1]
# # contour3 = contours[2]
#
# # print(contours)
#
# cv2.drawContours(blank1, [contours[0]], -1, 255, 1)
# cv2.drawContours(blank2, [contours[1]], -1, 255, 1)
# # cv2.drawContours(blank3, [contours[2]], -1, 255, 1)
#
# plt.imshow(blank1)
# plt.show()
# plt.imshow(blank2)
# plt.show()
# # plt.imshow(blank3)
# # plt.show()
#
#
# # 计算前两个轮廓的相似度
# similarity = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0.0)
#
# print('Similarity:', similarity)
