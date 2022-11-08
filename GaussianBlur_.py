import cv2
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

#attack_img0 = cv2.imread('./data/patch-all/4/patch_20.png')
#out_img_blur = cv2.GaussianBlur(attack_img0, (19, 19), 0)

#cv2.imwrite("./data/save/5.png", out_img_blur)
'''cv2.namedWindow("image")  # 创建一个image的窗口
cv2.imshow("image", out_img_blur)  # 显示图像
cv2.waitKey()  # 默认为0，无限等待
cv2.destroyAllWindows()  # 释放所有窗口'''


def test(img=Image.open("./data/res1.jpg")):
    print(type(img))
    img = torchvision.transforms.ToTensor()(img)
    print(img.shape)
    plt.subplot(1, 2, 1)
    plt.imshow(img.T.transpose(1, 0))
    plt.subplot(1, 2, 2)
    img = TF.perspective(img, [[0, 0], [0, 1259], [2789, 0], [2789, 1259]],
                         [[300, 200], [300, 900], [2000, 100], [2000, 1100]])
    print(img.shape)
    plt.imshow(img.T.transpose(1, 0))
    plt.show()
test()



