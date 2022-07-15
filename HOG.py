from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt

# Görseli Okumak
img = imread('Tolga_Ay.png')
plt.axis("off")
plt.imshow(img)
print(img.shape)

# Görselin pixel boyutunu 1:2 oranına getirmek
resized_img = resize(img, (128*4, 64*4))
plt.axis("off")
plt.imshow(resized_img)
print(resized_img.shape)

# Görselleştirme Aşaması
fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), visualize=True, multichannel=True)
plt.axis("off")
plt.imshow(hog_image, cmap="gray")
plt.show()