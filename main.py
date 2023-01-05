# importing cv2
import cv2

# Using cv2.imread() method
img = cv2.imread('Pyramid.jpeg')

# Displaying the image
cv2.imshow('image', img)

im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#Image.fromarray(im_rgb).save('data/dst/lena_rgb_pillow.jpg')

resized_img = cv2.resize(im_rgb  , (512 , 512))
cv2.imshow('img' , resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(resized_img.shape)

blur = cv2.GaussianBlur(resized_img,(5,5),0)

cv2.imshow('img' , blur)
cv2.waitKey(0)
cv2.destroyAllWindows()

src = cv2.pyrDown(blur, dstsize=(blur.shape[0] // 2, blur.shape[1] // 2))
cv2.imshow('canvasOutput', src);
cv2.waitKey(0)
cv2.destroyAllWindows()

def Create_Gaussian_Pyramid (img,size):
    src = cv2.pyrDown(img, dstsize=(size // 2, size // 2))
    cv2.imshow('canvasOutput', src);
    cv2.waitKey(0)
    cv2.destroyAllWindows()