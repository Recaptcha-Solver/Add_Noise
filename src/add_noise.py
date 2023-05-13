import numpy as np
import cv2, os, time
import matplotlib.pyplot as plt

# 참고: https://gist.github.com/Prasad9/28f6a2df8e8d463c6ddd040f4f6a028a

def add_gaussian_noise(img):
    noise = np.random.normal(loc=0, scale=1, size=img.shape)
    ### Overlaying noise over the image ###
    # The noise is added to the image by multiplying it with scaling factors of 0.2 and 0.4
    # The resulting noisy images are clipped to the range [0, 1] using `np.clip()`
    noisy = np.clip((img + noise*0.2), 0, 1)
    noisy2 = np.clip((img + noise*0.4), 0, 1)

    ### Multiplying noise by the image
    # The noise is multiplied by the image
    # This operation amplifies the noise while preserving the original image intensity
    noisy2mul = np.clip((img*(1 + noise*0.2)), 0, 1)
    noisy4mul = np.clip((img*(1 + noise*0.4)), 0, 1)

    ### Multiplying noise by the bottom and top half images ###
    # The image is multiplied by 2 to enhance the differences between the darker and brighter regions
    # The noise multiplied by bottom and top half images separately
    # The `np.where()` function is used to handle the differnet ranges of pixel values
    img2 = img*2
    n2 = np.clip(np.where(img2 <= 1, (img2*(1 + noise*0.2)), (1-img2+1)*(1 + noise*0.2)*-1 + 2)/2, 0,1)
    n4 = np.clip(np.where(img2 <= 1, (img2*(1 + noise*0.4)), (1-img2+1)*(1 + noise*0.4)*-1 + 2)/2, 0,1)

    ### Visualizing the images ###
    # noise2 = (noise - noise.min())/(noise.max()-noise.min())
    # plt.figure(figsize=(20,20))
    # plt.imshow(np.vstack((np.hstack((img, noise2)),
    #                     np.hstack((noisy, noisy2)),
    #                     np.hstack((noisy2mul, noisy4mul)),
    #                     np.hstack((n2, n4)))))
    # plt.show()

    ### Visualizing the normal distribution ###
    # plt.hist(noise.ravel(), bins=100)
    # plt.show()

    return n2
def train_image():
    train_img_path = "../training_dataset/train/images/"
    train_img_res_path = "../result/crosswalk/train/"
    train_img_list = os.listdir(train_img_path)
    for img_name in train_img_list:
        img = cv2.imread(train_img_path+img_name)[...,::-1]/255.0
        # [...,::-1] -> convert the BGR image to RGB
        # / 255.0 ->  normalize to the range [0, 1] by dividing by 255.
        res = add_gaussian_noise(img) 
        plt.imshow(res)
        plt.axis('off')
        plt.imsave(train_img_res_path+img_name, res)

def valid_image():
    valid_img_path = "../training_dataset/valid/images/"
    valid_img_res_path = "../result/crosswalk/valid/"
    valid_img_list = os.listdir(valid_img_path)
    for img_name in valid_img_list:
        img = cv2.imread(valid_img_path+img_name)[...,::-1]/255.0
        # [...,::-1] -> convert the BGR image to RGB
        # / 255.0 ->  normalize to the range [0, 1] by dividing by 255.
        res = add_gaussian_noise(img) 
        plt.imshow(res)
        plt.axis('off')
        plt.imsave(valid_img_res_path+img_name, res)

# Call function
# train_image()
valid_image()