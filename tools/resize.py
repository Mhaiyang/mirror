import os
import cv2

if __name__ =='__main__':

    filelist = os.listdir(os.path.join(os.getcwd(), 'image'))
    total_num = len(filelist)

    for item in filelist:
        image = cv2.imread(os.path.join(os.getcwd(), 'image', item))
        fixed = cv2.resize(image, (300,200), interpolation=cv2.INTER_CUBIC)
        print(fixed.shape)
        cv2.imwrite(os.path.join(os.getcwd(), 'dataset', item), fixed)
        cv2.imshow("image", fixed)

    print("Resize is Ok!")
