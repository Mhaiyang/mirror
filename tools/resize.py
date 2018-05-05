import os
import cv2

if __name__ =='__main__':

    filelist = os.listdir(os.path.join(os.getcwd(), 'data'))
    total_num = len(filelist)
    print(total_num)

    for item in filelist:
        image = cv2.imread(os.path.join(os.getcwd(), 'data', item))
        h, w, _ = image.shape
        if h > w:    # high
            fixed = cv2.resize(image, (200, 300), interpolation=cv2.INTER_CUBIC)
        else:        # wide
            fixed = cv2.resize(image, (300, 200), interpolation=cv2.INTER_CUBIC)
        print(item, fixed.shape)
        cv2.imwrite(os.path.join(os.getcwd(), 'image', item), fixed)

    print("Resize is Ok!")
