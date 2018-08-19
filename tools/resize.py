import os
import cv2

if __name__ =='__main__':

    input_dir = os.path.join(os.getcwd(), '/media/taylor/Seagate Backup Plus Drive/take_photo4')
    output_dir = os.path.join(os.getcwd(), '/media/taylor/mhy/mirror/take_photo4')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    filelist = os.listdir(input_dir)
    total_num = len(filelist)
    print(total_num)

    for item in filelist:
        image = cv2.imread(os.path.join(input_dir, item))
        h, w, _ = image.shape
        if h > w:    # high
            fixed = cv2.resize(image, (1024, 1280), interpolation=cv2.INTER_CUBIC)
        else:        # wide
            fixed = cv2.resize(image, (1280, 1024), interpolation=cv2.INTER_CUBIC)
        print(item, fixed.shape)
        cv2.imwrite(os.path.join(output_dir, item), fixed)

    print("Resize is Ok!")
