import os

class ImageRename():
    def __init__(self):
        self.path = os.path.join(os.getcwd(), '/media/taylor/mhy/mirror/take_photo6/original')

    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)

        i = 1

        for item in filelist:
            if item.endswith('.jpg') or item.endswith('.JPG'):
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), format(str(i)) + '.jpg')
                os.rename(src, dst)
                print('converting {} to {}' .format(src, dst))
                i = i + 1
        print('total {} to rename & converted {} jpgs'.format(total_num, i-1))

if __name__ == '__main__':
    newname = ImageRename()
    newname.rename()