import os

class ImageRename():
    def __init__(self):
        self.path = os.path.join(os.getcwd(), 'dataset')

    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)

        i = 0

        for item in filelist:
            if item.endswith('.jpg'):
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), format(str(i), '0>4s') + '.jpg')
                os.rename(src, dst)
                print('converting {} to {} ...' .format (src, dst))
                i = i + 1
        print('total {} to rename & converted {} jpgs'.format(total_num, i))

if __name__ == '__main__':
    newname = ImageRename()
    newname.rename()