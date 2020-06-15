import os
import matplotlib.pyplot as plt
from PIL import Image
import os.path


def convertjpg(jpgfile, outdir, width=224, height=224):
    img = Image.open(jpgfile)
    try:
        new_img = img.resize((width, height), Image.BILINEAR)
        path_join = os.path.join(outdir, os.path.basename(jpgfile))
        print(path_join)
        new_img.save(path_join)
    except Exception as e:
        print(e)


path = './data'
fileList = os.listdir(path)
print(len(fileList))

for d in fileList:
    s = path + '/' + d
    imgs = os.listdir(s)
    print(len(imgs))
    for i in imgs:
        fname = s + '/' + i
        pic = plt.imread(fname)
        print(fname, pic.shape)
        # convertjpg(fname, s)
        # break

# for jpgfile in glob.glob("E:\\img\\*.jpg"):
#     convertjpg(jpgfile,"E:\\lianhua")
