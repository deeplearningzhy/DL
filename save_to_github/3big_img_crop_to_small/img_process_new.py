#coding=utf-8
"""python实现图片切割"""
import os
from PIL import Image

def splitimage(canvas_path, nrows, ncols, newimg_path):
    """

    :param canvas_path: 大图存储路径
    :param nrows: 整个大图有多少行
    :param ncols: 整个大图有多少列
    :param newimg_path: 分割出来的小图最后存储的路径
    :return:无返回值
    """
    """
    print img.size  #图片的尺寸(100, 100)
    print img.format  #图片的格式JPG
    print img.mode  #图片的模式(是灰度图L还是彩色图RGBA)
    """

    img = Image.open(canvas_path)#打开大图
    w, h = img.size#获取大图片的尺寸宽和高
    if nrows <= h and ncols <= w:#如果大图的宽和高大于整个大图的行数和列数的话（必须是啊）
        #输出大图的信息：W*H,
        print('Original image information: %sx%s, %s, %s' % (w, h, img.format, img.mode))
        print('开始处理图片切割, 请稍候...')

        # s = os.path.split(canvas_path)#输出路径名s[0]和文件名s[1]
        # # if newimg_path == '':#如果没有这个路径的话
        # newimg_path = s[0] #就使用源路径
        # fn = s[1].split('.')
        # imgname = fn[0]
        # imgformat = fn[-1]

        rowheight = h // nrows
        colwidth = w // ncols
        num = 0
        for r in range(nrows):
            for c in range(ncols):
                #box是一个4元组，(left,upper,right,low)左上、右下
                box = (c * colwidth, r * rowheight, (c + 1) * colwidth, (r + 1) * rowheight)
                #下面这个代码不能分开执行，否则切割的图片有问题
                img.crop(box).save(os.path.join(newimg_path, "class0_" + str(num) + '.jpg' ))
                # img = img.crop(box)
                # img.save(os.path.join(dstpath, basename + "_" + str(num) + '.'+ ext))
                num = num + 1

        print('图片切割完毕，共生成 %s 张小图片。' % num)

    else:
        print('不合法的行列切割参数！')

###################调用图片切割函数##################
canvas_path="D:\\deep_keyan\\homework\\完整版\\sample.jpg"
newing_path="D:\\deep_keyan\\homework\\完整版\\小图\\"
# splitimage(canvas_path=canvas_path,nrows=8,ncols=8,newimg_path=newing_path)#一用默认参数就都要用默认参数
splitimage(canvas_path,8,8,newing_path)
###########################################################


# src = input('请输入图片文件路径：')
# if os.path.isfile(src):#如果path是一个存在的文件，返回True。否则返回False
#     dstpath = input('请输入图片输出目录（不输入路径则表示使用源图片所在目录）：')
#     if (dstpath == '') or os.path.exists(dstpath):
#         row = int(input('请输入切割行数：'))
#         col = int(input('请输入切割列数：'))
#         if row > 0 and col > 0:
#             splitimage(src, row, col, dstpath)
#         else:
#             print('无效的行列切割参数！')
#     else:
#         print('图片输出目录 %s 不存在！' % dstpath)
# else:
#     print('图片文件 %s 不存在！' % src)