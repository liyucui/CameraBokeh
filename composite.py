from skimage import io, color, transform

import numpy as np
import os
import random
import cv2
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'

# bk_dir = '/home/moonsin/gxy/data/用于合成新数据集的背景/背景'
bk_dir = 'bk_data/background'
fr_dir = 'to_add_background/image_resize'
frm_dir = 'to_add_background/mask_resize'
com_dir_image = 'dataset-horizontal/com_data/image'
com_dir_mask = 'dataset-horizontal/com_data/mask'

if os.path.exists(com_dir_image) is False:
    os.makedirs(com_dir_image)
if os.path.exists(com_dir_mask) is False:
    os.makedirs(com_dir_mask)

prefix = ''
resize_size = (192, 144)
# original_size = (600,450)
bk_dirs = os.listdir(bk_dir)
bk_files = []
for dir in bk_dirs:
    path = os.path.join(bk_dir, dir)
    tmp = os.listdir(path)
    for name in tmp:
        file = os.path.join(path, name)
        bk_files.append(file) # bk_files保存了图像路径

# bk_names = os.listdir(bk_dir)
# bk_files = []
# for bk_name in bk_names:
#     bk_files.append(os.path.join(bk_dir, bk_name))

random.shuffle(bk_files)
fr_files = os.listdir(fr_dir)
all_count = 0
bk_count = 0
bk_num = len(bk_files)
print(bk_num)
fr_count = 1
fr_num = len(fr_files)
bg_count_per_fg = 15
for fr_file in fr_files:
    fr_path = os.path.join(fr_dir, fr_file)
    frm_path = os.path.join(frm_dir, fr_file)
    print('foreground: ', fr_count, '/', fr_num)
    fr_count += 1
    for c in range(bg_count_per_fg):
        bk_file = bk_files[(bk_count+c)%bk_num]
        print('background: ', bk_file)
        try:
            # read images and resize the images
            src = cv2.imread(bk_file, 1)
            tar = cv2.imread(fr_path, 1)
            tarMask = cv2.imread(frm_path, 0)
        except:
            continue

        # 在此处修改尺寸
        tar_shape = np.shape(tar)
        tarm_shape = np.shape(tarMask)
        src_shape = np.shape(src)
        if len(tar_shape) !=3 or len(tarm_shape) !=2 or len(src_shape) != 3: continue

        tarRow = tarMask.shape[0] #前景宽高https://wx.qq.com/cgi-bin/mmwebwx-bin/webwxgetmsgimg?&MsgID=6596402574502186661&skey=%40crypt_8a54a8ec_497c27324811b9d6b963ce524bf9076f
        tarCol = tarMask.shape[1]
        srcRow = src.shape[0] # 背景宽高
        srcCol = src.shape[1]

        # if tarRow < srcRow*2/3: # 如果前景过小
        # temp = random.uniform(1.2, 1.5)
        # scale1 = temp*tarRow/srcRow
        # scale2 = temp*tarCol/srcCol
        # if scale1 > scale2:
        #     scale = scale1
        # else:
        #     scale = scale2
        src = cv2.resize(src, (tarCol, tarRow), interpolation=cv2.INTER_AREA) # 重采样背景
        dst = src


        # # composite
        fr_row = tar.shape[0]
        fr_col = tar.shape[1]
        # dstRow = dst.shape[0]
        # dstCol = dst.shape[1]
        start = 0
        count = 1

        for i in range(fr_row):
            if np.sum(tarMask[i, :]) <= 10:
                continue
            else:
                for j in range(fr_col):
                    for s in range(count):
                        if tarMask[i, j] > 10:
                            rate = tarMask[i, j] / 255.0
                            dst[i, j+s*fr_col, :] = rate * tar[i, j, :] +\
                                                          (1 - rate) * dst[i, j+s*fr_col, :]


        # if c == 0:  # save original image
        #     print('c=0')
        #     tmp_index = all_count + 1
            # tmp_path0 = os.path.join(org_dir_image, tmp_index + '.jpg')
            # print(tmp_path0)
            # print(tar)
            # cv2.imwrite(org_dir_image + '/original_%d.jpg' %tmp_index, tar)
            # tmp_path1 = os.path.join(com_dir_mask, tmp_index + '.jpg')
            # cv2.imwrite(org_dir_mask + '/original_%d.jpg' %tmp_index, tarMask)
            # all_count += 1

        for i in range(count):
            tmp = dst[start:(start+fr_row), (i*fr_col):((i+1)*fr_col), :]
            tmp_index = all_count + 1
            # tmp_path = os.path.join(com_dir_image, tmp_index + '.jpg')
            # print(tmp_path)
            tmp = cv2.resize(tmp, resize_size, interpolation=cv2.INTER_AREA)
            tarMask = cv2.resize(tarMask, resize_size, interpolation=cv2.INTER_AREA)
            cv2.imwrite(com_dir_image + '/toaddbk_composite_%d.jpg' %tmp_index, tmp)
            # tmp_path2 = os.path.join(com_dir_mask, tmp_index + '.jpg')
            cv2.imwrite(com_dir_mask + '/toaddbk_composite_%d.jpg' %tmp_index, tarMask)
            all_count += 1

    bk_count += bg_count_per_fg
    print('\n')

