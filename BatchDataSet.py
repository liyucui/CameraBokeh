
import os
import numpy as np
import cv2
import imageio

# data_dir = 'train'
class BatchDataSet:
    train_img_list = []
    train_annotation_list = []
    val_img_list = []
    val_annotation_list = []
    batch_offset = 0
    epochs_completed = 0
    train_index = []

    def __init__(self, data_dir,):
        for _, _, files in os.walk(data_dir+'/val/image'):
            for name in files:
                self.val_img_list.append(data_dir+'/val/image/'+name)
                self.val_annotation_list.append(data_dir + '/val/mask/' + name)
        for _, _, files in os.walk(data_dir+'/train/image'):
            self.train_index = np.arange(0, len(files) + 1)
            for name in files:
                self.train_img_list.append(data_dir+'/train/image/'+name)
                self.train_annotation_list.append(data_dir+'/train/mask/'+name)

    def train_next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        # a = len(self.train_img_list)
        if self.batch_offset > len(self.train_img_list):
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(len(self.train_img_list))
            np.random.shuffle(perm)
            self.train_index = np.array(self.train_index)[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size
        end = self.batch_offset

        imgdata = []
        matdata = []
        for n in self.train_index[start:end]:
            try:
                imgdata.append(imageio.imread(self.train_img_list[n]))
                matdata.append(imageio.imread(self.train_annotation_list[n]))
            except ValueError:
                if len(np.shape(imgdata)) > len(np.shape(matdata)):
                    imgdata.pop()
                    print('FileNotFoundError: %s' % self.train_img_list[n])
                elif len(np.shape(imgdata)) > len(np.shape(matdata)):
                    matdata.pop()
                    print('FileNotFoundError: %s' % self.train_annotation_list[n])
                else:
                    print('FileNotFoundError: %s, %s' % self.train_img_list[n], self.train_annotation_list[n])
                end += 1
                self.batch_offset += 1
                if self.batch_offset > len(self.train_img_list):
                    self.epochs_completed += 1
                    print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
                    # Shuffle the data
                    perm = np.arange(len(self.train_img_list))
                    np.random.shuffle(perm)
                    self.train_index = np.array(self.train_index)[perm]
                    # Start next epoch
                    start = 0
                    self.batch_offset = batch_size


        imgdata = np.array(imgdata, dtype=np.float32)
        matdata = np.round(np.array(matdata, dtype=np.float32)/255.0)
        matdata = np.expand_dims(np.array(matdata, dtype=np.float32), axis=3)
        return imgdata, matdata

    def val_random_batch(self, batch_size):
        indexes = np.random.randint(0, len(self.val_img_list), size=[batch_size]).tolist()

        imgdata = []
        matdata = []
        for n in indexes:
            # imgdata.append(imageio.imread(self.val_img_list[n]))
            # matdata.append(imageio.imread(self.val_annotation_list[n]))
            try:
                imgdata.append(imageio.imread(self.val_img_list[n]))
                matdata.append(imageio.imread(self.val_annotation_list[n]))
            except ValueError:
                if len(np.shape(imgdata)) > len(np.shape(matdata)):
                    imgdata.pop()
                    print('FileNotFoundError: %s' % self.val_img_list[n])
                elif len(np.shape(imgdata)) > len(np.shape(matdata)):
                    matdata.pop()
                    print('FileNotFoundError: %s' % self.val_annotation_list[n])
                else:
                    print('FileNotFoundError: %s, %s' % self.val_img_list[n], self.val_annotation_list[n])

        imgdata = np.array(imgdata, dtype=np.float32)
        matdata = np.round(np.array(matdata, dtype=np.float32)/255.0)
        matdata = np.expand_dims(np.array(matdata, dtype=np.int32), axis=3)
        return imgdata, matdata

    def val_image_batch(self, batch_size):
        indexes = np.random.randint(0, len(self.val_img_list), size=[batch_size]).tolist()

        imgdata = []
        matdata = []
        for n in indexes:
            imgdata.append(imageio.imread(self.val_img_list[n]))
            # matdata.append(imageio.imread(self.val_annotation_list[n]))
        imgdata = np.array(imgdata, dtype=np.float32)
        # matdata = np.round(np.array(matdata, dtype=np.float32)/255.0)
        # matdata = np.expand_dims(np.array(matdata, dtype=np.int32), axis=3)
        return imgdata

    def val_all_batch(self):
        imgdata = []
        matdata = []
        for i, name in enumerate(self.val_img_list):
            imgdata.append(imageio.imread(name))
            matdata.append(imageio.imread(self.val_annotation_list[i]))
        imgdata = np.array(imgdata, dtype=np.float32)
        matdata = np.round(np.array(matdata, dtype=np.float32)/255.0)
        matdata = np.expand_dims(np.array(matdata, dtype=np.int32), axis=3)
        return imgdata, matdata








