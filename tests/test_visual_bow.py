from unittest import TestCase
from visual_bow import *


class TestVisualBoW(TestCase):
    def test_bow(self):
        # 得到数据集
        labeled_images = binary_labeled_img_from_cal101("accordion")

        # 生成图像的 description 全集
        img_descs, _ = gen_sift_features(labeled_images)

        # 分组
        training_idxs, test_idxs, val_idxs = train_test_val_split_idxs(len(labeled_images), 0.3, 0.1)

        # 将用于训练的一组用于图像的 feature 分类
        cluster_features(img_descs, training_idxs)
