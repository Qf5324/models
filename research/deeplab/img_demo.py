import tensorflow as tf
import numpy as np
import cv2
from keras.preprocessing.image import load_img, img_to_array


def create_visual_anno(anno):
    """"""
    assert np.max(anno) <= 18, "only 18 classes are supported, add new color in label2color_dict"
    label2color_dict = {
        0: [128, 64, 128],
        1: [244, 35, 232],
        2: [70, 70, 70],
        3: [102, 102, 156],
        4: [190, 153, 153],
        5: [153, 153, 153],
        6: [250, 170, 30],
        7: [220, 220, 0],
        8: [107, 142, 35],
        9: [152, 251, 152],
        10: [70, 130, 180],
        11: [220, 20, 60],
        12: [255, 0, 0],
        13: [0, 0, 142],
        14: [0, 0, 70],
        15: [0, 60, 100],
        16: [0, 80, 100],
        17: [0, 0, 230],
        18: [119, 11, 32]
    }
    # visualize
    visual_anno = np.zeros((anno.shape[0], anno.shape[1], 3), dtype=np.uint8)
    for i in range(visual_anno.shape[0]):  # i for h
        for j in range(visual_anno.shape[1]):
            color = label2color_dict[anno[i, j]]
            visual_anno[i, j, 0] = color[0]
            visual_anno[i, j, 1] = color[1]
            visual_anno[i, j, 2] = color[2]

    return visual_anno

# 获取视频

#print('########', video_frame_cnt)

# img_path = '/home/zhou/deeplearning/models-master/research/deeplab/datasets/cityscapes/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png'
img_path = '/home/zhou/test.png'
img = load_img(img_path)  # 输入预测图片的url
tag = img_to_array(img)
h = tag.shape[0]
w = tag.shape[1]

img = img.resize((2048, 1024))
img = img_to_array(img)


img = np.expand_dims(img, axis=0).astype(np.uint8)  # uint8是之前导出模型时定义的

# Mat dst = Mat::zeros(512, 512, CV_8UC3); //我要转化为512*512大小的
# resize(img, dst, dst.size());

# while(vid.isOpened()):
#     ret, img_ori = vid.read()
#
#     if ret == False:
#         break;
#     cv2.imshow('input', img_ori)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# vid.release()

# 加载模型
with tf.Session() as sess:
    with open("exp/eval/frozen_inference_graph.pb", "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())


        output = tf.import_graph_def(graph_def, input_map={"ImageTensor:0": img},
                                        return_elements=["SemanticPredictions:0"])
        # input_map 就是指明 输入是什么；
        # return_elements 就是指明输出是什么；两者在前面已介绍

        result = sess.run(output)
        print(result[0].shape)
        result[0].shape = (result[0].shape[1], result[0].shape[2])

        # pre = result[0].astype(np.uint8)
        pre = create_visual_anno(result[0])
        # print(pre.dtype)  # (1, height, width)

        #
        # cv2.imshow('input', img_ori)

        pre = cv2.resize(pre, (w, h))

        cv2.imshow('output', pre)

        cv2.waitKey(0)

        # if args.save_video:
        #     videoWriter.release()
