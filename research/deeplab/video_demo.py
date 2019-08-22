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
vid = cv2.VideoCapture("/home/zhou/test.mp4")
# vid = cv2.VideoCapture(0)
video_frame_cnt = int(vid.get(7))
video_width = int(vid.get(3))
video_height = int(vid.get(4))
video_fps = int(vid.get(5))
#print('########', video_frame_cnt)

# img_path = '/home/zhou/dog.jpg'
# img = load_img(img_path)  # 输入预测图片的url
# img = img_to_array(img)
# img = np.expand_dims(img, axis=0).astype(np.uint8)  # uint8是之前导出模型时定义的

# while(vid.isOpened()):
#     ret, img_ori = vid.read()
#
#     if ret == False:
#         break;
#     cv2.imshow('input', img_ori)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# vid.release()
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
videoWriter = cv2.VideoWriter('video_result.mp4', fourcc, video_fps, (video_width, video_height))

graph = tf.Graph()
INPUT_TENSOR_NAME = 'ImageTensor:0'
OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
graph_def = None
graph_path = "exp/eval/frozen_inference_graph.pb"
with tf.gfile.FastGFile(graph_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

if graph_def is None:
    raise RuntimeError('Cannot find inference graph in tar archive.')

with graph.as_default():
    tf.import_graph_def(graph_def, name='')

sess = tf.Session(graph=graph)


for i in range(video_frame_cnt):
    ret, img_ori = vid.read()

    if ret == False:
        break
    #img = img_ori.resize((2048, 1024))
    img = cv2.resize(img_ori, (2048, 1024))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0).astype(np.uint8)  # uint8是之前导出模型时定义的
    result = sess.run(
        OUTPUT_TENSOR_NAME,
        feed_dict={INPUT_TENSOR_NAME: img})
    print(result.shape)
    result.shape = (result.shape[1], result.shape[2])

    pre = result.astype(np.uint8)
    pre = create_visual_anno(pre)
    # print(pre.dtype)  # (1, height, width)

    #
    # cv2.imshow('input', img_ori)
    pre = cv2.resize(pre, (video_width, video_height))
    cv2.imshow('output', pre)
    videoWriter.write(pre)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # cv2.waitKey(0)
vid.release()
videoWriter.release()
