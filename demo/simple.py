from __future__ import print_function
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import hwang
import PIL
import torch
import cv2

dec = hwang.Decoder('/big_fast_drive/orm/trip19/wnum_F97D34AC-3D84-447F-A22C-3D495E7578A8.1514912571.89312.MP4')
config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
#cfg.merge_from_list(["INPUT.TO_BGR255", False])

cd = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.3,
)
# load image and then run prediction

im = cv2.imread('/nvme_drive/orm/frame500.png')
r,p = cd.run_on_opencv_image(im)
cv2.imwrite('/nvme_drive/orm/frame500label.png', r)


images = dec.retrieve(torch.arange(start=0, end=dec.video_index.frames(), step=30).numpy())
print('done retrieving imgs')
preds = []
pos_preds = []
top_preds = []
for (i,image) in enumerate(images):
    cvim = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('/nvme_drive/orm/expout%d.png'%i, cvim)
    r, p = cd.run_on_opencv_image(cvim)
    cv2.imwrite('/nvme_drive/orm/expout%dlabel.png'%i, r)
    # img = cd.transforms(image)
    # predictions = cd.compute_prediction(img)
    if len(p) > 0:
        print('found one')
        pos_preds.append(p)
    preds.append(p)
    top_predictions = cd.select_top_predictions(p)
    top_preds.append(top_predictions)

    # assert False
    # result = image.copy()
    # result = cd.overlay_boxes(result, top_predictions)
    # result = cd.overlay_class_names(result, top_predictions)
    #
    # PIL.Image.fromarray(result).save('/nvme_drive/orm/frame%dout.png' % i)