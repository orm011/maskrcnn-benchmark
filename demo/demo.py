from __future__ import print_function, division
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import tqdm
import cv2
import argparse
import os


parser = argparse.ArgumentParser(description="Run objdet on video")
parser.add_argument(
    "--config-file",
    dest='config_file',
    help="path to config file",
)

parser.add_argument(
    "--output-dir",
    dest='output_dir',
    default=None,
    help="output dir for video and predictions"
)

parser.add_argument(
    "--input-file",
    dest='input_file',
    help="input file path"
)


parser.add_argument(
    "--output-video",
    dest='output_video',
    default=False,
    action='store_true',
    help="whether to output a video with the labels to output dir"
)

parser.add_argument(
    "--file-id",
    dest='file_id',
    default=-1,
    type=int,
    help="assign a file id to the output"
)

parser.add_argument(
    "--stop-at",
    dest='stop_at',
    default=-1,
    type=int,
    help="stop at frame (for testing)"
)

args = parser.parse_args()

config_file = args.config_file
fname = args.input_file

dirname = os.path.dirname(fname)
base = os.path.basename(fname)
assert os.path.exists(fname), 'bad input path'

#'/big_fast_drive/orm/dashcam/F97D34AC-3D84-447F-A22C-3D495E7578A8.1514912571.89312.mov.wnum.mp4'
output_dir = args.output_dir if args.output_dir is not None else dirname

assert os.path.exists(output_dir)
# os.mkdir(output_idr)
#"../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
#cfg.merge_from_list(["MODEL.DEVICE", "cpu"])


vid = cv2.VideoCapture(fname)
fc = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
fcc = int(vid.get(cv2.CAP_PROP_FOURCC))
fps = vid.get(cv2.CAP_PROP_FPS)

coco_demo = COCODemo(
    cfg,
    min_image_size=min(h,w), # keep original size?
    show_mask_heatmaps=False,
    confidence_threshold=0.0)

print('Input {}'.format(fname))
print('Input props:\nframe count: {} w: {} h: {} fourcc: {}'.format(fc, w, h, fcc))
from collections import OrderedDict
import pandas as pd

base = os.path.basename(fname)
obase = base.replace('.wnum.mp4', '.wlabel.mp4')
output_dir = args.output_dir

odata = output_dir + '/' + base.replace('.wnum.mp4', '.parquet')
print('Output predictions at:\n{}'.format(odata))

if args.output_video:
    opath = output_dir + '/' + obase
    print('Output labelled video at:\n{}'.format(opath))
    wr = cv2.VideoWriter(opath, fourcc=fcc, fps=fps, frameSize=(w,h))

dfs = []
if args.stop_at == -1:
    tot = fc
else:
    tot = min(fc, args.stop_at)

import torch

nparr =[]
for frame_num in tqdm.tqdm(range(tot), total=tot):
    r,f = vid.read()
    assert r, 'failed to read?'
    box_logits = coco_demo.run_on_opencv_image(f)
    # has 1000 boxes x 80 class logits.
    frame_maxes = box_logits.softmax(dim=-1).max(dim=1)[0]
    # shape should be max box score per class, so 80x1
    #print(type(predictions), predictions.shape, predictions)

    # if args.output_video:
    #     wr.write(predictions)
    #
    # df = pd.DataFrame(OrderedDict(
    #     [('file_id', args.file_id if args.file_id >= 0 else base),
    #      ('frame_id', frame_num),
    #      ('labels', dat.extra_fields['labels']),
    #      ('scores', dat.extra_fields['scores']),
    #      ('box_x0', dat.bbox[:,0]),
    #      ('box_y0', dat.bbox[:,1]),
    #      ('box_x1', dat.bbox[:,2]),
    #      ('box_y1', dat.bbox[:,3])]
    #     ))
    nparr.append(frame_maxes.cpu())

max_logits = torch.stack(frame_maxes)
torch.save(max_logits, './max_logits.pth')