# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
#
# Modified by Brad Ezard
# --------------------------------------------------------
from collections import defaultdict, OrderedDict
from multiprocessing import Pool
from pathlib import Path
from typing import *

import xml.etree.ElementTree as ET
import os
import pickle
import numpy as np
from numpy.lib.function_base import average
import torch

if __name__ == '__main__':
    from voc_annotation import _LABELS
else:
    from wsodlib.data.voc_annotation import _LABELS

__all__ = ['run_evaluation', 'parse_rec', 'voc_ap', 'voc_eval', 'get_voc_test_aps']


def run_evaluation(
    dataset_predictions: Dict[str, Dict[str, torch.Tensor]],
    root: str = './data/',
    test_split: bool = True
) -> Dict[str, Any]:
    # Construct the lines for the output files
    lines = defaultdict(list)
    for image_id, element_predictions in dataset_predictions.items():
        scores = element_predictions['scores']
        boxes = element_predictions['boxes'].round().long() + 1
        class_labels = element_predictions['labels']
        for box, score, label in zip(boxes, scores, class_labels):
            class_name = _LABELS[label]
            lines[class_name].append(f'{image_id} {float(score):8.08f} {box[0]} {box[1]} {box[2]} {box[3]}')

    # Start the computations for the average precisions
    savepath = os.path.join(root, 'dets')
    Path(savepath).mkdir(parents=True, exist_ok=True)
    pool = Pool(8)
    result_futures = []
    for klass, out_lines in lines.items():
        with open(os.path.join(savepath, f'{klass}.txt'), 'w') as f:
            f.write('\n'.join(out_lines))
        args = (
            os.path.join(savepath, '{}.txt'),
            os.path.join(root, 'VOCdevkit/VOC2007/Annotations/{}.xml'),
            os.path.join(root, f'VOCdevkit/VOC2007/ImageSets/Main/{"test" if test_split else "val"}.txt'),
            klass,
            savepath,
            0.5,
            True,
            test_split
        )
        result_futures.append((klass, pool.apply_async(voc_eval, args)))

    # Get the results
    aps = {}
    for klass, future in result_futures:
        recall, precision, average_precision = future.get()
        aps[klass] = average_precision
    aps = OrderedDict(sorted(aps.items()))
    aps['mean_average_precision'] = torch.as_tensor(float(sum(aps.values())) / float(len(aps)))  # type: ignore
    return {
        'mean_average_precision': aps['mean_average_precision'],
        'log': aps
    }


@no_type_check
def parse_rec(
    filename: str
) -> List[Dict[str, Any]]:
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                            int(bbox.find('ymin').text),
                            int(bbox.find('xmax').text),
                            int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects


def voc_ap(
    rec: np.ndarray,
    prec: np.ndarray,
    use_07_metric: bool = False
) -> float:
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


@no_type_check
def voc_eval(
    detpath: str,
    annopath: str,
    imagesetfile: str,
    classname: str,
    cachedir: str,
    ovthresh: float = 0.5,
    use_07_metric: bool = False,
    test_split: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots_test.pkl' if test_split else 'annots_val.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    if os.path.isfile(detfile):
        with open(detfile, 'r') as f:
            lines = f.readlines()
    else:
        lines = []
    if len(lines) == 0:
        return np.array([0.,]), np.array([0.,]), 0.

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
    if BB.min() == 0: BB += 1  # expects 1-indexed boxes, because MATLAB...

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :] 
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def get_voc_test_aps() -> Dict[str, float]:   
    aps = {}
    for l in _LABELS:
        _,_,ap = voc_eval('./data/dets/{}.txt',
                './data/VOCdevkit/VOC2007/Annotations/{}.xml',
                f'./data/VOCdevkit/VOC2007/ImageSets/Main/test.txt',
                l,
                './data/dets/',
                0.5,
                True)
        aps[l] = ap
    return aps


if __name__ == '__main__':
    aps = get_voc_test_aps()
    for l, ap in aps.items():
        print(f'{l:24}: {ap:6.04f}')
    print( '--------------------------------')
    print(f'average                 : {sum(aps.values())/len(aps):6.04f}')
    print( '--------------------------------')
