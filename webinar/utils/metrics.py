import tensorflow as tf
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_custom_metric, tensorleap_custom_loss

from webinar.config import CONFIG
from typing import List, Tuple, Any
from code_loader.helpers.detection.yolo.loss import YoloLoss
from code_loader.helpers.detection.yolo.grid import Grid
from code_loader.helpers.detection.yolo.utils import reshape_output_list
import numpy as np
from code_loader.helpers.detection.utils import true_coords_labels
from code_loader.contract.datasetclasses import ConfusionMatrixElement
from code_loader.contract.enums import (
    ConfusionMatrixValue, MetricDirection
)
from code_loader.helpers.detection.utils import xywh_to_xyxy_format, xyxy_to_xywh_format, jaccard
from code_loader.helpers.detection.yolo.decoder import Decoder


CLASSES = len(CONFIG['CATEGORIES'])
BACKGROUND_LABEL = CLASSES + 1

DECODER = Decoder(CLASSES,
                  background_label=BACKGROUND_LABEL,
                  top_k=20,
                  conf_thresh=CONFIG['CONF_THRESH'],
                  nms_thresh=CONFIG['NMS_THRESH'],
                  max_bb_per_layer=CONFIG['MAX_BB_PER_IMAGE'],
                  max_bb=CONFIG['MAX_BB_PER_IMAGE'])

BOXES_GENERATOR = Grid(image_size=CONFIG['IMAGE_SIZE'], feature_maps=CONFIG['FEATURE_MAPS'],
                       box_sizes=CONFIG['BOX_SIZES'],
                       strides=CONFIG['STRIDES'], offset=CONFIG['OFFSET'])

DEFAULT_BOXES = BOXES_GENERATOR.generate_anchors()
# LOSS_FN = YoloLoss(num_classes=CLASSES, overlap_thresh=OVERLAP_THRESH,
#                                 default_boxes=DEFAULT_BOXES, background_label=BACKGROUND_LABEL,
#                                 from_logits=False , weights=[4.0, 1.0, 0.4], max_match_per_gt=10)

LOSS_FN = YoloLoss(num_classes=CLASSES, overlap_thresh=CONFIG['OVERLAP_THRESH'],
                   features=CONFIG['FEATURE_MAPS'], anchors=np.array(CONFIG['BOX_SIZES']),
                   default_boxes=DEFAULT_BOXES, background_label=BACKGROUND_LABEL,
                   from_logits=False if CONFIG['MODEL_FORMAT'] == "inference" else True,
                   image_size=CONFIG['IMAGE_SIZE'], yolo_match=True)


def compute_losses(y_true: tf.Tensor, y_pred: tf.Tensor) -> Tuple[Any, Any, Any]:
    """
    Computes the sum of the classification (CE loss) and localization (regression) losses from all heads
    """
    decoded = False if CONFIG['MODEL_FORMAT'] != "inference" else True
    class_list_reshaped, loc_list_reshaped = reshape_output_list(y_pred, decoded=decoded,
                                                                 image_size=CONFIG['IMAGE_SIZE'],
                                                                 feature_maps=CONFIG['FEATURE_MAPS'])  # add batch
    loss_l, loss_c, loss_o = LOSS_FN(y_true=y_true, y_pred=(loc_list_reshaped, class_list_reshaped))
    return loss_l, loss_c, loss_o

@tensorleap_custom_loss('od_loss')
def od_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:  # return batch
    """
    Sums the classification and regression loss
    """
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    loss_l, loss_c, loss_o = compute_losses(y_true, y_pred)
    combined_losses = [l + c + o for l, c, o in zip(loss_l, loss_c, loss_o)]
    sum_loss = tf.reduce_sum(combined_losses, axis=0)
    return sum_loss.numpy().squeeze(0)


# -------------- metrics ---------------- #
@tensorleap_custom_metric('Classification_metric',direction=MetricDirection.Downward)
def classification_metric(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray: # return batch
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    _, loss_c, _ = compute_losses(y_true, y_pred)
    return tf.squeeze(tf.reduce_sum(loss_c, axis=0), axis=-1).numpy()

@tensorleap_custom_metric('Regression_metric',direction=MetricDirection.Downward)
def regression_metric(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:  # return batch
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    loss_l, _, _ = compute_losses(y_true, y_pred)
    return tf.squeeze(tf.reduce_sum(loss_l, axis=0), axis=-1).numpy()

@tensorleap_custom_metric('Objectness_metric',direction=MetricDirection.Downward)
def object_metric(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    _, _, loss_o = compute_losses(y_true, y_pred)
    return tf.squeeze(tf.reduce_sum(loss_o, axis=0), axis=-1).numpy()

@tensorleap_custom_metric('Confusion_metric',direction=MetricDirection.Downward)
def confusion_matrix_metric(y_true, y_pred):
    decoded = False if CONFIG['MODEL_FORMAT'] != "inference" else True
    from_logits = True if CONFIG['MODEL_FORMAT'] != "inference" else False
    threshold = 0.5
    class_list_reshaped, loc_list_reshaped = reshape_output_list(
        y_pred, decoded=decoded, image_size=CONFIG['IMAGE_SIZE'])  # add batch
    outputs = DECODER(loc_list_reshaped,
                      class_list_reshaped,
                      DEFAULT_BOXES,
                      from_logits=from_logits,
                      decoded=decoded
                      )

    ret = []
    for batch_i in range(len(outputs)):
        gt_loc, gt_class = true_coords_labels(batch_i, y_true, BACKGROUND_LABEL)
        gt_detected = np.zeros_like(gt_class, dtype=bool)
        confusion_matrix_elements = []
        if len(outputs[batch_i]) != 0:
            ious = jaccard(outputs[batch_i][:, 1:5],
                           xywh_to_xyxy_format(tf.cast(gt_loc, tf.double))).numpy()  # (#bb_predicted,#gt)
            prediction_detected = np.any((ious > threshold), axis=1)
            if len(gt_loc) != 0:
                max_iou_ind = np.argmax(ious, axis=1)
                gt_indices = [int(gt_class[max_iou_ind[i]]) for i in range(len(prediction_detected))]
            else:  # no gt
                max_iou_ind = np.zeros(prediction_detected.shape[0])
                gt_indices = [0] * len(prediction_detected)
            for i, prediction in enumerate(prediction_detected):
                gt_idx = int(gt_indices[i])
                pred_idx = int(outputs[batch_i][i, -1])
                same_class = gt_idx == pred_idx
                confidence = outputs[batch_i][i, 0]
                pred_class = CONFIG['CATEGORIES'][int(outputs[batch_i][i, -1])]
                if prediction and same_class:  # TP -> FN
                    gt_detected[max_iou_ind[i]] = True
                    confusion_matrix_elements.append(ConfusionMatrixElement(
                        str(pred_class),
                        ConfusionMatrixValue.Positive,
                        float(confidence)
                    ))
                else:  # FP -> TN
                    confusion_matrix_elements.append(ConfusionMatrixElement(
                        str(pred_class),
                        ConfusionMatrixValue.Negative,
                        float(confidence)
                    ))
        for k, gt_detection in enumerate(gt_detected):
            if not gt_detection:
                confusion_matrix_elements.append(ConfusionMatrixElement(  # FN
                    str(CONFIG['CATEGORIES'][int(gt_class[k])]),
                    ConfusionMatrixValue.Positive,
                    float(0)
                ))
        ret.append(confusion_matrix_elements)
    return ret
