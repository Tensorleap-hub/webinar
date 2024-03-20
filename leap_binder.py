import os
from typing import Dict
from PIL import ImageFile

from webinar.data.preprocess import generate_subset
from webinar.utils.gcs_utils import _download

ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
from numpy.typing import NDArray
from code_loader.contract.datasetclasses import PreprocessResponse
from typing import Union
import json
from code_loader import leap_binder
from code_loader.contract.enums import (
    LeapDataType,
)
from code_loader.contract.visualizer_classes import LeapImageWithBBox
from code_loader.contract.responsedataclasses import BoundingBox
from pathlib import Path
import cv2
from webinar.utils.metrics import *

NUM_FEATURES = len(CONFIG['FEATURE_MAPS'])
NUM_PRIORS = len(CONFIG['BOX_SIZES'][0]) * len(CONFIG['BOX_SIZES'])  # [3*3]
car_ind = CONFIG['CATEGORIES'].index("car")
truck_ind = CONFIG['CATEGORIES'].index("truck")
pedestrian_ind = CONFIG['CATEGORIES'].index("pedestrian")


# Preprocess Function
def subset_images_list() -> List[PreprocessResponse]:
    train_files = Path('dataset/anno_data.txt')
    validation_files = Path('dataset/cognata_v2_annotati.txt')

    train_image_paths, train_label_data = generate_subset(train_files)
    train_image_paths, train_label_data = train_image_paths[::5], train_label_data[::5]

    validation_image_paths, validation_label_data = generate_subset(validation_files)
    validation_image_paths, validation_label_data = validation_image_paths[::5], validation_label_data[::5]

    train = PreprocessResponse(length=len(train_image_paths),
                               data={'images': train_image_paths, 'labels': train_label_data})
    validation = PreprocessResponse(length=len(validation_image_paths),
                                    data={'images': validation_image_paths, 'labels': validation_label_data})

    return [train, validation]


def unlabled_subset_images() -> PreprocessResponse:
    path_to_txt_file = "s3_data/From-Algo/OD_partial2/foresight_unlabeled_paths.txt"
    with open(_download(path_to_txt_file), 'r') as f:
        list_of_image_paths = f.readlines()
    prefix = Path("s3_data/From-Algo/OD_partial2")
    list_of_image_paths = [str(prefix / pth.rstrip("\n")) for pth in list_of_image_paths]
    return PreprocessResponse(length=len(list_of_image_paths), data={'images': list_of_image_paths})


# -------------input
def get_image_size(img_path: str) -> Tuple[float, float]:
    with Image.open(img_path) as img:
        width, height = img.size
    return (float(width), float(height))


def load_mapping(fpath: str) -> Dict[int, int]:
    with open(fpath, 'r') as file:
        mapping_json = json.load(file)
    int_dict = {int(key): value for key, value in mapping_json.items()}
    return int_dict


def transform_image_list_to_labels(image_list):
    return [img_pth.replace("celeb_images", "yolo_labels").replace(".jpg", ".txt") for img_pth in image_list]


def get_all_images(pattth, must_have):
    images = []
    for root, dirs, files in os.walk(pattth):
        if must_have not in root:
            # print(f"passing throught: {must_have} is not in path: {root}")
            continue
        for file in files:
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                images.append(os.path.join(root, file))
    return images


def image_path(idx: int, data: PreprocessResponse) -> str:
    """
    Returns image path
    """
    return str(_download(data.data['images'][idx]))


def origin_path(idx: int, data: PreprocessResponse) -> str:
    """
    Returns a string that tells where this sample is from
    """

    if "/cognata/" in data.data['images'][idx]:
        str_out = "cognata_new"
    elif "/bdd100k/" in data.data['images'][idx]:
        str_out = "bdd100k"
    elif "/foresight/" in data.data['images'][idx]:
        str_out = "foresight"
    elif "/Rivian/" in data.data['images'][idx]:
        if "right" in data.data['images'][idx]:
            str_out = "rivian_right"
        else:
            str_out = "rivian_left"
    elif "/Volvo/" in data.data['images'][idx]:
        if "right" in data.data['images'][idx]:
            str_out = "volvo_right"
        else:
            str_out = "Volvo_left"
    elif "/Airport/" in data.data['images'][idx]:
        str_out = "airport"
    elif "/cognata_video_split/" in data.data['images'][idx]:
        str_out = "cognata"
    elif "/cognata_v2_with_gt/" in data.data['images'][idx]:
        str_out = "cognata"
    elif "/foresight_unlabeled_data/" in data.data['images'][idx]:
        str_out = "foresight_unlabeled"
    else:
        str_out = "none"
    return str_out


def input_image(idx: int, data: PreprocessResponse) -> NDArray[float]:
    """
    Returns a RGB image normalized and padded
    """
    imagePath = image_path(idx, data)
    img = cv2.imread(imagePath)[..., ::-1]
    img = cv2.resize(img, CONFIG['IMAGE_SIZE'][::-1]).astype('float32') / 255.0  # RGB!
    return img


# Ground truth encoder fetches the label with the index `idx` from the `labels` array set in
# the PreprocessResponse's data. Returns a numpy array containing a hot vector label correlated with the sample.
def get_bb(idx: int, preprocessing: PreprocessResponse) -> np.ndarray:
    BBOX = np.zeros([CONFIG['MAX_BB_PER_IMAGE'], 5])
    boxes = preprocessing.data['labels'][idx]  # x0, y0, x1,y1, class: [0,1,2]car, truck, pedestrian
    boxes = np.array(boxes).astype(float)
    (width, height) = get_image_size(image_path(idx, preprocessing))
    x0, y0, x1, y1, class_ = boxes[:, 0] / width, boxes[:, 1] / height, boxes[:, 2] / width, boxes[:,
                                                                                             3] / height, boxes[:, 4]
    currlen = len(x0)
    W, H = x1 - x0, y1 - y0
    XM, YM = (x1 + x0) / 2, (y1 + y0) / 2
    BBOX[:currlen, 0] = XM
    BBOX[:currlen, 1] = YM
    BBOX[:currlen, 2] = W
    BBOX[:currlen, 3] = H
    BBOX[:currlen, 4] = class_.astype(float)
    BBOX[currlen:, 4] = BACKGROUND_LABEL
    return BBOX


def bb_array_to_object(bb_array: Union[NDArray[float], tf.Tensor], iscornercoded: bool = True, bg_label: int = 0,
                       is_gt=False) -> List[BoundingBox]:
    """
    Assumes a (X,Y,W,H) Format for the BB text
    bb_array is (CLASSES,TOP_K,PROPERTIES) WHERE PROPERTIES =(conf,xmin,ymin,xmax,ymax)
    """

    bb_list = []
    if not isinstance(bb_array, np.ndarray):
        bb_array = bb_array.numpy()
    # fig, ax = plt.subplots(figsize=(6, 9)
    if len(bb_array.shape) == 3:
        bb_array = bb_array.reshape(-1, bb_array.shape[-1])
    for i in range(bb_array.shape[0]):
        if bb_array[i][-1] != bg_label:
            if iscornercoded:
                x, y, w, h = xyxy_to_xywh_format(bb_array[i][1:5])  # FIXED TOM
                # unormalize to image dimensions
            else:
                x, y = bb_array[i][0], bb_array[i][1]
                w, h = bb_array[i][2], bb_array[i][3]
            conf = 1 if is_gt else bb_array[i][0]
            curr_bb = BoundingBox(x=x, y=y, width=w, height=h, confidence=conf,
                                  label=CONFIG['CATEGORIES'][int(bb_array[i][-1])])
            bb_list.append(curr_bb)
    return bb_list


def gt_decoder(image, ground_truth) -> LeapImageWithBBox:
    bb_object = bb_array_to_object(ground_truth, iscornercoded=False, bg_label=BACKGROUND_LABEL, is_gt=True)
    return LeapImageWithBBox((image * 255).astype(np.float32), bb_object)


def bb_decoder(image, predictions):
    """
    Overlays the BB predictions on the image
    """
    from_logits = True if CONFIG['MODEL_FORMAT'] != "inference" else False
    decoded = False if CONFIG['MODEL_FORMAT'] != "inference" else True
    class_list_reshaped, loc_list_reshaped = reshape_output_list(np.reshape(predictions, (1, *predictions.shape)),
                                                                 decoded=decoded, image_size=CONFIG['IMAGE_SIZE'],
                                                                 feature_maps=CONFIG['FEATURE_MAPS'])
    # add batch
    outputs = DECODER(loc_list_reshaped,
                      class_list_reshaped,
                      DEFAULT_BOXES,
                      from_logits=from_logits,
                      decoded=decoded
                      )
    bb_object = bb_array_to_object(outputs[0], iscornercoded=True, bg_label=BACKGROUND_LABEL)
    return LeapImageWithBBox((image * 255).astype(np.float32), bb_object)


# ---------------------------------------------- #
# ----------- metadata ------------------------- #


def avg_bb_area_metadata(index: int, subset: PreprocessResponse) -> float:
    if "labels" not in subset.data:
        return float(-1)
    bbs = get_bb(index, subset)  # x,y,w,h
    valid_bbs = bbs[bbs[..., -1] != BACKGROUND_LABEL]
    areas = valid_bbs[:, 2] * valid_bbs[:, 3]
    if len(areas) > 0:
        return areas.mean()
    else:
        return float(0)


# add
def max_bb_area_metadata(index: int, subset: PreprocessResponse) -> float:
    if "labels" not in subset.data:
        return float(-1)
    bbs = get_bb(index, subset)  # x,y,w,h
    valid_bbs = bbs[bbs[..., -1] != BACKGROUND_LABEL]
    areas = valid_bbs[:, 2] * valid_bbs[:, 3]
    if len(areas) > 0:
        return float(areas.max())
    else:
        return float(0)


def min_bb_area_metadata(index: int, subset: PreprocessResponse) -> float:
    if "labels" not in subset.data:
        return float(-1)
    bbs = get_bb(index, subset)  # x,y,w,h
    valid_bbs = bbs[bbs[..., -1] != BACKGROUND_LABEL]
    areas = valid_bbs[:, 2] * valid_bbs[:, 3]
    if len(areas) > 0:
        return float(areas.min())
    else:
        return float(0)


def num_bbox_metadata(index: int, subset: PreprocessResponse) -> float:
    if "labels" not in subset.data:
        return float(-1)
    bbs = get_bb(index, subset)  # x,y,w,h
    valid_bbs = bbs[..., -1] != BACKGROUND_LABEL
    return float(valid_bbs.sum())


def num_bbox_car_metadata(index: int, subset: PreprocessResponse) -> float:
    if "labels" not in subset.data:
        return float(-1)
    bbs = get_bb(index, subset)  # x,y,w,h
    valid_bbs = bbs[..., -1] == car_ind
    return float(valid_bbs.sum())


def num_bbox_truck_metadata(index: int, subset: PreprocessResponse) -> float:
    if "labels" not in subset.data:
        return float(-1)
    bbs = get_bb(index, subset)  # x,y,w,h
    valid_bbs = bbs[..., -1] == truck_ind
    return float(valid_bbs.sum())


def num_bbox_pedestrian_metadata(index: int, subset: PreprocessResponse) -> float:
    if "labels" not in subset.data:
        return float(-1)
    bbs = get_bb(index, subset)  # x,y,w,h
    valid_bbs = bbs[..., -1] == pedestrian_ind
    return float(valid_bbs.sum())


def avg_bb_aspect_ratio(index: int, subset: PreprocessResponse) -> float:
    if "labels" not in subset.data:
        return float(-1)
    bbs = get_bb(index, subset)
    valid_bbs = bbs[bbs[..., -1] != BACKGROUND_LABEL]
    assert ((valid_bbs[:, 3] > 0).all())
    aspect_ratios = valid_bbs[:, 2] / valid_bbs[:, 3]
    if len(aspect_ratios) > 0:
        return aspect_ratios.mean()
    else:
        return float(0)


def sample_index(index: int, subset: PreprocessResponse) -> float:
    return float(index)



def metadata_color_brightness_mean(idx: int, preprocess: PreprocessResponse) -> dict:
    image = input_image(idx, preprocess)
    b, g, r = cv2.split(image)
    res = {"red": float(r.mean()), "green": float(g.mean()), "blue": float(b.mean())}

    return res


def metadata_color_brightness_std(idx: int, preprocess: PreprocessResponse) -> dict:
    image = input_image(idx, preprocess)
    b, g, r = cv2.split(image)
    res = {"red": float(r.std()), "green": float(g.std()), "blue": float(b.std())}

    return res


def metadata_contrast(idx: int, preprocess: PreprocessResponse) -> float:
    image = input_image(idx, preprocess)
    img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    df = abs(a - b)

    return float(np.mean(df))


def compute_image_temperature(idx: int, preprocess: PreprocessResponse) -> float:
    image = input_image(idx, preprocess)
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    mean_a = np.mean(lab_image[:, :, 1])
    mean_b = np.mean(lab_image[:, :, 2])
    color_temperature = 2000 + (mean_a + mean_b) / 2

    return float(color_temperature)


def extract_hsv_metadata(idx: int, preprocess: PreprocessResponse) -> Dict[str, float]:
    image = input_image(idx, preprocess)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hue_range = np.ptp(hsv_image[:, :, 0])  #
    saturation_level = np.mean(hsv_image[:, :, 1])

    res = {'hue_range': float(hue_range), 'saturation_level': float(saturation_level)}

    return res


def extract_lab_metadata(idx: int, preprocess: PreprocessResponse) -> Dict[str, float]:
    image = input_image(idx, preprocess)
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lightness_mean = np.mean(lab_image[:, :, 0])
    a_mean = np.mean(lab_image[:, :, 1])
    b_mean = np.mean(lab_image[:, :, 2])
    res = {'lightness_mean': float(lightness_mean), 'a_mean': float(a_mean), 'b_mean': float(b_mean)}

    return res


LABELS = ["x", "y", "w", "h", "object"] + CONFIG['CATEGORIES']
leap_binder.set_preprocess(subset_images_list)
leap_binder.set_unlabeled_data_preprocess(unlabled_subset_images)
leap_binder.set_input(input_image, 'images')
leap_binder.set_ground_truth(get_bb, 'bb')
leap_binder.add_prediction('flattened prediction', LABELS)
leap_binder.add_custom_loss(od_loss, 'od_loss')
leap_binder.set_visualizer(gt_decoder, 'bb_gt_decoder', LeapDataType.ImageWithBBox)
leap_binder.set_visualizer(bb_decoder, 'bb_decoder', LeapDataType.ImageWithBBox)
leap_binder.set_metadata(avg_bb_area_metadata, "BB area")
leap_binder.set_metadata(min_bb_area_metadata, "min BB area")
leap_binder.set_metadata(max_bb_area_metadata, "max BB area")
leap_binder.set_metadata(avg_bb_aspect_ratio, "BB aspect ratio")
leap_binder.set_metadata(num_bbox_metadata, "num objects")
leap_binder.set_metadata(num_bbox_car_metadata, "num cars")
leap_binder.set_metadata(num_bbox_truck_metadata, "num trucks")
leap_binder.set_metadata(num_bbox_pedestrian_metadata, "num pedestrians")
leap_binder.set_metadata(image_path, "image path")
leap_binder.set_metadata(origin_path, "origin")
leap_binder.set_metadata(sample_index, "sample_index")

leap_binder.set_metadata(metadata_color_brightness_mean, "color_brightness_mean")
leap_binder.set_metadata(metadata_color_brightness_std, "color_brightness_std")
leap_binder.set_metadata(metadata_contrast, "image_contrast")
leap_binder.set_metadata(compute_image_temperature, "image_temperature")
leap_binder.set_metadata(extract_hsv_metadata, "hsv")
leap_binder.set_metadata(extract_lab_metadata, "lab")

leap_binder.add_custom_metric(regression_metric, "Regression_metric")
leap_binder.add_custom_metric(classification_metric, "Classification_metric")
leap_binder.add_custom_metric(object_metric, "Objectness_metric")

leap_binder.add_custom_metric(confusion_matrix_metric, "Confusion_metric")

if __name__ == '__main__':
    leap_binder.check()
