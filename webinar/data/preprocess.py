from pathlib import Path
from webinar.utils.s3_utils import _download


def generate_subset(annotations_file: Path, xywh: bool = False):
    images_base_path = Path("s3_data/From-Algo/OD_partial2")

    anno_file = str(_download(str(annotations_file)))
    with open(anno_file, 'r') as f:
        lines = f.readlines()

    image_paths = []
    label_data = []
    for line in lines:
        splitted_line = line.split()
        if "bdd100k" in splitted_line[0]:
            continue
        image_paths.append(str(images_base_path / splitted_line[0]))
        list_of_bounding_boxes = [word.split(',') for word in splitted_line[1:]]
        list_of_bounding_boxes_int = list()
        for qq in list_of_bounding_boxes:
            box = [int(x) for x in qq]
            if xywh:  # convert x,y,w,h -> x0,y0,x1,y1
                box = [box[0], box[1], box[0] + box[2], box[1] + box[3], box[4]]
            list_of_bounding_boxes_int.append(box)
        label_data.append(list_of_bounding_boxes_int)
    return image_paths, label_data