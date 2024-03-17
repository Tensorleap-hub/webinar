from pathlib import Path
from webinar.utils.gcs_utils import _download


def generate_subset(annotations_file: Path):
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
            list_of_bounding_boxes_int.append([int(x) for x in qq])
        label_data.append(list_of_bounding_boxes_int)
    return image_paths, label_data