from leap_binder import *


def check_custom_test():
    LABELS = ["x", "y", "w", "h", "object"] + CONFIG['CATEGORIES']
    dir_path = os.path.dirname(os.path.abspath(__file__))
    model_path = 'model/yolov7-webinar.h5'
    model = tf.keras.models.load_model(os.path.join(dir_path, model_path))

    subset = subset_images_list()
    train = subset[0]
    val = subset[1]
    # val.data['images'] = #ddlete the first image

    set = train
    unlabeled_set = unlabled_subset_images()
    for idx in range(set.length):
        # get input and gt
        try:
            input = input_image(idx, set)
            concat = np.expand_dims(input, axis=0)
            gt = get_bb(idx, set)
            y_true = tf.convert_to_tensor(np.expand_dims(gt, axis=0))

            # model
            y_pred = model([concat])

            # get loss and metrices
            od_loss_ = od_loss(y_true, y_pred)
            regression_metric_ = regression_metric(y_true, y_pred)
            classification_metric_ = classification_metric(y_true, y_pred)
            object_metric_ = object_metric(y_true, y_pred)
            # confusion_matrix_metric_ = confusion_matrix_metric(y_true, y_pred)

            # get metadata
            avg_bb_area = avg_bb_area_metadata(idx, set)
            min_bb_area = min_bb_area_metadata(idx, set)
            max_bb_area = max_bb_area_metadata(idx, set)
            avg_bb_aspect = avg_bb_aspect_ratio(idx, set)
            num_bbox = num_bbox_metadata(idx, set)
            num_bbox_car = num_bbox_car_metadata(idx, set)
            num_bbox_truck = num_bbox_truck_metadata(idx, set)
            num_bbox_pedestrian = num_bbox_pedestrian_metadata(idx, set)
            image_path_ = image_path(idx, set)
            origin_path_ = origin_path(idx, set)
            sample_index_ = sample_index(idx, set)
            metadata_color_brightness_mean_ = metadata_color_brightness_mean(idx, set)
            metadata_color_brightness_std_ = metadata_color_brightness_std(idx, set)
            metadata_contrast_ = metadata_contrast(idx, set)
            compute_image_temperature_ = compute_image_temperature(idx, set)
            extract_hsv_metadata_ = extract_hsv_metadata(idx, set)
            extract_lab_metadata_ = extract_lab_metadata(idx, set)

            # get visualizers
        except:
            print(f"index {idx} has a problem")


if __name__ == '__main__':
    check_custom_test()
