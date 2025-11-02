from leap_binder import *
from code_loader.plot_functions.visualize import visualize
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_load_model, tensorleap_integration_test
from code_loader.contract.datasetclasses import PredictionTypeHandler

LABELS = ["x", "y", "w", "h", "object"] + CONFIG['CATEGORIES']
prediction_type1 = PredictionTypeHandler('flattened prediction', LABELS)

@tensorleap_load_model([prediction_type1])
def load_model():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    model_path = 'model/yolov7-webinar.h5'
    model = tf.keras.models.load_model(os.path.join(dir_path, model_path))
    return model

@tensorleap_integration_test()
def check_custom_test(idx, subset):
    plot_vis = True
    model = load_model()
    img = input_image(idx, subset)
    # model
    y_pred = model([img])

    # get loss and metrics
    if subset is None or subset.state != DataStateType.unlabeled:
        gt = get_bb(idx, subset)
        od_loss_ = od_loss(gt, y_pred)
        regression_metric_ = regression_metric(gt, y_pred)
        classification_metric_ = classification_metric(gt, y_pred)
        object_metric_ = object_metric(gt, y_pred)

        # get visualizers
        gt_decoder_ = gt_decoder(img, gt)
    bb_decoder_ = bb_decoder(img, y_pred)

    if plot_vis:
        if subset is None or subset.state != DataStateType.unlabeled:
            visualize(gt_decoder_)
        visualize(bb_decoder_)

    # get metadata
    bb_metadata = bbs_metadata(idx, subset)
    image_path_ = image_path(idx, subset)
    origin_path_ = origin_path(idx, subset)
    img_metadata = image_metadata(idx, subset)
    print(bb_metadata)
    print(img_metadata)
    print(image_path_)
    print(origin_path_)

if __name__ == '__main__':
    sets = subset_images_list()
    for curr_set in sets:
        for idx in range(3):
            check_custom_test(idx, curr_set)
