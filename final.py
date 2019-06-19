from tkinter import *
from tkinter import filedialog
from object_detection.utils import label_map_util
import re
import os
import glob
from PIL import Image
import pytesseract

root = Tk()
root.geometry('1000x1000')
defaultImage = PhotoImage(file="img.png")

num_steps = 100
MODELS_CONFIG = {
    'ssd_mobilenet_v2': {
        'model_name': 'ssd_mobilenet_v2_coco_2018_03_29',
        'pipeline_file': 'ssd_mobilenet_v2_coco.config',
        'batch_size': 12
    }
}

selected_model = 'ssd_mobilenet_v2'
MODEL = MODELS_CONFIG[selected_model]['model_name']
pipeline_file = MODELS_CONFIG[selected_model]['pipeline_file']
batch_size = MODELS_CONFIG[selected_model]['batch_size']

test_record_fname = 'annotations/test.record'
train_record_fname = 'annotations/train.record'
label_map_pbtxt_fname = 'annotations/label_map.pbtxt'
repo_dir_path = 'Dataset'
fine_tune_checkpoint = 'Files/pretrained_model/model.ckpt'
pipeline_fname = 'ssd_mobilenet_v2_coco.config'
pb_fname = 'frozen_inference_graph.pb'

def get_num_classes(pbtxt_fname):
    label_map = label_map_util.load_labelmap(pbtxt_fname)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return len(category_index.keys())

num_classes = get_num_classes(label_map_pbtxt_fname)

PATH_TO_CKPT = pb_fname
PATH_TO_LABELS = label_map_pbtxt_fname
PATH_TO_TEST_IMAGES_DIR = os.path.join(repo_dir_path, "Try")

'os.path.join(repo_dir_path, "Try")'

assert os.path.isfile(pb_fname)
assert os.path.isfile(PATH_TO_LABELS)
TEST_IMAGE_PATHS = glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, "*.*"))
assert len(TEST_IMAGE_PATHS) > 0, 'No image found in `{}`.'.format(PATH_TO_TEST_IMAGES_DIR)


def getPlate():
    import numpy as np
    import sys
    import tensorflow as tf
    from matplotlib import pyplot as plt
    from PIL import Image

    # This is needed since the notebook is stored in the object_detection folder.
    sys.path.append("..")
    from object_detection.utils import ops as utils_ops
    from object_detection.utils import label_map_util

    from object_detection.utils import visualization_utils as vis_util

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    # Size, in inches, of the output images.
    IMAGE_SIZE = (12, 8)

    def run_inference_for_single_image(image, graph):
        with graph.as_default():
            with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {
                    output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(
                        tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(
                        tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(
                        tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [
                        real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                        real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict,
                                       feed_dict={image_tensor: np.expand_dims(image, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(
                    output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict

    for image_path in TEST_IMAGE_PATHS:
        image = Image.open(image_path)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        output_dict = run_inference_for_single_image(image_np, detection_graph)
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)
        plt.figure(figsize=IMAGE_SIZE)
        plt.imshow(image_np)
        plt.savefig('Output/detected.png')

def detectPlate():
    getPlate()
    imgPath = 'Output/detected.png'
    x = defaultImage.configure(file=imgPath)

def applyOCR():
    file = 'Dataset/Use/1.jpg'
    text1 = pytesseract.image_to_string(Image.open(file))
    ocrLabel.configure(text=text1)
    print(text1)

topFrame = Frame(root)
topFrame.pack()

bottomFrame = Frame(root)
bottomFrame.pack()

quitFrame = Frame(root)
quitFrame.pack()

# Top frame widgets
uploadImgLabel = Label(topFrame, text="Click to Detect NumberPlate", font=("Arial", 16))
uploadImgLabel.pack()
uploadImgButton = Button(topFrame, text="Detect", height=2, width=40, command=detectPlate)
uploadImgButton.pack()
ocrButton = Button(topFrame, text="Apply OCR", height=2, width=40, command=applyOCR)
ocrButton.pack()

# Bottom frame widgets, example: displaying images as labels
photoLabel = Label(bottomFrame, image=defaultImage, height=500, width=550)
photoLabel.pack()

ocrLabel = Label(bottomFrame,text="Extracted Text from OCR is displayed here")
ocrLabel.pack()

label = Label(bottomFrame,text="  ")
label.pack()

label1 = Label(bottomFrame,text="  ")
label1.pack()
label2 = Label(bottomFrame,text="  ")
label2.pack()
label3 = Label(bottomFrame,text="  ")
label3.pack()
# Quit frame widgets
quitButton = Button(quitFrame, text="Quit", command=root.quit, height=2, width =20)
quitButton.grid(row=0, column=1)

root.mainloop()

