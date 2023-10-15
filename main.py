

import cv2
import tensorflow as tf

import tensorflow_hub as hub

# This file from TensorFlow facilitates the reading of the ".pbtxt" files that can contain class names for labels
import label_map_util

from Training_Class import ModelTraining

def calculate_bounding_area(x1, y1, x2, y2):
    return (x2 - x1) * (y2 - y1)

def determine_if_detection_is_bigger_then(inference_results, class_name_to_filter_by, frame_area, size_threshold=0.3):
    for record in inference_results:
        class_name, bounding_box_coordinates = record

        if class_name == class_name_to_filter_by:

            if calculate_bounding_area(*bounding_box_coordinates) > frame_area * size_threshold:
                # If the label of the current bounding box matches the specified label, and it is bigger than size_threshold times the size of the frame, return True.
                return True, bounding_box_coordinates

        # If the class does not match the specified class name, continue early.
        else:
            continue

    return False, None


def crop_image_to_bounding_box(img, x1, y1, x2, y2, offset=((0, 0), (0, 0))):
    return img[x1+offset[1][0]:x2+offset[1][1], y1+offset[0][0]:y2+offset[0][1], :]


class ResNet50:
    LABEL_PATH = r"mscoco_label_map.pbtxt"
    MODEL_PATH = r"https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512/1"
    min_score_threshold = 0.5


    def __init__(self):
        self.detector = hub.load(self.MODEL_PATH)

        # Determine the label names
        self.category_index = label_map_util.create_category_index_from_labelmap(self.LABEL_PATH)

    def _basic_inference(self, img):
        tensor = tf.convert_to_tensor(img)
        tensor = tf.expand_dims(img, axis=0)

        results = self.detector(tensor)
        result = {key: value.numpy() for key, value in results.items()}

        return result

    def inference(self, img):

        result = self._basic_inference(img)

        boxes = result['detection_boxes']
        classes = result["detection_classes"][0]

        classes = [self.category_index[result]['name'] for result in classes]

        detected_class_scores = result["detection_scores"][0]

        w, h = img.shape[0], img.shape[1]

        # For boxes [0] to get into empty dimension, [i] for current dimension, and the dimension that corresponds to the current coordinate of the box.
        return [(result, [int(boxes[0][i][0] * w), int(boxes[0][i][1] * h), int(boxes[0][i][2] * w), int(boxes[0][i][3] * h)]) for i, result in enumerate(classes) if detected_class_scores[i] > self.min_score_threshold]

    def inference_and_return_annotated_image(self, img):

        result = self._basic_inference(img)
        boxes = result['detection_boxes']
        classes = result['detection_classes']
        scores = result['detection_scores']


        for i in range(len(boxes[0])):
            # TODO: Figure out boxes!

            if not (scores[0][i] > self.min_score_threshold):
                continue

            current_box = boxes[0][i]
            w, h = img.shape[0], img.shape[1]
            current_box = [int(current_box[0] * w), int(current_box[1] * h), int(current_box[2] * w), int(current_box[3] * h)]

            # Get Labels
            class_name = self.category_index[classes[0][i]]['name']

            # image = np.array(image)

            cv2.rectangle(img, (current_box[0], current_box[1]), (current_box[2], current_box[3]), color=(255, 0, 0), thickness=2)

            other_options = (cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.putText(img, class_name, (current_box[0], current_box[1] - 10), *other_options)

        return img


if __name__ == "__main__":

    # Stage 1, Person Detection Setup
    resnet50 = ResNet50()

    def stage_1(image):
        # ResNet 50 Tests
        # frame = resnet50.inference_and_return_annotated_image(frame)
        # print(resnet50.inference(frame))

        inference_result = resnet50.inference(image)

        significant_result, significant_result_bounding_box = determine_if_detection_is_bigger_then(inference_result,
                                                                                                    "person",
                                                                                                    frame_area)

        # TODO: Hide These Tests!
        try:
            first_box_size = calculate_bounding_area(*(inference_result[0][1]))
            print(f"The detected person takes up {(first_box_size / frame_area * 100):.4}% of the frame.")
        except:
            pass

        if significant_result:
            return True, crop_image_to_bounding_box(image, *significant_result_bounding_box, offset=((0, 0), (-30, -20)))
        else:
            return False, None


    # Stage 2, Face Detection Setup
    def stage_2(image):
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalcatface_extended.xml")

        greyed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        results = face_cascade.detectMultiScale(greyed_image, 1.1, 3)

        print(results)

        significant_result = len(results) > 0

        if significant_result:
            x, y, w, h = results[0]
            return True, crop_image_to_bounding_box(image, x, y, x+w, y+h)
        else:
            return False, None

    # Stage 3, Face Matching Setup
    model_training_instance = ModelTraining()
    model_training_instance.train()
    def stage_3(image):
        result = model_training_instance.predict(image)

        if result == 1:
            return True
        else:
            return False

    # General Setup
    cap = cv2.VideoCapture(0)


    frame = None

    while True:
        ret, frame = cap.read()
        if frame is None:
            continue
        cv2.waitKey(1)
        cv2.imshow("Preview", frame)

        # Calculate the frame area
        frame_area = frame.shape[0] * frame.shape[1]

        proceed, processed_frame = stage_1(frame)

        if (not proceed) or (processed_frame.shape[0] < 10 or processed_frame.shape[1] < 10):
            continue

        cv2.imshow("Processed Frame 1", processed_frame)

        proceed, processed_frame = stage_2(processed_frame)

        if not proceed:
            continue

        result = stage_3(processed_frame)



