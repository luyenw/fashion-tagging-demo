import tensorflow as tf
import cv2
import numpy as np
from tqdm import tqdm
import six
import pandas as pd
import json

def convert_to_numpy(predictions):
    new_predictions = {}
    for key, value in predictions.items():
        if key != 'source_id':
            new_predictions[key] = value.numpy()
        else:
            new_predictions[key] = value

    return new_predictions

def process_predictions(predictions):
    image_scale = np.tile(predictions['image_info'][:, 2:3, :], (1, 1, 2))
    predictions['detection_boxes'] = (
        predictions['detection_boxes'].astype(np.float32))
    predictions['detection_boxes'] /= image_scale
    if 'detection_outer_boxes' in predictions:
        predictions['detection_outer_boxes'] = (
            predictions['detection_outer_boxes'].astype(np.float32))
        predictions['detection_outer_boxes'] /= image_scale
        
    return  predictions

def yxyx_to_xywh(boxes):
  """Converts boxes from ymin, xmin, ymax, xmax to xmin, ymin, width, height.

  Args:
    boxes: a numpy array whose last dimension is 4 representing the coordinates
      of boxes in ymin, xmin, ymax, xmax order.

  Returns:
    boxes: a numpy array whose shape is the same as `boxes` in new format.

  Raises:
    ValueError: If the last dimension of boxes is not 4.
  """
  if boxes.shape[-1] != 4:
    raise ValueError(
        'boxes.shape[-1] is {:d}, but must be 4.'.format(boxes.shape[-1]))

  boxes_ymin = boxes[..., 0]
  boxes_xmin = boxes[..., 1]
  boxes_width = boxes[..., 3] - boxes[..., 1]
  boxes_height = boxes[..., 2] - boxes[..., 0]
  new_boxes = np.stack(
      [boxes_xmin, boxes_ymin, boxes_width, boxes_height], axis=-1)

  return new_boxes

def get_new_image_size(image_size, output_size: int):
    image_height, image_width = image_size

    if image_width > image_height:
        scale = image_width / output_size
        new_width = output_size
        new_height = int(image_height / scale)
    else:
        scale = image_height / output_size
        new_height = output_size
        new_width = int(image_width / scale)

    return new_height, new_width

def paste_instance_masks(masks,
                         detected_boxes,
                         image_height,
                         image_width):
  """Paste instance masks to generate the image segmentation results.

  Args:
    masks: a numpy array of shape [N, mask_height, mask_width] representing the
      instance masks w.r.t. the `detected_boxes`.
    detected_boxes: a numpy array of shape [N, 4] representing the reference
      bounding boxes.
    image_height: an integer representing the height of the image.
    image_width: an integer representing the width of the image.

  Returns:
    segms: a numpy array of shape [N, image_height, image_width] representing
      the instance masks *pasted* on the image canvas.
  """

  def expand_boxes(boxes, scale):
    """Expands an array of boxes by a given scale."""
    # Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/boxes.py#L227  # pylint: disable=line-too-long
    # The `boxes` in the reference implementation is in [x1, y1, x2, y2] form,
    # whereas `boxes` here is in [x1, y1, w, h] form
    w_half = boxes[:, 2] * .5
    h_half = boxes[:, 3] * .5
    x_c = boxes[:, 0] + w_half
    y_c = boxes[:, 1] + h_half

    w_half *= scale
    h_half *= scale

    boxes_exp = np.zeros(boxes.shape)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half

    return boxes_exp

  # Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/core/test.py#L812  # pylint: disable=line-too-long
  # To work around an issue with cv2.resize (it seems to automatically pad
  # with repeated border values), we manually zero-pad the masks by 1 pixel
  # prior to resizing back to the original image resolution. This prevents
  # "top hat" artifacts. We therefore need to expand the reference boxes by an
  # appropriate factor.
  _, mask_height, mask_width = masks.shape
  scale = max((mask_width + 2.0) / mask_width,
              (mask_height + 2.0) / mask_height)

  ref_boxes = expand_boxes(detected_boxes, scale)
  ref_boxes = ref_boxes.astype(np.int32)
  padded_mask = np.zeros((mask_height + 2, mask_width + 2), dtype=np.float32)
  segms = []
  for mask_ind, mask in enumerate(masks):
    im_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    # Process mask inside bounding boxes.
    padded_mask[1:-1, 1:-1] = mask[:, :]

    ref_box = ref_boxes[mask_ind, :]
    w = ref_box[2] - ref_box[0] + 1
    h = ref_box[3] - ref_box[1] + 1
    w = np.maximum(w, 1)
    h = np.maximum(h, 1)

    mask = cv2.resize(padded_mask, (w, h))
    mask = np.array(mask > 0.5, dtype=np.uint8)

    x_0 = min(max(ref_box[0], 0), image_width)
    x_1 = min(max(ref_box[2] + 1, 0), image_width)
    y_0 = min(max(ref_box[1], 0), image_height)
    y_1 = min(max(ref_box[3] + 1, 0), image_height)

    im_mask[y_0:y_1, x_0:x_1] = mask[
        (y_0 - ref_box[1]):(y_1 - ref_box[1]),
        (x_0 - ref_box[0]):(x_1 - ref_box[0])
    ]
    segms.append(im_mask)

  segms = np.array(segms)
  assert masks.shape[0] == segms.shape[0]
  return segms

def encode_mask(mask: np.ndarray) -> str:
    pixels = mask.T.flatten()

    # We need to allow for cases where there is a '1' at either end of the sequence.
    # We do this by padding with a zero at each end when needed.
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded

    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle = rle - 1

    rle[1::2] = rle[1::2] - rle[:-1:2]

    return ' '.join(str(x) for x in rle)

def convert_predictions_to_coco_annotations(predictions, eval_image_sizes: dict = None, output_image_size: int = None,
                                            encode_mask_fn=None, score_threshold=0.05):
  """Converts a batch of predictions to annotations in COCO format.

  Args:
    predictions: a dictionary of lists of numpy arrays including the following
      fields. K below denotes the maximum number of instances per image.
      Required fields:
        - source_id: a list of numpy arrays of int or string of shape
            [batch_size].
        - num_detections: a list of numpy arrays of int of shape [batch_size].
        - detection_boxes: a list of numpy arrays of float of shape
            [batch_size, K, 4], where coordinates are in the original image
            space (not the scaled image space).
        - detection_classes: a list of numpy arrays of int of shape
            [batch_size, K].
        - detection_scores: a list of numpy arrays of float of shape
            [batch_size, K].
      Optional fields:
        - detection_masks: a list of numpy arrays of float of shape
            [batch_size, K, mask_height, mask_width].

  Returns:
    coco_predictions: prediction in COCO annotation format.
  """
  coco_predictions = []
  num_batches = len(predictions['source_id'])
  use_outer_box = 'detection_outer_boxes' in predictions

  if encode_mask_fn is None:
     raise Exception

  for i in tqdm(range(num_batches), total=num_batches):
    predictions['detection_boxes'][i] = yxyx_to_xywh(
        predictions['detection_boxes'][i])

    if use_outer_box:
      predictions['detection_outer_boxes'][i] = yxyx_to_xywh(
          predictions['detection_outer_boxes'][i])
      mask_boxes = predictions['detection_outer_boxes']
    else:
      mask_boxes = predictions['detection_boxes']

    batch_size = predictions['source_id'][i].shape[0]
    for j in range(batch_size):
      image_id = predictions['source_id'][i][j]
      orig_image_size = predictions['image_info'][i][j, 0]

      if eval_image_sizes:
        eval_image_size = eval_image_sizes[image_id] if eval_image_sizes else orig_image_size
      elif output_image_size:
        eval_image_size = get_new_image_size(orig_image_size, output_image_size)
      else:
        eval_image_size = orig_image_size

      eval_scale = orig_image_size[0] / eval_image_size[0]

      bbox_indices = np.argwhere(predictions['detection_scores'][i][j] >= score_threshold).flatten()

      if 'detection_masks' in predictions:
        predicted_masks = predictions['detection_masks'][i][j, bbox_indices]
        image_masks = paste_instance_masks(
            predicted_masks,
            mask_boxes[i][j, bbox_indices].astype(np.float32) / eval_scale,
            int(eval_image_size[0]),
            int(eval_image_size[1]))
        binary_masks = (image_masks > 0.0).astype(np.uint8)
        encoded_masks = [encode_mask_fn(binary_mask) for binary_mask in list(binary_masks)]

        mask_masks = (predicted_masks > 0.5).astype(np.float32)
        mask_areas = mask_masks.sum(axis=-1).sum(axis=-1)
        mask_area_fractions = (mask_areas / np.prod(predicted_masks.shape[1:])).tolist()
        mask_mean_scores = ((predicted_masks * mask_masks).sum(axis=-1).sum(axis=-1) / mask_areas).tolist()

      for m, k in enumerate(bbox_indices):
        ann = {
          'image_id': int(image_id),
          'category_id': int(predictions['detection_classes'][i][j, k]),
          'bbox': (predictions['detection_boxes'][i][j, k].astype(np.float32) / eval_scale).tolist(),
          'score': float(predictions['detection_scores'][i][j, k]),
        }

        if 'detection_masks' in predictions:
          ann['segmentation'] = encoded_masks[m]
          ann['mask_mean_score'] = mask_mean_scores[m]
          ann['mask_area_fraction'] = mask_area_fractions[m]

        if 'detection_attributes' in predictions:
          ann['attribute_probabilities'] = predictions['detection_attributes'][i][j, k].tolist()

        coco_predictions.append(ann)

  for i, ann in enumerate(coco_predictions):
    ann['id'] = i + 1

  return coco_predictions

def load_map():
    category_map = pd.read_csv('assets/category_map.csv', index_col = 0)
    attribute_map = pd.read_csv('assets/attribute_map.csv', index_col= 0)

    map_new_id_att = {key: value for value, key in enumerate(list(attribute_map.index))}
    category_to_attribute_content = json.load(open('assets/category-attributes.json', 'r'))
    category_to_attribute_map = {}

    for ix in range(46):
        value = np.zeros(294)
        if str(ix) in category_to_attribute_content.keys():
            for ele in category_to_attribute_content[str(ix)]:
                value[map_new_id_att[ele]] = 1.
        category_to_attribute_map[ix + 1] = value
    
    return category_map, attribute_map, category_to_attribute_map

def process_output(predictions, category_map, attribute_map, category_to_attribute):
    

    for i in range(len(predictions)):
        predictions[i]['category_info'] = category_map[category_map['id'] == predictions[i]['category_id']].iloc[0].to_dict()
        attributes_ele = np.array(predictions[i]['attribute_probabilities']) * category_to_attribute[predictions[i]['category_id']]
        predictions[i]['attribute_info'] = attribute_map[attributes_ele > attribute_map['threshold']].T.to_dict()
    return predictions

def filter_output(predictions):
    new_predictions = []

    for ele in predictions:
       new_predictions.append({'image_id': ele['image_id'],
                               'category_id': ele['category_id'],
                               'bbox': ele['bbox'],
                               'score': ele['score'],
                               'id': ele['id'],
                               'category_info': ele['category_info'],
                               'attribute_info': ele['attribute_info']
                               })
    
    return new_predictions

def postprocess(predictions, score_threshold = 0.8):

    predictions = convert_to_numpy(predictions)
    predictions = process_predictions(predictions)

    processed_predictions = {}
    for k, v in six.iteritems(predictions):
      if k not in processed_predictions:
        processed_predictions[k] = [v]
      else:
        processed_predictions[k].append(v)
    
    coco_result = convert_predictions_to_coco_annotations(processed_predictions,
                                                            output_image_size=1024,
                                                            encode_mask_fn=encode_mask,
                                                            score_threshold=score_threshold)
    

    category_map, attribute_map, category_to_attribute_map = load_map()

    final_result = process_output(coco_result, category_map, attribute_map, category_to_attribute_map)

    final_result = filter_output(final_result)
    
    return final_result
    

def process_one_image(model, img, id = 1):
    img = tf.constant(img, dtype = tf.uint8)
    img = tf.expand_dims(img, axis = 0)

    prediction = model.signatures['serving_default'](input = img)
    prediction['source_id'] = np.array([id])
    prediction = postprocess(prediction)

    return prediction

from PIL import Image
import random

def random_hex(length):
    """
    Generates a random hexadecimal string of the given length.
    """
    hex_digits = "0123456789abcdef"
    return ''.join(random.choice(hex_digits) for _ in range(length))
def is_image(filename):
    try:
        with Image.open(filename) as img:
            return True
    except:
        return False

def get_response_msg(data):
  output = []
  for i in range(len(data)):
      object = data[i]
      bbox = object['bbox']
      category = object["category_info"]['name']
      list_attr = {}
      list_attr['bbox'] = bbox
      list_attr['category'] = category
      for attr in object["attribute_info"]:
          r = object["attribute_info"][attr]
          list_attr[r['supercategory']] = r['name']
      output.append(list_attr)
  return output
    