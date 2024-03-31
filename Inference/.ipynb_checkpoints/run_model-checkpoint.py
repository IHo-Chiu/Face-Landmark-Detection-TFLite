import os
import argparse
import numpy as np
import tensorflow as tf
import tflite_runtime.interpreter as tflite
import cv2
import math


class YOLOv8_face:
    def __init__(self, conf_thres=0.2, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.class_names = ['face']
        self.num_classes = len(self.class_names)
        # Initialize model
        # self.net = cv2.dnn.readNet(path)
        self.input_height = 640
        self.input_width = 640
        self.reg_max = 16

        self.project = np.arange(self.reg_max)
        self.strides = (8, 16, 32)
        self.feats_hw = [(math.ceil(self.input_height / self.strides[i]), math.ceil(self.input_width / self.strides[i])) for i in range(len(self.strides))]
        self.anchors = self.make_anchors(self.feats_hw)

    def make_anchors(self, feats_hw, grid_cell_offset=0.5):
        """Generate anchors from features."""
        anchor_points = {}
        for i, stride in enumerate(self.strides):
            h,w = feats_hw[i]
            x = np.arange(0, w) + grid_cell_offset  # shift x
            y = np.arange(0, h) + grid_cell_offset  # shift y
            sx, sy = np.meshgrid(x, y)
            # sy, sx = np.meshgrid(y, x)
            anchor_points[stride] = np.stack((sx, sy), axis=-1).reshape(-1, 2)
        return anchor_points

    def softmax(self, x, axis=1):
        x_exp = np.exp(x)
        # 如果是列向量，则axis=0
        x_sum = np.sum(x_exp, axis=axis, keepdims=True)
        s = x_exp / x_sum
        return s
    
    def resize_image(self, srcimg, keep_ratio=True):
        top, left, newh, neww = 0, 0, self.input_width, self.input_height
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.input_height, int(self.input_width / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.input_width - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, left, self.input_width - neww - left, cv2.BORDER_CONSTANT,
                                         value=(0, 0, 0))  # add border
            else:
                newh, neww = int(self.input_height * hw_scale), self.input_width
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.input_height - newh) * 0.5)
                img = cv2.copyMakeBorder(img, top, self.input_height - newh - top, 0, 0, cv2.BORDER_CONSTANT,
                                         value=(0, 0, 0))
        else:
            img = cv2.resize(srcimg, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA)
        return img, newh, neww, top, left

    def detect(self, srcimg):
        input_img, newh, neww, padh, padw = self.resize_image(cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB))
        scale_h, scale_w = srcimg.shape[0]/newh, srcimg.shape[1]/neww

        input_img = input_img.transpose((2, 0, 1))
        input_img = np.expand_dims(input_img, 0)
        input_img = np.ascontiguousarray(input_img)
        
        input_img = input_img.astype(np.float32) / 255.0
        
        yolo_interpreter.set_tensor(yolo_input_details[0]['index'], input_img)
        yolo_interpreter.invoke()
        outputs = [
            yolo_interpreter.get_tensor(yolo_output_details[2]['index']),
            yolo_interpreter.get_tensor(yolo_output_details[0]['index']),
            yolo_interpreter.get_tensor(yolo_output_details[1]['index'])
        ]
        
        mlvl_bboxes, confidences = self.post_process(outputs, scale_h, scale_w, padh, padw)
        mlvl_bboxes /= np.array([[srcimg.shape[1], srcimg.shape[0], srcimg.shape[1], srcimg.shape[0]]])

        return mlvl_bboxes, confidences

    def post_process(self, preds, scale_h, scale_w, padh, padw):
        bboxes, scores, landmarks = [], [], []
        for i, pred in enumerate(preds):
            stride = int(self.input_height/pred.shape[2])
            pred = pred.transpose((0, 2, 3, 1))
            
            box = pred[..., :self.reg_max * 4]
            cls = 1 / (1 + np.exp(-pred[..., self.reg_max * 4:-15])).reshape((-1,1))

            # tmp = box.reshape(self.feats_hw[i][0], self.feats_hw[i][1], 4, self.reg_max)
            tmp = box.reshape(-1, 4, self.reg_max)
            bbox_pred = self.softmax(tmp, axis=-1)
            bbox_pred = np.dot(bbox_pred, self.project).reshape((-1,4))

            bbox = self.distance2bbox(self.anchors[stride], bbox_pred, max_shape=(self.input_height, self.input_width)) * stride

            bbox -= np.array([[padw, padh, padw, padh]])  ###合理使用广播法则
            bbox *= np.array([[scale_w, scale_h, scale_w, scale_h]])

            bboxes.append(bbox)
            scores.append(cls)

        bboxes = np.concatenate(bboxes, axis=0)
        scores = np.concatenate(scores, axis=0)
    
        bboxes_wh = bboxes.copy()
        bboxes_wh[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]  ####xywh
        confidences = np.max(scores, axis=1)  ####max_class_confidence

        mask = confidences>self.conf_threshold
        bboxes_wh = bboxes_wh[mask]  ###合理使用广播法则
        confidences = confidences[mask]

        indices = cv2.dnn.NMSBoxes(bboxes_wh.tolist(), confidences.tolist(), self.conf_threshold, self.iou_threshold).flatten()
        if len(indices) > 0:
            mlvl_bboxes = bboxes_wh[indices]
            confidences = confidences[indices]
            return mlvl_bboxes, confidences
        else:
            return np.array([]), np.array([])

    def distance2bbox(self, points, distance, max_shape=None):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)
    
    def draw_detections(self, image, boxes, scores):
        imw, imh, imc = image.shape
        for box, score in zip(boxes, scores):
            x, y, w, h = int(box[0]*imw), int(box[1]*imh), int(box[2]*imw), int(box[3]*imh)
            # Draw rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)
            cv2.putText(image, "face:"+str(round(score,2)), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
        return image


class GetCropMatrix():
    def __init__(self, image_size, target_face_scale, align_corners=False):
        self.image_size = image_size
        self.target_face_scale = target_face_scale
        self.align_corners = align_corners

    def _compose_rotate_and_scale(self, angle, scale, shift_xy, from_center, to_center):
        cosv = math.cos(angle)
        sinv = math.sin(angle)

        fx, fy = from_center
        tx, ty = to_center

        acos = scale * cosv
        asin = scale * sinv

        a0 = acos
        a1 = -asin
        a2 = tx - acos * fx + asin * fy + shift_xy[0]

        b0 = asin
        b1 = acos
        b2 = ty - asin * fx - acos * fy + shift_xy[1]

        rot_scale_m = np.array([
            [a0, a1, a2],
            [b0, b1, b2],
            [0.0, 0.0, 1.0]
        ], np.float32)
        return rot_scale_m

    def process(self, scale, center_w, center_h):
        if self.align_corners:
            to_w, to_h = self.image_size - 1, self.image_size - 1
        else:
            to_w, to_h = self.image_size, self.image_size

        rot_mu = 0
        scale_mu = self.image_size / (scale * self.target_face_scale * 200.0)
        shift_xy_mu = (0, 0)
        matrix = self._compose_rotate_and_scale(
            rot_mu, scale_mu, shift_xy_mu,
            from_center=[center_w, center_h],
            to_center=[to_w / 2.0, to_h / 2.0])
        return matrix


class TransformPerspective():
    def __init__(self, image_size):
        self.image_size = image_size

    def process(self, image, matrix):
        return cv2.warpPerspective(
            image, matrix, dsize=(self.image_size, self.image_size),
            flags=cv2.INTER_LINEAR, borderValue=0)

getCropMatrix = GetCropMatrix(image_size=256, target_face_scale=1.0, align_corners=True)
transformPerspective = TransformPerspective(image_size=256)

def star_postprocess(srcPoints, coeff):
    dstPoints = np.zeros(srcPoints.shape, dtype=np.float32)
    for i in range(srcPoints.shape[0]):
        dstPoints[i][0] = coeff[0][0] * srcPoints[i][0] + coeff[0][1] * srcPoints[i][1] + coeff[0][2]
        dstPoints[i][1] = coeff[1][0] * srcPoints[i][0] + coeff[1][1] * srcPoints[i][1] + coeff[1][2]
    return dstPoints
        
def star_process(input_image, faces):
    results = []
    for face in faces:
        w, h, c = input_image.shape
        center_w = w * (face[0] + face[2]/2)
        center_h = h * (face[1] + face[3]/2)
        scale = (max(face[2] * w, face[3] * h) + 10) / 200

        scale, center_w, center_h = float(scale), float(center_w), float(center_h)
        matrix = getCropMatrix.process(scale, center_w, center_h)
        input_tensor = transformPerspective.process(input_image, matrix)
        
        input_tensor = input_tensor[np.newaxis, :].astype(np.float32)
        input_tensor = np.transpose(input_tensor ,(0, 3, 1, 2))
        input_tensor = input_tensor / 255.0 * 2.0 - 1.0

        # run model
        star_interpreter.set_tensor(star_input_details[0]['index'], input_tensor)
        star_interpreter.invoke()

        for i, star_output_detail in enumerate(star_output_details):
            if star_output_detail['name'] == 'PartitionedCall:7':
                output = star_interpreter.get_tensor(star_output_details[i]['index'])
                output = output[0]
        
        # [-1, +1] -> [-0.5, SIZE-0.5]
        output = (output+1)/2 * 256
        output = star_postprocess(output, np.linalg.inv(matrix))

        results.append(output)
        
    return results
    

def draw_pts(img, pts, mode="pts", shift=4, color=(0, 255, 0), radius=1, thickness=1, save_path=None, dif=0,
             scale=0.3, concat=False, ):
    img_draw = img
    for cnt, p in enumerate(pts):
        if mode == "index":
            cv2.putText(img_draw, str(cnt), (int(float(p[0] + dif)), int(float(p[1] + dif))), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, thickness)
        elif mode == 'pts':
            if len(img_draw.shape) > 2:
                # 此处来回切换是因为opencv的bug
                img_draw = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
                img_draw = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)
            cv2.circle(img_draw, (int(p[0] * (1 << shift)), int(p[1] * (1 << shift))), radius << shift, color, -1,
                       cv2.LINE_AA, shift=shift)
        else:
            raise NotImplementedError
    if concat:
        img_draw = np.concatenate((img, img_draw), axis=1)
    if save_path is not None:
        cv2.imwrite(save_path, img_draw)
    return img_draw


    
if __name__ == '__main__':
    # read args
    parser = argparse.ArgumentParser()
    parser.add_argument("image_list", type=str)
    parser.add_argument("output_folder", type=str)
    parser.add_argument("--visualize", action='store_true')
    args = parser.parse_args()
    
    image_list = args.image_list
    output_folder = args.output_folder
    visualize = args.visualize

    # load images
    with open(image_list, 'r') as f:
        image_paths = f.readlines()
        
    # load YOLO
    YOLOv8_face_detector = YOLOv8_face(conf_thres=0.1, iou_thres=0.5)
    
    yolo_interpreter = tf.lite.Interpreter(model_path="yolo.tflite")
    yolo_interpreter.allocate_tensors()
    yolo_input_details = yolo_interpreter.get_input_details()
    yolo_output_details = yolo_interpreter.get_output_details()

    # load STAR
    star_interpreter = tf.lite.Interpreter(model_path="star.tflite")
    star_interpreter.allocate_tensors()
    star_input_details = star_interpreter.get_input_details()
    star_output_details = star_interpreter.get_output_details()

    
    for image_path in image_paths:
        image_path = image_path.replace('\n', '')
        image = cv2.imread(image_path)
        
        boxes, scores = YOLOv8_face_detector.detect(image)

        ixs = boxes[...,0].argsort()[...,None]
        boxes = np.take_along_axis(boxes, ixs, 0)
    
        result = star_process(image, boxes)
    
        output_file_name, _ = os.path.splitext(os.path.basename(image_path))
        output_file_name += '.txt'
        output_path = os.path.join(output_folder, output_file_name)
        os.makedirs(output_folder, exist_ok=True)
        with open(output_path, 'w') as f:
            for face in result:
                f.write('version: 1\n')
                f.write('n_points: 51\n')
                f.write('{\n')
                for point in face:
                    f.write(str(point[0]) + ' ' + str(point[1]) + '\n')
                f.write('}\n')

        # visualize
        if visualize == True:
            output_file_name, _ = os.path.splitext(os.path.basename(image_path))
            output_file_name += '.jpg'
            output_path = os.path.join(output_folder, output_file_name)
            dstimg = YOLOv8_face_detector.draw_detections(image, boxes, scores)
            for face in result:
                dstimg = draw_pts(dstimg, face)
            cv2.imwrite(output_path, dstimg)


