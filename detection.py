import cv2
import numpy as np
import os
from util.post_processing import gen_3D_box,draw_3D_box, draw_2D_box
from net.bbox_3D_net import bbox_3D_net
from util.process_data import get_cam_data, get_dect2D_data

from ultralytics import YOLO
from collections import deque

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Construct the network
model = bbox_3D_net((224,224,3))

model.load_weights(r'model_saved/weights_0415.h5')

image_dir = '/home/phon/project/kitti/test/image/'
calib_file = 'calib.txt'
# box2d_dir = 'F:/dataset/kitti/testing/label_2/' 

# classes = ['Car','Van','Truck','Pedestrian','Person_sitting','Cyclist','Tram']
classes = ['Pedestrian', 'Cyclist', 'Car', 'motorcycle', 'airplane', 'Van', 'train', 'Truck', 'boat']
# cls_to_ind = {cls:i for i,cls in enumerate(classes)}

# dims_avg = np.loadtxt(r'dataset/voc_dims.txt',delimiter=',')
dims_avg = {'Car': np.array([1.52131309, 1.64441358, 3.85728004]), # 高 寬 長
'Van': np.array([2.18560847, 1.91077601, 5.08042328]),
'Truck': np.array([3.07044968,  2.62877944, 11.17126338]),
'Pedestrian': np.array([1.75562272, 0.67027992, 0.87397566]),
'Person_sitting': np.array([1.28627907, 0.53976744, 0.96906977]),
'Cyclist': np.array([1.73456498, 0.58174006, 1.77485499]),
'Tram': np.array([3.56020305,  2.40172589, 18.60659898])}


all_image = sorted(os.listdir(image_dir))
# np.random.shuffle(all_image)

cam_to_img = get_cam_data(calib_file)
fx = cam_to_img[0][0]
u0 = cam_to_img[0][2]
v0 = cam_to_img[1][2]

# 2d rect
tracking_trajectories = {}
# Load a 2D model
bbox2d_model = YOLO('yolov8n-seg.pt')  # load an official model
# set model parameters
bbox2d_model.overrides['conf'] = 0.9  # NMS confidence threshold
bbox2d_model.overrides['iou'] = 0.45  # NMS IoU threshold
bbox2d_model.overrides['agnostic_nms'] = False  # NMS class-agnostic
bbox2d_model.overrides['max_det'] = 1000  # maximum number of detections per image
bbox2d_model.overrides['classes'] = 2 # 只有 car
# bbox2d_model.overrides['classes'] = 0, 1, 2, 3, 4, 5, 6, 7, 8 # 全部物品

def process2D(image, track = True, device ='cpu'):
    bboxes = []
    if track is True:
        results = bbox2d_model.track(image, verbose=False, device=device, persist=True)

        for id_ in list(tracking_trajectories.keys()):
            if id_ not in [int(bbox.id) for predictions in results if predictions is not None for bbox in predictions.boxes if bbox.id is not None]:
                del tracking_trajectories[id_]

        for predictions in results:
            if predictions is None:
                continue

            if predictions.boxes is None or predictions.masks is None or predictions.boxes.id is None:
                continue

            for bbox, masks in zip(predictions.boxes, predictions.masks):
                ## object detections
                for scores, classes, bbox_coords, id_ in zip(bbox.conf, bbox.cls, bbox.xyxy, bbox.id):
                    xmin    = bbox_coords[0]
                    ymin    = bbox_coords[1]
                    xmax    = bbox_coords[2]
                    ymax    = bbox_coords[3]
                    cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,0,225), 2)
                    bboxes.append([bbox_coords, scores, classes, id_])

                    label = (' '+f'ID: {int(id_)}'+' '+str(predictions.names[int(classes)]) + ' ' + str(round(float(scores) * 100, 1)) + '%')
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2, 1)
                    dim, baseline = text_size[0], text_size[1]
                    cv2.rectangle(image, (int(xmin), int(ymin)), ((int(xmin) + dim[0] //3) - 20, int(ymin) - dim[1] + baseline), (30,30,30), cv2.FILLED)
                    cv2.putText(image,label,(int(xmin), int(ymin) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                    centroid_x = (xmin + xmax) / 2
                    centroid_y = (ymin + ymax) / 2

                    # Append centroid to tracking_points
                    if id_ is not None and int(id_) not in tracking_trajectories:
                        tracking_trajectories[int(id_)] = deque(maxlen=5)
                    if id_ is not None:
                        tracking_trajectories[int(id_)].append((centroid_x, centroid_y))

                # Draw trajectories
                for id_, trajectory in tracking_trajectories.items():

                    # print(tracking_trajectories.items()) # test
                    # dict_items([(1, deque([(tensor(618.8979), tensor(499.7820)), (tensor(619.2719), tensor(499.2860)), (tensor(618.5344), tensor(499.2982)), (tensor(618.3712), tensor(499.1003)), (tensor(619.5695), tensor(499.5506))], maxlen=5))])
                    #  最大一次辨識 5 items

                    for i in range(1, len(trajectory)):
                        thickness = int(2 * (i / len(trajectory)) + 1)
                        cv2.line(image, (int(trajectory[i-1][0]), int(trajectory[i-1][1])), 
                                 (int(trajectory[i][0]), int(trajectory[i][1])), (255, 255, 255), thickness)

                ## object segmentations
                for mask in masks.xy:
                    polygon = mask
                    cv2.polylines(image, [np.int32(polygon)], True, (255, 0, 0), thickness=2)


    if not track:
        results = bbox2d_model.predict(image, verbose=False, device=device)  # predict on an image
        for predictions in results:
            if predictions is None:
                continue  # Skip this image if YOLO fails to detect any objects
            if predictions.boxes is None or predictions.masks is None:
                continue  # Skip this image if there are no boxes or masks

            for bbox, masks in zip(predictions.boxes, predictions.masks): 
                ## object detections
                for scores, classes, bbox_coords in zip(bbox.conf, bbox.cls, bbox.xyxy):
                    # print(scores)
                    if scores >= 0.5:
                        xmin    = bbox_coords[0]
                        ymin    = bbox_coords[1]
                        xmax    = bbox_coords[2]
                        ymax    = bbox_coords[3]
                        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,0,225), 2)
                        bboxes.append([bbox_coords, scores, classes])

                        label = (' '+str(predictions.names[int(classes)]) + ' ' + str(round(float(scores) * 100, 1)) + '%')
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2, 1)
                        dim, baseline = text_size[0], text_size[1]
                        cv2.rectangle(image, (int(xmin), int(ymin)), ((int(xmin) + dim[0] //3) - 20, int(ymin) - dim[1] + baseline), (30,30,30), cv2.FILLED)
                        cv2.putText(image,label,(int(xmin), int(ymin) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                    else:
                        continue

                ## object segmentations
                for mask in masks.xy:
                    polygon = mask
                    cv2.polylines(image, [np.int32(polygon)], True, (255, 0, 0), thickness=2)

    return image, bboxes

for f in all_image:
    image_file = image_dir + f
    # box2d_file = box2d_dir + f.replace('png', 'txt')

    img = cv2.imread(image_file)
    img_2d = img.copy()
    img_3d = img.copy()

    pro_img, box = process2D(img, track=False) # 因為單純辨識不連續影像，不用 track

    # dect2D_data,box2d_reserved = get_dect2D_data(box2d_file,classes)

    for data in box:
        # data: bbox_coords, scores, classes
        cls = int(data[2])
        box_2D = np.asarray(data[0],dtype=float)
        xmin = box_2D[0]
        ymin = box_2D[1]
        xmax = box_2D[2]
        ymax = box_2D[3]

        patch = img[int(ymin):int(ymax), int(xmin):int(xmax)]
        patch = cv2.resize(patch, (224, 224))
        patch = patch - np.array([[[103.939, 116.779, 123.68]]])
        patch = np.expand_dims(patch, 0)

        prediction = model.predict(patch)

        # compute dims
        dims = dims_avg[classes[cls]] + prediction[0][0]

        # Transform regressed angle
        box2d_center_x = (xmin + xmax) / 2.0
        # Transfer arctan() from (-pi/2,pi/2) to (0,pi)
        theta_ray = np.arctan(fx /(box2d_center_x - u0))
        if theta_ray<0:
            theta_ray = theta_ray+np.pi

        max_anc = np.argmax(prediction[2][0])
        anchors = prediction[1][0][max_anc]

        if anchors[1] > 0:
            angle_offset = np.arccos(anchors[0])
        else:
            angle_offset = -np.arccos(anchors[0])

        bin_num = prediction[2][0].shape[0]
        wedge = 2. * np.pi / bin_num
        theta_loc = angle_offset + max_anc * wedge

        theta = theta_loc + theta_ray
        # object's yaw angle
        yaw = np.pi/2 - theta

        points2D = gen_3D_box(yaw, dims, cam_to_img, box_2D)
        # print(points2D)
        draw_3D_box(img_3d, points2D)

    # for cls,box in box2d_reserved:
    #     draw_2D_box(img,box)

    cv2.imshow('2d', img)
    cv2.imshow('3d', img_3d)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite('output/'+ f.replace('png','jpg'), img)
