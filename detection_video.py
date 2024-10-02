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

# image_dir = '/home/phon/project/kitti/test/image/'
calib_file = 'calib_435i.txt'

# Load video
video = cv2.VideoCapture('/home/phon/下載/3D_detection-master/video_demo/output_video2024-04-28_17-21-25.mp4') # 中山路直行
# video = cv2.VideoCapture('/home/phon/下載/3D_detection-master/video_demo/cut.mp4') # 和平路部份
# video = cv2.VideoCapture('/home/phon/下載/3D_detection-master/video_demo/cut2.mp4') # 中山路部份
# video = cv2.VideoCapture('/home/phon/下載/3D_detection-master/video_demo/output_video2024-04-28_17-34-32.mp4') # 重慶路
# video = cv2.VideoCapture('/home/phon/下載/3D_detection-master/video_demo/output_video2024-04-28_17-23-03.mp4') # 中山路轉彎
# video = cv2.VideoCapture('/home/phon/下載/3D_detection-master/video_demo/output_video2024-04-28_17-37-57.mp4') # 和平路全段

# 運算時降低解析度
scale = 1

### svae results
# Get video information (frame width, height, frames per second)
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
fps = int(video.get(cv2.CAP_PROP_FPS))
# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change the codec if needed (e.g., 'XVID')
# out = cv2.VideoWriter('output_video.mp4', fourcc, 30, (frame_width, frame_height)) # 正常
out = cv2.VideoWriter('output_video.mp4', fourcc, 30/5, (frame_width+512, frame_height)) # tog


# classes = ['Car','Van','Truck','Pedestrian','Person_sitting','Cyclist','Tram']
classes = ['Pedestrian', 'Cyclist', 'Car', 'motorcycle', 'airplane', 'Van', 'train', 'Truck', 'boat']

dims_avg = {'Car': np.array([1.52131309, 1.64441358, 3.85728004]), # 高 寬 長
'Van': np.array([2.18560847, 1.91077601, 5.08042328]),
'Truck': np.array([3.07044968,  2.62877944, 11.17126338]),
'Pedestrian': np.array([1.75562272, 0.67027992, 0.87397566]),
'Person_sitting': np.array([1.28627907, 0.53976744, 0.96906977]),
'Cyclist': np.array([1.73456498, 0.58174006, 1.77485499]),
'Tram': np.array([3.56020305,  2.40172589, 18.60659898])}

cam_to_img = get_cam_data(calib_file)

# P2 = np.array([[607.711, 0.0, 419.265, 0.0], [0.0, 609.418, 237.783, 0.0], [0.0, 0.0, 1.0, 0]])
# fx = 7.215377000000e+02
# u0 = 6.095593000000e+02 // 改檔案才有用
# v0 = 7.215377000000e+02

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
# bbox2d_model.overrides['classes'] = 2 # 只有 car
bbox2d_model.overrides['classes'] = 2, 7 # 車卡車
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
                    if scores >= 0.5:
                        xmin    = bbox_coords[0]
                        ymin    = bbox_coords[1]
                        xmax    = bbox_coords[2]
                        ymax    = bbox_coords[3]
                        
                        if (xmin <= frame_width*0.03) or (xmax >= frame_width*0.97) or (ymin <= frame_height*0.03) or (ymax >= frame_height*0.97):
                            continue

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

def process3D(img, bboxes):
    # img = cv2.imread(img)
    img_3d = img.copy()
    bev = []

    # pro_img, box = process2D(img, track=True) # 因為單純辨識不連續影像，不用 track

    # dect2D_data,box2d_reserved = get_dect2D_data(box2d_file,classes)

    for data in bboxes:
        # data: bbox_coords, scores, classes
        cls = int(data[2])
        box_2D = np.asarray(data[0],dtype=float)
        xmin = box_2D[0]
        ymin = box_2D[1]
        xmax = box_2D[2]
        ymax = box_2D[3]

        patch = img[int(ymin):int(ymax), int(xmin):int(xmax)]
        patch = cv2.resize(patch, (224, 224))
        patch = patch - np.array([[[103.939, 116.779, 123.68]]]) # 可能為修正3D框用
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

        # print(anchors[0]) # 1.0000001
        if anchors[0] > 1.0:
            continue

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

        points2D, center = gen_3D_box(yaw, dims, cam_to_img, box_2D)
        # print(center)
        info = [cls, center, yaw]
        bev.append(info)
        # print(points2D)
        draw_3D_box(img_3d, points2D)

    return img, img_3d, bev

def process_BEV(bev):
    angle = np.radians(30); # 假設偏移30度, 換弧度
    canva = np.zeros((512, 512, 3), np.uint8)
    canva = cv2.line(canva, (256, 0), (256, 512), (255, 0, 0), 3)
    # 'Car': np.array([1.52131309, 1.64441358, 3.85728004]), # 高 寬 長
    cv2.rectangle(canva, (256-int(1.64441358*(512/30.0)/2), 512-int(3.85728004*(512/30.0)/4)), (256+int(1.64441358*(512/30.0)/2), 512), (0, 0, 255), 3)

    for info in bev:
        # [cls, center, yaw]
        cls = info[0]
        center = info[1]
        yaw = -info[2] + np.radians(10) # 補償角度10

        # print(item)
        x = center[0]*(512/30.0) # 橫向最遠30m
        y = center[2]*(512/70.0) # 縱向最遠70m, 降低中心點跳動

        # 修正誤差旋轉
        vector = np.array([x, y])
        rotation_matrix = np.array([
                                    [np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]
                                    ])
        rotation_vector = np.dot(rotation_matrix, vector)

        x = 256+rotation_vector[0] # 原點位置
        y = 512-rotation_vector[1] # 原點位置
        canva = cv2.circle(canva, (int(x), int(y)), 5, (0, 0, 255), -1) # 中心點

        # 方框
        dim = dims_avg[classes[cls]]
        # 'Car': np.array([1.52131309, 1.64441358, 3.85728004]), # 高 寬 長

        width = dim[1]*(512/30.0)
        length = dim[2]*(512/30.0)
        
        lt = np.array([-(0.5*width), -(0.5*length)])
        rt = np.array([+(0.5*width), -(0.5*length)])
        lb = np.array([-(0.5*width), +(0.5*length)])
        rb = np.array([+(0.5*width), +(0.5*length)])

        # yaw角
        box_vector = np.array([lt, 
                               rt, 
                               lb, 
                               rb])
        box_rotation_matrix = np.array([
                                    [np.cos(yaw), -np.sin(yaw)],
                                    [np.sin(yaw), np.cos(yaw)]
                                    ])
        box_rotation_result = np.dot(box_vector, box_rotation_matrix.T)
        # print(box_rotation_result.shape)
        # print(box_rotation_result)
        
        # 四個框點 left top, right buttom...
        lt = (int(x+int(box_rotation_result[0][0])), int(y-int(box_rotation_result[0][1])))
        rt = (int(x+int(box_rotation_result[1][0])), int(y-int(box_rotation_result[1][1])))
        lb = (int(x+int(box_rotation_result[2][0])), int(y-int(box_rotation_result[2][1])))
        rb = (int(x+int(box_rotation_result[3][0])), int(y-int(box_rotation_result[3][1])))

        cv2.line(canva, lt, rt, (0, 255, 0), 3)
        cv2.line(canva, rt, rb, (0, 255, 0), 3)
        cv2.line(canva, rb, lb, (0, 255, 0), 3)
        cv2.line(canva, lb, lt, (0, 255, 0), 3)

    return canva

do = 0
while True:
    success, frame = video.read()
    # 跳 5fps
    if not do == 5:
        do+=1
        continue
    else:
        do = 0

    if not success:
        break
    frame = cv2.resize(frame, (frame_width, frame_height))

    img = frame.copy()

    img_2d, bboxes = process2D(img, True)
    img , img_3d, bev = process3D(frame, bboxes)
    img_bev = process_BEV(bev)

    # 將 img_bev 擴展到 720 的高度
    img_bev_expanded = np.ones((720, 512, 3), dtype=np.uint8)
    img_bev_expanded *= 255
    img_bev_expanded[:512, :, :] = img_bev
    # 水平堆疊 img_bev 和 img_3d
    combined_img = np.hstack((img_bev_expanded, img_3d))
    cv2.imshow('tog', combined_img)

    # cv2.imshow('2d', img_2d)
    # cv2.imshow('3d', img_3d)
    cv2.imshow('Bev', img_bev)
    # out.write(img_3d)
    out.write(combined_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # if cv2.waitKey(-1) != -1:
    #     continue

cv2.destroyAllWindows()
video.release()