import cv2
import numpy as np
import os
from util.post_processing import gen_3D_box,draw_3D_box, draw_2D_box
from net.bbox_3D_net import bbox_3D_net
from util.process_data import get_cam_data, get_dect2D_data

# stream
import pyrealsense2 as rs

ctx = rs.context()
devices = ctx.query_devices()
dev0 = ctx.query_devices()[0]  
dev1 = ctx.query_devices()[1]

width = 1280
height = 720
fps = 30

print('Config ... ')
pipe1 = rs.pipeline()
cfg1 = rs.config()
cfg1.enable_device(dev0.get_info(rs.camera_info.serial_number))
cfg1.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)

pipe2 = rs.pipeline()
cfg2 = rs.config()
cfg2.enable_device(dev1.get_info(rs.camera_info.serial_number))
cfg2.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)


# 開始串流
profile = pipe1.start(cfg1)
profile = pipe2.start(cfg2)

from ultralytics import YOLO
from collections import deque

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Construct the network
model = bbox_3D_net((224,224,3))

model.load_weights(r'model_saved/weights_0415.h5')

# image_dir = '/home/phon/project/kitti/test/image/'
calib_file = 'calib_435i.txt'
# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change the codec if needed (e.g., 'XVID')
out = cv2.VideoWriter('output_video.mp4', fourcc, 30, (width+512, height))

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
                    if scores >= 0:
                        xmin    = bbox_coords[0]
                        ymin    = bbox_coords[1]
                        xmax    = bbox_coords[2]
                        ymax    = bbox_coords[3]
                        
                        if (xmin <= width*0.03) or (xmax >= width*0.97) or (ymin <= height*0.03) or (ymax >= height*0.97):
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
        patch = patch - np.array([[[103.939, 116.779, 123.68]]]) # 修正3D框用
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

def process_BEV(bev, oth_x, oth_y, loca_angle, oth_bev):
    loca_angle = np.radians(loca_angle)
    angle = np.radians(30); # 假設偏移30度, 換弧度
    canva = np.zeros((512, 512, 3), np.uint8)
    canva = cv2.line(canva, (256, 0), (256, 512), (255, 0, 0), 7)

    
    for info in bev:
        # [cls, center, yaw]
        cls = info[0]
        center = info[1]
        yaw = -info[2] + np.radians(10) # 補償角度10

        # print(item)
        x = center[0]-2.4 # 橫向最遠30m
        y = center[2]-8.4 # 縱向最遠70m, 降低中心點跳動
        # print(x, y)

        # 修正誤差旋轉
        vector = np.array([x, y])
        rotation_matrix = np.array([
                                    [np.cos(0), -np.sin(0)],
                                    [np.sin(0), np.cos(0)]
                                    ])
        rotation_vector = np.dot(rotation_matrix, vector)

        x = 256+rotation_vector[0] * (512/30.0) # 原點位置
        y = 512-rotation_vector[1] * (512/30.0) # 原點位置
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

        cv2.line(canva, lt, rt, (0, 255, 0), 1)
        cv2.line(canva, rt, rb, (0, 255, 0), 1)
        cv2.line(canva, rb, lb, (0, 255, 0), 1)
        cv2.line(canva, lb, lt, (0, 255, 0), 1)
    
    for info in oth_bev:
        # [cls, center, yaw]
        cls = info[0]
        center = info[1]
        yaw = -info[2]

        # print(center[0], center[2])
        x = center[0]-2.4 # 橫向最遠30m
        y = center[2]-8.4 # 縱向最遠70m, 降低中心點跳動
        # print(x, y)
        
        # 修正鏡頭角度
        vector = np.array([x, y])
        rotation_matrix = np.array([
                                    [np.cos(-loca_angle), -np.sin(-loca_angle)],
                                    [np.sin(-loca_angle), np.cos(-loca_angle)]
                                    ])
        rotation_vector = np.dot(rotation_matrix, vector)

        # print(rotation_vector[0], rotation_vector[1])
        # print((oth_y * (512/20.0)))
        x = 256 + (rotation_vector[0]) * (512/30.0) # 原點位置
        y = 512 - (rotation_vector[1] + oth_x) * (512/30.0) # 原點位置
        # canva = cv2.circle(canva, (int(x), int(y)), 5, (0, 0, 255), -1) # 中心點

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
                                    [np.cos(yaw-loca_angle), -np.sin(yaw-loca_angle)],
                                    [np.sin(yaw-loca_angle), np.cos(yaw-loca_angle)]
                                    ])
        box_rotation_result = np.dot(box_vector, box_rotation_matrix.T)
        
        # 四個框點 left top, right buttom...
        lt = (int(x+int(box_rotation_result[0][0])), int(y-int(box_rotation_result[0][1])))
        rt = (int(x+int(box_rotation_result[1][0])), int(y-int(box_rotation_result[1][1])))
        lb = (int(x+int(box_rotation_result[2][0])), int(y-int(box_rotation_result[2][1])))
        rb = (int(x+int(box_rotation_result[3][0])), int(y-int(box_rotation_result[3][1])))

        cv2.line(canva, lt, rt, (0, 255, 0), 1)
        cv2.line(canva, rt, rb, (0, 255, 0), 1)
        cv2.line(canva, rb, lb, (0, 255, 0), 1)
        cv2.line(canva, lb, lt, (0, 255, 0), 1)
    
    return canva

do = 0
cv2.namedWindow('tog', cv2.WINDOW_NORMAL)  # 視窗可調大小

while True:
    frames1 = pipe1.wait_for_frames()
    frames2 = pipe2.wait_for_frames()

    # 獲取顏色幀
    color_frame1 = frames1.get_color_frame()
    color_frame2 = frames2.get_color_frame()

    if not color_frame1 or not color_frame2:
        continue

    color_frame1 = np.asanyarray(color_frame1.get_data())
    color_frame2 = np.asanyarray(color_frame2.get_data())

    img1 = color_frame1.copy()
    img2 = color_frame2.copy()

    # 處理來自第一個攝影機的畫面
    img_2d1, bboxes1 = process2D(img1, True)
    img1, img_3d1, bev1 = process3D(color_frame1, bboxes1)

    # 處理來自第二個攝影機的畫面
    img_2d2, bboxes2 = process2D(img2, True)
    img2, img_3d2, bev2 = process3D(color_frame2, bboxes2)

    # 距離 0.2m, 實際乘62.43=12.49
    img_bev1 = process_BEV(bev1, 11.86, 11.86, 90, bev2)    # (bev, oth_x, oth_y, oth_angle, oth_bev)
    img_bev2 = process_BEV(bev2, 11.86, 11.86, -90, bev1)

    # 將 img_bev1 擴展到 720 的高度
    img_bev_expanded1 = np.ones((720, 512, 3), dtype=np.uint8)
    img_bev_expanded1 *= 255
    img_bev_expanded1[:512, :, :] = img_bev1
    combined_img1 = np.hstack((img_bev_expanded1, img_3d1))

    # 將 img_bev2 擴展到 720 的高度
    img_bev_expanded2 = np.ones((720, 512, 3), dtype=np.uint8)
    img_bev_expanded2 *= 255
    img_bev_expanded2[:512, :, :] = img_bev2
    combined_img2 = np.hstack((img_bev_expanded2, img_3d2))

    # 將兩個 combined_img 垂直堆疊在一起
    final_combined_img = np.vstack((combined_img1, combined_img2))
    
    cv2.imshow('tog', final_combined_img)

    out.write(final_combined_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()