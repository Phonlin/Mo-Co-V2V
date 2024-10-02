import pyrealsense2 as rs
import numpy as np
import cv2
import datetime

# counter = 0
# 若要保存到其他資料夾，修改此路徑
folder = "/home/phon/下載/3D_detection-master/video_demo/"

pipeline = rs.pipeline()
config = rs.config()

width = 1280
height = 720
fps = 30

# 配置顏色相機
config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

# 開始串流
profile = pipeline.start(config)

### svae results
# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change the codec if needed (e.g., 'XVID')
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
out = cv2.VideoWriter(folder + 'output_video' + current_time + '.mp4', fourcc, fps, (width, height))

try:
    while True:
        frames = pipeline.wait_for_frames()

        # 獲取顏色幀
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            continue
        color_frame = np.asanyarray(color_frame.get_data())

        color_frame = cv2.flip(color_frame, 0)
        color_frame = cv2.flip(color_frame, 1)

        cv2.imshow("color", color_frame)
        out.write(color_frame)

        c = cv2.waitKey(1)

        # 如果按下ESC則關閉視窗（ESC的ascii碼為27），同時跳出迴圈
        if c == 27:
            cv2.destroyAllWindows()
            break

finally:
    # 停止串流
    pipeline.stop()