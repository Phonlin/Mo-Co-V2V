import pyrealsense2 as rs

ctx = rs.context()
devices = ctx.query_devices()
dev0 = ctx.query_devices()[0]  
dev1 = ctx.query_devices()[1]

device_model0 = str(dev0.get_info(rs.camera_info.name))
device_model1 = str(dev1.get_info(rs.camera_info.name))

print(f'device_model0: {device_model0}') 
print(f'device_model1: {device_model1}')

print('Config ... ')
pipe1 = rs.pipeline()
cfg1 = rs.config()
cfg1.enable_device(dev0.get_info(rs.camera_info.serial_number))
cfg1.enable_stream(rs.stream.infrared, 1, 424, 240, rs.format.y8, 30)
cfg1.enable_stream(rs.stream.infrared, 2, 424, 240, rs.format.y8, 30)

pipe2 = rs.pipeline()
cfg2 = rs.config()
cfg2.enable_device(dev1.get_info(rs.camera_info.serial_number))
cfg2.enable_stream(rs.stream.infrared, 1,  424, 240, rs.format.y8, 30)
cfg2.enable_stream(rs.stream.infrared, 2,  424, 240, rs.format.y8, 30)