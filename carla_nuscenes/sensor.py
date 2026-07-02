import numpy as np
import carla
from .actor import Actor

def parse_image(image):
    array = np.ndarray(
            shape=(image.height, image.width, 4),
            dtype=np.uint8, buffer=image.raw_data,order="C")
    return array

def parse_lidar_data(lidar_data):

    data_array = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')))
    data_array = np.reshape(data_array, (int(data_array.shape[0] / 4), 4))

    channels = []

    for ch in range(lidar_data.channels):
        channels.extend(
            [ch] * lidar_data.get_point_count(ch)
        )

    channels = np.asarray(channels, dtype=np.float32)[:, None]
    #print(f"Parse Lidar Shape: {data_array.shape}, Channel Shape: {channels.shape}, {data_array[10:20]}, channels: {channels[10:20]}")
    #final_data = np.column_stack([data_array, points_channel],dtype=np.float32)
    final_data = np.hstack((data_array, channels))
    final_data[:,1] = -1 * final_data[:,1] # LH to RH coordinate system
    #return data_array
    return final_data

def parse_thi_lidar_data(lidar_data):
    #print("ParseTHILidar")
    dtype= np.dtype([
        ('x',np.float32),
        ('y',np.float32),
        ('z',np.float32),
        ('reflectivity',np.float32),
        ('intensity',np.float32),
        ('object_tag',np.float32)])
    pts = np.copy(np.frombuffer(lidar_data.raw_data,dtype=dtype))
    points = np.vstack([pts['x'],-pts['y'],pts['z'],pts['reflectivity'],pts['intensity']]).T
    return points

def parse_radar_data(radar_data):
    points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4')).copy()
    return points

def parse_data(data):
    if isinstance(data,carla.Image):
        return parse_image(data)
    elif isinstance(data,carla.RadarMeasurement):
        return parse_radar_data(data)
    elif isinstance(data,carla.LidarMeasurement):
        return parse_lidar_data(data)
    elif isinstance(data, carla.THILidarMeasurement):
        return parse_thi_lidar_data(data)

def get_data_shape(data):
    if isinstance(data,carla.Image):
        return data.height,data.width
    else:
        return 0,0
class Sensor(Actor):
    def __init__(self, name, **args):
        super().__init__(**args)
        self.name = name
        self.data_list = []
    
    def get_data_list(self):
        return self.data_list
    
    def set_actor(self, id):
        super().set_actor(id)
        self.actor.listen(self.add_data)
    
    def spawn_actor(self):
        super().spawn_actor()
        self.actor.listen(self.add_data)

    def get_last_data(self):
        if self.data_list:
            return self.data_list[-1]
        else:
            return None
            
    def add_data(self,data):
        self.data_list.append((self.actor.parent.get_transform(),data))

    def get_transform(self):
        return self.actor.get_transform()
