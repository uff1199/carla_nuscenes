import numpy as np
import carla
from .actor import Actor
from queue import Queue
from PIL import Image

class SensorSnapshot:
    __slots__ = ("timestamp")
    def __init__(self, sensor_data):
        self.timestamp = sensor_data.timestamp

class LidarSnapshot(SensorSnapshot):
    __slots__ = ("data_array", "channels", "_point_counts")
    def __init__(self, sensor_data):
        super().__init__(sensor_data)
        self.data_array = np.copy(np.frombuffer(sensor_data.raw_data, dtype=np.dtype('f4')))  # owned copy, not a view
        self.channels = sensor_data.channels
        self._point_counts = [sensor_data.get_point_count(ch) for ch in range(sensor_data.channels)]
    def get_point_count(self, ch):
        return self._point_counts[ch]

class THILidarSnapshot(SensorSnapshot):
    __slots__ = ("data_array","raw_data")
    def __init__(self,sensor_data):
        super().__init__(sensor_data)
        dtype= np.dtype([
            ('x',np.float32),
            ('y',np.float32),
            ('z',np.float32),
            ('reflectivity',np.float32),
            ('intensity',np.float32),
            ('object_tag',np.float32)])
        self.data_array = np.copy(np.frombuffer(sensor_data.raw_data,dtype=dtype))

class RadarSnapshot(SensorSnapshot):
    __slots__ = ("data_array","raw_data")
    def __init__(self,sensor_data):
         super().__init__(sensor_data)
         self.data_array = np.copy(np.frombuffer(sensor_data.raw_data, dtype=np.dtype('f4')))


class CameraSnapshot(SensorSnapshot):
    __slots__ = ("data_array","height", "width")
    def __init__(self,sensor_data):
         super().__init__(sensor_data)
         self.data_array = np.copy(np.ndarray(
            shape=(sensor_data.height, sensor_data.width, 4),
            dtype=np.uint8, buffer=sensor_data.raw_data,order="C"))
         self.height = sensor_data.height
         self.width = sensor_data.width

    def save_to_disk(self,path):
        image = self.data_array
        img = image.reshape((self.height, self.width, 4))
        img = img[:, :, :3]
        img = img[:, :, ::-1]
        
        pil_image = Image.fromarray(img)
        pil_image.save(path)

def parse_image(image):
    array = image.data_array
    return array

def parse_lidar_data(lidar_data):

    data_array = lidar_data.data_array
    data_array = np.reshape(data_array, (int(data_array.shape[0] / 4), 4))

    channels = []

    for ch in range(lidar_data.channels):
        channels.extend(
            [ch] * lidar_data.get_point_count(ch)
        )
    no_points_channel = sum(lidar_data.get_point_count(ch) for ch in range(lidar_data.channels))
    if no_points_channel != data_array.shape[0]:
        print(f"MISMATCH: no_points_channel={no_points_channel}, data_array.shape={data_array.shape}", flush=True)
        raise RuntimeError("point count mismatch")

    channels = np.asarray(channels, dtype=np.float32)[:, None]
    #print(f"Parse Lidar Shape: {data_array.shape}, Channel Shape: {channels.shape}, {data_array[10:20]}, channels: {channels[10:20]}")
    #final_data = np.column_stack([data_array, points_channel],dtype=np.float32)
    
    final_data = np.hstack((data_array, channels))
    final_data[:,3] = final_data[:,3] * 255.0
    final_data[:,1] = -1 * final_data[:,1] # LH to RH coordinate system
    #return data_array
    return final_data

def parse_thi_lidar_data(lidar_data):
    #print("ParseTHILidar")
    pts = lidar_data.data_array
    points = np.vstack([pts['x'],-pts['y'],pts['z'],pts['reflectivity']*255.0,pts['intensity']*255.0]).T
    return points, pts['object_tag']

def parse_radar_data(radar_data):
    points = radar_data.data_array
    return points

def parse_data(data):
    if isinstance(data,CameraSnapshot):
        return parse_image(data)
    elif isinstance(data,RadarSnapshot):
        return parse_radar_data(data)
    elif isinstance(data,LidarSnapshot):
        return parse_lidar_data(data)
    elif isinstance(data, THILidarSnapshot):
        return parse_thi_lidar_data(data)

def convert_data(data):
    if isinstance(data,carla.Image):
        return CameraSnapshot(data)
    elif isinstance(data,carla.RadarMeasurement):
        return RadarSnapshot(data)
    elif isinstance(data,carla.LidarMeasurement):
        return LidarSnapshot(data)
    elif isinstance(data, carla.THILidarMeasurement):
        return THILidarSnapshot(data)


def get_data_shape(data):
    if isinstance(data,carla.Image):
        return data.height,data.width
    else:
        return 0,0
class Sensor(Actor):
    def __init__(self, name, **args):
        #print(f"Spawn Sensor {name}: {args} ")
        super().__init__(**args)
        self.name = name
        self.data_queue = Queue()
        self.frame_rate = float(args['options']['sensor_tick']) if args['options'] is not None else 0.0
    
    def get_data_list(self):
        return self.data_queue
    
    def set_actor(self, id):
        super().set_actor(id)
        self.actor.listen(self.add_data)
    
    def spawn_actor(self):
        super().spawn_actor()
        self.actor.listen(self.add_data)

    # def get_last_data(self):
    #     if self.data_list:
    #         return self.data_list[-1]
    #     else:
    #         return None
            
    def add_data(self,data):
        self.data_queue.put((self.actor.parent.get_transform(),convert_data(data)))
        # print(f"Add Data to {self.name} - Length: {len(self.data_list)} - timestamp: {data.timestamp*10e6}")

    def get_transform(self):
        return self.actor.get_transform()
