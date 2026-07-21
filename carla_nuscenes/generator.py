from .client import Client
from .dataset import Dataset
import traceback
import datetime
import numpy as np
from queue import Empty


class Generator:
    def __init__(self,config):
        self.config = config
        self.collect_client = Client(self.config["client"])
        self.max_fov = config['max_fov']
        self.max_dist = config['max_dist']
        self.perception_sensors = ['sensor.camera.rgb','sensor.other.radar','sensor.lidar.ray_cast','sensor.lidar.thi_lidar', 'sensor.lidar.thi_rotational_lidar']
        self.debug=config['debug']
        self.collect_client.debug = self.debug

    def generate_dataset(self,load=False):
        self.dataset = Dataset(**self.config["dataset"],load=load)
        print(self.dataset.data["progress"])
        for sensor in self.config["sensors"]:
            self.dataset.update_sensor(sensor["name"],sensor["modality"])
        for category in self.config["categories"]:
            self.dataset.update_category(category["name"],category["description"])
        for attribute in self.config["attributes"]:
            self.dataset.update_attribute(attribute["name"],category["description"])
        for visibility in self.config["visibility"]:
            self.dataset.update_visibility(visibility["description"],visibility["level"])

        for world_config in self.config["worlds"][self.dataset.data["progress"]["current_world_index"]:]:
            try:
                self.collect_client.generate_world(world_config)
                map_token = self.dataset.update_map(world_config["map_name"],world_config["map_category"])
                for capture_config in world_config["captures"][self.dataset.data["progress"]["current_capture_index"]:]:
                    log_token = self.dataset.update_log(map_token,capture_config["date"],capture_config["time"],
                                            capture_config["timezone"],capture_config["capture_vehicle"],capture_config["location"])
                    for scene_config in capture_config["scenes"][self.dataset.data["progress"]["current_scene_index"]:]:
                        no_scenes = scene_config["count"]
                        for scene_count in range(self.dataset.data["progress"]["current_scene_count"],scene_config["count"]):
                            print(f"Run on Scene {scene_count} of {no_scenes} at {datetime.datetime.now()}")
                            self.dataset.update_scene_count()
                            self.add_one_scene(log_token,scene_config)
                            self.dataset.save()
                        self.dataset.update_scene_index()
                    self.dataset.update_capture_index()
                self.dataset.update_world_index()
            except:
                traceback.print_exc()
            finally:
                self.collect_client.destroy_world()
                
    def add_one_scene(self,log_token,scene_config):
        try:
            calibrated_sensors_token = {}
            samples_data_token = {}
            instances_token = {}
            samples_annotation_token = {}

            self.collect_client.generate_scene(scene_config)
            scene_token = self.dataset.update_scene(log_token,scene_config["description"])

            for instance in self.collect_client.walkers+self.collect_client.vehicles:
                instance_token = self.dataset.update_instance(*self.collect_client.get_instance(scene_token,instance))
                instances_token[instance.get_actor().id] = instance_token
                samples_annotation_token[instance.get_actor().id] = ""
            
            for sensor in self.collect_client.sensors:
                calibrated_sensor_token = self.dataset.update_calibrated_sensor(scene_token,*self.collect_client.get_calibrated_sensor(sensor))
                calibrated_sensors_token[sensor.name] = calibrated_sensor_token
                samples_data_token[sensor.name] = ""

            sample_token = ""
            # print(f"Delta Sim: {self.collect_client.settings.fixed_delta_seconds}, Max. Frame: {int(scene_config['collect_time']/self.collect_client.settings.fixed_delta_seconds)}, Divider {int(scene_config['keyframe_time']/self.collect_client.settings.fixed_delta_seconds)}")
            for frame_count in range(int(scene_config['collect_time']/self.collect_client.settings.fixed_delta_seconds)):
                #print("frame count:",frame_count)
                self.collect_client.tick()
                # snapshot = self.collect_client.world.get_snapshot()
                # print(f"World Tick: {snapshot.frame}, {snapshot.timestamp.elapsed_seconds}")
                if (frame_count+1)%int(scene_config["keyframe_time"]/self.collect_client.settings.fixed_delta_seconds) == 0:
                    sample_token = self.dataset.update_sample(sample_token,scene_token,*self.collect_client.get_sample())
                    #start_time = datetime.datetime.now()
                    principal_lidar_sensor_data = None
                    for sensor in self.collect_client.sensors:
                        if sensor.bp_name in self.perception_sensors:
                            #print(f"{frame_count} Do for Sensor: {sensor.bp_name}, #No. of Samples: {len(sensor.get_data_list())}")
                            #for idx in range(sensor.get_data_list().qsize):
                            idx = 0
                            timeout = .1
                            frame_rate = sensor.frame_rate if sensor.frame_rate != 0.0 else 1 / self.collect_client.settings.fixed_delta_seconds
                            #expected_no_of_samples = (frame_count + 1 )* self.collect_client.settings.fixed_delta_seconds / frame_rate
                            while True:
                                try:
                                    sample_data = sensor.get_data_list().get(timeout=timeout) # Blocks and waits until sensor data is available
                                    idx +=1
                                    if sensor.name == self.collect_client.principal_lidar:
                                        principal_lidar_sensor_data = sample_data
                                    ego_pose_token = self.dataset.update_ego_pose(scene_token,calibrated_sensors_token[sensor.name],*self.collect_client.get_ego_pose(sample_data))
                                    is_key_frame = False
                                    
                                    if scene_config["keyframe_time"] ==  idx * sensor.frame_rate:
                                        is_key_frame = True

                                    samples_data_token[sensor.name] = self.dataset.update_sample_data(samples_data_token[sensor.name],calibrated_sensors_token[sensor.name],sample_token,ego_pose_token,is_key_frame,*self.collect_client.get_sample_data(sample_data))
                                    
                                except Empty:
                                    #print(f"No Sensor data was available for {sensor.name}")
                                    if scene_config["keyframe_time"] <=  idx * sensor.frame_rate:
                                        break
                    #print(f"{frame_count}: Saving the data took: {(datetime.datetime.now() - start_time).total_seconds()}")
                    #start_time = datetime.datetime.now()
                    ego_vehicle = self.collect_client.ego_vehicle.get_actor()
                    num_annos = 0

                    if self.debug:
                        transformed_lidar_points = []
                        if principal_lidar_sensor_data is not None:
                            for data in principal_lidar_sensor_data[1]:
                                transformed_point = principal_lidar_sensor_data[0].transform(data.point)
                                transformed_lidar_points.append((transformed_point.x,transformed_point.y,transformed_point.z))
                        transformed_lidar_points = np.array(transformed_lidar_points)
                    else:
                        transformed_lidar_points=None
        
                    for instance in self.collect_client.walkers+self.collect_client.vehicles:
                        dist = instance.get_actor().get_location().distance(ego_vehicle.get_location())
                        # Check https://carla.readthedocs.io/en/latest/tuto_G_bounding_boxes/
                        forward_vec = ego_vehicle.get_transform().get_forward_vector()
                        ray_ego_target = instance.get_actor().get_transform().location - ego_vehicle.get_transform().location
                        angle = np.rad2deg(forward_vec.get_vector_angle(ray_ego_target)) # Angle in Radians converted to degrees 
                        #print(f"Angle: {angle}, Dist: {dist}, of {instance.get_actor().get_transform().location} with respecto to {ego_vehicle.get_transform().location}")
                        if dist < self.max_dist and abs(angle) <= self.max_fov: # NuScenes only uses object within 54 m distance, we furthermore filter to be within a parameterizable FOV
                            #print(f"Generate Annos: {num_annos} of {len(self.collect_client.walkers+self.collect_client.vehicles)} for Instance within dist")
                            #t1 = datetime.datetime.now()
                            vis = self.collect_client.get_visibility(instance,transformed_lidar_points)
                            if vis > 0:
                                sample_annos = self.collect_client.get_sample_annotation(scene_token,instance,vis,-1)
                                #print(f"{frame_count}: Getting CARLA annos took {(datetime.datetime.now()-t1).total_seconds()}")
                                #t1 = datetime.datetime.now()
                                samples_annotation_token[instance.get_actor().id]  = self.dataset.update_sample_annotation(samples_annotation_token[instance.get_actor().id],sample_token,*sample_annos)
                                #print(f"{frame_count}: Update annos dicts took {(datetime.datetime.now()-t1).total_seconds()}")
                                num_annos += 1
                    #print(f"{frame_count}: Getting Sample Annotations took: {(datetime.datetime.now()-start_time).total_seconds()}")
                    #start_time = datetime.datetime.now()
                    for sensor in self.collect_client.sensors:
                        sensor.get_data_list().queue.clear()
                    #print(f"{frame_count}: Clearing Sensor data took: {(datetime.datetime.now()-start_time).total_seconds()}")
        except:
            traceback.print_exc()
        finally:
            self.collect_client.destroy_scene()
