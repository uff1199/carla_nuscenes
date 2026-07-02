import carla
from .sensor import *
from .vehicle import Vehicle
from .walker import Walker
import math
from .utils import generate_token,get_nuscenes_rt,get_intrinsic,transform_timestamp,clamp
import random
import time


'''
        NuScenes
        NameMapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
        }
        DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
        }
        Current CARLA Blueprints
        see actor_list.txt
'''

def get_category(bp):
    # Helper to check if a keyword is in the blueprint ID
    bp_id = bp.id
    
    # 1. Check for specific vehicle types
    if 'vehicle' in bp_id:
        # Bicycles
        if 'diamondback' in bp_id or 'gazelle' in bp_id:
            return 'vehicle.bicycle'
        
        # Motorcycles
        if 'kawasaki' in bp_id or 'yamaha' in bp_id or 'harley' in bp_id or 'vespa' in bp_id:
            return 'vehicle.motorcycle'
        
        # Trucks
        if 'firetruck' in bp_id or 'ambulance' in bp_id or 'carlacola' in bp_id:
            return 'vehicle.truck'
        
        # Buses
        if 'sprinter' in bp_id or 't2' in bp_id or 'fusorosa' in bp_id:
            return 'vehicle.bus.rigid'
        
        # Construction Vehicles
        if 'charger_police' in bp_id:
            return 'vehicle.construction' # Treating police car as construction for this mapping or specific logic
        
        # Default to car for other vehicles
        return 'vehicle.car'

    # 2. Check for pedestrians
    elif 'walker' in bp_id:
        return 'human.pedestrian.adult'

    # 3. Check for static props (Barriers and Cones)
    elif 'static' in bp_id:
        if 'trafficcone' in bp_id or 'constructioncone' in bp_id:
            return 'movable_object.trafficcone'
        if 'streetbarrier' in bp_id:
            return 'movable_object.barrier'
        
    return None

def get_attribute(bp):
    category = get_category(bp)
    
    if category is None:
        return None

    # Map NuScenes categories to default attributes
    if category == 'vehicle.car':
        return ['vehicle.parked']
    elif category == 'vehicle.truck':
        return ['vehicle.parked']
    elif category == 'vehicle.bus.bendy':
        return ['vehicle.moving']
    elif category == 'vehicle.motorcycle':
        return ['cycle.without_rider']
    elif category == 'vehicle.bicycle':
        return ['cycle.without_rider']
    elif category == 'human.pedestrian.adult':
        return ['pedestrian.moving']
    elif category == 'movable_object.trafficcone':
        return ['']
    elif category == 'movable_object.barrier':
        return ['']
    
    # Fallback for other mapped categories (e.g. construction)
    return ['vehicle.parked']


class Client:
    def __init__(self,client_config):
        self.client = carla.Client(client_config["host"],client_config["port"])
        self.client.set_timeout(client_config["time_out"])

    def generate_world(self,world_config):
        print("generate world start!")
        self.client.load_world_if_different(world_config["map_name"])
        self.world = self.client.get_world()
        self.original_settings = self.world.get_settings()
        self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        self.ego_vehicle = None
        self.sensors = None
        self.vehicles = None
        self.walkers = None


        # Here the assignments of blueprint ids to categories happens
        #get_category = lambda bp: "vehicle.car" if bp.id.split(".")[0] == "vehicle" else "human.pedestrian.adult" if bp.id.split(".")[0] == "walker" else None
        self.category_dict = {bp.id: get_category(bp) for bp in self.world.get_blueprint_library()}
        #get_attribute = lambda bp: ["vehicle.moving"] if bp.id.split(".")[0] == "vehicle" else ["pedestrian.moving"] if bp.id.split(".")[0] == "walker" else None
        self.attribute_dict = {bp.id: get_attribute(bp) for bp in self.world.get_blueprint_library()}

        self.trafficmanager = self.client.get_trafficmanager()
        self.trafficmanager.set_synchronous_mode(True)
        self.trafficmanager.set_respawn_dormant_vehicles(True)
        self.settings = carla.WorldSettings(**world_config["settings"])
        self.settings.synchronous_mode = True
        self.settings.no_rendering_mode = False
        self.world.apply_settings(self.settings)
        self.world.set_pedestrians_cross_factor(1)
        print("generate world success!")

    def generate_scene(self,scene_config):
        print("generate scene start!")
        if scene_config["custom"]:
            self.generate_custom_scene(scene_config)
        else:
            self.generate_random_scene(scene_config)
        print("generate scene success!")

    def generate_custom_scene(self,scene_config):
        
        if scene_config["weather_mode"] == "custom":
            self.weather = carla.WeatherParameters(**scene_config["weather"])
        else:
            self.weather = getattr(carla.WeatherParameters, scene_config["weather_mode"])
        
        self.world.set_weather(self.weather)
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        self.ego_vehicle = Vehicle(world=self.world,**scene_config["ego_vehicle"])
        self.ego_vehicle.blueprint.set_attribute('role_name', 'hero')
        self.ego_vehicle.spawn_actor()
        self.ego_vehicle.get_actor().set_autopilot()
        self.trafficmanager.ignore_lights_percentage(self.ego_vehicle.get_actor(),10)
        self.trafficmanager.ignore_signs_percentage(self.ego_vehicle.get_actor(),10)
        self.trafficmanager.ignore_vehicles_percentage(self.ego_vehicle.get_actor(),10)
        self.trafficmanager.distance_to_leading_vehicle(self.ego_vehicle.get_actor(),5.0)
        self.trafficmanager.vehicle_percentage_speed_difference(self.ego_vehicle.get_actor(),-20)
        self.trafficmanager.auto_lane_change(self.ego_vehicle.get_actor(), True)

        self.vehicles = [Vehicle(world=self.world,**vehicle_config) for vehicle_config in scene_config["vehicles"]]
        vehicles_batch = [SpawnActor(vehicle.blueprint,vehicle.transform)
                            .then(SetAutopilot(FutureActor, True, self.trafficmanager.get_port())) 
                            for vehicle in self.vehicles]
        print("Spawn Vehicles")
        for i,response in enumerate(self.client.apply_batch_sync(vehicles_batch)):
            if not response.error:
                self.vehicles[i].set_actor(response.actor_id)
            else:
                print(response.error)
        self.vehicles = list(filter(lambda vehicle:vehicle.get_actor(),self.vehicles))

        for vehicle in self.vehicles:
            self.trafficmanager.set_path(vehicle.get_actor(),vehicle.path)
        print("Spawn Walkers")
        self.walkers = [Walker(world=self.world,**walker_config) for walker_config in scene_config["walkers"]]
        walkers_batch = [SpawnActor(walker.blueprint,walker.transform) for walker in self.walkers]
        for i,response in enumerate(self.client.apply_batch_sync(walkers_batch)):
            if not response.error:
                self.walkers[i].set_actor(response.actor_id)
            else:
                print(response.error)
        self.walkers = list(filter(lambda walker:walker.get_actor(),self.walkers))

        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        walkers_controller_batch = [SpawnActor(walker_controller_bp,carla.Transform(),walker.get_actor()) for walker in self.walkers]
        for i,response in enumerate(self.client.apply_batch_sync(walkers_controller_batch)):
                    if not response.error:
                        self.walkers[i].set_controller(response.actor_id)
                    else:
                        print(response.error)
        self.world.tick()
        for walker in self.walkers:
            walker.start()

        
        ## Wait for a few simulation steps until the vehicles have stopped falling down
        print("Wait for Actors to settle")
        for i in range(0,70):
            self.world.tick()
            time.sleep(0.05)

        print("Spawn Sensors")
        self.sensors = [Sensor(world=self.world, attach_to=self.ego_vehicle.get_actor(), **sensor_config) for sensor_config in scene_config["calibrated_sensors"]["sensors"]]
        sensors_batch = [SpawnActor(sensor.blueprint,sensor.transform,sensor.attach_to) for sensor in self.sensors]
        for i,response in enumerate(self.client.apply_batch_sync(sensors_batch)):
            if not response.error:
                self.sensors[i].set_actor(response.actor_id)
            else:
                print(response.error)
        self.sensors = list(filter(lambda sensor:sensor.get_actor(),self.sensors))

    def tick(self):
        self.world.tick()

    def generate_random_scene(self,scene_config):
        print("generate random scene start!")
        self.weather = carla.WeatherParameters(**self.get_random_weather())
        self.world.set_weather(self.weather)


        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        
        
        ego_bp_name=scene_config["ego_bp_name"]
        ego_location={attr:getattr(spawn_points[0].location,attr) for attr in ["x","y","z"]}
        ego_rotation={attr:getattr(spawn_points[0].rotation,attr) for attr in ["yaw","pitch","roll"]}
        self.ego_vehicle = Vehicle(world=self.world,bp_name=ego_bp_name,location=ego_location,rotation=ego_rotation)
        self.ego_vehicle.blueprint.set_attribute('role_name', 'hero')
        self.ego_vehicle.spawn_actor()
        self.ego_vehicle.get_actor().set_autopilot()
        self.trafficmanager.ignore_lights_percentage(self.ego_vehicle.get_actor(),10)
        self.trafficmanager.ignore_signs_percentage(self.ego_vehicle.get_actor(),10)
        self.trafficmanager.ignore_vehicles_percentage(self.ego_vehicle.get_actor(),10)
        self.trafficmanager.distance_to_leading_vehicle(self.ego_vehicle.get_actor(),5.0)
        self.trafficmanager.vehicle_percentage_speed_difference(self.ego_vehicle.get_actor(),-20)
        self.trafficmanager.auto_lane_change(self.ego_vehicle.get_actor(), True)
        print("Spawn Vehicles")
        vehicle_bp_list = self.world.get_blueprint_library().filter("vehicle")
        self.vehicles = []
        for spawn_point in spawn_points[1:random.randint(1,len(spawn_points))]:
            location = {attr:getattr(spawn_point.location,attr) for attr in ["x","y","z"]}
            rotation = {attr:getattr(spawn_point.rotation,attr) for attr in ["yaw","pitch","roll"]}
            bp_name = random.choice(vehicle_bp_list).id
            self.vehicles.append(Vehicle(world=self.world,bp_name=bp_name,location=location,rotation=rotation))
        vehicles_batch = [SpawnActor(vehicle.blueprint,vehicle.transform)
                            .then(SetAutopilot(FutureActor, True, self.trafficmanager.get_port())) 
                            for vehicle in self.vehicles]

        for i,response in enumerate(self.client.apply_batch_sync(vehicles_batch)):
            if not response.error:
                self.vehicles[i].set_actor(response.actor_id)
            else:
                print(response.error)
        self.vehicles = list(filter(lambda vehicle:vehicle.get_actor(),self.vehicles))

        # Disable spawining of dummy target + ISO target by filtering out
        walker_bp_list = self.world.get_blueprint_library().filter("*.pedestrian.[0-9][0-9][0-5][0-2]")
        print(f"Found: {len(walker_bp_list)} walkers")      
        self.walkers = []
        for i in range(random.randint(len(spawn_points),len(spawn_points)*2)):
            spawn = self.world.get_random_location_from_navigation()
            if spawn != None:
                bp_name=random.choice(walker_bp_list).id
                spawn_location = {attr:getattr(spawn,attr) for attr in ["x","y","z"]}
                destination=self.world.get_random_location_from_navigation()
                destination_location={attr:getattr(destination,attr) for attr in ["x","y","z"]}
                rotation = {"yaw":random.random()*360,"pitch":random.random()*360,"roll":random.random()*360}
                self.walkers.append(Walker(world=self.world,location=spawn_location,rotation=rotation,destination=destination_location,bp_name=bp_name))
            else:
                print("walker generate fail")
        walkers_batch = [SpawnActor(walker.blueprint,walker.transform) for walker in self.walkers]
        for i,response in enumerate(self.client.apply_batch_sync(walkers_batch)):
            if not response.error:
                self.walkers[i].set_actor(response.actor_id)
            else:
                print(response.error)
        self.walkers = list(filter(lambda walker:walker.get_actor(),self.walkers))
        print("Spawn Walkers")
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        walkers_controller_batch = [SpawnActor(walker_controller_bp,carla.Transform(),walker.get_actor()) for walker in self.walkers]
        for i,response in enumerate(self.client.apply_batch_sync(walkers_controller_batch)):
                    if not response.error:
                        self.walkers[i].set_controller(response.actor_id)
                    else:
                        print(response.error)
        self.world.tick()
        for walker in self.walkers:
            walker.start()

        ## Wait for a few simulation steps until the vehicles have stopped falling down
        print("Wait for Actors to settle")
        for i in range(0,70):
            self.world.tick()
            time.sleep(0.05)

        print("Spawn Sensors")
        self.sensors = [Sensor(world=self.world, attach_to=self.ego_vehicle.get_actor(), **sensor_config) for sensor_config in scene_config["calibrated_sensors"]["sensors"]]
        sensors_batch = [SpawnActor(sensor.blueprint,sensor.transform,sensor.attach_to) for sensor in self.sensors]
        for i,response in enumerate(self.client.apply_batch_sync(sensors_batch)):
            if not response.error:
                self.sensors[i].set_actor(response.actor_id)
            else:
                print(response.error)
        self.sensors = list(filter(lambda sensor:sensor.get_actor(),self.sensors))
        print("generate random scene success!")        

    def destroy_scene(self):
        if self.walkers is not None:
            for walker in self.walkers:
                walker.controller.stop()
                walker.destroy()
        if self.vehicles is not None:
            for vehicle in self.vehicles:
                vehicle.destroy()
        if self.sensors is not None:
            for sensor in self.sensors:
                sensor.destroy()
        if self.ego_vehicle is not None:
            self.ego_vehicle.destroy()


    def destroy_world(self):
        self.trafficmanager.set_synchronous_mode(False)
        self.ego_vehicle = None
        self.sensors = None
        self.vehicles = None
        self.walkers = None
        self.world.apply_settings(self.original_settings)

    def get_calibrated_sensor(self,sensor):
        sensor_token = generate_token("sensor",sensor.name)
        channel = sensor.name
        if sensor.bp_name == "sensor.camera.rgb":
            intrinsic = get_intrinsic(float(sensor.get_actor().attributes["fov"]),
                            float(sensor.get_actor().attributes["image_size_x"]),
                            float(sensor.get_actor().attributes["image_size_y"])).tolist()
            rotation,translation = get_nuscenes_rt(sensor.transform,"zxy")
        else:
            intrinsic = []
            rotation,translation = get_nuscenes_rt(sensor.transform)
        return sensor_token,channel,translation,rotation,intrinsic
        
    def get_ego_pose(self,sample_data):
        timestamp = transform_timestamp(sample_data[1].timestamp)
        rotation,translation = get_nuscenes_rt(sample_data[0])
        return timestamp,translation,rotation
    
    def get_sample_data(self,sample_data):
        height = 0
        width = 0
        if isinstance(sample_data[1],carla.Image):
            height = sample_data[1].height
            width = sample_data[1].width
        return sample_data,height,width

    def get_sample(self):
        return (transform_timestamp(self.world.get_snapshot().timestamp.elapsed_seconds),)

    def get_instance(self,scene_token,instance):
        category_token = generate_token("category",self.category_dict[instance.blueprint.id])
        id = hash((scene_token,instance.get_actor().id))
        return category_token,id

    def get_sample_annotation(self,scene_token,instance,visibility=-1,no_pts=1):
        instance_token = generate_token("instance",hash((scene_token,instance.get_actor().id)))
        if visibility < 0:
            visibility_token = str(self.get_visibility(instance))
        else:
            visibility_token = str(visibility)
        
        attribute_tokens = [generate_token("attribute",attribute) for attribute in self.get_attributes(instance)]
        # get_nuscenes_rt transfroms from left hand to right hand coordinate system
        rotation,translation = get_nuscenes_rt(instance.get_transform())
        size = [instance.get_size().y,instance.get_size().x,instance.get_size().z]#xyz to whl
        num_lidar_pts = 0
        num_radar_pts = 0
        if no_pts>0:
            for sensor in self.sensors:
                if ((sensor.bp_name == 'sensor.lidar.ray_cast') or (sensor.bp_name == 'sensor.lidar.thi_lidar')):
                    num_lidar_pts += self.get_num_lidar_pts(instance,sensor.get_last_data(),sensor.get_transform())
                elif sensor.bp_name == 'sensor.other.radar':
                    num_radar_pts += self.get_num_radar_pts(instance,sensor.get_last_data(),sensor.get_transform())
        return instance_token,visibility_token,attribute_tokens,translation,rotation,size,num_lidar_pts,num_radar_pts

    def get_visibility(self,instance):
        max_visible_point_count = 0
        for sensor in self.sensors:
            if ((sensor.bp_name == 'sensor.lidar.ray_cast') or (sensor.bp_name == 'sensor.lidar.thi_lidar')):
                ego_position = sensor.get_transform().location
                ego_position.z += self.ego_vehicle.get_size().z*0.5
                instance_position = instance.get_transform().location
                visible_point_count1 = 0
                visible_point_count2 = 0
                for i in range(5):
                    size = instance.get_size()
                    size.z = 0
                    check_point = instance_position-(i-2)*size*0.5
                    ray_points =  self.world.cast_ray(ego_position,check_point)
                    points = list(filter(lambda point:not self.ego_vehicle.get_actor().bounding_box.contains(point.location,self.ego_vehicle.get_actor().get_transform()) 
                                        and not instance.get_actor().bounding_box.contains(point.location,instance.get_actor().get_transform()) 
                                        and point.label is not carla.libcarla.CityObjectLabel.NONE,ray_points))
                    if not points:
                        visible_point_count1+=1
                    size.x = -size.x
                    check_point = instance_position-(i-2)*size*0.5
                    ray_points =  self.world.cast_ray(ego_position,check_point)
                    points = list(filter(lambda point:not self.ego_vehicle.get_actor().bounding_box.contains(point.location,self.ego_vehicle.get_actor().get_transform()) 
                                        and not instance.get_actor().bounding_box.contains(point.location,instance.get_actor().get_transform()) 
                                        and point.label is not carla.libcarla.CityObjectLabel.NONE,ray_points))
                    if not points:
                        visible_point_count2+=1
                if max(visible_point_count1,visible_point_count2)>max_visible_point_count:
                    max_visible_point_count = max(visible_point_count1,visible_point_count2)
        visibility_dict = {0:0,1:1,2:1,3:2,4:3,5:4}
        return visibility_dict[max_visible_point_count]

    def get_attributes(self,instance):
        return self.attribute_dict[instance.bp_name]
    

    def get_num_lidar_pts(self,instance,lidar_data,lidar_transform):
        #print("Check No. of Lidar Points on Annotation")
        num_lidar_pts = 0
        if lidar_data is not None:
            for data in lidar_data[1]:
                point = lidar_transform.transform(data.point)
                if instance.get_actor().bounding_box.contains(point,instance.get_actor().get_transform()):
                    num_lidar_pts+=1
        #print("Check No. of Lidar Points on Annotation - finished")
        return num_lidar_pts

    def get_num_radar_pts(self,instance,radar_data,radar_transform):
        num_radar_pts = 0
        if radar_data is not None:
            for data in radar_data[1]:
                point = carla.Location(data.depth*math.cos(data.altitude)*math.cos(data.azimuth),
                        data.depth*math.sin(data.altitude)*math.cos(data.azimuth),
                        data.depth*math.sin(data.azimuth)
                        )
                point = radar_transform.transform(point)
                if instance.get_actor().bounding_box.contains(point,instance.get_actor().get_transform()):
                    num_radar_pts+=1
        return num_radar_pts

    def get_random_weather(self):
        weather_param = {
            "cloudiness":clamp(random.gauss(0,30)),
            #"sun_azimuth_angle":random.random()*360,
            #"sun_altitude_angle":random.random()*120-30,
            #"precipitation":clamp(random.gauss(0,30)),
            #"precipitation_deposits":clamp(random.gauss(0,30)),
            "wind_intensity":random.random()*100,
            #"fog_density":clamp(random.gauss(0,30)),
            #"fog_distance":random.random()*100,
            #"wetness":clamp(random.gauss(0,1)),
            #"fog_falloff":random.random()*5,
            "scattering_intensity":max(random.random()*2-1,0),
            "mie_scattering_scale":max(random.random()*2-1,0),
            "rayleigh_scattering_scale":max(random.random()*2-1,0),
            #"dust_storm":clamp(random.gauss(0,30))
        }
        return weather_param

    
