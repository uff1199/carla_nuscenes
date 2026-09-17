from pyquaternion import Quaternion
import numpy as np
import json
import argparse
from nuscenes.nuscenes import NuScenes
from pathlib import Path
import os

from tqdm import tqdm

def check_sample(nusc, scene, sensor_type, data_root =Path(), duration=20, key_frame=0.5):

    if scene['nbr_samples'] != duration/key_frame:
        return None
    sample = nusc.get('sample',scene['first_sample_token'])
    
    while True:
        try:
            if sensor_type not in sample["data"]:
                break

            sample_data = nusc.get("sample_data", sample["data"][sensor_type]) # Sample Data itself has also prev and next flag
            nusc.get("ego_pose", sample_data["ego_pose_token"])

            while True: 
                if sample_data['next'] == "":
                    break
                if not os.path.exists(data_root / sample_data['filename']):
                    print(f"Fil: {data_root / sample_data['filename']} does not exist")
                    return None
                sample_data = nusc.get("sample_data", sample_data['next'])

            if sample["token"] == scene["last_sample_token"]:
                return scene

            if sample["next"] == "":
                break

            sample = nusc.get("sample", sample["next"])

        except KeyError as e:
            print(e)
            break

    return None


def main(args):

    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=True)

    dataroot = args.data_root

    processed_annos = []
    removed_annos = {}

    for sensor in ['CAM_FRONT','LIDAR_TOP', 'LIDAR_TOP_DEFAULT','LIDAR_TOP_RAYCAST','LIDAR_TOP_SPINNING_KITTI']:
        valid_scenes = []
        for scene in nusc.scene:
            valid_scene = check_sample(nusc,scene,sensor,Path(dataroot))
            if valid_scene is not None:
                valid_scenes.append(valid_scene)
        print(f"Sensor: {sensor} - No. of Valid Scenes: {len(valid_scenes)}, No. of Original Scenes: {len(nusc.scene)}")


    files = {'annotations':'sample_annotation.json',
         'ego_pose' : 'ego_pose.json',
         'instance':'instance.json',
         'sample_data':'sample_data.json',
         'sample': 'sample.json',
         'scene': 'scene.json'}

    file_tokens = {'annotations':{},
                   'ego_pose': {},
                    'instance':{},
                    'sample_data':{},
                    'sample': {},
                    'scene':{}}

    dataset_version=args.version

    for file in files:
        with open(Path(dataroot) / f'{dataset_version}/{files[file]}','r') as f:
            for element in json.load(f):
                #print(element)
                file_tokens[file].update({element['token'] : element})


    scene_tokens = [scene['token'] for scene in valid_scenes]
    sample_tokens = []
    sample_data_tokens = []
    annotation_tokens = []
    instance_tokens = []
    print(f"No of Instances: {len(file_tokens['instance'])}, Annotations: {len(file_tokens['annotations'])}, Samples: {len(file_tokens['sample_data'])}, Pose: {len(file_tokens['ego_pose'])} ")
    
    for scene in nusc.scene:
        if not scene['token'] in scene_tokens:
            first_sample_token = scene['first_sample_token']
            if first_sample_token != '':
                sample = nusc.get('sample',first_sample_token)
                while True:
                    # Delete Sensor Data
                    for sensor in sample['data']:
                        sample_data = nusc.get('sample_data',sample['data'][sensor])
                        while True:
                            ego_pose_token = sample_data['ego_pose_token']
                            # Delete Ego Pose + Sample Data
                            if ego_pose_token in file_tokens['ego_pose']:
                                file_tokens['ego_pose'].pop(ego_pose_token)
        
                            if sample_data['token'] in file_tokens['sample_data']:
                                file_tokens['sample_data'].pop(sample_data['token'])
                            if sample_data['next'] != '':
                                sample_data = nusc.get('sample_data',sample_data['next'])
                            else:
                                break                  
    
                # Delete Annotations
                for ann in sample['anns']:
                    file_tokens['annotations'].pop(ann)
    
                if sample['next'] != '':
                    sample = nusc.get('sample',sample['next'])
                else:
                    break
                file_tokens['sample'].pop(sample['token'])
            file_tokens['scene'].pop(scene['token'])
    
    # Filter Instances
    
    for instance in nusc.instance:
        if not instance['first_annotation_token'] in file_tokens['annotations'] and not instance['last_annotation_token'] in file_tokens['annotations']:
            file_tokens['instance'].pop(instance['token'])
    
    
    print(f"No of Instances: {len(file_tokens['instance'])}, Annotations: {len(file_tokens['annotations'])}, Samples: {len(file_tokens['sample_data'])}, Pose: {len(file_tokens['ego_pose'])} ")
    
    for sample in nusc.sample:
        try:
            nusc.get('scene',sample['scene_token'])
        except KeyError:
            print(f"sample token {sample['token']} not found")
            file_tokens['sample'].pop(sample['token'])


    # save Files
    for file in files:
        with open(Path(dataroot) / f'{dataset_version}/{files[file]}','w') as f:
            store_dict = []
            for item in file_tokens[file]:
                store_dict.append(file_tokens[file][item])
    
            print(f"No. of elements in {file}: {len(store_dict)}")
    
            json.dump(store_dict,f,indent=4)
            print(f"Saved-{f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root"
        )
    parser.add_argument("" \
    "--version",
    default="v1.16-trainval")
    parser.add_argument(
        "--output",
        default="./"
    )
    
    args = parser.parse_args()


    
    main(args)