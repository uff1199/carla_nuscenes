#! /usr/bin/env python
 
 
import sys
import numpy as np
from pathlib import Path
import argparse
from matplotlib import pyplot as plt
import carla
import json
import time

def main(world_name,timeout):
    client = carla.Client()

    client.set_timeout(timeout)

    world = client.load_world_if_different(world_name)

    if world is None:
        world = client.get_world()

    spawns_walkers = set()
    length = 0

    try:
        while True:
            
            spawn = world.get_random_location_from_navigation()
            if spawn != None:
                spawns_walkers.add(spawn)
                if len(spawns_walkers) > length:
                    print(f"New Spawnpoint added - {time.time_ns()}")
                    length +=1
    
    finally:
        with open(world_name+'_walker_spawn.json','w') as f:
            spawn_points = []
            for point in spawns_walkers:
                spawn_points.append({'x':point.x, 'y':point.y, 'z':point.z})

            json.dump(spawn_points,f,indent=4)
            
 
 
if __name__ == '__main__':

    argparser = argparse.ArgumentParser()


    argparser.add_argument(
        "--world",
        type=str
        )

    argparser.add_argument(
        "--timeout",
        type=float,
        default=10.0)
    
    

    args = argparser.parse_args()

    main(args.world, args.timeout)


