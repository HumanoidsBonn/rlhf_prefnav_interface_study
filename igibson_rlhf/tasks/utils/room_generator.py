import numpy as np
import os
from gym_hrl.robots.pedestrian.pedestrian_walker import Pedestrian
from gym_hrl.envs.utils.gen_turtlebot_world2 import *
from jorges_ml_utils.pb import *

config_room = {
    "width": 10,
    "length": 14,
    "height": 3,
    "wall_thickness": 0.2,
    "obstacle_number_range": [4, 6],
    "pillar_number_range": [2, 4],

}
def create_room(_bullet_client, configs):
    obstacles = []
    obstacles.append(
        place_wall(
            _bullet_client,
            sidelength[0],
            0,
            configs["thickness"],
            sidelength[1],
            height,
        )
    )
    obstacles.append(
        place_wall(
            _bullet_client,
            -sidelength[0],
            0,
            configs["thickness"],
            sidelength[1],
            height,
        )
    )
    obstacles.append(
        place_wall(
            _bullet_client,
            0,
            sidelength[1],
            sidelength[0],
            configs["thickness"],
            height,
        )
    )
    obstacles.append(
        place_wall(
            _bullet_client,
            0,
            -sidelength[1],
            sidelength[0],
            configs["thickness"],
            height,
        )
    )

    place_ceiling(
            _bullet_client,
            sidelength[0],
            sidelength[1],
            height,
        )

    return obstacles, sidelength

def place_ceiling(_bullet_client, room_width, room_length, height, thickness=0.2):
    height = height + thickness / 2
    return _bullet_client.createMultiBody(
        0,
        _bullet_client.createCollisionShape(
            shapeType=_bullet_client.GEOM_BOX,
            halfExtents=[room_width, room_length, thickness/2],
            collisionFramePosition=[0, 0, height],
        ),
        _bullet_client.createVisualShape(
            shapeType=_bullet_client.GEOM_BOX,
            halfExtents=[room_width, room_length, thickness/2],
            visualFramePosition=[0, 0, height],
            rgbaColor=[0.5, 0.5, 0.5, 1.0],
        ),
    )

def create_human_env(_bullet_client, configs, sampler_type="training"):
    collision_detector = CollisionDetector(_bullet_client)

    obstacles, sidelength = create_room(_bullet_client, configs)

    # Place clutter behind
    sides_to_fill = 4
    clutter_num = np.random.choice([2, 3, 4], (sides_to_fill,))
    clutter_list = [
        # [place_3chairs, 0.3, -0.5 * np.pi],
        [place_tableNchair, 0.55, 0.5 * np.pi],
        [place_malmNplant, 0.3, 0.5 * np.pi],
        [place_malm, 0.275, 0.5 * np.pi],
        [place_galant, 0.25, 0.5 * np.pi],
        [place_cabinet, 0.2, np.pi],
    ]
    clutter_list_floor = [
        # [place_chair, 0.3, -0.5 * np.pi],
        [place_malm, 0.275, 0.5 * np.pi],
        [place_galant, 0.25, 0.5 * np.pi],
    ]
    shifts = [
        [
            [
                -(0.2 + np.random.random_sample() * 0.55),
                (0.2 + np.random.random_sample() * 0.55),
            ],
            [
                -(0.5 + np.random.random_sample() * 0.25),
                np.random.choice([-1.0, 1.0]) * np.random.random_sample() * 0.1,
                (0.5 + np.random.random_sample() * 0.25),
            ],
            [
                -(0.65 + np.random.random_sample() * 0.1),
                -(0.2 + np.random.random_sample() * 0.05),
                (0.2 + np.random.random_sample() * 0.05),
                (0.65 + np.random.random_sample() * 0.1),
            ],
        ][clutter_num[i] - 2]
        for i in range(sides_to_fill)
    ]
    # POPULATE WALLS
    # ================================================================================
    for i, choice in enumerate(np.random.choice(len(clutter_list), (clutter_num[0],))):
        fn, offset, yaw = clutter_list[choice]
        obstacles += fn(
            _bullet_client,
            sidelength[0] - offset,
            shifts[0][i] * sidelength[1],
            yaw,
        )
    for i, choice in enumerate(np.random.choice(len(clutter_list), (clutter_num[1],))):
        fn, offset, yaw = clutter_list[choice]
        obstacles += fn(
            _bullet_client,
            shifts[1][i] * sidelength[0],
            sidelength[1] - offset,
            yaw + 0.5 * np.pi,
        )
    for i, choice in enumerate(np.random.choice(len(clutter_list), (clutter_num[2],))):
        fn, offset, yaw = clutter_list[choice]
        obstacles += fn(
            _bullet_client,
            -sidelength[0] + offset,
            shifts[2][i] * sidelength[1],
            yaw + np.pi,
        )
    for i, choice in enumerate(np.random.choice(len(clutter_list), (clutter_num[3],))):
        fn, offset, yaw = clutter_list[choice]
        obstacles += fn(
            _bullet_client,
            shifts[3][i] * sidelength[0],
            -sidelength[1] + offset,
            yaw - 0.5 * np.pi,
        )

    collision_detector.add_objects(obstacles)

    # POPULATE FLOOR
    # ================================================================================
    clutter_objects_floor = np.random.choice(
        list(range(configs["floor_clutter"][0], configs["floor_clutter"][1] + 1, 1))
    )
    clutter_objects_floor = int(np.sqrt(sidelength[0] * sidelength[1] * 2.5))
    # print("Clutter:", clutter_objects_floor)

    for _ in range(clutter_objects_floor):
        object_id = np.random.choice(np.arange(len(clutter_list_floor)))
        fn, _, _ = clutter_list_floor[object_id]

        i = 0
        while True:
            x = np.random.choice([-1.0, 1.0]) * np.random.random() * sidelength[0] * 0.75
            y = np.random.choice([-1.0, 1.0]) * np.random.random() * sidelength[1] * 0.75
            if i > 10000:
                raise RuntimeWarning("Error in Placement Method of Room CLutter.")
            i += 1

            # print("Floor sample xy:", x, y)

            yaw = np.random.choice([i * np.pi/2 for i in range(4)])
            obatscle_tmp = fn(
                _bullet_client,
                x,
                y,
                yaw,
            )
            if collision_detector.overlap(obatscle_tmp[0], inflate_distance=0.5):
                _bullet_client.removeBody(obatscle_tmp[0])
                continue
            else:
                collision_detector.add_objects(obatscle_tmp)
                obstacles.append(obatscle_tmp[0])
                break
    # collision_detector.plot_polygons()

    def _sample_helper():
        _size_x = sidelength[0] - configs["distance"][0] - configs["distance"][1]
        _size_y = sidelength[1] - configs["distance"][0] - configs["distance"][1]
        areas = np.array(
            [
                _size_x * _size_y,
                _size_x * configs["distance"][0],
                _size_y * configs["distance"][0],
            ]
        )
        areas /= np.sum(areas)

        choice = np.random.choice([0, 1, 2], p=areas)
        if choice == 0:
            _x = configs["distance"][0] + np.random.random_sample() * _size_x
            _y = configs["distance"][0] + np.random.random_sample() * _size_y
        elif choice == 1:
            _x = configs["distance"][0] + np.random.random_sample() * _size_x
            _y = np.random.random_sample() * configs["distance"][0]
        elif choice == 2:
            _x = np.random.random_sample() * configs["distance"][0]
            _y = configs["distance"][0] + np.random.random_sample() * _size_y
        return np.random.choice([-1.0, 1.0]) * _x, np.random.choice([-1.0, 1.0]) * _y

    def _normalize_angle(angle):
        if angle < 0.0:
            angle += 2.0 * np.pi
        return angle

    def _angle_helper(x, y):
        sign_x, sign_y = x / np.abs(x), y / np.abs(y)
        inner_corners = [
            [sign_x * configs["distance"][0], -sign_y * configs["distance"][0]],
            [-sign_x * configs["distance"][0], sign_y * configs["distance"][0]],
        ]
        outer_corners = [
            [
                sign_x * (sidelength[0] - configs["distance"][1]),
                -sign_y * (sidelength[1] - configs["distance"][1]),
            ],
            [
                -sign_x * (sidelength[0] - configs["distance"][1]),
                sign_y * (sidelength[1] - configs["distance"][1]),
            ],
        ]
        regions = []
        weights = []
        for i in range(2):
            inner_angle = np.arctan2(
                inner_corners[i][1] - y,
                inner_corners[i][0] - x,
            )
            outer_angle = np.arctan2(
                outer_corners[i][1] - y,
                outer_corners[i][0] - x,
            )
            if inner_angle * outer_angle < -0.25:
                regions.append(
                    [
                        max(inner_angle, outer_angle),
                        min(inner_angle, outer_angle) + 2.0 * np.pi,
                    ]
                )
            else:
                regions.append(
                    [
                        min(inner_angle, outer_angle),
                        max(inner_angle, outer_angle),
                    ]
                )
            weights.append(regions[-1][1] - regions[-1][0])
        weights = np.array(weights)
        weights /= np.sum(weights)
        angles = np.array(
            [
                region[0] + np.random.random_sample() * (region[1] - region[0])
                for region in regions
            ]
        )
        return _normalize_angle(np.random.choice(angles, p=weights))

    def human_sampler():
        _list = []
        for _ in range(
            np.random.choice(
                list(range(configs["humanoids"][0], configs["humanoids"][1] + 1, 1))
            )
        ):
            _x, _y = _sample_helper()
            _list += [[_x, _y, _angle_helper(_x, _y)]]
        return _list

    def robot_sampler():
        return 0.0, 0.0, np.random.random_sample() * 2.0 * np.pi

    return obstacles, human_sampler, robot_sampler



# Environment parameter
configs_all = {"thickness": 0.05}
configs_2021Jul21 = [
    {  # humans
        "limit": np.array([[7.0, 7.0], [9.0, 9.0]]),
        "distance": [1.0, 1.5],
        "humanoids": [5, 10],
        "floor_clutter": [2, 4],
        "height": [2.5, 3],
        **configs_all,
    },
]
funcs_2021Jul21 = [create_human_env]


def create_world(
    _bullet_client,
    _index,
    sampler_type="training",
):
    return funcs_2021Jul21[_index](
        _bullet_client, configs_2021Jul21[_index], sampler_type
    )
