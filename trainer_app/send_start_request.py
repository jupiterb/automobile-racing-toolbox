import grequests 
import time 
import logging 

data = {
    "game_config": {
        "game_id": "TMNF",
        "process_name": "Trackmania Nations Forever",
        "window_size": [
            750,
            600
        ],
        "observation_frame": {
            "top": 0.475,
            "bottom": 0.9125,
            "left": 0.01,
            "right": 0.99
        },
        "discrete_actions_mapping": {
            "FORWARD": "up",
            "BREAK": "down",
            "RIGHT": "right",
            "LEFT": "left"
        },
        "continous_actions_mapping": {
            "FORWARD": "XUSB_GAMEPAD_A"
        },
        "ocrs": {
            "instances": {
                "speed": [
                    {
                        "top": 0.945,
                        "bottom": 0.9875,
                        "left": 0.918,
                        "right": 0.9825
                    },
                    {
                        "threshold": 190,
                        "max_digits": 3,
                        "segemnts_definitions": {
                            "0": {
                                "top": 0.0,
                                "bottom": 0.09,
                                "left": 0.42,
                                "right": 0.6
                            },
                            "1": {
                                "top": 0.15,
                                "bottom": 0.28,
                                "left": 0.14,
                                "right": 0.28
                            },
                            "2": {
                                "top": 0.15,
                                "bottom": 0.28,
                                "left": 0.85,
                                "right": 1.0
                            },
                            "3": {
                                "top": 0.38,
                                "bottom": 0.5,
                                "left": 0.42,
                                "right": 0.6
                            },
                            "4": {
                                "top": 0.58,
                                "bottom": 0.73,
                                "left": 0.14,
                                "right": 0.28
                            },
                            "5": {
                                "top": 0.58,
                                "bottom": 0.73,
                                "left": 0.85,
                                "right": 1.0
                            },
                            "6": {
                                "top": 0.82,
                                "bottom": 0.94,
                                "left": 0.42,
                                "right": 0.6
                            }
                        }
                    }
                ]
            }
        },
        "reset_seconds": 3,
        "reset_keys_sequence": [
            "enter"
        ],
        "reset_gamepad_sequence": [
            "XUSB_GAMEPAD_X"
        ],
        "frequency_per_second": 8
    },
    "env_config": {
        "reward_config": {
            "speed_diff_thresh": 3,
            "memory_length": 2,
            "off_track_termination": True,
            "clip_range": [
                -300.0,
                300.0
            ],
            "baseline": 20.0,
            "scale": 300.0
        },
        "action_config": {
            "available_actions": {
                "FORWARD": [
                    0,
                    1,
                    2
                ],
                "BREAK": [],
                "RIGHT": [
                    1,
                    3
                ],
                "LEFT": [
                    2,
                    4
                ]
            }
        },
        "observation_config": {
            "frame": {
                "top": 0.475,
                "bottom": 0.9125,
                "left": 0.01,
                "right": 0.99
            },
            "shape": [
                60,
                60
            ],
            "stack_size": 4,
            "lidar_config": None,
            "track_segmentation_config": None
        },
        "max_episode_length": 1000
    },
    "training_config": {
        "num_rollout_workers": 1,
        "rollout_fragment_length": 128,
        "compress_observations": False,
        "gamma": 0.99,
        "lr": 0.0001,
        "train_batch_size": 256,
        "max_iterations": 200,
        "stop_reward": 1000,
        "log_level": "INFO",
        "model": {
            "fcnet_hiddens": [
                100,
                256
            ],
            "fcnet_activation": "relu",
            "conv_filters": [
                [
                    32,
                    [8, 8],
                    4
                ],
                [
                    64,
                    [4, 4],
                    2
                ],
                [
                    64,
                    [3,3],
                    1
                ],
                [
                    64,
                    [11, 11],
                    1
                ]
            ],
            "conv_activation": "relu"
        },
        "algorithm": {
            "v_min": -100.0,
            "v_max": 100.0,
            "dueling": True,
            "double_q": True,
            "hiddens": [
                256
            ],
            "replay_buffer_config": {
                "capacity": 50000
            }
        }
    },
    "wandb_api_key": "ff2b2b54b5862102dcdd75a25fd3b8f45e028305"
}

# resp = requests.put("http://localhost:8000/start", json=data)
# print(resp)

if __name__ == "__main__":
    urls = ["http://192.168.1.14:8000/worker/probe"] * 10 

    rs = (grequests.get(u) for u in urls)

    time.sleep(10)
    logging.error("start")
    responses = grequests.map(rs)

    logging.error(responses)