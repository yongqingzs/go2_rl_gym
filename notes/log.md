## base
```bash
CUDA_VISIBLE_DEVICES=1 python3 /workspace/go2_rl_gym/legged_gym/scripts/train.py --task go1h_cts --headless --num_envs 4096 --max_iterations 30000 --seed 1 --run_name XX

python3 /workspace/go2_rl_gym/legged_gym/scripts/export_policy.py --headless --task go1h_cts --load_run Mar01_07-44-00_ --checkpoint 19500
```

## log
```python
# like himloco mr_lag13
# b1
"""
运行速度很快，能上6级楼梯，但是 hip 姿态奇怪，20000 没有发散
"""

# like himloco mr_lag13，but:
default_joint_angles = { # = target angles [rad] when action = 0.0
    'FL_hip_joint': 0.0,   # [rad]
    'RL_hip_joint': 0.0,   # [rad]
    'FR_hip_joint': -0.0 ,  # [rad]
    'RR_hip_joint': -0.0,   # [rad]

    'FL_thigh_joint': 0.8,     # [rad]
    'RL_thigh_joint': 0.8,   # [rad]
    'FR_thigh_joint': 0.8,     # [rad]
    'RR_thigh_joint': 0.8,   # [rad]

    'FL_calf_joint': -1.5,   # [rad]
    'RL_calf_joint': -1.5,    # [rad]
    'FR_calf_joint': -1.5,  # [rad]
    'RR_calf_joint': -1.5,    # [rad]
}
hip_pos0 = -0.1
# b2
"""
姿态表现良好，但是丧失了横侧向移动的能力(?)
"""

# like b2, but:
hip_pos0 = -0.05
# b3
"""
姿态表现不好，30000 后发散
"""

# like b2, but:
collision = -1.0
dof_pos_limits = -2.0
feet_regulation = -0.05
feet_air_time = 0
# b5
"""
姿态表现不好，19000 尚未到达 6
"""


# like b2, but:
hip_pos0 = -0.05
x_command_hip_regular = -0.5
# b6
"""
35000 后发散，hip_regular 会影响到横向移动
"""


# like b2, but:
hip_pos0 = -0.05
x_command_hip_regular = -0.5
feet_regulation = -0.05
feet_air_time = 0
# b7
"""
22000 尚未到 6
"""
```