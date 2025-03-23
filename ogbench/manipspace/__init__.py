from gymnasium.envs.registration import register

visual_dict = dict(
    ob_type='pixels',
    width=64,
    height=64,
    visualize_info=False,
)
cube_singletask_dict = dict(
    permute_blocks=False,
)
scene_singletask_dict = dict(
    permute_blocks=False,
)
puzzle_singletask_dict = dict()

# Environments for offline goal-conditioned RL.
register(
    id='cube-single-v0',
    entry_point='ogbench.manipspace.envs.cube_env:CubeEnv',
    max_episode_steps=200,
    kwargs=dict(
        env_type='single',
    ),
)
register(
    id='visual-cube-single-v0',
    entry_point='ogbench.manipspace.envs.cube_env:CubeEnv',
    max_episode_steps=200,
    kwargs=dict(
        env_type='single',
        **visual_dict,
    ),
)
register(
    id='cube-double-v0',
    entry_point='ogbench.manipspace.envs.cube_env:CubeEnv',
    max_episode_steps=500,
    kwargs=dict(
        env_type='double',
    ),
)
register(
    id='visual-cube-double-v0',
    entry_point='ogbench.manipspace.envs.cube_env:CubeEnv',
    max_episode_steps=500,
    kwargs=dict(
        env_type='double',
        **visual_dict,
    ),
)
register(
    id='cube-triple-v0',
    entry_point='ogbench.manipspace.envs.cube_env:CubeEnv',
    max_episode_steps=1000,
    kwargs=dict(
        env_type='triple',
    ),
)
register(
    id='visual-cube-triple-v0',
    entry_point='ogbench.manipspace.envs.cube_env:CubeEnv',
    max_episode_steps=1000,
    kwargs=dict(
        env_type='triple',
        **visual_dict,
    ),
)
register(
    id='cube-quadruple-v0',
    entry_point='ogbench.manipspace.envs.cube_env:CubeEnv',
    max_episode_steps=1000,
    kwargs=dict(
        env_type='quadruple',
    ),
)
register(
    id='visual-cube-quadruple-v0',
    entry_point='ogbench.manipspace.envs.cube_env:CubeEnv',
    max_episode_steps=1000,
    kwargs=dict(
        env_type='quadruple',
        **visual_dict,
    ),
)

register(
    id='scene-v0',
    entry_point='ogbench.manipspace.envs.scene_env:SceneEnv',
    max_episode_steps=750,
    kwargs=dict(
        env_type='scene',
    ),
)
register(
    id='visual-scene-v0',
    entry_point='ogbench.manipspace.envs.scene_env:SceneEnv',
    max_episode_steps=750,
    kwargs=dict(
        env_type='scene',
        **visual_dict,
    ),
)

register(
    id='puzzle-3x3-v0',
    entry_point='ogbench.manipspace.envs.puzzle_env:PuzzleEnv',
    max_episode_steps=500,
    kwargs=dict(
        env_type='3x3',
    ),
)
register(
    id='visual-puzzle-3x3-v0',
    entry_point='ogbench.manipspace.envs.puzzle_env:PuzzleEnv',
    max_episode_steps=500,
    kwargs=dict(
        env_type='3x3',
        **visual_dict,
    ),
)
register(
    id='puzzle-4x4-v0',
    entry_point='ogbench.manipspace.envs.puzzle_env:PuzzleEnv',
    max_episode_steps=500,
    kwargs=dict(
        env_type='4x4',
    ),
)
register(
    id='visual-puzzle-4x4-v0',
    entry_point='ogbench.manipspace.envs.puzzle_env:PuzzleEnv',
    max_episode_steps=500,
    kwargs=dict(
        env_type='4x4',
        **visual_dict,
    ),
)
register(
    id='puzzle-4x5-v0',
    entry_point='ogbench.manipspace.envs.puzzle_env:PuzzleEnv',
    max_episode_steps=1000,
    kwargs=dict(
        env_type='4x5',
    ),
)
register(
    id='visual-puzzle-4x5-v0',
    entry_point='ogbench.manipspace.envs.puzzle_env:PuzzleEnv',
    max_episode_steps=1000,
    kwargs=dict(
        env_type='4x5',
        **visual_dict,
    ),
)
register(
    id='puzzle-4x6-v0',
    entry_point='ogbench.manipspace.envs.puzzle_env:PuzzleEnv',
    max_episode_steps=1000,
    kwargs=dict(
        env_type='4x6',
    ),
)
register(
    id='visual-puzzle-4x6-v0',
    entry_point='ogbench.manipspace.envs.puzzle_env:PuzzleEnv',
    max_episode_steps=1000,
    kwargs=dict(
        env_type='4x6',
        **visual_dict,
    ),
)

# Environments for reward-based single-task offline RL.
for task_id in [None, 1, 2, 3, 4, 5]:
    task_suffix = '' if task_id is None else f'-task{task_id}'
    reward_task_id = 0 if task_id is None else task_id  # 0 means the default task.

    register(
        id=f'cube-single-singletask{task_suffix}-v0',
        entry_point='ogbench.manipspace.envs.cube_env:CubeEnv',
        max_episode_steps=200,
        kwargs=dict(
            env_type='single',
            reward_task_id=reward_task_id,
            **cube_singletask_dict,
        ),
    )
    register(
        id=f'visual-cube-single-singletask{task_suffix}-v0',
        entry_point='ogbench.manipspace.envs.cube_env:CubeEnv',
        max_episode_steps=200,
        kwargs=dict(
            env_type='single',
            **visual_dict,
            reward_task_id=reward_task_id,
            **cube_singletask_dict,
        ),
    )
    register(
        id=f'cube-double-singletask{task_suffix}-v0',
        entry_point='ogbench.manipspace.envs.cube_env:CubeEnv',
        max_episode_steps=500,
        kwargs=dict(
            env_type='double',
            reward_task_id=reward_task_id,
            **cube_singletask_dict,
        ),
    )
    register(
        id=f'visual-cube-double-singletask{task_suffix}-v0',
        entry_point='ogbench.manipspace.envs.cube_env:CubeEnv',
        max_episode_steps=500,
        kwargs=dict(
            env_type='double',
            **visual_dict,
            reward_task_id=reward_task_id,
            **cube_singletask_dict,
        ),
    )
    register(
        id=f'cube-triple-singletask{task_suffix}-v0',
        entry_point='ogbench.manipspace.envs.cube_env:CubeEnv',
        max_episode_steps=1000,
        kwargs=dict(
            env_type='triple',
            reward_task_id=reward_task_id,
            **cube_singletask_dict,
        ),
    )
    register(
        id=f'visual-cube-triple-singletask{task_suffix}-v0',
        entry_point='ogbench.manipspace.envs.cube_env:CubeEnv',
        max_episode_steps=1000,
        kwargs=dict(
            env_type='triple',
            **visual_dict,
            reward_task_id=reward_task_id,
            **cube_singletask_dict,
        ),
    )
    register(
        id=f'cube-quadruple-singletask{task_suffix}-v0',
        entry_point='ogbench.manipspace.envs.cube_env:CubeEnv',
        max_episode_steps=1000,
        kwargs=dict(
            env_type='quadruple',
            reward_task_id=reward_task_id,
            **cube_singletask_dict,
        ),
    )
    register(
        id=f'visual-cube-quadruple-singletask{task_suffix}-v0',
        entry_point='ogbench.manipspace.envs.cube_env:CubeEnv',
        max_episode_steps=1000,
        kwargs=dict(
            env_type='quadruple',
            **visual_dict,
            reward_task_id=reward_task_id,
            **cube_singletask_dict,
        ),
    )

    register(
        id=f'scene-singletask{task_suffix}-v0',
        entry_point='ogbench.manipspace.envs.scene_env:SceneEnv',
        max_episode_steps=750,
        kwargs=dict(
            env_type='scene',
            reward_task_id=reward_task_id,
            **scene_singletask_dict,
        ),
    )
    register(
        id=f'visual-scene-singletask{task_suffix}-v0',
        entry_point='ogbench.manipspace.envs.scene_env:SceneEnv',
        max_episode_steps=750,
        kwargs=dict(
            env_type='scene',
            **visual_dict,
            reward_task_id=reward_task_id,
            **scene_singletask_dict,
        ),
    )

    register(
        id=f'puzzle-3x3-singletask{task_suffix}-v0',
        entry_point='ogbench.manipspace.envs.puzzle_env:PuzzleEnv',
        max_episode_steps=500,
        kwargs=dict(
            env_type='3x3',
            reward_task_id=reward_task_id,
            **puzzle_singletask_dict,
        ),
    )
    register(
        id=f'visual-puzzle-3x3-singletask{task_suffix}-v0',
        entry_point='ogbench.manipspace.envs.puzzle_env:PuzzleEnv',
        max_episode_steps=500,
        kwargs=dict(
            env_type='3x3',
            **visual_dict,
            reward_task_id=reward_task_id,
            **puzzle_singletask_dict,
        ),
    )
    register(
        id=f'puzzle-4x4-singletask{task_suffix}-v0',
        entry_point='ogbench.manipspace.envs.puzzle_env:PuzzleEnv',
        max_episode_steps=500,
        kwargs=dict(
            env_type='4x4',
            reward_task_id=reward_task_id,
            **puzzle_singletask_dict,
        ),
    )
    register(
        id=f'visual-puzzle-4x4-singletask{task_suffix}-v0',
        entry_point='ogbench.manipspace.envs.puzzle_env:PuzzleEnv',
        max_episode_steps=500,
        kwargs=dict(
            env_type='4x4',
            **visual_dict,
            reward_task_id=reward_task_id,
            **puzzle_singletask_dict,
        ),
    )
    register(
        id=f'puzzle-4x5-singletask{task_suffix}-v0',
        entry_point='ogbench.manipspace.envs.puzzle_env:PuzzleEnv',
        max_episode_steps=1000,
        kwargs=dict(
            env_type='4x5',
            reward_task_id=reward_task_id,
            **puzzle_singletask_dict,
        ),
    )
    register(
        id=f'visual-puzzle-4x5-singletask{task_suffix}-v0',
        entry_point='ogbench.manipspace.envs.puzzle_env:PuzzleEnv',
        max_episode_steps=1000,
        kwargs=dict(
            env_type='4x5',
            **visual_dict,
            reward_task_id=reward_task_id,
            **puzzle_singletask_dict,
        ),
    )
    register(
        id=f'puzzle-4x6-singletask{task_suffix}-v0',
        entry_point='ogbench.manipspace.envs.puzzle_env:PuzzleEnv',
        max_episode_steps=1000,
        kwargs=dict(
            env_type='4x6',
            reward_task_id=reward_task_id,
            **puzzle_singletask_dict,
        ),
    )
    register(
        id=f'visual-puzzle-4x6-singletask{task_suffix}-v0',
        entry_point='ogbench.manipspace.envs.puzzle_env:PuzzleEnv',
        max_episode_steps=1000,
        kwargs=dict(
            env_type='4x6',
            **visual_dict,
            reward_task_id=reward_task_id,
            **puzzle_singletask_dict,
        ),
    )
