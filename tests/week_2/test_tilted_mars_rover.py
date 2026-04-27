import numpy as np
from rl_exercises.week_2.tilted_mars_rover import TiltedMarsRover


def test_tilted_mars_rover_friction_zero_matches_tilt_only_transitions():
    env = TiltedMarsRover(
        transition_probabilities=np.full((5, 2), 0.5),
        tilt_angle=15.0,
        friction=0.0,
    )

    T = env.get_transition_matrix()

    assert np.allclose(T.sum(axis=2), 1.0)
    assert np.isclose(T[2, 1, 3] + T[2, 1, 1], 1.0)
    assert np.isclose(T[2, 1, 2], 0.0)


def test_tilted_mars_rover_friction_adds_stay_probability():
    env = TiltedMarsRover(
        transition_probabilities=np.full((5, 2), 0.5),
        tilt_angle=0.0,
        friction=0.25,
    )

    T = env.get_transition_matrix()

    assert np.allclose(T.sum(axis=2), 1.0)
    assert np.isclose(T[2, 1, 2], 0.25)
    assert np.isclose(T[2, 1, 3], 0.375)
    assert np.isclose(T[2, 1, 1], 0.375)


def test_tilted_mars_rover_full_friction_stays_put():
    env = TiltedMarsRover(
        transition_probabilities=np.full((5, 2), 0.5),
        tilt_angle=10.0,
        friction=1.0,
        seed=0,
    )

    obs, _ = env.reset()
    next_obs, _, _, _, _ = env.step(1)

    assert next_obs == obs
    assert np.allclose(env.get_transition_matrix()[2, 1], np.eye(5)[2])
