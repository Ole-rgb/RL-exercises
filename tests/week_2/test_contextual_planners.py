import unittest

import numpy as np
from rl_exercises.week_2.contextual_policy_iteration import ContextualPolicyIteration
from rl_exercises.week_2.contextual_value_iteration import ContextualValueIteration
from rl_exercises.week_2.tilted_mars_rover import TiltedMarsRover


class TestContextualPlanners(unittest.TestCase):
    def test_contextual_value_iteration_updates_internal_context_from_info(self):
        env = TiltedMarsRover(
            transition_probabilities=np.full((5, 2), 0.5),
            tilt_angle=0.0,
            friction=0.0,
            seed=0,
        )
        agent = ContextualValueIteration(env=env, seed=0)

        agent.predict_action(
            observation=2,
            info={"context": {"tilt_angle": 15.0, "friction": 0.25}},
            evaluate=True,
        )

        self.assertAlmostEqual(agent.env.tilt_angle, 15.0)
        self.assertAlmostEqual(agent.env.friction, 0.25)

    def test_contextual_policy_iteration_replans_for_new_context(self):
        env = TiltedMarsRover(
            transition_probabilities=np.full((5, 2), 0.5),
            tilt_angle=0.0,
            friction=0.0,
            seed=0,
        )
        agent = ContextualPolicyIteration(env=env, seed=0)

        agent.predict_action(
            observation=2,
            info={"context": {"tilt_angle": 0.0, "friction": 0.0}},
            evaluate=True,
        )
        T_before = agent.T.copy()

        agent.predict_action(
            observation=2,
            info={"context": {"tilt_angle": 15.0, "friction": 0.25}},
            evaluate=True,
        )
        T_after = agent.T.copy()

        self.assertFalse(np.allclose(T_before, T_after))
        self.assertAlmostEqual(agent.env.tilt_angle, 15.0)
        self.assertAlmostEqual(agent.env.friction, 0.25)


if __name__ == "__main__":
    unittest.main()
