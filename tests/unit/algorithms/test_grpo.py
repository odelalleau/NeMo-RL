# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import MagicMock, patch

import pytest
import ray
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_rl.algorithms.grpo import (
    _default_grpo_save_state,
    async_grpo_train,
    grpo_train,
)
from nemo_rl.algorithms.loss_functions import ClippedPGLossFn
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)
from nemo_rl.experience.rollouts import calculate_rewards

# ============================================================================
# Stub classes for async GRPO testing (non-Ray versions for easy mocking)
# ============================================================================


class StubReplayBuffer:
    """Non-Ray stub of ReplayBuffer for unit testing

    Each method returns a MagicMock with a 'remote' attribute that can be called.
    """

    def __init__(self, initial_size=10, mock_batch=None, mock_rollout_metrics=None):
        self._size = initial_size
        self._trajectories = []
        self._mock_batch = mock_batch
        self._mock_rollout_metrics = mock_rollout_metrics or {}

    @property
    def size(self):
        """Return a mock that returns buffer size when .remote() is called"""
        mock = MagicMock()
        mock.remote = MagicMock(return_value=self._size)  # ray.get will extract this
        return mock

    @property
    def sample(self):
        """Return a mock that returns sample result when .remote() is called"""

        def _sample(num_prompt_groups, current_weight_version, max_age_steps):
            # Return proper trajectory structure expected by async GRPO
            trajectories = [
                {
                    "batch": self._mock_batch,
                    "rollout_metrics": self._mock_rollout_metrics,
                }
                for _ in range(num_prompt_groups)
            ]
            return {
                "trajectories": trajectories,
                "avg_trajectory_age": 0.5,
            }

        mock = MagicMock()
        mock.remote = MagicMock(
            side_effect=lambda *args, **kwargs: _sample(*args, **kwargs)
        )
        return mock

    @property
    def get_debug_info(self):
        """Return a mock that returns debug info when .remote() is called"""
        mock = MagicMock()
        mock.remote = MagicMock(
            return_value={
                "total_trajectories": self._size,
                "trajectory_versions": [0],
                "target_weight_versions": [0],
                "max_size": 100,
            }
        )
        return mock


class StubAsyncTrajectoryCollector:
    """Non-Ray stub of AsyncTrajectoryCollector for unit testing

    Each method is a property that returns a MagicMock with a 'remote' attribute.
    """

    @property
    def start_collection(self):
        """Start collection - returns a remote-callable mock"""
        mock = MagicMock()
        mock.remote = MagicMock(return_value=MagicMock())  # Returns a fake ObjectRef
        return mock

    @property
    def set_weight_version(self):
        """Set weight version - returns a remote-callable mock"""
        mock = MagicMock()
        mock.remote = MagicMock(return_value=MagicMock())
        return mock

    @property
    def pause(self):
        """Pause collection - returns a remote-callable mock"""
        mock = MagicMock()
        mock.remote = MagicMock(return_value=MagicMock())
        return mock

    @property
    def resume(self):
        """Resume collection - returns a remote-callable mock"""
        mock = MagicMock()
        mock.remote = MagicMock(return_value=MagicMock())
        return mock

    @property
    def stop(self):
        """Stop collection - returns a remote-callable mock"""
        mock = MagicMock()
        mock.remote = MagicMock(return_value=MagicMock())
        return mock

    @property
    def wait_for_stop(self):
        """Wait for stop - returns a remote-callable mock"""
        mock = MagicMock()
        mock.remote = MagicMock(return_value=MagicMock())
        return mock


def mock_async_grpo_infrastructure(mock_batch, mock_rollout_metrics):
    """
    Context manager that mocks all async GRPO infrastructure (Ray actors, venv, etc).

    Returns a dict of patches that can be used as a context manager stack.
    """
    from contextlib import ExitStack

    stack = ExitStack()

    # Create stub instances with mock data
    stub_buffer = StubReplayBuffer(
        initial_size=10,
        mock_batch=mock_batch,
        mock_rollout_metrics=mock_rollout_metrics,
    )
    stub_collector = StubAsyncTrajectoryCollector()

    # Patch venv creation
    stack.enter_context(
        patch(
            "nemo_rl.algorithms.grpo.create_local_venv_on_each_node",
            return_value="/fake/venv",
        )
    )
    stack.enter_context(
        patch(
            "nemo_rl.algorithms.grpo.get_actor_python_env", return_value="/fake/python"
        )
    )

    # Patch Ray actor classes to return our stubs
    mock_buffer_cls = MagicMock()
    mock_buffer_cls.options.return_value.remote.return_value = stub_buffer
    stack.enter_context(
        patch("nemo_rl.algorithms.async_utils.ReplayBuffer", mock_buffer_cls)
    )

    mock_collector_cls = MagicMock()
    mock_collector_cls.options.return_value.remote.return_value = stub_collector
    stack.enter_context(
        patch(
            "nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector",
            mock_collector_cls,
        )
    )

    # Patch ray.get to return values from our stubs (not remote refs)
    def mock_ray_get(ref):
        # If it's already a plain value (from our stubs), return it
        if isinstance(ref, (int, str, dict, list)):
            return ref
        # If it's a MagicMock, return a default response
        return None

    stack.enter_context(patch("ray.get", side_effect=mock_ray_get))
    stack.enter_context(
        patch("ray.wait", side_effect=lambda refs, **kwargs: (refs, []))
    )
    stack.enter_context(
        patch("ray.kill", return_value=None)
    )  # Mock ray.kill for cleanup

    # Patch the rollout functions used inside async_grpo_train
    stack.enter_context(
        patch(
            "nemo_rl.algorithms.grpo.run_multi_turn_rollout",
            return_value=(mock_batch, mock_rollout_metrics),
        )
    )
    stack.enter_context(
        patch(
            "nemo_rl.algorithms.grpo.run_async_multi_turn_rollout",
            return_value=(mock_batch, mock_rollout_metrics),
        )
    )

    # Patch refit and validate functions
    stack.enter_context(
        patch("nemo_rl.algorithms.grpo.refit_policy_generation", return_value=None)
    )
    stack.enter_context(
        patch("nemo_rl.algorithms.grpo.validate", return_value=({}, {}))
    )

    # Mock print_performance_metrics to avoid needing real timing metrics
    stack.enter_context(
        patch("nemo_rl.algorithms.grpo.print_performance_metrics", return_value={})
    )

    return stack


@ray.remote(num_cpus=0)
class MockEnvironment(EnvironmentInterface):
    def __init__(self, rewards: list[float]):
        self.rewards = rewards
        self._calls = 0

    def step(
        self, messages: list[LLMMessageLogType], env_info: list[dict]
    ) -> EnvironmentReturn:
        self._calls += 1
        return (
            [{"role": "environment", "content": "observation"}] * len(messages),
            [{}] * len(messages),
            [[]] * len(messages),
            self.rewards,
            [True] * len(messages),
            [None] * len(messages),
        )

    def get_calls(self):
        return self._calls

    def reset_calls(self):
        self._calls = 0
        return True

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        return batch, {}


def create_mock_batch(
    num_samples: int,
    task_names: list[str],
    message_logs: list[LLMMessageLogType],
    extra_env_info: list[dict] = None,
) -> BatchedDataDict[DatumSpec]:
    """Helper function to create a mock batch for testing."""
    if extra_env_info is None:
        extra_env_info = [{} for _ in range(num_samples)]

    return BatchedDataDict[DatumSpec](
        {
            "task_name": task_names,
            "message_log": message_logs,
            "extra_env_info": extra_env_info,
            "loss_multiplier": torch.ones(num_samples),
        }
    )


@pytest.fixture(scope="module")
def mock_env():
    """Create a mock environment for single task tests."""
    env = MockEnvironment.remote(rewards=[1.0, 2.0])
    yield env
    ray.kill(env)


@pytest.fixture(scope="module")
def mock_envs():
    """Create mock environments for multiple task tests."""
    math_env = MockEnvironment.remote(rewards=[1.0, 2.0])
    code_env = MockEnvironment.remote(rewards=[3.0, 4.0])
    yield {"math": math_env, "code": code_env}
    ray.kill(math_env)
    ray.kill(code_env)


@pytest.fixture(autouse=True)
def reset_env_calls(mock_env, mock_envs):
    """Reset call counters before each test."""
    ray.get(mock_env.reset_calls.remote())
    ray.get(mock_envs["math"].reset_calls.remote())
    ray.get(mock_envs["code"].reset_calls.remote())
    yield


def test_calculate_rewards_single_task(mock_env):
    """Test reward calculation with a single task type."""
    task_to_env = {"math": mock_env}

    # Create test data
    task_names = ["math", "math"]
    message_logs = [
        [{"role": "user", "content": "1+1"}, {"role": "assistant", "content": "2"}],
        [{"role": "user", "content": "2+2"}, {"role": "assistant", "content": "4"}],
    ]
    batch = create_mock_batch(2, task_names, message_logs)

    # Calculate rewards
    env_observations, metadata, next_stop_strings, rewards, terminateds, answers = (
        calculate_rewards(batch, task_to_env)
    )

    # Verify results
    assert torch.allclose(rewards, torch.tensor([1.0, 2.0]))
    assert len(env_observations) == 2
    assert len(terminateds) == 2
    assert len(next_stop_strings) == 2
    assert len(metadata) == 2
    assert len(answers) == 2
    assert torch.allclose(rewards, torch.tensor([1.0, 2.0]))
    assert (
        ray.get(mock_env.get_calls.remote()) == 1
    )  # Should only call once for all samples of same task


def test_calculate_rewards_multiple_tasks(mock_envs):
    """Test reward calculation with multiple task types."""
    # Create test data
    task_names = ["math", "math", "code", "code"]
    message_logs = [
        [{"role": "user", "content": "1+1"}, {"role": "assistant", "content": "2"}],
        [{"role": "user", "content": "2+2"}, {"role": "assistant", "content": "4"}],
        [
            {"role": "user", "content": "print('hello')"},
            {"role": "assistant", "content": "hello"},
        ],
        [
            {"role": "user", "content": "print('world')"},
            {"role": "assistant", "content": "world"},
        ],
    ]
    batch = create_mock_batch(4, task_names, message_logs)

    # Calculate rewards
    env_observations, metadata, next_stop_strings, rewards, terminateds, answers = (
        calculate_rewards(batch, mock_envs)
    )

    # Verify results
    assert torch.allclose(rewards, torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert len(env_observations) == 4
    assert len(terminateds) == 4
    assert len(next_stop_strings) == 4
    assert len(metadata) == 4
    assert len(answers) == 4
    assert torch.allclose(rewards, torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert (
        ray.get(mock_envs["math"].get_calls.remote()) == 1
    )  # One call for all math samples
    assert (
        ray.get(mock_envs["code"].get_calls.remote()) == 1
    )  # One call for all code samples


def test_calculate_rewards_empty_batch(mock_env):
    """Test reward calculation with an empty batch."""
    task_to_env = {"math": mock_env}

    # Create empty test data
    batch = create_mock_batch(0, [], [])

    # Calculate rewards
    env_observations, metadata, next_stop_strings, rewards, terminateds, answers = (
        calculate_rewards(batch, task_to_env)
    )

    # Verify results
    assert len(rewards) == 0
    assert len(env_observations) == 0
    assert len(terminateds) == 0
    assert len(next_stop_strings) == 0
    assert len(metadata) == 0
    assert len(answers) == 0
    assert (
        ray.get(mock_env.get_calls.remote()) == 0
    )  # Should not call environment for empty batch


def test_calculate_rewards_missing_environment():
    """Test reward calculation with a missing environment."""
    # Create test data with unknown task
    task_names = ["unknown_task"]
    message_logs = [[{"role": "user", "content": "test"}]]
    batch = create_mock_batch(1, task_names, message_logs)

    # Try to calculate rewards with missing environment
    task_to_env = {}  # Empty dict means no environments available
    with pytest.raises(
        ValueError, match="No environment found for task type: unknown_task"
    ):
        calculate_rewards(batch, task_to_env)
