"""Actual manager time/query methods on CPU; no scene or physics construction."""
from types import SimpleNamespace

import pytest
import torch

from instinct_mj.motion_reference.motion_reference_data import MotionReferenceData, MotionReferenceState
from instinct_mj.motion_reference.motion_reference_manager import MotionReferenceManager


class Rows:
    def __init__(self, source_dt):
        self.source_dt = source_dt
        self.starts = torch.tensor([3, 7])
        self.queries = []

    def fill_motion_data(self, ids, timestamp, origins, target):
        assert timestamp.dtype == torch.float64
        self.queries.append(timestamp.clone())
        rows = self.starts[ids, None] + torch.floor(timestamp / self.source_dt + 1e-6).long()
        target.base_pos_w[ids, :, 0] = rows.to(target.base_pos_w.dtype)
        target.validity[ids] = True


def manager(source_dt=.02, interval=.02):
    cfg = SimpleNamespace(entity_name='robot', num_frames=3, data_class_type=MotionReferenceData,
        state_class_type=MotionReferenceState, robot_model_path=None, link_of_interests=None,
        frame_interval_s=interval, data_start_from='one_frame_interval', update_period=.02,
        symmetric_augmentation_joint_mapping=None, symmetric_augmentation_joint_reverse_buf=None,
        symmetric_augmentation_link_mapping=None)
    value = MotionReferenceManager(cfg)
    value.device = 'cpu'
    value._entities = {'robot': SimpleNamespace(data=SimpleNamespace(default_root_state=torch.zeros(2, 13)),
                                               num_joints=1, joint_names=['joint'])}
    value._initialize_data()
    value._is_initialized = True
    value._resample_buffer_collate_params()
    rows = Rows(source_dt)
    value._motion_buffers = {'rows': rows}
    value._motion_buffer_assignment = {'rows': slice(0, 2)}
    def reset_rows(ids):
        rows.starts[ids] += 20
    value._reset_motion_buffers = reset_rows
    return value, rows


@pytest.mark.parametrize('source_centiseconds', [2, 4])
@pytest.mark.parametrize('interval_centiseconds', [2, 3])
def test_499_control_current_and_future_rows_preserve_source_seconds(source_centiseconds, interval_centiseconds):
    value, rows = manager(source_centiseconds / 100, interval_centiseconds / 100)
    for control in range(499):
        expected = rows.starts + (control * 2) // source_centiseconds
        torch.testing.assert_close(value.reference_frame.base_pos_w[:, 0, 0], expected.float())
        value._update_buffers_impl(value.ALL_INDICES)
        offsets = torch.tensor([(control * 2 + future * interval_centiseconds) // source_centiseconds
                                for future in range(1, 4)])
        torch.testing.assert_close(value._data.base_pos_w[:, :, 0], (rows.starts[:, None] + offsets).float())
        assert value._data.time_to_target_frame.dtype == torch.float32
        assert value._data.base_pos_w.dtype == torch.float32
        for _ in range(4):
            value.update(.005)
    assert value._elapsed_ticks.tolist() == [1996, 1996]
    assert value.elapsed_time.dtype == torch.float64
    assert value._timestamp_last_update.dtype == value._data_timestamp_last_update.dtype == torch.float64


def test_long_elapsed_ticks_partial_reset_and_reads_have_one_time_owner():
    value, rows = manager()
    # Cover late controls across a complete 320,000-control budget without millions of CPU kernel launches.
    for control in (319_987, 319_999, 320_000):
        value._elapsed_ticks[:] = control * 4
        for _ in range(4):
            value.update(.005)
        expected = torch.full((2,), (control + 1) * .02, dtype=torch.float64)
        torch.testing.assert_close(value.elapsed_time, expected, rtol=0, atol=1e-12)
        torch.testing.assert_close(value.reference_frame.base_pos_w[:, 0, 0],
                                   (rows.starts + control + 1).float())
    before, ticks = value.elapsed_time, value._elapsed_ticks.clone()
    detached = value.elapsed_time
    detached.zero_()
    value.update(0.)
    torch.testing.assert_close(value.elapsed_time, before)
    assert torch.equal(value._elapsed_ticks, ticks)
    value.reset(torch.tensor([0]))
    assert value._elapsed_ticks.tolist() == [0, int(ticks[1])]
    assert value.elapsed_time.tolist() == [0., before[1].item()]
    value._update_buffers_impl(value.ALL_INDICES)
    assert value._data.base_pos_w[0, :, 0].tolist() == [24., 25., 26.]
    torch.testing.assert_close(value._data.base_pos_w[1, :, 0],
                               (rows.starts[1] + control + 1 + torch.arange(1, 4)).float())
    value.reset()
    assert not value._elapsed_ticks.any() and not value.elapsed_time.any()


def test_zero_refresh_does_not_select_dt_and_variable_physics_step_is_rejected():
    value, _ = manager()
    value.update(0.)
    assert value._clock_dt is None and not value._elapsed_ticks.any()
    value.update(.005)
    before = value.elapsed_time
    for invalid in (.004, float('nan'), float('inf'), -.005):
        with pytest.raises(ValueError, match='fixed|finite and nonnegative'):
            value.update(invalid)
        torch.testing.assert_close(value.elapsed_time, before)
    value.reset()
    with pytest.raises(ValueError, match='fixed'):
        value.update(.01)


def test_time_observations_and_commands_keep_policy_dtype():
    from instinct_mj.envs.mdp.observations.motion_reference import time_from_reference_update, ref_frame_interval
    from instinct_mj.envs.mdp.commands.shadowing_command import TimeToTargetCommand
    value, _ = manager()
    value.update(.005)
    env = SimpleNamespace(scene={'motion_reference': value})
    for multiple in (False, True):
        for current in (False, True):
            assert time_from_reference_update(env, multi_frames=multiple,
                                             current_state_mask_at_last=current).dtype == torch.float32
    assert ref_frame_interval(env).dtype == torch.float32
    command = TimeToTargetCommand.__new__(TimeToTargetCommand)
    command._motion_reference = value
    command._command = torch.zeros(2, 3, 1)
    command._update_command_by_env_ids(torch.arange(2))
    assert command._command.dtype == torch.float32
