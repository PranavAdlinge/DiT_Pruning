import torch
import torch.nn as nn

from nexus.losses.depth_distillation import (
    PruningDepthDistillationLoss,
    parse_pruning_intervals,
)


class _DummyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(5)])
        self.single_transformer_blocks = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(4)])


def test_parse_pruning_intervals_with_unified_stream_tokens():
    double_intervals, single_intervals = parse_pruning_intervals(
        pruned_blocks=["d0", "d1", "d3", "s1", "s2"]
    )

    assert double_intervals == [(0, 1), (3, 3)]
    assert single_intervals == [(1, 2)]


def test_parse_pruning_intervals_with_per_stream_lists():
    double_intervals, single_intervals = parse_pruning_intervals(
        double_stream_pruned_blocks=[0, 1, 4],
        single_stream_pruned_blocks=["s0", "s1"],
    )

    assert double_intervals == [(0, 1), (4, 4)]
    assert single_intervals == [(0, 1)]


def test_parse_pruning_intervals_with_explicit_interval_entries():
    double_intervals, single_intervals = parse_pruning_intervals(
        pruned_blocks=[["d1", "d2"], ["s2", "s4"]]
    )

    assert double_intervals == [(1, 2)]
    assert single_intervals == [(2, 4)]


def test_prepare_for_training_uses_midpoint_source_block_by_default():
    transformer = _DummyTransformer()
    loss = PruningDepthDistillationLoss(
        pretrained_model_name_or_path="dummy",
        pruned_blocks=[["d1", "d3"], ["s0", "s1"]],
        transformer_cls=object,
        device=torch.device("cpu"),
    )

    loss.prepare_for_training(transformer)

    assert len(loss.student_double_blocks) == 1
    assert len(loss.student_single_blocks) == 1
    assert loss.double_interval_specs[0].start == 1
    assert loss.double_interval_specs[0].end == 3
    assert loss.double_interval_specs[0].source_index == 2
    assert loss.single_interval_specs[0].start == 0
    assert loss.single_interval_specs[0].end == 1
    assert loss.single_interval_specs[0].source_index == 0

    with torch.no_grad():
        transformer.transformer_blocks[2].weight.fill_(3.0)

    student_param = next(iter(loss.student_double_blocks[0].parameters()))
    assert not torch.equal(
        student_param,
        transformer.transformer_blocks[2].weight,
    )


def test_prepare_for_training_supports_start_block_initialization():
    transformer = _DummyTransformer()
    loss = PruningDepthDistillationLoss(
        pretrained_model_name_or_path="dummy",
        pruned_blocks=[["d1", "d3"]],
        student_block_init="start",
        transformer_cls=object,
        device=torch.device("cpu"),
    )

    loss.prepare_for_training(transformer)

    assert loss.double_interval_specs[0].source_index == 1


def test_teacher_is_not_registered_in_state_dict():
    loss = PruningDepthDistillationLoss(
        pretrained_model_name_or_path="dummy",
        double_stream_pruned_blocks=[0, 1],
        transformer_cls=object,
        device=torch.device("cpu"),
    )
    loss.__dict__["_teacher"] = nn.Linear(4, 4, bias=False)

    assert all("_teacher" not in key for key in loss.state_dict().keys())
