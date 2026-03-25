"""Tests for datasets/precompute.py"""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    torch = None
    HAS_CUDA = False

# Add datasets dir for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precompute import (
    parse_args,
    _caption_sample_weights,
    _datadir_to_streams,
    discover_groups,
    _partition,
    _sample_caption,
)


def _get_precompute_main():
    from precompute import main
    return main


class TestParseArgs:
    def test_required_datadir(self):
        with patch.object(sys, "argv", ["precompute.py", "--datadir", "/path/to/mds"]):
            args = parse_args()
            assert args.datadir == "/path/to/mds"

    def test_default_values(self):
        with patch.object(sys, "argv", ["precompute.py", "--datadir", "/path/to/mds"]):
            args = parse_args()
            assert args.savedir == ""
            assert args.num_proc is None
            assert args.resolution == 512


class TestDatadirToStreams:
    def test_single_str_becomes_list(self):
        from streaming import Stream

        streams = _datadir_to_streams("/path/to/mds")
        assert len(streams) == 1
        assert isinstance(streams[0], Stream)

    def test_list_of_strs_becomes_list_of_streams(self):
        from streaming import Stream

        streams = _datadir_to_streams(["/a", "/b"])
        assert len(streams) == 2
        assert all(isinstance(s, Stream) for s in streams)


class TestDiscoverGroups:
    def test_raises_when_not_dir(self, tmp_path):
        p = tmp_path / "nonexistent"
        with pytest.raises(ValueError, match="does not exist"):
            discover_groups(str(p), "/out")

    def test_raises_when_no_shards(self, tmp_path):
        (tmp_path / "meta").mkdir()
        with pytest.raises(ValueError, match="No MDS shards"):
            discover_groups(str(tmp_path), "/out")

    def test_flat_layout(self, tmp_path):
        (tmp_path / "0").mkdir()
        (tmp_path / "1").mkdir()
        (tmp_path / "meta").mkdir()
        groups = discover_groups(str(tmp_path), "/out")
        assert len(groups) == 1
        outdir, shards = groups[0]
        assert outdir == "/out"
        assert len(shards) == 2
        assert any("0" in s for s in shards)
        assert any("1" in s for s in shards)

    def test_nested_layout(self, tmp_path):
        (tmp_path / "a" / "0").mkdir(parents=True)
        (tmp_path / "a" / "1").mkdir()
        (tmp_path / "b" / "0").mkdir(parents=True)
        groups = discover_groups(str(tmp_path), "/out")
        assert len(groups) == 2
        outdirs = {g[0] for g in groups}
        assert outdirs == {"/out/a", "/out/b"}


class TestPartition:
    def test_num_geq_shards(self):
        shards = ["/a", "/b", "/c"]
        parts = _partition(shards, 5)
        assert len(parts) == 5
        assert parts[0] == ["/a"]
        assert parts[1] == ["/b"]
        assert parts[2] == ["/c"]
        assert parts[3] == []
        assert parts[4] == []


class TestCaptionSampleWeights:
    def test_uniform_when_none(self):
        w = _caption_sample_weights(3, None)
        assert w.shape == (3,)
        np.testing.assert_array_almost_equal(w, [1 / 3] * 3)
