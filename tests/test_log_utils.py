"""Tests for nexus.utils.log_utils"""
from types import SimpleNamespace

import pytest

from nexus.utils.log_utils import _require_mlflow, get_output_dir, log_dataset_input, setup_mlflow_log_with, setup_tracking


class TestRequireMlflow:
    def test_raises_when_report_to_not_mlflow(self):
        cfg = SimpleNamespace(report_to="tensorboard", mlflow=SimpleNamespace(experiment_name="x"))
        with pytest.raises(ValueError, match="report_to must be 'mlflow'"):
            _require_mlflow(cfg)

    def test_raises_when_no_mlflow_config(self):
        cfg = SimpleNamespace(report_to="mlflow")
        with pytest.raises(ValueError, match="mlflow config required"):
            _require_mlflow(cfg)

    def test_raises_when_no_experiment_name(self):
        cfg = SimpleNamespace(report_to="mlflow", mlflow=SimpleNamespace(run_name="x"))
        with pytest.raises(ValueError, match="experiment_name is required"):
            _require_mlflow(cfg)

    def test_raises_when_no_run_name(self):
        cfg = SimpleNamespace(report_to="mlflow", mlflow=SimpleNamespace(experiment_name="x"))
        with pytest.raises(ValueError, match="run_name is required"):
            _require_mlflow(cfg)

    def test_passes_when_configured(self):
        cfg = SimpleNamespace(report_to="mlflow", mlflow=SimpleNamespace(experiment_name="my-exp", run_name="my-run"))
        _require_mlflow(cfg)  # no raise


class TestSetupMlflowLogWith:
    def test_returns_tracker(self, tmp_path):
        cfg = SimpleNamespace(experiment_name="my-exp", run_name="my-run", tracking_uri=None)
        result = setup_mlflow_log_with(tmp_path, cfg)
        from accelerate.tracking import MLflowTracker
        assert isinstance(result, MLflowTracker)
        assert (tmp_path / "mlruns").exists()


class TestSetupTracking:
    def test_sets_output_dir_and_returns_log_with(self, tmp_path):
        cfg = SimpleNamespace(
            report_to="mlflow",
            log_root=str(tmp_path),
            mlflow=SimpleNamespace(
                experiment_name="my-exp",
                run_name="my-run",
                tracking_uri=None,
                user=None,
            ),
        )
        output_dir, log_with = setup_tracking(cfg)
        assert cfg.output_dir == output_dir
        assert "experiments" in output_dir and "my-exp-my-run" in output_dir
        from accelerate.tracking import MLflowTracker
        assert isinstance(log_with, MLflowTracker)


class TestLogDatasetInput:
    def test_logs_class_and_kwargs_without_path(self, tmp_path):
        import mlflow

        mlruns = tmp_path / "mlruns"
        mlruns.mkdir()
        mlflow.set_tracking_uri(mlruns.as_uri())
        mlflow.set_experiment("test-dataset-no-path")
        with mlflow.start_run():
            log_dataset_input(
                class_name="nexus.data.precomputed_sstk_dataset:PrecomputedSSTKDataset",
                kwargs={"resolution": 512, "shuffle": True},
                context="training",
            )
            run = mlflow.get_run(mlflow.active_run().info.run_id)
            assert run.data.params["dataset.class"] == "nexus.data.precomputed_sstk_dataset:PrecomputedSSTKDataset"
            assert "512" in run.data.params["dataset.kwargs"]
            assert len(run.inputs.dataset_inputs) == 0

    def test_logs_dataset_with_path_and_name_from_config(self, tmp_path):
        import mlflow

        mlruns = tmp_path / "mlruns2"
        mlruns.mkdir()
        mlflow.set_tracking_uri(mlruns.as_uri())
        mlflow.set_experiment("test-dataset-with-path")
        data_dir = tmp_path / "mds"
        data_dir.mkdir()
        with mlflow.start_run():
            log_dataset_input(
                class_name="nexus.data.precomputed_sstk_dataset:PrecomputedSSTKDataset",
                kwargs={"resolution": 512},
                name="my-training-data",
                source_path=str(data_dir),
                context="training",
            )
            run = mlflow.get_run(mlflow.active_run().info.run_id)
            inputs = run.inputs.dataset_inputs
            assert len(inputs) == 1
            assert inputs[0].dataset.name == "my-training-data"
            assert run.data.params["dataset.class"] == "nexus.data.precomputed_sstk_dataset:PrecomputedSSTKDataset"

    def test_uses_class_name_as_dataset_name_when_name_not_in_config(self, tmp_path):
        import mlflow

        mlruns = tmp_path / "mlruns3"
        mlruns.mkdir()
        mlflow.set_tracking_uri(mlruns.as_uri())
        mlflow.set_experiment("test-dataset-name-fallback")
        data_dir = tmp_path / "mds"
        data_dir.mkdir()
        with mlflow.start_run():
            log_dataset_input(
                class_name="nexus.data.precomputed_sstk_dataset:PrecomputedSSTKDataset",
                source_path=str(data_dir),
                context="training",
            )
            run = mlflow.get_run(mlflow.active_run().info.run_id)
            inputs = run.inputs.dataset_inputs
            assert len(inputs) == 1
            assert inputs[0].dataset.name == "PrecomputedSSTKDataset"


class TestGetOutputDir:
    def test_output_dir_path_default_user(self, tmp_path):
        out = get_output_dir(tmp_path, "nexus-flux2", "flux2-dreambooth-lora")
        assert out == tmp_path / "experiments" / "default" / "nexus-flux2-flux2-dreambooth-lora"

    def test_output_dir_path_with_user(self, tmp_path):
        out = get_output_dir(tmp_path, "nexus-flux2", "flux2-dreambooth-lora", user="alice")
        assert out == tmp_path / "experiments" / "alice" / "nexus-flux2-flux2-dreambooth-lora"
