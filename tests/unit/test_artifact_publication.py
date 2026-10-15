# AI-generated test code: generated with OpenAI Codex.
from pathlib import Path
import pytest
from airline.ml import model_contract as c


class Saved:
    def write(self): return self
    def save(self, path):
        Path(path).mkdir()
        (Path(path) / "data").write_text("complete")


@pytest.mark.parametrize("failure", ["save", "reload", "validate", "publish", "none"])
def test_publication_is_complete_or_preserves_previous(tmp_path, monkeypatch, failure):
    destination = tmp_path / "models"
    destination.mkdir()
    (destination / "manifest.json").write_text('"previous usable bundle"')
    monkeypatch.setattr(c, "validate_manifest", lambda _: None)
    def load(path):
        if failure == "reload": raise RuntimeError("reload failed")
        return Saved()
    monkeypatch.setattr(c.PipelineModel, "load", load)
    monkeypatch.setattr(c.LogisticRegressionModel, "load", load)
    def validate(*args):
        if failure == "validate": raise RuntimeError("incompatible model")
    monkeypatch.setattr(c, "validate_models", validate)
    if failure == "save":
        monkeypatch.setattr(Saved, "save", lambda *args: (_ for _ in ()).throw(RuntimeError("save failed")))
    rename = Path.rename
    def fail_publish(self, target):
        if failure == "publish" and ".tmp-" in self.name:
            raise OSError("publication failed")
        return rename(self, target)
    monkeypatch.setattr(Path, "rename", fail_publish)
    if failure != "none":
        with pytest.raises((RuntimeError, OSError)):
            c.publish_bundle(destination, Saved(), Saved(), {"complete": True, "selected_classifier": "lr"})
        assert (destination / "manifest.json").read_text() == '"previous usable bundle"'
    else:
        c.publish_bundle(destination, Saved(), Saved(), {"complete": True, "selected_classifier": "lr"})
        assert (destination / "preprocessing/data").read_text() == "complete"
        assert (destination / "classifier/data").read_text() == "complete"
        assert '"complete": true' in (destination / "manifest.json").read_text()
    assert list(tmp_path.iterdir()) == [destination]
