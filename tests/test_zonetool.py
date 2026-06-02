"""Tests for the headless zone-authoring helpers."""
import yaml

from firewatch.zonetool import auto_bands, save_zones
from firewatch.floors.mapper import FloorMapper


def test_auto_bands_descend_from_top():
    zones = auto_bands(width=200, height=300, n_floors=3)
    assert [z["floor"] for z in zones] == [3, 2, 1]
    # Bands cover the full height contiguously.
    assert zones[0]["polygon"][0] == [0, 0]
    assert zones[-1]["polygon"][2] == [200, 300]


def test_auto_bands_map_points_to_expected_floor():
    zones = auto_bands(width=200, height=300, n_floors=3)
    mapper = FloorMapper.from_config({"floors": zones})
    # bottom-center y=50 -> top band floor 3; y=150 -> floor 2; y=250 -> floor 1.
    assert mapper.locate((95, 40, 105, 50)) == 3
    assert mapper.locate((95, 140, 105, 150)) == 2
    assert mapper.locate((95, 240, 105, 250)) == 1


def test_save_zones_round_trips(tmp_path):
    cfg = {"camera_id": "cam1", "source": {"type": "webcam", "index": 0}}
    zones = auto_bands(100, 100, 2)
    path = tmp_path / "cam1.yaml"
    save_zones(str(path), cfg, zones)

    loaded = yaml.safe_load(path.read_text())
    assert loaded["camera_id"] == "cam1"
    assert len(loaded["floors"]) == 2
