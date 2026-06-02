"""Tests for polygon zones and bbox-to-floor mapping."""
import pytest

from firewatch.floors.zones import FloorZone, point_in_polygon
from firewatch.floors.mapper import FloorMapper

SQUARE = [(0, 0), (100, 0), (100, 100), (0, 100)]


def test_point_inside_polygon():
    assert point_in_polygon((50, 50), SQUARE) is True


def test_point_outside_polygon():
    assert point_in_polygon((150, 50), SQUARE) is False
    assert point_in_polygon((50, 150), SQUARE) is False


def test_point_in_polygon_handles_concave_shape():
    # An L-shape: the notch in the top-right must read as outside.
    l_shape = [(0, 0), (100, 0), (100, 40), (40, 40), (40, 100), (0, 100)]
    assert point_in_polygon((20, 80), l_shape) is True   # in the leg
    assert point_in_polygon((80, 80), l_shape) is False  # in the notch


# Two stacked zones: floor 2 is the top band, floor 1 the bottom band.
ZONES = [
    FloorZone(floor=2, polygon=[(0, 0), (200, 0), (200, 100), (0, 100)]),
    FloorZone(floor=1, polygon=[(0, 100), (200, 100), (200, 200), (0, 200)]),
]


def _bbox_centered_at(cx, bottom):
    # bbox is (x1, y1, x2, y2); bottom-center is what the mapper uses.
    return (cx - 5, bottom - 10, cx + 5, bottom)


def test_mapper_uses_bottom_center_to_pick_floor():
    mapper = FloorMapper(ZONES)
    assert mapper.locate(_bbox_centered_at(100, bottom=50)) == 2
    assert mapper.locate(_bbox_centered_at(100, bottom=150)) == 1


def test_mapper_returns_none_when_outside_all_zones():
    mapper = FloorMapper(ZONES)
    assert mapper.locate(_bbox_centered_at(100, bottom=500)) is None


def test_mapper_from_config_parses_zones():
    cfg = {
        "floors": [
            {"floor": 5, "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]},
            {"floor": 4, "polygon": [[0, 10], [10, 10], [10, 20], [0, 20]]},
        ]
    }
    mapper = FloorMapper.from_config(cfg)
    assert mapper.locate((4, 0, 6, 5)) == 5
    assert mapper.locate((4, 10, 6, 15)) == 4


def test_mapper_from_config_rejects_empty_zones():
    with pytest.raises(ValueError):
        FloorMapper.from_config({"floors": []})


def test_zone_requires_at_least_three_points():
    with pytest.raises(ValueError):
        FloorZone(floor=1, polygon=[(0, 0), (1, 1)])
