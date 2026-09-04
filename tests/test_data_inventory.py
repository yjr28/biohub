"""Unit tests for metadata-only data inventory behavior."""

import pytest

from biohub.data.inventory import DEFAULT_SCALE, DataInventoryError, _parse_scale


def test_missing_multiscales_uses_organizer_default_scale() -> None:
    assert _parse_scale({}) == DEFAULT_SCALE


def test_ome_scale_uses_last_three_spatial_axes() -> None:
    attrs = {
        "multiscales": [
            {
                "datasets": [
                    {
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0, 1.625, 0.40625, 0.40625]}
                        ]
                    }
                ]
            }
        ]
    }
    assert _parse_scale(attrs) == pytest.approx((1.625, 0.40625, 0.40625))


def test_unsupported_transform_fails_closed() -> None:
    attrs = {
        "multiscales": [
            {
                "datasets": [
                    {"coordinateTransformations": [{"type": "translation", "translation": [0, 0, 0, 0]}]}
                ]
            }
        ]
    }
    with pytest.raises(DataInventoryError, match="Unsupported coordinate transform"):
        _parse_scale(attrs)
