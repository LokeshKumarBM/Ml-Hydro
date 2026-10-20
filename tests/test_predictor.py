"""Unit tests for the prediction logic. Run from the project root: pytest"""

import math

import pytest

from backend.predictor import FatiguePredictor, InvalidConditionError


@pytest.fixture(scope="module")
def predictor():
    return FatiguePredictor()


def valid_request(**overrides):
    request = {
        "material": "7475",
        "environment": "H2: 103 / RT",
        "pressure_mpa": 103.0,
        "delta_k_mpa_sqrt_m": 8.0,
        "r_ratio": 0.3,
        "frequency_hz": 1.0,
        "orientation": "SL/ST",
        "test_method": "ASTM E647",
    }
    request.update(overrides)
    return request


def test_valid_request_returns_positive_rate(predictor):
    result = predictor.predict(valid_request())
    assert result["status"] == "experimental_estimate"
    assert result["notes"] == []
    assert 0 < result["predicted_da_dn_m_per_cycle"] < 1
    assert math.isclose(
        10 ** result["predicted_log10_da_dn"],
        result["predicted_da_dn_m_per_cycle"],
        rel_tol=1e-3,
    )


def test_hydrogen_accelerates_crack_growth_in_x42_steel(predictor):
    # In the measured data, X42 grows cracks ~40x faster in H2 than in N2.
    # (7475 aluminium does not show this: in the data it is slightly
    # slower in dry H2 than in lab air, so it is not used here.)
    base = valid_request(
        material="X42",
        pressure_mpa=6.9,
        delta_k_mpa_sqrt_m=12.0,
        r_ratio=0.1,
        orientation="CL",
        test_method="ASTM E647-83",
    )
    h2 = predictor.predict({**base, "environment": "H2: 6.9 / RT", "frequency_hz": 1.0})
    n2 = predictor.predict({**base, "environment": "N2: 6.9 / RT", "frequency_hz": 5.0})
    assert h2["predicted_da_dn_m_per_cycle"] > 5 * n2["predicted_da_dn_m_per_cycle"]


def test_rate_increases_with_delta_k(predictor):
    low = predictor.predict(valid_request(delta_k_mpa_sqrt_m=5.0))
    high = predictor.predict(valid_request(delta_k_mpa_sqrt_m=11.0))
    assert high["predicted_da_dn_m_per_cycle"] > low["predicted_da_dn_m_per_cycle"]


def test_unknown_environment_is_rejected(predictor):
    # The original dashboard sent this string; the model never saw it.
    with pytest.raises(InvalidConditionError):
        predictor.predict(valid_request(environment="H2 103 MPa RT"))


def test_untested_material_environment_pair_is_rejected(predictor):
    with pytest.raises(InvalidConditionError):
        predictor.predict(valid_request(material="X42"))


def test_unknown_orientation_is_rejected(predictor):
    with pytest.raises(InvalidConditionError):
        predictor.predict(valid_request(orientation="L-T"))


def test_out_of_range_delta_k_is_flagged(predictor):
    result = predictor.predict(valid_request(delta_k_mpa_sqrt_m=40.0))
    assert result["status"] == "extrapolated_estimate"
    assert any("Delta K" in note for note in result["notes"])


def test_options_list_every_tested_condition(predictor):
    options = predictor.options()
    pairs = {(c["material"], c["environment"]) for c in options["tested_conditions"]}
    assert ("4130X", "H2: 45 / RT") in pairs
    assert len(pairs) == 5
