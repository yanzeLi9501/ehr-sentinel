"""Pytest fixtures — synthetic data only."""
from __future__ import annotations

import pytest

from ehr_sentinel import EpidemicConfig, generate_admissions, generate_epidemic_signal, generate_fhir_bundle


@pytest.fixture(scope="session")
def synthetic_admissions():
    return generate_admissions(n_patients=200, n_admissions_per_patient=(2, 20),
                               start_date="2016-01-01", end_date="2022-12-31", seed=42)


@pytest.fixture(scope="session")
def synthetic_admissions_with_signal(synthetic_admissions):
    return generate_epidemic_signal(
        synthetic_admissions,
        target_icd10=["U07.1"],
        outbreak_start="2020-01-15",
        outbreak_end="2020-04-30",
        target_group="Respiratory",
        effect_size=0.6,
        seed=7,
    )


@pytest.fixture(scope="session")
def synthetic_fhir_bundle():
    return generate_fhir_bundle(n_patients=30, seed=42, start_date="2019-01-01", end_date="2021-12-31")


@pytest.fixture(scope="session")
def synthetic_config_covid():
    return EpidemicConfig(
        target_disease="COVID-19",
        reference_icd10_codes=["U07.1", "U07.2"],
        reference_years=[2020],
        reference_months=[1, 2, 3, 4],
        baseline_start="2016-01-01",
        baseline_end="2018-12-31",
        monitoring_start="2019-01-01",
        target_group="Respiratory",
        epidemic_season_months=list(range(1, 13)),
        min_visit_order=2,
    )


@pytest.fixture(scope="session")
def synthetic_config_flu():
    return EpidemicConfig(
        target_disease="Influenza",
        reference_icd10_codes=["J11.1", "J10"],
        reference_years=[2018, 2019],
        reference_months=[1, 2, 12],
        baseline_start="2016-01-01",
        baseline_end="2017-12-31",
        monitoring_start="2018-01-01",
        target_group="Respiratory",
        epidemic_season_months=[11, 12, 1, 2, 3],
        min_visit_order=2,
    )
