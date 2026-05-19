import pandas as pd

from ehr_sentinel import ConsensusRule, SeasonFilter, SustainedRule, EpidemicPredictor


def _toy_mase():
    weeks = pd.date_range("2020-01-06", periods=20, freq="W-MON")
    rows = []
    for w in weeks:
        for g in ("Respiratory", "Cardiovascular", "Diabetes"):
            base = 1.0
            spike = 3.0 if g == "Respiratory" and w >= pd.Timestamp("2020-02-10") and w <= pd.Timestamp("2020-03-30") else 0.0
            rows.append({"week": w, "group": g, "S": base + spike})
    return pd.DataFrame(rows)


def test_consensus_and_sustained():
    mase = _toy_mase()
    cr = ConsensusRule(k=1, threshold_sd=1.0).fit(mase)
    alerts = cr.evaluate(mase)
    assert alerts["alert"].sum() > 0
    season = SeasonFilter(months=[2, 3]).apply(alerts)
    sustained = SustainedRule(n_weeks=2).apply(season)
    assert "alert_sustained" in sustained.columns
    assert sustained["alert_sustained"].sum() > 0


def test_epidemic_predictor():
    weeks = pd.date_range("2020-01-06", periods=20, freq="W-MON")
    lgdi = pd.DataFrame({"week": weeks, "lgdi": [0.1] * 5 + [2.5] * 10 + [0.1] * 5})
    mase = _toy_mase()
    pred = EpidemicPredictor("COVID-19", alert_threshold_sd=1.0)
    warn = pred.generate_warning(lgdi, mase)
    assert warn.onset_week is not None
    assert warn.peak_week_estimate is not None
    assert len(warn.at_risk_groups) == 3
    assert warn.at_risk_groups[0][0] == "Respiratory"
