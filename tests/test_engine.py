import unittest
import numpy as np
import pandas as pd
import engine as E


class TestOmniTwinEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_df = pd.read_csv("data/omnitwin_pune_historical.csv")
        cls.df, cls.notes = E.prepare_df(cls.raw_df)

    def test_data_preparation(self):
        self.assertGreater(len(self.df), 1000)
        self.assertEqual(self.df["station"].nunique(), 5)
        for col in E.NUM_COLS:
            self.assertIn(col, self.df.columns)
            self.assertFalse(self.df[col].isnull().any())

    def test_series_extraction_and_dust_proxy(self):
        s = E.get_series(self.df, "Shivajinagar")
        self.assertIn("dust", s.columns)
        self.assertTrue((s["dust"] >= 0).all())
        self.assertEqual(len(s), 1096)

    def test_forecaster_fit_and_prediction(self):
        s = E.get_series(self.df, "Hadapsar")
        model = E.fit_forecaster(s, "Ridge regression")
        origin = s.index[-15]
        fc = E.forecast(model, s, origin)

        self.assertEqual(len(fc), E.HORIZON)
        self.assertTrue((fc > 0).all())
        self.assertTrue((fc.index > origin).all())

    def test_attribution_and_non_negativity(self):
        origin = self.df["date"].max() - pd.Timedelta(days=15)
        attr = E.fit_attribution(self.df, origin)
        self.assertEqual(len(attr.coef_), 3)
        self.assertTrue((attr.coef_ >= 0).all(), "Coefficients must be non-negative")
        self.assertGreaterEqual(attr.intercept_, 0.0)

    def test_action_scenario_reductions(self):
        s = E.get_series(self.df, "Pimpri-Chinchwad")
        model = E.fit_forecaster(s, "Ridge regression")
        origin = s.index[-15]
        fc = E.forecast(model, s, origin)
        attr = E.fit_attribution(self.df, origin)

        # Action 1: Odd-Even (Vehicular -30%)
        fc_act1 = E.apply_actions(fc, attr, s, {"Vehicular": 0.30})
        self.assertLess(fc_act1.mean(), fc.mean())

        # Action 2: Halt Heavy Industry (Industrial -80%)
        fc_act2 = E.apply_actions(fc, attr, s, {"Industrial": 0.80})
        self.assertLess(fc_act2.mean(), fc.mean())

        # Action 3: Sprinklers (Dust -40%)
        fc_act3 = E.apply_actions(fc, attr, s, {"Dust/Weather": 0.40})
        self.assertLess(fc_act3.mean(), fc.mean())

        # Combined actions should reduce even more
        fc_all = E.apply_actions(fc, attr, s, {
            "Vehicular": 0.30,
            "Industrial": 0.80,
            "Dust/Weather": 0.40
        })
        self.assertLess(fc_all.mean(), fc_act1.mean())
        self.assertLess(fc_all.mean(), fc_act2.mean())

    def test_backtest_scoring(self):
        s = E.get_series(self.df, "Kothrud")
        model = E.fit_forecaster(s, "Ridge regression")
        bt = E.backtest(model, s)
        self.assertGreater(len(bt), 0)
        self.assertIn("pred", bt.columns)
        self.assertIn("actual", bt.columns)
        self.assertIn("persistence", bt.columns)

        scores = E.score(bt)
        self.assertIn("OmniTwin model", scores.index)
        self.assertIn("Persistence baseline", scores.index)
        model_mae = scores.loc["OmniTwin model", "MAE (µg/m³)"]
        persist_mae = scores.loc["Persistence baseline", "MAE (µg/m³)"]
        self.assertLess(model_mae, persist_mae, "Model MAE must outperform baseline persistence")


if __name__ == "__main__":
    unittest.main()
