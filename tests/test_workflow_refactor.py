"""Offline regression tests for workflow options and treatment timing.

Run with: python -m unittest discover -s tests -v
Requires numpy and pandas, the same libraries used by the treatment pipeline.
"""

import argparse
import unittest

import functions
from run import validate_task_options


class TaskOptionTests(unittest.TestCase):
    def options(self, **updates):
        values = {
            "distance_miles": None,
            "array_worker_id": None,
            "array_worker_count": None,
            "array_run_id": None,
        }
        values.update(updates)
        return argparse.Namespace(**values)

    def test_non_array_task(self):
        validate_task_options(self.options(), ["build_wui_singlepart_dataset"])

    def test_worker_arguments(self):
        task = "build_fire_wui_treatment_definitions_rolling5y_mixed_fires"
        validate_task_options(
            self.options(array_worker_id=0, array_worker_count=3, array_run_id="trial"),
            [task],
        )
        with self.assertRaises(SystemExit):
            validate_task_options(
                self.options(array_worker_id=0, array_worker_count=3),
                [task],
            )

    def test_fixed_distance_variant(self):
        task = "build_fire_wui_treatment_definitions_rolling5y_mixed_fires_1p5mi"
        with self.assertRaises(SystemExit):
            validate_task_options(self.options(distance_miles=3.0), [task])

    def test_finalizer_requires_run_metadata(self):
        task = "finalize_fire_wui_treatment_definitions"
        with self.assertRaises(SystemExit):
            validate_task_options(self.options(), [task])


class TreatmentTimingTests(unittest.TestCase):
    def test_scalar_and_vectorized_memberships_agree(self):
        windows = functions.build_treatment_time_windows(
            rolling_years=[5],
            lag_bands=[("lag_1_2y", 1, 2)],
        )
        cases = [
            ("2022-06-15", 2022, "exact"),
            ("2023-08-01", 2023, "exact"),
            ("2017-01-01", 2017, "exact"),
            (None, 2022, "calendar_year"),
            (None, 2023, "calendar_year"),
            (None, 2022, "fiscal_year"),
            (None, 2022, "award_year_proxy"),
        ]
        fire_date = "2023-07-15"
        fire_year = 2023
        matrix = functions.treatment_window_membership_matrix(
            [case[0] for case in cases],
            [case[1] for case in cases],
            [case[2] for case in cases],
            fire_date,
            fire_year,
            windows,
        )
        for i, (completion_date, completion_year, precision) in enumerate(cases):
            for j, window in enumerate(windows):
                with self.subTest(case=i, window=window["name"]):
                    scalar = functions.treatment_window_membership(
                        completion_date,
                        completion_year,
                        fire_date,
                        fire_year,
                        window,
                        date_precision=precision,
                    )
                    self.assertEqual(matrix[i, j], scalar)

    def test_known_time_boundaries_and_uncertainty(self):
        windows = {
            item["name"]: item
            for item in functions.build_treatment_time_windows(
                rolling_years=[5], lag_bands=[("lag_1_2y", 1, 2)]
            )
        }
        classify = functions.treatment_window_membership
        event = ("2023-07-15", 2023)
        self.assertTrue(
            classify("2022-06-15", 2022, *event, windows["previous_calendar_year"])
        )
        self.assertFalse(
            classify("2023-08-01", 2023, *event, windows["rolling_5y"])
        )
        self.assertTrue(
            classify(None, 2022, *event, windows["previous_calendar_year"],
                     date_precision="calendar_year")
        )
        self.assertIsNone(
            classify(None, 2022, *event, windows["lag_1_2y"],
                     date_precision="calendar_year")
        )
        self.assertIsNone(
            classify(None, 2022, *event, windows["rolling_5y"],
                     date_precision="award_year_proxy")
        )


if __name__ == "__main__":
    unittest.main()
