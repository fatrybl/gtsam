"""
GTSAM Copyright 2010-2026, Georgia Tech Research Corporation,
Atlanta, Georgia 30332-0415
All Rights Reserved

See LICENSE for the license information

Unit tests for SmartProjectionPoseFactor.fixPose / fixKeys and the
fixed-lag smoother integration (setFixSmartFactorsOnMarginalize).
See doc/SmartFactorFixPose.md.

Author: Mark (Python)
"""
import unittest

import numpy as np
from gtsam.utils.test_case import GtsamTestCase

import gtsam
from gtsam import (BatchFixedLagSmoother, Cal3_S2,
                   IncrementalFixedLagSmoother, NonlinearFactorGraph,
                   PinholePoseCal3_S2, Pose3, Rot3,
                   SmartProjectionPoseFactorCal3_S2, Values)
from gtsam.symbol_shorthand import X

# Pixel noise model used by the unit-level tests.
PIXEL = gtsam.noiseModel.Isotropic.Sigma(2, 0.1)


def make_three_view():
    """A factor observing one landmark from three poses, plus its Values."""
    K = Cal3_S2(60.0, 640, 480)
    landmark = np.array([5.0, 0.5, 1.2])
    level = Pose3(Rot3.Ypr(-np.pi / 2, 0.0, -np.pi / 2),
                  np.array([0., 0., 1.]))
    poses = [
        level,
        level.compose(Pose3(Rot3(), np.array([1., 0., 0.]))),
        level.compose(Pose3(Rot3(), np.array([0., -1., 0.]))),
    ]

    factor = SmartProjectionPoseFactorCal3_S2(PIXEL, K)
    values = Values()
    for i, pose in enumerate(poses):
        z = PinholePoseCal3_S2(pose, K).project(landmark)
        factor.add(z, X(i))
        values.insert(X(i), pose)
    return factor, values


class TestSmartFactorFixPose(GtsamTestCase):
    """Tests for the fixPose capability and its smoother integration."""

    def test_fix_pose_structure(self):
        """fixPose drops the fixed key and records it as anchored."""
        factor, values = make_three_view()
        fixed = factor.fixPose(X(0), values.atPose3(X(0)))

        self.assertEqual(len(fixed.keys()), 2)
        self.assertTrue(fixed.hasFixedPoses())
        self.assertEqual(list(fixed.fixedKeys()), [X(0)])
        # Original factor is untouched.
        self.assertEqual(len(factor.keys()), 3)
        self.assertFalse(factor.hasFixedPoses())

    def test_fix_pose_error_preserved(self):
        """Anchoring a pose at its value preserves the nonlinear error."""
        factor, values = make_three_view()
        full_error = factor.error(values)

        fixed = factor.fixPose(X(0), values.atPose3(X(0)))
        live = Values()
        live.insert(X(1), values.atPose3(X(1)))
        live.insert(X(2), values.atPose3(X(2)))
        self.assertAlmostEqual(full_error, fixed.error(live), places=9)

    def test_fix_keys(self):
        """fixKeys conditions every shared key, returning a smaller factor."""
        factor, values = make_three_view()
        fixed = factor.fixKeys([X(0), X(1)], values)
        # Only X(2) remains live.
        self.assertEqual(len(fixed.keys()), 1)
        self.assertEqual(fixed.keys()[0], X(2))


class FixedLagScenario:
    """Sideways-translating trajectory looking down +Z, 3 landmarks."""

    def __init__(self):
        self.K = Cal3_S2(500, 500, 0, 320, 240)
        self.pixel = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
        self.odo = gtsam.noiseModel.Diagonal.Sigmas(
            np.concatenate([np.full(3, 0.01), np.full(3, 0.05)]))
        self.prior = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-3))
        self.gt = [Pose3(Rot3(), np.array([i * 1.0, 0., 0.]))
                   for i in range(5)]
        self.landmarks = {
            'A': (np.array([0.0, 1.0, 5.0]), [0, 1, 2]),
            'B': (np.array([1.5, -1.0, 6.0]), [1, 2, 3]),
            'C': (np.array([3.0, 0.5, 5.5]), [2, 3, 4]),
        }

    def smart(self, landmark, pose_idxs):
        f = SmartProjectionPoseFactorCal3_S2(self.pixel, self.K)
        for i in pose_idxs:
            z = PinholePoseCal3_S2(self.gt[i], self.K).project(landmark)
            f.add(z, X(i))
        return f

    def run(self, smoother):
        for i in range(5):
            new_factors = NonlinearFactorGraph()
            new_values = Values()
            timestamps = {}
            if i == 0:
                new_factors.push_back(
                    gtsam.PriorFactorPose3(X(0), self.gt[0], self.prior))
            else:
                new_factors.push_back(gtsam.BetweenFactorPose3(
                    X(i - 1), X(i),
                    self.gt[i - 1].between(self.gt[i]), self.odo))
            new_values.insert(X(i), self.gt[i])
            timestamps[X(i)] = float(i)
            for _, (lm, idxs) in self.landmarks.items():
                if max(idxs) == i:
                    new_factors.push_back(self.smart(lm, idxs))
            smoother.update(new_factors, new_values, timestamps)


def count_smart(graph):
    """(number of surviving smart factors, number with fixed poses)."""
    n_smart = n_fixed = 0
    for k in range(graph.size()):
        f = graph.at(k)
        if isinstance(f, SmartProjectionPoseFactorCal3_S2):
            n_smart += 1
            if f.hasFixedPoses():
                n_fixed += 1
    return n_smart, n_fixed


class TestSmartFactorBatchFixedLagSmoother(GtsamTestCase):
    """SmartProjectionPoseFactor + BatchFixedLagSmoother."""

    def test_fixing_enabled(self):
        s = FixedLagScenario()
        smoother = BatchFixedLagSmoother(2.0)
        self.assertTrue(smoother.getFixSmartFactorsOnMarginalize())
        s.run(smoother)

        est = smoother.calculateEstimate()
        self.assertFalse(est.exists(X(0)))
        self.assertFalse(est.exists(X(1)))
        for i in (2, 3, 4):
            self.gtsamAssertEquals(s.gt[i], est.atPose3(X(i)), 1e-3)

        # A, B, C all survive as smart factors; A and B had a pose fixed.
        n_smart, n_fixed = count_smart(smoother.getFactors())
        self.assertEqual(n_smart, 3)
        self.assertEqual(n_fixed, 2)

    def test_fixing_disabled(self):
        s = FixedLagScenario()
        smoother = BatchFixedLagSmoother(2.0)
        smoother.setFixSmartFactorsOnMarginalize(False)
        self.assertFalse(smoother.getFixSmartFactorsOnMarginalize())
        s.run(smoother)

        est = smoother.calculateEstimate()
        for i in (2, 3, 4):
            self.gtsamAssertEquals(s.gt[i], est.atPose3(X(i)), 1e-3)
        # Only C never touched a marginalized pose; A and B were frozen.
        n_smart, n_fixed = count_smart(smoother.getFactors())
        self.assertEqual(n_smart, 1)
        self.assertEqual(n_fixed, 0)


class TestSmartFactorIncrementalFixedLagSmoother(GtsamTestCase):
    """SmartProjectionPoseFactor + IncrementalFixedLagSmoother (iSAM2)."""

    def test_fixing_enabled(self):
        s = FixedLagScenario()
        smoother = IncrementalFixedLagSmoother(2.0)
        self.assertTrue(smoother.getFixSmartFactorsOnMarginalize())
        s.run(smoother)

        est = smoother.calculateEstimate()
        self.assertFalse(est.exists(X(0)))
        self.assertFalse(est.exists(X(1)))
        for i in (2, 3, 4):
            self.gtsamAssertEquals(s.gt[i], est.atPose3(X(i)), 1e-3)

        n_smart, n_fixed = count_smart(smoother.getFactors())
        self.assertEqual(n_smart, 3)
        self.assertEqual(n_fixed, 2)


if __name__ == "__main__":
    unittest.main()
