/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 *  @file  testSmartFactorBatchFixedLagSmoother.cpp
 *  @brief Integration test: SmartProjectionPoseFactor + BatchFixedLagSmoother,
 *         exercising the fixPose-on-marginalization path.
 *         See doc/SmartFactorFixPose.md.
 *  @author Mark
 *  @date   2026
 */

#include <gtsam/slam/SmartProjectionPoseFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/BatchFixedLagSmoother.h>
#include <gtsam/geometry/PinholePose.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <CppUnitLite/TestHarness.h>

#include <vector>

using namespace std;
using namespace gtsam;
using symbol_shorthand::X;

namespace {
typedef PinholePose<Cal3_S2> Camera;
typedef SmartProjectionPoseFactor<Cal3_S2> SmartFactor;

// A short sideways-translating trajectory looking down the world +Z axis, so
// that the landmarks (in front of the cameras) have parallax for triangulation.
struct Scenario {
  Cal3_S2::shared_ptr K;
  SharedIsotropic pixelNoise;
  SharedDiagonal odoNoise, priorNoise;
  vector<Pose3> gt;            // ground-truth poses
  Point3 lmA, lmB, lmC;        // landmarks

  Scenario()
      : K(new Cal3_S2(500, 500, 0, 320, 240)),
        pixelNoise(noiseModel::Isotropic::Sigma(2, 1.0)),
        odoNoise(noiseModel::Diagonal::Sigmas(
            (Vector(6) << Vector3::Constant(0.01), Vector3::Constant(0.05))
                .finished())),
        priorNoise(noiseModel::Diagonal::Sigmas(
            (Vector(6) << Vector3::Constant(1e-3), Vector3::Constant(1e-3))
                .finished())),
        lmA(0.0, 1.0, 5.0),
        lmB(1.5, -1.0, 6.0),
        lmC(3.0, 0.5, 5.5) {
    for (int i = 0; i < 5; i++)
      gt.push_back(Pose3(Rot3(), Point3(i * 1.0, 0.0, 0.0)));
  }

  Point2 project(int poseIdx, const Point3& L) const {
    return Camera(gt[poseIdx], K).project(L);
  }

  SmartFactor::shared_ptr makeSmart(const Point3& L,
                                    const vector<int>& poseIdxs) const {
    auto f = std::make_shared<SmartFactor>(pixelNoise, K);
    for (int i : poseIdxs) f->add(project(i, L), X(i));
    return f;
  }
};

// Count factors that are still SmartProjectionPoseFactors (i.e. not collapsed
// into a linear marginal), and how many of those have fixed poses.
void countSmartFactors(const NonlinearFactorGraph& graph, size_t* nSmart,
                       size_t* nFixed) {
  *nSmart = 0;
  *nFixed = 0;
  for (const auto& f : graph) {
    auto s = std::dynamic_pointer_cast<SmartFactor>(f);
    if (!s) continue;
    (*nSmart)++;
    if (s->hasFixedPoses()) (*nFixed)++;
  }
}

// Drive the smoother across the 5-pose trajectory, adding smart factors as
// their last observing pose arrives. Lag = 2 forces X(0), X(1) to marginalize.
void runTrajectory(const Scenario& s, BatchFixedLagSmoother* smoother) {
  for (int i = 0; i < 5; i++) {
    NonlinearFactorGraph newFactors;
    Values newValues;
    FixedLagSmoother::KeyTimestampMap ts;

    if (i == 0) {
      newFactors.addPrior(X(0), s.gt[0], s.priorNoise);
    } else {
      newFactors.emplace_shared<BetweenFactor<Pose3> >(
          X(i - 1), X(i), s.gt[i - 1].between(s.gt[i]), s.odoNoise);
    }
    newValues.insert(X(i), s.gt[i]);
    ts[X(i)] = static_cast<double>(i);

    if (i == 2) newFactors.add(s.makeSmart(s.lmA, {0, 1, 2}));
    if (i == 3) newFactors.add(s.makeSmart(s.lmB, {1, 2, 3}));
    if (i == 4) newFactors.add(s.makeSmart(s.lmC, {2, 3, 4}));

    smoother->update(newFactors, newValues, ts);
  }
}
}  // namespace

/* ************************************************************************* */
// With fixing enabled (default), marginalizing a pose used by a smart factor
// does not crash, keeps accurate estimates, and the surviving smart factors
// remain nonlinear (with the retired poses fixed).
TEST(SmartFactorBatchFixedLagSmoother, FixingEnabled) {
  Scenario s;
  BatchFixedLagSmoother smoother(2.0);
  CHECK(smoother.getFixSmartFactorsOnMarginalize());

  runTrajectory(s, &smoother);

  const Values estimate = smoother.calculateEstimate();

  // X(0) and X(1) have been marginalized; X(2..4) survive.
  EXPECT(!estimate.exists(X(0)));
  EXPECT(!estimate.exists(X(1)));
  EXPECT(estimate.exists(X(2)));
  EXPECT(estimate.exists(X(3)));
  EXPECT(estimate.exists(X(4)));

  // Estimates stay accurate (noise-free data).
  EXPECT(assert_equal(s.gt[2], estimate.at<Pose3>(X(2)), 1e-3));
  EXPECT(assert_equal(s.gt[3], estimate.at<Pose3>(X(3)), 1e-3));
  EXPECT(assert_equal(s.gt[4], estimate.at<Pose3>(X(4)), 1e-3));

  // All three landmark smart factors survive as (re-triangulatable) smart
  // factors; factors A and B have had retired poses fixed.
  size_t nSmart = 0, nFixed = 0;
  countSmartFactors(smoother.getFactors(), &nSmart, &nFixed);
  EXPECT_LONGS_EQUAL(3, nSmart);
  EXPECT_LONGS_EQUAL(2, nFixed);
}

/* ************************************************************************* */
// With fixing disabled we recover the legacy behavior: smart factors touching
// a marginalized pose collapse into linear marginals (so fewer survive as
// SmartProjectionPoseFactors). Estimates are still accurate.
TEST(SmartFactorBatchFixedLagSmoother, FixingDisabled) {
  Scenario s;
  BatchFixedLagSmoother smoother(2.0);
  smoother.setFixSmartFactorsOnMarginalize(false);

  runTrajectory(s, &smoother);

  const Values estimate = smoother.calculateEstimate();
  EXPECT(assert_equal(s.gt[2], estimate.at<Pose3>(X(2)), 1e-3));
  EXPECT(assert_equal(s.gt[3], estimate.at<Pose3>(X(3)), 1e-3));
  EXPECT(assert_equal(s.gt[4], estimate.at<Pose3>(X(4)), 1e-3));

  // Only factor C never touched a marginalized pose, so it is the only smart
  // factor left; A and B were frozen into linear marginals.
  size_t nSmart = 0, nFixed = 0;
  countSmartFactors(smoother.getFactors(), &nSmart, &nFixed);
  EXPECT_LONGS_EQUAL(1, nSmart);
  EXPECT_LONGS_EQUAL(0, nFixed);
}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
/* ************************************************************************* */
