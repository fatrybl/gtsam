/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 *  @file  testSmartFactorIncrementalFixedLagSmoother.cpp
 *  @brief Integration test: SmartProjectionPoseFactor + IncrementalFixedLagSmoother
 *         (iSAM2-based), exercising the fixPose-on-marginalization path.
 *         See doc/SmartFactorFixPose.md.
 *  @author Mark
 *  @date   2026
 */

#include <gtsam/slam/SmartProjectionPoseFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/IncrementalFixedLagSmoother.h>
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

// Same scenario as the batch test: a sideways-translating trajectory looking
// down +Z, so the landmarks in front of the cameras have triangulation
// parallax.
struct Scenario {
  Cal3_S2::shared_ptr K;
  SharedIsotropic pixelNoise;
  SharedDiagonal odoNoise, priorNoise;
  vector<Pose3> gt;
  Point3 lmA, lmB, lmC;

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

void runTrajectory(const Scenario& s, IncrementalFixedLagSmoother* smoother) {
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
// Marginalizing a pose used by a smart factor in the iSAM2-based smoother does
// not crash, keeps accurate estimates, and the surviving smart factors remain
// nonlinear with the retired poses fixed.
TEST(SmartFactorIncrementalFixedLagSmoother, FixingEnabled) {
  Scenario s;
  IncrementalFixedLagSmoother smoother(2.0);
  CHECK(smoother.getFixSmartFactorsOnMarginalize());

  runTrajectory(s, &smoother);

  const Values estimate = smoother.calculateEstimate();
  EXPECT(!estimate.exists(X(0)));
  EXPECT(!estimate.exists(X(1)));

  EXPECT(assert_equal(s.gt[2], smoother.calculateEstimate<Pose3>(X(2)), 1e-3));
  EXPECT(assert_equal(s.gt[3], smoother.calculateEstimate<Pose3>(X(3)), 1e-3));
  EXPECT(assert_equal(s.gt[4], smoother.calculateEstimate<Pose3>(X(4)), 1e-3));

  // Factors A, B, C survive as re-triangulatable smart factors; A and B have
  // had retired poses fixed.
  size_t nSmart = 0, nFixed = 0;
  countSmartFactors(smoother.getISAM2().getFactorsUnsafe(), &nSmart, &nFixed);
  EXPECT_LONGS_EQUAL(3, nSmart);
  EXPECT_LONGS_EQUAL(2, nFixed);
}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
/* ************************************************************************* */
