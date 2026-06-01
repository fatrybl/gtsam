/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 *  @file  testSmartProjectionFactorFixPose.cpp
 *  @brief Unit tests for SmartProjectionPoseFactor::fixPose (anchoring a pose).
 *         See doc/SmartFactorFixPose.md for the underlying math.
 *  @author Mark
 *  @date   2026
 */

#include "smartFactorScenarios.h"
#include <gtsam/linear/HessianFactor.h>
#include <CppUnitLite/TestHarness.h>

using namespace std;
using namespace gtsam;

namespace {
// Pixel noise model
static const double sigma = 0.1;
static const SharedIsotropic model(noiseModel::Isotropic::Sigma(2, sigma));

using symbol_shorthand::X;

// A factor observing landmark1 from three poses, plus matching Values.
struct ThreeViewFixture {
  vanillaPose::SmartFactor::shared_ptr factor;
  Values values;
  Point2 z1, z2, z3;

  ThreeViewFixture() {
    using namespace vanillaPose;
    z1 = cam1.project(landmark1);
    z2 = cam2.project(landmark1);
    z3 = cam3.project(landmark1);
    factor = std::make_shared<SmartFactor>(model, sharedK);
    factor->add(z1, X(1));
    factor->add(z2, X(2));
    factor->add(z3, X(3));
    values.insert(X(1), level_pose);
    values.insert(X(2), pose_right);
    values.insert(X(3), pose_above);
  }
};
}  // namespace

/* ************************************************************************* */
// fixPose removes the fixed key from the live set and records it as anchored.
TEST(SmartProjectionFactorFixPose, Structure) {
  ThreeViewFixture f;
  auto fixed = f.factor->fixPose(X(1), f.values.at<Pose3>(X(1)));

  EXPECT_LONGS_EQUAL(2, fixed->keys().size());
  EXPECT(fixed->keys()[0] == X(2));
  EXPECT(fixed->keys()[1] == X(3));
  EXPECT(fixed->hasFixedPoses());
  EXPECT_LONGS_EQUAL(1, fixed->fixedKeys().size());
  EXPECT(fixed->fixedKeys()[0] == X(1));

  // Original factor is untouched.
  EXPECT_LONGS_EQUAL(3, f.factor->keys().size());
  EXPECT(!f.factor->hasFixedPoses());

  // Fixing an unused key throws.
  CHECK_EXCEPTION(f.factor->fixPose(X(9), Pose3()), std::invalid_argument);
}

/* ************************************************************************* */
// The nonlinear error is preserved: anchoring a pose at its current value and
// evaluating over the remaining keys equals the full factor's error.
TEST(SmartProjectionFactorFixPose, ErrorConsistency) {
  ThreeViewFixture f;
  const double fullError = f.factor->error(f.values);

  auto fixed = f.factor->fixPose(X(1), f.values.at<Pose3>(X(1)));
  Values live;
  live.insert(X(2), f.values.at<Pose3>(X(2)));
  live.insert(X(3), f.values.at<Pose3>(X(3)));

  EXPECT_DOUBLES_EQUAL(fullError, fixed->error(live), 1e-9);
}

/* ************************************************************************* */
// Core math: linearizing the fixed factor equals the full factor's augmented
// Hessian with the fixed pose's block dropped (conditioning at the value).
TEST(SmartProjectionFactorFixPose, LinearizationEqualsConditioned) {
  ThreeViewFixture f;

  auto Hfull = std::dynamic_pointer_cast<HessianFactor>(
      f.factor->linearize(f.values));
  CHECK(Hfull);
  const Matrix augFull = Hfull->augmentedInformation();  // [x1 x2 x3 | b]

  // Fix x1; linearize over {x2, x3}.
  auto fixed = f.factor->fixPose(X(1), f.values.at<Pose3>(X(1)));
  Values live;
  live.insert(X(2), f.values.at<Pose3>(X(2)));
  live.insert(X(3), f.values.at<Pose3>(X(3)));
  auto Hfixed = std::dynamic_pointer_cast<HessianFactor>(fixed->linearize(live));
  CHECK(Hfixed);

  // Expected = augFull with the first (x1) 6x6 block dropped: keep [x2 x3 | b].
  const Matrix expected = augFull.block(6, 6, 13, 13);
  EXPECT(assert_equal(expected, Hfixed->augmentedInformation(), 1e-6));
}

/* ************************************************************************* */
// The fixed factor remains nonlinear: it re-triangulates from the anchored +
// live cameras as the live poses move.
TEST(SmartProjectionFactorFixPose, ReTriangulation) {
  ThreeViewFixture f;
  const Pose3 anchor = f.values.at<Pose3>(X(1));
  auto fixed = f.factor->fixPose(X(1), anchor);

  // Perturb the live poses.
  const Pose3 pose2b =
      f.values.at<Pose3>(X(2)) *
      Pose3(Rot3::Rodrigues(0.01, -0.02, 0.015), Point3(0.05, -0.03, 0.02));
  const Pose3 pose3b =
      f.values.at<Pose3>(X(3)) *
      Pose3(Rot3::Rodrigues(-0.02, 0.01, 0.005), Point3(-0.04, 0.02, 0.01));

  Values moved;
  moved.insert(X(2), pose2b);
  moved.insert(X(3), pose3b);

  // Reference: full factor with x1 held at the anchor and the moved live poses.
  Values ref;
  ref.insert(X(1), anchor);
  ref.insert(X(2), pose2b);
  ref.insert(X(3), pose3b);

  EXPECT_DOUBLES_EQUAL(f.factor->error(ref), fixed->error(moved), 1e-9);
}

/* ************************************************************************* */
// fixPose can be applied repeatedly as more poses leave the window.
TEST(SmartProjectionFactorFixPose, ChainedFixing) {
  ThreeViewFixture f;
  const Matrix augFull = std::dynamic_pointer_cast<HessianFactor>(
                             f.factor->linearize(f.values))
                             ->augmentedInformation();

  auto fixed1 = f.factor->fixPose(X(1), f.values.at<Pose3>(X(1)));
  auto fixed2 = fixed1->fixPose(X(2), f.values.at<Pose3>(X(2)));

  EXPECT_LONGS_EQUAL(1, fixed2->keys().size());
  EXPECT(fixed2->keys()[0] == X(3));
  EXPECT_LONGS_EQUAL(2, fixed2->fixedKeys().size());

  Values onlyX3;
  onlyX3.insert(X(3), f.values.at<Pose3>(X(3)));

  // Error still matches the full factor.
  EXPECT_DOUBLES_EQUAL(f.factor->error(f.values), fixed2->error(onlyX3), 1e-9);

  // Linearization equals augFull restricted to the x3 block + b.
  auto Hfixed2 =
      std::dynamic_pointer_cast<HessianFactor>(fixed2->linearize(onlyX3));
  CHECK(Hfixed2);
  const Matrix expected = augFull.block(12, 12, 7, 7);  // [x3 | b]
  EXPECT(assert_equal(expected, Hfixed2->augmentedInformation(), 1e-6));
}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
/* ************************************************************************* */
