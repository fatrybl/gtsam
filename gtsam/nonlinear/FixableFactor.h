/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    FixableFactor.h
 * @brief   Interface for factors that can fix (condition) a subset of their
 *          variables at given values, returning a factor over the remaining
 *          variables.
 * @author  Mark
 * @date    2026
 */

#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/Values.h>

namespace gtsam {

/**
 * Mixin interface for factors that can "fix" (condition) some of their
 * variables at given values, baking the fixed variables' contribution into a
 * new factor over the remaining (live) variables.
 *
 * The canonical implementer is SmartProjectionPoseFactor (see
 * doc/SmartFactorFixPose.md). A fixed-lag smoother uses this to retire an
 * out-of-window pose from a smart factor without discarding the landmark
 * information contributed by the surviving views: rather than deleting the
 * factor (losing the surviving views) or marginalizing it into a frozen linear
 * factor (losing re-triangulation), it fixes the retiring pose at its current
 * estimate and keeps a smaller, still-nonlinear factor.
 *
 * The interface is deliberately independent of the concrete value type so that
 * the smoother (in gtsam/nonlinear) does not depend on any SLAM type: values
 * are passed through a Values container.
 */
class FixableFactor {
 public:
  virtual ~FixableFactor() = default;

  /**
   * Return a copy of this factor with every key of this factor that also
   * appears in @p keysToFix conditioned at its value in @p values. The result
   * depends only on the remaining (live) keys.
   *
   * @param keysToFix the keys to fix (only those that this factor uses are
   *        acted upon; others are ignored)
   * @param values    must contain a value for every key of this factor that is
   *        being fixed
   * @return a new factor over the surviving keys, or nullptr if no live key
   *         remains (the factor then reduces to a constant and can be dropped)
   */
  virtual NonlinearFactor::shared_ptr fixKeys(const KeyVector& keysToFix,
                                              const Values& values) const = 0;
};

}  // namespace gtsam
