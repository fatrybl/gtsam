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
 * new factor over the remaining (live) variables. The canonical implementer is
 * SmartProjectionPoseFactor (see doc/SmartFactorFixPose.md); a fixed-lag
 * smoother uses it to retire an out-of-window pose from a smart factor while
 * keeping the landmark information of the surviving views.
 *
 * The interface is independent of the concrete value type (values pass through
 * a Values container) so the smoother does not depend on any SLAM type.
 */
class FixableFactor {
 public:
  virtual ~FixableFactor() = default;

  /**
   * Return a copy of this factor with every key it shares with @p keysToFix
   * conditioned at its value in @p values; the result depends only on the
   * remaining (live) keys.
   *
   * @param keysToFix keys to fix (only those used by this factor act)
   * @param values    must hold a value for each fixed key
   * @return a factor over the surviving keys, or nullptr if none remain
   */
  virtual NonlinearFactor::shared_ptr fixKeys(const KeyVector& keysToFix,
                                              const Values& values) const = 0;
};

}  // namespace gtsam
