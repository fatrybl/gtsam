/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file FixedLagSmootherExample_SmartFactor.cpp
 * @brief Smart projection factors in the fixed-lag smoothers, with landmark
 * tracks that outlive the lag window.
 *
 * A vehicle drives a loop with a forward-looking camera. Landmarks are placed
 * at random; a landmark is measured when it lies in the field of view and the
 * detector does not drop it, so tracks start, pause, and end at random. Each
 * landmark is one SmartProjectionPoseFactor that grows with the track. A factor
 * inside the smoother is never modified: a new factor holding all measurements
 * replaces the old one through factorsToRemove (the replace protocol).
 *
 * When the oldest pose of a track leaves the window, the smoother either
 * marginalizes it (FixedLagSmoother::MARGINALIZE, the default: the factor is
 * consumed into a linear marginal and the track restarts) or conditions on it
 * (FixedLagSmoother::CONDITION: the factor survives with that camera held at
 * its estimate). Both modes are run on both smoothers and compared with the
 * full batch solution. See gtsam/slam/doc/SmartFactorsFixedLag.md.
 */

#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/PinholePose.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/BatchFixedLagSmoother.h>
#include <gtsam/nonlinear/IncrementalFixedLagSmoother.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/SmartProjectionPoseFactor.h>

#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <vector>

using namespace gtsam;
using symbol_shorthand::X;
typedef SmartProjectionPoseFactor<Cal3_S2> SmartFactor;

/* ************************************************************************* */
/// Simulated world: trajectory, landmarks, and the measurements they produce.
struct Simulation {
  size_t numFrames = 60;
  double frameInterval = 0.5;  ///< seconds
  double loopRadius = 25.0;    ///< meters
  size_t numLandmarks = 120;
  /// Landmark distance from the loop center and height above ground, meters.
  double landmarkMinRadius = 20.0, landmarkMaxRadius = 40.0;
  double landmarkMinHeight = 1.0, landmarkMaxHeight = 4.0;
  /// Least depth and greatest distance at which a landmark is seen, meters.
  double minRange = 2.0, maxRange = 40.0;
  /// Probability that the detector misses a visible landmark.
  double dropout = 0.3;
  double imageWidth = 640.0, imageHeight = 480.0;
  Cal3_S2::shared_ptr K = std::make_shared<Cal3_S2>(
      450.0, 450.0, 0.0, imageWidth / 2, imageHeight / 2);
  // The camera looks along the body x axis.
  Pose3 body_P_camera{Rot3::Ypr(-M_PI / 2, 0.0, -M_PI / 2),
                      Point3(0.2, 0.0, 0.5)};
  SharedNoiseModel pixelNoise = noiseModel::Isotropic::Sigma(2, 1.0);
  SharedNoiseModel odometryNoise = noiseModel::Diagonal::Sigmas(
      Vector6{0.004, 0.004, 0.004, 0.02, 0.02, 0.02});
  SharedNoiseModel priorNoise = noiseModel::Isotropic::Sigma(6, 1e-3);

  std::vector<Pose3> truth, odometry, deadReckoning;
  std::vector<Point3> landmarks;
  /// measurements[i] maps each landmark seen in frame i to its pixel.
  std::vector<std::map<size_t, Point2>> measurements;

  explicit Simulation(unsigned seed = 7) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> gauss(0.0, 1.0);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    // A loop on the ground, heading along the tangent.
    for (size_t i = 0; i < numFrames; ++i) {
      const double angle = 2 * M_PI * i / numFrames;
      truth.emplace_back(Rot3::Yaw(angle + M_PI / 2),
                         Point3(loopRadius * std::cos(angle),
                                loopRadius * std::sin(angle), 0.0));
    }
    // Landmarks scattered around the loop.
    for (size_t j = 0; j < numLandmarks; ++j) {
      const double angle = 2 * M_PI * uniform(rng),
                   radius =
                       landmarkMinRadius +
                       (landmarkMaxRadius - landmarkMinRadius) * uniform(rng);
      landmarks.emplace_back(
          radius * std::cos(angle), radius * std::sin(angle),
          landmarkMinHeight +
              (landmarkMaxHeight - landmarkMinHeight) * uniform(rng));
    }
    // Visibility: in front of the camera, in range, inside the image, and not
    // missed by the detector.
    measurements.resize(numFrames);
    for (size_t i = 0; i < numFrames; ++i) {
      const PinholePose<Cal3_S2> camera(truth[i] * body_P_camera, K);
      for (size_t j = 0; j < landmarks.size(); ++j) {
        const Point3 local = camera.pose().transformTo(landmarks[j]);
        if (local.z() < minRange || local.norm() > maxRange ||
            uniform(rng) < dropout)
          continue;
        const double u = gauss(rng), v = gauss(rng);
        const Point2 pixel = camera.project(landmarks[j]) + Point2(u, v);
        if (pixel.x() < 0 || pixel.x() > imageWidth || pixel.y() < 0 ||
            pixel.y() > imageHeight)
          continue;
        measurements[i][j] = pixel;
      }
    }
    // Odometry with noise, integrated for the initial values.
    const Vector6 sigmas = odometryNoise->sigmas();
    odometry.resize(numFrames);
    deadReckoning.push_back(truth[0]);
    for (size_t i = 1; i < numFrames; ++i) {
      Vector6 noise;
      for (int k = 0; k < 6; ++k) noise(k) = sigmas(k) * gauss(rng);
      odometry[i] = truth[i - 1].between(truth[i]) * Pose3::Expmap(noise);
      deadReckoning.push_back(deadReckoning[i - 1] * odometry[i]);
    }
  }

  /// Prior or odometry factor, and the initial value of frame i: the previous
  /// estimate composed with the odometry step if given, else dead reckoning.
  void frameFactors(size_t i, NonlinearFactorGraph* graph, Values* values,
                    const Pose3* previousEstimate = nullptr) const {
    if (i == 0)
      graph->addPrior(X(0), truth[0], priorNoise);
    else
      graph->emplace_shared<BetweenFactor<Pose3>>(X(i - 1), X(i), odometry[i],
                                                  odometryNoise);
    values->insert(X(i), previousEstimate ? *previousEstimate * odometry[i]
                                          : deadReckoning[i]);
  }

  SmartFactor::shared_ptr newSmartFactor() const {
    return std::make_shared<SmartFactor>(pixelNoise, K, body_P_camera);
  }

  /// Full batch solution with all measurements up to frame upTo.
  Values fullBatch(size_t upTo) const {
    NonlinearFactorGraph graph;
    Values initial;
    for (size_t i = 0; i <= upTo; ++i) frameFactors(i, &graph, &initial);
    std::map<size_t, SmartFactor::shared_ptr> factors;
    for (size_t i = 0; i <= upTo; ++i)
      for (const auto& [j, pixel] : measurements[i]) {
        if (!factors.count(j)) factors[j] = newSmartFactor();
        factors[j]->add(pixel, X(i));
      }
    for (const auto& [j, factor] : factors)
      if (factor->size() >= 2) graph.push_back(factor);
    return LevenbergMarquardtOptimizer(graph, initial).optimize();
  }
};

/* ************************************************************************* */
/// The caller's view of one landmark: its factor in the smoother and its slot.
struct Track {
  SmartFactor::shared_ptr factor;
  long slot = -1;
  size_t restarts = 0;
};

/// Whether every key of a factor still has a timestamp in the smoother.
template <class SMOOTHER>
bool allKeysInWindow(const NonlinearFactor& factor, const SMOOTHER& smoother) {
  for (Key key : factor.keys())
    if (!smoother.timestamps().count(key)) return false;
  return true;
}

struct RunSummary {
  double meanErrorToBatch = 0.0, maxErrorToBatch = 0.0, finalErrorToTruth = 0.0;
  size_t restarts = 0, longestTrack = 0;
};

/// Restart a track whose factor is no longer usable.
void restartTrack(Track* track) {
  track->factor.reset();
  track->slot = -1;
  ++track->restarts;
}

/**
 * Replace protocol for one frame: for every landmark seen now, a new factor
 * holding all measurements replaces the previous one. Returns the slots to
 * remove and fills newFactors; pending lists (landmark, index in newFactors).
 */
template <class SMOOTHER>
FactorIndices replaceTrackFactors(
    const SMOOTHER& smoother, const Simulation& sim, size_t frame,
    std::map<size_t, Track>* tracks, NonlinearFactorGraph* newFactors,
    std::vector<std::pair<size_t, size_t>>* pending) {
  FactorIndices toRemove;
  const NonlinearFactorGraph& graph = smoother.getFactors();
  for (const auto& [j, pixel] : sim.measurements[frame]) {
    Track& track = (*tracks)[j];
    const bool inGraph = track.slot >= 0 && size_t(track.slot) < graph.size() &&
                         graph[track.slot] == track.factor;
    // Consumed by marginalization, or seen once and not again until that
    // pose had left the window: start the landmark afresh.
    if (track.factor && track.slot >= 0 && !inGraph) restartTrack(&track);
    if (track.factor && track.slot < 0 &&
        !allKeysInWindow(*track.factor, smoother))
      restartTrack(&track);
    auto factor = sim.newSmartFactor();
    if (track.factor) *factor = *track.factor;  // keeps fixed cameras, if any
    factor->add(pixel, X(frame));
    if (inGraph) toRemove.push_back(size_t(track.slot));
    track.factor = factor;
    track.slot = -1;
    if (factor->size() >= 2) {
      pending->emplace_back(j, newFactors->size());
      newFactors->push_back(factor);
    }
  }
  return toRemove;
}

/**
 * After an update: our factors' slots, then the slots of conditioned
 * replacements, then refresh the factor pointers. A slot that no longer holds
 * a smart factor was consumed by marginalization.
 */
template <class SMOOTHER>
void refreshTracks(const SMOOTHER& smoother,
                   const FixedLagSmoother::Result& result, size_t callerFactors,
                   const std::vector<std::pair<size_t, size_t>>& pending,
                   std::map<size_t, Track>* tracks, size_t* longestTrack) {
  for (const auto& [j, index] : pending)
    (*tracks)[j].slot = long(result.newFactorsIndices.at(index));
  for (size_t k = 0; k < result.conditionedFactorIndices.size(); ++k)
    for (auto& [j, track] : *tracks)
      if (track.slot == long(result.conditionedFactorIndices[k]))
        track.slot = long(result.newFactorsIndices.at(callerFactors + k));
  for (auto& [j, track] : *tracks) {
    if (track.slot < 0) continue;
    track.factor = std::dynamic_pointer_cast<SmartFactor>(
        smoother.getFactors()[track.slot]);
    if (!track.factor)
      restartTrack(&track);
    else
      *longestTrack = std::max(*longestTrack, track.factor->measured().size());
  }
}

/**
 * Run one smoother through the simulation with the replace protocol and
 * report the newest-pose error against the full batch solution.
 */
template <class SMOOTHER>
RunSummary run(SMOOTHER& smoother, const Simulation& sim) {
  std::map<size_t, Track> tracks;
  RunSummary summary;
  size_t comparisons = 0;
  const size_t comparisonInterval = 5;  // frames between batch comparisons

  for (size_t i = 0; i < sim.numFrames; ++i) {
    NonlinearFactorGraph newFactors;
    Values newValues;
    FixedLagSmoother::KeyTimestampMap timestamps;
    const Pose3 previousEstimate =
        i > 0 ? smoother.calculateEstimate().template at<Pose3>(X(i - 1))
              : Pose3();
    sim.frameFactors(i, &newFactors, &newValues,
                     i > 0 ? &previousEstimate : nullptr);
    timestamps[X(i)] = i * sim.frameInterval;

    std::vector<std::pair<size_t, size_t>> pending;
    const FactorIndices toRemove =
        replaceTrackFactors(smoother, sim, i, &tracks, &newFactors, &pending);
    const size_t callerFactors = newFactors.size();
    const FixedLagSmoother::Result result =
        smoother.update(newFactors, newValues, timestamps, toRemove);
    refreshTracks(smoother, result, callerFactors, pending, &tracks,
                  &summary.longestTrack);

    // Compare the newest pose with the full batch solution.
    if ((i + 1) % comparisonInterval == 0) {
      const Pose3 estimate =
          smoother.calculateEstimate().template at<Pose3>(X(i));
      const Pose3 reference = sim.fullBatch(i).at<Pose3>(X(i));
      const double error =
          (estimate.translation() - reference.translation()).norm();
      summary.meanErrorToBatch += error;
      summary.maxErrorToBatch = std::max(summary.maxErrorToBatch, error);
      ++comparisons;
      if (i + 1 == sim.numFrames)
        summary.finalErrorToTruth =
            (estimate.translation() - sim.truth[i].translation()).norm();
    }
  }
  summary.meanErrorToBatch /= comparisons;
  for (const auto& [j, track] : tracks) summary.restarts += track.restarts;
  return summary;
}

/* ************************************************************************* */
int main(int argc, char* argv[]) {
  const Simulation sim(argc > 1 ? unsigned(std::atoi(argv[1]))
                                : 7u);  // optional seed
  size_t total = 0;
  for (const auto& frame : sim.measurements) total += frame.size();
  const Point3 finalTruth = sim.truth.back().translation();
  std::cout << "Simulation: " << sim.numFrames << " frames, "
            << sim.landmarks.size() << " landmarks, " << total
            << " measurements, final dead-reckoning error "
            << (sim.deadReckoning.back().translation() - finalTruth).norm()
            << " m\n";
  const Values batch = sim.fullBatch(sim.numFrames - 1);
  const Point3 finalBatch = batch.at<Pose3>(X(sim.numFrames - 1)).translation();
  std::cout << "Full batch final error to truth: "
            << (finalBatch - finalTruth).norm() << " m\n\n";

  const size_t lagFrames = 8;
  const double lag = lagFrames * sim.frameInterval;  // seconds
  const size_t windowFrames = lagFrames + 1;         // poses in the window
  const double maxPlausibleError = 2.0;  // meters from the full batch
  ISAM2Params isamParams;
  isamParams.findUnusedFactorSlots = true;
  isamParams.relinearizeSkip = 1;
  isamParams.relinearizeThreshold = 0.01;

  std::cout << std::fixed << std::setprecision(3) << std::left << std::setw(30)
            << "smoother / mode" << std::setw(16) << "mean|fl-batch|"
            << std::setw(15) << "max|fl-batch|" << std::setw(14)
            << "final|fl-gt|" << std::setw(10) << "restarts"
            << "longest track\n";
  bool plausible = true;
  for (auto mode :
       {FixedLagSmoother::MARGINALIZE, FixedLagSmoother::CONDITION}) {
    const std::string modeName =
        mode == FixedLagSmoother::MARGINALIZE ? "MARGINALIZE" : "CONDITION";
    IncrementalFixedLagSmoother incremental(lag, isamParams);
    incremental.setMarginalizationMode(mode);
    BatchFixedLagSmoother batchSmoother(lag);
    batchSmoother.setMarginalizationMode(mode);
    const std::vector<std::pair<std::string, RunSummary>> runs = {
        {"Incremental / " + modeName, run(incremental, sim)},
        {"Batch / " + modeName, run(batchSmoother, sim)}};
    for (const auto& [name, summary] : runs) {
      std::cout << std::setw(30) << name << std::setw(16)
                << summary.meanErrorToBatch << std::setw(15)
                << summary.maxErrorToBatch << std::setw(14)
                << summary.finalErrorToTruth << std::setw(10)
                << summary.restarts << summary.longestTrack << "\n";
      // Tracks outlive the window only with CONDITION; errors stay bounded.
      if (mode == FixedLagSmoother::MARGINALIZE &&
          summary.longestTrack > windowFrames)
        plausible = false;
      if (mode == FixedLagSmoother::CONDITION &&
          summary.longestTrack <= windowFrames)
        plausible = false;
      if (summary.maxErrorToBatch > maxPlausibleError) plausible = false;
    }
  }
  std::cout << "\nMARGINALIZE is exact marginalization but ends each track at "
               "the window;\n"
               "CONDITION keeps whole tracks but freezes each pose's "
               "estimation error when it\n"
               "is fixed. Which one is closer to the batch solution depends on "
               "the data.\n";
  std::cout << (plausible ? "Results plausible.\n"
                          : "Results NOT plausible.\n");
  return plausible ? 0 : 1;
}
