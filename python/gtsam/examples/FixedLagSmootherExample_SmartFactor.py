"""
GTSAM Copyright 2010-2026, Georgia Tech Research Corporation,
Atlanta, Georgia 30332-0415
All Rights Reserved
Authors: Frank Dellaert, et al. (see THANKS for the full author list)

See LICENSE for the license information

Smart projection factors in the fixed-lag smoothers, with landmark tracks
that outlive the lag window. Python version of
examples/FixedLagSmootherExample_SmartFactor.cpp.

A vehicle drives a loop with a forward-looking camera. Landmarks are placed at
random; a landmark is measured when it lies in the field of view and the
detector does not drop it, so tracks start, pause, and end at random. Each
landmark is one SmartProjectionPoseFactor that grows with the track. A factor
inside the smoother is never modified: a clone holding all measurements
replaces the old one through factorsToRemove (the replace protocol).

When the oldest pose of a track leaves the window, the smoother either
marginalizes it (MARGINALIZE, the default: the factor is consumed into a
linear marginal and the track restarts) or conditions on it (CONDITION: the
factor survives with that camera held at its estimate). Both modes are run on
both smoothers and compared with the full batch solution. See
gtsam/slam/doc/SmartFactorsFixedLag.md.
"""

# pylint: disable=invalid-name, no-name-in-module, no-member

import math
import sys

import numpy as np

import gtsam
from gtsam.symbol_shorthand import X


class Simulation:
    """Trajectory, landmarks, and the measurements they produce."""

    num_frames = 60
    frame_interval = 0.5  # seconds
    loop_radius = 25.0  # meters
    num_landmarks = 120
    landmark_radius = (20.0, 40.0)  # meters from the center
    landmark_height = (1.0, 4.0)  # meters above ground
    visible_range = (2.0, 40.0)  # least depth, greatest distance; meters
    dropout = 0.3  # probability that the detector misses a landmark
    image_size = (640.0, 480.0)

    def __init__(self, seed=7):
        rng = np.random.default_rng(seed)
        width, height = self.image_size
        self.K = gtsam.Cal3_S2(450.0, 450.0, 0.0, width / 2, height / 2)
        # Camera looks along the body x axis.
        self.body_P_camera = gtsam.Pose3(
            gtsam.Rot3.Ypr(-math.pi / 2, 0.0, -math.pi / 2),
            gtsam.Point3(0.2, 0.0, 0.5),
        )
        self.pixel_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
        odometry_sigmas = np.array([0.004, 0.004, 0.004, 0.02, 0.02, 0.02])
        self.odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(odometry_sigmas)
        self.prior_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)

        # A loop on the ground, heading along the tangent.
        self.truth = []
        for i in range(self.num_frames):
            angle = 2 * math.pi * i / self.num_frames
            position = gtsam.Point3(
                self.loop_radius * math.cos(angle),
                self.loop_radius * math.sin(angle),
                0.0,
            )
            self.truth.append(
                gtsam.Pose3(gtsam.Rot3.Yaw(angle + math.pi / 2), position)
            )
        # Landmarks scattered around the loop.
        min_radius, max_radius = self.landmark_radius
        min_height, max_height = self.landmark_height
        self.landmarks = []
        for _ in range(self.num_landmarks):
            angle, radius = (
                2 * math.pi * rng.uniform(),
                min_radius + (max_radius - min_radius) * rng.uniform(),
            )
            self.landmarks.append(
                gtsam.Point3(
                    radius * math.cos(angle),
                    radius * math.sin(angle),
                    min_height + (max_height - min_height) * rng.uniform(),
                )
            )
        # Visibility: in front of the camera, in range, inside the image, and
        # not missed by the detector.
        min_range, max_range = self.visible_range
        self.measurements = []  # per frame: {landmark: pixel}
        for i in range(self.num_frames):
            camera = gtsam.PinholePoseCal3_S2(
                self.truth[i].compose(self.body_P_camera), self.K
            )
            frame = {}
            for j, landmark in enumerate(self.landmarks):
                local = camera.pose().transformTo(landmark)
                if (
                    local[2] < min_range
                    or np.linalg.norm(local) > max_range
                    or rng.uniform() < self.dropout
                ):
                    continue
                pixel = camera.project(landmark) + rng.normal(size=2)
                if 0 <= pixel[0] <= width and 0 <= pixel[1] <= height:
                    frame[j] = pixel
            self.measurements.append(frame)
        # Odometry with noise, integrated for the initial values.
        self.odometry = [None]
        self.dead_reckoning = [self.truth[0]]
        for i in range(1, self.num_frames):
            noise = odometry_sigmas * rng.normal(size=6)
            step = self.truth[i - 1].between(self.truth[i])
            self.odometry.append(step.compose(gtsam.Pose3.Expmap(noise)))
            self.dead_reckoning.append(
                self.dead_reckoning[i - 1].compose(self.odometry[i])
            )

    def frame_factors(self, i, graph, values, previous_estimate=None):
        """Prior or odometry factor, and the initial value of frame i: the
        previous estimate composed with the odometry step if given."""
        if i == 0:
            graph.push_back(
                gtsam.PriorFactorPose3(X(0), self.truth[0], self.prior_noise)
            )
        else:
            graph.push_back(
                gtsam.BetweenFactorPose3(
                    X(i - 1), X(i), self.odometry[i], self.odometry_noise
                )
            )
        if previous_estimate is None:
            initial = self.dead_reckoning[i]
        else:
            initial = previous_estimate.compose(self.odometry[i])
        values.insert(X(i), initial)

    def new_smart_factor(self):
        return gtsam.SmartProjectionPoseFactorCal3_S2(
            self.pixel_noise, self.K, self.body_P_camera
        )

    def full_batch(self, up_to):
        """Full batch solution with all measurements up to frame up_to."""
        graph, initial = gtsam.NonlinearFactorGraph(), gtsam.Values()
        for i in range(up_to + 1):
            self.frame_factors(i, graph, initial)
        factors = {}
        for i in range(up_to + 1):
            for j, pixel in self.measurements[i].items():
                factors.setdefault(j, self.new_smart_factor()).add(pixel, X(i))
        for factor in factors.values():
            if factor.size() >= 2:
                graph.push_back(factor)
        return gtsam.LevenbergMarquardtOptimizer(graph, initial).optimize()


class Track:
    """One landmark as the caller sees it: its factor and its slot."""

    def __init__(self):
        self.factor = None
        self.slot = -1
        self.restarts = 0


def in_graph(track, graph):
    """Whether the track's factor is the one in the smoother at its slot."""
    return (
        track.slot >= 0
        and track.slot < graph.size()
        and graph.exists(track.slot)
        and graph.at(track.slot).equals(track.factor, 1e-12)
    )


def all_keys_in_window(factor, smoother):
    """Whether every key of the factor still has a timestamp."""
    timestamps = smoother.timestamps()
    return all(key in timestamps for key in factor.keys())


def run(smoother, sim):
    """Run one smoother through the simulation with the replace protocol."""
    tracks = {}
    errors_to_batch, restarts, longest = [], 0, 0
    final_error_to_truth = 0.0
    comparison_interval = 5  # frames between batch comparisons

    for i in range(sim.num_frames):
        new_factors, new_values = gtsam.NonlinearFactorGraph(), gtsam.Values()
        previous_estimate = (
            smoother.calculateEstimatePose3(X(i - 1)) if i > 0 else None
        )
        sim.frame_factors(i, new_factors, new_values, previous_estimate)
        timestamps = {X(i): i * sim.frame_interval}
        to_remove = []

        # Replace protocol: a copy with all measurements replaces the factor.
        graph = smoother.getFactors()
        pending = []  # (landmark, index in new_factors)
        for j, pixel in sim.measurements[i].items():
            track = tracks.setdefault(j, Track())
            present = in_graph(track, graph)
            if track.factor is not None and track.slot >= 0 and not present:
                track.factor, restarts = (
                    None,
                    restarts + 1,
                )  # consumed by marginalization
            pending_factor = track.factor is not None and track.slot < 0
            if pending_factor and not all_keys_in_window(
                track.factor, smoother
            ):
                track.factor, restarts = (
                    None,
                    restarts + 1,
                )  # seen once, pose left since
            factor = (
                track.factor.clone()
                if track.factor is not None
                else sim.new_smart_factor()
            )
            factor.add(pixel, X(i))
            if present:
                to_remove.append(track.slot)
            track.factor, track.slot = factor, -1
            if factor.size() >= 2:
                pending.append((j, new_factors.size()))
                new_factors.push_back(factor)

        caller_factors = new_factors.size()
        result = smoother.update(
            new_factors, new_values, timestamps, to_remove
        )

        # Our factors' slots, then the slots of conditioned replacements.
        new_slots = result.getNewFactorsIndices()
        for j, index in pending:
            tracks[j].slot = new_slots[index]
        for k, old_slot in enumerate(result.getConditionedFactorIndices()):
            for track in tracks.values():
                if track.slot == old_slot:
                    track.slot = new_slots[caller_factors + k]
        graph = smoother.getFactors()
        for track in tracks.values():
            if track.slot < 0:
                continue
            factor = graph.at(track.slot) if graph.exists(track.slot) else None
            if isinstance(factor, gtsam.SmartProjectionPoseFactorCal3_S2):
                track.factor = factor
                longest = max(
                    longest, factor.dim() // 2
                )  # two rows per measurement
            else:
                track.factor, track.slot, restarts = None, -1, restarts + 1

        # Compare the newest pose with the full batch solution every 5 frames.
        if (i + 1) % comparison_interval == 0:
            estimate = smoother.calculateEstimatePose3(X(i))
            reference = sim.full_batch(i).atPose3(X(i))
            errors_to_batch.append(
                np.linalg.norm(
                    estimate.translation() - reference.translation()
                )
            )
            if i + 1 == sim.num_frames:
                truth = sim.truth[i].translation()
                final_error_to_truth = np.linalg.norm(
                    estimate.translation() - truth
                )
    return dict(
        mean=np.mean(errors_to_batch),
        max=np.max(errors_to_batch),
        final=final_error_to_truth,
        restarts=restarts,
        longest=longest,
    )


def main():
    sim = Simulation()
    total = sum(len(frame) for frame in sim.measurements)
    drift = np.linalg.norm(
        sim.dead_reckoning[-1].translation() - sim.truth[-1].translation()
    )
    print(
        f"Simulation: {sim.num_frames} frames, {len(sim.landmarks)} "
        f"landmarks, {total} measurements, final dead-reckoning error "
        f"{drift:.3f} m"
    )
    batch = (
        sim.full_batch(sim.num_frames - 1)
        .atPose3(X(sim.num_frames - 1))
        .translation()
    )
    batch_error = np.linalg.norm(batch - sim.truth[-1].translation())
    print(f"Full batch final error to truth: {batch_error:.3f} m\n")

    lag_frames = 8
    lag = lag_frames * sim.frame_interval  # seconds
    window_frames = lag_frames + 1  # poses in the window
    max_plausible_error = 2.0  # meters from the full batch

    isam_params = gtsam.ISAM2Params()
    isam_params.setFactorization("CHOLESKY")
    isam_params.relinearizeSkip = 1
    isam_params.setRelinearizeThreshold(0.01)
    isam_params.findUnusedFactorSlots = True

    print(
        f"{'smoother / mode':30}{'mean|fl-batch|':16}{'max|fl-batch|':15}"
        f"{'final|fl-gt|':14}{'restarts':10}longest track"
    )
    plausible = True
    modes = [
        (
            "MARGINALIZE",
            gtsam.FixedLagSmoother.MarginalizationMode.MARGINALIZE,
        ),
        ("CONDITION", gtsam.FixedLagSmoother.MarginalizationMode.CONDITION),
    ]
    for mode_name, mode in modes:
        smoothers = [
            (
                "Incremental",
                gtsam.IncrementalFixedLagSmoother(lag, isam_params),
            ),
            ("Batch", gtsam.BatchFixedLagSmoother(lag)),
        ]
        for smoother_name, smoother in smoothers:
            smoother.setMarginalizationMode(mode)
            s = run(smoother, sim)
            name = f"{smoother_name} / {mode_name}"
            print(
                f"{name:30}{s['mean']:<16.3f}{s['max']:<15.3f}"
                f"{s['final']:<14.3f}{s['restarts']:<10}{s['longest']}"
            )
            # Only CONDITION tracks outlive the window; errors stay bounded.
            if mode_name == "MARGINALIZE" and s["longest"] > window_frames:
                plausible = False
            if mode_name == "CONDITION" and s["longest"] <= window_frames:
                plausible = False
            if s["max"] > max_plausible_error:
                plausible = False
    print(
        "\nMARGINALIZE is exact marginalization but ends each track at the "
        "window;\nCONDITION keeps whole tracks but freezes each pose's "
        "estimation error when it\nis fixed. Which one is closer to the batch "
        "solution depends on the data."
    )
    print("Results plausible." if plausible else "Results NOT plausible.")
    return 0 if plausible else 1


if __name__ == "__main__":
    sys.exit(main())
