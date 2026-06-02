/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file   SmartProjectionPoseFactor.h
 * @brief  Smart factor on poses, assuming camera calibration is fixed
 * @author Luca Carlone
 * @author Chris Beall
 * @author Zsolt Kira
 */

#pragma once

#include <gtsam/slam/SmartProjectionFactor.h>
#include <gtsam/nonlinear/FixableFactor.h>

namespace gtsam {
/**
 *
 * @ingroup slam
 *
 * If you are using the factor, please cite:
 * L. Carlone, Z. Kira, C. Beall, V. Indelman, F. Dellaert, Eliminating conditionally
 * independent sets in factor graphs: a unifying perspective based on smart factors,
 * Int. Conf. on Robotics and Automation (ICRA), 2014.
 *
 */

/**
 * This factor assumes that camera calibration is fixed, and that
 * the calibration is the same for all cameras involved in this factor.
 * The factor only constrains poses (variable dimension is 6).
 * This factor requires that values contains the involved poses (Pose3).
 * If the calibration should be optimized, as well, use SmartProjectionFactor instead!
 * @ingroup slam
 */
template <class CALIBRATION>
class SmartProjectionPoseFactor
    : public SmartProjectionFactor<PinholePose<CALIBRATION> >,
      public FixableFactor {
 private:
  typedef PinholePose<CALIBRATION> Camera;
  typedef SmartProjectionFactor<Camera> Base;
  typedef SmartProjectionPoseFactor<CALIBRATION> This;
  typedef typename Camera::Measurement Z;              ///< single measurement
  typedef typename Camera::MeasurementVector ZVector;  ///< measurement vector

protected:

  std::shared_ptr<CALIBRATION> K_; ///< calibration object (one for all cameras)

  /// @name Anchored (fixed) poses, see doc/SmartFactorFixPose.md
  /// @{
  KeyVector fixedKeys_;  ///< original keys of the anchored poses
  std::vector<Pose3, Eigen::aligned_allocator<Pose3> >
      fixedPoses_;         ///< anchored world_P_body values
  ZVector fixedMeasured_;  ///< measurements of the anchored views
  /// @}

public:

  /// shorthand for a smart pointer to a factor
  typedef std::shared_ptr<This> shared_ptr;

  /**
   * Default constructor, only for serialization
   */
  SmartProjectionPoseFactor() {}

  /**
   * Constructor
   * @param sharedNoiseModel isotropic noise model for the 2D feature measurements
   * @param K (fixed) calibration, assumed to be the same for all cameras
   * @param params parameters for the smart projection factors
   */
  SmartProjectionPoseFactor(
      const SharedNoiseModel& sharedNoiseModel,
      const std::shared_ptr<CALIBRATION> K,
      const SmartProjectionParams& params = SmartProjectionParams())
      : Base(sharedNoiseModel, params), K_(K) {
  }

  /**
   * Constructor
   * @param sharedNoiseModel isotropic noise model for the 2D feature measurements
   * @param K (fixed) calibration, assumed to be the same for all cameras
   * @param body_P_sensor pose of the camera in the body frame (optional)
   * @param params parameters for the smart projection factors
   */
  SmartProjectionPoseFactor(
      const SharedNoiseModel& sharedNoiseModel,
      const std::shared_ptr<CALIBRATION> K,
      const std::optional<Pose3> body_P_sensor,
      const SmartProjectionParams& params = SmartProjectionParams())
      : SmartProjectionPoseFactor(sharedNoiseModel, K, params) {
    this->body_P_sensor_ = body_P_sensor;
  }

  /** Virtual destructor */
  ~SmartProjectionPoseFactor() override {
  }

  /**
   * print
   * @param s optional string naming the factor
   * @param keyFormatter optional formatter useful for printing Symbols
   */
  void print(const std::string& s = "", const KeyFormatter& keyFormatter =
      DefaultKeyFormatter) const override {
    std::cout << s << "SmartProjectionPoseFactor, z = \n ";
    for (size_t i = 0; i < fixedKeys_.size(); i++) {
      std::cout << "fixed pose for key " << keyFormatter(fixedKeys_[i]) << ":\n";
      fixedPoses_[i].print();
    }
    Base::print("", keyFormatter);
  }

  /// equals
  bool equals(const NonlinearFactor& p, double tol = 1e-9) const override {
    const This *e = dynamic_cast<const This*>(&p);
    if (!e || !Base::equals(p, tol)) return false;
    if (fixedKeys_ != e->fixedKeys_ ||
        fixedMeasured_.size() != e->fixedMeasured_.size()) return false;
    for (size_t i = 0; i < fixedPoses_.size(); i++) {
      if (!fixedPoses_[i].equals(e->fixedPoses_[i], tol) ||
          !traits<Z>::Equals(fixedMeasured_[i], e->fixedMeasured_[i], tol))
        return false;
    }
    return true;
  }

  /**
   * error calculates the error of the factor.
   */
  double error(const Values& values) const override {
    if (!this->active(values)) return 0.0;
    if (fixedPoses_.empty())
      return this->totalReprojectionError(cameras(values));
    // Triangulate from all (live + anchored) cameras; anchored views contribute.
    const typename Base::Cameras cameras = allCameras(values);
    const ZVector measured = allMeasured();
    this->result_ =
        gtsam::triangulateSafe(cameras, measured, this->params_.triangulation);
    if (!this->result_) return 0.0;
    Vector e = cameras.reprojectionError(*this->result_, measured);
    this->noiseModel_->whitenInPlace(e);
    return 0.5 * e.dot(e);
  }

  /** return calibration shared pointers */
  inline const std::shared_ptr<CALIBRATION> calibration() const {
    return K_;
  }

  /**
   * Collect all cameras involved in this factor
   * @param values Values structure which must contain camera poses corresponding
   * to keys involved in this factor
   * @return vector of Values
   */
  typename Base::Cameras cameras(const Values& values) const override {
    typename Base::Cameras cameras;
    for (const Key& k : this->keys_) {
      const Pose3 world_P_sensor_k =
          Base::body_P_sensor_ ? values.at<Pose3>(k) * *Base::body_P_sensor_
                               : values.at<Pose3>(k);
      cameras.emplace_back(world_P_sensor_k, K_);
    }
    return cameras;
  }

  /// @name Fixing (anchoring) poses
  /// @{

  /// Return true if this factor has any anchored (fixed) pose.
  bool hasFixedPoses() const { return !fixedPoses_.empty(); }

  /// Keys of the poses that have been fixed in this factor.
  const KeyVector& fixedKeys() const { return fixedKeys_; }

  /// Cameras for the anchored poses, built at their stored world_P_body values.
  typename Base::Cameras fixedCameras() const {
    typename Base::Cameras cameras;
    cameras.reserve(fixedPoses_.size());
    for (const Pose3& world_P_body : fixedPoses_) {
      const Pose3 world_P_sensor =
          Base::body_P_sensor_ ? world_P_body * *Base::body_P_sensor_
                               : world_P_body;
      cameras.emplace_back(world_P_sensor, K_);
    }
    return cameras;
  }

  /**
   * Fix the pose with the given key at the supplied value, returning a new
   * factor that no longer depends on that key. The view's measurement is
   * retained (it still constrains the landmark and the surviving poses) but the
   * pose is conditioned (held constant) instead of being a variable.
   *
   * @param key         the pose key to fix (must be one of this->keys())
   * @param world_P_body the value at which to anchor the pose
   * @return a new SmartProjectionPoseFactor over the remaining keys
   */
  shared_ptr fixPose(Key key, const Pose3& world_P_body) const {
    const auto it = std::find(this->keys_.begin(), this->keys_.end(), key);
    if (it == this->keys_.end())
      throw std::invalid_argument(
          "SmartProjectionPoseFactor::fixPose: key is not used by this factor");
    const size_t idx = std::distance(this->keys_.begin(), it);

    // Move (key, measurement) from the live set into the anchored set.
    auto result = std::make_shared<This>(*this);
    result->fixedKeys_.push_back(key);
    result->fixedPoses_.push_back(world_P_body);
    result->fixedMeasured_.push_back(this->measured_[idx]);
    result->keys_.erase(result->keys_.begin() + idx);
    result->measured_.erase(result->measured_.begin() + idx);
    result->cameraPosesTriangulation_.clear();  // camera count changed
    return result;
  }

  /**
   * FixableFactor interface: fix every key of this factor that also appears in
   * `keysToFix` at its value in `values`. Returns the reduced factor, or
   * nullptr if no live key remains (the factor would reduce to a constant).
   */
  NonlinearFactor::shared_ptr fixKeys(const KeyVector& keysToFix,
                                      const Values& values) const override {
    const KeySet toFix(keysToFix.begin(), keysToFix.end());
    shared_ptr current;
    bool anyFixed = false;
    for (const Key k : this->keys_) {
      if (!toFix.exists(k)) continue;
      const Pose3& world_P_body = values.at<Pose3>(k);
      current = anyFixed ? current->fixPose(k, world_P_body)
                         : this->fixPose(k, world_P_body);
      anyFixed = true;
    }
    if (!anyFixed) return std::make_shared<This>(*this);  // nothing to fix
    if (current->keys().empty()) return nullptr;          // fully fixed: drop
    return current;
  }

  /// @}

  /// Anchored factors linearize to a Hessian; otherwise defer to the base.
  std::shared_ptr<GaussianFactor> linearize(
      const Values& values) const override {
    if (fixedPoses_.empty()) return Base::linearize(values);
    return createHessianFactorFixed(values);
  }

  /**
   * Linearize to a Hessian factor over the live keys, with the anchored poses
   * fixed at their stored values. The landmark is triangulated from all (live +
   * anchored) cameras and Schur-complemented out; the anchored pose blocks are
   * then dropped (conditioning at the fixed value), keeping the residual energy.
   */
  std::shared_ptr<RegularHessianFactor<Base::Dim> > createHessianFactorFixed(
      const Values& values, const double lambda = 0.0,
      bool diagonalDamping = false) const {
    const size_t nrLive = this->keys_.size();
    const size_t m = nrLive + fixedPoses_.size();

    // Live cameras first, then anchored cameras, so the live blocks form a
    // prefix of the Schur-complement matrix (see SelectLiveBlocks).
    typename Base::Cameras cameras = allCameras(values);
    const ZVector measured = allMeasured();

    this->result_ =
        gtsam::triangulateSafe(cameras, measured, this->params_.triangulation);

    if (!this->result_) {  // degenerate: zero Hessian over the live keys
      std::vector<Matrix> Gs(nrLive * (nrLive + 1) / 2,
                             Matrix::Zero(Base::Dim, Base::Dim));
      std::vector<Vector> gs(nrLive, Vector::Zero(Base::Dim));
      return std::make_shared<RegularHessianFactor<Base::Dim> >(this->keys_, Gs,
                                                                gs, 0.0);
    }

    typename Base::FBlocks Fs;
    Matrix E;
    Vector b;
    computeJacobiansFixed(Fs, E, b, cameras, measured);
    this->whitenJacobians(Fs, E, b);

    // Schur-complement the landmark over all m cameras, then drop the anchored
    // pose blocks to obtain the augmented Hessian over the live keys only.
    Eigen::Matrix<double, 3, 3> P;
    Base::Cameras::template ComputePointCovariance<3>(P, E, lambda,
                                                      diagonalDamping);
    const SymmetricBlockMatrix augmentedHessian =
        Base::Cameras::template SchurComplement<3, Base::Dim>(Fs, E, P, b);

    return std::make_shared<RegularHessianFactor<Base::Dim> >(
        this->keys_, SelectLiveBlocks(augmentedHessian, nrLive, m));
  }

 protected:
  /// Cameras for the live keys (from values) followed by the anchored cameras.
  typename Base::Cameras allCameras(const Values& values) const {
    typename Base::Cameras cameras = this->cameras(values);
    const typename Base::Cameras fixed = fixedCameras();
    cameras.insert(cameras.end(), fixed.begin(), fixed.end());
    return cameras;
  }

  /// Live measurements followed by the anchored measurements (same order as
  /// allCameras()).
  ZVector allMeasured() const {
    ZVector measured = this->measured_;
    measured.insert(measured.end(), fixedMeasured_.begin(),
                    fixedMeasured_.end());
    return measured;
  }

  /**
   * Like the base-class Jacobian computation, but takes an explicit measurement
   * vector so the anchored views can be included alongside the live ones.
   */
  void computeJacobiansFixed(typename Base::FBlocks& Fs, Matrix& E, Vector& b,
                             const typename Base::Cameras& cameras,
                             const ZVector& measured) const {
    b = -cameras.reprojectionError(*this->result_, measured, Fs, E);
    // Chain rule to the body pose for a sensor offset (cf. unwhitenedError).
    if (Base::body_P_sensor_) {
      const Pose3 sensor_P_body = Base::body_P_sensor_->inverse();
      for (size_t i = 0; i < Fs.size(); i++) {
        const Pose3 world_P_body = cameras[i].pose() * sensor_P_body;
        Eigen::Matrix<double, 6, 6> H;
        world_P_body.compose(*Base::body_P_sensor_, H);
        Fs[i] = Fs[i] * H;
      }
    }
  }

  /**
   * Keep the first nrLive pose blocks and the trailing info block of the full
   * augmented Hessian, dropping the anchored blocks in between. This is the
   * conditioning step derived in doc/SmartFactorFixPose.md.
   */
  static SymmetricBlockMatrix SelectLiveBlocks(
      const SymmetricBlockMatrix& full, size_t nrLive, size_t m) {
    std::vector<DenseIndex> dims(nrLive + 1, Base::Dim);
    dims.back() = 1;
    SymmetricBlockMatrix reduced(
        dims, Matrix::Zero(Base::Dim * nrLive + 1, Base::Dim * nrLive + 1));
    for (size_t i = 0; i < nrLive; i++) {
      reduced.setDiagonalBlock(i, full.block(i, i));
      for (size_t j = i + 1; j < nrLive; j++)
        reduced.setOffDiagonalBlock(i, j, full.block(i, j));
      reduced.setOffDiagonalBlock(i, nrLive, full.block(i, m));  // info vector
    }
    reduced.setDiagonalBlock(nrLive, full.block(m, m));  // constant term f
    return reduced;
  }

 private:

#if GTSAM_ENABLE_BOOST_SERIALIZATION  ///
  /// Serialization function
  friend class boost::serialization::access;
  template<class ARCHIVE>
  void serialize(ARCHIVE & ar, const unsigned int /*version*/) {
    ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Base);
    ar & BOOST_SERIALIZATION_NVP(K_);
    ar & BOOST_SERIALIZATION_NVP(fixedKeys_);
    ar & BOOST_SERIALIZATION_NVP(fixedPoses_);
    ar & BOOST_SERIALIZATION_NVP(fixedMeasured_);
  }
#endif
};
// end of class declaration

/// traits
template<class CALIBRATION>
struct traits<SmartProjectionPoseFactor<CALIBRATION> > : public Testable<
    SmartProjectionPoseFactor<CALIBRATION> > {
};

} // \ namespace gtsam
