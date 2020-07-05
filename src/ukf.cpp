#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

auto normAngle = [](double ang) {
    while (ang > M_PI)
      ang -= 2. * M_PI;
    while (ang < -M_PI)
      ang += 2. * M_PI;
    return ang;
  };
/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.8;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;

  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
   * End DO NOT MODIFY section for measurement noise values
   */

  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */

  // move this to list initializer at the beginning
  is_initialized_ = false;

  n_x_ = 5;
  n_aug_ = n_x_ + 2;
  lambda_ = 3 - n_aug_;

  uint num_sp = n_aug_ * 2 + 1;
  P_ = MatrixXd::Identity(n_x_, n_x_);
  P_(0, 0) = std_laspx_*std_laspx_;
  P_(1, 1) = std_laspy_*std_laspy_;
  P_(2, 2) = 0.5;
  P_(3, 3) = 0.5;
  P_(4, 4) = 0.5;
  //P_(4, 4) = 10;
  Xsig_pred_ = MatrixXd(n_x_, num_sp);
  weights_ = VectorXd::Constant(num_sp, 0.5 / 3);
  weights_(0) = lambda_ / 3;

  time_us_ = 0;

  Q_ = MatrixXd(2, 2);
  Q_ << std_a_ * std_a_, 0, 0, std_yawdd_ * std_yawdd_;

  R_rad_ = MatrixXd::Zero(3, 3);
  R_rad_(0, 0) = std_radr_ * std_radr_;
  R_rad_(1, 1) = std_radphi_ * std_radphi_;
  R_rad_(2, 2) = std_radrd_ * std_radrd_;

  R_las_ = MatrixXd::Zero(2, 2);
  R_las_(0, 0) = std_laspx_ * std_laspx_;
  R_las_(1, 1) = std_laspy_ * std_laspy_;

  H_las_ = MatrixXd::Zero(2, n_x_);
  H_las_(0, 0) = 1;
  H_las_(1, 1) = 1;
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  if (!is_initialized_) {
    if (meas_package.sensor_type_ == meas_package.LASER) {
      x_ << meas_package.raw_measurements_[0],
          meas_package.raw_measurements_[1], 0, 0, 0;
    } else {
      auto radius = meas_package.raw_measurements_[0];
      auto theta = meas_package.raw_measurements_[1];
      // auto v = meas_package.raw_measurements_[2];
      // auto vx = v*cos(theta);
      // auto vy = v*sin(theta);
      x_ << radius * cos(theta),
            radius * sin(theta),
            0,
            0,
            0;
    }

    is_initialized_ = true;
    time_us_ = meas_package.timestamp_;
    return; // ?
  }
  auto delta_t = meas_package.timestamp_ - time_us_;
  time_us_ = meas_package.timestamp_;

  Prediction(delta_t / 1.0e6f);
  if (meas_package.sensor_type_ == meas_package.LASER) {
    if (use_laser_) UpdateLidar(meas_package);
  } else {
    if (use_radar_) UpdateRadar(meas_package);
  }
  return ;
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location.
   * Modify the state vector, x_. Predict sigma points, the state,
   * and the state covariance matrix.
   */

  // create augmented mean vector
  VectorXd x_aug = VectorXd(7);
  // create augmented state covariance
  MatrixXd P_aug = MatrixXd(7, 7);

  // create augmented mean state
  x_aug << x_, 0, 0;
  // create augmented covariance matrix
  P_aug.block(0, 0, 5, 5) = P_;
  P_aug.block(0, 5, 2, 2) = MatrixXd::Zero(2, 2);
  P_aug.block(5, 0, 2, 2) = MatrixXd::Zero(2, 2);
  P_aug.block(5, 5, 2, 2) = Q_;
  // calculate square root of P
  MatrixXd A = P_aug.llt().matrixL();

  // calculate sigma points
  // set first column of sigma point matrix
  MatrixXd Xsig_aug = Eigen::MatrixXd::Zero(n_aug_, n_aug_ * 2 + 1);
  Xsig_aug.col(0) = x_aug;
  // set remaining sigma points
  for (int i = 0; i < n_aug_; ++i) {
    Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * A.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * A.col(i);
  }

  // predict sigma points
  Xsig_pred_ = Xsig_aug.block(0, 0, 5, 2 * n_aug_ + 1);
  Xsig_pred_.row(0) +=
      (Xsig_aug.row(5).array() * Xsig_aug.row(3).array().cos()).matrix() * 0.5 *
      delta_t * delta_t;
  Xsig_pred_.row(1) +=
      (Xsig_aug.row(5).array() * Xsig_aug.row(3).array().sin()).matrix() * 0.5 *
      delta_t * delta_t;
  Xsig_pred_.row(2) += Xsig_aug.row(5) * delta_t;
  Xsig_pred_.row(3) += Xsig_aug.row(6) * delta_t * delta_t *0.5;
  Xsig_pred_.row(4) += Xsig_aug.row(6) * delta_t;

  for (int i = 0; i <= 2 * n_aug_; i++) {
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yawd = Xsig_aug(4, i);
    if (fabs(yawd) <= 1e-3f) {
      Xsig_pred_(0, i) += v * cos(yaw) * delta_t;
      Xsig_pred_(1, i) += v * sin(yaw) * delta_t;
      // Xsig_pred_(3,i) += yawd*delta_t;
    } else {
      Xsig_pred_(0, i) += v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      Xsig_pred_(1, i) += v / yawd * (-cos(yaw + yawd * delta_t) + cos(yaw));
      Xsig_pred_(3, i) += yawd * delta_t;
      Xsig_pred_(3, i) = normAngle(Xsig_pred_(3, i));
    }
  }

  // predict state mean
  x_ = Xsig_pred_ * weights_;
    
  x_(3) = normAngle(x_(3));
  // predict state covariance matrix
  MatrixXd centered = Xsig_pred_.colwise() - x_;
  for (int i = 0; i < n_aug_*2 + 1; i++)
    centered(3, i) = normAngle(centered(3, i));

  P_ = (centered.array().rowwise() * weights_.transpose().array()).matrix() *
       centered.transpose();

  std::cout << "Predicted x: \n" << x_ << "\n";
  std::cout << "Predicted P: \n" << P_ << "\n";

  // int a;
  // std::cin >> a;
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief
   * about the object's position. Modify the state vector, x_, and
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */

  MatrixXd z_pred = H_las_ * x_;
  VectorXd z = meas_package.raw_measurements_; // 2 rows

  MatrixXd Ht = H_las_.transpose();
  MatrixXd S = H_las_ * P_ * Ht + R_las_;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_ * Ht * Si;

  x_ = x_ + K * (z - z_pred);
  P_ = (MatrixXd::Identity(n_x_, n_x_) - K * H_las_) * P_;
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief
   * about the object's position. Modify the state vector, x_, and
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

  /**
   * Student part begin
   */
  const int n_z = 3;
  VectorXd z = meas_package.raw_measurements_;
  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);

  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);

  // transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) { // 2n+1 simga points
    // extract values for better readability
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;

    // measurement model
    Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y);                         // r
    Zsig(1, i) = atan2(p_y, p_x);                                     // phi
    Zsig(2, i) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y); // r_dot
  }

  // mean predicted measurement
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // innovation covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) { // 2n+1 simga points
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    z_diff(1) = normAngle(z_diff(1));

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // add measurement noise covariance matrix
  S = S + R_rad_;

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  // calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) { // 2n+1 simga points
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // angle normalization
    z_diff(1) = normAngle(z_diff(1));

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    x_diff(3) = normAngle(x_diff(3));

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // residual
  VectorXd z_diff = z - z_pred;
  z_diff(1) = normAngle(z_diff(1));

  // update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
}