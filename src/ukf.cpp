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
UKF::UKF()
    : use_laser_(true), use_radar_(true), std_a_(1), std_yawdd_(0.5),
      std_laspx_(0.15), std_laspy_(0.15), std_radr_(0.3), std_radphi_(0.03),
      std_radrd_(0.3), is_initialized_(false), n_x_(5), n_aug_(7), lambda_(-4) {

  // initial state vector
  x_ = VectorXd(n_x_);
  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  P_ = MatrixXd::Identity(n_x_, n_x_);
  P_(0, 0) = std_laspx_ * std_laspx_;
  P_(1, 1) = std_laspy_ * std_laspy_;
  P_(2, 2) = 1;
  P_(3, 3) = 1;
  P_(4, 4) = 1;

  uint num_sp = n_aug_ * 2 + 1;
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
  if (!is_initialized_) {
    if (meas_package.sensor_type_ == meas_package.LASER) {
      x_ << meas_package.raw_measurements_[0],
          meas_package.raw_measurements_[1], 0, 0, 0;
    } else {
      auto radius = meas_package.raw_measurements_[0];
      auto theta = meas_package.raw_measurements_[1];
      x_ << radius * cos(theta), radius * sin(theta), 0, 0, 0;
    }
    is_initialized_ = true;
    time_us_ = meas_package.timestamp_;
    return; // ?
  }

  auto delta_t = meas_package.timestamp_ - time_us_;
  time_us_ = meas_package.timestamp_;
  Prediction(delta_t / 1.0e6f);
  if (meas_package.sensor_type_ == meas_package.LASER && use_laser_) {
    UpdateLidar(meas_package);
  } else if (meas_package.sensor_type_ == meas_package.RADAR && use_radar_) {
    UpdateRadar(meas_package);
  }
  return;
}

void UKF::Prediction(double delta_t) {
  // create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);
  // create augmented state covariance
  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
  MatrixXd Xsig_aug = Eigen::MatrixXd::Zero(n_aug_, n_aug_ * 2 + 1);

  // create augmented mean state
  x_aug << x_, 0, 0;
  // create augmented covariance matrix
  P_aug.block(0, 0, 5, 5) = P_;
  P_aug.block(5, 5, 2, 2) = Q_;
  // calculate square root of P
  MatrixXd A = P_aug.llt().matrixL();

  // calculate sigma points
  // set first column of sigma point matrix
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
  Xsig_pred_.row(3) += Xsig_aug.row(6) * delta_t * delta_t * 0.5;
  Xsig_pred_.row(4) += Xsig_aug.row(6) * delta_t;

  for (int i = 0; i <= 2 * n_aug_; i++) {
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yawd = Xsig_aug(4, i);
    if (fabs(yawd) <= 1e-3f) {
      Xsig_pred_(0, i) += v * cos(yaw) * delta_t;
      Xsig_pred_(1, i) += v * sin(yaw) * delta_t;
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
  centered.row(3) = centered.row(3).unaryExpr(normAngle);
  P_ = (centered.array().rowwise() * weights_.transpose().array()).matrix() *
       centered.transpose();
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  MatrixXd z_pred = H_las_ * x_;
  VectorXd z = meas_package.raw_measurements_; // 2 rows

  // calculate K matrix
  MatrixXd Ht = H_las_.transpose();
  MatrixXd S = H_las_ * P_ * Ht + R_las_;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_ * Ht * Si;

  x_ = x_ + K * (z - z_pred);
  P_ = (MatrixXd::Identity(n_x_, n_x_) - K * H_las_) * P_;
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  const int n_z = 3;
  VectorXd z = meas_package.raw_measurements_;
  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  // mean predicted measurement
  VectorXd z_pred = VectorXd::Zero(n_z);

  // transform sigma points into measurement space
  auto px = Xsig_pred_.row(0).array();
  auto py = Xsig_pred_.row(1).array();
  auto v = Xsig_pred_.row(2).array();
  auto th = Xsig_pred_.row(3).array();
  Zsig.row(0) = (px.pow(2) + py.pow(2)).sqrt().matrix();
  auto atan2_lambda = [](double y, double x) { return atan2(y, x); };
  Zsig.row(1) = py.matrix().binaryExpr(px.matrix(), atan2_lambda);
  Zsig.row(2) =
      (px * v * (th.cos()) + py * v * (th.sin())) / Zsig.row(0).array();

  // mean predicted measurement
  z_pred = Zsig * weights_;

  MatrixXd z_centered = Zsig.colwise() - z_pred;
  z_centered.row(1) = z_centered.row(1).unaryExpr(normAngle);
  // innovation covariance matrix S
  MatrixXd S =
      (z_centered.array().rowwise() * weights_.transpose().array()).matrix() *
          z_centered.transpose() + R_rad_;

  MatrixXd x_centered = Xsig_pred_.colwise() - x_;
  x_centered.row(3) = x_centered.row(3).unaryExpr(normAngle);
  // create matrix for cross correlation Tc
  MatrixXd Tc =
      (x_centered.array().rowwise() * weights_.transpose().array()).matrix() *
      z_centered.transpose();

  // Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // residual
  VectorXd z_diff = z - z_pred;
  z_diff(1) = normAngle(z_diff(1));

  // update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
}