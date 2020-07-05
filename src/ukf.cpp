#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

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
  std_a_ = 3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1;
  
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

  uint num_sp = n_aug_*2 + 1;
  P_ = MatrixXd::Identity(n_x_, n_x_);
  Xsig_pred_ = MatrixXd(n_aug_, num_sp);
  weights_ = VectorXd::Constant(n_aug_, 0.5/(lambda_+n_x_));
  weights_(0) = lambda_ / (lambda_ + n_aug_); 

  time_us_ = 0;

  Q_ << 
    std_a_*std_a_, 0,
    0, std_yawdd_*std_yawdd_;
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
            meas_package.raw_measurements_[1], 
            0,
            0,
            0;
    } else {
      auto radius = meas_package.raw_measurements_[0];
      auto theta = meas_package.raw_measurements_[1];

      x_ << radius*cos(theta),
            radius*sin(theta),
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

  Prediction(delta_t);
  if (meas_package.sensor_type_ == meas_package.LASER) {
    UpdateLidar(meas_package);
  } else {
    UpdateRadar(meas_package);
  }

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
  P_aug.block(0,0,5,5)  = P_;
  P_aug.block(0, 5, 2, 2) = MatrixXd::Zero(2,2);
  P_aug.block(5, 0, 2, 2) = MatrixXd::Zero(2,2);
  P_aug.block(5,5,2,2) = Q_;
  // calculate square root of P
  MatrixXd A = P_aug.llt().matrixL();

  // calculate sigma points
  // set first column of sigma point matrix
  MatrixXd Xsig_aug = Eigen::MatrixXd::Zero(n_aug_, n_aug_*2+1);
  Xsig_aug.col(0) = x_aug;
  // set remaining sigma points
  for (int i = 0; i < n_aug_; ++i) {
    Xsig_aug.col(i+1)     = x_aug + sqrt(lambda_+n_aug_) * A.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * A.col(i);
  }

  // predict sigma points
  Xsig_pred_ = Xsig_aug.block(0, 0, 5, 2*n_aug_+1);
  Xsig_pred_.row(0) += (Xsig_aug.row(5).array() * Xsig_aug.row(3).array().cos()).matrix()*0.5*delta_t*delta_t;
  Xsig_pred_.row(1) += (Xsig_aug.row(5).array() * Xsig_aug.row(3).array().sin()).matrix()*0.5*delta_t*delta_t;
  Xsig_pred_.row(2) += Xsig_aug.row(5)*delta_t;
  Xsig_pred_.row(3) += Xsig_aug.row(6)*delta_t*delta_t/2;
  Xsig_pred_.row(4) += Xsig_aug.row(6)*delta_t;
  
  for (int i = 0; i <= 2*n_aug_; i++){
      double v = Xsig_aug(2, i);
      double yaw = Xsig_aug(3, i);
      double yawd = Xsig_aug(4, i);
      if (fabs(yawd) <= 1e-3) {
        Xsig_pred_(0,i) += v*cos(yaw)*delta_t;
        Xsig_pred_(1,i) += v*sin(yaw)*delta_t;
        //Xsig_pred_(3,i) += yawd*delta_t;
      }
      else {
        Xsig_pred_(0,i) += v/yawd*(sin(yaw+yawd*delta_t) - sin(yaw));
        Xsig_pred_(1,i) += v/yawd*(-cos(yaw+yawd*delta_t) + cos(yaw));
        Xsig_pred_(3,i) += yawd*delta_t;  
      } 
  }

  // predict state mean
  x_ = Xsig_pred_*weights_;
  // predict state covariance matrix
  MatrixXd centered = Xsig_pred_.colwise() - x_;
  P_ = (centered.array().rowwise() * weights_.transpose().array()).matrix()*centered.transpose();
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
}