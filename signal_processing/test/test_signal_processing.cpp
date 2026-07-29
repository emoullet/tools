#include <cmath>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include "signal_processing/dead_zone.hpp"
#include "signal_processing/low_pass_filter.hpp"
#include "signal_processing/one_euro_filter.hpp"
#include "signal_processing/saturation.hpp"

namespace
{
  Eigen::VectorXd vector3(double x, double y, double z)
  {
    Eigen::VectorXd value(3);
    value << x, y, z;
    return value;
  }
} // namespace

TEST(SaturationTest, RateLimitBehavesAsExpected)
{
  // Deltas larger than 0.75 are clipped per axis, while smaller deltas pass through.
  const Eigen::VectorXd current = vector3(2.0, -2.0, 0.5);
  const Eigen::VectorXd previous = Eigen::VectorXd::Zero(3);
  const Eigen::VectorXd limited = signal_processing::rateLimitPerAxis(current, previous, 0.75);

  EXPECT_NEAR(limited(0), 0.75, 1e-12);
  EXPECT_NEAR(limited(1), -0.75, 1e-12);
  EXPECT_NEAR(limited(2), 0.5, 1e-12);
}

TEST(SaturationTest, LimitNormScalesVector)
{
  // The 3-4-5 vector is scaled by 0.5 so its direction is preserved at norm 2.5.
  const Eigen::VectorXd input = vector3(3.0, 4.0, 0.0);
  const Eigen::VectorXd limited = signal_processing::limitNorm(input, 2.5);

  EXPECT_NEAR(limited.norm(), 2.5, 1e-12);
  EXPECT_NEAR(limited(0), 1.5, 1e-12);
  EXPECT_NEAR(limited(1), 2.0, 1e-12);
}

TEST(DeadZoneTest, NormDeadZoneMatchesRampShape)
{
  // The dead-zone boundary itself is still considered inactive input.
  const Eigen::VectorXd boundary =
      signal_processing::applyNormDeadZone(vector3(0.1, 0.0, 0.0), 0.1, 1.0);
  EXPECT_NEAR(boundary.norm(), 0.0, 1e-12);

  // Magnitudes inside the spherical dead-zone are fully zeroed.
  const Eigen::VectorXd zeroed =
      signal_processing::applyNormDeadZone(vector3(0.05, 0.0, 0.0), 0.1, 1.0);
  EXPECT_NEAR(zeroed.norm(), 0.0, 1e-12);

  // Magnitudes beyond the saturation radius keep direction but clamp to unit norm.
  const Eigen::VectorXd saturated =
      signal_processing::applyNormDeadZone(vector3(2.0, 0.0, 0.0), 0.1, 1.0);
  EXPECT_NEAR(saturated(0), 1.0, 1e-12);
  EXPECT_NEAR(saturated(1), 0.0, 1e-12);
  EXPECT_NEAR(saturated(2), 0.0, 1e-12);

  // Between dead-zone and saturation, the norm follows a linear ramp: (0.55 - 0.1) / 1.0.
  const Eigen::VectorXd ramped =
      signal_processing::applyNormDeadZone(vector3(0.55, 0.0, 0.0), 0.1, 1.1);
  EXPECT_NEAR(ramped(0), 0.45, 1e-12);

  const Eigen::VectorXd ramped_to_max =
      signal_processing::applyNormDeadZone(vector3(0.55, 0.0, 0.0), 0.1, 1.1, 2.0);
  EXPECT_NEAR(ramped_to_max(0), 0.9, 1e-12);
}

TEST(DeadZoneTest, ScaledDeadZonePerAxisMatchesJoystickDeadband)
{
  // Each component is treated independently, matching per-axis joystick deadband behavior.
  const Eigen::VectorXd input = vector3(0.05, -0.55, 2.0);
  const Eigen::VectorXd output = signal_processing::applyScaledDeadZonePerAxis(input, 0.1, 1.0);

  EXPECT_NEAR(output(0), 0.0, 1e-12);
  EXPECT_NEAR(output(1), -0.5, 1e-12);
  EXPECT_NEAR(output(2), 1.0, 1e-12);

  EXPECT_NEAR(signal_processing::applyScaledDeadZone(-0.55, 0.1, 1.1, 10.0), -4.5, 1e-12);
  EXPECT_NEAR(signal_processing::applyScaledDeadZone(0.55, 0.1, 1.1, 2.0), 0.9, 1e-12);

  const Eigen::VectorXd output_with_max =
      signal_processing::applyScaledDeadZonePerAxis(input, 0.1, 1.0, 2.0);
  EXPECT_NEAR(output_with_max(0), 0.0, 1e-12);
  EXPECT_NEAR(output_with_max(1), -1.0, 1e-12);
  EXPECT_NEAR(output_with_max(2), 2.0, 1e-12);
}

TEST(LowPassFilterTest, ComputesExpectedAlphaAndState)
{
  // Equal sample period and time constant give alpha = T / (T + tau) = 0.5.
  EXPECT_NEAR(signal_processing::computeLowPassAlpha(0.1, 0.1), 0.5, 1e-12);

  // The first sample warm-starts the state; the next sample blends halfway toward input.
  signal_processing::FirstOrderLowPassFilterVector filter;
  const Eigen::VectorXd first = filter.filter(vector3(1.0, 2.0, 3.0), 0.2);
  const Eigen::VectorXd second = filter.filter(vector3(3.0, 2.0, 1.0), 0.5);

  EXPECT_DOUBLE_EQ(first(0), 1.0);
  EXPECT_NEAR(second(0), 2.0, 1e-12);
  EXPECT_NEAR(second(1), 2.0, 1e-12);
  EXPECT_NEAR(second(2), 2.0, 1e-12);
}

TEST(OneEuroFilterTest, SmoothesAfterFirstSample)
{
  signal_processing::OneEuroFilter filter(30.0, 1.0, 0.0, 1.0);

  // The first sample initializes the filter; the next one is smoothed toward the new value.
  const double first = filter.filter(0.0, 0.0);
  const double second = filter.filter(1.0, 0.1);

  EXPECT_DOUBLE_EQ(first, 0.0);
  EXPECT_GT(second, 0.0);
  EXPECT_LT(second, 1.0);
}

TEST(OneEuroFilterTest, FiltersThreeDimensionsIndependently)
{
  signal_processing::OneEuroFilterVector filter(3, 30.0, 1.0, 0.0, 1.0);

  // The vector wrapper warm-starts all three scalar filters from the first sample.
  const Eigen::VectorXd first = filter.filter(Eigen::VectorXd::Zero(3), 0.0);
  const Eigen::VectorXd second = filter.filter(vector3(1.0, -1.0, 2.0), 0.1);

  // Each axis should move toward its own target without crossing or reaching it in one step.
  EXPECT_NEAR(first.norm(), 0.0, 1e-12);
  EXPECT_GT(second(0), 0.0);
  EXPECT_LT(second(0), 1.0);
  EXPECT_LT(second(1), 0.0);
  EXPECT_GT(second(1), -1.0);
  EXPECT_GT(second(2), 0.0);
  EXPECT_LT(second(2), 2.0);
}
