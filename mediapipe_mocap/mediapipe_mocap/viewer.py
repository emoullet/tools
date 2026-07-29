# Copyright 2026 Etienne Moullet
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
#
# * Neither the name of the Etienne Moullet nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""ROS-independent OpenCV visualization for hand landmarks."""

from dataclasses import dataclass
from enum import Enum
import time

import cv2

from mediapipe_mocap.mediapipe_runtime import PeriodicPerformanceTracker
import numpy as np
from signal_processing import (
    apply_norm_dead_zone,
    apply_scaled_dead_zone_per_axis,
)


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

_CUBOID_EDGES = (
    (0, 1), (0, 2), (0, 4),
    (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6),
    (5, 7),
    (6, 7),
)


class OverlayNormalizationMode(Enum):
    """Control normalization policies supported by the debug overlay."""

    AXIS = 'axis'
    VECTOR = 'vector'


def parse_overlay_normalization_mode(value, warn=None):
    """Parse an overlay mode, falling back to vector normalization."""
    normalized = str(value).lower()
    try:
        return OverlayNormalizationMode(normalized)
    except ValueError:
        if warn is not None:
            warn(
                'Invalid overlay_normalization_mode '
                f"'{normalized}', falling back to 'vector'."
            )
        return OverlayNormalizationMode.VECTOR


@dataclass(frozen=True)
class ControlOverlayConfig:
    """Display-only control-zone and normalization configuration."""

    dead_zone: float
    saturation_zone: float
    normalization_mode: OverlayNormalizationMode

    @property
    def effective_saturation_zone(self):
        """Return the saturation boundary used by shared signal processing."""
        return max(abs(float(self.saturation_zone)), abs(float(self.dead_zone)))

    def normalize(self, displacement):
        """Normalize a filtered displacement for display only."""
        if self.normalization_mode is OverlayNormalizationMode.AXIS:
            return apply_scaled_dead_zone_per_axis(
                displacement,
                self.dead_zone,
                self.saturation_zone,
            )
        return apply_norm_dead_zone(
            displacement,
            self.dead_zone,
            self.saturation_zone,
        )


class HandLandmarksViewer:
    """Render hand-landmark diagnostics and manage one OpenCV window."""

    def __init__(
        self,
        window_name,
        *,
        performance_interval_sec=0.5,
        clock=time.monotonic,
    ):
        """Initialize the viewer without opening a window."""
        self.window_name = str(window_name)
        self._performance = PeriodicPerformanceTracker(
            interval_sec=performance_interval_sec,
            clock=clock,
        )
        self._performance_snapshot = None

    def show_2d(
        self,
        image,
        hands,
        mediapipe_time_sec=None,
        reference_xyz=None,
        tracked_landmark_index=0,
        control_overlay=None,
        displacement_scale=(1.0, 1.0, 1.0),
    ):
        """
        Render a 2D hand-landmark frame and return whether exit was requested.

        ``hands`` and ``reference_xyz`` use normalized image coordinates.
        """
        annotated = image.copy()
        self._draw_hands(annotated, hands)
        self._draw_performance_overlay(annotated, mediapipe_time_sec)
        self._draw_2d_reference_overlay(
            annotated,
            reference_xyz,
            hands[0] if hands else None,
            tracked_landmark_index,
            control_overlay,
            displacement_scale,
        )
        return self._show(annotated)

    def show_3d(
        self,
        image,
        image_hands,
        primary_metric_hand=None,
        missing_depth_count=0,
        mediapipe_time_sec=None,
        reference_metric=None,
        reference_image=None,
        reference_initialized=False,
        tracked_landmark_index=0,
        control_overlay=None,
        camera_intrinsics=None,
    ):
        """
        Render a metric 3D frame and return whether exit was requested.

        ``image_hands`` uses normalized image coordinates. Metric landmarks and
        the metric reference are expressed in meters.
        """
        annotated = image.copy()
        self._draw_hands(annotated, image_hands)
        self._draw_performance_overlay(
            annotated,
            mediapipe_time_sec,
            status_text=f'missing depth: {int(missing_depth_count)}',
        )
        self._draw_3d_reference_overlay(
            annotated,
            primary_metric_hand,
            reference_metric,
            reference_image,
            reference_initialized,
            tracked_landmark_index,
            control_overlay,
            camera_intrinsics,
        )
        return self._show(annotated)

    def close(self):
        """Close this viewer's OpenCV window."""
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass

    @staticmethod
    def _draw_hands(image, hands):
        """Draw all supplied hands in normalized image coordinates."""
        height, width = image.shape[:2]
        for hand_landmarks in hands:
            points_px = []
            for landmark in hand_landmarks:
                point = (
                    int(float(landmark.x) * width),
                    int(float(landmark.y) * height),
                )
                points_px.append(point)
                cv2.circle(image, point, 3, (0, 255, 0), -1)

            for start_idx, end_idx in HAND_CONNECTIONS:
                if start_idx < len(points_px) and end_idx < len(points_px):
                    cv2.line(
                        image,
                        points_px[start_idx],
                        points_px[end_idx],
                        (0, 200, 255),
                        2,
                    )

    def _draw_performance_overlay(
        self,
        image,
        mediapipe_time_sec=None,
        status_text=None,
    ):
        """Draw cached window metrics and a current-frame status line."""
        completed = self._performance.tick(mediapipe_time_sec)
        if completed is not None:
            self._performance_snapshot = completed

        if self._performance_snapshot is not None:
            cv2.putText(
                image,
                f'FPS: {self._performance_snapshot.rate_hz:.1f}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        status_parts = []
        if (
            self._performance_snapshot is not None
            and self._performance_snapshot.average_duration_sec is not None
        ):
            status_parts.append(
                'MP avg: '
                f'{self._performance_snapshot.average_duration_sec * 1000.0:.1f}ms'
            )
        if status_text is not None:
            status_parts.append(status_text)

        if status_parts:
            cv2.putText(
                image,
                '  '.join(status_parts),
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

    @staticmethod
    def _draw_2d_reference_overlay(
        image,
        reference_xyz,
        primary_hand,
        tracked_landmark_index,
        control_overlay,
        displacement_scale,
    ):
        """Draw normalized reference and control-zone feedback."""
        if reference_xyz is None:
            return

        ref_x, ref_y, ref_z = reference_xyz
        height, width = image.shape[:2]
        ref_px = (
            int(np.clip(ref_x, 0.0, 1.0) * width),
            int(np.clip(ref_y, 0.0, 1.0) * height),
        )
        HandLandmarksViewer._draw_reference_marker(image, ref_px)
        cv2.putText(
            image,
            f'Ref: ({ref_x:.2f}, {ref_y:.2f}, {ref_z:.2f})',
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 255),
            2,
        )

        if control_overlay is None:
            return

        min_dimension = min(width, height)
        dead_zone = abs(float(control_overlay.dead_zone))
        saturation_zone = control_overlay.effective_saturation_zone
        if (
            control_overlay.normalization_mode
            is OverlayNormalizationMode.AXIS
        ):
            HandLandmarksViewer._draw_2d_axis_zone(
                image,
                ref_px,
                dead_zone,
                min_dimension,
                (0, 255, 255),
            )
            HandLandmarksViewer._draw_2d_axis_zone(
                image,
                ref_px,
                saturation_zone,
                min_dimension,
                (255, 128, 0),
            )
        else:
            cv2.circle(
                image,
                ref_px,
                max(1, int(dead_zone * min_dimension)),
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.circle(
                image,
                ref_px,
                max(1, int(saturation_zone * min_dimension)),
                (255, 128, 0),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            image,
            f'CTRL {control_overlay.normalization_mode.value}  '
            f'DZ: {dead_zone:.2f}  SAT: {saturation_zone:.2f}',
            (10, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        if primary_hand and 0 <= tracked_landmark_index < len(primary_hand):
            landmark = primary_hand[tracked_landmark_index]
            landmark_px = (
                int(np.clip(float(landmark.x), 0.0, 1.0) * width),
                int(np.clip(float(landmark.y), 0.0, 1.0) * height),
            )
            dx = float(landmark.x) - ref_x
            dy = float(landmark.y) - ref_y
            dz = float(landmark.z) - ref_z
            displacement = tuple(
                value * float(scale)
                for value, scale in zip(
                    (dx, dy, dz),
                    displacement_scale,
                )
            )
            normalized = control_overlay.normalize(displacement)
            cv2.circle(image, landmark_px, 7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.line(image, ref_px, landmark_px, (255, 0, 255), 1, cv2.LINE_AA)
            status = HandLandmarksViewer._control_status(
                displacement,
                normalized,
                control_overlay,
            )
            cv2.putText(
                image,
                f'LM[{tracked_landmark_index}] {status} '
                f'd=({displacement[0]:.2f}, {displacement[1]:.2f}, '
                f'{displacement[2]:.2f}) '
                f'n=({normalized[0]:.2f}, {normalized[1]:.2f}, '
                f'{normalized[2]:.2f})',
                (10, 178),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 220, 255),
                2,
            )

    @staticmethod
    def _draw_3d_reference_overlay(
        image,
        primary_metric_hand,
        reference_metric,
        reference_image,
        reference_initialized,
        tracked_landmark_index,
        control_overlay,
        camera_intrinsics,
    ):
        """Draw metric reference and control-zone feedback."""
        if not reference_initialized or reference_metric is None:
            return

        height, width = image.shape[:2]
        if reference_image is None:
            ref_px = (int(width * 0.5), int(height * 0.5))
        else:
            ref_px = (
                int(np.clip(reference_image[0], 0.0, 1.0) * width),
                int(np.clip(reference_image[1], 0.0, 1.0) * height),
            )
        HandLandmarksViewer._draw_reference_marker(image, ref_px)

        cv2.putText(
            image,
            f'Ref3D: ({reference_metric[0]:.2f}, {reference_metric[1]:.2f}, '
            f'{reference_metric[2]:.2f}) m',
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 255),
            2,
        )
        if control_overlay is None:
            return

        dead_zone = abs(float(control_overlay.dead_zone))
        saturation_zone = control_overlay.effective_saturation_zone
        if camera_intrinsics is not None:
            if (
                control_overlay.normalization_mode
                is OverlayNormalizationMode.AXIS
            ):
                HandLandmarksViewer._draw_projected_cuboid(
                    image,
                    reference_metric,
                    dead_zone,
                    camera_intrinsics,
                    (0, 255, 255),
                )
                HandLandmarksViewer._draw_projected_cuboid(
                    image,
                    reference_metric,
                    saturation_zone,
                    camera_intrinsics,
                    (255, 128, 0),
                )
            else:
                HandLandmarksViewer._draw_projected_sphere(
                    image,
                    ref_px,
                    reference_metric,
                    dead_zone,
                    camera_intrinsics,
                    (0, 255, 255),
                )
                HandLandmarksViewer._draw_projected_sphere(
                    image,
                    ref_px,
                    reference_metric,
                    saturation_zone,
                    camera_intrinsics,
                    (255, 128, 0),
                )
        cv2.putText(
            image,
            f'CTRL {control_overlay.normalization_mode.value}  '
            f'DZ: {dead_zone:.2f}m  SAT: {saturation_zone:.2f}m',
            (10, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        tracked_index_valid = (
            primary_metric_hand
            and 0 <= tracked_landmark_index < len(primary_metric_hand)
        )
        if tracked_index_valid:
            landmark = primary_metric_hand[tracked_landmark_index]
            dx = float(landmark.x) - float(reference_metric[0])
            dy = float(landmark.y) - float(reference_metric[1])
            dz = float(landmark.z) - float(reference_metric[2])
            displacement = (dx, dy, dz)
            normalized = control_overlay.normalize(displacement)
            status = HandLandmarksViewer._control_status(
                displacement,
                normalized,
                control_overlay,
            )
            cv2.putText(
                image,
                f'LM[{tracked_landmark_index}] {status} '
                f'd=({dx:.2f}, {dy:.2f}, {dz:.2f})m '
                f'n=({normalized[0]:.2f}, {normalized[1]:.2f}, '
                f'{normalized[2]:.2f})',
                (10, 178),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 220, 255),
                2,
            )

    @staticmethod
    def _draw_2d_axis_zone(image, center, zone, min_dimension, color):
        """Draw one component-wise 2D control boundary."""
        radius = max(1, int(float(zone) * min_dimension))
        cv2.rectangle(
            image,
            (center[0] - radius, center[1] - radius),
            (center[0] + radius, center[1] + radius),
            color,
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_projected_sphere(
        image,
        center,
        reference_metric,
        radius_m,
        camera_intrinsics,
        color,
    ):
        """Draw an image-plane approximation of one metric sphere."""
        fx, fy, _, _ = camera_intrinsics
        depth_m = float(reference_metric[2])
        if depth_m <= 1e-9:
            return
        axes = (
            max(1, int(abs(float(fx) * float(radius_m) / depth_m))),
            max(1, int(abs(float(fy) * float(radius_m) / depth_m))),
        )
        cv2.ellipse(
            image,
            center,
            axes,
            0.0,
            0.0,
            360.0,
            color,
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_projected_cuboid(
        image,
        reference_metric,
        half_extent_m,
        camera_intrinsics,
        color,
    ):
        """Project and draw one axis-aligned metric cuboid."""
        ref_x, ref_y, ref_z = (float(value) for value in reference_metric)
        extent = abs(float(half_extent_m))
        corners = [
            (ref_x + sx * extent, ref_y + sy * extent, ref_z + sz * extent)
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ]
        projected = [
            HandLandmarksViewer._metric_point_to_pixel(point, camera_intrinsics)
            for point in corners
        ]
        if any(point is None for point in projected):
            return
        for start, end in _CUBOID_EDGES:
            cv2.line(
                image,
                projected[start],
                projected[end],
                color,
                2,
                cv2.LINE_AA,
            )

    @staticmethod
    def _metric_point_to_pixel(point, camera_intrinsics):
        """Project one metric camera point into image pixels."""
        fx, fy, cx, cy = (float(value) for value in camera_intrinsics)
        x, y, z = (float(value) for value in point)
        if z <= 1e-9:
            return None
        return (
            int(round(fx * x / z + cx)),
            int(round(fy * y / z + cy)),
        )

    @staticmethod
    def _control_status(displacement, normalized, control_overlay):
        """Classify a display-only normalized control preview."""
        if all(abs(float(value)) <= 1e-12 for value in normalized):
            return 'DEAD'
        saturation = control_overlay.effective_saturation_zone
        if (
            control_overlay.normalization_mode
            is OverlayNormalizationMode.AXIS
        ):
            saturated = any(
                abs(float(value)) >= saturation for value in displacement
            )
        else:
            magnitude = sum(float(value) ** 2 for value in displacement) ** 0.5
            saturated = magnitude >= saturation
        return 'SATURATED' if saturated else 'ACTIVE'

    @staticmethod
    def _draw_reference_marker(image, point):
        """Draw the common reference marker."""
        cv2.drawMarker(
            image,
            point,
            (255, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=16,
            thickness=2,
            line_type=cv2.LINE_AA,
        )

    def _show(self, image):
        """Display one annotated frame and read the exit key."""
        cv2.imshow(self.window_name, image)
        key = cv2.waitKey(1)
        return key == 27 or key == ord('q')
