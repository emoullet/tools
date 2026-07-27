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

import time

import cv2

import numpy as np


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


class HandLandmarksViewer:
    """Render hand-landmark diagnostics and manage one OpenCV window."""

    def __init__(self, window_name):
        """Initialize the viewer without opening a window."""
        self.window_name = str(window_name)
        self._last_frame_time = time.time()

    def show_2d(
        self,
        image,
        hands,
        mediapipe_time_sec=None,
        reference_xyz=None,
        tracked_landmark_index=0,
        show_control_zones=True,
        dead_zone=0.0,
        saturation_zone=0.3,
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
            show_control_zones,
            dead_zone,
            saturation_zone,
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
        show_control_zones=True,
        dead_zone=0.0,
        saturation_zone=0.3,
        normalization_mode='axis',
        focal_length_px=None,
    ):
        """
        Render a metric 3D frame and return whether exit was requested.

        ``image_hands`` uses normalized image coordinates. Metric landmarks and
        the metric reference are expressed in meters.
        """
        annotated = image.copy()
        self._draw_hands(annotated, image_hands)
        performance_text = (
            f'MP: {float(mediapipe_time_sec) * 1000.0:.1f}ms  '
            f'missing depth: {int(missing_depth_count)}'
            if mediapipe_time_sec is not None
            else f'missing depth: {int(missing_depth_count)}'
        )
        self._draw_performance_overlay(annotated, status_text=performance_text)
        self._draw_3d_reference_overlay(
            annotated,
            primary_metric_hand,
            reference_metric,
            reference_image,
            reference_initialized,
            tracked_landmark_index,
            show_control_zones,
            dead_zone,
            saturation_zone,
            normalization_mode,
            focal_length_px,
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
        """Draw frame rate and a processing-status line."""
        now = time.time()
        elapsed = now - self._last_frame_time
        if elapsed > 0.0:
            cv2.putText(
                image,
                f'FPS: {1.0 / elapsed:.1f}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        self._last_frame_time = now

        if status_text is None and mediapipe_time_sec is not None:
            status_text = f'MP: {float(mediapipe_time_sec) * 1000.0:.1f}ms'
        if status_text is not None:
            cv2.putText(
                image,
                status_text,
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
        show_control_zones,
        dead_zone,
        saturation_zone,
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

        if show_control_zones:
            min_dimension = min(width, height)
            cv2.circle(
                image,
                ref_px,
                max(1, int(float(dead_zone) * min_dimension)),
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.circle(
                image,
                ref_px,
                max(1, int(float(saturation_zone) * min_dimension)),
                (255, 128, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
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
            distance = (dx * dx + dy * dy + dz * dz) ** 0.5
            saturated = (
                abs(dx) >= float(saturation_zone),
                abs(dy) >= float(saturation_zone),
                abs(dz) >= float(saturation_zone),
            )
            cv2.circle(image, landmark_px, 7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.line(image, ref_px, landmark_px, (255, 0, 255), 1, cv2.LINE_AA)
            status = 'DEAD' if distance < float(dead_zone) else 'ACTIVE'
            cv2.putText(
                image,
                f'LM[{tracked_landmark_index}] {status} '
                f'SAT[x:{int(saturated[0])} y:{int(saturated[1])} '
                f'z:{int(saturated[2])}]',
                (10, 178),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
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
        show_control_zones,
        dead_zone,
        saturation_zone,
        normalization_mode,
        focal_length_px,
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

        if show_control_zones and focal_length_px is not None:
            ref_depth = max(abs(float(reference_metric[2])), 1e-6)
            radius = max(
                1,
                int(
                    float(focal_length_px)
                    * float(saturation_zone)
                    / ref_depth
                ),
            )
            cv2.circle(image, ref_px, radius, (255, 128, 0), 2, cv2.LINE_AA)

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
        cv2.putText(
            image,
            f'SAT m: {saturation_zone:.2f} ({normalization_mode})',
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
            distance = (dx * dx + dy * dy + dz * dz) ** 0.5
            status = 'DEAD' if distance < float(dead_zone) else 'ACTIVE'
            cv2.putText(
                image,
                f'LM[{tracked_landmark_index}] {status} '
                f'd=({dx:.2f}, {dy:.2f}, {dz:.2f}) m',
                (10, 178),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 220, 255),
                2,
            )

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
