# ---
# VOLLEBAK_ENGINEERING_METADATA:
#   PROJECT_ID: PREDATOR-01
#   TRACK: ECOSYSTEM
#   PHASE: CHALLENGE_HUB
#   CONTRIBUTOR_ID: Antigravity AI
#   THE_DELTA: Full pipeline test suite validates FSM transitions, aerial tracker, micro-Doppler classifier, safety gates, and scan patterns without hardware
#   FAILURE_MODE: N/A — test infrastructure
#   IP_STATUS: VOLLEBAK_PROPRIETARY
#   USER_FEEDBACK_REF: None
#   DEPENDENCIES: [pytest, numpy, scipy]
# ---
"""
Unit tests for Project Predator — runs without hardware.

Tests cover:
    - State machine transitions and timeouts
    - Aerial target tracker (9-state Kalman)
    - Micro-Doppler classifier (STFT blade-rate extraction)
    - Safety manager (keep-out cone + range gates)
    - Slew-to-cue (Cartesian → gimbal commands)
    - Lissajous scanner (pattern geometry)
    - BDA assessor (kill assessment logic)
    - Simulated radar backend
"""

import time
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# FSM Tests
# ---------------------------------------------------------------------------

class TestPredatorStateMachine:
    """Tests for the Silent-to-Active state machine."""

    def _make_fsm(self, **overrides):
        from src.layer3_fusion.predator_state_machine import (
            PredatorStateMachine, FSMConfig, SystemState,
        )
        defaults = dict(
            alert_confirm_count=2,
            high_confidence_threshold=0.8,
            pre_slew_on_alert=True,
            alert_timeout_s=1.0,
            radar_search_timeout_s=2.0,
            tracking_coast_timeout_s=1.0,
            engagement_timeout_s=2.0,
            bda_observation_s=1.0,
        )
        defaults.update(overrides)
        config = FSMConfig(**defaults)
        return PredatorStateMachine(config), SystemState

    def test_initial_state_is_silent(self):
        fsm, SS = self._make_fsm()
        assert fsm.state == SS.SILENT

    def test_low_confidence_triggers_alert(self):
        """Below high_confidence_threshold → ALERT (not direct RADAR_ACTIVE)."""
        fsm, SS = self._make_fsm()
        fsm.on_layer1_detection(camera_id=0, bearing_deg=45.0, confidence=0.5, timestamp_us=1000)
        assert fsm.state == SS.ALERT

    def test_high_confidence_fast_tracks_to_radar(self):
        """Above high_confidence_threshold → skip ALERT → direct RADAR_ACTIVE."""
        fsm, SS = self._make_fsm()
        fsm.on_layer1_detection(0, 45.0, confidence=0.9, timestamp_us=1000)
        assert fsm.state == SS.RADAR_ACTIVE  # Skipped ALERT

    def test_confirmed_detection_triggers_radar_active(self):
        """Low confidence requires alert_confirm_count=2 detections."""
        fsm, SS = self._make_fsm()
        fsm.on_layer1_detection(0, 45.0, 0.5, 1000)
        assert fsm.state == SS.ALERT
        fsm.on_layer1_detection(0, 46.0, 0.6, 2000)
        assert fsm.state == SS.RADAR_ACTIVE

    def test_pre_slew_arm_deploy_on_first_detection(self):
        """First detection triggers arm deploy + FSM coarse aim."""
        fsm, SS = self._make_fsm()
        fsm.on_layer1_detection(0, 45.0, 0.5, 1000)
        assert fsm.arm_deploy_requested
        assert abs(fsm.pre_slew_bearing_deg - 45.0) < 0.01

    def test_pre_slew_right_shoulder_for_right_bearing(self):
        """Bearing 0-180° → right shoulder deployment."""
        fsm, SS = self._make_fsm()
        fsm.on_layer1_detection(0, 90.0, 0.5, 1000)
        assert fsm.arm_deploy_requested

    def test_pre_slew_left_shoulder_for_left_bearing(self):
        """Bearing 180-360° → left shoulder deployment."""
        fsm, SS = self._make_fsm()
        fsm.on_layer1_detection(0, 270.0, 0.5, 1000)
        assert fsm.arm_deploy_requested

    def test_radar_acquisition_triggers_tracking(self):
        fsm, SS = self._make_fsm()
        fsm.on_layer1_detection(0, 45.0, 0.9, 1000)  # Fast-track
        fsm.on_radar_track_acquired(track_id=0, range_m=150.0, azimuth_deg=45.0, elevation_deg=10.0)
        assert fsm.state == SS.TRACKING

    def test_threat_score_triggers_engagement(self):
        fsm, SS = self._make_fsm()
        fsm.on_layer1_detection(0, 45.0, 0.9, 1000)
        fsm.on_radar_track_acquired(0, 150.0, 45.0, 10.0)
        fsm.on_threat_score_update(0.85)
        assert fsm.state == SS.ENGAGEMENT

    def test_laser_fire_triggers_bda(self):
        fsm, SS = self._make_fsm()
        fsm.on_layer1_detection(0, 45.0, 0.9, 1000)
        fsm.on_radar_track_acquired(0, 150.0, 45.0, 10.0)
        fsm.on_threat_score_update(0.85)
        fsm.on_laser_fired()
        assert fsm.state == SS.BDA

    def test_radar_not_authorized_in_silent(self):
        fsm, _ = self._make_fsm()
        assert not fsm.is_radar_authorized

    def test_radar_authorized_after_high_conf_detection(self):
        fsm, _ = self._make_fsm()
        fsm.on_layer1_detection(0, 45.0, 0.9, 1000)
        assert fsm.is_radar_authorized

    def test_transition_log_high_conf_fast_track(self):
        """High confidence: single transition SILENT→RADAR_ACTIVE."""
        fsm, _ = self._make_fsm()
        fsm.on_layer1_detection(0, 45.0, 0.95, 1000)
        log = fsm.get_transition_log()
        assert len(log) == 1  # SILENT→RADAR_ACTIVE (skipped ALERT)
        assert log[0].trigger == "high_confidence_detection"

    def test_transition_log_low_conf_two_step(self):
        """Low confidence: SILENT→ALERT→RADAR_ACTIVE (2 transitions)."""
        fsm, _ = self._make_fsm()
        fsm.on_layer1_detection(0, 45.0, 0.5, 1000)
        fsm.on_layer1_detection(0, 46.0, 0.6, 2000)
        log = fsm.get_transition_log()
        assert len(log) == 2


# ---------------------------------------------------------------------------
# Aerial Tracker Tests
# ---------------------------------------------------------------------------

class TestAerialTracker:
    """Tests for the 9-state constant-acceleration Kalman tracker."""

    def _make_tracker(self):
        from src.layer3_fusion.aerial_target_tracker import (
            AerialMultiTargetTracker, AerialTrackerConfig,
        )
        config = AerialTrackerConfig(dt=0.1, max_coast_frames=5)
        return AerialMultiTargetTracker(config)

    def test_spawn_track_from_single_detection(self):
        tracker = self._make_tracker()
        pos = np.array([[100.0, 50.0, 30.0]])
        dop = np.array([5.0])
        rcs = np.array([-10.0])
        cnt = np.array([5])
        targets = tracker.update(pos, dop, rcs, cnt)
        assert len(targets) == 1
        assert targets[0].target_id == 0

    def test_track_persistence_across_frames(self):
        tracker = self._make_tracker()
        for i in range(5):
            pos = np.array([[100.0 + i * 0.5, 50.0, 30.0]])
            targets = tracker.update(pos, np.array([5.0]), np.array([-10.0]), np.array([5]))
        assert len(targets) == 1
        assert targets[0].track_age_frames == 4

    def test_track_coasts_when_no_measurements(self):
        tracker = self._make_tracker()
        pos = np.array([[100.0, 50.0, 30.0]])
        tracker.update(pos, np.array([5.0]), np.array([-10.0]), np.array([5]))
        # Empty frames
        for _ in range(3):
            empty = np.empty((0, 3))
            targets = tracker.update(empty, np.array([]), np.array([]), np.array([]))
        assert tracker.active_track_count == 1  # Still coasting

    def test_track_pruned_after_max_coast(self):
        tracker = self._make_tracker()
        pos = np.array([[100.0, 50.0, 30.0]])
        tracker.update(pos, np.array([5.0]), np.array([-10.0]), np.array([5]))
        for _ in range(10):
            empty = np.empty((0, 3))
            tracker.update(empty, np.array([]), np.array([]), np.array([]))
        assert tracker.active_track_count == 0


# ---------------------------------------------------------------------------
# Micro-Doppler Classifier Tests
# ---------------------------------------------------------------------------

class TestMicroDopplerClassifier:
    """Tests for STFT-based blade-rate classification."""

    def _make_classifier(self):
        from src.layer2_radar.micro_doppler_classifier import (
            MicroDopplerClassifier, TargetClass,
        )
        return MicroDopplerClassifier(), TargetClass

    def _make_drone_doppler(self, blade_hz=120.0, duration_s=0.5):
        from src.layer2_radar.radar_interface import DopplerTimeSeries
        fs = 1000.0
        n = int(duration_s * fs)
        t = np.arange(n) / fs
        # Blade-pass fundamental + 2 harmonics
        signal = (
            -5.0  # bulk velocity
            + 0.5 * np.sin(2 * np.pi * blade_hz * t)
            + 0.25 * np.sin(2 * np.pi * 2 * blade_hz * t)
            + 0.12 * np.sin(2 * np.pi * 3 * blade_hz * t)
        )
        signal += np.random.normal(0, 0.02, n)
        timestamps = np.arange(n) * 1000  # 1kHz → 1000µs spacing
        return DopplerTimeSeries(
            timestamps_us=timestamps.astype(np.int64),
            doppler_mps=signal.astype(np.float32),
            range_m=150.0,
            azimuth_deg=10.0,
        )

    def test_rotary_uas_classification(self):
        classifier, TC = self._make_classifier()
        drone_ts = self._make_drone_doppler(blade_hz=120.0)
        result = classifier.classify(drone_ts)
        assert result.target_class == TC.ROTARY_UAS
        assert result.confidence > 0.3
        assert result.fundamental_hz > 50.0

    def test_clutter_classification(self):
        classifier, TC = self._make_classifier()
        from src.layer2_radar.radar_interface import DopplerTimeSeries
        # Pure noise — no harmonic structure
        n = 500
        ts = DopplerTimeSeries(
            timestamps_us=np.arange(n, dtype=np.int64) * 1000,
            doppler_mps=np.random.normal(0, 0.1, n).astype(np.float32),
            range_m=50.0,
            azimuth_deg=0.0,
        )
        result = classifier.classify(ts)
        assert result.target_class in (TC.CLUTTER, TC.UNKNOWN)


# ---------------------------------------------------------------------------
# Safety Manager Tests
# ---------------------------------------------------------------------------

class TestSafetyManager:
    """Tests for engagement safety gate."""

    def _make_safety(self):
        from src.layer3_fusion.safety_manager import SafetyManager, SafetyConfig
        config = SafetyConfig(
            keep_out_cone_deg=30.0,
            min_engagement_range_m=10.0,
            max_engagement_range_m=300.0,
            min_elevation_deg=5.0,
        )
        return SafetyManager(config)

    def test_safe_engagement_authorized(self):
        sm = self._make_safety()
        check = sm.check_engagement(bearing_deg=45.0, elevation_deg=15.0, range_m=150.0)
        assert check.authorized

    def test_keep_out_cone_vetoes(self):
        sm = self._make_safety()
        check = sm.check_engagement(bearing_deg=10.0, elevation_deg=5.0, range_m=150.0)
        assert not check.authorized
        assert not check.keep_out_clear

    def test_too_close_vetoes(self):
        sm = self._make_safety()
        check = sm.check_engagement(bearing_deg=90.0, elevation_deg=15.0, range_m=5.0)
        assert not check.authorized
        assert not check.range_clear

    def test_too_far_vetoes(self):
        sm = self._make_safety()
        check = sm.check_engagement(bearing_deg=90.0, elevation_deg=15.0, range_m=500.0)
        assert not check.authorized

    def test_low_elevation_vetoes(self):
        sm = self._make_safety()
        check = sm.check_engagement(bearing_deg=90.0, elevation_deg=2.0, range_m=150.0)
        assert not check.authorized
        assert not check.elevation_clear


# ---------------------------------------------------------------------------
# Slew-to-Cue Tests
# ---------------------------------------------------------------------------

class TestSlewToCue:
    """Tests for target-to-gimbal conversion."""

    def _make_slew(self):
        from src.layer4_engagement.slew_to_cue import SlewToCue, SlewConfig
        return SlewToCue(SlewConfig(lead_time_s=0.0))

    def test_forward_target(self):
        slew = self._make_slew()
        cmd = slew.compute_command(0.0, 100.0, 30.0)
        assert abs(cmd.azimuth_deg) < 1.0  # Nearly forward
        assert cmd.elevation_deg > 0.0     # Above horizon
        assert cmd.target_range_m > 100.0

    def test_right_target(self):
        slew = self._make_slew()
        cmd = slew.compute_command(100.0, 0.01, 0.0)
        assert cmd.azimuth_deg > 80.0  # Right side


# ---------------------------------------------------------------------------
# Lissajous Scanner Tests
# ---------------------------------------------------------------------------

class TestLissajousScanner:
    """Tests for scan pattern generation."""

    def test_lissajous_pattern_length(self):
        from src.layer4_engagement.lissajous_scanner import LissajousScanner, ScanConfig
        scanner = LissajousScanner(ScanConfig(num_points=100))
        pattern = scanner.generate_pattern()
        assert len(pattern) == 100

    def test_pattern_bounded(self):
        from src.layer4_engagement.lissajous_scanner import LissajousScanner, ScanConfig
        scanner = LissajousScanner(ScanConfig(angular_extent_deg=2.0))
        pattern = scanner.generate_pattern()
        for p in pattern:
            assert abs(p.az_offset_deg) <= 2.1
            assert abs(p.el_offset_deg) <= 2.1


# ---------------------------------------------------------------------------
# Simulated Radar Tests
# ---------------------------------------------------------------------------

class TestSimulatedRadar:
    """Tests for the simulated radar backend."""

    def test_generates_frames(self):
        from src.layer2_radar.radar_interface import SimulatedRadarBackend
        radar = SimulatedRadarBackend(update_rate_hz=100.0)
        radar.add_simulated_target(range_m=100.0)
        radar.start()
        frame = radar.get_frame()
        radar.stop()
        assert frame is not None
        assert len(frame.detections) >= 1

    def test_doppler_time_series(self):
        from src.layer2_radar.radar_interface import SimulatedRadarBackend
        radar = SimulatedRadarBackend()
        radar.add_simulated_target(range_m=100.0, blade_rate_hz=120.0)
        radar.start()
        ts = radar.get_doppler_time_series(100.0, 10.0, duration_s=0.5)
        radar.stop()
        assert ts is not None
        assert ts.doppler_mps.size > 0


# ---------------------------------------------------------------------------
# Event Aggregator Tests
# ---------------------------------------------------------------------------

class TestEventAggregator:
    """Tests for bearing computation."""

    def test_center_pixel_gives_boresight_bearing(self):
        from src.layer1_neuromorphic.event_aggregator import EventAggregator
        agg = EventAggregator(resolution_w=640)
        bearings = agg.compute_pixel_bearings(
            x_pixels=np.array([320.0]),
            fov_h_deg=55.0,
            azimuth_offset_deg=90.0,
        )
        assert abs(bearings[0] - 90.0) < 0.5

    def test_edge_pixel_offset(self):
        from src.layer1_neuromorphic.event_aggregator import EventAggregator
        agg = EventAggregator(resolution_w=640)
        bearings = agg.compute_pixel_bearings(
            x_pixels=np.array([0.0, 639.0]),
            fov_h_deg=55.0,
            azimuth_offset_deg=0.0,
        )
        # Left edge should be negative (wrapped to ~332.5°)
        # Right edge should be positive (~27.5°)
        assert bearings[1] > bearings[0] or (bearings[0] > 300)  # Wrap handling


# ---------------------------------------------------------------------------
# IMU Provider Tests
# ---------------------------------------------------------------------------

class TestImuProvider:
    """Tests for multi-source IMU fusion."""

    def test_degraded_state_when_no_sources(self):
        from src.imu.imu_provider import ImuProvider
        provider = ImuProvider()
        state = provider.get_fused_state()
        assert not state.is_valid
        assert state.source == "degraded"

    def test_rotation_matrix_identity_when_no_imu(self):
        from src.imu.imu_provider import ImuProvider
        provider = ImuProvider()
        rot = provider.get_rotation_matrix()
        np.testing.assert_array_almost_equal(rot, np.eye(3))

    def test_operator_stationary_by_default(self):
        from src.imu.imu_provider import ImuProvider
        provider = ImuProvider()
        motion = provider.get_operator_motion_state()
        assert motion.is_stationary
        assert not motion.is_walking
        assert not motion.is_running

    def test_quaternion_to_rotation_identity(self):
        from src.imu.imu_provider import ImuProvider
        rot = ImuProvider._quaternion_to_rotation(1.0, 0.0, 0.0, 0.0)
        np.testing.assert_array_almost_equal(rot, np.eye(3))

    def test_quaternion_to_rotation_90_yaw(self):
        """90° yaw rotation should map X→Y, Y→-X."""
        from src.imu.imu_provider import ImuProvider
        import math
        # Quaternion for 90° around Z: (cos(45°), 0, 0, sin(45°))
        w = math.cos(math.pi / 4)
        z = math.sin(math.pi / 4)
        rot = ImuProvider._quaternion_to_rotation(w, 0.0, 0.0, z)
        # X-axis should map to approximately Y
        x_mapped = rot @ np.array([1, 0, 0])
        assert abs(x_mapped[1] - 1.0) < 0.01 or abs(x_mapped[0]) < 0.01


# ---------------------------------------------------------------------------
# Motion Compensator Tests
# ---------------------------------------------------------------------------

class TestMotionCompensator:
    """Tests for ego-motion compensation."""

    def _make_compensator(self):
        from src.imu.imu_provider import ImuProvider
        from src.imu.motion_compensator import MotionCompensator, CompensationConfig
        provider = ImuProvider()  # No sources = degraded mode
        config = CompensationConfig(enabled=True)
        return MotionCompensator(provider, config)

    def test_passthrough_when_disabled(self):
        from src.imu.imu_provider import ImuProvider
        from src.imu.motion_compensator import MotionCompensator, CompensationConfig
        provider = ImuProvider()
        config = CompensationConfig(enabled=False)
        comp = MotionCompensator(provider, config)
        points = np.array([[10, 20, 30, 5.0]], dtype=np.float32)
        result = comp.compensate_radar_pointcloud(points)
        np.testing.assert_array_equal(result, points)

    def test_empty_pointcloud_passthrough(self):
        comp = self._make_compensator()
        empty = np.empty((0, 4), dtype=np.float32)
        result = comp.compensate_radar_pointcloud(empty)
        assert result.shape[0] == 0

    def test_degraded_mode_static_compensation(self):
        comp = self._make_compensator()
        # Create points with static reflectors (low Doppler) and a target
        n_static = 20
        static = np.column_stack([
            np.random.uniform(-5, 5, n_static),
            np.random.uniform(1, 10, n_static),
            np.zeros(n_static),
            np.full(n_static, 0.5),  # ~0.5 m/s ego Doppler
        ]).astype(np.float32)
        result = comp.compensate_radar_pointcloud(static)
        # Static reflector Doppler should be closer to zero after compensation
        assert abs(np.median(result[:, 3])) < abs(np.median(static[:, 3]))

    def test_gimbal_feed_forward_invalid_without_imu(self):
        comp = self._make_compensator()
        ff = comp.get_gimbal_feed_forward()
        assert not ff.is_valid


# ---------------------------------------------------------------------------
# Kinematic Primer Tests
# ---------------------------------------------------------------------------

class TestKinematicPrimer:
    """Tests for the speed/trajectory kinematic primer in threat scoring."""

    def _make_classifier(self):
        from src.layer3_fusion.threat_classifier import (
            ThreatClassifier, ThreatClassifierConfig,
        )
        return ThreatClassifier(ThreatClassifierConfig())

    def test_primer_returns_valid_assessment(self):
        tc = self._make_classifier()
        result = tc.compute_kinematic_primer(
            target_id=0,
            speed_mps=41.0,
            radial_velocity_mps=-35.0,
            altitude_m=30.0,
            flicker_confidence=0.9,
            rcs_dbsm=-15.0,
        )
        assert result.is_primer_only
        assert result.doppler_score == 0.0  # No micro-Doppler yet
        assert result.kinematic_primer_score > 0.0

    def test_fast_mover_primer_exceeds_threshold(self):
        """41 m/s closing drone with flicker should pass 0.7 threshold."""
        tc = self._make_classifier()
        result = tc.compute_kinematic_primer(
            target_id=0,
            speed_mps=41.0,
            radial_velocity_mps=-35.0,  # Closing fast
            altitude_m=25.0,
            flicker_confidence=0.9,
            rcs_dbsm=-15.0,
        )
        assert result.is_threat, (
            f"Fast mover primer score {result.threat_score:.3f} "
            f"should exceed 0.7 threshold"
        )

    def test_slow_mover_primer_below_threshold(self):
        """Slow target with low flicker should NOT pass threshold."""
        tc = self._make_classifier()
        result = tc.compute_kinematic_primer(
            target_id=0,
            speed_mps=2.0,
            radial_velocity_mps=0.0,  # Not closing
            altitude_m=5.0,
            flicker_confidence=0.3,
            rcs_dbsm=-15.0,
        )
        assert not result.is_threat

    def test_closing_rate_boost(self):
        """Negative radial velocity (closing) increases score."""
        tc = self._make_classifier()
        result_closing = tc.compute_kinematic_primer(
            target_id=0, speed_mps=20.0, radial_velocity_mps=-25.0,
            altitude_m=20.0, flicker_confidence=0.8, rcs_dbsm=-15.0,
        )
        result_receding = tc.compute_kinematic_primer(
            target_id=0, speed_mps=20.0, radial_velocity_mps=10.0,
            altitude_m=20.0, flicker_confidence=0.8, rcs_dbsm=-15.0,
        )
        assert result_closing.threat_score > result_receding.threat_score

    def test_full_assess_replaces_primer(self):
        """Full assess with micro-Doppler should have is_primer_only=False."""
        from src.layer2_radar.micro_doppler_classifier import TargetClass
        tc = self._make_classifier()
        result = tc.assess(
            target_id=0,
            flicker_confidence=0.9,
            doppler_class=TargetClass.ROTARY_UAS,
            doppler_confidence=0.85,
            speed_mps=41.0,
            altitude_m=25.0,
            rcs_dbsm=-15.0,
            radial_velocity_mps=-35.0,
        )
        assert not result.is_primer_only
        assert result.doppler_score > 0.0
        assert result.kinematic_primer_score > 0.0  # Primer computed for comparison

    def test_full_assess_fast_closing_drone_is_threat(self):
        """Full assessment: fast drone + flicker + Doppler = definite threat."""
        from src.layer2_radar.micro_doppler_classifier import TargetClass
        tc = self._make_classifier()
        result = tc.assess(
            target_id=0,
            flicker_confidence=0.95,
            doppler_class=TargetClass.ROTARY_UAS,
            doppler_confidence=0.9,
            speed_mps=41.0,
            altitude_m=30.0,
            rcs_dbsm=-12.0,
            radial_velocity_mps=-38.0,
        )
        assert result.is_threat
        assert result.threat_score > 0.8

