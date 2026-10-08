const $ = (id) => document.getElementById(id);

const ui = {
  driverBadge: $("driverBadge"), cameraBadge: $("cameraBadge"), statusBadge: $("statusBadge"),
  fps: $("fpsValue"), feedMode: $("feedMode"), resolution: $("resolutionValue"),
  target: $("targetValue"), calibration: $("calibrationValue"), mode: $("modeChip"),
  pan: $("panValue"), tilt: $("tiltValue"), panTrack: $("panTrack"), tiltTrack: $("tiltTrack"),
  panLimits: $("panLimits"), tiltLimits: $("tiltLimits"), tracking: $("trackingButton"),
  laserTest: $("laserTestButton"), autoLaser: $("autoLaserButton"),
  panRange: $("panRange"), tiltRange: $("tiltRange"), panOutput: $("panOutput"),
  tiltOutput: $("tiltOutput"), log: $("eventLog"), toast: $("toast"), cameraStage: $("cameraStage"),
  safetyMode: $("safetyMode"), l1Status: $("l1Status"), l1Sensor: $("l1Sensor"),
  l1Detector: $("l1Detector"), l1Event: $("l1Event"), flickerStats: $("flickerStats"),
  neuralStats: $("neuralStats"),
  roiStats: $("roiStats"),
  l2Status: $("l2Status"), l2Sensor: $("l2Sensor"), l2Simulation: $("l2Simulation"),
  l3Status: $("l3Status"), l3Mode: $("l3Mode"), trackCount: $("trackCount"),
  trackAngles: $("trackAngles"), trackConfidence: $("trackConfidence"),
  l4Status: $("l4Status"), l4Driver: $("l4Driver"), l4Control: $("l4Control"),
  profile: $("profileValue"), upstream: $("upstreamValue"),
  cameraFeed: $("cameraFeed"), rgbFeed: $("rgbFeedButton"),
  eventFeed: $("eventFeedButton"), eventRate: $("eventRateValue"),
  clickHint: $("clickHint"), targetReticle: $("targetReticle"),
  trimPanRange: $("trimPanRange"), trimTiltRange: $("trimTiltRange"),
  trimPanOutput: $("trimPanOutput"), trimTiltOutput: $("trimTiltOutput"),
  alignmentDistanceBadge: $("alignmentDistanceBadge"),
  alignmentParallaxBadge: $("alignmentParallaxBadge"),
  alignmentSource: $("alignmentSource"),
  resetTrimButton: $("resetTrimButton"),
};

const canvas = $("twinCanvas");
const ctx = canvas.getContext("2d");
let state = null;
let priorSignature = "";
let priorMessage = "";
let toastTimer = null;
let selectedFeed = "rgb";
let isDraggingTrimPan = false;
let isDraggingTrimTilt = false;

function signed(value, digits = 1) {
  return `${value >= 0 ? "+" : "\u2212"}${Math.abs(value).toFixed(digits)}\u00b0`;
}

function valueOrNA(value, suffix = "") {
  return value === null || value === undefined ? "N/A" : `${value}${suffix}`;
}

function addLog(message) {
  const entry = document.createElement("li");
  const time = document.createElement("time");
  time.textContent = new Date().toLocaleTimeString([], {hour12: false});
  entry.append(time, document.createTextNode(message));
  ui.log.prepend(entry);
  while (ui.log.children.length > 5) ui.log.lastElementChild.remove();
}

function showToast(message) {
  ui.toast.textContent = message;
  ui.toast.classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => ui.toast.classList.remove("visible"), 2600);
}

async function api(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body || {}),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

function updatePredatorUI(predator) {
  if (!predator) return;
  const l1 = predator.layer1;
  const l2 = predator.layer2;
  const l3 = predator.layer3;
  const l4 = predator.layer4;

  ui.safetyMode.textContent = predator.engagement.enabled ? "ENGAGEMENT STATE INVALID" : "ENGAGEMENT HARD DISABLED";
  ui.safetyMode.classList.toggle("fault", predator.engagement.enabled);
  ui.l1Status.textContent = l1.status;
  ui.l1Sensor.textContent = l1.sensor;
  ui.l1Detector.textContent = l1.detector;
  ui.l1Event.textContent = l1.event_camera;
  ui.flickerStats.textContent = l1.bpf_hz === null
    ? (l1.event_rate_mev_s === null
      ? "N/A (RGB INPUT)"
      : l1.event_mode === "flicker_detector"
        ? "SCANNING / NO FLICKER LOCK"
        : "RAW EVENT STREAM / FFT READY")
    : `${Number(l1.bpf_hz).toFixed(1)} Hz / ${Number(l1.rotor_rpm).toFixed(0)} / ${Number(l1.snr_db).toFixed(1)} dB`;
  const imu = l1.imu_connected ? `200 Hz (${l1.imu_packets})` : "OFFLINE";
  const combnet = l1.spectral_combnet_active
    ? `FP16 / ${l1.spectral_detections} LOCK`
    : "OFF";
  ui.neuralStats.textContent = `${imu} / ${combnet}`;
  ui.roiStats.textContent = `${l1.roi_active_cells || 0} / ${l1.event_tracks || 0}`;
  ui.l2Status.textContent = l2.status;
  ui.l2Sensor.textContent = l2.sensor;
  ui.l2Simulation.textContent = l2.simulated ? "ON" : "OFF";
  ui.l3Status.textContent = l3.status;
  ui.l3Mode.textContent = l3.mode;
  ui.trackCount.textContent = String(l3.track_count);
  ui.trackAngles.textContent = l3.azimuth_deg === null
    ? "N/A"
    : `${signed(l3.azimuth_deg)} / ${signed(l3.elevation_deg)}`;
  ui.trackConfidence.textContent = valueOrNA(l3.confidence === null ? null : Math.round(l3.confidence * 100), "%");
  ui.l4Status.textContent = l4.status;
  ui.l4Driver.textContent = l4.driver.toUpperCase();
  ui.l4Control.textContent = l4.control_state;
  ui.profile.textContent = predator.profile;
  ui.upstream.textContent = `${predator.upstream.repository} @ ${predator.upstream.commit}`;
}

function displayedImageRect(frameWidth, frameHeight) {
  const width = ui.cameraStage.clientWidth;
  const height = ui.cameraStage.clientHeight;
  const mediaAspect = Math.max(1, frameWidth) / Math.max(1, frameHeight);
  const stageAspect = width / Math.max(1, height);
  if (mediaAspect > stageAspect) {
    const imageHeight = width / mediaAspect;
    return {left: 0, top: (height - imageHeight) / 2, width, height: imageHeight};
  }
  const imageWidth = height * mediaAspect;
  return {left: (width - imageWidth) / 2, top: 0, width: imageWidth, height};
}

function updateTargetOverlay(target, frameWidth, frameHeight) {
  const matchesFeed = target && (
    (selectedFeed === "rgb" && target.source === "rgb") ||
    (selectedFeed === "event" && target.source === "event")
  );
  if (!matchesFeed) {
    ui.targetReticle.style.display = "none";
    return;
  }
  const rect = displayedImageRect(frameWidth, frameHeight);
  ui.targetReticle.style.left = `${rect.left + Number(target.x) * rect.width}px`;
  ui.targetReticle.style.top = `${rect.top + Number(target.y) * rect.height}px`;
  ui.targetReticle.style.display = "block";
}

function updateUI(next) {
  state = next;
  const eventCamera = next.event_camera || {connected: false};
  ui.eventFeed.disabled = !eventCamera.connected;
  if (selectedFeed === "event" && !eventCamera.connected) setFeed("rgb");
  ui.driverBadge.textContent = `DRIVER \u00b7 ${next.driver.toUpperCase()}`;
  ui.cameraBadge.textContent = eventCamera.connected
    ? "CAMERAS \u00b7 RGB + EVENT ONLINE"
    : `CAMERA \u00b7 ${next.camera_mode.toUpperCase()}`;
  ui.statusBadge.textContent = `SYSTEM \u00b7 ${next.status}`;
  ui.fps.textContent = Number(selectedFeed === "event" ? eventCamera.stream_fps : next.fps).toFixed(1);
  ui.feedMode.textContent = selectedFeed === "event"
    ? (eventCamera.num_targets ? "EVENT TARGET LOCK" : "EVENT LIVE / SCANNING")
    : (next.camera_mode === "live" ? "RGB LIVE INPUT" : "RGB SIMULATION");
  ui.resolution.textContent = selectedFeed === "event"
    ? (eventCamera.resolution || "1280x720").replace("x", " \u00d7 ")
    : `${next.frame_width} \u00d7 ${next.frame_height}`;
  ui.eventRate.textContent = eventCamera.connected
    ? `${Number(eventCamera.event_rate_mev_s).toFixed(2)} MEv/s`
    : "OFFLINE";
  ui.target.textContent = next.target ? `${next.target.label.toUpperCase()} / LOCK` : "NONE";
  ui.calibration.textContent = next.calibrated ? "MAPPED" : "REQUIRED";
  ui.mode.textContent = next.driver === "mock" ? "SAFE / MOCK" : "GIMBAL / LIVE";
  ui.pan.textContent = signed(next.pan);
  ui.tilt.textContent = signed(next.tilt);
  ui.tracking.textContent = next.tracking_enabled ? "PAUSE TRACKING" : "START TRACKING";
  ui.tracking.classList.toggle("button-primary", next.tracking_enabled);
  const laserTest = next.laser_test || {available: false, pulse_ms: 100, cooldown_remaining_s: 0};
  const laserCoolingDown = Number(laserTest.cooldown_remaining_s) > 0;
  ui.laserTest.disabled = !laserTest.available || next.tracking_enabled || laserCoolingDown;
  ui.laserTest.textContent = laserCoolingDown
    ? `LASER COOLDOWN · ${Number(laserTest.cooldown_remaining_s).toFixed(1)} S`
    : `LASER TEST PULSE · ${(Number(laserTest.pulse_ms) / 1000).toFixed(1)} S`;
  if (ui.autoLaser) {
    ui.autoLaser.textContent = next.auto_laser_enabled
      ? (next.laser_active ? "LASER ENGAGED [ACTIVE]" : "AUTO LASER: ARMED")
      : "AUTO LASER: OFF";
    ui.autoLaser.classList.toggle("button-danger", Boolean(next.auto_laser_enabled));
  }

  const limits = next.limits;
  const panPercent = 100 * (next.pan - limits.pan_min) / (limits.pan_max - limits.pan_min);
  const tiltPercent = 100 * (next.tilt - limits.tilt_min) / (limits.tilt_max - limits.tilt_min);
  ui.panTrack.style.width = `${Math.max(0, Math.min(100, panPercent))}%`;
  ui.tiltTrack.style.width = `${Math.max(0, Math.min(100, tiltPercent))}%`;
  ui.panLimits.textContent = `${signed(limits.pan_min, 0)} \u2014 ${signed(limits.pan_max, 0)}`;
  ui.tiltLimits.textContent = `${signed(limits.tilt_min, 0)} \u2014 ${signed(limits.tilt_max, 0)}`;
  ui.panRange.min = limits.pan_min; ui.panRange.max = limits.pan_max;
  ui.tiltRange.min = limits.tilt_min; ui.tiltRange.max = limits.tilt_max;
  updateTargetOverlay(next.target, next.frame_width, next.frame_height);
  updatePredatorUI(next.predator);

  if (next.alignment_live && ui.alignmentDistanceBadge) {
    const al = next.alignment_live;
    ui.alignmentDistanceBadge.textContent = `${al.distance_ft} FT (${al.distance_m} M)`;
    ui.alignmentParallaxBadge.textContent = `P ${signed(al.parallax_pan_deg)} / T ${signed(al.parallax_tilt_deg)}`;
    if (ui.alignmentSource) {
      ui.alignmentSource.textContent = al.distance_source.toUpperCase();
    }
    if (!isDraggingTrimPan && ui.trimPanRange) {
      ui.trimPanRange.value = al.trim_pan_deg;
      ui.trimPanOutput.textContent = signed(al.trim_pan_deg);
    }
    if (!isDraggingTrimTilt && ui.trimTiltRange) {
      ui.trimTiltRange.value = al.trim_tilt_deg;
      ui.trimTiltOutput.textContent = signed(al.trim_tilt_deg);
    }
  }

  const signature = `${next.status}|${next.camera_mode}|${Boolean(next.target)}|${next.tracking_enabled}|${eventCamera.connected}`;
  if (signature !== priorSignature) {
    addLog(`${next.status} \u00b7 ${next.target ? "target acquired" : "no target"}`);
    priorSignature = signature;
  }
  if (next.message && next.message !== priorMessage) showToast(next.message);
  priorMessage = next.message;
  drawTwin(next.pan, next.tilt, Boolean(next.target));
}

function setFeed(feed) {
  if (feed === "event" && (!state || !state.event_camera || !state.event_camera.connected)) {
    showToast("IDS event stream is not connected");
    return;
  }
  selectedFeed = feed;
  const source = feed === "event" ? "/event-stream.mjpg" : "/stream.mjpg";
  if (ui.cameraFeed.getAttribute("src") !== source) ui.cameraFeed.setAttribute("src", source);
  ui.rgbFeed.classList.toggle("active", feed === "rgb");
  ui.eventFeed.classList.toggle("active", feed === "event");
  ui.clickHint.textContent = feed === "event" ? "BART FFT TRACKING" : "CLICK FEED TO AIM";
  ui.cameraStage.style.cursor = feed === "event" ? "default" : "crosshair";
  if (state) updateUI(state);
}

function drawTwin(pan, tilt, locked) {
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#0c0f0d"; ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "#242b27"; ctx.lineWidth = 1;
  for (let x = 0; x <= w; x += 36) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
  for (let y = 0; y <= h; y += 36) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }

  const acid = "#49f8ae", muted = "#66736d", text = "#dce5e0";
  const panCx = 190, cy = 202, r = 118;
  ctx.strokeStyle = "#354039"; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(panCx, cy, r, 0, Math.PI * 2); ctx.stroke();
  ctx.setLineDash([3, 8]); ctx.strokeStyle = muted; ctx.beginPath(); ctx.arc(panCx, cy, r - 20, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]);
  for (let angle = -180; angle < 180; angle += 30) {
    const a = angle * Math.PI / 180;
    ctx.beginPath(); ctx.moveTo(panCx + Math.sin(a) * (r - 8), cy - Math.cos(a) * (r - 8));
    ctx.lineTo(panCx + Math.sin(a) * r, cy - Math.cos(a) * r); ctx.stroke();
  }
  const pa = pan * Math.PI / 180;
  ctx.strokeStyle = acid; ctx.shadowColor = acid; ctx.shadowBlur = 12; ctx.lineWidth = 4;
  ctx.beginPath(); ctx.moveTo(panCx, cy); ctx.lineTo(panCx + Math.sin(pa) * 91, cy - Math.cos(pa) * 91); ctx.stroke();
  ctx.shadowBlur = 0; ctx.fillStyle = acid; ctx.beginPath(); ctx.arc(panCx, cy, 8, 0, Math.PI * 2); ctx.fill();

  const sideCx = 532, sideCy = 230;
  ctx.strokeStyle = "#354039"; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(sideCx, sideCy, 100, Math.PI, Math.PI * 2); ctx.stroke();
  ctx.strokeStyle = muted; ctx.setLineDash([4, 7]); ctx.beginPath(); ctx.moveTo(sideCx - 120, sideCy); ctx.lineTo(sideCx + 120, sideCy); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = "#181d1a"; ctx.strokeStyle = "#4b5750"; ctx.lineWidth = 2;
  ctx.fillRect(sideCx - 58, sideCy + 20, 116, 24); ctx.strokeRect(sideCx - 58, sideCy + 20, 116, 24);
  ctx.fillRect(sideCx - 12, sideCy - 8, 24, 30); ctx.strokeRect(sideCx - 12, sideCy - 8, 24, 30);
  const ta = -tilt * Math.PI / 180;
  ctx.save(); ctx.translate(sideCx, sideCy - 8); ctx.rotate(ta);
  ctx.fillStyle = "#26302b"; ctx.strokeStyle = acid; ctx.fillRect(-9, -27, 104, 54); ctx.strokeRect(-9, -27, 104, 54);
  ctx.fillStyle = "#080a09"; ctx.fillRect(67, -18, 17, 36); ctx.strokeRect(67, -18, 17, 36);
  if (locked) { ctx.strokeStyle = acid; ctx.setLineDash([4,5]); ctx.beginPath(); ctx.moveTo(95, 0); ctx.lineTo(158, 0); ctx.stroke(); ctx.setLineDash([]); }
  ctx.restore();

  ctx.font = "11px monospace"; ctx.fillStyle = muted; ctx.fillText("TOP VIEW / PAN", 34, 38); ctx.fillText("SIDE VIEW / TILT", 398, 38);
  ctx.font = "20px monospace"; ctx.fillStyle = text; ctx.fillText(signed(pan), 142, 365); ctx.fillText(signed(tilt), 486, 365);
  ctx.font = "10px monospace"; ctx.fillStyle = locked ? acid : muted; ctx.fillText(locked ? "TARGET VECTOR LOCKED" : "AWAITING TARGET", 486, 70);
}

async function poll() {
  try {
    const response = await fetch("/api/state", {cache: "no-store"});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    updateUI(await response.json());
  } catch (error) {
    ui.statusBadge.textContent = "SYSTEM \u00b7 OFFLINE";
    showToast(`Dashboard connection lost: ${error.message}`);
  } finally {
    setTimeout(poll, 100);
  }
}

ui.tracking.addEventListener("click", async () => {
  try { updateUI(await api("/api/tracking", {enabled: !state.tracking_enabled})); }
  catch (error) { showToast(error.message); }
});

$("homeButton").addEventListener("click", async () => {
  try { updateUI(await api("/api/home")); addLog("Return-to-home command sent"); }
  catch (error) { showToast(error.message); }
});

ui.laserTest.addEventListener("click", async () => {
  const confirmed = window.confirm(
    "Confirm the beam is terminated by a matte nonreflective stop and the path is clear. Fire one 0.1 second test pulse?"
  );
  if (!confirmed) return;
  ui.laserTest.disabled = true;
  try {
    updateUI(await api("/api/laser-test/pulse"));
    addLog("Laser test pulse complete; output OFF");
  } catch (error) {
    showToast(error.message);
  }
});

if (ui.autoLaser) {
  ui.autoLaser.addEventListener("click", async () => {
    try {
      const nextEnabled = !(state && state.auto_laser_enabled);
      updateUI(await api("/api/laser-auto", {enabled: nextEnabled}));
      addLog(`Auto laser engagement ${nextEnabled ? "ARMED" : "DISARMED"}`);
    } catch (error) {
      showToast(error.message);
    }
  });
}

$("centerButton").addEventListener("click", async () => {
  try { updateUI(await api("/api/move", {pan: 0, tilt: 0})); }
  catch (error) { showToast(error.message); }
});

document.querySelectorAll(".manual-grid button[data-pan]").forEach((button) => {
  button.addEventListener("click", async () => {
    try {
      updateUI(await api("/api/move", {
        pan: state.pan + Number(button.dataset.pan), tilt: state.tilt + Number(button.dataset.tilt),
      }));
    } catch (error) { showToast(error.message); }
  });
});

ui.panRange.addEventListener("input", () => { ui.panOutput.textContent = `${ui.panRange.value}\u00b0`; });
ui.tiltRange.addEventListener("input", () => { ui.tiltOutput.textContent = `${ui.tiltRange.value}\u00b0`; });
$("sendButton").addEventListener("click", async () => {
  try { updateUI(await api("/api/move", {pan: Number(ui.panRange.value), tilt: Number(ui.tiltRange.value)})); }
  catch (error) { showToast(error.message); }
});

if (ui.trimPanRange) {
  ui.trimPanRange.addEventListener("mousedown", () => { isDraggingTrimPan = true; });
  ui.trimPanRange.addEventListener("touchstart", () => { isDraggingTrimPan = true; }, {passive: true});
  ui.trimPanRange.addEventListener("input", () => {
    ui.trimPanOutput.textContent = signed(Number(ui.trimPanRange.value));
  });
  ui.trimPanRange.addEventListener("change", async () => {
    isDraggingTrimPan = false;
    try {
      await api("/api/alignment", { trim_pan_deg: Number(ui.trimPanRange.value) });
      addLog(`Pan trim set to ${signed(Number(ui.trimPanRange.value))}`);
    } catch (e) { showToast(e.message); }
  });
}

if (ui.trimTiltRange) {
  ui.trimTiltRange.addEventListener("mousedown", () => { isDraggingTrimTilt = true; });
  ui.trimTiltRange.addEventListener("touchstart", () => { isDraggingTrimTilt = true; }, {passive: true});
  ui.trimTiltRange.addEventListener("input", () => {
    ui.trimTiltOutput.textContent = signed(Number(ui.trimTiltRange.value));
  });
  ui.trimTiltRange.addEventListener("change", async () => {
    isDraggingTrimTilt = false;
    try {
      await api("/api/alignment", { trim_tilt_deg: Number(ui.trimTiltRange.value) });
      addLog(`Tilt trim set to ${signed(Number(ui.trimTiltRange.value))}`);
    } catch (e) { showToast(e.message); }
  });
}

document.querySelectorAll(".nudge-btn").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const axis = btn.dataset.axis;
    const step = Number(btn.dataset.step);
    if (axis === "pan" && ui.trimPanRange) {
      const cur = Number(ui.trimPanRange.value);
      const nextVal = Math.round((cur + step) * 10) / 10;
      ui.trimPanRange.value = nextVal;
      ui.trimPanOutput.textContent = signed(nextVal);
      try {
        await api("/api/alignment", { trim_pan_deg: nextVal });
        addLog(`Pan trim nudged to ${signed(nextVal)}`);
      } catch (e) { showToast(e.message); }
    } else if (axis === "tilt" && ui.trimTiltRange) {
      const cur = Number(ui.trimTiltRange.value);
      const nextVal = Math.round((cur + step) * 10) / 10;
      ui.trimTiltRange.value = nextVal;
      ui.trimTiltOutput.textContent = signed(nextVal);
      try {
        await api("/api/alignment", { trim_tilt_deg: nextVal });
        addLog(`Tilt trim nudged to ${signed(nextVal)}`);
      } catch (e) { showToast(e.message); }
    }
  });
});

if (ui.resetTrimButton) {
  ui.resetTrimButton.addEventListener("click", async () => {
    if (ui.trimPanRange) {
      ui.trimPanRange.value = 0.0;
      ui.trimPanOutput.textContent = signed(0.0);
    }
    if (ui.trimTiltRange) {
      ui.trimTiltRange.value = 0.0;
      ui.trimTiltOutput.textContent = signed(0.0);
    }
    try {
      await api("/api/alignment", { trim_pan_deg: 0.0, trim_tilt_deg: 0.0 });
      addLog("Trim reset to 0.0° / 0.0°");
    } catch (e) { showToast(e.message); }
  });
}

ui.cameraStage.addEventListener("click", async (event) => {
  if (selectedFeed === "event") {
    showToast("Event targets drive the mock gimbal from Bart's bearing telemetry");
    return;
  }
  const stage = ui.cameraStage.getBoundingClientRect();
  const image = displayedImageRect(state.frame_width, state.frame_height);
  const x = Math.max(0, Math.min(1, (event.clientX - stage.left - image.left) / image.width));
  const y = Math.max(0, Math.min(1, (event.clientY - stage.top - image.top) / image.height));
  try {
    updateUI(await api("/api/point", {x, y}));
    addLog("Click-to-aim command sent");
  } catch (error) { showToast(error.message); }
});

ui.rgbFeed.addEventListener("click", () => setFeed("rgb"));
ui.eventFeed.addEventListener("click", () => setFeed("event"));

drawTwin(0, 0, false);
poll();
