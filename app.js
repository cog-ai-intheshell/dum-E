"use strict";

const GRAVITY = 9.81;
const DT = 0.02;
const MAX_STEPS = 250;
const PLAYER_X = 0.0;
const PLAYER_Y = 0.0;
const HOOP_Y = 0.0;
const HOOP_RADIUS = 0.23;
const TWO_PI = Math.PI * 2;
const CAMERA_PITCH_LIMIT = Math.PI / 2;
const CAMERA_MIN_ZOOM = 0.28;
const CAMERA_MAX_ZOOM = 4.0;
const FIXED_SCENE_BOUNDS = Object.freeze({
  xMin: -0.75,
  xMax: 13.15,
  yMin: -4.5,
  yMax: 4.5,
  zMin: 0,
  zMax: 5.6,
});
const DEFAULT_CAMERA = Object.freeze({
  yaw: -Math.PI / 2,
  pitch: 0,
  zoom: 1,
});
const CAMERA_VIEWS = Object.freeze({
  profile: DEFAULT_CAMERA,
  top: { yaw: DEFAULT_CAMERA.yaw, pitch: CAMERA_PITCH_LIMIT },
  front: { yaw: Math.PI, pitch: 0 },
  back: { yaw: 0, pitch: 0 },
  left: { yaw: Math.PI / 2, pitch: 0 },
  right: { yaw: -Math.PI / 2, pitch: 0 },
});
const WIND_VISUALIZATION_MODES = ["vector", "field", "volume"];
const WIND_VISUALIZATION_LABELS = Object.freeze({
  vector: "Champ",
  field: "Volume",
  volume: "Vecteurs",
});
const WIND_VISUALIZATION_TITLES = Object.freeze({
  vector: "Passer en carte de champ",
  field: "Passer en volume 3D",
  volume: "Revenir en visualisation vecteurs",
});
const WORLD_MODEL_ENDPOINT = "http://127.0.0.1:8765/predict";
const WORLD_MODEL_HISTORY_STEPS = 12;
const WORLD_MODEL_HORIZON_STEPS = 30;
const WORLD_MODEL_MIN_INTERVAL_MS = 120;

const WIND_REGIME_NAMES = {
  0: "vent nul",
  1: "vent constant",
  2: "vent random",
  3: "vent markovien",
  4: "vent cyclique",
  5: "rafales",
  6: "changement brutal",
  7: "mean reversion",
  8: "clustering volatilité",
  9: "jump / news shock",
  10: "pump & fade",
  11: "cascade liquidations",
  12: "liquidity wall",
  13: "squeeze breakout",
  14: "hidden regime switching",
  15: "vent chaotique",
};

const SHOTS_COLUMNS = [
  "shot_id",
  "initial_force",
  "vertical_angle",
  "horizontal_angle",
  "distance_to_hoop",
  "boy_height",
  "ball_mass",
  "hoop_height",
  "wind_regime_id",
  "wind_regime",
  "wind_strength",
  "wind_orientation",
  "wind_vertical_orientation",
  "wind_spatial_coupling",
  "drag_coeff",
  "gravity",
];

const TRAJECTORY_COLUMNS = [
  "shot_id",
  "timestep",
  "time",
  "ball_x",
  "ball_y",
  "ball_z",
  "ball_vx",
  "ball_vy",
  "ball_vz",
  "ball_ax",
  "ball_ay",
  "ball_az",
  "distance_ball_to_hoop",
  "horizontal_distance_to_hoop",
  "speed_norm",
];

const WIND_FIELD_COLUMNS = [
  "shot_id",
  "timestep",
  "time",
  "wind_vector_id",
  "wind_grid_x",
  "wind_grid_y",
  "wind_grid_z",
  "wind_x",
  "wind_y",
  "wind_z",
  "wind_norm",
];

const LABEL_COLUMNS = [
  "shot_id",
  "label",
  "result",
  "distance_to_hoop",
  "final_min_hoop_distance",
  "collision_time",
];

const PREVIEW_COLUMNS = [
  "shot_id",
  "timestep",
  "time",
  "ball_x",
  "ball_y",
  "ball_z",
  "ball_vx",
  "ball_vy",
  "ball_vz",
  "ball_ax",
  "ball_ay",
  "ball_az",
  "distance_ball_to_hoop",
  "speed_norm",
  "...",
];

const INTEGER_COLUMNS = new Set([
  "shot_id",
  "timestep",
  "label",
  "wind_spatial_coupling",
  "wind_regime_id",
  "wind_vector_id",
]);

const sliderFormat = {
  initialForce: 1,
  verticalAngle: 1,
  horizontalAngle: 1,
  distanceToHoop: 1,
  boyHeight: 2,
  ballMass: 2,
  hoopHeight: 2,
  windStrength: 1,
  windOrientation: 1,
  windVerticalOrientation: 1,
  dragCoeff: 3,
};

const controls = {
  initialForce: document.getElementById("initialForce"),
  verticalAngle: document.getElementById("verticalAngle"),
  horizontalAngle: document.getElementById("horizontalAngle"),
  distanceToHoop: document.getElementById("distanceToHoop"),
  boyHeight: document.getElementById("boyHeight"),
  ballMass: document.getElementById("ballMass"),
  hoopHeight: document.getElementById("hoopHeight"),
  windRegime: document.getElementById("windRegime"),
  windCoupling: document.getElementById("windCoupling"),
  windStrength: document.getElementById("windStrength"),
  windOrientation: document.getElementById("windOrientation"),
  windVerticalOrientation: document.getElementById("windVerticalOrientation"),
  dragCoeff: document.getElementById("dragCoeff"),
};

const dom = {
  canvas: document.getElementById("sceneCanvas"),
  resultTitle: document.getElementById("resultTitle"),
  timeTitle: document.getElementById("timeTitle"),
  goButton: document.getElementById("goButton"),
  aiCamerasLink: document.getElementById("aiCamerasLink"),
  resetCamera: document.getElementById("resetCamera"),
  windVizToggle: document.getElementById("windVizToggle"),
  worldModelToggle: document.getElementById("worldModelToggle"),
  themeToggle: document.getElementById("themeToggle"),
  viewCubeToggle: document.getElementById("viewCubeToggle"),
  viewCubeMenu: document.getElementById("viewCubeMenu"),
  lightTheme: document.getElementById("lightTheme"),
  darkTheme: document.getElementById("darkTheme"),
  playButton: document.getElementById("playButton"),
  stepBackButton: document.getElementById("stepBackButton"),
  stepForwardButton: document.getElementById("stepForwardButton"),
  timeSlider: document.getElementById("timeSlider"),
  resultsBody: document.getElementById("resultsBody"),
  csvPreview: document.getElementById("csvPreview"),
  downloadCsv: document.getElementById("downloadCsv"),
  csvDownloadButtons: Array.from(document.querySelectorAll("[data-csv-download]")),
  showDragVector: document.getElementById("showDragVector"),
};

let shotCounter = 0;
let animationFrame = null;
let previousAnimationTime = 0;
let state = null;
let cameraDrag = null;
let windVisualizationMode = "vector";
let worldModelEnabled = false;
let worldModelStatus = "idle";
let worldModelAbortController = null;
let worldModelRequestId = 0;
let worldModelLastRequestTime = 0;

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function smoothstep(edge0, edge1, value) {
  const t = clamp((value - edge0) / Math.max(edge1 - edge0, 1e-9), 0, 1);
  return t * t * (3 - 2 * t);
}

function add(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function sub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function scale(v, scalar) {
  return [v[0] * scalar, v[1] * scalar, v[2] * scalar];
}

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function norm(v) {
  return Math.hypot(v[0], v[1], v[2]);
}

function normalize(v) {
  const length = norm(v);
  if (length < 1e-12) {
    return [0, 0, 0];
  }
  return [v[0] / length, v[1] / length, v[2] / length];
}

function degToRad(degrees) {
  return (degrees * Math.PI) / 180;
}

function hoopX(params) {
  return params.distanceToHoop;
}

function hoopCenter(params) {
  return [hoopX(params), HOOP_Y, params.hoopHeight];
}

function makeDefaultCamera() {
  return {
    yaw: DEFAULT_CAMERA.yaw,
    pitch: DEFAULT_CAMERA.pitch,
    zoom: DEFAULT_CAMERA.zoom,
  };
}

function rawCssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function resolveCssValue(value, fallback, depth = 0) {
  if (!value || depth > 5) {
    return fallback;
  }
  return value.replace(/var\(\s*(--[\w-]+)\s*(?:,\s*([^)]+))?\)/g, (_match, name, localFallback) => {
    const nested = rawCssVar(name) || localFallback || fallback;
    return resolveCssValue(nested.trim(), fallback, depth + 1);
  }).trim();
}

function cssVar(name, fallback) {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value ? resolveCssValue(value, fallback) : fallback;
}

function cssVarAny(names, fallback) {
  for (const name of names) {
    const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    if (value) {
      return resolveCssValue(value, fallback);
    }
  }
  return fallback;
}

function isDarkTheme() {
  return document.body.dataset.theme === "dark";
}

function applyTheme(theme) {
  const nextTheme = theme === "dark" ? "dark" : "light";
  document.body.dataset.theme = nextTheme;
  dom.lightTheme.disabled = nextTheme !== "light";
  dom.darkTheme.disabled = nextTheme !== "dark";
  dom.themeToggle.textContent = nextTheme === "dark" ? "☀" : "☾";
  dom.themeToggle.setAttribute(
    "aria-label",
    nextTheme === "dark" ? "Activer le light mode" : "Activer le dark mode"
  );
  dom.themeToggle.title = nextTheme === "dark" ? "Activer le light mode" : "Activer le dark mode";
  dom.themeToggle.setAttribute("aria-pressed", String(nextTheme === "dark"));
  localStorage.setItem("basketball-simulator-theme", nextTheme);
  requestAnimationFrame(drawScene);
}

function toggleTheme() {
  applyTheme(isDarkTheme() ? "light" : "dark");
}

function readStoredTheme() {
  return localStorage.getItem("basketball-simulator-theme") === "dark" ? "dark" : "light";
}

function readStoredWindVisualizationMode() {
  const storedMode = localStorage.getItem("basketball-simulator-wind-visualization");
  return WIND_VISUALIZATION_MODES.includes(storedMode) ? storedMode : "vector";
}

function updateWindVisualizationButton() {
  dom.windVizToggle.textContent = WIND_VISUALIZATION_LABELS[windVisualizationMode] || "Champ";
  dom.windVizToggle.title = WIND_VISUALIZATION_TITLES[windVisualizationMode] || "Passer en carte de champ";
  dom.windVizToggle.setAttribute("aria-pressed", String(windVisualizationMode !== "vector"));
  dom.windVizToggle.setAttribute("aria-label", dom.windVizToggle.title);
}

function toggleWindVisualizationMode() {
  const currentIndex = WIND_VISUALIZATION_MODES.indexOf(windVisualizationMode);
  windVisualizationMode = WIND_VISUALIZATION_MODES[(currentIndex + 1) % WIND_VISUALIZATION_MODES.length];
  localStorage.setItem("basketball-simulator-wind-visualization", windVisualizationMode);
  updateWindVisualizationButton();
  drawScene();
}

function updateWorldModelButton() {
  if (!dom.worldModelToggle) {
    return;
  }
  const probability = state?.worldModelMomentum?.probability;
  const hasMomentum = Number.isFinite(probability);
  const momentumPercent = hasMomentum ? Math.round(probability * 100) : null;
  const titles = {
    idle: "Show world-model predictions",
    loading: "World model: prediction in progress",
    ready: hasMomentum
      ? `World model active: blue trajectory, momentum ${momentumPercent}%`
      : "World model active: predicted trajectory in blue",
    offline: "World model unavailable: run python -m basketball_sim.serving.world_model_server",
  };
  dom.worldModelToggle.textContent = worldModelEnabled && hasMomentum ? `M ${momentumPercent}%` : "WM";
  dom.worldModelToggle.setAttribute("aria-pressed", String(worldModelEnabled));
  dom.worldModelToggle.title = titles[worldModelStatus] || titles.idle;
  dom.worldModelToggle.setAttribute("aria-label", dom.worldModelToggle.title);
  dom.worldModelToggle.classList.toggle("is-loading", worldModelStatus === "loading");
}

function worldModelHistoryRows() {
  if (!state) {
    return [];
  }
  const start = Math.max(0, state.frameIndex - WORLD_MODEL_HISTORY_STEPS + 1);
  return state.rows.slice(start, state.frameIndex + 1);
}

function makeWorldModelPayload() {
  return {
    params: {
      ...state.params,
      gravity: GRAVITY,
    },
    history: worldModelHistoryRows(),
    horizon: WORLD_MODEL_HORIZON_STEPS,
    dt: DT,
    frameIndex: state.frameIndex,
  };
}

async function requestWorldModelPrediction(force = false) {
  if (!worldModelEnabled || !state || state.rows.length === 0) {
    return;
  }
  const now = performance.now();
  if (!force && now - worldModelLastRequestTime < WORLD_MODEL_MIN_INTERVAL_MS) {
    return;
  }
  worldModelLastRequestTime = now;
  if (worldModelAbortController) {
    worldModelAbortController.abort();
  }

  const requestId = worldModelRequestId + 1;
  const frameIndex = state.frameIndex;
  const controller = new AbortController();
  worldModelRequestId = requestId;
  worldModelAbortController = controller;
  worldModelStatus = "loading";
  updateWorldModelButton();

  try {
    const response = await fetch(WORLD_MODEL_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(makeWorldModelPayload()),
      signal: controller.signal,
    });
    const data = await response.json();
    if (!response.ok || !data.ok) {
      throw new Error(data.error || "world model request failed");
    }
    if (!state || requestId !== worldModelRequestId) {
      return;
    }
    if (state.frameIndex !== frameIndex) {
      worldModelStatus = "idle";
      updateWorldModelButton();
      requestAnimationFrame(() => requestWorldModelPrediction(true));
      return;
    }
    state.worldModelPrediction = Array.isArray(data.predictions) ? data.predictions : [];
    state.worldModelMomentum = data.momentum || null;
    worldModelStatus = "ready";
    updateWorldModelButton();
    drawScene();
  } catch (error) {
    if (error.name === "AbortError") {
      return;
    }
    if (requestId === worldModelRequestId && state) {
      state.worldModelPrediction = [];
      state.worldModelMomentum = null;
      worldModelStatus = "offline";
      updateWorldModelButton();
      drawScene();
    }
  } finally {
    if (requestId === worldModelRequestId) {
      worldModelAbortController = null;
    }
  }
}

function toggleWorldModel() {
  worldModelEnabled = !worldModelEnabled;
  if (!worldModelEnabled) {
    if (worldModelAbortController) {
      worldModelAbortController.abort();
      worldModelAbortController = null;
    }
    if (state) {
      state.worldModelPrediction = [];
      state.worldModelMomentum = null;
    }
    worldModelStatus = "idle";
    updateWorldModelButton();
    drawScene();
    return;
  }
  worldModelStatus = "loading";
  updateWorldModelButton();
  requestWorldModelPrediction(true);
}

function hashString(text) {
  let hash = 2166136261;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function mulberry32(seed) {
  return function random() {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function normalSample(random) {
  const u1 = Math.max(random(), 1e-12);
  const u2 = Math.max(random(), 1e-12);
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(TWO_PI * u2);
}

function vectorLimited(v, maxNorm) {
  const length = norm(v);
  if (length <= maxNorm || length < 1e-12) {
    return v;
  }
  return scale(v, maxNorm / length);
}

function updateSliderOutput(id) {
  const input = controls[id];
  const output = document.getElementById(`${id}Value`);
  if (!input || !output) {
    return;
  }
  updateRangeFill(input);
  output.value = Number(input.value).toFixed(sliderFormat[id]);
}

function updateRangeFill(input) {
  const min = Number(input.min || 0);
  const max = Number(input.max || 100);
  const value = Number(input.value || 0);
  const progress = max === min ? 0 : clamp(((value - min) / (max - min)) * 100, 0, 100);
  input.style.setProperty("--range-progress", `${progress}%`);
}

function updateAllRangeFills() {
  document.querySelectorAll('input[type="range"]').forEach(updateRangeFill);
}

function readParams() {
  const windRegimeId = Number(controls.windRegime.value);
  return {
    shotId: shotCounter,
    initialForce: Number(controls.initialForce.value),
    verticalAngle: Number(controls.verticalAngle.value),
    horizontalAngle: Number(controls.horizontalAngle.value),
    distanceToHoop: Number(controls.distanceToHoop.value),
    boyHeight: Number(controls.boyHeight.value),
    ballMass: Number(controls.ballMass.value),
    hoopHeight: Number(controls.hoopHeight.value),
    windRegimeId,
    windRegime: WIND_REGIME_NAMES[windRegimeId],
    windSpatialCoupling: controls.windCoupling.checked ? 1 : 0,
    windStrength: Number(controls.windStrength.value),
    windOrientation: Number(controls.windOrientation.value),
    windVerticalOrientation: Number(controls.windVerticalOrientation.value),
    dragCoeff: Number(controls.dragCoeff.value),
  };
}

function makeBaseWind(params) {
  const azimuth = degToRad(params.windOrientation);
  const elevation = degToRad(params.windVerticalOrientation);
  const horizontalStrength = params.windStrength * Math.cos(elevation);
  return [
    horizontalStrength * Math.cos(azimuth),
    horizontalStrength * Math.sin(azimuth),
    params.windStrength * Math.sin(elevation),
  ];
}

function createModes(random, count) {
  const modes = [];
  for (let index = 0; index < count; index += 1) {
    const direction = normalize([
      random() * 2 - 1,
      random() * 2 - 1,
      0.65 * (random() * 2 - 1),
    ]);
    modes.push({
      k: [
        (0.35 + 1.15 * random()) * (random() < 0.5 ? -1 : 1),
        (0.35 + 1.25 * random()) * (random() < 0.5 ? -1 : 1),
        (0.28 + 0.95 * random()) * (random() < 0.5 ? -1 : 1),
      ],
      omega: 0.45 + 1.9 * random(),
      phase: TWO_PI * random(),
      direction,
      amplitude: 0.65 + 0.7 * random(),
    });
  }
  return modes;
}

function spectralNoise(x, y, z, t, modes) {
  let total = [0, 0, 0];
  let amplitudeTotal = 0;
  for (const mode of modes) {
    const phase =
      mode.k[0] * x +
      mode.k[1] * y +
      mode.k[2] * z +
      mode.omega * t +
      mode.phase;
    const wave = Math.sin(phase);
    total = add(total, scale(mode.direction, mode.amplitude * wave));
    amplitudeTotal += mode.amplitude;
  }
  if (amplitudeTotal <= 1e-12) {
    return total;
  }
  return scale(total, 1 / amplitudeTotal);
}

function lorenzDerivative(state) {
  const sigma = 10;
  const rho = 28;
  const beta = 8 / 3;
  return [
    sigma * (state[1] - state[0]),
    state[0] * (rho - state[2]) - state[1],
    state[0] * state[1] - beta * state[2],
  ];
}

function lorenzStep(state, dt) {
  const k1 = lorenzDerivative(state);
  const k2 = lorenzDerivative(add(state, scale(k1, dt / 2)));
  const k3 = lorenzDerivative(add(state, scale(k2, dt / 2)));
  const k4 = lorenzDerivative(add(state, scale(k3, dt)));
  return state.map((value, index) => (
    value + (dt / 6) * (k1[index] + 2 * k2[index] + 2 * k3[index] + k4[index])
  ));
}

function createChaoticStates(random, strength) {
  const states = [];
  let lorenzState = [
    0.72 + 0.34 * random(),
    0.95 + 0.42 * random(),
    1.08 + 0.38 * random(),
  ];
  const lorenzDt = 0.01;

  for (let burn = 0; burn < 140; burn += 1) {
    lorenzState = lorenzStep(lorenzState, lorenzDt);
  }

  for (let step = 0; step < MAX_STEPS; step += 1) {
    for (let substep = 0; substep < 8; substep += 1) {
      lorenzState = lorenzStep(lorenzState, lorenzDt);
    }
    states.push(vectorLimited([
      lorenzState[0] / 20,
      lorenzState[1] / 27,
      (lorenzState[2] - 25) / 25,
    ], 1.45 + 0.05 * Math.log1p(strength)));
  }

  return states;
}

function createWindContext(params) {
  const seedText = [
    params.initialForce,
    params.verticalAngle,
    params.horizontalAngle,
    params.distanceToHoop,
    params.boyHeight,
    params.ballMass,
    params.hoopHeight,
    params.windRegimeId,
    params.windSpatialCoupling,
    params.windStrength,
    params.windOrientation,
    params.windVerticalOrientation,
    params.dragCoeff,
  ].join("|");
  const random = mulberry32(hashString(seedText));
  const modes = createModes(random, 12);
  const base = makeBaseWind(params);
  const markovStates = [];
  const tau = 0.82;
  const alpha = Math.exp(-DT / tau);
  const sigma = 0.52 * params.windStrength * Math.sqrt(1 - alpha * alpha);
  let markovState = base.slice();

  for (let step = 0; step < MAX_STEPS; step += 1) {
    const innovation = [
      sigma * normalSample(random),
      sigma * normalSample(random),
      0.38 * sigma * normalSample(random),
    ];
    const target = add(base, scale(spectralNoise(-2, 0, 2, step * DT, modes), 0.2 * params.windStrength));
    markovState = add(target, add(scale(sub(markovState, target), alpha), innovation));
    markovState = vectorLimited(markovState, Math.max(0.1, 2.25 * params.windStrength));
    markovStates.push(markovState.slice());
  }

  const gusts = Array.from({ length: 5 }, (_, index) => ({
    center: 0.55 + index * 0.86 + (random() - 0.5) * 0.2,
    width: 0.07 + random() * 0.08,
    amplitude: 0.85 + random() * 1.45,
    lateralPhase: TWO_PI * random(),
  }));

  const volatilityStates = [];
  const volatilityAlpha = Math.exp(-DT / 0.9);
  const volatilitySigma = Math.sqrt(1 - volatilityAlpha * volatilityAlpha);
  let volatilityState = 0;
  for (let step = 0; step < MAX_STEPS; step += 1) {
    volatilityState = volatilityAlpha * volatilityState + volatilitySigma * normalSample(random);
    volatilityStates.push(clamp(Math.exp(0.72 * volatilityState), 0.28, 3.8));
  }

  const shocks = Array.from({ length: 3 }, (_, index) => {
    const shockAngle = degToRad(params.windOrientation) + (random() - 0.5) * Math.PI * 1.35;
    return {
      center: 0.85 + index * 1.22 + (random() - 0.5) * 0.38,
      width: 0.045 + random() * 0.105,
      amplitude: 1.25 + random() * 1.85,
      direction: normalize([
        Math.cos(shockAngle),
        Math.sin(shockAngle),
        0.18 * (random() * 2 - 1),
      ]),
      spatialPhase: TWO_PI * random(),
    };
  });

  const liquidityWalls = Array.from({ length: 3 }, (_, index) => ({
    x: params.distanceToHoop * (0.34 + index * 0.18 + (random() - 0.5) * 0.08),
    y: (random() - 0.5) * 3.2,
    z: 1.05 + random() * 2.95,
    widthX: 0.24 + random() * 0.22,
    widthY: 0.65 + random() * 0.65,
    widthZ: 0.75 + random() * 0.65,
    amplitude: 0.75 + random() * 0.85,
    polarity: random() < 0.72 ? 1 : -1,
  }));

  const hiddenStates = [];
  let hiddenState = Math.floor(random() * 4);
  for (let step = 0; step < MAX_STEPS; step += 1) {
    if (step > 0 && random() < 0.018) {
      hiddenState = (hiddenState + 1 + Math.floor(random() * 3)) % 4;
    }
    hiddenStates.push(hiddenState);
  }
  const chaoticStates = createChaoticStates(random, params.windStrength);

  return {
    base,
    modes,
    markovStates,
    gusts,
    volatilityStates,
    shocks,
    liquidityWalls,
    hiddenStates,
    chaoticStates,
    squeezeTime: 1.35 + random() * 1.45,
  };
}

function spatialCouplingWindAt(x, y, z, t, base, perpendicular, strength) {
  const mixed = [
    0.42 * strength * Math.sin(1.15 * y + 1.22 * t) + 0.18 * strength * Math.cos(0.55 * z + 0.32 * x),
    0.42 * strength * Math.cos(0.78 * x - 1.05 * t) + 0.2 * strength * Math.sin(0.82 * z + 0.55 * y),
    0.32 * strength * Math.sin(1.1 * z + 1.72 * t) + 0.23 * strength * Math.cos(0.7 * x + 0.62 * y - t),
  ];
  const swirl = [
    -0.18 * strength * Math.sin(0.55 * y + t),
    0.18 * strength * Math.sin(0.45 * x - 0.7 * t),
    0.12 * strength * Math.sin(0.5 * x - 0.4 * y + t),
  ];
  const directionalShear =
    0.16 * strength * Math.sin(0.42 * x + 0.68 * y + 0.55 * z + 0.9 * t);
  return add(add(mixed, swirl), add(scale(base, 0.1 * Math.sin(0.65 * z + t)), scale(perpendicular, directionalShear)));
}

function windAt(x, y, z, t, params, windContext) {
  const strength = params.windStrength;
  if (strength <= 1e-9 || params.windRegimeId === 0) {
    return [0, 0, 0];
  }

  const base = windContext.base;
  const orientation = degToRad(params.windOrientation);
  const perpendicular = [-Math.sin(orientation), Math.cos(orientation), 0];
  const baseDirection = normalize(base);
  const distanceScale = Math.max(params.distanceToHoop, 1);
  const withSpatialCoupling = (wind) => {
    if (!params.windSpatialCoupling) {
      return wind;
    }
    return add(wind, spatialCouplingWindAt(x, y, z, t, base, perpendicular, strength));
  };

  switch (params.windRegimeId) {
    case 1:
      return withSpatialCoupling(base.slice());

    case 2: {
      const turbulent = spectralNoise(x, y, z, t, windContext.modes);
      const pulsingBase = scale(base, 0.82 + 0.18 * Math.sin(1.7 * t + 0.4 * x));
      return withSpatialCoupling(add(pulsingBase, scale(turbulent, 0.72 * strength)));
    }

    case 3: {
      const step = clamp(Math.round(t / DT), 0, windContext.markovStates.length - 1);
      const local = spectralNoise(x, y, z, t, windContext.modes);
      return withSpatialCoupling(add(windContext.markovStates[step], scale(local, 0.22 * strength)));
    }

    case 4: {
      const phase = TWO_PI * t / 2.35;
      return withSpatialCoupling(add(
        add(scale(base, Math.sin(phase)), scale(perpendicular, 0.34 * strength * Math.cos(phase))),
        [0, 0, 0.22 * strength * Math.sin(2 * phase + 0.45 * x)]
      ));
    }

    case 5: {
      let gustFactor = 1.0;
      for (const gust of windContext.gusts) {
        const timePulse = Math.exp(-((t - gust.center) ** 2) / (2 * gust.width ** 2));
        const spatialPulse = 0.76 + 0.24 * Math.sin(0.7 * x - 0.5 * y + gust.lateralPhase);
        gustFactor += gust.amplitude * timePulse * spatialPulse;
      }
      const turbulent = spectralNoise(x, y, z, t, windContext.modes);
      return withSpatialCoupling(add(
        add(scale(base, gustFactor), scale(perpendicular, 0.16 * strength * Math.sin(2.8 * t + y))),
        scale(turbulent, 0.26 * strength)
      ));
    }

    case 6: {
      const switchTime = (MAX_STEPS * DT) * 0.44;
      const sign = t < switchTime ? 1 : -1;
      const shear = 0.18 * strength * Math.sin(0.9 * z + 0.5 * y);
      return withSpatialCoupling(add(scale(base, sign), scale(perpendicular, shear)));
    }

    case 7: {
      const progress = clamp(x / distanceScale, 0, 1.15);
      const anchorX = 0.55 * distanceScale;
      const targetZ = lerp(params.releaseHeight, params.hoopHeight, progress);
      const meanPull = [
        -0.32 * strength * Math.tanh((x - anchorX) / (0.35 * distanceScale)),
        -0.78 * strength * Math.tanh((y - HOOP_Y) / 2.1),
        0.25 * strength * Math.tanh((targetZ - z) / 1.35),
      ];
      const residual = scale(spectralNoise(x, y, z, t, windContext.modes), 0.12 * strength);
      return withSpatialCoupling(add(add(scale(base, 0.18 + 0.12 * Math.cos(1.4 * t)), meanPull), residual));
    }

    case 8: {
      const step = clamp(Math.round(t / DT), 0, windContext.volatilityStates.length - 1);
      const volatility = windContext.volatilityStates[step] ?? 1;
      const turbulent = spectralNoise(x, y, z, t, windContext.modes);
      const clusteredNoise = scale(turbulent, strength * (0.22 + 0.58 * volatility));
      const persistentDrift = scale(base, 0.32 + 0.16 * volatility);
      const lateralChop = scale(perpendicular, 0.14 * strength * volatility * Math.sin(2.2 * t + 0.45 * x - 0.3 * y));
      return withSpatialCoupling(add(add(persistentDrift, clusteredNoise), lateralChop));
    }

    case 9: {
      let shockWind = scale(base, 0.22);
      for (const shock of windContext.shocks) {
        const timePulse = Math.exp(-((t - shock.center) ** 2) / (2 * shock.width ** 2));
        const spatialPulse = 0.72 + 0.28 * Math.sin(0.65 * x - 0.48 * y + 0.25 * z + shock.spatialPhase);
        shockWind = add(shockWind, scale(shock.direction, strength * shock.amplitude * timePulse * spatialPulse));
      }
      return withSpatialCoupling(add(shockWind, scale(spectralNoise(x, y, z, t, windContext.modes), 0.1 * strength)));
    }

    case 10: {
      const pump = smoothstep(0.15, 0.75, t) * (1 - smoothstep(1.35, 2.25, t));
      const fade = smoothstep(1.75, 3.45, t);
      const directionalFactor = 0.16 + 1.65 * pump - 0.82 * fade;
      const lateralUnwind = scale(perpendicular, strength * (0.12 + 0.25 * fade) * Math.sin(2.3 * t + 0.6 * x));
      const verticalUnwind = [0, 0, 0.14 * strength * fade * Math.sin(1.6 * t + 0.4 * y)];
      return withSpatialCoupling(add(add(scale(base, directionalFactor), lateralUnwind), verticalUnwind));
    }

    case 11: {
      const progress = smoothstep(0.12 * distanceScale, distanceScale, x);
      const lateralSignal = Math.tanh((y + 0.3 * Math.sin(0.9 * x + 1.1 * t)) / 1.2);
      const amplification = 0.26 + 0.62 * progress + 0.5 * Math.abs(lateralSignal);
      const cascade = scale(perpendicular, strength * lateralSignal * (0.25 + 0.9 * progress));
      const lift = [0, 0, 0.1 * strength * Math.abs(lateralSignal) * Math.sin(1.8 * t + 0.35 * x)];
      const turbulent = scale(spectralNoise(x, y, z, t, windContext.modes), 0.15 * strength * (1 + progress));
      return withSpatialCoupling(add(add(add(scale(base, amplification), cascade), lift), turbulent));
    }

    case 12: {
      let wallWind = scale(base, 0.22);
      for (const wall of windContext.liquidityWalls) {
        const dx = (x - wall.x) / wall.widthX;
        const dy = (y - wall.y) / wall.widthY;
        const dz = (z - wall.z) / wall.widthZ;
        const envelope = Math.exp(-0.5 * (dx * dx + dy * dy + 0.55 * dz * dz));
        const resistance = scale(baseDirection, -wall.polarity * strength * wall.amplitude * envelope);
        const lateralGradient = scale(perpendicular, -0.42 * strength * wall.amplitude * dy * envelope);
        const verticalGradient = [0, 0, -0.22 * strength * wall.amplitude * dz * envelope];
        wallWind = add(wallWind, add(add(resistance, lateralGradient), verticalGradient));
      }
      return withSpatialCoupling(add(wallWind, scale(spectralNoise(x, y, z, t, windContext.modes), 0.08 * strength)));
    }

    case 13: {
      const t0 = windContext.squeezeTime;
      const compression = 1 - smoothstep(t0 - 0.65, t0, t);
      const breakout = smoothstep(t0, t0 + 0.35, t);
      const directionalFactor = 0.08 * compression + (1.65 + 0.22 * Math.sin(6 * (t - t0) + 0.5 * x)) * breakout;
      const turbulenceLevel = 0.06 * compression + 0.35 * breakout;
      const breakoutLift = [0, 0, 0.15 * strength * breakout * Math.sin(2.1 * t + 0.7 * y)];
      return withSpatialCoupling(add(add(scale(base, directionalFactor), scale(spectralNoise(x, y, z, t, windContext.modes), turbulenceLevel * strength)), breakoutLift));
    }

    case 14: {
      const step = clamp(Math.round(t / DT), 0, windContext.hiddenStates.length - 1);
      const hiddenState = windContext.hiddenStates[step] ?? 0;
      const local = spectralNoise(x, y, z, t, windContext.modes);
      if (hiddenState === 0) {
        return withSpatialCoupling(add(scale(base, 0.92), scale(local, 0.1 * strength)));
      }
      if (hiddenState === 1) {
        return withSpatialCoupling(add(add(scale(base, -0.62), scale(perpendicular, 0.28 * strength * Math.sin(1.9 * t + 0.4 * x))), scale(local, 0.12 * strength)));
      }
      if (hiddenState === 2) {
        return withSpatialCoupling(add(scale(base, 0.15), scale(local, 1.15 * strength)));
      }
      return withSpatialCoupling(add(
        add(scale(base, 0.22), [0, -0.74 * strength * Math.tanh((y - HOOP_Y) / 2), 0.2 * strength * Math.tanh((params.hoopHeight - z) / 1.5)]),
        scale(local, 0.16 * strength)
      ));
    }

    case 15: {
      const step = clamp(Math.round(t / DT), 0, windContext.chaoticStates.length - 1);
      const chaotic = windContext.chaoticStates[step] ?? [0, 0, 0];
      const local = spectralNoise(
        x + 0.9 * chaotic[1],
        y - 0.7 * chaotic[0],
        z + 0.55 * chaotic[2],
        t * 1.85,
        windContext.modes
      );
      const vortex = [
        -Math.sin(1.28 * y + 1.7 * t + chaotic[2]) + 0.36 * chaotic[0] * Math.cos(0.58 * z + 2.1 * t),
        Math.sin(1.06 * x - 1.4 * t + chaotic[0]) + 0.36 * chaotic[1] * Math.sin(0.52 * z - 1.6 * t),
        0.58 * Math.sin(0.88 * z + 0.55 * x - 0.42 * y + 1.9 * t + chaotic[2]),
      ];
      const drift = scale(base, 0.18 + 0.22 * Math.sin(2.4 * t + chaotic[0]));
      const chaoticWind = add(add(drift, scale(chaotic, 1.25 * strength)), add(scale(vortex, 0.34 * strength), scale(local, 0.62 * strength)));
      return withSpatialCoupling(vectorLimited(chaoticWind, Math.max(0.2, 3.4 * strength)));
    }

    default:
      return [0, 0, 0];
  }
}

function accelerationAt(position, velocity, t, params, windContext) {
  const wind = windAt(position[0], position[1], position[2], t, params, windContext);
  const relativeVelocity = sub(velocity, wind);
  const relativeSpeed = norm(relativeVelocity);
  const dragAcceleration = scale(
    relativeVelocity,
    -(params.dragCoeff * relativeSpeed) / Math.max(params.ballMass, 1e-6)
  );
  return {
    wind,
    acceleration: [
      dragAcceleration[0],
      dragAcceleration[1],
      -GRAVITY + dragAcceleration[2],
    ],
  };
}

function derivative(stateVector, t, params, windContext) {
  const position = stateVector.slice(0, 3);
  const velocity = stateVector.slice(3, 6);
  const { acceleration } = accelerationAt(position, velocity, t, params, windContext);
  return [
    velocity[0],
    velocity[1],
    velocity[2],
    acceleration[0],
    acceleration[1],
    acceleration[2],
  ];
}

function combineState(stateVector, derivativeVector, scalar) {
  return stateVector.map((value, index) => value + derivativeVector[index] * scalar);
}

function rk4Step(position, velocity, t, dt, params, windContext) {
  const stateVector = [...position, ...velocity];
  const k1 = derivative(stateVector, t, params, windContext);
  const k2 = derivative(combineState(stateVector, k1, dt / 2), t + dt / 2, params, windContext);
  const k3 = derivative(combineState(stateVector, k2, dt / 2), t + dt / 2, params, windContext);
  const k4 = derivative(combineState(stateVector, k3, dt), t + dt, params, windContext);
  const next = stateVector.map(
    (value, index) => value + (dt / 6) * (k1[index] + 2 * k2[index] + 2 * k3[index] + k4[index])
  );
  return {
    position: next.slice(0, 3),
    velocity: next.slice(3, 6),
  };
}

function computeCrossing(prevPosition, position, prevVelocity, velocity, prevTime, time, params) {
  if (!prevPosition || prevPosition[2] < params.hoopHeight || position[2] > params.hoopHeight) {
    return null;
  }

  const dz = prevPosition[2] - position[2];
  if (Math.abs(dz) < 1e-9) {
    return null;
  }

  const alpha = clamp((prevPosition[2] - params.hoopHeight) / dz, 0, 1);
  const crossingPosition = [
    lerp(prevPosition[0], position[0], alpha),
    lerp(prevPosition[1], position[1], alpha),
    params.hoopHeight,
  ];
  const crossingVelocityZ = lerp(prevVelocity[2], velocity[2], alpha);
  const horizontalDistance = Math.hypot(crossingPosition[0] - hoopX(params), crossingPosition[1] - HOOP_Y);

  if (crossingVelocityZ < 0 && horizontalDistance <= HOOP_RADIUS) {
    return {
      time: lerp(prevTime, time, alpha),
      horizontalDistance,
    };
  }
  return null;
}

function makeRow(params, step, t, position, velocity, acceleration, wind, minDistance, prevPosition) {
  const center = hoopCenter(params);
  const distanceBallToHoop = norm(sub(position, center));
  const horizontalDistanceToHoop = Math.hypot(position[0] - center[0], position[1] - center[1]);
  const windNorm = norm(wind);
  const speedNorm = norm(velocity);
  const crossedPlane =
    prevPosition &&
    (prevPosition[2] - params.hoopHeight) * (position[2] - params.hoopHeight) <= 0;

  return {
    shot_id: params.shotId,
    timestep: step,
    time: t,
    label: 0,
    initial_force: params.initialForce,
    vertical_angle: params.verticalAngle,
    horizontal_angle: params.horizontalAngle,
    distance_to_hoop: params.distanceToHoop,
    initial_ball_x: PLAYER_X,
    initial_ball_y: PLAYER_Y,
    initial_ball_z: params.releaseHeight,
    boy_height: params.boyHeight,
    release_height: params.releaseHeight,
    ball_mass: params.ballMass,
    hoop_height: params.hoopHeight,
    hoop_radius: HOOP_RADIUS,
    wind_enabled: params.windRegimeId === 0 ? 0 : 1,
    wind_strength: params.windStrength,
    wind_orientation: params.windOrientation,
    wind_vertical_orientation: params.windVerticalOrientation,
    wind_spatial_coupling: params.windSpatialCoupling,
    wind_regime: params.windRegime,
    wind_regime_id: params.windRegimeId,
    drag_coeff: params.dragCoeff,
    gravity: GRAVITY,
    ball_x: position[0],
    ball_y: position[1],
    ball_z: position[2],
    ball_vx: velocity[0],
    ball_vy: velocity[1],
    ball_vz: velocity[2],
    ball_ax: acceleration[0],
    ball_ay: acceleration[1],
    ball_az: acceleration[2],
    wind_x: wind[0],
    wind_y: wind[1],
    wind_z: wind[2],
    wind_norm: windNorm,
    distance_ball_to_hoop: distanceBallToHoop,
    horizontal_distance_to_hoop: horizontalDistanceToHoop,
    speed_norm: speedNorm,
    is_crossing_hoop_plane: crossedPlane ? 1 : 0,
    is_inside_hoop_radius: horizontalDistanceToHoop <= HOOP_RADIUS ? 1 : 0,
    is_descending: velocity[2] < 0 ? 1 : 0,
    min_hoop_distance_so_far: minDistance,
    collision_time: null,
    final_min_hoop_distance: null,
  };
}

function simulateShot(params) {
  params.releaseHeight = params.boyHeight * 0.85;
  const windContext = createWindContext(params);
  const verticalAngle = degToRad(params.verticalAngle);
  const horizontalAngle = degToRad(params.horizontalAngle);

  // The UI keeps the requested label "Force initiale"; numerically it controls launch speed in m/s.
  const initialSpeed = params.initialForce;
  let position = [PLAYER_X, PLAYER_Y, params.releaseHeight];
  let velocity = [
    initialSpeed * Math.cos(verticalAngle) * Math.cos(horizontalAngle),
    initialSpeed * Math.cos(verticalAngle) * Math.sin(horizontalAngle),
    initialSpeed * Math.sin(verticalAngle),
  ];

  const rows = [];
  let label = 0;
  let collisionTime = null;
  let finalMinDistance = Infinity;
  let prevPosition = null;
  let prevVelocity = null;
  let prevTime = null;

  for (let step = 0; step < MAX_STEPS; step += 1) {
    const time = step * DT;
    const { acceleration, wind } = accelerationAt(position, velocity, time, params, windContext);
    const center = hoopCenter(params);
    const distanceBallToHoop = norm(sub(position, center));
    finalMinDistance = Math.min(finalMinDistance, distanceBallToHoop);

    if (label === 0 && prevPosition) {
      const crossing = computeCrossing(prevPosition, position, prevVelocity, velocity, prevTime, time, params);
      if (crossing) {
        label = 1;
        collisionTime = crossing.time;
      }
    }

    rows.push(makeRow(params, step, time, position, velocity, acceleration, wind, finalMinDistance, prevPosition));

    if (position[2] < -0.2 && velocity[2] < 0 && step > 6) {
      break;
    }

    prevPosition = position.slice();
    prevVelocity = velocity.slice();
    prevTime = time;

    const nextState = rk4Step(position, velocity, time, DT, params, windContext);
    position = nextState.position;
    velocity = nextState.velocity;
  }

  for (const row of rows) {
    row.label = label;
    row.collision_time = collisionTime;
    row.final_min_hoop_distance = finalMinDistance;
  }

  const metadata = {
    label,
    result: label === 1 ? "PANIER" : "ÉCHEC",
    minHoopDistance: finalMinDistance,
    collisionTime,
    windContext,
  };

  return {
    rows,
    metadata,
    windContext,
  };
}

function computeSceneBounds() {
  return { ...FIXED_SCENE_BOUNDS };
}

function buildWindGrid(bounds) {
  const points = [];
  const nx = 18;
  const ny = 9;
  const nz = 6;

  for (let ix = 0; ix < nx; ix += 1) {
    for (let iy = 0; iy < ny; iy += 1) {
      for (let iz = 0; iz < nz; iz += 1) {
        points.push([
          lerp(bounds.xMin + 0.25, bounds.xMax - 0.25, ix / (nx - 1)),
          lerp(bounds.yMin + 0.25, bounds.yMax - 0.25, iy / (ny - 1)),
          lerp(0.42, Math.max(0.65, bounds.zMax - 0.28), iz / (nz - 1)),
        ]);
      }
    }
  }

  return points;
}

function setFrame(index) {
  if (!state) {
    return;
  }
  state.frameIndex = clamp(index, 0, state.rows.length - 1);
  dom.timeSlider.value = String(state.frameIndex);
  updateRangeFill(dom.timeSlider);
  updateTitles();
  drawScene();
  updatePlaybackControls();
  requestWorldModelPrediction();
}

function isAnimationAtEnd() {
  return Boolean(state && state.frameIndex >= state.rows.length - 1);
}

function updatePlaybackControls() {
  if (!state) {
    dom.playButton.dataset.mode = "play";
    dom.playButton.setAttribute("aria-label", "Lire");
    dom.playButton.title = "Lire";
    dom.stepBackButton.disabled = true;
    dom.stepForwardButton.disabled = true;
    return;
  }

  const mode = state.playing ? "pause" : isAnimationAtEnd() ? "replay" : "play";
  const labels = {
    play: "Lire",
    pause: "Pause",
    replay: "Rejouer depuis le début",
  };
  dom.playButton.dataset.mode = mode;
  dom.playButton.setAttribute("aria-label", labels[mode]);
  dom.playButton.title = labels[mode];
  dom.stepBackButton.disabled = state.frameIndex <= 0;
  dom.stepForwardButton.disabled = isAnimationAtEnd();
}

function updateTitles() {
  if (!state) {
    return;
  }
  const row = state.rows[state.frameIndex];
  const result = state.metadata.result;
  dom.resultTitle.textContent = result;
  dom.resultTitle.className = result === "PANIER" ? "result-made" : "result-fail";
  dom.timeTitle.innerHTML = `t = ${row.time.toFixed(2)} s &nbsp; (timestep ${row.timestep} / ${MAX_STEPS})`;
}

function renderResults() {
  const resultClass = state.metadata.result === "PANIER" ? "metric-made" : "metric-fail";
  const collision = state.metadata.collisionTime == null ? "-" : `${state.metadata.collisionTime.toFixed(2)} s`;
  const rows = [
    ["Label", state.metadata.label],
    ["Résultat", `<span class="${resultClass}">${state.metadata.result}</span>`],
    ["Distance au panier", `${state.params.distanceToHoop.toFixed(1)} m`],
    ["Distance minimale au panier", `${state.metadata.minHoopDistance.toFixed(3)} m`],
    ["Collision time", collision],
    ["Régime de vent", state.params.windRegime],
    ["Couplage x,y,z", state.params.windSpatialCoupling ? "activé" : "désactivé"],
    ["Force du vent", `${state.params.windStrength.toFixed(1)} m/s`],
    ["Orientation horizontale", `${state.params.windOrientation.toFixed(1)}°`],
    ["Orientation verticale", `${state.params.windVerticalOrientation.toFixed(1)}°`],
  ];
  dom.resultsBody.innerHTML = rows
    .map(([key, value]) => `<tr><td>${key}</td><td>${value}</td></tr>`)
    .join("");
}

function formatPreviewValue(value, column) {
  if (column === "...") {
    return "...";
  }
  if (value == null || Number.isNaN(value)) {
    return "-";
  }
  if (typeof value === "number") {
    if (INTEGER_COLUMNS.has(column)) {
      return String(Math.round(value));
    }
    if (column === "time") {
      return value.toFixed(2);
    }
    return value.toFixed(3);
  }
  return String(value);
}

function makeShotCsvRow() {
  return {
    shot_id: state.params.shotId,
    initial_force: state.params.initialForce,
    vertical_angle: state.params.verticalAngle,
    horizontal_angle: state.params.horizontalAngle,
    distance_to_hoop: state.params.distanceToHoop,
    boy_height: state.params.boyHeight,
    ball_mass: state.params.ballMass,
    hoop_height: state.params.hoopHeight,
    wind_regime_id: state.params.windRegimeId,
    wind_regime: state.params.windRegime,
    wind_strength: state.params.windStrength,
    wind_orientation: state.params.windOrientation,
    wind_vertical_orientation: state.params.windVerticalOrientation,
    wind_spatial_coupling: state.params.windSpatialCoupling,
    drag_coeff: state.params.dragCoeff,
    gravity: GRAVITY,
  };
}

function makeTrajectoryCsvRow(row) {
  return {
    shot_id: row.shot_id,
    timestep: row.timestep,
    time: row.time,
    ball_x: row.ball_x,
    ball_y: row.ball_y,
    ball_z: row.ball_z,
    ball_vx: row.ball_vx,
    ball_vy: row.ball_vy,
    ball_vz: row.ball_vz,
    ball_ax: row.ball_ax,
    ball_ay: row.ball_ay,
    ball_az: row.ball_az,
    distance_ball_to_hoop: row.distance_ball_to_hoop,
    horizontal_distance_to_hoop: row.horizontal_distance_to_hoop,
    speed_norm: row.speed_norm,
  };
}

function makeWindFieldCsvRow(row, windPoint, windVectorId) {
  const wind = windAt(
    windPoint[0],
    windPoint[1],
    windPoint[2],
    row.time,
    state.params,
    state.windContext
  );

  return {
    shot_id: row.shot_id,
    timestep: row.timestep,
    time: row.time,
    wind_vector_id: windVectorId,
    wind_grid_x: windPoint[0],
    wind_grid_y: windPoint[1],
    wind_grid_z: windPoint[2],
    wind_x: wind[0],
    wind_y: wind[1],
    wind_z: wind[2],
    wind_norm: norm(wind),
  };
}

function makeLabelCsvRow() {
  return {
    shot_id: state.params.shotId,
    label: state.metadata.label,
    result: state.metadata.result,
    distance_to_hoop: state.params.distanceToHoop,
    final_min_hoop_distance: state.metadata.minHoopDistance,
    collision_time: state.metadata.collisionTime,
  };
}

function renderCsvPreview() {
  const head = `<thead><tr>${PREVIEW_COLUMNS.map((column) => `<th>${column}</th>`).join("")}</tr></thead>`;
  const previewRows = state.rows.slice(0, 3).map((row) => ({
    ...makeTrajectoryCsvRow(row),
    "...": "...",
  }));
  previewRows.push(Object.fromEntries(PREVIEW_COLUMNS.map((column) => [column, "..."])));
  const body = previewRows
    .map((row) => {
      const cells = PREVIEW_COLUMNS.map((column) => `<td>${formatPreviewValue(row[column], column)}</td>`).join("");
      return `<tr>${cells}</tr>`;
    })
    .join("");
  dom.csvPreview.innerHTML = `${head}<tbody>${body}</tbody>`;
}

function renderAllPanels() {
  renderResults();
  renderCsvPreview();
  updateTitles();
  updatePlaybackControls();
}

function windColorScaleMax(params) {
  const regimeMultipliers = {
    5: 2.8,
    8: 3.2,
    9: 3.4,
    10: 2.7,
    11: 2.9,
    12: 3.0,
    13: 2.8,
    14: 2.7,
    15: 3.8,
  };
  const couplingBoost = params.windSpatialCoupling ? 0.45 : 0;
  return Math.max(1, Math.ceil(params.windStrength * ((regimeMultipliers[params.windRegimeId] || 1.5) + couplingBoost)));
}

function stopAnimation() {
  state && (state.playing = false);
  if (animationFrame) {
    cancelAnimationFrame(animationFrame);
    animationFrame = null;
  }
  updatePlaybackControls();
}

function playAnimation() {
  if (!state || state.playing) {
    return;
  }
  state.playing = true;
  previousAnimationTime = 0;
  updatePlaybackControls();

  function tick(timestamp) {
    if (!state || !state.playing) {
      return;
    }
    if (previousAnimationTime === 0 || timestamp - previousAnimationTime >= 34) {
      previousAnimationTime = timestamp;
      const nextIndex = state.frameIndex + 1;
      if (nextIndex >= state.rows.length) {
        state.playing = false;
        animationFrame = null;
        updatePlaybackControls();
        return;
      }
      setFrame(nextIndex);
    }
    animationFrame = requestAnimationFrame(tick);
  }

  animationFrame = requestAnimationFrame(tick);
}

function togglePlayback() {
  if (!state) {
    return;
  }
  if (state.playing) {
    stopAnimation();
    return;
  }
  if (isAnimationAtEnd()) {
    setFrame(0);
  }
  playAnimation();
}

function stepFrame(delta) {
  if (!state) {
    return;
  }
  stopAnimation();
  setFrame(state.frameIndex + delta);
}

function runSimulation(isInitial = false) {
  stopAnimation();
  if (!isInitial) {
    shotCounter += 1;
  }
  const params = readParams();
  const result = simulateShot(params);
  const bounds = computeSceneBounds();
  const camera = state?.camera ? { ...state.camera } : makeDefaultCamera();

  state = {
    params,
    rows: result.rows,
    metadata: result.metadata,
    windContext: result.windContext,
    bounds,
    windGrid: buildWindGrid(bounds),
    camera,
    frameIndex: 0,
    playing: false,
    colorMax: windColorScaleMax(params),
    worldModelPrediction: [],
    worldModelMomentum: null,
  };

  dom.timeSlider.max = String(Math.max(0, state.rows.length - 1));
  dom.timeSlider.value = "0";
  updateRangeFill(dom.timeSlider);
  renderAllPanels();
  drawScene();
  requestWorldModelPrediction(true);
}

function buildAiCamerasUrl() {
  const params = state?.params ?? readParams();
  const query = new URLSearchParams({
    initial_force: params.initialForce,
    vertical_angle: params.verticalAngle,
    horizontal_angle: params.horizontalAngle,
    distance_to_hoop: params.distanceToHoop,
    boy_height: params.boyHeight,
    ball_mass: params.ballMass,
    hoop_height: params.hoopHeight,
    wind_regime_id: params.windRegimeId,
    wind_strength: params.windStrength,
    wind_orientation: params.windOrientation,
    wind_vertical_orientation: params.windVerticalOrientation,
    wind_spatial_coupling: params.windSpatialCoupling,
    drag_coeff: params.dragCoeff,
    show_wind_field: 1,
    field_view: windVisualizationMode === "volume" ? 2 : windVisualizationMode === "field" ? 1 : 0,
    show_past_trail: 1,
    autoplay: 1,
    loop: 1,
    return_to: "index.html",
  });
  return `ai-cameras.html?${query.toString()}`;
}

function openAiCameras(event) {
  event.preventDefault();
  window.location.href = buildAiCamerasUrl();
}

function prepareCanvas() {
  const rect = dom.canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, rect.width);
  const height = Math.max(1, rect.height);
  const targetWidth = Math.round(width * dpr);
  const targetHeight = Math.round(height * dpr);
  if (dom.canvas.width !== targetWidth || dom.canvas.height !== targetHeight) {
    dom.canvas.width = targetWidth;
    dom.canvas.height = targetHeight;
  }
  const ctx = dom.canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return { ctx, width, height };
}

function createProjector(width, height, bounds, camera = DEFAULT_CAMERA) {
  const center = [
    (bounds.xMin + bounds.xMax) / 2,
    (bounds.yMin + bounds.yMax) / 2,
    (bounds.zMin + bounds.zMax) / 2,
  ];
  const xSpan = bounds.xMax - bounds.xMin;
  const ySpan = bounds.yMax - bounds.yMin;
  const zSpan = bounds.zMax - bounds.zMin;
  const radius = Math.max(xSpan, ySpan, zSpan) * 2.15;
  const pitch = clamp(camera.pitch, -CAMERA_PITCH_LIMIT, CAMERA_PITCH_LIMIT);
  const horizontalRadius = Math.cos(pitch) * radius;
  const eye = [
    center[0] + horizontalRadius * Math.cos(camera.yaw),
    center[1] + horizontalRadius * Math.sin(camera.yaw),
    center[2] + radius * Math.sin(pitch),
  ];
  const forward = normalize(sub(center, eye));
  let right = cross(forward, [0, 0, 1]);
  if (norm(right) < 1e-6) {
    right = [-Math.sin(camera.yaw), Math.cos(camera.yaw), 0];
  } else {
    right = normalize(right);
  }
  const up = normalize(cross(right, forward));

  const corners = [
    [bounds.xMin, bounds.yMin, bounds.zMin],
    [bounds.xMin, bounds.yMin, bounds.zMax],
    [bounds.xMin, bounds.yMax, bounds.zMin],
    [bounds.xMin, bounds.yMax, bounds.zMax],
    [bounds.xMax, bounds.yMin, bounds.zMin],
    [bounds.xMax, bounds.yMin, bounds.zMax],
    [bounds.xMax, bounds.yMax, bounds.zMin],
    [bounds.xMax, bounds.yMax, bounds.zMax],
  ];

  const views = corners.map((point) => {
    const rel = sub(point, eye);
    return {
      x: dot(rel, right),
      y: dot(rel, up),
      z: dot(rel, forward),
    };
  });

  const minX = Math.min(...views.map((view) => view.x));
  const maxX = Math.max(...views.map((view) => view.x));
  const minY = Math.min(...views.map((view) => view.y));
  const maxY = Math.max(...views.map((view) => view.y));
  const margin = {
    left: 82,
    right: 118,
    top: 32,
    bottom: 52,
  };
  const availableWidth = width - margin.left - margin.right;
  const availableHeight = height - margin.top - margin.bottom;
  const zoom = clamp(camera.zoom ?? DEFAULT_CAMERA.zoom, CAMERA_MIN_ZOOM, CAMERA_MAX_ZOOM);
  const baseScaleX = availableWidth / Math.max(maxX - minX, 1e-6);
  const uniformScale = Math.min(
    baseScaleX,
    availableHeight / Math.max(maxY - minY, 1e-6)
  );
  const baseScaleY = Math.min(
    availableHeight / Math.max(maxY - minY, 1e-6),
    uniformScale * 1.72
  );
  const scaleX = baseScaleX * zoom;
  const scaleY = baseScaleY * zoom;
  const offsetX = margin.left + (availableWidth - (maxX - minX) * scaleX) / 2;
  const offsetY = margin.top + (availableHeight - (maxY - minY) * scaleY) / 2;

  function project(point) {
    const rel = sub(point, eye);
    const viewX = dot(rel, right);
    const viewY = dot(rel, up);
    const viewZ = dot(rel, forward);
    return {
      x: offsetX + (viewX - minX) * scaleX,
      y: offsetY + (maxY - viewY) * scaleY,
      depth: viewZ,
    };
  }

  return {
    project,
    right,
    up,
    forward,
  };
}

function jetColor(t) {
  const value = clamp(t, 0, 1);
  const r = clamp(1.5 - Math.abs(4 * value - 3), 0, 1);
  const g = clamp(1.5 - Math.abs(4 * value - 2), 0, 1);
  const b = clamp(1.5 - Math.abs(4 * value - 1), 0, 1);
  return `rgb(${Math.round(255 * r)}, ${Math.round(255 * g)}, ${Math.round(255 * b)})`;
}

function rgbString(rgb) {
  return `rgb(${Math.round(rgb[0])}, ${Math.round(rgb[1])}, ${Math.round(rgb[2])})`;
}

function interpolateColor(stops, t) {
  const value = clamp(t, 0, 1);
  const scaled = value * (stops.length - 1);
  const index = Math.min(stops.length - 2, Math.floor(scaled));
  const localT = scaled - index;
  return rgbString([
    lerp(stops[index][0], stops[index + 1][0], localT),
    lerp(stops[index][1], stops[index + 1][1], localT),
    lerp(stops[index][2], stops[index + 1][2], localT),
  ]);
}

function parseCssColor(color, fallback) {
  const text = color.trim();
  const hex = text.match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i);
  if (hex) {
    const raw = hex[1];
    const full = raw.length === 3
      ? raw.split("").map((char) => char + char).join("")
      : raw;
    return [
      Number.parseInt(full.slice(0, 2), 16),
      Number.parseInt(full.slice(2, 4), 16),
      Number.parseInt(full.slice(4, 6), 16),
    ];
  }

  const rgb = text.match(/^rgba?\(([^)]+)\)$/i);
  if (rgb) {
    const channels = rgb[1]
      .split(",")
      .slice(0, 3)
      .map((part) => Number.parseFloat(part.trim()));
    if (channels.every((channel) => Number.isFinite(channel))) {
      return channels;
    }
  }

  return fallback;
}

function rgbaFromCssColor(color, alpha, fallback = [0, 0, 0]) {
  const [r, g, b] = parseCssColor(color, fallback);
  return `rgba(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)}, ${alpha})`;
}

function cssColorStops(names, fallbacks) {
  return names.map((name, index) => parseCssColor(cssVar(name, fallbacks[index]), parseCssColor(fallbacks[index], [0, 0, 0])));
}

function themeWindColor(t) {
  if (!isDarkTheme()) {
    return jetColor(t);
  }

  return interpolateColor(
    cssColorStops(
      ["--wind-low", "--wind-mid", "--wind-high", "--wind-warm", "--wind-hot", "--wind-max"],
      ["#0d0887", "#7e03a8", "#cc4778", "#ff2a00", "#f89540", "#f0f921"]
    ),
    t
  );
}

function glowFromCss(name) {
  const value = cssVar(name, "");
  const colorMatch = value.match(/rgba?\([^)]+\)|#[0-9a-f]{3,6}/i);
  const blurMatch = value.match(/0\s+0\s+([0-9.]+)px/i);
  if (!colorMatch) {
    return null;
  }
  return {
    color: colorMatch[0],
    blur: blurMatch ? Number.parseFloat(blurMatch[1]) : 10,
  };
}

function withGlow(ctx, glow, draw) {
  ctx.save();
  if (glow) {
    ctx.shadowColor = glow.color;
    ctx.shadowBlur = glow.blur;
  }
  draw();
  ctx.restore();
}

function canvasPalette() {
  return {
    background: cssVarAny(["--canvas-bg", "--scene-bg", "--panel"], "#ffffff"),
    text: cssVarAny(["--canvas-text", "--text", "--ink"], "#111827"),
    label: cssVarAny(["--canvas-label", "--canvas-text", "--text"], "#273142"),
    grid: cssVar("--canvas-grid", "#d8dee8"),
    axis: cssVar("--canvas-axis", "#8b96a8"),
    box: cssVar("--canvas-box", "#c6ceda"),
    backboard: cssVar("--canvas-backboard", "#aab4c2"),
    pole: cssVar("--canvas-pole", "#2f3338"),
    hoop: cssVar("--canvas-hoop", "#d3111c"),
    hoopGlow: glowFromCss("--canvas-hoop-glow"),
    net: cssVar("--canvas-net", "#b8c1cc"),
    netSoft: cssVar("--canvas-net-soft", "#c6ccd5"),
    playerShoe: cssVar("--canvas-player-shoe", "#0f172a"),
    playerShort: cssVar("--canvas-player-short", "#0f5fbd"),
    playerShirt: cssVar("--canvas-player-shirt", "#0f72df"),
    playerLimb: cssVarAny(["--canvas-player-limb", "--orange"], "#9a4b21"),
    playerSkin: cssVar("--canvas-player-skin", "#a95528"),
    playerSkinStroke: cssVar("--canvas-player-skin-stroke", "#6b2e18"),
    trajectory: cssVar("--canvas-trajectory", "#df7a12"),
    trajectoryDot: cssVar("--canvas-trajectory-dot", "#e87511"),
    trajectoryStroke: cssVar("--canvas-trajectory-stroke", "#ab520e"),
    trajectoryGlow: glowFromCss("--trajectory-glow"),
    worldPrediction: cssVar("--canvas-world-prediction", "#2563eb"),
    worldPredictionDot: cssVar("--canvas-world-prediction-dot", "#60a5fa"),
    worldPredictionStroke: cssVar("--canvas-world-prediction-stroke", "#1d4ed8"),
    worldPredictionGlow: glowFromCss("--world-model-glow") || { color: "rgba(37, 99, 235, 0.45)", blur: 12 },
    ball: cssVar("--canvas-ball", "#f28a1d"),
    ballStroke: cssVar("--canvas-ball-stroke", "#a94908"),
    ballGlow: glowFromCss("--canvas-ball-glow"),
    drag: cssVar("--canvas-drag", "#5D71FC"),
    colorbarBorder: cssVar("--colorbar-border", "#b6bfcb"),
    colorbarTick: cssVar("--colorbar-tick", "#5d6675"),
  };
}

function drawGlowingPolyline3D(ctx, projector, points, strokeStyle, lineWidth, alpha, glow) {
  withGlow(ctx, glow, () => {
    drawPolyline3D(ctx, projector, points, strokeStyle, lineWidth, alpha);
  });
}

function drawGlowingCircleAt(ctx, projector, point, radius, fillStyle, strokeStyle, lineWidth, glow) {
  withGlow(ctx, glow, () => {
    drawCircleAt(ctx, projector, point, radius, fillStyle, strokeStyle, lineWidth);
  });
}

function windColor(t) {
  return themeWindColor(t);
}

function drawLine3D(ctx, projector, a, b, strokeStyle, lineWidth = 1, alpha = 1) {
  const pa = projector.project(a);
  const pb = projector.project(b);
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = strokeStyle;
  ctx.lineWidth = lineWidth;
  ctx.beginPath();
  ctx.moveTo(pa.x, pa.y);
  ctx.lineTo(pb.x, pb.y);
  ctx.stroke();
  ctx.restore();
}

function drawPolyline3D(ctx, projector, points, strokeStyle, lineWidth = 1, alpha = 1) {
  if (points.length < 2) {
    return;
  }
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = strokeStyle;
  ctx.lineWidth = lineWidth;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.beginPath();
  const first = projector.project(points[0]);
  ctx.moveTo(first.x, first.y);
  for (let index = 1; index < points.length; index += 1) {
    const projected = projector.project(points[index]);
    ctx.lineTo(projected.x, projected.y);
  }
  ctx.stroke();
  ctx.restore();
}

function niceTicks(min, max, count) {
  if (max <= min) {
    return [min];
  }
  const rawStep = (max - min) / Math.max(1, count - 1);
  const power = 10 ** Math.floor(Math.log10(rawStep));
  const fraction = rawStep / power;
  let niceFraction = 1;
  if (fraction > 5) {
    niceFraction = 10;
  } else if (fraction > 2) {
    niceFraction = 5;
  } else if (fraction > 1) {
    niceFraction = 2;
  }
  const step = niceFraction * power;
  const first = Math.ceil(min / step) * step;
  const ticks = [];
  for (let value = first; value <= max + step * 1e-6; value += step) {
    ticks.push(Number(value.toFixed(8)));
  }
  return ticks;
}

function drawText(ctx, text, x, y, options = {}) {
  ctx.save();
  ctx.fillStyle = options.color || cssVarAny(["--canvas-text", "--text", "--ink"], "#111827");
  ctx.font = options.font || "13px Inter, system-ui, sans-serif";
  ctx.textAlign = options.align || "center";
  ctx.textBaseline = options.baseline || "middle";
  ctx.fillText(text, x, y);
  ctx.restore();
}

function drawGrid(ctx, projector, bounds) {
  const colors = canvasPalette();
  const gridColor = colors.grid;
  const axisColor = colors.axis;
  const boxColor = colors.box;
  const xTicks = niceTicks(bounds.xMin, bounds.xMax, 7);
  const yTicks = niceTicks(bounds.yMin, bounds.yMax, 7);
  const zTicks = niceTicks(bounds.zMin, bounds.zMax, 6);

  for (const x of xTicks) {
    drawLine3D(ctx, projector, [x, bounds.yMin, 0], [x, bounds.yMax, 0], gridColor, 1, 0.75);
    const tick = projector.project([x, bounds.yMin, 0]);
    drawText(ctx, String(Math.round(x)), tick.x, tick.y + 16, { color: colors.label, font: "12px Inter, system-ui, sans-serif" });
  }

  for (const y of yTicks) {
    drawLine3D(ctx, projector, [bounds.xMin, y, 0], [bounds.xMax, y, 0], gridColor, 1, 0.75);
    const tick = projector.project([bounds.xMin, y, 0]);
    drawText(ctx, String(Math.round(y)), tick.x - 14, tick.y + 4, { color: colors.label, font: "12px Inter, system-ui, sans-serif" });
  }

  const edges = [
    [[bounds.xMin, bounds.yMin, 0], [bounds.xMax, bounds.yMin, 0]],
    [[bounds.xMax, bounds.yMin, 0], [bounds.xMax, bounds.yMax, 0]],
    [[bounds.xMax, bounds.yMax, 0], [bounds.xMin, bounds.yMax, 0]],
    [[bounds.xMin, bounds.yMax, 0], [bounds.xMin, bounds.yMin, 0]],
    [[bounds.xMin, bounds.yMin, 0], [bounds.xMin, bounds.yMin, bounds.zMax]],
    [[bounds.xMax, bounds.yMin, 0], [bounds.xMax, bounds.yMin, bounds.zMax]],
    [[bounds.xMax, bounds.yMax, 0], [bounds.xMax, bounds.yMax, bounds.zMax]],
    [[bounds.xMin, bounds.yMax, 0], [bounds.xMin, bounds.yMax, bounds.zMax]],
    [[bounds.xMin, bounds.yMin, bounds.zMax], [bounds.xMax, bounds.yMin, bounds.zMax]],
    [[bounds.xMax, bounds.yMin, bounds.zMax], [bounds.xMax, bounds.yMax, bounds.zMax]],
    [[bounds.xMax, bounds.yMax, bounds.zMax], [bounds.xMin, bounds.yMax, bounds.zMax]],
    [[bounds.xMin, bounds.yMax, bounds.zMax], [bounds.xMin, bounds.yMin, bounds.zMax]],
  ];
  for (const [a, b] of edges) {
    drawLine3D(ctx, projector, a, b, boxColor, 1, 0.85);
  }

  for (const z of zTicks) {
    const tickStart = projector.project([bounds.xMin, bounds.yMin, z]);
    const tickEnd = projector.project([bounds.xMin - 0.08, bounds.yMin, z]);
    ctx.strokeStyle = axisColor;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(tickStart.x, tickStart.y);
    ctx.lineTo(tickEnd.x, tickEnd.y);
    ctx.stroke();
    drawText(ctx, String(Number(z.toFixed(1))), tickEnd.x - 16, tickEnd.y, { align: "right", color: colors.label, font: "12px Inter, system-ui, sans-serif" });
  }

  drawLine3D(ctx, projector, [bounds.xMin, bounds.yMin, 0], [bounds.xMax, bounds.yMin, 0], axisColor, 1.3, 0.95);
  drawLine3D(ctx, projector, [bounds.xMin, bounds.yMin, 0], [bounds.xMin, bounds.yMax, 0], axisColor, 1.3, 0.95);
  drawLine3D(ctx, projector, [bounds.xMin, bounds.yMin, 0], [bounds.xMin, bounds.yMin, bounds.zMax], axisColor, 1.3, 0.95);

  const xLabel = projector.project([bounds.xMax, bounds.yMin, 0]);
  const yLabel = projector.project([bounds.xMin, bounds.yMax, 0]);
  const zLabel = projector.project([bounds.xMin, bounds.yMin, bounds.zMax * 0.52]);
  drawText(ctx, "x (m)", xLabel.x + 22, xLabel.y + 10, { font: "15px Inter, system-ui, sans-serif" });
  drawText(ctx, "y (m)", yLabel.x - 14, yLabel.y + 22, { font: "15px Inter, system-ui, sans-serif" });
  drawText(ctx, "z (m)", zLabel.x - 42, zLabel.y, { font: "15px Inter, system-ui, sans-serif" });
}

function drawArrow(ctx, tail, head, color, options = {}) {
  const dx = head.x - tail.x;
  const dy = head.y - tail.y;
  const length = Math.hypot(dx, dy);
  if (length < 1.5) {
    return;
  }
  const ux = dx / length;
  const uy = dy / length;
  const headSize = clamp(length * 0.34, options.minHeadSize ?? 2.8, options.maxHeadSize ?? 5.4);

  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = options.lineWidth ?? 1.05;
  ctx.globalAlpha = options.alpha ?? 0.82;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(tail.x, tail.y);
  ctx.lineTo(head.x, head.y);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(head.x, head.y);
  ctx.lineTo(head.x - ux * headSize - uy * headSize * 0.48, head.y - uy * headSize + ux * headSize * 0.48);
  ctx.lineTo(head.x - ux * headSize + uy * headSize * 0.48, head.y - uy * headSize - ux * headSize * 0.48);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawWindField(ctx, projector, row) {
  const arrows = [];
  for (const point of state.windGrid) {
    const wind = windAt(point[0], point[1], point[2], row.time, state.params, state.windContext);
    const windNorm = norm(wind);
    const direction = normalize(wind);
    const length = 0.12 + 0.1 * clamp(windNorm / Math.max(state.colorMax, 1e-6), 0, 1.4);
    const headPoint = add(point, scale(direction, length));
    const projectedTail = projector.project(point);
    const projectedHead = projector.project(headPoint);
    arrows.push({
      tail: projectedTail,
      head: projectedHead,
      color: windColor(windNorm / state.colorMax),
      depth: projectedTail.depth,
    });
  }
  arrows.sort((a, b) => a.depth - b.depth);
  for (const arrow of arrows) {
    drawArrow(ctx, arrow.tail, arrow.head, arrow.color);
  }
}

function drawDragVector(ctx, projector, row) {
  if (!dom.showDragVector.checked) {
    return;
  }

  const colors = canvasPalette();
  const dragAcceleration = [row.ball_ax, row.ball_ay, row.ball_az + GRAVITY];
  const dragNorm = norm(dragAcceleration);
  if (dragNorm < 1e-5) {
    return;
  }

  const ballPosition = [row.ball_x, row.ball_y, Math.max(row.ball_z, 0)];
  const dragDirection = normalize(dragAcceleration);
  const dragLength = clamp(0.18 + 0.09 * dragNorm, 0.18, 0.72);
  const tailPoint = add(ballPosition, scale(dragDirection, 0.1));
  const headPoint = add(ballPosition, scale(dragDirection, dragLength));
  const tail = projector.project(tailPoint);
  const head = projector.project(headPoint);

  drawArrow(ctx, tail, head, colors.drag, {
    alpha: 0.94,
    lineWidth: 2.2,
    minHeadSize: 5.2,
    maxHeadSize: 9.2,
  });
}

function makeWindFieldPlane(bounds, params, camera) {
  const useHorizontalSlice = Math.abs(camera.pitch) > 1.08;
  if (useHorizontalSlice) {
    return {
      type: "xy",
      fixed: clamp(params.releaseHeight, 0.45, Math.max(0.45, bounds.zMax - 0.18)),
      uMin: bounds.xMin + 0.22,
      uMax: bounds.xMax - 0.22,
      vMin: bounds.yMin + 0.22,
      vMax: bounds.yMax - 0.22,
    };
  }

  const isLookingAlongX = Math.abs(Math.cos(camera.yaw)) >= Math.abs(Math.sin(camera.yaw));
  if (isLookingAlongX) {
    return {
      type: "yz",
      fixed: clamp(params.distanceToHoop * 0.5, bounds.xMin + 0.28, bounds.xMax - 0.28),
      uMin: bounds.yMin + 0.22,
      uMax: bounds.yMax - 0.22,
      vMin: 0.08,
      vMax: bounds.zMax - 0.18,
    };
  }

  return {
    type: "xz",
    fixed: 0,
    uMin: bounds.xMin + 0.22,
    uMax: bounds.xMax - 0.22,
    vMin: 0.08,
    vMax: bounds.zMax - 0.18,
  };
}

function pointOnWindFieldPlane(plane, u, v) {
  if (plane.type === "xy") {
    return [u, v, plane.fixed];
  }
  if (plane.type === "yz") {
    return [plane.fixed, u, v];
  }
  return [u, plane.fixed, v];
}

function windOnFieldPlane(plane, u, v, time) {
  const point = pointOnWindFieldPlane(plane, u, v);
  const wind = windAt(point[0], point[1], point[2], time, state.params, state.windContext);
  let planeVector = [wind[0], wind[2]];
  if (plane.type === "xy") {
    planeVector = [wind[0], wind[1]];
  } else if (plane.type === "yz") {
    planeVector = [wind[1], wind[2]];
  }
  return {
    wind,
    planeVector,
    norm: norm(wind),
  };
}

function drawProjectedQuad(ctx, projector, corners, fillStyle, alpha) {
  const projected = corners.map((point) => projector.project(point));
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.fillStyle = fillStyle;
  ctx.beginPath();
  ctx.moveTo(projected[0].x, projected[0].y);
  for (let index = 1; index < projected.length; index += 1) {
    ctx.lineTo(projected[index].x, projected[index].y);
  }
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawContourSegment(ctx, projector, plane, a, b, color) {
  const pa = projector.project(pointOnWindFieldPlane(plane, a.u, a.v));
  const pb = projector.project(pointOnWindFieldPlane(plane, b.u, b.v));
  ctx.beginPath();
  ctx.moveTo(pa.x, pa.y);
  ctx.lineTo(pb.x, pb.y);
  ctx.strokeStyle = color;
  ctx.stroke();
}

function interpolateContourPoint(a, b, level) {
  const span = b.value - a.value;
  const t = Math.abs(span) < 1e-9 ? 0.5 : clamp((level - a.value) / span, 0, 1);
  return {
    u: lerp(a.u, b.u, t),
    v: lerp(a.v, b.v, t),
  };
}

function drawWindContours(ctx, projector, plane, samples, cols, rows) {
  const lineColor = rgbaFromCssColor(cssVarAny(["--canvas-text", "--text", "--ink"], "#ffffff"), isDarkTheme() ? 0.42 : 0.36);
  const levels = [0.18, 0.3, 0.42, 0.54, 0.66, 0.78, 0.9];

  ctx.save();
  ctx.lineWidth = isDarkTheme() ? 0.72 : 0.62;
  ctx.lineCap = "round";
  for (const level of levels) {
    for (let j = 0; j < rows; j += 1) {
      for (let i = 0; i < cols; i += 1) {
        const c0 = samples[j][i];
        const c1 = samples[j][i + 1];
        const c2 = samples[j + 1][i + 1];
        const c3 = samples[j + 1][i];
        const corners = [c0, c1, c2, c3];
        const edges = [[c0, c1], [c1, c2], [c2, c3], [c3, c0]];
        const intersections = [];

        for (const [a, b] of edges) {
          if ((a.value - level) * (b.value - level) < 0) {
            intersections.push(interpolateContourPoint(a, b, level));
          }
        }

        if (intersections.length === 2) {
          drawContourSegment(ctx, projector, plane, intersections[0], intersections[1], lineColor);
        } else if (intersections.length === 4) {
          const average = corners.reduce((sum, point) => sum + point.value, 0) / corners.length;
          const order = average > level ? [0, 3, 1, 2] : [0, 1, 2, 3];
          drawContourSegment(ctx, projector, plane, intersections[order[0]], intersections[order[1]], lineColor);
          drawContourSegment(ctx, projector, plane, intersections[order[2]], intersections[order[3]], lineColor);
        }
      }
    }
  }
  ctx.restore();
}

function traceFieldLine(plane, seedU, seedV, time, direction, stepLength, maxSteps) {
  let u = seedU;
  let v = seedV;
  const points = [];

  for (let step = 0; step < maxSteps; step += 1) {
    if (u < plane.uMin || u > plane.uMax || v < plane.vMin || v > plane.vMax) {
      break;
    }

    points.push(pointOnWindFieldPlane(plane, u, v));
    const field = windOnFieldPlane(plane, u, v, time);
    const vectorNorm = Math.hypot(field.planeVector[0], field.planeVector[1]);
    if (vectorNorm < 1e-6) {
      break;
    }

    u += direction * stepLength * field.planeVector[0] / vectorNorm;
    v += direction * stepLength * field.planeVector[1] / vectorNorm;
  }

  return points;
}

function drawWindStreamlines(ctx, projector, plane, time) {
  const uSeeds = 10;
  const vSeeds = 6;
  const uSpan = plane.uMax - plane.uMin;
  const vSpan = plane.vMax - plane.vMin;
  const stepLength = Math.min(uSpan, vSpan) / 42;
  const maxSteps = 36;
  const lineColor = cssVarAny(["--canvas-text", "--text", "--ink"], "#ffffff");

  ctx.save();
  ctx.lineWidth = isDarkTheme() ? 1.05 : 0.9;
  for (let i = 1; i < uSeeds; i += 1) {
    for (let j = 1; j < vSeeds; j += 1) {
      const seedU = lerp(plane.uMin, plane.uMax, i / uSeeds);
      const seedV = lerp(plane.vMin, plane.vMax, j / vSeeds);
      const backward = traceFieldLine(plane, seedU, seedV, time, -1, stepLength, maxSteps).reverse();
      const forward = traceFieldLine(plane, seedU, seedV, time, 1, stepLength, maxSteps);
      const points = backward.concat(forward.slice(1));
      if (points.length > 4) {
        drawPolyline3D(ctx, projector, points, lineColor, ctx.lineWidth, isDarkTheme() ? 0.48 : 0.55);
      }
    }
  }
  ctx.restore();
}

function drawWindFieldMap(ctx, projector, row) {
  const plane = makeWindFieldPlane(state.bounds, state.params, state.camera);
  const cols = 38;
  const rows = 24;
  const samples = [];

  for (let j = 0; j <= rows; j += 1) {
    const v = lerp(plane.vMin, plane.vMax, j / rows);
    const sampleRow = [];
    for (let i = 0; i <= cols; i += 1) {
      const u = lerp(plane.uMin, plane.uMax, i / cols);
      const wind = windOnFieldPlane(plane, u, v, row.time);
      sampleRow.push({
        u,
        v,
        value: clamp(wind.norm / Math.max(state.colorMax, 1e-6), 0, 1),
      });
    }
    samples.push(sampleRow);
  }

  const fieldAlpha = isDarkTheme() ? 0.66 : 0.46;
  for (let j = 0; j < rows; j += 1) {
    for (let i = 0; i < cols; i += 1) {
      const c0 = samples[j][i];
      const c1 = samples[j][i + 1];
      const c2 = samples[j + 1][i + 1];
      const c3 = samples[j + 1][i];
      const value = (c0.value + c1.value + c2.value + c3.value) / 4;
      drawProjectedQuad(
        ctx,
        projector,
        [
          pointOnWindFieldPlane(plane, c0.u, c0.v),
          pointOnWindFieldPlane(plane, c1.u, c1.v),
          pointOnWindFieldPlane(plane, c2.u, c2.v),
          pointOnWindFieldPlane(plane, c3.u, c3.v),
        ],
        windColor(value),
        fieldAlpha
      );
    }
  }

  drawWindContours(ctx, projector, plane, samples, cols, rows);
  drawWindStreamlines(ctx, projector, plane, row.time);
}

function pointInsideBounds(point, bounds, padding = 0) {
  return (
    point[0] >= bounds.xMin + padding &&
    point[0] <= bounds.xMax - padding &&
    point[1] >= bounds.yMin + padding &&
    point[1] <= bounds.yMax - padding &&
    point[2] >= bounds.zMin + padding &&
    point[2] <= bounds.zMax - padding
  );
}

function drawWindVolumeSlices(ctx, projector, row) {
  const bounds = state.bounds;
  const cols = 14;
  const rows = 8;
  const levels = 5;
  const cells = [];

  for (let iz = 0; iz < levels; iz += 1) {
    const z = lerp(0.52, bounds.zMax - 0.34, iz / Math.max(1, levels - 1));
    for (let iy = 0; iy < rows; iy += 1) {
      const y0 = lerp(bounds.yMin + 0.28, bounds.yMax - 0.28, iy / rows);
      const y1 = lerp(bounds.yMin + 0.28, bounds.yMax - 0.28, (iy + 1) / rows);
      for (let ix = 0; ix < cols; ix += 1) {
        const x0 = lerp(bounds.xMin + 0.28, bounds.xMax - 0.28, ix / cols);
        const x1 = lerp(bounds.xMin + 0.28, bounds.xMax - 0.28, (ix + 1) / cols);
        const center = [(x0 + x1) / 2, (y0 + y1) / 2, z];
        const wind = windAt(center[0], center[1], center[2], row.time, state.params, state.windContext);
        const value = clamp(norm(wind) / Math.max(state.colorMax, 1e-6), 0, 1);
        cells.push({
          corners: [
            [x0, y0, z],
            [x1, y0, z],
            [x1, y1, z],
            [x0, y1, z],
          ],
          color: windColor(value),
          alpha: (isDarkTheme() ? 0.13 : 0.1) + value * (isDarkTheme() ? 0.13 : 0.11),
          depth: projector.project(center).depth,
        });
      }
    }
  }

  cells.sort((a, b) => a.depth - b.depth);
  for (const cell of cells) {
    drawProjectedQuad(ctx, projector, cell.corners, cell.color, cell.alpha);
  }
}

function drawWindVolumeGlyphs(ctx, projector, row) {
  const glyphs = [];
  for (const point of state.windGrid) {
    const wind = windAt(point[0], point[1], point[2], row.time, state.params, state.windContext);
    const windNorm = norm(wind);
    const value = clamp(windNorm / Math.max(state.colorMax, 1e-6), 0, 1);
    if (value < 0.012) {
      continue;
    }

    const direction = normalize(wind);
    const headPoint = add(point, scale(direction, 0.08 + 0.12 * value));
    glyphs.push({
      center: projector.project(point),
      head: projector.project(headPoint),
      color: windColor(value),
      radius: 1.25 + 3.4 * value,
      lineWidth: 0.65 + 1.25 * value,
      alpha: 0.34 + 0.42 * value,
      depth: projector.project(point).depth,
    });
  }

  glyphs.sort((a, b) => a.depth - b.depth);
  ctx.save();
  ctx.lineCap = "round";
  for (const glyph of glyphs) {
    ctx.globalAlpha = glyph.alpha;
    ctx.strokeStyle = glyph.color;
    ctx.fillStyle = glyph.color;
    ctx.lineWidth = glyph.lineWidth;
    ctx.beginPath();
    ctx.moveTo(glyph.center.x, glyph.center.y);
    ctx.lineTo(glyph.head.x, glyph.head.y);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(glyph.center.x, glyph.center.y, glyph.radius, 0, TWO_PI);
    ctx.fill();
  }
  ctx.restore();
}

function traceWindFieldLine3D(seed, time, direction, stepLength, maxSteps) {
  let point = seed.slice();
  const points = [];

  for (let step = 0; step < maxSteps; step += 1) {
    if (!pointInsideBounds(point, state.bounds, 0.18)) {
      break;
    }

    points.push(point);
    const wind = windAt(point[0], point[1], point[2], time, state.params, state.windContext);
    const windNorm = norm(wind);
    if (windNorm < 1e-6) {
      break;
    }
    point = add(point, scale(wind, (direction * stepLength) / windNorm));
  }

  return points;
}

function drawWindVolumeStreamlines(ctx, projector, row) {
  const bounds = state.bounds;
  const xSeeds = 5;
  const ySeeds = 4;
  const zSeeds = 3;
  const stepLength = 0.34;
  const maxSteps = 30;
  const lines = [];

  for (let ix = 1; ix <= xSeeds; ix += 1) {
    for (let iy = 1; iy <= ySeeds; iy += 1) {
      for (let iz = 1; iz <= zSeeds; iz += 1) {
        const seed = [
          lerp(bounds.xMin + 0.65, bounds.xMax - 0.65, ix / (xSeeds + 1)),
          lerp(bounds.yMin + 0.65, bounds.yMax - 0.65, iy / (ySeeds + 1)),
          lerp(0.72, bounds.zMax - 0.62, iz / (zSeeds + 1)),
        ];
        const seedWind = windAt(seed[0], seed[1], seed[2], row.time, state.params, state.windContext);
        const seedValue = clamp(norm(seedWind) / Math.max(state.colorMax, 1e-6), 0, 1);
        if (seedValue < 0.03) {
          continue;
        }
        const backward = traceWindFieldLine3D(seed, row.time, -1, stepLength, maxSteps).reverse();
        const forward = traceWindFieldLine3D(seed, row.time, 1, stepLength, maxSteps);
        const points = backward.concat(forward.slice(1));
        if (points.length > 5) {
          const depth = points.reduce((sum, point) => sum + projector.project(point).depth, 0) / points.length;
          lines.push({
            points,
            color: windColor(seedValue),
            alpha: isDarkTheme() ? 0.56 : 0.48,
            lineWidth: 1.15 + 1.1 * seedValue,
            depth,
          });
        }
      }
    }
  }

  lines.sort((a, b) => a.depth - b.depth);
  for (const line of lines) {
    drawPolyline3D(ctx, projector, line.points, line.color, line.lineWidth, line.alpha);
  }
}

function drawWindVolumeModel(ctx, projector, row) {
  drawWindVolumeSlices(ctx, projector, row);
  drawWindVolumeStreamlines(ctx, projector, row);
  drawWindVolumeGlyphs(ctx, projector, row);
}

function drawHoop(ctx, projector, params) {
  const colors = canvasPalette();
  const hoopPoints = [];
  const lowerNetPoints = [];
  const x = hoopX(params);
  for (let index = 0; index <= 96; index += 1) {
    const theta = (TWO_PI * index) / 96;
    hoopPoints.push([
      x + HOOP_RADIUS * Math.cos(theta),
      HOOP_Y + HOOP_RADIUS * Math.sin(theta),
      params.hoopHeight,
    ]);
    lowerNetPoints.push([
      x + 0.15 * Math.cos(theta),
      HOOP_Y + 0.15 * Math.sin(theta),
      params.hoopHeight - 0.42,
    ]);
  }

  const backboard = [
    [x + 0.18, -0.72, params.hoopHeight - 0.44],
    [x + 0.18, 0.72, params.hoopHeight - 0.44],
    [x + 0.18, 0.72, params.hoopHeight + 0.56],
    [x + 0.18, -0.72, params.hoopHeight + 0.56],
    [x + 0.18, -0.72, params.hoopHeight - 0.44],
  ];

  drawPolyline3D(ctx, projector, backboard, colors.backboard, 2.1, 0.7);
  drawLine3D(ctx, projector, [x + 0.44, 0, 0], [x + 0.44, 0, params.hoopHeight + 0.34], colors.pole, 7, 0.95);
  drawLine3D(ctx, projector, [x + 0.44, 0, params.hoopHeight], [x + 0.12, 0, params.hoopHeight], colors.pole, 4, 0.95);
  drawPolyline3D(
    ctx,
    projector,
    [[x + 0.2, -0.22, 0], [x + 0.72, -0.22, 0], [x + 0.72, 0.22, 0], [x + 0.2, 0.22, 0], [x + 0.2, -0.22, 0]],
    colors.pole,
    5,
    0.95
  );
  drawGlowingPolyline3D(ctx, projector, hoopPoints, colors.hoop, 4.4, 0.98, colors.hoopGlow);

  for (let index = 0; index < 12; index += 1) {
    const hoopIndex = Math.floor((index * 96) / 12);
    drawLine3D(ctx, projector, hoopPoints[hoopIndex], lowerNetPoints[hoopIndex], colors.net, 1.2, 0.85);
  }
  drawPolyline3D(ctx, projector, lowerNetPoints, colors.netSoft, 1.1, 0.8);
}

function drawCircleAt(ctx, projector, point, radius, fillStyle, strokeStyle = null, lineWidth = 1) {
  const projected = projector.project(point);
  ctx.save();
  ctx.fillStyle = fillStyle;
  ctx.beginPath();
  ctx.arc(projected.x, projected.y, radius, 0, TWO_PI);
  ctx.fill();
  if (strokeStyle) {
    ctx.strokeStyle = strokeStyle;
    ctx.lineWidth = lineWidth;
    ctx.stroke();
  }
  ctx.restore();
}

function drawPlayer(ctx, projector, params) {
  const colors = canvasPalette();
  const x = PLAYER_X;
  const y = PLAYER_Y;
  const ankleZ = 0;
  const hipZ = params.boyHeight * 0.43;
  const shoulderZ = params.boyHeight * 0.72;
  const headZ = params.boyHeight * 0.91;
  const release = [x, y, params.releaseHeight];

  drawLine3D(ctx, projector, [x, y - 0.09, ankleZ], [x, y - 0.04, hipZ], colors.playerShoe, 3, 0.95);
  drawLine3D(ctx, projector, [x, y + 0.09, ankleZ], [x, y + 0.04, hipZ], colors.playerShoe, 3, 0.95);
  drawLine3D(ctx, projector, [x, y - 0.04, hipZ], [x, y + 0.04, hipZ], colors.playerShort, 6, 0.96);
  drawLine3D(ctx, projector, [x, y, hipZ], [x, y, shoulderZ], colors.playerShirt, 8, 0.96);
  drawLine3D(ctx, projector, [x, y, shoulderZ], release, colors.playerLimb, 3.2, 0.96);
  drawLine3D(ctx, projector, [x, y + 0.04, shoulderZ], [x + 0.16, y, params.releaseHeight + 0.12], colors.playerLimb, 3.2, 0.96);
  drawCircleAt(ctx, projector, [x, y, headZ], 6.2, colors.playerSkin, colors.playerSkinStroke, 1);
  drawCircleAt(ctx, projector, release, 5.3, colors.trajectoryDot, colors.trajectoryStroke, 1);
}

function drawWorldModelPrediction(ctx, projector) {
  if (!worldModelEnabled || !state?.worldModelPrediction?.length) {
    return;
  }
  const colors = canvasPalette();
  const row = state.rows[state.frameIndex];
  const points = [
    [row.ball_x, row.ball_y, Math.max(row.ball_z, 0)],
    ...state.worldModelPrediction
      .filter((point) => Number.isFinite(point.ball_x) && Number.isFinite(point.ball_y) && Number.isFinite(point.ball_z))
      .filter((point) => point.ball_z >= -0.15)
      .map((point) => [point.ball_x, point.ball_y, Math.max(point.ball_z, 0)]),
  ];
  if (points.length < 2) {
    return;
  }

  drawGlowingPolyline3D(ctx, projector, points, colors.worldPrediction, 2.6, 0.88, colors.worldPredictionGlow);

  ctx.save();
  ctx.fillStyle = colors.worldPredictionDot;
  ctx.strokeStyle = colors.worldPredictionStroke;
  ctx.lineWidth = 1;
  for (let index = 1; index < points.length; index += 3) {
    const projected = projector.project(points[index]);
    ctx.beginPath();
    ctx.arc(projected.x, projected.y, 2.8, 0, TWO_PI);
    ctx.fill();
    ctx.stroke();
  }
  ctx.restore();
}

function drawTrajectory(ctx, projector) {
  const colors = canvasPalette();
  const currentRows = state.rows.slice(0, state.frameIndex + 1).filter((row) => row.ball_z >= -0.15);
  if (currentRows.length === 0) {
    return;
  }
  const points = currentRows.map((row) => [row.ball_x, row.ball_y, Math.max(row.ball_z, 0)]);
  drawGlowingPolyline3D(ctx, projector, points, colors.trajectory, 3.2, 0.98, colors.trajectoryGlow);

  ctx.save();
  ctx.fillStyle = colors.trajectoryDot;
  ctx.strokeStyle = colors.trajectoryStroke;
  ctx.lineWidth = 1;
  for (let index = 0; index < points.length; index += 4) {
    const projected = projector.project(points[index]);
    ctx.beginPath();
    ctx.arc(projected.x, projected.y, 3.3, 0, TWO_PI);
    ctx.fill();
    ctx.stroke();
  }
  ctx.restore();

  drawWorldModelPrediction(ctx, projector);

  const row = state.rows[state.frameIndex];
  drawGlowingCircleAt(ctx, projector, [row.ball_x, row.ball_y, Math.max(row.ball_z, 0)], 8.2, colors.ball, colors.ballStroke, 1.4, colors.ballGlow);
}

function drawColorbar(ctx, width, height) {
  const colors = canvasPalette();
  const barWidth = 22;
  const barHeight = clamp(height * 0.54, 180, 380);
  const x = width - 58;
  const y = 74;

  for (let index = 0; index < barHeight; index += 1) {
    const t = 1 - index / Math.max(1, barHeight - 1);
    ctx.strokeStyle = windColor(t);
    ctx.beginPath();
    ctx.moveTo(x, y + index);
    ctx.lineTo(x + barWidth, y + index);
    ctx.stroke();
  }

  ctx.strokeStyle = colors.colorbarBorder;
  ctx.lineWidth = 1;
  ctx.strokeRect(x, y, barWidth, barHeight);
  drawText(ctx, "||V vent||", x + barWidth / 2, y - 42, { font: "15px Inter, system-ui, sans-serif" });
  drawText(ctx, "(m/s)", x + barWidth / 2, y - 20, { font: "15px Inter, system-ui, sans-serif" });

  const ticks = niceTicks(0, state.colorMax, 6);
  for (const tick of ticks) {
    const ty = y + barHeight * (1 - tick / state.colorMax);
    ctx.strokeStyle = colors.colorbarTick;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x + barWidth, ty);
    ctx.lineTo(x + barWidth + 5, ty);
    ctx.stroke();
    drawText(ctx, String(Number(tick.toFixed(1))), x + barWidth + 13, ty, {
      align: "left",
      color: colors.text,
      font: "12px Inter, system-ui, sans-serif",
    });
  }
}

function drawSceneBackground(ctx, width, height, colors) {
  ctx.fillStyle = colors.background;
  ctx.fillRect(0, 0, width, height);
}

function drawScene() {
  if (!state) {
    return;
  }
  const { ctx, width, height } = prepareCanvas();
  const projector = createProjector(width, height, state.bounds, state.camera);
  const row = state.rows[state.frameIndex];
  const colors = canvasPalette();

  drawSceneBackground(ctx, width, height, colors);
  drawGrid(ctx, projector, state.bounds);
  if (windVisualizationMode === "field") {
    drawWindFieldMap(ctx, projector, row);
  } else if (windVisualizationMode === "volume") {
    drawWindVolumeModel(ctx, projector, row);
  } else {
    drawWindField(ctx, projector, row);
  }
  drawHoop(ctx, projector, state.params);
  drawPlayer(ctx, projector, state.params);
  drawTrajectory(ctx, projector);
  drawDragVector(ctx, projector, row);
  drawColorbar(ctx, width, height);
}

function csvValue(value, column) {
  if (value == null || Number.isNaN(value)) {
    return "";
  }
  if (typeof value === "number") {
    if (INTEGER_COLUMNS.has(column)) {
      return String(Math.round(value));
    }
    return value.toFixed(6);
  }
  const text = String(value);
  if (text.includes(",") || text.includes("\n") || text.includes('"')) {
    return `"${text.replaceAll('"', '""')}"`;
  }
  return text;
}

function makeCsvFromRows(columns, rows) {
  const lines = [columns.join(",")];
  for (const row of rows) {
    lines.push(columns.map((column) => csvValue(row[column], column)).join(","));
  }
  return `${lines.join("\n")}\n`;
}

function makeShotsCsv() {
  return makeCsvFromRows(SHOTS_COLUMNS, [makeShotCsvRow()]);
}

function makeTrajectoryCsv() {
  return makeCsvFromRows(TRAJECTORY_COLUMNS, state.rows.map(makeTrajectoryCsvRow));
}

function makeWindFieldCsv() {
  const lines = [WIND_FIELD_COLUMNS.join(",")];
  for (const row of state.rows) {
    state.windGrid.forEach((windPoint, windVectorId) => {
      const windFieldRow = makeWindFieldCsvRow(row, windPoint, windVectorId);
      lines.push(WIND_FIELD_COLUMNS.map((column) => csvValue(windFieldRow[column], column)).join(","));
    });
  }
  return `${lines.join("\n")}\n`;
}

function makeLabelsCsv() {
  return makeCsvFromRows(LABEL_COLUMNS, [makeLabelCsvRow()]);
}

function makeDatasetCsv(kind) {
  if (kind === "shots") {
    return makeShotsCsv();
  }
  if (kind === "trajectory") {
    return makeTrajectoryCsv();
  }
  if (kind === "wind_field") {
    return makeWindFieldCsv();
  }
  if (kind === "labels") {
    return makeLabelsCsv();
  }
  return "";
}

function downloadTextFile(filename, text) {
  const blob = new Blob([text], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function csvFilename(kind) {
  return `basketball_shot_${state.params.shotId}_${kind}.csv`;
}

function downloadCsv(kind = "all") {
  if (!state) {
    return;
  }
  const kinds = kind === "all" ? ["shots", "trajectory", "wind_field", "labels"] : [kind];
  for (const csvKind of kinds) {
    downloadTextFile(csvFilename(csvKind), makeDatasetCsv(csvKind));
  }
}

function resetCamera() {
  if (!state) {
    return;
  }
  state.camera = makeDefaultCamera();
  drawScene();
}

function closeViewMenu() {
  dom.viewCubeMenu.hidden = true;
  dom.viewCubeToggle.setAttribute("aria-expanded", "false");
}

function toggleViewMenu(event) {
  event.stopPropagation();
  const nextHidden = !dom.viewCubeMenu.hidden;
  dom.viewCubeMenu.hidden = nextHidden;
  dom.viewCubeToggle.setAttribute("aria-expanded", String(!nextHidden));
}

function setCameraView(viewName) {
  if (!state || !CAMERA_VIEWS[viewName]) {
    return;
  }
  const view = CAMERA_VIEWS[viewName];
  state.camera = {
    yaw: view.yaw,
    pitch: view.pitch,
    zoom: state.camera.zoom ?? DEFAULT_CAMERA.zoom,
  };
  closeViewMenu();
  drawScene();
}

function zoomCamera(event) {
  if (!state) {
    return;
  }
  event.preventDefault();
  const zoomFactor = Math.exp(-event.deltaY * 0.0012);
  state.camera.zoom = clamp(
    (state.camera.zoom ?? DEFAULT_CAMERA.zoom) * zoomFactor,
    CAMERA_MIN_ZOOM,
    CAMERA_MAX_ZOOM
  );
  drawScene();
}

function startCameraDrag(event) {
  if (!state || event.button !== 0) {
    return;
  }
  cameraDrag = {
    pointerId: event.pointerId,
    lastX: event.clientX,
    lastY: event.clientY,
  };
  dom.canvas.classList.add("is-dragging");
  if (dom.canvas.setPointerCapture) {
    dom.canvas.setPointerCapture(event.pointerId);
  }
  event.preventDefault();
}

function moveCameraDrag(event) {
  if (!state || !cameraDrag || cameraDrag.pointerId !== event.pointerId) {
    return;
  }

  const dx = event.clientX - cameraDrag.lastX;
  const dy = event.clientY - cameraDrag.lastY;
  cameraDrag.lastX = event.clientX;
  cameraDrag.lastY = event.clientY;

  state.camera.yaw += dx * 0.006;
  state.camera.pitch = clamp(
    state.camera.pitch - dy * 0.006,
    -CAMERA_PITCH_LIMIT,
    CAMERA_PITCH_LIMIT
  );
  drawScene();
  event.preventDefault();
}

function endCameraDrag(event) {
  if (!cameraDrag || cameraDrag.pointerId !== event.pointerId) {
    return;
  }
  if (dom.canvas.releasePointerCapture && dom.canvas.hasPointerCapture?.(event.pointerId)) {
    dom.canvas.releasePointerCapture(event.pointerId);
  }
  cameraDrag = null;
  dom.canvas.classList.remove("is-dragging");
}

for (const id of Object.keys(sliderFormat)) {
  controls[id].addEventListener("input", () => updateSliderOutput(id));
  updateSliderOutput(id);
}
updateAllRangeFills();

dom.goButton.addEventListener("click", () => runSimulation(false));
dom.aiCamerasLink.addEventListener("click", openAiCameras);
dom.resetCamera.addEventListener("click", resetCamera);
dom.windVizToggle.addEventListener("click", toggleWindVisualizationMode);
dom.worldModelToggle.addEventListener("click", toggleWorldModel);
dom.themeToggle.addEventListener("click", toggleTheme);
dom.viewCubeToggle.addEventListener("click", toggleViewMenu);
dom.viewCubeMenu.addEventListener("click", (event) => {
  const button = event.target.closest("[data-view]");
  if (!button) {
    return;
  }
  event.stopPropagation();
  setCameraView(button.dataset.view);
});
dom.playButton.addEventListener("click", togglePlayback);
dom.stepBackButton.addEventListener("click", () => stepFrame(-1));
dom.stepForwardButton.addEventListener("click", () => stepFrame(1));
dom.downloadCsv.addEventListener("click", () => downloadCsv("all"));
for (const button of dom.csvDownloadButtons) {
  button.addEventListener("click", () => downloadCsv(button.dataset.csvDownload));
}
dom.showDragVector.addEventListener("change", drawScene);
dom.timeSlider.addEventListener("input", (event) => {
  stopAnimation();
  updateRangeFill(event.target);
  setFrame(Number(event.target.value));
});
dom.canvas.addEventListener("pointerdown", startCameraDrag);
dom.canvas.addEventListener("pointermove", moveCameraDrag);
dom.canvas.addEventListener("pointerup", endCameraDrag);
dom.canvas.addEventListener("pointercancel", endCameraDrag);
dom.canvas.addEventListener("wheel", zoomCamera, { passive: false });
dom.canvas.addEventListener("lostpointercapture", () => {
  cameraDrag = null;
  dom.canvas.classList.remove("is-dragging");
});
document.addEventListener("click", (event) => {
  if (!dom.viewCubeMenu.hidden && !event.target.closest(".view-menu")) {
    closeViewMenu();
  }
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeViewMenu();
  }
});

const observer = new ResizeObserver(() => drawScene());
observer.observe(dom.canvas);

windVisualizationMode = readStoredWindVisualizationMode();
updateWindVisualizationButton();
updateWorldModelButton();
applyTheme(readStoredTheme());
runSimulation(true);
