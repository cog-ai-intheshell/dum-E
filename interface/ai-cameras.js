"use strict";

const GRAVITY = 9.81;
const DT = 0.02;
const MAX_STEPS = 250;
const PLAYER_X = 0.0;
const PLAYER_Y = 0.0;
const TARGET_Y = 0.0;
const CATCH_RADIUS = 0.34;
const ROBOT_BASE_OFFSET = 1.14;
const RESULT_SUCCESS = "CAUGHT";
const RESULT_FAIL = "MISSED";
// Legacy names remain aliases so existing query/data contracts still work.
const HOOP_Y = TARGET_Y;
const HOOP_RADIUS = CATCH_RADIUS;
const TWO_PI = Math.PI * 2;
const CAMERA_PITCH_LIMIT = Math.PI / 2;
const TRAIL_CONTEXT_FRAMES = 18;

const FIXED_SCENE_BOUNDS = Object.freeze({
  xMin: -0.75,
  xMax: 13.85,
  yMin: -4.5,
  yMax: 4.5,
  zMin: 0,
  zMax: 5.6,
});

const CAMERA_FEEDS = Object.freeze({
  profile: { yaw: -Math.PI / 2, pitch: 0, zoom: 1.06 },
  top: { yaw: -Math.PI / 2, pitch: CAMERA_PITCH_LIMIT, zoom: 1.08 },
  rear: { yaw: Math.PI, pitch: 0.03, zoom: 1.05 },
  hoop: { yaw: 0, pitch: 0.02, zoom: 1.05 },
  left: { yaw: Math.PI / 2, pitch: 0.02, zoom: 1.05 },
  oblique: { yaw: -Math.PI * 0.32, pitch: 0.34, zoom: 1.0 },
});

const WIND_REGIME_NAMES = {
  0: "no wind",
  1: "constant wind",
  2: "random wind",
  3: "Markov wind",
  4: "cyclic wind",
  5: "gusts",
  6: "sudden shift",
  7: "mean reversion",
  8: "volatility clustering",
  9: "jump / news shock",
  10: "pump & fade",
  11: "cascade liquidations",
  12: "liquidity wall",
  13: "squeeze breakout",
  14: "hidden regime switching",
  15: "chaotic wind",
};

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

const DEFAULT_CONTROL_VALUES = Object.freeze({
  initialForce: 12,
  verticalAngle: 45,
  horizontalAngle: 0,
  distanceToHoop: 6,
  boyHeight: 1.75,
  ballMass: 0.62,
  hoopHeight: 3.05,
  windRegime: 1,
  windCoupling: 1,
  windStrength: 4,
  windOrientation: 45,
  windVerticalOrientation: 0,
  dragCoeff: 0.02,
  showWindField: 1,
  useFieldView: 0,
  showPastTrail: 1,
});

const QUERY_PARAM_ALIASES = Object.freeze({
  initialForce: ["initialForce", "initial_force", "force"],
  verticalAngle: ["verticalAngle", "vertical_angle"],
  horizontalAngle: ["horizontalAngle", "horizontal_angle"],
  distanceToHoop: ["distanceToHoop", "distance_to_hoop", "distance"],
  boyHeight: ["boyHeight", "boy_height"],
  ballMass: ["ballMass", "ball_mass"],
  hoopHeight: ["hoopHeight", "hoop_height"],
  windRegime: ["windRegime", "wind_regime_id", "wind_regime"],
  windCoupling: ["windCoupling", "wind_spatial_coupling", "coupled_wind"],
  windStrength: ["windStrength", "wind_strength"],
  windOrientation: ["windOrientation", "wind_orientation"],
  windVerticalOrientation: ["windVerticalOrientation", "wind_vertical_orientation", "wind_elevation"],
  dragCoeff: ["dragCoeff", "drag_coeff"],
  showWindField: ["showWindField", "show_wind_field", "wind_visible"],
  useFieldView: ["useFieldView", "field_view", "champ"],
  showPastTrail: ["showPastTrail", "show_past_trail", "trail"],
});

const queryParams = new URLSearchParams(window.location.search);

function queryFlag(name, defaultValue = false) {
  const value = queryParams.get(name);
  if (value == null || value === "") {
    return defaultValue;
  }
  return !["0", "false", "off", "no"].includes(value.trim().toLowerCase());
}

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
  lightTheme: document.getElementById("lightTheme"),
  darkTheme: document.getElementById("darkTheme"),
  themeToggle: document.getElementById("themeToggle"),
  resultTitle: document.getElementById("resultTitle"),
  timeTitle: document.getElementById("timeTitle"),
  goButton: document.getElementById("goButton"),
  playButton: document.getElementById("playButton"),
  stepBackButton: document.getElementById("stepBackButton"),
  stepForwardButton: document.getElementById("stepForwardButton"),
  timeSlider: document.getElementById("timeSlider"),
  showWindField: document.getElementById("showWindField"),
  useFieldView: document.getElementById("useFieldView"),
  showPastTrail: document.getElementById("showPastTrail"),
  aiStatsBody: document.getElementById("aiStatsBody"),
  canvases: Array.from(document.querySelectorAll("[data-camera]")),
};

let shotCounter = 0;
let animationFrame = null;
let previousAnimationTime = 0;
let state = null;

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

function targetX(params) {
  return params.distanceToHoop;
}

function catchCenter(params) {
  return [targetX(params), TARGET_Y, params.hoopHeight];
}

function robotBaseX(params) {
  return targetX(params) + ROBOT_BASE_OFFSET;
}

function hoopX(params) {
  return targetX(params);
}

function hoopCenter(params) {
  return catchCenter(params);
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
  const value = rawCssVar(name);
  return value ? resolveCssValue(value, fallback) : fallback;
}

function cssVarAny(names, fallback) {
  for (const name of names) {
    const value = rawCssVar(name);
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
  if (dom.themeToggle) {
    dom.themeToggle.textContent = nextTheme === "dark" ? "☀" : "☾";
    dom.themeToggle.setAttribute(
      "aria-label",
      nextTheme === "dark" ? "Enable light mode" : "Enable dark mode"
    );
    dom.themeToggle.title = nextTheme === "dark" ? "Enable light mode" : "Enable dark mode";
    dom.themeToggle.setAttribute("aria-pressed", String(nextTheme === "dark"));
    localStorage.setItem("dum-e-playground-theme", nextTheme);
  }
  requestAnimationFrame(drawAllCameras);
}

function readStoredTheme() {
  return (localStorage.getItem("dum-e-playground-theme") || localStorage.getItem("basketball-simulator-theme")) === "dark"
    ? "dark"
    : "light";
}

function toggleTheme() {
  applyTheme(isDarkTheme() ? "light" : "dark");
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

function controlNumber(id) {
  const rawValue = queryValueFor(id) ?? controls[id]?.value ?? DEFAULT_CONTROL_VALUES[id];
  const value = Number(rawValue);
  return Number.isFinite(value) ? value : Number(DEFAULT_CONTROL_VALUES[id]);
}

function controlChecked(id) {
  const queryValue = queryValueFor(id);
  if (queryValue != null) {
    return !["0", "false", "off", "no"].includes(String(queryValue).trim().toLowerCase());
  }
  const control = controls[id] ?? dom[id];
  if (!control) {
    return Boolean(DEFAULT_CONTROL_VALUES[id]);
  }
  return control.checked;
}

function fieldViewMode() {
  const queryValue = queryValueFor("useFieldView");
  if (queryValue != null) {
    const normalized = String(queryValue).trim().toLowerCase();
    if (["2", "volume", "volume3d", "3d", "volumique"].includes(normalized)) {
      return 2;
    }
    if (["1", "true", "on", "yes", "field", "champ", "map", "coupe"].includes(normalized)) {
      return 1;
    }
    return 0;
  }
  return controlChecked("useFieldView") ? 1 : 0;
}

function queryValueFor(id) {
  const keys = QUERY_PARAM_ALIASES[id] ?? [id];
  for (const key of keys) {
    const value = queryParams.get(key);
    if (value != null && value !== "") {
      return value;
    }
  }
  return null;
}

function readInitialFrameIndex(rowCount) {
  const rawFrame = queryParams.get("frame") ?? queryParams.get("timestep");
  if (rawFrame != null && rawFrame !== "") {
    const frame = Number(rawFrame);
    if (Number.isFinite(frame)) {
      return clamp(Math.round(frame), 0, rowCount - 1);
    }
  }

  const rawTime = queryParams.get("time");
  if (rawTime != null && rawTime !== "") {
    const time = Number(rawTime);
    if (Number.isFinite(time)) {
      return clamp(Math.round(time / DT), 0, rowCount - 1);
    }
  }

  return 0;
}

function updateSliderOutput(id) {
  const input = controls[id];
  const output = document.getElementById(`${id}Value`);
  if (!input || !output) {
    return;
  }
  output.value = Number(input.value).toFixed(sliderFormat[id]);
}

function readParams() {
  const windRegimeId = controlNumber("windRegime");
  return {
    shotId: shotCounter,
    initialForce: controlNumber("initialForce"),
    verticalAngle: controlNumber("verticalAngle"),
    horizontalAngle: controlNumber("horizontalAngle"),
    distanceToHoop: controlNumber("distanceToHoop"),
    boyHeight: controlNumber("boyHeight"),
    ballMass: controlNumber("ballMass"),
    hoopHeight: controlNumber("hoopHeight"),
    windRegimeId,
    windRegime: WIND_REGIME_NAMES[windRegimeId],
    windSpatialCoupling: controlChecked("windCoupling") ? 1 : 0,
    windStrength: controlNumber("windStrength"),
    windOrientation: controlNumber("windOrientation"),
    windVerticalOrientation: controlNumber("windVerticalOrientation"),
    dragCoeff: controlNumber("dragCoeff"),
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

function closestPointOnSegment(a, b, point) {
  const segment = sub(b, a);
  const lengthSquared = dot(segment, segment);
  const alpha = lengthSquared < 1e-12
    ? 0
    : clamp(dot(sub(point, a), segment) / lengthSquared, 0, 1);
  const closest = add(a, scale(segment, alpha));
  return {
    alpha,
    point: closest,
    distance: norm(sub(closest, point)),
  };
}

function computeCatch(prevPosition, position, prevTime, time, params) {
  if (!prevPosition) {
    return null;
  }
  const closest = closestPointOnSegment(prevPosition, position, catchCenter(params));
  if (closest.distance <= CATCH_RADIUS) {
    return {
      time: lerp(prevTime, time, closest.alpha),
      distance: closest.distance,
    };
  }
  return null;
}

function makeRow(params, step, t, position, velocity, acceleration, wind, minDistance) {
  const center = hoopCenter(params);
  return {
    timestep: step,
    time: t,
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
    wind_norm: norm(wind),
    speed_norm: norm(velocity),
    distance_ball_to_hoop: norm(sub(position, center)),
    horizontal_distance_to_hoop: Math.hypot(position[0] - center[0], position[1] - center[1]),
    min_hoop_distance_so_far: minDistance,
  };
}

function simulateShot(params) {
  params.releaseHeight = params.boyHeight * 0.85;
  const windContext = createWindContext(params);
  const verticalAngle = degToRad(params.verticalAngle);
  const horizontalAngle = degToRad(params.horizontalAngle);
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
  let prevTime = null;

  for (let step = 0; step < MAX_STEPS; step += 1) {
    const time = step * DT;
    const { acceleration, wind } = accelerationAt(position, velocity, time, params, windContext);
    finalMinDistance = Math.min(finalMinDistance, norm(sub(position, hoopCenter(params))));

    if (label === 0 && prevPosition) {
      const catchEvent = computeCatch(prevPosition, position, prevTime, time, params);
      if (catchEvent) {
        label = 1;
        collisionTime = catchEvent.time;
      }
    }

    rows.push(makeRow(params, step, time, position, velocity, acceleration, wind, finalMinDistance));

    if (position[2] < -0.2 && velocity[2] < 0 && step > 6) {
      break;
    }

    prevPosition = position.slice();
    prevTime = time;

    const nextState = rk4Step(position, velocity, time, DT, params, windContext);
    position = nextState.position;
    velocity = nextState.velocity;
  }

  return {
    rows,
    windContext,
    metadata: {
      label,
      result: label === 1 ? RESULT_SUCCESS : RESULT_FAIL,
      minHoopDistance: finalMinDistance,
      collisionTime,
    },
  };
}

function buildWindGrid(bounds) {
  const points = [];
  const nx = 12;
  const ny = 7;
  const nz = 4;

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

function jetColor(t) {
  const value = clamp(t, 0, 1);
  const r = clamp(1.5 - Math.abs(4 * value - 3), 0, 1);
  const g = clamp(1.5 - Math.abs(4 * value - 2), 0, 1);
  const b = clamp(1.5 - Math.abs(4 * value - 1), 0, 1);
  return `rgb(${Math.round(255 * r)}, ${Math.round(255 * g)}, ${Math.round(255 * b)})`;
}

function themeWindColor(t) {
  if (!isDarkTheme()) {
    return jetColor(t);
  }
  return interpolateColor(
    [
      parseCssColor(cssVar("--wind-low", "#0d0887"), [13, 8, 135]),
      parseCssColor(cssVar("--wind-mid", "#7e03a8"), [126, 3, 168]),
      parseCssColor(cssVar("--wind-high", "#cc4778"), [204, 71, 120]),
      parseCssColor(cssVar("--wind-warm", "#ff2a00"), [255, 42, 0]),
      parseCssColor(cssVar("--wind-hot", "#f89540"), [248, 149, 64]),
      parseCssColor(cssVar("--wind-max", "#f0f921"), [240, 249, 33]),
    ],
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
    robotBase: cssVar("--canvas-robot-base", cssVar("--canvas-pole", "#16191d")),
    robotBaseEdge: cssVar("--canvas-robot-base-edge", "#2f3338"),
    robotArm: cssVar("--canvas-robot-arm", "#e87511"),
    robotArmLight: cssVar("--canvas-robot-arm-light", "#f6a24a"),
    robotArmShadow: cssVar("--canvas-robot-arm-shadow", "#7a2f0b"),
    robotJoint: cssVar("--canvas-robot-joint", "#121417"),
    robotJointStroke: cssVar("--canvas-robot-joint-stroke", "#3b4148"),
    robotJointAccent: cssVar("--canvas-robot-accent", "#f4d23c"),
    robotClaw: cssVar("--canvas-robot-claw", "#1f252b"),
    robotCable: cssVar("--canvas-robot-cable", "#111827"),
    catchZone: cssVar("--canvas-catch-zone", "#22c55e"),
    catchZoneSoft: cssVar("--canvas-catch-zone-soft", "#86efac"),
    catchZoneGlow: glowFromCss("--canvas-catch-zone-glow"),
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
    ball: cssVar("--canvas-ball", "#f28a1d"),
    ballStroke: cssVar("--canvas-ball-stroke", "#a94908"),
    ballGlow: glowFromCss("--canvas-ball-glow"),
  };
}

function prepareCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, rect.width);
  const height = Math.max(1, rect.height);
  const targetWidth = Math.round(width * dpr);
  const targetHeight = Math.round(height * dpr);
  if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
    canvas.width = targetWidth;
    canvas.height = targetHeight;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return { ctx, width, height };
}

function createProjector(width, height, bounds, camera) {
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
  const margin = { left: 20, right: 20, top: 16, bottom: 20 };
  const availableWidth = Math.max(1, width - margin.left - margin.right);
  const availableHeight = Math.max(1, height - margin.top - margin.bottom);
  const baseScaleX = availableWidth / Math.max(maxX - minX, 1e-6);
  const baseScaleY = availableHeight / Math.max(maxY - minY, 1e-6);
  const scaleBase = Math.min(baseScaleX, baseScaleY);
  const scaleX = scaleBase * camera.zoom;
  const scaleY = scaleBase * camera.zoom;
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

  return { project, right, up, forward };
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

function ringPoints(center, radius, plane = "xy", segments = 72) {
  const points = [];
  for (let index = 0; index <= segments; index += 1) {
    const theta = (TWO_PI * index) / segments;
    const c = Math.cos(theta) * radius;
    const s = Math.sin(theta) * radius;
    if (plane === "xz") {
      points.push([center[0] + c, center[1], center[2] + s]);
    } else if (plane === "yz") {
      points.push([center[0], center[1] + c, center[2] + s]);
    } else {
      points.push([center[0] + c, center[1] + s, center[2]]);
    }
  }
  return points;
}

function drawRing3D(ctx, projector, center, radius, plane, strokeStyle, lineWidth = 1, alpha = 1, glow = null) {
  withGlow(ctx, glow, () => {
    drawPolyline3D(ctx, projector, ringPoints(center, radius, plane), strokeStyle, lineWidth, alpha);
  });
}

function drawDashedRing3D(ctx, projector, center, radius, plane, strokeStyle, lineWidth = 1, alpha = 1) {
  ctx.save();
  ctx.setLineDash([6, 5]);
  drawPolyline3D(ctx, projector, ringPoints(center, radius, plane), strokeStyle, lineWidth, alpha);
  ctx.restore();
}

function drawText(ctx, text, x, y, options = {}) {
  ctx.save();
  ctx.fillStyle = options.color || cssVarAny(["--canvas-text", "--text", "--ink"], "#111827");
  ctx.font = options.font || "12px Inter, system-ui, sans-serif";
  ctx.textAlign = options.align || "center";
  ctx.textBaseline = options.baseline || "middle";
  ctx.fillText(text, x, y);
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

function drawGrid(ctx, projector, bounds) {
  const colors = canvasPalette();
  const xTicks = niceTicks(bounds.xMin, bounds.xMax, 7);
  const yTicks = niceTicks(bounds.yMin, bounds.yMax, 7);
  const zTicks = niceTicks(bounds.zMin, bounds.zMax, 5);

  for (const x of xTicks) {
    drawLine3D(ctx, projector, [x, bounds.yMin, 0], [x, bounds.yMax, 0], colors.grid, 1, 0.74);
  }
  for (const y of yTicks) {
    drawLine3D(ctx, projector, [bounds.xMin, y, 0], [bounds.xMax, y, 0], colors.grid, 1, 0.74);
  }
  for (const z of zTicks) {
    drawLine3D(ctx, projector, [bounds.xMin, bounds.yMin, z], [bounds.xMax, bounds.yMin, z], colors.grid, 0.8, 0.34);
    drawLine3D(ctx, projector, [bounds.xMin, bounds.yMax, z], [bounds.xMax, bounds.yMax, z], colors.grid, 0.8, 0.34);
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
    drawLine3D(ctx, projector, a, b, colors.box, 1, 0.9);
  }

  drawLine3D(ctx, projector, [bounds.xMin, bounds.yMin, 0], [bounds.xMax, bounds.yMin, 0], colors.axis, 1.15, 0.94);
  drawLine3D(ctx, projector, [bounds.xMin, bounds.yMin, 0], [bounds.xMin, bounds.yMax, 0], colors.axis, 1.15, 0.94);
  drawLine3D(ctx, projector, [bounds.xMin, bounds.yMin, 0], [bounds.xMin, bounds.yMin, bounds.zMax], colors.axis, 1.15, 0.94);
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
  const headSize = clamp(length * 0.34, options.minHeadSize ?? 2.2, options.maxHeadSize ?? 4.8);

  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = options.lineWidth ?? 0.95;
  ctx.globalAlpha = options.alpha ?? 0.74;
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
  if (!controlChecked("showWindField")) {
    return;
  }
  const arrows = [];
  for (const point of state.windGrid) {
    const wind = windAt(point[0], point[1], point[2], row.time, state.params, state.windContext);
    const windNorm = norm(wind);
    const direction = normalize(wind);
    const length = 0.1 + 0.09 * clamp(windNorm / Math.max(state.colorMax, 1e-6), 0, 1.4);
    const headPoint = add(point, scale(direction, length));
    const projectedTail = projector.project(point);
    const projectedHead = projector.project(headPoint);
    arrows.push({
      tail: projectedTail,
      head: projectedHead,
      color: themeWindColor(windNorm / state.colorMax),
      depth: projectedTail.depth,
    });
  }
  arrows.sort((a, b) => a.depth - b.depth);
  for (const arrow of arrows) {
    drawArrow(ctx, arrow.tail, arrow.head, arrow.color);
  }
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
  const lineColor = rgbaFromCssColor(
    cssVarAny(["--canvas-text", "--text", "--ink"], "#ffffff"),
    isDarkTheme() ? 0.4 : 0.34,
    [255, 255, 255]
  );
  const levels = [0.18, 0.3, 0.42, 0.54, 0.66, 0.78, 0.9];

  ctx.save();
  ctx.lineWidth = isDarkTheme() ? 0.62 : 0.55;
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
  const uSeeds = 8;
  const vSeeds = 5;
  const uSpan = plane.uMax - plane.uMin;
  const vSpan = plane.vMax - plane.vMin;
  const stepLength = Math.min(uSpan, vSpan) / 38;
  const maxSteps = 32;
  const lineColor = cssVarAny(["--canvas-text", "--text", "--ink"], "#ffffff");

  ctx.save();
  ctx.lineWidth = isDarkTheme() ? 0.9 : 0.78;
  for (let i = 1; i < uSeeds; i += 1) {
    for (let j = 1; j < vSeeds; j += 1) {
      const seedU = lerp(plane.uMin, plane.uMax, i / uSeeds);
      const seedV = lerp(plane.vMin, plane.vMax, j / vSeeds);
      const backward = traceFieldLine(plane, seedU, seedV, time, -1, stepLength, maxSteps).reverse();
      const forward = traceFieldLine(plane, seedU, seedV, time, 1, stepLength, maxSteps);
      const points = backward.concat(forward.slice(1));
      if (points.length > 4) {
        drawPolyline3D(ctx, projector, points, lineColor, ctx.lineWidth, isDarkTheme() ? 0.42 : 0.5);
      }
    }
  }
  ctx.restore();
}

function drawWindFieldMap(ctx, projector, row, camera) {
  if (!controlChecked("showWindField")) {
    return;
  }

  const plane = makeWindFieldPlane(state.bounds, state.params, camera);
  const cols = 24;
  const rows = 16;
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

  const fieldAlpha = isDarkTheme() ? 0.52 : 0.38;
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
        themeWindColor(value),
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
  const cols = 10;
  const rows = 6;
  const levels = 4;
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
          color: themeWindColor(value),
          alpha: (isDarkTheme() ? 0.1 : 0.08) + value * (isDarkTheme() ? 0.11 : 0.09),
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
    const headPoint = add(point, scale(direction, 0.07 + 0.1 * value));
    const center = projector.project(point);
    glyphs.push({
      center,
      head: projector.project(headPoint),
      color: themeWindColor(value),
      radius: 0.9 + 2.4 * value,
      lineWidth: 0.5 + 0.95 * value,
      alpha: 0.28 + 0.42 * value,
      depth: center.depth,
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
  const lines = [];
  const xSeeds = 4;
  const ySeeds = 3;
  const zSeeds = 3;

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

        const backward = traceWindFieldLine3D(seed, row.time, -1, 0.34, 24).reverse();
        const forward = traceWindFieldLine3D(seed, row.time, 1, 0.34, 24);
        const points = backward.concat(forward.slice(1));
        if (points.length > 5) {
          const depth = points.reduce((sum, point) => sum + projector.project(point).depth, 0) / points.length;
          lines.push({
            points,
            color: themeWindColor(seedValue),
            alpha: isDarkTheme() ? 0.48 : 0.4,
            lineWidth: 0.85 + 0.85 * seedValue,
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
  if (!controlChecked("showWindField")) {
    return;
  }
  drawWindVolumeSlices(ctx, projector, row);
  drawWindVolumeStreamlines(ctx, projector, row);
  drawWindVolumeGlyphs(ctx, projector, row);
}

function robotPose(params) {
  const catchPoint = catchCenter(params);
  const baseX = robotBaseX(params);
  const shoulder = [baseX, TARGET_Y, 0.76];
  const elbow = [
    lerp(baseX, catchPoint[0], 0.34),
    TARGET_Y,
    clamp(catchPoint[2] - 0.86, 1.15, 3.1),
  ];
  const wrist = [catchPoint[0] + 0.2, TARGET_Y, catchPoint[2] - 0.12];
  const palm = [catchPoint[0] + 0.07, TARGET_Y, catchPoint[2]];
  return { baseX, catchPoint, shoulder, elbow, wrist, palm };
}

function drawRobotLink(ctx, projector, a, b, colors, width) {
  drawLine3D(ctx, projector, a, b, colors.robotArmShadow, width + 2.8, 0.94);
  drawLine3D(ctx, projector, a, b, colors.robotArm, width, 0.98);
  drawLine3D(
    ctx,
    projector,
    [a[0], a[1] - 0.045, a[2] + 0.02],
    [b[0], b[1] - 0.045, b[2] + 0.02],
    colors.robotArmLight,
    Math.max(1.8, width * 0.28),
    0.72
  );
}

function drawRobotJoint(ctx, projector, point, radius, colors) {
  drawRing3D(ctx, projector, point, radius * 0.034, "xz", colors.robotJointStroke, 2, 0.9);
  drawCircleAt(ctx, projector, point, radius, colors.robotJoint, colors.robotJointStroke, 1.1);
  drawCircleAt(ctx, projector, [point[0], point[1] - 0.012, point[2] + 0.006], radius * 0.42, colors.robotJointAccent, null, 0);
}

function drawDumERobot(ctx, projector, params) {
  const colors = canvasPalette();
  const { baseX, catchPoint, shoulder, elbow, wrist, palm } = robotPose(params);
  const platform = [
    [baseX - 0.46, TARGET_Y - 0.34, 0],
    [baseX + 0.46, TARGET_Y - 0.34, 0],
    [baseX + 0.52, TARGET_Y + 0.34, 0],
    [baseX - 0.52, TARGET_Y + 0.34, 0],
    [baseX - 0.46, TARGET_Y - 0.34, 0],
  ];

  drawPolyline3D(ctx, projector, platform, colors.robotBase, 5.4, 0.96);
  drawRing3D(ctx, projector, [baseX, TARGET_Y, 0.16], 0.33, "xy", colors.robotBaseEdge, 2.6, 0.88);
  drawLine3D(ctx, projector, [baseX, TARGET_Y - 0.2, 0.08], [baseX, TARGET_Y - 0.2, 0.56], colors.robotBase, 7, 0.96);
  drawLine3D(ctx, projector, [baseX, TARGET_Y + 0.2, 0.08], [baseX, TARGET_Y + 0.2, 0.56], colors.robotBase, 7, 0.96);
  drawRing3D(ctx, projector, [baseX, TARGET_Y, 0.56], 0.24, "xy", colors.robotJointAccent, 2.4, 0.96);

  drawDashedRing3D(ctx, projector, catchPoint, CATCH_RADIUS, "xy", colors.catchZone, 1.35, 0.72);
  drawDashedRing3D(ctx, projector, catchPoint, CATCH_RADIUS, "xz", colors.catchZoneSoft, 1.1, 0.54);
  drawRing3D(ctx, projector, catchPoint, CATCH_RADIUS * 0.58, "yz", colors.catchZoneSoft, 0.9, 0.32, colors.catchZoneGlow);

  drawRobotLink(ctx, projector, shoulder, elbow, colors, 9.4);
  drawRobotLink(ctx, projector, elbow, wrist, colors, 7.8);
  drawLine3D(ctx, projector, [shoulder[0] + 0.05, TARGET_Y + 0.16, shoulder[2] + 0.08], [elbow[0] + 0.02, TARGET_Y + 0.14, elbow[2] - 0.08], colors.robotCable, 1.15, 0.72);
  drawLine3D(ctx, projector, [elbow[0] - 0.04, TARGET_Y + 0.14, elbow[2] + 0.06], [wrist[0] - 0.02, TARGET_Y + 0.13, wrist[2] - 0.02], colors.robotCable, 1.1, 0.72);

  drawRobotJoint(ctx, projector, shoulder, 7.3, colors);
  drawRobotJoint(ctx, projector, elbow, 6.8, colors);
  drawRobotJoint(ctx, projector, wrist, 5.8, colors);

  drawLine3D(ctx, projector, wrist, palm, colors.robotClaw, 4.4, 0.96);
  drawLine3D(ctx, projector, palm, [catchPoint[0] - 0.1, TARGET_Y - 0.17, catchPoint[2] + 0.11], colors.robotClaw, 3, 0.96);
  drawLine3D(ctx, projector, palm, [catchPoint[0] - 0.1, TARGET_Y + 0.17, catchPoint[2] + 0.11], colors.robotClaw, 3, 0.96);
  drawLine3D(ctx, projector, palm, [catchPoint[0] - 0.14, TARGET_Y - 0.16, catchPoint[2] - 0.1], colors.robotClaw, 2.7, 0.92);
  drawLine3D(ctx, projector, palm, [catchPoint[0] - 0.14, TARGET_Y + 0.16, catchPoint[2] - 0.1], colors.robotClaw, 2.7, 0.92);
  drawCircleAt(ctx, projector, palm, 3.8, colors.robotJoint, colors.robotJointStroke, 0.9);

  const label = projector.project([baseX, TARGET_Y, 0.34]);
  drawText(ctx, "DUM-E", label.x, label.y - 18, {
    color: colors.robotJointAccent,
    font: "700 11px Inter, system-ui, sans-serif",
  });
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

  drawLine3D(ctx, projector, [x, y - 0.09, ankleZ], [x, y - 0.04, hipZ], colors.playerShoe, 2.4, 0.95);
  drawLine3D(ctx, projector, [x, y + 0.09, ankleZ], [x, y + 0.04, hipZ], colors.playerShoe, 2.4, 0.95);
  drawLine3D(ctx, projector, [x, y - 0.04, hipZ], [x, y + 0.04, hipZ], colors.playerShort, 5, 0.96);
  drawLine3D(ctx, projector, [x, y, hipZ], [x, y, shoulderZ], colors.playerShirt, 6.4, 0.96);
  drawLine3D(ctx, projector, [x, y, shoulderZ], release, colors.playerLimb, 2.8, 0.96);
  drawLine3D(ctx, projector, [x, y + 0.04, shoulderZ], [x + 0.16, y, params.releaseHeight + 0.12], colors.playerLimb, 2.8, 0.96);
  drawCircleAt(ctx, projector, [x, y, headZ], 4.8, colors.playerSkin, colors.playerSkinStroke, 0.9);
}

function drawPastContext(ctx, projector) {
  const colors = canvasPalette();
  const start = Math.max(0, state.frameIndex - TRAIL_CONTEXT_FRAMES);
  const samples = state.rows
    .slice(start, state.frameIndex)
    .filter((sample) => sample.ball_z >= -0.15);

  if (samples.length === 0) {
    return;
  }

  ctx.save();
  ctx.fillStyle = colors.trajectoryDot;
  ctx.strokeStyle = colors.trajectoryStroke;
  ctx.lineWidth = 0.85;
  for (const sample of samples) {
    const ageFrames = state.frameIndex - sample.timestep;
    const recency = 1 - clamp(ageFrames / TRAIL_CONTEXT_FRAMES, 0, 1);
    const projected = projector.project([sample.ball_x, sample.ball_y, Math.max(sample.ball_z, 0)]);
    ctx.globalAlpha = 0.18 + 0.5 * recency;
    ctx.beginPath();
    ctx.arc(projected.x, projected.y, 2.7, 0, TWO_PI);
    ctx.fill();
    ctx.stroke();
  }
  ctx.restore();
}

function drawBallAndPastTrail(ctx, projector, row) {
  const colors = canvasPalette();
  if (controlChecked("showPastTrail")) {
    drawPastContext(ctx, projector);
  }

  withGlow(ctx, colors.ballGlow, () => {
    drawCircleAt(ctx, projector, [row.ball_x, row.ball_y, Math.max(row.ball_z, 0)], 6.4, colors.ball, colors.ballStroke, 1.1);
  });
}

function drawSceneBackground(ctx, width, height, colors) {
  ctx.fillStyle = colors.background;
  ctx.fillRect(0, 0, width, height);
}

function drawCamera(canvas) {
  if (!state) {
    return;
  }
  const cameraId = canvas.dataset.camera;
  const camera = CAMERA_FEEDS[cameraId] || CAMERA_FEEDS.oblique;
  const { ctx, width, height } = prepareCanvas(canvas);
  const colors = canvasPalette();
  const projector = createProjector(width, height, state.bounds, camera);
  const row = state.rows[state.frameIndex];

  drawSceneBackground(ctx, width, height, colors);
  drawGrid(ctx, projector, state.bounds);
  const windMode = fieldViewMode();
  if (windMode === 2) {
    drawWindVolumeModel(ctx, projector, row);
  } else if (windMode === 1) {
    drawWindFieldMap(ctx, projector, row, camera);
  } else {
    drawWindField(ctx, projector, row);
  }
  drawDumERobot(ctx, projector, state.params);
  drawPlayer(ctx, projector, state.params);
  drawBallAndPastTrail(ctx, projector, row);
}

function drawAllCameras() {
  if (!state) {
    return;
  }
  for (const canvas of dom.canvases) {
    drawCamera(canvas);
  }
}

function updateTitles() {
  if (!state || !dom.resultTitle || !dom.timeTitle) {
    return;
  }
  const row = state.rows[state.frameIndex];
  const result = state.metadata.result;
  dom.resultTitle.textContent = result;
  dom.resultTitle.className = result === RESULT_SUCCESS ? "result-made" : "result-fail";
  dom.timeTitle.textContent = `t = ${row.time.toFixed(2)} s (timestep ${row.timestep} / ${MAX_STEPS})`;
}

function renderStats() {
  if (!state || !dom.aiStatsBody) {
    return;
  }
  const collision = state.metadata.collisionTime == null ? "-" : `${state.metadata.collisionTime.toFixed(2)} s`;
  const row = state.rows[state.frameIndex];
  const resultClass = state.metadata.result === RESULT_SUCCESS ? "metric-made" : "metric-fail";
  const rows = [
    ["Target label", state.metadata.label, "Result", `<span class="${resultClass}">${state.metadata.result}</span>`],
    ["DUM-E distance", `${state.params.distanceToHoop.toFixed(1)} m`, "Claw distance", `${state.metadata.minHoopDistance.toFixed(3)} m`],
    ["Catch time", collision, "Wind regime", state.params.windRegime],
    ["Active frame", row.timestep, "Ball speed", `${row.speed_norm.toFixed(2)} m/s`],
  ];
  dom.aiStatsBody.innerHTML = rows
    .map(([k1, v1, k2, v2]) => `<tr><td>${k1}</td><td>${v1}</td><td>${k2}</td><td>${v2}</td></tr>`)
    .join("");
}

function setFrame(index) {
  if (!state) {
    return;
  }
  state.frameIndex = clamp(index, 0, state.rows.length - 1);
  if (dom.timeSlider) {
    dom.timeSlider.value = String(state.frameIndex);
  }
  updateTitles();
  renderStats();
  drawAllCameras();
  updatePlaybackControls();
}

function isAnimationAtEnd() {
  return Boolean(state && state.frameIndex >= state.rows.length - 1);
}

function updatePlaybackControls() {
  if (!dom.playButton || !dom.stepBackButton || !dom.stepForwardButton) {
    return;
  }
  if (!state) {
    dom.playButton.dataset.mode = "play";
    dom.playButton.setAttribute("aria-label", "Play");
    dom.playButton.title = "Play";
    dom.stepBackButton.disabled = true;
    dom.stepForwardButton.disabled = true;
    return;
  }

  const mode = state.playing ? "pause" : isAnimationAtEnd() ? "replay" : "play";
  const labels = {
    play: "Play",
    pause: "Pause",
    replay: "Replay from start",
  };
  dom.playButton.dataset.mode = mode;
  dom.playButton.setAttribute("aria-label", labels[mode]);
  dom.playButton.title = labels[mode];
  dom.stepBackButton.disabled = state.frameIndex <= 0;
  dom.stepForwardButton.disabled = isAnimationAtEnd();
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
        if (queryFlag("loop")) {
          setFrame(0);
          animationFrame = requestAnimationFrame(tick);
          return;
        }
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

function returnToSimulation() {
  const returnTo = queryParams.get("return_to") || "index.html";
  window.location.href = returnTo;
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
  window.__AI_CAMERAS_READY__ = false;
  stopAnimation();
  if (!isInitial) {
    shotCounter += 1;
  }
  const params = readParams();
  const result = simulateShot(params);
  const initialFrameIndex = readInitialFrameIndex(result.rows.length);

  state = {
    params,
    rows: result.rows,
    metadata: result.metadata,
    windContext: result.windContext,
    bounds: { ...FIXED_SCENE_BOUNDS },
    windGrid: buildWindGrid(FIXED_SCENE_BOUNDS),
    frameIndex: initialFrameIndex,
    playing: false,
    colorMax: windColorScaleMax(params),
  };

  if (dom.timeSlider) {
    dom.timeSlider.max = String(Math.max(0, state.rows.length - 1));
    dom.timeSlider.value = String(initialFrameIndex);
  }
  updateTitles();
  renderStats();
  updatePlaybackControls();
  drawAllCameras();
  window.__AI_CAMERAS_READY__ = true;
}

for (const id of Object.keys(sliderFormat)) {
  controls[id]?.addEventListener("input", () => updateSliderOutput(id));
  updateSliderOutput(id);
}

dom.goButton?.addEventListener("click", () => runSimulation(false));
dom.themeToggle?.addEventListener("click", toggleTheme);
dom.playButton?.addEventListener("click", togglePlayback);
dom.stepBackButton?.addEventListener("click", () => stepFrame(-1));
dom.stepForwardButton?.addEventListener("click", () => stepFrame(1));
dom.timeSlider?.addEventListener("input", (event) => {
  stopAnimation();
  setFrame(Number(event.target.value));
});
dom.showWindField?.addEventListener("change", drawAllCameras);
dom.useFieldView?.addEventListener("change", drawAllCameras);
dom.showPastTrail?.addEventListener("change", drawAllCameras);
document.addEventListener("keydown", (event) => {
  if (event.key.toLowerCase() === "r") {
    returnToSimulation();
  }
});

const observer = new ResizeObserver(drawAllCameras);
for (const canvas of dom.canvases) {
  observer.observe(canvas);
}

applyTheme("dark");
runSimulation(true);
if (queryFlag("autoplay")) {
  playAnimation();
}
