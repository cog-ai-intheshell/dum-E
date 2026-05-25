#!/usr/bin/env node
"use strict";

const childProcess = require("node:child_process");
const fs = require("node:fs");
const fsp = require("node:fs/promises");
const http = require("node:http");
const net = require("node:net");
const os = require("node:os");
const path = require("node:path");
const { pathToFileURL } = require("node:url");

const CAMERA_VIEWS = ["profile", "top", "rear", "hoop", "left", "oblique"];

function parseArgs(argv) {
  const args = {
    datasetDir: "generated_dataset",
    manifest: null,
    app: path.resolve("interface/ai-cameras.html"),
    browser: null,
    width: 1536,
    height: 960,
    scale: 1,
    limitGroups: 0,
    force: false,
    dryRun: false,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = () => {
      index += 1;
      if (index >= argv.length) {
        throw new Error(`Missing value after ${arg}`);
      }
      return argv[index];
    };

    if (arg === "--dataset-dir") args.datasetDir = next();
    else if (arg === "--manifest") args.manifest = next();
    else if (arg === "--app") args.app = next();
    else if (arg === "--browser") args.browser = next();
    else if (arg === "--width") args.width = Number(next());
    else if (arg === "--height") args.height = Number(next());
    else if (arg === "--scale") args.scale = Number(next());
    else if (arg === "--limit-groups") args.limitGroups = Number(next());
    else if (arg === "--force") args.force = true;
    else if (arg === "--dry-run") args.dryRun = true;
    else if (arg === "--help" || arg === "-h") {
      printHelp();
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  args.datasetDir = path.resolve(args.datasetDir);
  args.manifest = path.resolve(args.manifest ?? path.join(args.datasetDir, "frames_manifest.csv"));
  args.app = path.resolve(args.app);
  return args;
}

function printHelp() {
  console.log(`Usage:
  node dataset_generation/render_frames.js --dataset-dir generated_dataset_test

Options:
  --dataset-dir DIR       Directory containing frames_manifest.csv
  --manifest FILE         Explicit path to frames_manifest.csv
  --app FILE              Path to interface/ai-cameras.html
  --browser FILE          Path to Brave/Chromium/Chrome
  --width PX              Viewport width, default 1536
  --height PX             Viewport height, default 960
  --scale N               Device scale factor, default 1
  --limit-groups N        Limit rendered timestep groups, 0 = all
  --force                 Rewrite existing PNG files
  --dry-run               Print the plan without launching the browser
`);
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;

  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    const next = text[index + 1];

    if (quoted) {
      if (char === '"' && next === '"') {
        field += '"';
        index += 1;
      } else if (char === '"') {
        quoted = false;
      } else {
        field += char;
      }
    } else if (char === '"') {
      quoted = true;
    } else if (char === ",") {
      row.push(field);
      field = "";
    } else if (char === "\n") {
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else if (char !== "\r") {
      field += char;
    }
  }

  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }

  if (rows.length === 0) return [];
  const headers = rows[0];
  return rows.slice(1).filter((values) => values.length > 1).map((values) => {
    const record = {};
    headers.forEach((header, index) => {
      record[header] = values[index] ?? "";
    });
    return record;
  });
}

async function findFreePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.on("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address();
      server.close(() => resolve(port));
    });
  });
}

function findBrowser(explicitPath) {
  const candidates = [
    explicitPath,
    process.env.BROWSER_PATH,
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/usr/bin/google-chrome",
  ].filter(Boolean);

  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }

  throw new Error("Aucun navigateur Chromium/Brave trouve. Utilise --browser /chemin/vers/le/navigateur.");
}

function httpJson(url, options = {}) {
  return new Promise((resolve, reject) => {
    const request = http.request(url, { method: options.method ?? "GET" }, (response) => {
      let body = "";
      response.setEncoding("utf8");
      response.on("data", (chunk) => {
        body += chunk;
      });
      response.on("end", () => {
        if (response.statusCode < 200 || response.statusCode >= 300) {
          reject(new Error(`HTTP ${response.statusCode}: ${body.slice(0, 200)}`));
          return;
        }
        try {
          resolve(JSON.parse(body));
        } catch (error) {
          reject(error);
        }
      });
    });
    request.on("error", reject);
    request.end();
  });
}

async function waitForJson(url, timeoutMs = 10000) {
  const start = Date.now();
  let lastError = null;
  while (Date.now() - start < timeoutMs) {
    try {
      return await httpJson(url);
    } catch (error) {
      lastError = error;
      await new Promise((resolve) => setTimeout(resolve, 120));
    }
  }
  throw lastError ?? new Error(`Timeout sur ${url}`);
}

class CdpClient {
  constructor(wsUrl) {
    this.ws = new WebSocket(wsUrl);
    this.nextId = 1;
    this.pending = new Map();
    this.listeners = new Map();
    this.opened = new Promise((resolve, reject) => {
      this.ws.addEventListener("open", resolve, { once: true });
      this.ws.addEventListener("error", reject, { once: true });
    });
    this.ws.addEventListener("message", (event) => this.handleMessage(event));
  }

  handleMessage(event) {
    const data = typeof event.data === "string" ? event.data : Buffer.from(event.data).toString("utf8");
    const message = JSON.parse(data);
    if (message.id && this.pending.has(message.id)) {
      const { resolve, reject } = this.pending.get(message.id);
      this.pending.delete(message.id);
      if (message.error) reject(new Error(message.error.message));
      else resolve(message.result);
      return;
    }

    if (message.method && this.listeners.has(message.method)) {
      for (const listener of this.listeners.get(message.method)) {
        listener(message.params);
      }
    }
  }

  async send(method, params = {}) {
    await this.opened;
    const id = this.nextId;
    this.nextId += 1;
    const payload = JSON.stringify({ id, method, params });
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.ws.send(payload);
    });
  }

  once(method) {
    return new Promise((resolve) => {
      const listener = (params) => {
        const listeners = this.listeners.get(method) ?? [];
        this.listeners.set(method, listeners.filter((item) => item !== listener));
        resolve(params);
      };
      const listeners = this.listeners.get(method) ?? [];
      listeners.push(listener);
      this.listeners.set(method, listeners);
    });
  }

  close() {
    this.ws.close();
  }
}

async function createPage(port) {
  const target = await httpJson(`http://127.0.0.1:${port}/json/new?about:blank`, { method: "PUT" });
  return new CdpClient(target.webSocketDebuggerUrl);
}

function normalizeRenderQuery(renderQuery) {
  const params = new URLSearchParams(renderQuery);
  params.delete("camera");
  params.sort();
  return params.toString();
}

function groupManifestRows(rows, datasetDir, force) {
  const groups = new Map();
  for (const row of rows) {
    const absolutePath = path.resolve(datasetDir, row.relative_path);
    if (!force && fs.existsSync(absolutePath)) {
      continue;
    }
    const key = normalizeRenderQuery(row.render_query);
    if (!groups.has(key)) {
      groups.set(key, []);
    }
    groups.get(key).push({ ...row, absolutePath });
  }
  return Array.from(groups.entries()).map(([query, groupRows]) => ({ query, rows: groupRows }));
}

async function navigateAndExport(page, url) {
  const loaded = page.once("Page.loadEventFired");
  await page.send("Page.navigate", { url });
  await loaded;
  const ready = await page.send("Runtime.evaluate", {
    awaitPromise: true,
    returnByValue: true,
    expression: `
      new Promise((resolve) => {
        const deadline = Date.now() + 10000;
        function check() {
          if (window.__AI_CAMERAS_READY__ && document.querySelectorAll("canvas[data-camera]").length) {
            requestAnimationFrame(() => requestAnimationFrame(() => resolve(true)));
          } else if (Date.now() > deadline) {
            resolve(false);
          } else {
            setTimeout(check, 16);
          }
        }
        check();
      })
    `,
  });

  if (!ready.result.value) {
    throw new Error("ai-cameras.html n'a pas signale que les canvases etaient prets.");
  }

  const result = await page.send("Runtime.evaluate", {
    returnByValue: true,
    expression: `
      Array.from(document.querySelectorAll("canvas[data-camera]")).map((canvas) => ({
        camera: canvas.dataset.camera,
        width: canvas.width,
        height: canvas.height,
        dataUrl: canvas.toDataURL("image/png")
      }))
    `,
  });
  return result.result.value;
}

async function writeCanvasPng(filePath, dataUrl) {
  const base64 = dataUrl.replace(/^data:image\/png;base64,/, "");
  await fsp.mkdir(path.dirname(filePath), { recursive: true });
  await fsp.writeFile(filePath, Buffer.from(base64, "base64"));
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const manifestText = await fsp.readFile(args.manifest, "utf8");
  const rows = parseCsv(manifestText);
  const groups = groupManifestRows(rows, args.datasetDir, args.force);
  const selectedGroups = args.limitGroups > 0 ? groups.slice(0, args.limitGroups) : groups;
  const pngCount = selectedGroups.reduce((sum, group) => sum + group.rows.length, 0);

  console.log(`Manifest        : ${args.manifest}`);
  console.log(`Dataset folder  : ${args.datasetDir}`);
  console.log(`Pages to render : ${selectedGroups.length}`);
  console.log(`PNG to write    : ${pngCount}`);
  console.log(`Viewport        : ${args.width} x ${args.height} @${args.scale}`);

  if (args.dryRun || selectedGroups.length === 0) {
    return;
  }

  const browserPath = findBrowser(args.browser);
  const port = await findFreePort();
  const userDataDir = await fsp.mkdtemp(path.join(os.tmpdir(), "basquett-headless-"));
  const browser = childProcess.spawn(
    browserPath,
    [
      "--headless=new",
      "--disable-gpu",
      "--disable-extensions",
      "--disable-background-networking",
      "--hide-scrollbars",
      "--mute-audio",
      `--remote-debugging-port=${port}`,
      `--user-data-dir=${userDataDir}`,
      "about:blank",
    ],
    { stdio: ["ignore", "ignore", "pipe"] }
  );

  browser.stderr.on("data", (chunk) => {
    const text = chunk.toString();
    if (/error|failed/i.test(text)) {
      process.stderr.write(text);
    }
  });

  let page = null;
  try {
    await waitForJson(`http://127.0.0.1:${port}/json/version`);
    page = await createPage(port);
    await page.send("Page.enable");
    await page.send("Runtime.enable");
    await page.send("Emulation.setDeviceMetricsOverride", {
      width: args.width,
      height: args.height,
      deviceScaleFactor: args.scale,
      mobile: false,
    });

    const baseUrl = pathToFileURL(args.app).href;
    let written = 0;
    for (let index = 0; index < selectedGroups.length; index += 1) {
      const group = selectedGroups[index];
      const canvasData = await navigateAndExport(page, `${baseUrl}?${group.query}`);
      const byCamera = new Map(canvasData.map((item) => [item.camera, item]));

      for (const row of group.rows) {
        if (!CAMERA_VIEWS.includes(row.camera_view)) {
          throw new Error(`Vue camera inconnue: ${row.camera_view}`);
        }
        const canvas = byCamera.get(row.camera_view);
        if (!canvas) {
          throw new Error(`Canvas absent pour la vue ${row.camera_view}`);
        }
        await writeCanvasPng(row.absolutePath, canvas.dataUrl);
        written += 1;
      }

      if ((index + 1) % 10 === 0 || index + 1 === selectedGroups.length) {
        console.log(`render ${index + 1}/${selectedGroups.length} - png ${written}/${pngCount}`);
      }
    }
  } finally {
    if (page) page.close();
    browser.kill();
    await fsp.rm(userDataDir, { recursive: true, force: true });
  }
}

main().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
