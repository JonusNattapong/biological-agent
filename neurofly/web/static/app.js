// NeuroFly Lab: 3D Interior House & Connectome Runtime Client

// --- Global State ---
let ws = null;
let brainTopology = null;
let isPaused = false;
let cameraMode = "room"; // "room", "chase", "free"

// DOM Elements
const hudStep = document.getElementById("hud-step");
const hudBehaviorState = document.getElementById("hud-behavior-state");
const hudEnergy = document.getElementById("hud-energy");
const hudFood = document.getElementById("hud-food");
const hudCollisions = document.getElementById("hud-collisions");
const hudDistance = document.getElementById("hud-distance");
const hudActiveSpikes = document.getElementById("hud-active-spikes");
const hudSparsity = document.getElementById("hud-sparsity");
const hudNeuronTag = document.getElementById("hud-neuron-tag");
const neuropilMetersGrid = document.getElementById("neuropil-meters-grid");
const overlayPosition = document.getElementById("overlay-position");
const overlayKinematics = document.getElementById("overlay-kinematics");
const overlayHeading = document.getElementById("overlay-heading");
const overlayFoodDist = document.getElementById("overlay-food-dist");
const hudFps = document.getElementById("hud-fps");
const hudAltitudeBadge = document.getElementById("hud-altitude-badge");

// Inside Fly 05 Telemetry & Retinal Elements
const leftEyeCanvas = document.getElementById("left-eye-canvas");
const rightEyeCanvas = document.getElementById("right-eye-canvas");
const smellBarLeft = document.getElementById("smell-bar-left");
const smellBarRight = document.getElementById("smell-bar-right");
const smellValLeft = document.getElementById("smell-val-left");
const smellValRight = document.getElementById("smell-val-right");
const insideAvgV = document.getElementById("inside-avg-v");
const insideFractionFiring = document.getElementById("inside-fraction-firing");
const insideMeanRate = document.getElementById("inside-mean-rate");
const flyStatusActive = document.getElementById("fly-status-active");
const flyStatusEscaped = document.getElementById("fly-status-escaped");
const flyStatusCaught = document.getElementById("fly-status-caught");

// Raster History
const RASTER_TICKS = 100;
let rasterBuffer = [];
let rateBuffer = [];

// ==========================================================================
// 1. Three.js 3D House Room View (Matching User's Sketch)
// ==========================================================================
let roomScene, roomCamera, roomRenderer, roomControls;
let flyGroup, flyInnerGroup, flyLeftWing, flyRightWing;
let flyLeftHaltere = null, flyRightHaltere = null;
let roomTable, roomDoor, foodObjects = [];
let clickableObjects = [];
let flyTrajectoryLine, trajectoryGeometry;
let flyFloorShadow = null, flyShadowMat = null;
let ripples = [];
let targetFlyPos = new THREE.Vector3(-10, 42, -20);
let flyVelocity = new THREE.Vector3(0, 0, 0);
let targetHeading = 0.0, smoothHeading = 0.0;
let targetPitch = 0.0, smoothPitch = 0.0;
let prevFlyHeading = 0.0;
let currentBankAngle = 0.0;
let lastServerTickTime = performance.now();
let lastLoopTime = performance.now();
let fpsFrameCount = 0, lastFpsTime = performance.now();
const MAX_TRAIL_POINTS = 80;
const trailPositions = new Float32Array(MAX_TRAIL_POINTS * 3);
let trailCount = 0;

const roomContainer = document.getElementById("room-three-container");

// --------------------------------------------------------------------------
// Procedural Biological & Interior Textures
// --------------------------------------------------------------------------
function createFloorTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext("2d");

  // Rich warm wood base
  ctx.fillStyle = "#221914";
  ctx.fillRect(0, 0, 512, 512);

  ctx.strokeStyle = "#140f0c";
  ctx.lineWidth = 3;
  const plankH = 64;
  const plankW = 128;

  for (let y = 0; y < 512; y += plankH) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(512, y);
    ctx.stroke();

    const row = Math.floor(y / plankH);
    const offsetX = (row % 2 === 0) ? 0 : plankW / 2;

    for (let x = -plankW; x < 512 + plankW; x += plankW) {
      ctx.beginPath();
      ctx.moveTo(x + offsetX, y);
      ctx.lineTo(x + offsetX, y + plankH);
      ctx.stroke();

      const grainTint = (Math.sin(x * 12.3 + y * 7.1) * 0.5 + 0.5) * 15;
      ctx.fillStyle = `rgba(255, 220, 180, ${grainTint * 0.003})`;
      ctx.fillRect(x + offsetX + 1, y + 1, plankW - 2, plankH - 2);
    }
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(5, 3.5);
  return texture;
}

function createRugTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext("2d");

  ctx.fillStyle = "#1e293b";
  ctx.fillRect(0, 0, 256, 256);

  ctx.strokeStyle = "#0ea5e9";
  ctx.lineWidth = 8;
  ctx.strokeRect(12, 12, 232, 232);

  ctx.strokeStyle = "#f59e0b";
  ctx.lineWidth = 3;
  ctx.strokeRect(22, 22, 212, 212);

  ctx.fillStyle = "#0f172a";
  ctx.beginPath();
  ctx.arc(128, 128, 48, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 3;
  ctx.stroke();

  const texture = new THREE.CanvasTexture(canvas);
  return texture;
}

function createDropShadowTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 128;
  canvas.height = 128;
  const ctx = canvas.getContext("2d");
  const grad = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
  grad.addColorStop(0, "rgba(0, 0, 0, 0.75)");
  grad.addColorStop(0.35, "rgba(0, 0, 0, 0.45)");
  grad.addColorStop(0.7, "rgba(0, 0, 0, 0.12)");
  grad.addColorStop(1, "rgba(0, 0, 0, 0)");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(canvas);
}

function createClickRipple(x, y, z) {
  if (!roomScene) return;
  const ringGeo = new THREE.RingGeometry(0.8, 1.8, 24);
  const ringMat = new THREE.MeshBasicMaterial({
    color: 0x10b981,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.85,
    depthWrite: false,
  });
  const ring = new THREE.Mesh(ringGeo, ringMat);
  ring.rotation.x = -Math.PI / 2;
  ring.position.set(x, y + 0.15, z);
  roomScene.add(ring);
  ripples.push({ mesh: ring, age: 0, maxAge: 35 });
}

function createWingTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 256;
  const ctx = canvas.getContext("2d");

  // Transparent glass membrane with thin-film interference rainbow sheen
  ctx.clearRect(0, 0, 512, 256);

  // Soft iridescent base gradient
  const sheen = ctx.createLinearGradient(0, 0, 512, 256);
  sheen.addColorStop(0.0, "rgba(224, 242, 254, 0.28)"); // crystal clear
  sheen.addColorStop(0.3, "rgba(56, 189, 248, 0.20)");  // cyan highlight
  sheen.addColorStop(0.55, "rgba(192, 132, 252, 0.16)"); // violet sheen
  sheen.addColorStop(0.8, "rgba(251, 191, 36, 0.18)");  // amber shimmer
  sheen.addColorStop(1.0, "rgba(167, 243, 208, 0.14)"); // soft emerald
  ctx.fillStyle = sheen;
  ctx.fillRect(0, 0, 512, 256);

  // Micro-texture stippling (corneal/cuticle reflection)
  ctx.fillStyle = "rgba(255, 255, 255, 0.04)";
  for (let i = 0; i < 1200; i++) {
    const rx = Math.random() * 512;
    const ry = Math.random() * 256;
    ctx.fillRect(rx, ry, 1.2, 1.2);
  }

  // --- Drosophila Wing Venation ---
  const primaryVein = "rgba(45, 34, 26, 0.95)";
  const secondaryVein = "rgba(75, 58, 44, 0.85)";
  const fineVein = "rgba(105, 85, 68, 0.75)";

  // 1. Costal Vein (C) - Leading anterior edge
  ctx.strokeStyle = primaryVein;
  ctx.lineWidth = 4.5;
  ctx.beginPath();
  ctx.moveTo(10, 35);
  ctx.bezierCurveTo(120, 20, 320, 25, 490, 110);
  ctx.stroke();

  // Costal fringe micro-bristles along leading edge
  ctx.strokeStyle = "rgba(35, 26, 20, 0.75)";
  ctx.lineWidth = 1.0;
  for (let x = 30; x < 460; x += 3.5) {
    const yRatio = (x - 30) / 430;
    const y = 30 - Math.sin(yRatio * Math.PI) * 10 + yRatio * 60;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + 1.5, y - 3.5);
    ctx.stroke();
  }

  // 2. Subcostal (Sc) with break at ~25% span
  ctx.strokeStyle = primaryVein;
  ctx.lineWidth = 3.0;
  ctx.beginPath();
  ctx.moveTo(15, 38);
  ctx.bezierCurveTo(70, 32, 110, 30, 135, 26);
  ctx.stroke();

  // 3. Radial Vein R1
  ctx.strokeStyle = primaryVein;
  ctx.lineWidth = 3.2;
  ctx.beginPath();
  ctx.moveTo(15, 45);
  ctx.bezierCurveTo(90, 42, 170, 35, 220, 27);
  ctx.stroke();

  // 4. Radial Vein R2+3
  ctx.strokeStyle = secondaryVein;
  ctx.lineWidth = 2.8;
  ctx.beginPath();
  ctx.moveTo(15, 50);
  ctx.bezierCurveTo(140, 52, 330, 48, 485, 95);
  ctx.stroke();

  // 5. Radial Vein R4+5 (terminating at apex)
  ctx.strokeStyle = primaryVein;
  ctx.lineWidth = 3.0;
  ctx.beginPath();
  ctx.moveTo(20, 60);
  ctx.bezierCurveTo(150, 70, 340, 80, 500, 125);
  ctx.stroke();

  // 6. Anterior Crossvein (r-m) connecting R4+5 and M1+2
  ctx.strokeStyle = secondaryVein;
  ctx.lineWidth = 2.4;
  ctx.beginPath();
  ctx.moveTo(215, 74);
  ctx.lineTo(225, 112);
  ctx.stroke();

  // 7. Media Vein M1+2
  ctx.strokeStyle = secondaryVein;
  ctx.lineWidth = 2.6;
  ctx.beginPath();
  ctx.moveTo(22, 75);
  ctx.bezierCurveTo(140, 95, 330, 130, 470, 165);
  ctx.stroke();

  // 8. Posterior Crossvein (dm-cu)
  ctx.strokeStyle = secondaryVein;
  ctx.lineWidth = 2.4;
  ctx.beginPath();
  ctx.moveTo(350, 133);
  ctx.lineTo(365, 185);
  ctx.stroke();

  // 9. Cubitus Vein CuA1
  ctx.strokeStyle = fineVein;
  ctx.lineWidth = 2.4;
  ctx.beginPath();
  ctx.moveTo(25, 90);
  ctx.bezierCurveTo(130, 125, 250, 165, 420, 205);
  ctx.stroke();

  // 10. Anal Vein A1 + Anal lobe margin
  ctx.strokeStyle = fineVein;
  ctx.lineWidth = 2.0;
  ctx.beginPath();
  ctx.moveTo(25, 105);
  ctx.bezierCurveTo(90, 150, 160, 205, 250, 225);
  ctx.stroke();

  // Sclerotized hinge plate / basicosta at wing base
  ctx.fillStyle = "rgba(40, 28, 20, 0.9)";
  ctx.beginPath();
  ctx.ellipse(22, 65, 16, 32, -0.2, 0, Math.PI * 2);
  ctx.fill();

  const texture = new THREE.CanvasTexture(canvas);
  return texture;
}

function createAbdomenTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 512;
  const ctx = canvas.getContext("2d");

  // Warm golden-amber chitin base
  const bgGrad = ctx.createLinearGradient(0, 0, 256, 0);
  bgGrad.addColorStop(0.0, "#92400e");
  bgGrad.addColorStop(0.5, "#d97706");
  bgGrad.addColorStop(1.0, "#92400e");
  ctx.fillStyle = bgGrad;
  ctx.fillRect(0, 0, 256, 512);

  // Micro-cuticle stippling
  ctx.fillStyle = "rgba(0, 0, 0, 0.05)";
  for (let i = 0; i < 2000; i++) {
    ctx.fillRect(Math.random() * 256, Math.random() * 512, 1.5, 1.5);
  }

  // Tergites T1 to T4: Dark posterior bands with median point
  const bands = [
    { y: 55, h: 32 },
    { y: 130, h: 38 },
    { y: 215, h: 44 },
    { y: 300, h: 52 },
  ];

  ctx.fillStyle = "#18181b"; // Dark melanin
  for (let b of bands) {
    ctx.beginPath();
    ctx.moveTo(0, b.y);
    ctx.bezierCurveTo(70, b.y - 12, 186, b.y - 12, 256, b.y);
    ctx.lineTo(256, b.y + b.h);
    ctx.lineTo(0, b.y + b.h);
    ctx.closePath();
    ctx.fill();

    // Central anterior triangular projection (diagnostic for D. melanogaster)
    ctx.beginPath();
    ctx.moveTo(105, b.y - 4);
    ctx.lineTo(128, b.y - 20);
    ctx.lineTo(151, b.y - 4);
    ctx.closePath();
    ctx.fill();

    // Subtle golden highlight on anterior margin of tergite
    ctx.strokeStyle = "rgba(251, 191, 36, 0.35)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(10, b.y + b.h + 4);
    ctx.lineTo(246, b.y + b.h + 4);
    ctx.stroke();
  }

  // Tergites T5 & T6: Solid black male apical pigmentation (MaleCNS)
  ctx.fillStyle = "#0f172a";
  ctx.beginPath();
  ctx.moveTo(0, 385);
  ctx.bezierCurveTo(80, 365, 176, 365, 256, 385);
  ctx.lineTo(256, 512);
  ctx.lineTo(0, 512);
  ctx.closePath();
  ctx.fill();

  // Lateral abdominal spiracles
  ctx.fillStyle = "#09090b";
  for (let y of [75, 150, 235, 325, 410]) {
    ctx.beginPath();
    ctx.arc(22, y, 3.5, 0, Math.PI * 2);
    ctx.arc(234, y, 3.5, 0, Math.PI * 2);
    ctx.fill();
  }

  const texture = new THREE.CanvasTexture(canvas);
  return texture;
}

function createThoraxTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext("2d");

  // Tan-gold chitin base
  const bg = ctx.createLinearGradient(0, 0, 256, 256);
  bg.addColorStop(0, "#78350f");
  bg.addColorStop(0.5, "#92400e");
  bg.addColorStop(1, "#451a03");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, 256, 256);

  // 3 Subtle darker longitudinal vittae (stripes on Drosophila scutum)
  ctx.fillStyle = "rgba(24, 24, 27, 0.45)";
  ctx.fillRect(80, 20, 22, 216);
  ctx.fillRect(118, 15, 20, 226);
  ctx.fillRect(154, 20, 22, 216);

  // Fine setal stippling
  ctx.fillStyle = "rgba(0, 0, 0, 0.25)";
  for (let i = 0; i < 1500; i++) {
    ctx.fillRect(Math.random() * 256, Math.random() * 256, 1.2, 1.2);
  }

  const texture = new THREE.CanvasTexture(canvas);
  return texture;
}

function createEyeTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext("2d");

  // Deep wine/burgundy base
  ctx.fillStyle = "#4a044e";
  ctx.fillRect(0, 0, 256, 256);

  // Ruby red ommatidial lenses with specular corneal reflections
  const r = 3.2;
  const rowHeight = 6.2;
  const colWidth = 7.2;

  for (let row = 0; row < 44; row++) {
    const y = row * rowHeight + 4;
    const xOffset = (row % 2 === 0) ? 0 : colWidth / 2;
    for (let col = -1; col < 38; col++) {
      const x = col * colWidth + xOffset;

      const grad = ctx.createRadialGradient(x - 0.8, y - 0.8, 0.2, x, y, r);
      grad.addColorStop(0.0, "#f87171"); // bright corneal reflection
      grad.addColorStop(0.35, "#dc2626"); // ruby red
      grad.addColorStop(0.75, "#991b1b"); // deep crimson
      grad.addColorStop(1.0, "#450a0a"); // dark pigment cell border

      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  const texture = new THREE.CanvasTexture(canvas);
  return texture;
}

function createDrosophilaWingGeometry() {
  const shape = new THREE.Shape();
  // Start at wing hinge / axillary base
  shape.moveTo(0, 0);
  // Anterior costal margin (slight anterior bulge)
  shape.bezierCurveTo(2.5, 1.4, 6.0, 3.2, 10.0, 3.1);
  // Toward apex (curving gently)
  shape.bezierCurveTo(13.0, 2.9, 15.6, 1.6, 16.0, 0.0);
  // Rounded wing tip and posterior margin
  shape.bezierCurveTo(15.8, -1.5, 13.0, -3.2, 9.0, -3.1);
  // Posterior anal lobe (tapering back to hinge)
  shape.bezierCurveTo(4.5, -2.8, 1.8, -1.8, 0, 0);

  const geo = new THREE.ShapeGeometry(shape, 32);

  // Compute normalized UV coordinates [0..1] based on bounding box
  geo.computeBoundingBox();
  const bb = geo.boundingBox;
  const pos = geo.attributes.position;
  const uvs = new Float32Array(pos.count * 2);
  const w = bb.max.x - bb.min.x;
  const h = bb.max.y - bb.min.y;

  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const y = pos.getY(i);
    uvs[i * 2 + 0] = (x - bb.min.x) / w;
    uvs[i * 2 + 1] = 1.0 - (y - bb.min.y) / h;
  }
  geo.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
  return geo;
}

function createJointedLeg(side, pairType, legMat) {
  // side: -1 (left), +1 (right)
  // pairType: 0 (fore), 1 (mid), 2 (hind)
  const legGroup = new THREE.Group();

  let coxaPos, femurLen, tibiaLen, tarsusLen;
  let femurRot, tibiaRot, tarsusRot;

  if (pairType === 0) {
    // Foreleg: reaches forward and down
    coxaPos = new THREE.Vector3(side * 1.5, -1.2, 1.8);
    femurLen = 3.2; tibiaLen = 3.4; tarsusLen = 3.6;
    femurRot = new THREE.Euler(0.4, side * 0.35, side * 0.65);
    tibiaRot = new THREE.Euler(0.3, 0, -side * 0.4);
    tarsusRot = new THREE.Euler(0.2, 0, -side * 0.2);
  } else if (pairType === 1) {
    // Midleg: extends lateral and down
    coxaPos = new THREE.Vector3(side * 1.7, -1.3, 0.0);
    femurLen = 3.6; tibiaLen = 3.8; tarsusLen = 4.0;
    femurRot = new THREE.Euler(0.0, side * 0.1, side * 0.85);
    tibiaRot = new THREE.Euler(0.0, 0, -side * 0.55);
    tarsusRot = new THREE.Euler(0.1, 0, -side * 0.25);
  } else {
    // Hindleg: reaches backward and down
    coxaPos = new THREE.Vector3(side * 1.6, -1.2, -1.8);
    femurLen = 4.2; tibiaLen = 4.4; tarsusLen = 4.6;
    femurRot = new THREE.Euler(-0.5, -side * 0.25, side * 0.75);
    tibiaRot = new THREE.Euler(-0.4, 0, -side * 0.45);
    tarsusRot = new THREE.Euler(-0.2, 0, -side * 0.2);
  }

  legGroup.position.copy(coxaPos);

  // 1. Femur
  const femurGeo = new THREE.CylinderGeometry(0.28, 0.22, femurLen, 8);
  femurGeo.translate(0, -femurLen / 2, 0);
  const femur = new THREE.Mesh(femurGeo, legMat);
  femur.rotation.copy(femurRot);
  legGroup.add(femur);

  // 2. Tibia (connected to end of femur)
  const tibiaGeo = new THREE.CylinderGeometry(0.20, 0.15, tibiaLen, 8);
  tibiaGeo.translate(0, -tibiaLen / 2, 0);
  const tibia = new THREE.Mesh(tibiaGeo, legMat);
  tibia.position.set(0, -femurLen, 0);
  tibia.rotation.copy(tibiaRot);
  femur.add(tibia);

  // 3. Tarsus & claws (connected to end of tibia)
  const tarsusGeo = new THREE.CylinderGeometry(0.13, 0.08, tarsusLen, 8);
  tarsusGeo.translate(0, -tarsusLen / 2, 0);
  const tarsus = new THREE.Mesh(tarsusGeo, legMat);
  tarsus.position.set(0, -tibiaLen, 0);
  tarsus.rotation.copy(tarsusRot);
  tibia.add(tarsus);

  // Tiny pretarsal claw at foot tip
  const clawGeo = new THREE.ConeGeometry(0.14, 0.4, 6);
  const claw = new THREE.Mesh(clawGeo, legMat);
  claw.position.set(0, -tarsusLen, 0);
  claw.rotation.x = Math.PI / 2;
  tarsus.add(claw);

  return legGroup;
}

function createHaltere(side) {
  const group = new THREE.Group();
  const stalkMat = new THREE.MeshStandardMaterial({ color: 0x78716c, roughness: 0.5 });
  const bulbMat = new THREE.MeshStandardMaterial({ color: 0xfef08a, roughness: 0.3, emissive: 0xfef08a, emissiveIntensity: 0.25 });

  // Slender stalk (scabellum + pedicel)
  const stalkGeo = new THREE.CylinderGeometry(0.08, 0.12, 1.8, 6);
  stalkGeo.translate(0, 0.9, 0);
  const stalk = new THREE.Mesh(stalkGeo, stalkMat);
  stalk.rotation.z = side * (Math.PI / 2 + 0.3);
  stalk.rotation.x = -0.3;
  group.add(stalk);

  // Ivory/cream bulb (capitulum)
  const bulbGeo = new THREE.SphereGeometry(0.35, 12, 12);
  const bulb = new THREE.Mesh(bulbGeo, bulbMat);
  bulb.position.set(side * 1.8, 0.5, -0.4);
  group.add(bulb);

  group.position.set(side * 2.2, 0.8, -2.2);
  return group;
}

function createAntennaWithArista(side) {
  const antGroup = new THREE.Group();
  const antMat = new THREE.MeshStandardMaterial({ color: 0x57534e, roughness: 0.6 });
  const aristaMat = new THREE.MeshBasicMaterial({ color: 0x1c1917 });

  // 1. Pedicel (2nd segment)
  const pedGeo = new THREE.SphereGeometry(0.35, 10, 10);
  const ped = new THREE.Mesh(pedGeo, antMat);
  ped.position.set(0, 0, 0);
  antGroup.add(ped);

  // 2. Funiculus (3rd bulbous segment with olfactory sensilla)
  const funGeo = new THREE.SphereGeometry(0.42, 10, 10);
  funGeo.scale(1.0, 1.3, 0.9);
  const fun = new THREE.Mesh(funGeo, antMat);
  fun.position.set(0, -0.4, 0.2);
  antGroup.add(fun);

  // 3. Main Arista Stalk (curving outward and forward)
  const aristaGeo = new THREE.CylinderGeometry(0.04, 0.08, 2.4, 6);
  aristaGeo.translate(0, 1.2, 0);
  const arista = new THREE.Mesh(aristaGeo, aristaMat);
  arista.rotation.set(0.7, side * 0.45, side * 0.3);
  antGroup.add(arista);

  // 4. Feathery micro-branches on the arista (4-5 lateral branches)
  for (let b = 0; b < 5; b++) {
    const branchGeo = new THREE.CylinderGeometry(0.02, 0.03, 0.6 + b * 0.1, 4);
    branchGeo.translate(0, 0.3, 0);
    const branch = new THREE.Mesh(branchGeo, aristaMat);
    branch.position.set(side * 0.05, 0.5 + b * 0.35, 0);
    branch.rotation.set(0.4, side * 0.8, side * 0.5);
    arista.add(branch);
  }

  antGroup.position.set(side * 0.7, 0.9, 4.6);
  return antGroup;
}

function createOcelliTriangle() {
  const ocelliGroup = new THREE.Group();
  const ocellusMat = new THREE.MeshStandardMaterial({
    color: 0xef4444,
    emissive: 0xdc2626,
    emissiveIntensity: 0.6,
    roughness: 0.1,
  });

  const ocellusGeo = new THREE.SphereGeometry(0.18, 8, 8);
  // Anterior median ocellus
  const antOcellus = new THREE.Mesh(ocellusGeo, ocellusMat);
  antOcellus.position.set(0, 2.05, 3.8);
  ocelliGroup.add(antOcellus);

  // Two posterior lateral ocelli
  const lOcellus = new THREE.Mesh(ocellusGeo, ocellusMat);
  lOcellus.position.set(-0.35, 2.15, 3.35);
  ocelliGroup.add(lOcellus);

  const rOcellus = new THREE.Mesh(ocellusGeo, ocellusMat);
  rOcellus.position.set(0.35, 2.15, 3.35);
  ocelliGroup.add(rOcellus);

  return ocelliGroup;
}

function createThoracicBristles() {
  const group = new THREE.Group();
  const bristleMat = new THREE.MeshBasicMaterial({ color: 0x09090b });

  const bristlePositions = [
    // Dorsocentrals: [x, y, z, rotX, rotZ]
    [-1.2, 2.8, -0.6, -0.35, -0.2],
    [1.2, 2.8, -0.6, -0.35, 0.2],
    [-1.1, 2.7, -1.8, -0.45, -0.25],
    [1.1, 2.7, -1.8, -0.45, 0.25],
    // Scutellar bristles
    [-0.7, 2.1, -3.2, -0.55, -0.3],
    [0.7, 2.1, -3.2, -0.55, 0.3],
    [-0.3, 2.0, -3.4, -0.6, -0.1],
    [0.3, 2.0, -3.4, -0.6, 0.1],
  ];

  for (let bp of bristlePositions) {
    const bGeo = new THREE.ConeGeometry(0.06, 2.2, 5);
    bGeo.translate(0, 1.1, 0);
    const b = new THREE.Mesh(bGeo, bristleMat);
    b.position.set(bp[0], bp[1], bp[2]);
    b.rotation.set(bp[3], 0, bp[4]);
    group.add(b);
  }
  return group;
}

function createProboscis() {
  const group = new THREE.Group();
  const mouthMat = new THREE.MeshStandardMaterial({ color: 0x78350f, roughness: 0.5 });
  const labellumMat = new THREE.MeshStandardMaterial({ color: 0x92400e, roughness: 0.4 });

  // Rostrum
  const rostGeo = new THREE.CylinderGeometry(0.5, 0.4, 1.2, 8);
  rostGeo.translate(0, -0.6, 0);
  const rost = new THREE.Mesh(rostGeo, mouthMat);
  rost.rotation.x = -0.3;
  group.add(rost);

  // Labellum (sponging lobes)
  const labGeo = new THREE.SphereGeometry(0.45, 10, 10);
  labGeo.scale(1.2, 0.7, 1.0);
  const lab = new THREE.Mesh(labGeo, labellumMat);
  lab.position.set(0, -1.2, 0.2);
  group.add(lab);

  group.position.set(0, -0.8, 3.4);
  return group;
}

function initRoom3D() {
  roomScene = new THREE.Scene();
  roomScene.background = new THREE.Color(0x0a0e17);
  roomScene.fog = new THREE.FogExp2(0x0a0e17, 0.002);

  const width = roomContainer.clientWidth || 700;
  const height = roomContainer.clientHeight || 550;

  roomCamera = new THREE.PerspectiveCamera(45, width / height, 1, 1500);
  roomCamera.position.set(0, 48, 160);
  roomCamera.up.set(0, 1, 0);
  roomCamera.lookAt(0, 36, 0);

  roomRenderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  roomRenderer.setSize(width, height);
  roomRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  roomRenderer.shadowMap.enabled = true;
  roomRenderer.shadowMap.type = THREE.PCFSoftShadowMap;
  roomContainer.appendChild(roomRenderer.domElement);

  roomControls = new THREE.OrbitControls(roomCamera, roomRenderer.domElement);
  roomControls.enableDamping = true;
  roomControls.dampingFactor = 0.08;
  roomControls.target.set(0, 36, 0);
  roomControls.enabled = false;

  // --- Lighting ---
  const ambientLight = new THREE.AmbientLight(0xffedd5, 0.65);
  roomScene.add(ambientLight);

  // Warm sunlight from window
  const sunLight = new THREE.DirectionalLight(0xfff7ed, 0.95);
  sunLight.position.set(30, 95, 20);
  sunLight.castShadow = true;
  sunLight.shadow.mapSize.width = 1024;
  sunLight.shadow.mapSize.height = 1024;
  roomScene.add(sunLight);

  // Ceiling warm ambient fill
  const fillLight = new THREE.PointLight(0xfef08a, 0.45, 300);
  fillLight.position.set(-30, 65, 0);
  roomScene.add(fillLight);

  // Window blue sky tint
  const windowLight = new THREE.PointLight(0x38bdf8, 0.4, 250);
  windowLight.position.set(10, 50, -80);
  roomScene.add(windowLight);

  buildRoomGeometry();
  build3DFly();

  // Floor drop shadow for fly altitude perception
  const shadowTex = createDropShadowTexture();
  const shadowGeo = new THREE.PlaneGeometry(1, 1);
  flyShadowMat = new THREE.MeshBasicMaterial({
    map: shadowTex,
    transparent: true,
    opacity: 0.65,
    depthWrite: false,
  });
  flyFloorShadow = new THREE.Mesh(shadowGeo, flyShadowMat);
  flyFloorShadow.rotation.x = -Math.PI / 2;
  flyFloorShadow.position.set(0, 0.16, 0);
  roomScene.add(flyFloorShadow);

  // Click & hover to drop sugar interaction
  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2();

  roomRenderer.domElement.addEventListener("pointermove", (event) => {
    if (cameraMode === "free") {
      roomRenderer.domElement.style.cursor = "grab";
      return;
    }
    const rect = roomRenderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(mouse, roomCamera);
    const intersects = raycaster.intersectObjects(clickableObjects, true);
    roomRenderer.domElement.style.cursor = (intersects.length > 0) ? "crosshair" : "default";
  });

  roomRenderer.domElement.addEventListener("pointerdown", (event) => {
    if (cameraMode === "free") return;
    const rect = roomRenderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, roomCamera);
    const intersects = raycaster.intersectObjects(clickableObjects, true);
    if (intersects.length > 0) {
      const pt = intersects[0].point;
      const targetZ = pt.y > 20 ? 32.0 : 2.0; // table top vs floor
      createClickRipple(pt.x, pt.y, pt.z);
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ action: "stimulus", type: "drop_food", x: pt.x, y: pt.z, z: targetZ }));
      }
    }
  });

  window.addEventListener("resize", onRoomWindowResize);
}

function buildRoomGeometry() {
  clickableObjects = [];

  // 1. Floor with warm parquet wood texture
  const floorGeo = new THREE.PlaneGeometry(280, 180);
  const floorTex = createFloorTexture();
  const floorMat = new THREE.MeshStandardMaterial({
    map: floorTex,
    roughness: 0.45,
    metalness: 0.15,
  });
  const floor = new THREE.Mesh(floorGeo, floorMat);
  floor.rotation.x = -Math.PI / 2;
  floor.position.set(0, 0, 0);
  floor.receiveShadow = true;
  roomScene.add(floor);
  clickableObjects.push(floor);

  // Decorative woven area rug in center
  const rugGeo = new THREE.PlaneGeometry(120, 80);
  const rugTex = createRugTexture();
  const rugMat = new THREE.MeshStandardMaterial({
    map: rugTex,
    roughness: 0.85,
    metalness: 0.05,
  });
  const rug = new THREE.Mesh(rugGeo, rugMat);
  rug.rotation.x = -Math.PI / 2;
  rug.position.set(0, 0.15, 5);
  rug.receiveShadow = true;
  roomScene.add(rug);
  clickableObjects.push(rug);

  // 2. Walls (Back, Left, Right)
  const wallMat = new THREE.MeshStandardMaterial({ color: 0x181f2e, roughness: 0.88 });

  // Back Wall (Z = -85)
  const backWallGeo = new THREE.PlaneGeometry(280, 95);
  const backWall = new THREE.Mesh(backWallGeo, wallMat);
  backWall.position.set(0, 47.5, -85);
  roomScene.add(backWall);

  // Left Wall (X = -135)
  const leftWallGeo = new THREE.PlaneGeometry(180, 95);
  const leftWall = new THREE.Mesh(leftWallGeo, wallMat);
  leftWall.rotation.y = Math.PI / 2;
  leftWall.position.set(-135, 47.5, 0);
  roomScene.add(leftWall);

  // Right Wall (X = 135)
  const rightWall = new THREE.Mesh(leftWallGeo, wallMat);
  rightWall.rotation.y = -Math.PI / 2;
  rightWall.position.set(135, 47.5, 0);
  roomScene.add(rightWall);

  // Baseboard trim
  const baseboardMat = new THREE.MeshStandardMaterial({ color: 0x2d3748, roughness: 0.6 });
  const trimBack = new THREE.Mesh(new THREE.BoxGeometry(280, 3, 2), baseboardMat);
  trimBack.position.set(0, 1.5, -84);
  roomScene.add(trimBack);

  // 3. Living Room Window on Back Wall
  const windowGroup = new THREE.Group();
  const skyGeo = new THREE.PlaneGeometry(68, 48);
  const skyMat = new THREE.MeshBasicMaterial({ color: 0x60a5fa });
  const skyMesh = new THREE.Mesh(skyGeo, skyMat);
  skyMesh.position.set(10, 52, -84.8);
  windowGroup.add(skyMesh);

  const wFrameMat = new THREE.MeshStandardMaterial({ color: 0xe2e8f0, roughness: 0.3 });
  const outerFrame = new THREE.Mesh(new THREE.BoxGeometry(72, 52, 2.5), wFrameMat);
  outerFrame.position.set(10, 52, -84.2);
  windowGroup.add(outerFrame);

  const dividerH = new THREE.Mesh(new THREE.BoxGeometry(68, 2, 3), wFrameMat);
  dividerH.position.set(10, 52, -84.0);
  windowGroup.add(dividerH);
  const dividerV = new THREE.Mesh(new THREE.BoxGeometry(2, 48, 3), wFrameMat);
  dividerV.position.set(10, 52, -84.0);
  windowGroup.add(dividerV);

  // Sunlight shaft beam
  const beamGeo = new THREE.ConeGeometry(45, 110, 16, 1, true);
  const beamMat = new THREE.MeshBasicMaterial({
    color: 0xfef08a,
    transparent: true,
    opacity: 0.08,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  const beamMesh = new THREE.Mesh(beamGeo, beamMat);
  beamMesh.position.set(10, 35, -40);
  beamMesh.rotation.x = Math.PI / 4;
  windowGroup.add(beamMesh);
  roomScene.add(windowGroup);

  // 4. Table on the LEFT (Matching sketch)
  const tableGroup = new THREE.Group();
  const topGeo = new THREE.BoxGeometry(65, 3.5, 55);
  const woodMat = new THREE.MeshStandardMaterial({ color: 0x451a03, roughness: 0.45 });
  const tableTop = new THREE.Mesh(topGeo, woodMat);
  tableTop.position.set(-75, 30, 0);
  tableTop.castShadow = true;
  tableTop.receiveShadow = true;
  tableGroup.add(tableTop);
  clickableObjects.push(tableTop);

  // 4 Table Legs
  const legGeo = new THREE.CylinderGeometry(1.4, 1.4, 28.5, 12);
  const legPositions = [
    [-102, 14.25, -22],
    [-48, 14.25, -22],
    [-102, 14.25, 22],
    [-48, 14.25, 22],
  ];
  for (let lp of legPositions) {
    const leg = new THREE.Mesh(legGeo, woodMat);
    leg.position.set(lp[0], lp[1], lp[2]);
    leg.castShadow = true;
    tableGroup.add(leg);
  }

  // Ceramic Plate on Table
  const plateGeo = new THREE.CylinderGeometry(9, 7, 0.8, 24);
  const plateMat = new THREE.MeshStandardMaterial({ color: 0xf8fafc, roughness: 0.2 });
  const plate = new THREE.Mesh(plateGeo, plateMat);
  plate.position.set(-75, 32.2, 0);
  plate.receiveShadow = true;
  tableGroup.add(plate);

  // Red Apple on plate
  const appleGeo = new THREE.SphereGeometry(3.2, 16, 16);
  const appleMat = new THREE.MeshStandardMaterial({ color: 0xdc2626, roughness: 0.35 });
  const apple = new THREE.Mesh(appleGeo, appleMat);
  apple.position.set(-76, 35.0, -1.5);
  apple.castShadow = true;
  tableGroup.add(apple);

  // Orange Fruit
  const orangeGeo = new THREE.SphereGeometry(2.6, 16, 16);
  const orangeMat = new THREE.MeshStandardMaterial({ color: 0xf97316, roughness: 0.4 });
  const orange = new THREE.Mesh(orangeGeo, orangeMat);
  orange.position.set(-72, 34.4, 2.0);
  orange.castShadow = true;
  tableGroup.add(orange);

  roomScene.add(tableGroup);

  // 5. Door on the RIGHT
  const doorGroup = new THREE.Group();
  const doorGeo = new THREE.BoxGeometry(34, 76, 2);
  const doorMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.65 });
  const doorMesh = new THREE.Mesh(doorGeo, doorMat);
  doorMesh.position.set(80, 38, -84);
  doorGroup.add(doorMesh);

  // Door Trim Frame
  const frameGeo = new THREE.BoxGeometry(38, 79, 3);
  const frameMat = new THREE.MeshStandardMaterial({ color: 0x334155 });
  const frameMesh = new THREE.Mesh(frameGeo, frameMat);
  frameMesh.position.set(80, 39.5, -84.5);
  doorGroup.add(frameMesh);

  // Round Brass Door Knob
  const knobGeo = new THREE.SphereGeometry(2.2, 16, 16);
  const knobMat = new THREE.MeshStandardMaterial({ color: 0xf59e0b, metalness: 0.85, roughness: 0.2 });
  const knobMesh = new THREE.Mesh(knobGeo, knobMat);
  knobMesh.position.set(68, 36, -82);
  doorGroup.add(knobMesh);

  roomScene.add(doorGroup);

  // 6. Trajectory Ribbon
  trajectoryGeometry = new THREE.BufferGeometry();
  trajectoryGeometry.setAttribute("position", new THREE.BufferAttribute(trailPositions, 3));
  const trailMat = new THREE.LineBasicMaterial({
    color: 0x00f0ff,
    transparent: true,
    opacity: 0.75,
    linewidth: 2,
  });
  flyTrajectoryLine = new THREE.Line(trajectoryGeometry, trailMat);
  roomScene.add(flyTrajectoryLine);
}

function build3DFly() {
  flyGroup = new THREE.Group();

  // Inner group for smooth flight banking and pitch
  flyInnerGroup = new THREE.Group();
  flyGroup.add(flyInnerGroup);

  const wingTex = createWingTexture();
  const abdomenTex = createAbdomenTexture();
  const thoraxTex = createThoraxTexture();
  const eyeTex = createEyeTexture();

  // Chitin material for body parts
  const chitinMat = new THREE.MeshStandardMaterial({
    map: thoraxTex,
    roughness: 0.45,
    metalness: 0.25,
  });

  const darkChitinMat = new THREE.MeshStandardMaterial({
    color: 0x1c1917,
    roughness: 0.35,
    metalness: 0.4,
  });

  const legMat = new THREE.MeshStandardMaterial({
    color: 0x292524,
    roughness: 0.55,
    metalness: 0.25,
  });

  // 1. Thorax (Scutum + Mesothorax)
  const thoraxGeo = new THREE.SphereGeometry(3.5, 24, 24);
  const thorax = new THREE.Mesh(thoraxGeo, chitinMat);
  thorax.scale.set(1.15, 1.05, 1.35);
  thorax.castShadow = true;
  flyInnerGroup.add(thorax);

  // Scutellum: Distinct posterior triangular shield overlapping abdomen
  const scutGeo = new THREE.SphereGeometry(1.6, 16, 16);
  scutGeo.scale(1.2, 0.55, 1.1);
  const scutellum = new THREE.Mesh(scutGeo, chitinMat);
  scutellum.position.set(0, 1.5, -2.4);
  scutellum.rotation.x = -0.3;
  scutellum.castShadow = true;
  flyInnerGroup.add(scutellum);

  // Thoracic Macrochaetae (Major bristles on scutum & scutellum)
  const bristles = createThoracicBristles();
  flyInnerGroup.add(bristles);

  // 2. Abdomen (Segmented with MaleCNS dark melanin bands & black apical tip)
  const abdomenGeo = new THREE.SphereGeometry(3.6, 24, 24);
  const abdomenMat = new THREE.MeshStandardMaterial({
    map: abdomenTex,
    roughness: 0.4,
    metalness: 0.2,
  });
  const abdomen = new THREE.Mesh(abdomenGeo, abdomenMat);
  abdomen.scale.set(1.0, 0.88, 1.95);
  abdomen.position.set(0, -0.4, -4.8);
  abdomen.rotation.x = -0.18;
  abdomen.castShadow = true;
  flyInnerGroup.add(abdomen);

  // 3. Head (Broad reniform shape)
  const headGeo = new THREE.SphereGeometry(2.5, 20, 20);
  headGeo.scale(1.25, 0.95, 0.95);
  const head = new THREE.Mesh(headGeo, darkChitinMat);
  head.position.set(0, 0.4, 3.4);
  head.castShadow = true;
  flyInnerGroup.add(head);

  // 4. Ruby Compound Eyes (Anterolateral, bulging, faceted ommatidia)
  const eyeGeo = new THREE.SphereGeometry(1.9, 28, 28);
  const eyeMat = new THREE.MeshStandardMaterial({
    map: eyeTex,
    roughness: 0.12,
    metalness: 0.4,
    emissive: 0x881337,
    emissiveIntensity: 0.35,
  });

  const leftEye = new THREE.Mesh(eyeGeo, eyeMat);
  leftEye.scale.set(0.9, 1.35, 1.35);
  leftEye.position.set(-1.65, 0.75, 3.9);
  leftEye.rotation.set(0.12, -0.35, 0.15);
  flyInnerGroup.add(leftEye);

  const rightEye = new THREE.Mesh(eyeGeo, eyeMat);
  rightEye.scale.set(0.9, 1.35, 1.35);
  rightEye.position.set(1.65, 0.75, 3.9);
  rightEye.rotation.set(0.12, 0.35, -0.15);
  flyInnerGroup.add(rightEye);

  // 5. Ocellar Triangle (3 simple eyes on vertex crown)
  const ocelli = createOcelliTriangle();
  flyInnerGroup.add(ocelli);

  // 6. Antennae with Feathery Arista
  const leftAnt = createAntennaWithArista(-1);
  flyInnerGroup.add(leftAnt);
  const rightAnt = createAntennaWithArista(1);
  flyInnerGroup.add(rightAnt);

  // 7. Proboscis (Mouthparts & Labellum)
  const proboscis = createProboscis();
  flyInnerGroup.add(proboscis);

  // 8. Halteres (Dipteran Gyroscopic Balancers)
  flyLeftHaltere = createHaltere(-1);
  flyInnerGroup.add(flyLeftHaltere);
  flyRightHaltere = createHaltere(1);
  flyInnerGroup.add(flyRightHaltere);

  // 9. Delicate Veined Translucent Wings (Biological Drosophila Shape)
  const wingGeo = createDrosophilaWingGeometry();
  const wingMat = new THREE.MeshPhysicalMaterial({
    map: wingTex,
    transparent: true,
    opacity: 0.82,
    roughness: 0.08,
    transmission: 0.88,
    ior: 1.48,
    clearcoat: 1.0,
    clearcoatRoughness: 0.1,
    side: THREE.DoubleSide,
    depthWrite: false,
  });

  // Left Wing (Hinged at base, extended laterally and back)
  const leftWingGroup = new THREE.Group();
  leftWingGroup.position.set(-2.4, 1.9, -0.4);
  const leftWingMesh = new THREE.Mesh(wingGeo, wingMat);
  leftWingMesh.rotation.set(-Math.PI / 2, 0, Math.PI + 0.25);
  leftWingGroup.add(leftWingMesh);
  flyLeftWing = leftWingGroup;
  flyInnerGroup.add(flyLeftWing);

  // Right Wing
  const rightWingGroup = new THREE.Group();
  rightWingGroup.position.set(2.4, 1.9, -0.4);
  const rightWingMesh = new THREE.Mesh(wingGeo, wingMat);
  rightWingMesh.rotation.set(-Math.PI / 2, 0, -0.25);
  rightWingMesh.scale.set(-1, 1, 1); // mirror for right side
  rightWingGroup.add(rightWingMesh);
  flyRightWing = rightWingGroup;
  flyInnerGroup.add(flyRightWing);

  // 10. Jointed Articulated Legs (6 legs, 3 pairs)
  // Forelegs (pair 0)
  flyInnerGroup.add(createJointedLeg(-1, 0, legMat));
  flyInnerGroup.add(createJointedLeg(1, 0, legMat));
  // Midlegs (pair 1)
  flyInnerGroup.add(createJointedLeg(-1, 1, legMat));
  flyInnerGroup.add(createJointedLeg(1, 1, legMat));
  // Hindlegs (pair 2)
  flyInnerGroup.add(createJointedLeg(-1, 2, legMat));
  flyInnerGroup.add(createJointedLeg(1, 2, legMat));

  flyGroup.position.set(-10, 42, -20);
  roomScene.add(flyGroup);
}

function onRoomWindowResize() {
  if (!roomContainer || !roomCamera || !roomRenderer) return;
  const w = roomContainer.clientWidth;
  const h = roomContainer.clientHeight;
  roomCamera.aspect = w / h;
  roomCamera.updateProjectionMatrix();
  roomRenderer.setSize(w, h);
}

function updateFoodObjects(foods) {
  if (foodObjects.length === 0) {
    for (let f of foods) {
      const fGroup = new THREE.Group();

      const coreGeo = new THREE.SphereGeometry(3.5, 16, 16);
      const coreMat = new THREE.MeshStandardMaterial({
        color: f.name.includes("Fruit") ? 0xef4444 : 0x10b981,
        roughness: 0.3,
        emissive: 0x10b981,
        emissiveIntensity: 0.15,
      });
      const core = new THREE.Mesh(coreGeo, coreMat);
      fGroup.add(core);

      const haloGeo = new THREE.RingGeometry(4, 9, 16);
      const haloMat = new THREE.MeshBasicMaterial({
        color: 0x10b981,
        transparent: true,
        opacity: 0.35,
        side: THREE.DoubleSide,
      });
      const halo = new THREE.Mesh(haloGeo, haloMat);
      halo.position.z = 0.5;
      fGroup.add(halo);

      fGroup.position.set(f.x, f.z, f.y);
      roomScene.add(fGroup);

      // Soft drop shadow for food
      const shadowGeo = new THREE.PlaneGeometry(1, 1);
      const shadowMat = new THREE.MeshBasicMaterial({
        map: createDropShadowTexture(),
        transparent: true,
        opacity: 0.55,
        depthWrite: false,
      });
      const shadowMesh = new THREE.Mesh(shadowGeo, shadowMat);
      shadowMesh.rotation.x = -Math.PI / 2;
      const surfaceY = f.z > 20 ? 31.8 : 0.15;
      shadowMesh.position.set(f.x, surfaceY, f.y);
      shadowMesh.scale.set(9.0, 9.0, 1.0);
      roomScene.add(shadowMesh);

      foodObjects.push({ group: fGroup, shadow: shadowMesh, id: f.id });
    }
  } else {
    for (let i = 0; i < foods.length; i++) {
      if (foodObjects[i]) {
        foodObjects[i].group.position.set(foods[i].x, foods[i].z, foods[i].y);
        foodObjects[i].group.visible = !foods[i].consumed;
        if (foodObjects[i].shadow) {
          const surfaceY = foods[i].z > 20 ? 31.8 : 0.15;
          foodObjects[i].shadow.position.set(foods[i].x, surfaceY, foods[i].y);
          foodObjects[i].shadow.visible = !foods[i].consumed;
        }
      }
    }
  }
}

function updateRoomFly(flyState, trajectory) {
  if (!flyGroup) return;

  // Update target coordinates (Three.js coordinates: X=flyState.x, Y=flyState.z, Z=flyState.y)
  targetFlyPos.set(flyState.x, flyState.z, flyState.y);
  targetHeading = flyState.heading;
  targetPitch = flyState.pitch || 0;

  if (flyState.vx !== undefined && flyState.vy !== undefined && flyState.vz !== undefined) {
    flyVelocity.set(flyState.vx, flyState.vz, flyState.vy);
  } else {
    const fwd = flyState.speed || 3.5;
    const h = flyState.heading || 0;
    flyVelocity.set(Math.cos(h) * fwd, 0, Math.sin(h) * fwd);
  }
  lastServerTickTime = performance.now();

  // Update 3D Trajectory Ribbon (X, Y=z, Z=y)
  if (trajectory && trajectory.length > 1) {
    const pts = trajectory.slice(-MAX_TRAIL_POINTS);
    for (let i = 0; i < pts.length; i++) {
      trailPositions[i * 3 + 0] = pts[i][0];
      trailPositions[i * 3 + 1] = pts[i][2] || 38.0;
      trailPositions[i * 3 + 2] = pts[i][1];
    }
    trajectoryGeometry.setDrawRange(0, pts.length);
    trajectoryGeometry.attributes.position.needsUpdate = true;
  }

  // Telemetry HUD readouts
  overlayPosition.textContent = `X: ${flyState.x.toFixed(1)} | Y: ${flyState.y.toFixed(1)} | Z: ${flyState.z.toFixed(1)} cm`;
  const deg = (((flyState.heading * 180) / Math.PI) % 360).toFixed(1);
  overlayHeading.textContent = `Yaw: ${deg}° | Pitch: ${((flyState.pitch || 0) * 57.3).toFixed(1)}°`;

  if (hudAltitudeBadge) {
    hudAltitudeBadge.textContent = `ALT: ${flyState.z.toFixed(1)} cm`;
  }

  // Nearest food distance calculation
  if (overlayFoodDist && foodObjects.length > 0) {
    let minDist = 9999;
    for (let f of foodObjects) {
      if (f.group.visible) {
        const dx = flyState.x - f.group.position.x;
        const dy = flyState.z - f.group.position.y;
        const dz = flyState.y - f.group.position.z;
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (dist < minDist) minDist = dist;
      }
    }
    if (minDist < 9000) {
      overlayFoodDist.textContent = `Nearest: ${minDist.toFixed(1)} cm`;
    } else {
      overlayFoodDist.textContent = "Consumed";
    }
  }
}

// --------------------------------------------------------------------------
// Inside Fly 05: Ommatidia Retinal Pixel Renderer
// --------------------------------------------------------------------------
function renderEyeCanvas(canvas, pixels) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const cw = canvas.width;
  const ch = canvas.height;
  const cellSize = cw / 8;

  ctx.fillStyle = "#020408";
  ctx.fillRect(0, 0, cw, ch);

  if (!pixels || pixels.length < 64) return;

  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const val = Math.min(1.0, Math.max(0.0, pixels[r * 8 + c]));
      let rCol, gCol, bCol;
      if (val < 0.2) {
        // Ambient room dark cyan
        rCol = Math.floor(10 + val * 30);
        gCol = Math.floor(25 + val * 120);
        bCol = Math.floor(40 + val * 180);
      } else if (val < 0.6) {
        // Emerald / phosphor green
        rCol = Math.floor(20 + (val - 0.2) * 100);
        gCol = Math.floor(120 + (val - 0.2) * 250);
        bCol = Math.floor(140 - (val - 0.2) * 80);
      } else {
        // Sunlight / warm golden food glow
        rCol = Math.floor(220 + (val - 0.6) * 80);
        gCol = Math.floor(200 + (val - 0.6) * 130);
        bCol = Math.floor(80 + (val - 0.6) * 100);
      }

      ctx.fillStyle = `rgb(${rCol}, ${gCol}, ${bCol})`;
      ctx.fillRect(c * cellSize + 0.5, r * cellSize + 0.5, cellSize - 1, cellSize - 1);
    }
  }
}

// --------------------------------------------------------------------------
// Inside Fly 05: Real-time Rolling Sparkline Waveforms
// --------------------------------------------------------------------------
const sparklineHistory = [];
const MAX_SPARK_POINTS = 60;

function renderInsideSparkline(smellL, smellR, fractionFiring) {
  const canvas = document.getElementById("inside-sparkline-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  sparklineHistory.push({ l: smellL, r: smellR, ff: fractionFiring });
  if (sparklineHistory.length > MAX_SPARK_POINTS) sparklineHistory.shift();

  ctx.fillStyle = "#04060a";
  ctx.fillRect(0, 0, w, h);

  // Center subtle baseline
  ctx.strokeStyle = "rgba(255, 255, 255, 0.06)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2);
  ctx.stroke();

  if (sparklineHistory.length < 2) return;
  const dx = w / (MAX_SPARK_POINTS - 1);

  // 1. Left Smell (Emerald solid line)
  ctx.strokeStyle = "#10b981";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < sparklineHistory.length; i++) {
    const x = i * dx;
    const y = h - (sparklineHistory[i].l * (h - 4)) - 2;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // 2. Right Smell (Mint dashed line)
  ctx.strokeStyle = "#34d399";
  ctx.lineWidth = 1.2;
  ctx.setLineDash([3, 2]);
  ctx.beginPath();
  for (let i = 0; i < sparklineHistory.length; i++) {
    const x = i * dx;
    const y = h - (sparklineHistory[i].r * (h - 4)) - 2;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.setLineDash([]);

  // 3. Spiking Fraction % (Cyan line)
  ctx.strokeStyle = "#00f0ff";
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  for (let i = 0; i < sparklineHistory.length; i++) {
    const x = i * dx;
    const y = h - (Math.min(10.0, sparklineHistory[i].ff) / 10.0 * (h - 4)) - 2;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function animateRoomLoop() {
  requestAnimationFrame(animateRoomLoop);

  const now = performance.now();
  const dt = Math.min(0.05, Math.max(0.001, (now - lastLoopTime) * 0.001));
  lastLoopTime = now;

  // Measure real-time rendering FPS
  fpsFrameCount++;
  if (now - lastFpsTime >= 500) {
    const currentFps = Math.round((fpsFrameCount * 1000) / (now - lastFpsTime));
    if (hudFps) hudFps.textContent = currentFps;
    fpsFrameCount = 0;
    lastFpsTime = now;
  }

  // Smooth biological flight interpolation & kinematics
  if (flyGroup) {
    // 1. Continuous velocity integration + frame-rate independent exponential smoothing
    // Extrapolate predicted target slightly along velocity vector between ticks to avoid deceleration stutter
    const timeSinceTick = (now - lastServerTickTime) * 0.001;
    const blendFactor = 1.0 - Math.exp(-14.0 * dt);

    const extraTime = Math.min(0.045, timeSinceTick);
    const predictedX = targetFlyPos.x + flyVelocity.x * extraTime * 0.75;
    const predictedY = targetFlyPos.y + flyVelocity.y * extraTime * 0.75;
    const predictedZ = targetFlyPos.z + flyVelocity.z * extraTime * 0.75;

    flyGroup.position.x += (predictedX - flyGroup.position.x) * blendFactor;
    flyGroup.position.y += (predictedY - flyGroup.position.y) * blendFactor;
    flyGroup.position.z += (predictedZ - flyGroup.position.z) * blendFactor;

    // 2. Shortest-arc smooth angular interpolation for yaw & pitch
    let dHeading = ((targetHeading - smoothHeading + Math.PI) % (2 * Math.PI)) - Math.PI;
    const headingBlend = 1.0 - Math.exp(-12.0 * dt);
    smoothHeading += dHeading * headingBlend;

    let dPitch = targetPitch - smoothPitch;
    const pitchBlend = 1.0 - Math.exp(-12.0 * dt);
    smoothPitch += dPitch * pitchBlend;

    // Orient fly smoothly forward along its flight vector
    const lookDist = 12.0;
    const targetX = flyGroup.position.x + Math.cos(smoothHeading) * lookDist;
    const targetY = flyGroup.position.y + smoothPitch * 8.0;
    const targetZ = flyGroup.position.z + Math.sin(smoothHeading) * lookDist;
    flyGroup.lookAt(targetX, targetY, targetZ);

    // Dynamic bank roll based on angular turning rate
    const turnRate = dHeading * 20.0;
    const targetBank = -THREE.MathUtils.clamp(turnRate * 1.6, -0.65, 0.65);
    currentBankAngle = THREE.MathUtils.lerp(currentBankAngle, targetBank, 1.0 - Math.exp(-10.0 * dt));
    if (flyInnerGroup) {
      flyInnerGroup.rotation.z = currentBankAngle;
    }

    // Biological wing flapping oscillation with stroke & pitch torsion
    const flutter = Math.sin(now * 0.09) * 0.48;
    if (flyLeftWing) {
      flyLeftWing.rotation.y = flutter;
      flyLeftWing.rotation.z = Math.sin(now * 0.09) * 0.12;
    }
    if (flyRightWing) {
      flyRightWing.rotation.y = -flutter;
      flyRightWing.rotation.z = -Math.sin(now * 0.09) * 0.12;
    }

    // Dipteran halteres gyroscopic anti-phase oscillation
    if (flyLeftHaltere) flyLeftHaltere.rotation.x = -flutter * 0.8;
    if (flyRightHaltere) flyRightHaltere.rotation.x = flutter * 0.8;

    // Floor drop shadow follow & scale with altitude
    if (flyFloorShadow && flyShadowMat) {
      const altitude = Math.max(1.0, flyGroup.position.y);
      const shadowScale = Math.min(22.0, 7.0 + altitude * 0.16);
      flyFloorShadow.scale.set(shadowScale, shadowScale, 1.0);
      flyFloorShadow.position.set(flyGroup.position.x, 0.16, flyGroup.position.z);
      flyShadowMat.opacity = Math.max(0.12, 0.75 - altitude * 0.007);
    }

    // 3. Camera tracking modes running at native 60+ FPS
    if (cameraMode === "chase") {
      const chaseDistance = 46.0;
      const chaseHeight = 16.0;
      const desiredCamX = flyGroup.position.x - Math.cos(smoothHeading) * chaseDistance;
      const desiredCamY = flyGroup.position.y + chaseHeight;
      const desiredCamZ = flyGroup.position.z - Math.sin(smoothHeading) * chaseDistance;

      const camBlend = 1.0 - Math.exp(-9.0 * dt);
      roomCamera.position.x += (desiredCamX - roomCamera.position.x) * camBlend;
      roomCamera.position.y += (desiredCamY - roomCamera.position.y) * camBlend;
      roomCamera.position.z += (desiredCamZ - roomCamera.position.z) * camBlend;
      roomCamera.lookAt(flyGroup.position.x, flyGroup.position.y + 3.8, flyGroup.position.z);
    } else if (cameraMode === "pov") {
      const headPos = new THREE.Vector3(
        flyGroup.position.x + Math.cos(smoothHeading) * 4.2,
        flyGroup.position.y + 1.2,
        flyGroup.position.z + Math.sin(smoothHeading) * 4.2
      );
      const forward = new THREE.Vector3(
        Math.cos(smoothHeading) * 35.0,
        smoothPitch * 14.0,
        Math.sin(smoothHeading) * 35.0
      );
      roomCamera.position.copy(headPos);
      roomCamera.lookAt(headPos.clone().add(forward));
    }
  }

  // Update click ripples
  for (let i = ripples.length - 1; i >= 0; i--) {
    const r = ripples[i];
    r.age++;
    const progress = r.age / r.maxAge;
    const s = 1.0 + progress * 6.5;
    r.mesh.scale.set(s, s, 1.0);
    r.mesh.material.opacity = (1.0 - progress) * 0.85;
    if (r.age >= r.maxAge) {
      roomScene.remove(r.mesh);
      r.mesh.geometry.dispose();
      r.mesh.material.dispose();
      ripples.splice(i, 1);
    }
  }

  if (roomControls && cameraMode === "free") {
    roomControls.update();
  }
  if (roomRenderer && roomScene && roomCamera) {
    roomRenderer.render(roomScene, roomCamera);
  }
}

// ==========================================================================
// 2. Three.js 3D Connectome Hologram (Right Panel)
// ==========================================================================
let connectomeScene, connectomeCamera, connectomeRenderer, connectomeControls;
let brainPointCloud = null, synapticLines = null;
let spikeGlowAttribute = null, dimAttribute = null;
let currentNeuropilFilter = "ALL";
const connectomeContainer = document.getElementById("connectome-three-container");

const NEUROPIL_METRICS = {
  OPTIC_LOBE_LEFT: { name: "L-Optic Lobe (Medulla/LPTC)", color: "#00f0ff" },
  OPTIC_LOBE_RIGHT: { name: "R-Optic Lobe (Medulla/LPTC)", color: "#00f0ff" },
  ANTENNAL_LOBE_LEFT: { name: "L-Antennal Lobe (ORN/PN)", color: "#10b981" },
  ANTENNAL_LOBE_RIGHT: { name: "R-Antennal Lobe (ORN/PN)", color: "#10b981" },
  CENTRAL_COMPLEX: { name: "Central Complex (Compass/CX)", color: "#f59e0b" },
  MUSHROOM_BODY: { name: "Mushroom Body (Kenyon/MBON)", color: "#a855f7" },
  DESCENDING_MOTOR: { name: "Descending Motor (DNs)", color: "#f43f5e" },
  INTERNEURON: { name: "Subesophageal Interneurons", color: "#64748b" },
};

function createParticleTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 64;
  const ctx = canvas.getContext("2d");
  const grad = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, "rgba(255, 255, 255, 1.0)");
  grad.addColorStop(0.25, "rgba(255, 255, 255, 0.85)");
  grad.addColorStop(0.55, "rgba(255, 255, 255, 0.35)");
  grad.addColorStop(0.85, "rgba(255, 255, 255, 0.08)");
  grad.addColorStop(1, "rgba(255, 255, 255, 0)");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(canvas);
}

function initConnectome3D() {
  connectomeScene = new THREE.Scene();
  connectomeScene.background = new THREE.Color(0x030509);
  connectomeScene.fog = new THREE.FogExp2(0x030509, 0.0012);

  const width = connectomeContainer.clientWidth || 520;
  const height = connectomeContainer.clientHeight || 280;

  connectomeCamera = new THREE.PerspectiveCamera(40, width / height, 1, 3000);
  connectomeCamera.position.set(0, 70, 480);

  connectomeRenderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  connectomeRenderer.setSize(width, height);
  connectomeRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  connectomeContainer.appendChild(connectomeRenderer.domElement);

  connectomeControls = new THREE.OrbitControls(connectomeCamera, connectomeRenderer.domElement);
  connectomeControls.enableDamping = true;
  connectomeControls.dampingFactor = 0.06;
  connectomeControls.autoRotate = true;
  connectomeControls.autoRotateSpeed = 0.45;

  const gridHelper = new THREE.GridHelper(380, 12, 0x1e293b, 0x0f172a);
  gridHelper.position.y = -140;
  connectomeScene.add(gridHelper);

  // Wire connectome toolbar controls
  const btnAutoRotate = document.getElementById("btn-autorotate-brain");
  if (btnAutoRotate) {
    btnAutoRotate.classList.add("active");
    btnAutoRotate.addEventListener("click", () => {
      connectomeControls.autoRotate = !connectomeControls.autoRotate;
      btnAutoRotate.classList.toggle("active", connectomeControls.autoRotate);
    });
  }

  const btnResetCam = document.getElementById("btn-reset-brain-cam");
  if (btnResetCam) {
    btnResetCam.addEventListener("click", () => {
      connectomeCamera.position.set(0, 70, 480);
      connectomeControls.target.set(0, 0, 0);
      connectomeControls.update();
    });
  }

  // Neuropil isolation filter buttons
  document.querySelectorAll(".np-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      document.querySelectorAll(".np-chip").forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      applyNeuropilFilter(chip.getAttribute("data-neuropil"));
    });
  });

  window.addEventListener("resize", onConnectomeResize);
}

function onConnectomeResize() {
  if (!connectomeContainer || !connectomeCamera || !connectomeRenderer) return;
  const w = connectomeContainer.clientWidth;
  const h = connectomeContainer.clientHeight;
  connectomeCamera.aspect = w / h;
  connectomeCamera.updateProjectionMatrix();
  connectomeRenderer.setSize(w, h);
}

function buildConnectome3D(topology) {
  if (brainPointCloud) connectomeScene.remove(brainPointCloud);
  if (synapticLines) connectomeScene.remove(synapticLines);

  const coords = topology.coordinates;
  const neuropils = topology.neuropils;
  const tracts = topology.synaptic_tracts || [];
  const numNeurons = coords.length;

  hudNeuronTag.textContent = `${numNeurons.toLocaleString()} NEURONS // ${topology.num_synapses.toLocaleString()} SYNAPSES`;

  const positions = new Float32Array(numNeurons * 3);
  const colors = new Float32Array(numNeurons * 3);
  const sizes = new Float32Array(numNeurons);
  const glows = new Float32Array(numNeurons);
  const dims = new Float32Array(numNeurons);

  for (let i = 0; i < numNeurons; i++) {
    positions[i * 3 + 0] = coords[i][0];
    positions[i * 3 + 1] = coords[i][1];
    positions[i * 3 + 2] = coords[i][2];

    const npKey = neuropils[i] || "INTERNEURON";
    const meta = NEUROPIL_METRICS[npKey] || { color: "#64748b" };
    const rgb = hexToRgb(meta.color);

    colors[i * 3 + 0] = rgb.r / 255;
    colors[i * 3 + 1] = rgb.g / 255;
    colors[i * 3 + 2] = rgb.b / 255;

    sizes[i] = 4.8;
    glows[i] = 0.0;
    dims[i] = 1.0;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("aColor", new THREE.BufferAttribute(colors, 3));
  geometry.setAttribute("aSize", new THREE.BufferAttribute(sizes, 1));
  spikeGlowAttribute = new THREE.BufferAttribute(glows, 1);
  geometry.setAttribute("aSpikeGlow", spikeGlowAttribute);
  dimAttribute = new THREE.BufferAttribute(dims, 1);
  geometry.setAttribute("aDim", dimAttribute);

  // Dynamic biological spiking shader: expansion from 4.8px to 15px + intense white-hot luminescence
  const brainShaderMaterial = new THREE.ShaderMaterial({
    uniforms: {
      pointTexture: { value: createParticleTexture() },
    },
    vertexShader: `
      attribute vec3 aColor;
      attribute float aSize;
      attribute float aSpikeGlow;
      attribute float aDim;
      varying vec3 vColor;
      varying float vGlow;
      void main() {
        vGlow = aSpikeGlow;
        // Dimming for isolated neuropils
        vec3 base = mix(vec3(0.04, 0.06, 0.10), aColor, aDim);
        // Biological spike flash: white-hot core
        if (aSpikeGlow > 0.01) {
          base = mix(base, vec3(1.0, 1.0, 1.0), clamp(aSpikeGlow * 1.15, 0.0, 1.0));
        }
        vColor = base;
        vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
        // Dynamic expansion with distance perspective
        float pSize = (aSize + aSpikeGlow * 10.5) * (360.0 / -mvPos.z);
        gl_PointSize = clamp(pSize, 2.0, 38.0);
        gl_Position = projectionMatrix * mvPos;
      }
    `,
    fragmentShader: `
      uniform sampler2D pointTexture;
      varying vec3 vColor;
      varying float vGlow;
      void main() {
        vec4 tex = texture2D(pointTexture, gl_PointCoord);
        if (tex.a < 0.04) discard;
        gl_FragColor = vec4(vColor, tex.a * (0.85 + vGlow * 0.15));
      }
    `,
    transparent: true,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });

  brainPointCloud = new THREE.Points(geometry, brainShaderMaterial);
  connectomeScene.add(brainPointCloud);

  // Synaptic tracts with stratified LOD and dual-color gradient
  if (tracts.length >= 2 && tracts[0].length > 0) {
    const posts = tracts[0];
    const pres = tracts[1];
    const lineCount = Math.min(posts.length, 1200);
    const linePositions = new Float32Array(lineCount * 2 * 3);
    const lineColors = new Float32Array(lineCount * 2 * 3);

    for (let e = 0; e < lineCount; e++) {
      const p1 = pres[e];
      const p2 = posts[e];
      if (p1 < numNeurons && p2 < numNeurons) {
        linePositions[e * 6 + 0] = coords[p1][0];
        linePositions[e * 6 + 1] = coords[p1][1];
        linePositions[e * 6 + 2] = coords[p1][2];

        linePositions[e * 6 + 3] = coords[p2][0];
        linePositions[e * 6 + 4] = coords[p2][1];
        linePositions[e * 6 + 5] = coords[p2][2];

        lineColors[e * 6 + 0] = 0.0;
        lineColors[e * 6 + 1] = 0.94;
        lineColors[e * 6 + 2] = 1.0;

        lineColors[e * 6 + 3] = 0.96;
        lineColors[e * 6 + 4] = 0.62;
        lineColors[e * 6 + 5] = 0.07;
      }
    }

    const lineGeom = new THREE.BufferGeometry();
    lineGeom.setAttribute("position", new THREE.BufferAttribute(linePositions, 3));
    lineGeom.setAttribute("color", new THREE.BufferAttribute(lineColors, 3));
    const lineMat = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.16,
      blending: THREE.AdditiveBlending,
    });
    synapticLines = new THREE.LineSegments(lineGeom, lineMat);
    connectomeScene.add(synapticLines);
  }

  if (currentNeuropilFilter !== "ALL") {
    applyNeuropilFilter(currentNeuropilFilter);
  }
}

function applyNeuropilFilter(filterKey) {
  currentNeuropilFilter = filterKey;
  if (!dimAttribute || !brainTopology) return;
  const dims = dimAttribute.array;
  const neuropils = brainTopology.neuropils;

  for (let i = 0; i < dims.length; i++) {
    const np = neuropils[i] || "";
    if (filterKey === "ALL") {
      dims[i] = 1.0;
    } else if (filterKey === "OPTIC" && np.includes("OPTIC")) {
      dims[i] = 1.0;
    } else if (filterKey === "ANTENNAL" && np.includes("ANTENNAL")) {
      dims[i] = 1.0;
    } else if (filterKey === "CENTRAL_COMPLEX" && np.includes("CENTRAL_COMPLEX")) {
      dims[i] = 1.0;
    } else if (filterKey === "MUSHROOM_BODY" && np.includes("MUSHROOM_BODY")) {
      dims[i] = 1.0;
    } else if (filterKey === "DESCENDING_MOTOR" && (np.includes("DESCENDING") || np.includes("MOTOR"))) {
      dims[i] = 1.0;
    } else {
      dims[i] = 0.12;
    }
  }
  dimAttribute.needsUpdate = true;

  if (synapticLines) {
    synapticLines.material.opacity = (filterKey === "ALL") ? 0.16 : 0.05;
  }
}

function updateBrainLuminescence(activeSpikes) {
  if (!spikeGlowAttribute || !activeSpikes) return;
  const glow = spikeGlowAttribute.array;
  for (let idx of activeSpikes) {
    if (idx < glow.length) {
      glow[idx] = 1.0;
    }
  }
  spikeGlowAttribute.needsUpdate = true;
}

function animateConnectomeLoop() {
  requestAnimationFrame(animateConnectomeLoop);
  if (connectomeControls) connectomeControls.update();

  if (spikeGlowAttribute) {
    const glow = spikeGlowAttribute.array;
    let anyGlow = false;
    for (let i = 0; i < glow.length; i++) {
      if (glow[i] > 0.005) {
        glow[i] *= 0.88;
        anyGlow = true;
      }
    }
    if (anyGlow) {
      spikeGlowAttribute.needsUpdate = true;
    }
  }

  if (connectomeRenderer && connectomeScene && connectomeCamera) {
    connectomeRenderer.render(connectomeScene, connectomeCamera);
  }
}

// ==========================================================================
// 3. Oscilloscope Spike Raster
// ==========================================================================
const rasterCanvas = document.getElementById("raster-canvas");
const rasterCtx = rasterCanvas.getContext("2d");

function renderOscilloscope(activeSpikes, totalNeurons) {
  rasterBuffer.push(activeSpikes || []);
  rateBuffer.push((activeSpikes || []).length);

  if (rasterBuffer.length > RASTER_TICKS) {
    rasterBuffer.shift();
    rateBuffer.shift();
  }

  const dpr = window.devicePixelRatio || 1;
  const rect = rasterCanvas.parentElement.getBoundingClientRect();
  const w = rect.width;
  const h = rect.height;

  if (rasterCanvas.width !== w * dpr || rasterCanvas.height !== h * dpr) {
    rasterCanvas.width = w * dpr;
    rasterCanvas.height = h * dpr;
  }

  rasterCtx.save();
  rasterCtx.scale(dpr, dpr);
  rasterCtx.clearRect(0, 0, w, h);

  // Grid lines
  rasterCtx.strokeStyle = "rgba(255, 255, 255, 0.05)";
  rasterCtx.lineWidth = 1;
  const gridX = w / 4;
  for (let i = 1; i < 4; i++) {
    rasterCtx.beginPath();
    rasterCtx.moveTo(i * gridX, 0);
    rasterCtx.lineTo(i * gridX, h);
    rasterCtx.stroke();
  }
  const gridY = h / 3;
  for (let j = 1; j < 3; j++) {
    rasterCtx.beginPath();
    rasterCtx.moveTo(0, j * gridY);
    rasterCtx.lineTo(w, j * gridY);
    rasterCtx.stroke();
  }

  // Draw spikes
  const dx = w / RASTER_TICKS;
  const numSteps = rasterBuffer.length;
  rasterCtx.fillStyle = "#00f0ff";

  for (let t = 0; t < numSteps; t++) {
    const spikes = rasterBuffer[t];
    const x = t * dx;
    for (let s of spikes) {
      const y = h - (s / Math.max(1, totalNeurons)) * h;
      rasterCtx.fillRect(x, y, 1.8, 1.8);
    }
  }

  // Population rate curve
  if (rateBuffer.length > 1) {
    rasterCtx.beginPath();
    rasterCtx.strokeStyle = "rgba(245, 158, 11, 0.85)";
    rasterCtx.lineWidth = 1.5;
    const maxSpikesScale = 60;

    for (let t = 0; t < rateBuffer.length; t++) {
      const x = t * dx;
      const y = h - (rateBuffer[t] / maxSpikesScale) * (h * 0.85);
      if (t === 0) rasterCtx.moveTo(x, y);
      else rasterCtx.lineTo(x, y);
    }
    rasterCtx.stroke();
  }

  rasterCtx.restore();
}

function updateNeuropilMeters(neuropils) {
  if (!neuropils) return;
  neuropilMetersGrid.innerHTML = "";
  const CALIBRATED_MAX_HZ = 350.0;

  for (let [key, hz] of Object.entries(neuropils)) {
    const meta = NEUROPIL_METRICS[key] || { name: key, color: "#00f0ff" };
    const pct = Math.min(100, (hz / CALIBRATED_MAX_HZ) * 100);

    const row = document.createElement("div");
    row.className = "meter-row";
    row.innerHTML = `
      <span class="meter-label" title="${meta.name}">${meta.name}</span>
      <div class="meter-bar-track">
        <div class="meter-bar-fill" style="width: ${pct}%; background: ${meta.color}; box-shadow: 0 0 8px ${meta.color}66;"></div>
      </div>
      <span class="meter-value" style="color: ${meta.color};">${hz.toFixed(1)} <small style="font-size:8px;color:#64748b">Hz</small></span>
    `;
    neuropilMetersGrid.appendChild(row);
  }
}

// ==========================================================================
// 4. WebSocket Telemetry Stream Client
// ==========================================================================
function connectWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws`;

  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    console.log("[NeuroFly] Connected to 3D House Simulation Stream.");
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.type === "init") {
      brainTopology = data.brain_topology;
      if (brainTopology && brainTopology.coordinates.length > 0) {
        buildConnectome3D(brainTopology);
      }
    } else if (data.type === "tick") {
      // 1. Update 3D House Room
      if (data.arena.fly) {
        updateRoomFly(data.arena.fly, data.arena.trajectory);
      }
      if (data.arena.foods) {
        updateFoodObjects(data.arena.foods);
      }

      // 2. Update 3D Connectome & Electrophysiology
      if (data.brain) {
        updateBrainLuminescence(data.brain.active_spikes);
        renderOscilloscope(
          data.brain.active_spikes,
          brainTopology ? brainTopology.coordinates.length : 1520
        );
        updateNeuropilMeters(data.brain.neuropils);

        hudActiveSpikes.textContent = data.brain.active_count || 0;
        hudSparsity.textContent = `${data.brain.sparsity_pct || 98.5}%`;
      }

      // 3. Inside Fly 05: Sensory Retinal Vision & Olfaction
      if (data.arena && data.arena.inside_fly) {
        const inside = data.arena.inside_fly;
        renderEyeCanvas(leftEyeCanvas, inside.left_eye_pixels);
        renderEyeCanvas(rightEyeCanvas, inside.right_eye_pixels);

        const smellL = Math.min(100, Math.round((inside.smell_left || 0) * 100));
        const smellR = Math.min(100, Math.round((inside.smell_right || 0) * 100));
        if (smellBarLeft) smellBarLeft.style.width = `${smellL}%`;
        if (smellBarRight) smellBarRight.style.width = `${smellR}%`;
        if (smellValLeft) smellValLeft.textContent = (inside.smell_left || 0).toFixed(2);
        if (smellValRight) smellValRight.textContent = (inside.smell_right || 0).toFixed(2);

        // Real-time Inside Fly 05 Sparkline Waveform
        const fracFiring = (data.brain && data.brain.fraction_firing_pct != null) ? data.brain.fraction_firing_pct : 0.0;
        renderInsideSparkline(inside.smell_left || 0, inside.smell_right || 0, fracFiring);
      }

      // 4. Inside Fly 05: Brain Electrophysiology Metrics
      if (data.brain) {
        if (insideAvgV) {
          insideAvgV.innerHTML = `${(data.brain.avg_electrical_state_mv ?? -65.0).toFixed(1)} <small>mV</small>`;
        }
        if (insideFractionFiring) {
          insideFractionFiring.textContent = `${(data.brain.fraction_firing_pct ?? 0.0).toFixed(1)}%`;
        }
        if (insideMeanRate) {
          insideMeanRate.innerHTML = `${(data.brain.mean_rate_hz ?? 0.0).toFixed(1)} <small>Hz</small>`;
        }
      }

      // 5. HUD Stats
      hudStep.textContent = String(data.arena.step).padStart(6, "0");
      hudEnergy.textContent = `${Math.max(0, data.arena.fly.energy).toFixed(1)}%`;
      hudFood.textContent = data.arena.stats.food_eaten;
      if (hudCollisions) hudCollisions.textContent = data.arena.stats.collisions || 0;
      hudDistance.textContent = `${data.arena.stats.distance.toFixed(1)} mm`;
      hudBehaviorState.textContent = data.arena.behavior_state || "INTERIOR_PATROL";

      if (data.action) {
        overlayKinematics.textContent = `v: ${data.action.forward_velocity.toFixed(2)} mm/s | w: ${data.action.angular_velocity.toFixed(3)} rad/s`;
      }

      // State chip color
      if (data.arena.behavior_state === "ESCAPE_REFLEX") {
        hudBehaviorState.style.color = "#f43f5e";
      } else if (data.arena.behavior_state && (data.arena.behavior_state.includes("NUTRIENT") || data.arena.behavior_state.includes("TROPOTAXIS"))) {
        hudBehaviorState.style.color = "#10b981";
      } else {
        hudBehaviorState.style.color = "#00f0ff";
      }
    }
  };

  ws.onclose = () => {
    setTimeout(connectWebSocket, 1500);
  };
}

// ==========================================================================
// 5. User Controls & Stimulus Handlers
// ==========================================================================
const ctrlSelect = document.getElementById("controller-select");
if (ctrlSelect) {
  ctrlSelect.addEventListener("change", (e) => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: "set_controller", controller: e.target.value }));
    }
  });
}

const envSelect = document.getElementById("env-select");
if (envSelect) {
  envSelect.addEventListener("change", (e) => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: "set_env", env: e.target.value }));
    }
  });
}

// Camera segmented buttons
document.querySelectorAll("#camera-modes .seg-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#camera-modes .seg-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    cameraMode = btn.getAttribute("data-cam");
    if (roomControls) {
      roomControls.enabled = (cameraMode === "free");
      if (cameraMode === "room") {
        roomControls.target.set(0, 36, 0);
      }
    }
    if (cameraMode === "room") {
      roomCamera.position.set(0, 48, 160);
      roomCamera.up.set(0, 1, 0);
      roomCamera.lookAt(0, 36, 0);
    }
  });
});

const btnReset = document.getElementById("btn-reset");
if (btnReset) {
  btnReset.addEventListener("click", () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: "reset" }));
    }
  });
}

const btnPause = document.getElementById("btn-pause");
if (btnPause) {
  btnPause.addEventListener("click", (e) => {
    isPaused = !isPaused;
    e.currentTarget.innerHTML = isPaused
      ? `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> RESUME`
      : `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg> PAUSE`;

    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: "pause" }));
    }
  });
}

document.querySelectorAll(".command-deck .seg-btn[data-speed]").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".command-deck .seg-btn[data-speed]").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    const speed = btn.getAttribute("data-speed");
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: "set_speed", speed: parseInt(speed) }));
    }
  });
});

// Stimulus Actions
const btnFoodTable = document.getElementById("stim-food-table");
if (btnFoodTable) {
  btnFoodTable.addEventListener("click", () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: "stimulus", type: "drop_food_table" }));
    }
  });
}

const btnFoodFloor = document.getElementById("stim-food-floor");
if (btnFoodFloor) {
  btnFoodFloor.addEventListener("click", () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: "stimulus", type: "drop_food_floor" }));
    }
  });
}

const btnThreat = document.getElementById("stim-threat");
if (btnThreat) {
  btnThreat.addEventListener("click", () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: "stimulus", type: "spawn_threat" }));
    }
  });
}

const btnLightL = document.getElementById("stim-light-l");
if (btnLightL) {
  btnLightL.addEventListener("click", () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: "stimulus", type: "flash_light", side: "left" }));
    }
  });
}

const btnLightR = document.getElementById("stim-light-r");
if (btnLightR) {
  btnLightR.addEventListener("click", () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: "stimulus", type: "flash_light", side: "right" }));
    }
  });
}

const btnZap = document.getElementById("stim-zap");
if (btnZap) {
  btnZap.addEventListener("click", () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: "stimulus", type: "zap_cx" }));
    }
  });
}

// Viewport Fullscreen Toggle
const btnFullscreenRoom = document.getElementById("btn-fullscreen-room");
if (btnFullscreenRoom) {
  btnFullscreenRoom.addEventListener("click", () => {
    const panel = document.getElementById("arena-panel");
    if (panel) {
      panel.classList.toggle("fullscreen");
      btnFullscreenRoom.classList.toggle("active", panel.classList.contains("fullscreen"));
      setTimeout(onRoomWindowResize, 60);
    }
  });
}

// ==========================================================================
// Primary View Navigation & Multi-Page View Switcher (แยกหน้าจอ 5 มุมมอง)
// ==========================================================================
const mainWorkspace = document.querySelector(".main-workspace");
let activeView = "flight"; // "flight", "connectome", "inside-fly", "electrophys", "dual"
let activePanelTab = "connectome"; // "connectome", "inside-fly", "electrophys"

function switchView(viewKey) {
  if (!mainWorkspace) return;
  activeView = viewKey;
  mainWorkspace.setAttribute("data-active-view", viewKey);

  document.querySelectorAll(".view-nav-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.getAttribute("data-view") === viewKey);
  });

  // Re-adjust Three.js viewports smoothly
  setTimeout(() => {
    onRoomWindowResize();
    onConnectomeResize();
  }, 40);
  setTimeout(() => {
    onRoomWindowResize();
    onConnectomeResize();
  }, 160);
}

function switchPanelTab(panelKey) {
  activePanelTab = panelKey;
  document.querySelectorAll(".panel-tab-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.getAttribute("data-panel") === panelKey);
  });
  document.querySelectorAll(".panel-module").forEach((mod) => {
    mod.classList.toggle("active", mod.id === `panel-mod-${panelKey}`);
  });

  if (panelKey === "connectome") {
    setTimeout(onConnectomeResize, 40);
  }
}

// Bind primary view nav buttons
document.querySelectorAll(".view-nav-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const view = btn.getAttribute("data-view");
    if (view) switchView(view);
  });
});

// Bind panel sub-tab buttons
document.querySelectorAll(".panel-tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const panel = btn.getAttribute("data-panel");
    if (panel) switchPanelTab(panel);
  });
});

// Keyboard shortcuts 1-5
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.key === "1") switchView("flight");
  else if (e.key === "2") switchView("connectome");
  else if (e.key === "3") switchView("inside-fly");
  else if (e.key === "4") switchView("electrophys");
  else if (e.key === "5") switchView("dual");
});

// Bootstrap Both 3D Canvas
window.addEventListener("DOMContentLoaded", () => {
  initRoom3D();
  animateRoomLoop();

  initConnectome3D();
  animateConnectomeLoop();

  connectWebSocket();
  switchView("flight");
});

function hexToRgb(hex) {
  const clean = hex.replace("#", "");
  const num = parseInt(clean, 16);
  return {
    r: ((num >> 16) & 255) / 255,
    g: ((num >> 8) & 255) / 255,
    b: (num & 255) / 255,
  };
}
