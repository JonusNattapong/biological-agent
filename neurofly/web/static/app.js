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
let flyGroup, flyLeftWing, flyRightWing;
let roomTable, roomDoor, foodObjects = [];
let flyTrajectoryLine, trajectoryGeometry;
const MAX_TRAIL_POINTS = 80;
const trailPositions = new Float32Array(MAX_TRAIL_POINTS * 3);
let trailCount = 0;

const roomContainer = document.getElementById("room-three-container");

function initRoom3D() {
  roomScene = new THREE.Scene();
  roomScene.background = new THREE.Color(0x0a0e17);
  roomScene.fog = new THREE.FogExp2(0x0a0e17, 0.002);

  const width = roomContainer.clientWidth || 700;
  const height = roomContainer.clientHeight || 550;

  roomCamera = new THREE.PerspectiveCamera(45, width / height, 1, 1500);
  // Perfectly level horizontal camera view (0 roll tilt)
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
  const ambientLight = new THREE.AmbientLight(0xffeedd, 0.55);
  roomScene.add(ambientLight);

  // Sunlight from window / warm ceiling lamp
  const sunLight = new THREE.DirectionalLight(0xfff5e6, 0.85);
  sunLight.position.set(60, 90, 40);
  sunLight.castShadow = true;
  roomScene.add(sunLight);

  const fillLight = new THREE.PointLight(0x38bdf8, 0.35, 300);
  fillLight.position.set(0, 60, 120);
  roomScene.add(fillLight);

  buildRoomGeometry();
  build3DFly();

  window.addEventListener("resize", onRoomWindowResize);
}

function buildRoomGeometry() {
  // 1. Floor: Lies flat horizontally in XZ plane (Y = 0)
  const floorGeo = new THREE.PlaneGeometry(280, 180);
  const floorMat = new THREE.MeshStandardMaterial({
    color: 0x182030,
    roughness: 0.6,
    metalness: 0.1,
  });
  const floor = new THREE.Mesh(floorGeo, floorMat);
  floor.rotation.x = -Math.PI / 2;
  floor.position.set(0, 0, 0);
  floor.receiveShadow = true;
  roomScene.add(floor);

  // Grid on floor
  const grid = new THREE.GridHelper(260, 26, 0x334155, 0x1e293b);
  grid.position.set(0, 0.1, 0);
  roomScene.add(grid);

  // 2. Walls (Back, Left, Right) - All perfectly upright
  const wallMat = new THREE.MeshStandardMaterial({ color: 0x111624, roughness: 0.85 });

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

  // Baseboard trim along bottom of walls
  const baseboardMat = new THREE.MeshStandardMaterial({ color: 0x1e293b });
  const trimBack = new THREE.Mesh(new THREE.BoxGeometry(280, 3, 2), baseboardMat);
  trimBack.position.set(0, 1.5, -84);
  roomScene.add(trimBack);

  // 3. Table on the LEFT (Directly matching user's sketch)
  const tableGroup = new THREE.Group();
  const topGeo = new THREE.BoxGeometry(65, 3, 55);
  const woodMat = new THREE.MeshStandardMaterial({ color: 0x3d2817, roughness: 0.55 });
  const tableTop = new THREE.Mesh(topGeo, woodMat);
  tableTop.position.set(-75, 30, 0);
  tableTop.castShadow = true;
  tableTop.receiveShadow = true;
  tableGroup.add(tableTop);

  // 4 Table Legs (Vertical cylinders from Y = 0 to Y = 28.5)
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
  roomScene.add(tableGroup);

  // 4. Door on the RIGHT / Back Wall (Matching user's sketch with round knob)
  const doorGroup = new THREE.Group();
  const doorGeo = new THREE.BoxGeometry(34, 76, 2);
  const doorMat = new THREE.MeshStandardMaterial({ color: 0x1e283d, roughness: 0.7 });
  const doorMesh = new THREE.Mesh(doorGeo, doorMat);
  doorMesh.position.set(80, 38, -84);
  doorGroup.add(doorMesh);

  // Door Trim Frame
  const frameGeo = new THREE.BoxGeometry(38, 79, 3);
  const frameMat = new THREE.MeshStandardMaterial({ color: 0x334155 });
  const frameMesh = new THREE.Mesh(frameGeo, frameMat);
  frameMesh.position.set(80, 39.5, -84.5);
  doorGroup.add(frameMesh);

  // Round Door Knob (Brass sphere matching user's sketch)
  const knobGeo = new THREE.SphereGeometry(2.2, 16, 16);
  const knobMat = new THREE.MeshStandardMaterial({ color: 0xf59e0b, metalness: 0.85, roughness: 0.25 });
  const knobMesh = new THREE.Mesh(knobGeo, knobMat);
  knobMesh.position.set(68, 36, -82);
  doorGroup.add(knobMesh);

  roomScene.add(doorGroup);

  // 5. Trajectory Ribbon
  trajectoryGeometry = new THREE.BufferGeometry();
  trajectoryGeometry.setAttribute("position", new THREE.BufferAttribute(trailPositions, 3));
  const trailMat = new THREE.LineBasicMaterial({
    color: 0x00f0ff,
    transparent: true,
    opacity: 0.65,
    linewidth: 2,
  });
  flyTrajectoryLine = new THREE.Line(trajectoryGeometry, trailMat);
  roomScene.add(flyTrajectoryLine);
}

function build3DFly() {
  flyGroup = new THREE.Group();

  // Chitin material
  const bodyMat = new THREE.MeshStandardMaterial({
    color: 0x1f2937,
    roughness: 0.35,
    metalness: 0.4,
  });

  // 1. Thorax (Center)
  const thoraxGeo = new THREE.SphereGeometry(3.5, 16, 16);
  const thorax = new THREE.Mesh(thoraxGeo, bodyMat);
  thorax.scale.set(1.1, 0.9, 1.3);
  thorax.position.set(0, 0, 0);
  thorax.castShadow = true;
  flyGroup.add(thorax);

  // Abdomen (Rear: -Z)
  const abdomenGeo = new THREE.SphereGeometry(3.6, 16, 16);
  const abdomen = new THREE.Mesh(abdomenGeo, bodyMat);
  abdomen.scale.set(1.0, 0.85, 1.8);
  abdomen.position.set(0, -0.4, -4.5);
  abdomen.castShadow = true;
  flyGroup.add(abdomen);

  // 2. Head (Front: +Z)
  const headGeo = new THREE.SphereGeometry(2.4, 16, 16);
  const head = new THREE.Mesh(headGeo, bodyMat);
  head.position.set(0, 0.3, 3.4);
  head.castShadow = true;
  flyGroup.add(head);

  // 3. Two Big Compound Eyes (Prominent red/amber facets facing forward/angled)
  const eyeGeo = new THREE.SphereGeometry(1.8, 20, 20);
  const eyeMat = new THREE.MeshStandardMaterial({
    color: 0xd97706,
    roughness: 0.15,
    metalness: 0.6,
    emissive: 0xb45309,
    emissiveIntensity: 0.3,
  });

  const leftEye = new THREE.Mesh(eyeGeo, eyeMat);
  leftEye.scale.set(0.9, 1.3, 1.3);
  leftEye.position.set(-1.6, 0.7, 4.0);
  leftEye.rotation.set(0.1, -0.3, 0.1);
  flyGroup.add(leftEye);

  const rightEye = new THREE.Mesh(eyeGeo, eyeMat);
  rightEye.scale.set(0.9, 1.3, 1.3);
  rightEye.position.set(1.6, 0.7, 4.0);
  rightEye.rotation.set(0.1, 0.3, -0.1);
  flyGroup.add(rightEye);

  // Antennae
  const antMat = new THREE.MeshBasicMaterial({ color: 0x94a3b8 });
  for (let side of [-1, 1]) {
    const antGeo = new THREE.CylinderGeometry(0.1, 0.1, 2.5, 4);
    const ant = new THREE.Mesh(antGeo, antMat);
    ant.position.set(side * 0.6, 1.2, 5.0);
    ant.rotation.set(0.6, side * 0.3, 0);
    flyGroup.add(ant);
  }

  // 4. Fluttering Translucent Wings (Attached on dorsal thorax +Y, extending horizontally)
  const wingGeo = new THREE.PlaneGeometry(12.0, 5.0);
  const wingMat = new THREE.MeshPhysicalMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.6,
    roughness: 0.1,
    transmission: 0.8,
    ior: 1.4,
    side: THREE.DoubleSide,
  });

  flyLeftWing = new THREE.Mesh(wingGeo, wingMat);
  flyLeftWing.position.set(-7.0, 1.6, -0.5);
  flyLeftWing.rotation.set(-Math.PI / 2, 0, 0.2);
  flyGroup.add(flyLeftWing);

  flyRightWing = new THREE.Mesh(wingGeo, wingMat);
  flyRightWing.position.set(7.0, 1.6, -0.5);
  flyRightWing.rotation.set(-Math.PI / 2, 0, -0.2);
  flyGroup.add(flyRightWing);

  // 5. Six jointed legs angled downwards (-Y)
  const legMat = new THREE.MeshBasicMaterial({ color: 0x111827 });
  for (let side of [-1, 1]) {
    for (let i = 0; i < 3; i++) {
      const legGeo = new THREE.CylinderGeometry(0.2, 0.2, 5.5, 6);
      const leg = new THREE.Mesh(legGeo, legMat);
      leg.position.set(side * 2.8, -1.8, (i - 1) * 2.5);
      leg.rotation.set(0, 0, side * 0.7);
      flyGroup.add(leg);
    }
  }

  // Position fly initially
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
  // Synchronize 3D food props in room
  if (foodObjects.length === 0) {
    for (let f of foods) {
      const fGroup = new THREE.Group();

      // Plate / Food Core (Red apple / fruit on table)
      const coreGeo = new THREE.SphereGeometry(3.5, 16, 16);
      const coreMat = new THREE.MeshStandardMaterial({
        color: f.name.includes("Fruit") ? 0xef4444 : 0x10b981,
        roughness: 0.3,
        emissive: 0x10b981,
        emissiveIntensity: 0.15,
      });
      const core = new THREE.Mesh(coreGeo, coreMat);
      fGroup.add(core);

      // Scent Halo
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

      // Place food in Three.js: X=f.x, Y=f.z (height above floor), Z=f.y (depth)
      fGroup.position.set(f.x, f.z, f.y);
      roomScene.add(fGroup);
      foodObjects.push({ group: fGroup, id: f.id });
    }
  } else {
    for (let i = 0; i < foods.length; i++) {
      if (foodObjects[i]) {
        foodObjects[i].group.position.set(foods[i].x, foods[i].z, foods[i].y);
        foodObjects[i].group.visible = !foods[i].consumed;
      }
    }
  }
}

function updateRoomFly(flyState, trajectory) {
  if (!flyGroup) return;

  // Position in Three.js: X=fly.x, Y=fly.z (height above floor), Z=fly.y (depth)
  flyGroup.position.set(flyState.x, flyState.z, flyState.y);

  // Point fly directly forward along heading vector
  const targetX = flyState.x + Math.cos(flyState.heading) * 10;
  const targetY = flyState.z + (flyState.pitch || 0) * 8;
  const targetZ = flyState.y + Math.sin(flyState.heading) * 10;
  flyGroup.lookAt(targetX, targetY, targetZ);

  // Wing stroke flapping animation
  const flutter = Math.sin(Date.now() * 0.06) * 0.35;
  if (flyLeftWing) flyLeftWing.rotation.y = flutter;
  if (flyRightWing) flyRightWing.rotation.y = -flutter;

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

  // Camera tracking modes
  if (cameraMode === "chase") {
    const offset = new THREE.Vector3(
      -Math.cos(flyState.heading) * 45,
      16,
      -Math.sin(flyState.heading) * 45
    );
    roomCamera.position.copy(flyGroup.position).add(offset);
    roomCamera.lookAt(flyGroup.position.x, flyGroup.position.y + 4, flyGroup.position.z);
  } else if (cameraMode === "room") {
    // Level eye-level room view (0 roll tilt)
    roomCamera.position.set(0, 48, 160);
    roomCamera.up.set(0, 1, 0);
    roomCamera.lookAt(0, 36, 0);
  }

  // Telemetry overlay
  overlayPosition.textContent = `X: ${flyState.x.toFixed(1)} | Y: ${flyState.y.toFixed(1)} | Z: ${flyState.z.toFixed(1)} cm`;
  const deg = (((flyState.heading * 180) / Math.PI) % 360).toFixed(1);
  overlayHeading.textContent = `Yaw: ${deg}° | Pitch: ${((flyState.pitch || 0) * 57.3).toFixed(1)}°`;
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

function animateRoomLoop() {
  requestAnimationFrame(animateRoomLoop);
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
let brainPointCloud = null, synapticLines = null, colorAttribute = null, originalColors = null;
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
  canvas.width = 32;
  canvas.height = 32;
  const ctx = canvas.getContext("2d");
  const grad = ctx.createRadialGradient(16, 16, 0, 16, 16, 16);
  grad.addColorStop(0, "rgba(255, 255, 255, 1.0)");
  grad.addColorStop(0.3, "rgba(255, 255, 255, 0.7)");
  grad.addColorStop(0.8, "rgba(255, 255, 255, 0.15)");
  grad.addColorStop(1, "rgba(255, 255, 255, 0)");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 32, 32);
  return new THREE.CanvasTexture(canvas);
}

function initConnectome3D() {
  connectomeScene = new THREE.Scene();
  connectomeScene.background = new THREE.Color(0x030509);

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
  connectomeControls.dampingFactor = 0.08;
  connectomeControls.autoRotate = true;
  connectomeControls.autoRotateSpeed = 0.5;

  const gridHelper = new THREE.GridHelper(380, 12, 0x1e293b, 0x0f172a);
  gridHelper.position.y = -140;
  connectomeScene.add(gridHelper);

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
  originalColors = new Float32Array(numNeurons * 3);

  for (let i = 0; i < numNeurons; i++) {
    positions[i * 3] = coords[i][0];
    positions[i * 3 + 1] = coords[i][1];
    positions[i * 3 + 2] = coords[i][2];

    const npKey = neuropils[i] || "INTERNEURON";
    const meta = NEUROPIL_METRICS[npKey] || { color: "#64748b" };
    const rgb = hexToRgb(meta.color);

    colors[i * 3] = rgb.r * 0.55;
    colors[i * 3 + 1] = rgb.g * 0.55;
    colors[i * 3 + 2] = rgb.b * 0.55;

    originalColors[i * 3] = colors[i * 3];
    originalColors[i * 3 + 1] = colors[i * 3 + 1];
    originalColors[i * 3 + 2] = colors[i * 3 + 2];
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  colorAttribute = new THREE.BufferAttribute(colors, 3);
  geometry.setAttribute("color", colorAttribute);

  const pointMaterial = new THREE.PointsMaterial({
    size: 6.0,
    map: createParticleTexture(),
    vertexColors: true,
    transparent: true,
    opacity: 0.9,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });

  brainPointCloud = new THREE.Points(geometry, pointMaterial);
  connectomeScene.add(brainPointCloud);

  // Synaptic tracts
  if (tracts.length >= 2 && tracts[0].length > 0) {
    const posts = tracts[0];
    const pres = tracts[1];
    const lineCount = Math.min(posts.length, 500);
    const linePositions = new Float32Array(lineCount * 2 * 3);

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
      }
    }

    const lineGeom = new THREE.BufferGeometry();
    lineGeom.setAttribute("position", new THREE.BufferAttribute(linePositions, 3));
    const lineMat = new THREE.LineBasicMaterial({
      color: 0x00f0ff,
      transparent: true,
      opacity: 0.12,
      blending: THREE.AdditiveBlending,
    });
    synapticLines = new THREE.LineSegments(lineGeom, lineMat);
    connectomeScene.add(synapticLines);
  }
}

function updateBrainLuminescence(activeSpikes) {
  if (!colorAttribute || !originalColors) return;
  const colors = colorAttribute.array;
  const count = colors.length / 3;

  for (let i = 0; i < count * 3; i++) {
    colors[i] += (originalColors[i] - colors[i]) * 0.22;
  }

  if (activeSpikes) {
    for (let idx of activeSpikes) {
      if (idx < count) {
        colors[idx * 3] = 1.0;
        colors[idx * 3 + 1] = 1.0;
        colors[idx * 3 + 2] = 1.0;
      }
    }
  }
  colorAttribute.needsUpdate = true;
}

function animateConnectomeLoop() {
  requestAnimationFrame(animateConnectomeLoop);
  if (connectomeControls) connectomeControls.update();
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

// Bootstrap Both 3D Canvas
window.addEventListener("DOMContentLoaded", () => {
  initRoom3D();
  animateRoomLoop();

  initConnectome3D();
  animateConnectomeLoop();

  connectWebSocket();
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
