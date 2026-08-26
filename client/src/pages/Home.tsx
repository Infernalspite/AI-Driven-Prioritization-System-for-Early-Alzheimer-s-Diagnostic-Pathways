/*
 * Clinical Observatory style reminder: editorial clinical data art, ink navy workspace,
 * warm parchment evidence panels, observatory indigo signals, restrained teal/amber semantics,
 * and motion that clarifies inference. Keep the hierarchy asymmetric and clinically legible.
 */
import { useEffect, useMemo, useRef, useState, type CSSProperties, type ReactNode, type PointerEvent as ReactPointerEvent, type WheelEvent as ReactWheelEvent } from "react";
import { AnimatePresence, motion, useInView } from "framer-motion";
import {
  Activity,
  ArrowDownRight,
  ArrowUpRight,
  BarChart3,
  Brain,
  CalendarDays,
  Check,
  ChevronDown,
  CircleHelp,
  ClipboardCheck,
  Clock3,
  Download,
  FlaskConical,
  Gauge,
  GitBranch,
  Info,
  LayoutDashboard,
  Menu,
  MoreHorizontal,
  Orbit,
  PanelLeftClose,
  PanelLeftOpen,
  ScanLine,
  Search,
  Settings2,
  ShieldCheck,
  Sparkles,
  Stethoscope,
  Target,
  Users,
  X,
  Zap,
} from "lucide-react";

const HERO_IMAGE = "/manus-storage/alz-hero-atmosphere_41a4656c.png";
const NEURAL_TEXTURE = "/manus-storage/alz-neural-texture_479d2057.png";
const TRAJECTORY_TEXTURE = "/manus-storage/alz-trajectory-texture_341dd4fe.png";
const BRAND_MARK = "/manus-storage/alz-brand-mark_f915e5d3.png";

const navItems = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "trajectory", label: "Trajectory", icon: Activity },
  { id: "pathway", label: "Pathway", icon: GitBranch },
  { id: "explainability", label: "Explainability", icon: Orbit },
  { id: "causal", label: "Causal analysis", icon: ScanLine },
  { id: "optimization", label: "Optimization", icon: BarChart3 },
];

const patients = [
  {
    id: "SUBJ2-0000",
    label: "Elena R.",
    age: 72,
    education: 16,
    visits: 5,
    lastSeen: "18 Aug 2026",
    diagnosis: "CN",
    diagnosisLabel: "Cognitively normal",
    confidence: 78.8,
    uncertainty: 8.4,
    score: 83.2,
    delta: -1.8,
    probabilities: { CN: 78.8, MCI: 16.7, AD: 4.5 },
    health: "stable",
    historical: [85, 84, 83, 82, 83, 82, 83],
    forecast: [83, 82, 80, 79, 77, 76, 74, 72, 70],
    range: [3, 4, 4, 5, 5, 6, 6, 7, 8],
    biomarkers: [
      { name: "MMSE", value: "28 / 30", note: "within expected range", tone: "teal" },
      { name: "Aβ42 / 40", value: "0.118", note: "above threshold", tone: "teal" },
      { name: "p-tau181", value: "15.4 pg/mL", note: "low signal", tone: "teal" },
      { name: "Hippocampal vol.", value: "6,420 mm³", note: "age-adjusted", tone: "indigo" },
    ],
    factors: [
      { label: "ADAS13", value: 82, contribution: "Protective", direction: "negative", detail: "Low cognitive symptom burden" },
      { label: "MMSE", value: 68, contribution: "Protective", direction: "negative", detail: "Strong global cognition" },
      { label: "Hippocampal volume", value: 47, contribution: "Neutral", direction: "neutral", detail: "Within age-adjusted range" },
      { label: "Aβ42 / 40 ratio", value: 31, contribution: "Protective", direction: "negative", detail: "No amyloid pattern detected" },
      { label: "Age", value: 25, contribution: "Risk", direction: "positive", detail: "Age-related prior" },
    ],
    pathway: {
      current: "Cognitive screen complete",
      recommendation: "MONITOR",
      expectedBenefit: "+0.08",
      cost: "0.0 pts",
      rationale: "Current cognitive and blood signals are concordant. A 6-month follow-up preserves sensitivity without adding imaging burden.",
    },
    causal: [
      { from: "Age", to: "Cognition", strength: 0.34, tone: "amber" },
      { from: "Education", to: "Cognition", strength: -0.28, tone: "teal" },
      { from: "Amyloid", to: "p-tau", strength: 0.22, tone: "indigo" },
    ],
  },
  {
    id: "SUBJ2-0042",
    label: "Marcus T.",
    age: 69,
    education: 14,
    visits: 4,
    lastSeen: "12 Aug 2026",
    diagnosis: "MCI",
    diagnosisLabel: "Mild cognitive impairment",
    confidence: 67.4,
    uncertainty: 14.9,
    score: 62.4,
    delta: -6.7,
    probabilities: { CN: 17.9, MCI: 67.4, AD: 14.7 },
    health: "watch",
    historical: [78, 76, 73, 70, 68, 66, 62],
    forecast: [62, 59, 56, 53, 49, 45, 42, 39, 36],
    range: [5, 6, 7, 8, 9, 10, 11, 12, 13],
    biomarkers: [
      { name: "MMSE", value: "24 / 30", note: "down 2 points", tone: "amber" },
      { name: "Aβ42 / 40", value: "0.071", note: "borderline low", tone: "amber" },
      { name: "p-tau181", value: "24.8 pg/mL", note: "elevated signal", tone: "coral" },
      { name: "Hippocampal vol.", value: "5,840 mm³", note: "mildly reduced", tone: "amber" },
    ],
    factors: [
      { label: "ADAS13", value: 89, contribution: "Risk", direction: "positive", detail: "Recent symptom acceleration" },
      { label: "MMSE", value: 76, contribution: "Risk", direction: "positive", detail: "Decline across two visits" },
      { label: "p-tau181", value: 63, contribution: "Risk", direction: "positive", detail: "Elevated phosphorylated tau" },
      { label: "Hippocampal volume", value: 52, contribution: "Risk", direction: "positive", detail: "Mild volume loss" },
      { label: "Education", value: 21, contribution: "Protective", direction: "negative", detail: "Cognitive reserve signal" },
    ],
    pathway: {
      current: "Cognitive decline detected",
      recommendation: "ORDER_BLOOD",
      expectedBenefit: "+0.21",
      cost: "1.0 pts",
      rationale: "A blood panel is the next lowest-burden action with the highest expected information gain for separating stable MCI from AD-pattern biology.",
    },
    causal: [
      { from: "Age", to: "Cognition", strength: 0.41, tone: "amber" },
      { from: "p-tau", to: "Cognition", strength: 0.68, tone: "coral" },
      { from: "Hippocampus", to: "Cognition", strength: -0.53, tone: "indigo" },
    ],
  },
  {
    id: "SUBJ2-0015",
    label: "George K.",
    age: 77,
    education: 12,
    visits: 7,
    lastSeen: "09 Aug 2026",
    diagnosis: "AD",
    diagnosisLabel: "Alzheimer's disease pattern",
    confidence: 91.6,
    uncertainty: 5.2,
    score: 31.8,
    delta: -12.3,
    probabilities: { CN: 2.1, MCI: 6.3, AD: 91.6 },
    health: "escalate",
    historical: [66, 62, 57, 51, 45, 39, 32],
    forecast: [32, 28, 24, 21, 18, 15, 13, 11, 9],
    range: [4, 5, 5, 6, 7, 8, 9, 10, 11],
    biomarkers: [
      { name: "MMSE", value: "18 / 30", note: "down 4 points", tone: "coral" },
      { name: "Aβ42 / 40", value: "0.041", note: "low amyloid ratio", tone: "coral" },
      { name: "p-tau181", value: "38.2 pg/mL", note: "high signal", tone: "coral" },
      { name: "Hippocampal vol.", value: "4,920 mm³", note: "markedly reduced", tone: "coral" },
    ],
    factors: [
      { label: "Hippocampal volume", value: 94, contribution: "Risk", direction: "positive", detail: "Marked age-adjusted loss" },
      { label: "ADAS13", value: 91, contribution: "Risk", direction: "positive", detail: "High symptom burden" },
      { label: "p-tau181", value: 83, contribution: "Risk", direction: "positive", detail: "Strong tau signal" },
      { label: "Aβ42 / 40 ratio", value: 74, contribution: "Risk", direction: "positive", detail: "Amyloid pattern confirmed" },
      { label: "MMSE", value: 69, contribution: "Risk", direction: "positive", detail: "Multi-visit decline" },
    ],
    pathway: {
      current: "Convergent AD pattern",
      recommendation: "DIAGNOSE",
      expectedBenefit: "+0.04",
      cost: "0.0 pts",
      rationale: "Cognition, blood, and structural MRI are aligned. Additional PET would add cost without materially reducing posterior uncertainty.",
    },
    causal: [
      { from: "Amyloid", to: "p-tau", strength: 0.81, tone: "coral" },
      { from: "p-tau", to: "Cognition", strength: 0.74, tone: "coral" },
      { from: "Hippocampus", to: "Cognition", strength: -0.69, tone: "indigo" },
    ],
  },
];

type Patient = (typeof patients)[number];
type SectionId = (typeof navItems)[number]["id"];

type AnimatedNumberProps = {
  value: number;
  decimals?: number;
  suffix?: string;
  duration?: number;
  className?: string;
};

function AnimatedNumber({ value, decimals = 0, suffix = "", duration = 850, className }: AnimatedNumberProps) {
  const [display, setDisplay] = useState(0);
  useEffect(() => {
    let frame = 0;
    const start = performance.now();
    const tick = (time: number) => {
      const progress = Math.min(1, (time - start) / duration);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplay(value * eased);
      if (progress < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [value, duration]);
  return <span className={className}>{display.toFixed(decimals)}{suffix}</span>;
}

function SectionReveal({ children, className = "", delay = 0 }: { children: ReactNode; className?: string; delay?: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 18 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.12 }}
      transition={{ duration: 0.62, delay, ease: [0.23, 1, 0.32, 1] }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

function ConfidenceRing({ confidence, diagnosis }: { confidence: number; diagnosis: Patient["diagnosis"] }) {
  const radius = 88;
  const circumference = 2 * Math.PI * radius;
  const stroke = "var(--indigo)";
  return (
    <div className="confidence-ring" aria-label={`${confidence.toFixed(1)} percent confidence in ${diagnosis} prediction`}>
      <svg viewBox="0 0 220 220" role="img" aria-hidden="true">
        <circle cx="110" cy="110" r={radius} className="ring-track" />
        <motion.circle
          cx="110"
          cy="110"
          r={radius}
          className="ring-progress"
          style={{ stroke }}
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: circumference * (1 - confidence / 100) }}
          transition={{ duration: 1.2, ease: [0.23, 1, 0.32, 1], delay: 0.2 }}
        />
        <circle cx="110" cy="22" r="4" className="ring-node" style={{ fill: stroke }} />
        <circle cx="110" cy="110" r="70" className="ring-inner" />
      </svg>
      <div className="ring-copy">
        <span className="eyebrow">AI confidence</span>
        <AnimatedNumber value={confidence} decimals={1} suffix="%" className="ring-value" />
        <span className="ring-caption">posterior certainty</span>
      </div>
    </div>
  );
}

type BrainRegionId = "hippocampus" | "temporal" | "frontal" | "parietal";

const brainRegions: Array<{ id: BrainRegionId; label: string; short: string; x: number; y: number; detail: string; impact: string; change: string; tone: "indigo" | "teal" | "amber" }> = [
  { id: "hippocampus", label: "Hippocampus", short: "memory", x: 337, y: 218, detail: "Memory-related region", impact: "+18%", change: "−7.2%", tone: "amber" },
  { id: "temporal", label: "Temporal lobe", short: "language", x: 392, y: 185, detail: "Learning and language", impact: "+11%", change: "−3.8%", tone: "indigo" },
  { id: "frontal", label: "Frontal regions", short: "planning", x: 182, y: 130, detail: "Planning and attention", impact: "+6%", change: "−1.4%", tone: "teal" },
  { id: "parietal", label: "Parietal regions", short: "orientation", x: 332, y: 102, detail: "Spatial orientation", impact: "+9%", change: "−2.5%", tone: "indigo" },
];

const networkNodes = [
  [96, 128], [117, 96], [145, 73], [181, 59], [221, 69], [259, 49], [300, 62], [344, 52], [390, 67], [434, 78], [478, 101], [510, 132], [540, 165], [548, 204], [527, 236], [488, 257], [445, 278], [400, 288], [356, 300], [308, 291], [265, 302], [224, 286], [186, 278], [150, 258], [121, 230], [102, 194], [173, 121], [216, 104], [263, 93], [312, 89], [363, 94], [414, 110], [462, 132], [483, 176], [463, 211], [423, 230], [374, 244], [327, 246], [282, 239], [235, 229], [190, 211], [153, 180], [142, 148], [204, 151], [253, 129], [305, 126], [355, 131], [402, 148], [428, 178], [407, 203], [362, 216], [316, 214], [270, 204], [226, 188], [186, 168],
] as const;

const networkEdges = [[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11],[11,12],[12,13],[13,14],[14,15],[15,16],[16,17],[17,18],[18,19],[19,20],[20,21],[21,22],[22,23],[23,24],[24,25],[2,26],[3,27],[4,27],[5,28],[6,29],[7,30],[8,31],[9,32],[10,33],[11,34],[12,35],[26,27],[27,28],[28,29],[29,30],[30,31],[31,32],[32,33],[33,34],[34,35],[35,36],[36,37],[37,38],[38,39],[39,40],[40,41],[41,42],[42,43],[43,44],[44,45],[45,46],[46,47],[47,48],[48,49],[49,50],[50,51],[51,52],[52,53],[53,54],[26,42],[28,44],[30,46],[32,48],[34,50],[36,52],[38,54]] as const;

function BrainExplorer({ patient, selectedRegion, onSelectRegion, onReset }: { patient: Patient; selectedRegion: BrainRegionId | null; onSelectRegion: (id: BrainRegionId) => void; onReset: () => void }) {
  const [rotation, setRotation] = useState(0);
  const [zoom, setZoom] = useState(1);
  const [dragging, setDragging] = useState(false);
  const dragStart = useRef(0);
  const dragRotation = useRef(0);
  const region = brainRegions.find((item) => item.id === selectedRegion) ?? null;
  const brainScale = (selectedRegion === "hippocampus" ? 1.12 : selectedRegion ? 1.07 : 1) * zoom;
  const brainX = selectedRegion === "hippocampus" ? -29 : selectedRegion === "temporal" ? -52 : selectedRegion === "frontal" ? 16 : selectedRegion === "parietal" ? -12 : 0;
  const brainY = selectedRegion === "hippocampus" ? -20 : selectedRegion ? -10 : 0;
  const focusRotation = selectedRegion === "hippocampus" ? -6 : selectedRegion === "temporal" ? 5 : selectedRegion === "frontal" ? -3 : selectedRegion === "parietal" ? 3 : 0;
  const handlePointerDown = (event: ReactPointerEvent<SVGSVGElement>) => { setDragging(true); dragStart.current = event.clientX; dragRotation.current = rotation; event.currentTarget.setPointerCapture(event.pointerId); };
  const handlePointerMove = (event: ReactPointerEvent<SVGSVGElement>) => { if (!dragging) return; setRotation(dragRotation.current + (event.clientX - dragStart.current) * 0.18); };
  const handlePointerUp = () => setDragging(false);
  const handleWheel = (event: ReactWheelEvent<SVGSVGElement>) => setZoom((value) => Math.min(1.16, Math.max(.88, value - event.deltaY * .0007)));
  const evidenceText = patient.diagnosis === "CN" ? "Model-indicated region · low concern" : patient.diagnosis === "MCI" ? "Model-indicated region · watch signal" : "Model-indicated region · elevated concern";
  return (
    <div className={`brain-explorer ${selectedRegion ? "brain-explorer-focused" : ""}`}>
      <div className="brain-toolbar"><div><span className="eyebrow">Neural map / 2.5D view</span><strong>{selectedRegion ? `Inspecting ${region?.label}` : "Select a region to inspect"}</strong></div><button className="brain-reset" onClick={() => { onReset(); setRotation(0); setZoom(1); }}><ScanLine size={13} /> reset view</button></div>
      <div className="brain-stage">
        <svg className="brain-svg" viewBox="0 0 620 360" role="img" aria-label="Interactive anatomical brain map. Select hippocampus, temporal, frontal, or parietal regions." onPointerDown={handlePointerDown} onPointerMove={handlePointerMove} onPointerUp={handlePointerUp} onPointerCancel={handlePointerUp} onWheel={handleWheel} style={{ cursor: dragging ? "grabbing" : "grab" }}>
          <defs><filter id="brain-glow"><feGaussianBlur stdDeviation="3" result="blur" /><feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge></filter><linearGradient id="brain-surface" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stopColor="#d5dae4" stopOpacity=".56" /><stop offset="100%" stopColor="#7887a0" stopOpacity=".2" /></linearGradient></defs>
          <g className="brain-orbit-guides"><ellipse cx="313" cy="181" rx="246" ry="145" /><ellipse cx="313" cy="181" rx="257" ry="151" className="orbit-dashed" /><path d="M 71 181 H 556 M 313 34 V 330" /></g>
          <motion.g className="brain-art" animate={{ x: brainX, y: brainY, scale: brainScale, rotate: rotation + focusRotation }} transition={{ type: "spring", stiffness: 110, damping: 18 }} style={{ transformOrigin: "313px 181px" }}>
            <path className="brain-silhouette" d="M 93 188 C 79 163 89 133 113 121 C 105 91 130 66 160 69 C 173 38 210 34 232 55 C 257 25 298 36 311 63 C 338 37 380 47 388 77 C 424 56 462 75 466 105 C 500 102 527 129 517 158 C 549 173 550 211 522 229 C 527 263 497 283 471 277 C 454 309 413 318 384 295 C 354 322 314 315 295 288 C 265 312 222 304 210 277 C 175 285 144 263 143 235 C 112 236 88 216 93 188 Z" />
            <path className="brain-fissure" d="M 311 63 C 300 100 302 131 311 163 C 316 190 311 219 295 288" />
            <path className="brain-gyrus" d="M 126 137 C 156 111 188 103 222 111 C 246 116 257 103 270 84 M 110 173 C 148 149 175 137 215 145 C 244 151 263 140 280 121 M 112 207 C 151 187 191 178 220 187 C 246 195 267 180 289 160 M 143 239 C 172 218 199 213 226 222 C 250 229 271 214 292 194 M 346 87 C 368 108 389 110 414 101 C 445 91 469 110 480 131 M 329 119 C 354 137 379 140 409 130 C 441 119 469 138 493 158 M 328 153 C 356 170 383 174 414 164 C 448 153 477 173 506 190 M 326 190 C 354 204 381 210 410 201 C 445 190 475 208 499 222 M 321 224 C 347 239 372 248 401 238 C 427 229 454 245 471 259" />
            <path className="brain-subtle-fold" d="M 164 83 C 183 101 199 83 213 69 M 173 254 C 192 238 208 253 223 268 M 432 82 C 419 99 432 116 451 122 M 427 275 C 412 259 430 246 451 252" />
            <g className="brain-network">{networkEdges.map(([from, to], index) => { const a = networkNodes[from]; const b = networkNodes[to]; return <line key={`${from}-${to}`} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} className={`network-edge ${index % 9 === 0 ? "network-edge-signal" : ""}`} />; })}{networkNodes.map(([x, y], index) => <circle key={`${x}-${y}`} cx={x} cy={y} r={index % 8 === 0 ? 2.5 : 1.45} className={`network-node ${index % 8 === 0 ? "network-node-bright" : ""}`} />)}</g>
            <g className="region-targets">{brainRegions.map((item) => <g key={item.id} className={`region-target ${selectedRegion === item.id ? "region-target-active" : ""}`} onPointerDown={(event) => { event.stopPropagation(); onSelectRegion(item.id); }}><ellipse cx={item.x} cy={item.y} rx={item.id === "hippocampus" ? 24 : 33} ry={item.id === "hippocampus" ? 11 : 25} className={`region-halo region-halo-${item.tone}`} /><circle cx={item.x} cy={item.y} r="12" className="region-hit" /><circle cx={item.x} cy={item.y} r={selectedRegion === item.id ? 6 : 3.5} className={`region-node region-node-${item.tone}`} /></g>)}</g>
          </motion.g>
          <motion.g className="brain-callout" animate={{ opacity: selectedRegion ? 1 : 0, x: selectedRegion ? 0 : -5 }} transition={{ duration: .3 }}><line x1="336" y1="218" x2="542" y2="262" className="brain-callout-line" /><circle cx="542" cy="262" r="3" className="brain-callout-dot" /></motion.g>
        </svg>
        <div className="brain-stage-note"><span className="drag-hint"><ArrowUpRight size={12} /> drag to rotate · wheel to zoom</span><span><span className="status-dot" /> neutral anatomy / predicted impact only</span></div>
      </div>
      <AnimatePresence mode="wait"><motion.div key={selectedRegion ?? "empty"} className={`brain-context ${region ? `brain-context-${region.tone}` : "brain-context-empty"}`} initial={{ opacity: 0, y: 9 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: .34 }}><div className="brain-context-heading">{region ? <><span className="eyebrow">{region.label} / {region.short}</span><strong>{region.detail}</strong></> : <><span className="eyebrow">Choose a region</span><strong>Follow the evidence to its source.</strong></>}</div>{region ? <><div className="brain-context-stats"><div><span>Patient indicator</span><strong>{patient.diagnosis === "CN" ? "Low concern" : patient.diagnosis === "MCI" ? "Watch signal" : "Elevated concern"}</strong></div><div><span>Model contribution</span><strong>{region.impact}</strong></div><div><span>Observed change</span><strong>{region.change}</strong></div></div><p>{evidenceText}. This is a model-indicated region, not a direct anatomical finding.</p><button className="text-action" onClick={() => document.getElementById("trajectory")?.scrollIntoView({ behavior: "smooth", block: "center" })}>Connect to trajectory <ArrowDownRight size={14} /></button></> : <p>Start with a neutral view. Select the hippocampus to see how a memory-related signal connects to the patient’s trajectory.</p>}</motion.div></AnimatePresence>
      <div className="brain-region-nav">{brainRegions.map((item) => <button key={item.id} className={selectedRegion === item.id ? "active" : ""} onClick={() => onSelectRegion(item.id)}><span className={`region-nav-dot region-nav-dot-${item.tone}`} /><span>{item.label}</span>{selectedRegion === item.id && <Check size={13} />}</button>)}</div>
    </div>
  );
}

function Sparkline({ values, tone = "indigo" }: { values: number[]; tone?: "indigo" | "teal" | "amber" }) {
  const min = Math.min(...values) - 2;
  const max = Math.max(...values) + 2;
  const points = values.map((value, index) => `${(index / (values.length - 1)) * 100},${100 - ((value - min) / (max - min)) * 100}`).join(" ");
  return (
    <svg className={`sparkline sparkline-${tone}`} viewBox="0 0 100 36" preserveAspectRatio="none" aria-hidden="true">
      <polyline points={points} fill="none" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

function trajectoryPoints(values: number[], width: number, height: number) {
  const min = 0;
  const max = 100;
  return values.map((value, index) => ({
    x: (index / (values.length - 1)) * width,
    y: height - ((value - min) / (max - min)) * height,
    value,
  }));
}

function TrajectoryChart({ patient }: { patient: Patient }) {
  const width = 840;
  const height = 280;
  const historic = trajectoryPoints(patient.historical, width * 0.47, height);
  const forecast = trajectoryPoints(patient.forecast, width * 0.53, height).map((point, index) => ({
    ...point,
    x: width * 0.47 + (index / (patient.forecast.length - 1)) * (width * 0.53),
  }));
  const all = [...historic, ...forecast.slice(1)];
  const range = trajectoryPoints(patient.range, width, 44).map((point) => ({ ...point, y: 120 - point.y }));
  const linePath = all.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x.toFixed(1)} ${point.y.toFixed(1)}`).join(" ");
  const historicPath = historic.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x.toFixed(1)} ${point.y.toFixed(1)}`).join(" ");
  const forecastPath = [historic[historic.length - 1], ...forecast.slice(1)].map((point, index) => `${index === 0 ? "M" : "L"} ${point.x.toFixed(1)} ${point.y.toFixed(1)}`).join(" ");
  const upper = all.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x.toFixed(1)} ${(point.y - (range[index]?.y ?? 8)).toFixed(1)}`).join(" ");
  const lower = [...all].reverse().map((point, reverseIndex) => {
    const index = all.length - 1 - reverseIndex;
    return `L ${point.x.toFixed(1)} ${(point.y + (range[index]?.y ?? 8)).toFixed(1)}`;
  }).join(" ");
  const uncertaintyPath = `${upper} ${lower} Z`;
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const hoverPoint = hoverIndex === null ? null : all[hoverIndex];
  const labels = ["-36m", "-30m", "-24m", "-18m", "-12m", "-6m", "now", "+6m", "+12m", "+18m", "+24m", "+30m", "+36m", "+42m", "+48m"];
  return (
    <div className="trajectory-chart-wrap">
      <div className="chart-legend">
        <span><i className="legend-line legend-history" /> Observed history</span>
        <span><i className="legend-line legend-forecast" /> 48-month projection</span>
        <span><i className="legend-band" /> 90% uncertainty</span>
      </div>
      <div className="trajectory-chart" onMouseLeave={() => setHoverIndex(null)}>
        <svg viewBox={`0 0 ${width} ${height + 36}`} role="img" aria-label="Cognitive trajectory with a 48 month projection">
          <defs>
            <linearGradient id="uncertainty-fill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--indigo)" stopOpacity="0.18" />
              <stop offset="100%" stopColor="var(--indigo)" stopOpacity="0.01" />
            </linearGradient>
            <linearGradient id="history-stroke" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="var(--teal)" />
              <stop offset="100%" stopColor="var(--indigo)" />
            </linearGradient>
          </defs>
          {[0, 25, 50, 75, 100].map((tick) => {
            const y = height - (tick / 100) * height;
            return <g key={tick}><line x1="0" x2={width} y1={y} y2={y} className="chart-grid" /><text x="-12" y={y + 4} className="chart-y-label" textAnchor="end">{tick}</text></g>;
          })}
          <line x1={width * 0.47} x2={width * 0.47} y1="0" y2={height} className="chart-now-line" />
          <text x={width * 0.47} y={height + 28} className="chart-now-label" textAnchor="middle">NOW</text>
          <motion.path d={uncertaintyPath} className="uncertainty-area" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.8, delay: 0.9 }} />
          <motion.path d={historicPath} className="history-path" initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1.5, ease: "easeOut" }} />
          <motion.path d={forecastPath} className="forecast-path" initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1.2, delay: 0.55, ease: "easeOut" }} />
          {all.map((point, index) => (
            <g key={`${point.x}-${point.y}`} onMouseEnter={() => setHoverIndex(index)} className="chart-point-group">
              <circle cx={point.x} cy={point.y} r="16" className="chart-point-hit" />
              <motion.circle cx={point.x} cy={point.y} r={index === 6 ? 5 : 3.5} className={index >= 6 ? "chart-point chart-point-forecast" : "chart-point"} initial={{ opacity: 0, scale: 0 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.06 * index + 0.6, duration: 0.3 }} />
            </g>
          ))}
          {hoverPoint && <><line x1={hoverPoint.x} x2={hoverPoint.x} y1="0" y2={height} className="chart-hover-line" /><circle cx={hoverPoint.x} cy={hoverPoint.y} r="7" className="chart-hover-dot" /></>}
          {labels.slice(0, 9).map((label, index) => <text key={label} x={(index / 8) * width} y={height + 28} className="chart-x-label" textAnchor={index === 0 ? "start" : index === 8 ? "end" : "middle"}>{label}</text>)}
        </svg>
        <AnimatePresence>
          {hoverPoint && (
            <motion.div className="chart-tooltip" initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} style={{ left: `${Math.min(78, Math.max(12, (hoverPoint.x / width) * 100))}%` }}>
              <span className="tooltip-label">{hoverIndex !== null && hoverIndex < 7 ? "Observed" : "Projected"}</span>
              <strong>{hoverPoint.value.toFixed(0)} <small>/ 100</small></strong>
              <span className="tooltip-sub">{hoverIndex !== null && hoverIndex < 7 ? `${Math.abs(36 - hoverIndex * 6)} months ago` : `month ${(hoverIndex! - 6) * 6} ahead`}</span>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

function MetricChip({ label, value, note, icon: Icon, tone = "indigo" }: { label: string; value: string; note: string; icon: typeof Activity; tone?: string }) {
  return (
    <div className={`metric-chip metric-chip-${tone}`}>
      <div className="metric-chip-icon"><Icon size={15} strokeWidth={1.8} /></div>
      <div><span className="eyebrow">{label}</span><strong>{value}</strong><small>{note}</small></div>
    </div>
  );
}

export default function Home() {
  const [patientIndex, setPatientIndex] = useState(0);
  const [selectedRegion, setSelectedRegion] = useState<BrainRegionId | null>(null);
  const [activeSection, setActiveSection] = useState<SectionId>("overview");
  const [sidebarOpen, setSidebarOpen] = useState(() => typeof window === "undefined" || window.innerWidth > 980);
  const [showPatientMenu, setShowPatientMenu] = useState(false);
  const [pathwayAction, setPathwayAction] = useState("blood");
  const [scenario, setScenario] = useState<"observed" | "intervene">("observed");
  const [interventionStrength, setInterventionStrength] = useState(30);
  const [capacity, setCapacity] = useState(50);
  const [notice, setNotice] = useState<string | null>(null);
  const patient = patients[patientIndex];
  const overviewRef = useRef<HTMLElement>(null);
  const sectionRefs = useRef<Record<string, HTMLElement | null>>({});
  const inView = useInView(overviewRef, { once: true, amount: 0.1 });

  useEffect(() => {
    const observer = new IntersectionObserver((entries) => {
      const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (visible?.target.id) setActiveSection(visible.target.id as SectionId);
    }, { rootMargin: "-18% 0px -62% 0px", threshold: [0.05, 0.2, 0.55] });
    Object.values(sectionRefs.current).forEach((element) => element && observer.observe(element));
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!notice) return;
    const timer = window.setTimeout(() => setNotice(null), 3200);
    return () => window.clearTimeout(timer);
  }, [notice]);

  useEffect(() => {
    setPathwayAction(patient.diagnosis === "AD" ? "diagnose" : patient.diagnosis === "MCI" ? "blood" : "monitor");
    setSelectedRegion(null);
  }, [patientIndex, patient.diagnosis]);

  const interventionRisk = useMemo(() => Math.max(4, patient.probabilities.AD - interventionStrength * 0.38), [patient, interventionStrength]);
  const observedRisk = patient.probabilities.AD;
  const selectedCount = Math.round(8 + capacity * 0.12);
  const expectedYield = (1.47 + capacity * 0.0415).toFixed(2);
  const utilization = Math.min(100, Math.round(78 + capacity * 0.44));
  const pathwayOptions = [
    { id: "blood", label: "Blood", detail: "low burden", icon: FlaskConical, color: "purple" },
    { id: "mri", label: "MRI", detail: "structural", icon: Brain, color: "blue" },
    { id: "pet", label: "PET", detail: "high specificity", icon: Orbit, color: "coral" },
    { id: "diagnose", label: "Diagnose", detail: "commit state", icon: Stethoscope, color: "amber" },
    { id: "monitor", label: "Monitor", detail: "reassess later", icon: Clock3, color: "teal" },
  ];
  const selectedPathway = pathwayOptions.find((option) => option.id === pathwayAction) ?? pathwayOptions[0];

  const scrollToSection = (id: SectionId) => {
    setActiveSection(id);
    sectionRefs.current[id]?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const selectPatient = (index: number) => {
    setPatientIndex(index);
    setShowPatientMenu(false);
    setNotice(`Loaded ${patients[index].id} · ${patients[index].diagnosisLabel}`);
  };

  const handleAction = (message: string) => setNotice(message);

  return (
    <div className={`app-shell ${sidebarOpen ? "sidebar-visible" : "sidebar-collapsed"}`}>
      <aside className="app-sidebar">
        <div className="brand-lockup">
          <img src={BRAND_MARK} alt="Sentinel mark" className="brand-mark" />
          <div className="brand-type"><strong>sentinel</strong><span>clinical intelligence</span></div>
          <button className="icon-button sidebar-toggle" onClick={() => setSidebarOpen((value) => !value)} aria-label={sidebarOpen ? "Collapse sidebar" : "Expand sidebar"}>{sidebarOpen ? <PanelLeftClose size={17} /> : <PanelLeftOpen size={17} />}</button>
        </div>
        <div className="workspace-switcher"><div className="workspace-avatar">SR</div><div><span className="eyebrow">Workspace</span><strong>St. Raphael Memory</strong></div><ChevronDown size={15} /></div>
        <div className="sidebar-label">Intelligence layers</div>
        <nav className="side-nav" aria-label="Dashboard sections">
          {navItems.map(({ id, label, icon: Icon }) => (
            <button key={id} className={`side-nav-item ${activeSection === id ? "active" : ""}`} onClick={() => scrollToSection(id)}>
              <Icon size={17} strokeWidth={activeSection === id ? 2.2 : 1.7} /><span>{label}</span>{activeSection === id && <i className="nav-pulse" />}
            </button>
          ))}
        </nav>
        <div className="sidebar-foot">
          <div className="model-status"><span className="status-dot" /><div><span className="eyebrow">Model status</span><strong>Synced 4 min ago</strong></div><ShieldCheck size={16} /></div>
          <button className="side-nav-item" onClick={() => handleAction("Settings are available in the full clinical workspace.")}><Settings2 size={17} /><span>Workspace settings</span></button>
          <div className="sidebar-user"><div className="user-avatar">DR</div><div><strong>Dr. R. Malik</strong><span>Neurology · Admin</span></div><MoreHorizontal size={17} /></div>
        </div>
      </aside>

      <main className="app-main">
        <header className="topbar">
          <div className="topbar-left"><button className="mobile-menu icon-button" onClick={() => setSidebarOpen((value) => !value)} aria-label="Toggle navigation"><Menu size={18} /></button><div className="breadcrumb"><span>Clinical workspace</span><i>/</i><strong>Patient intelligence</strong></div></div>
          <div className="topbar-actions"><span className="research-pill"><span className="status-dot" /> Research mode</span><button className="topbar-icon icon-button" onClick={() => handleAction("Search is scoped to the current cohort.")} aria-label="Search"><Search size={17} /></button><button className="topbar-icon icon-button" onClick={() => handleAction("Snapshot queued for secure export.")} aria-label="Export snapshot"><Download size={17} /></button><button className="avatar-button" onClick={() => handleAction("Profile menu opened.")}>RM</button></div>
        </header>

        <div className="content-wrap">
          <section id="overview" ref={(node) => { overviewRef.current = node; sectionRefs.current.overview = node; }} className="hero-section">
            <div className="hero-ambient" style={{ backgroundImage: `url(${HERO_IMAGE})` }} />
            <div className="hero-grid-overlay" />
            <div className="hero-copy">
              <div className="section-kicker"><span className="kicker-line" /> Patient intelligence <span className="kicker-meta">last refreshed 09:42:18</span></div>
              <AnimatePresence mode="wait">
                <motion.div key={patient.id} initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -12 }} transition={{ duration: 0.35 }}>
                  <h1>{patient.id}</h1>
                  <div className="patient-meta"><span>{patient.label}</span><i>•</i><span>Age {patient.age}</span><i>•</i><span>{patient.education} years education</span><i>•</i><span>{patient.visits} visits</span></div>
                </motion.div>
              </AnimatePresence>
              <BrainExplorer patient={patient} selectedRegion={selectedRegion} onSelectRegion={setSelectedRegion} onReset={() => setSelectedRegion(null)} />
              <div className="hero-actions"><div className="patient-select-wrap"><button className="patient-select" onClick={() => setShowPatientMenu((value) => !value)}><div className="select-avatar">{patient.label.split(" ").map((part) => part[0]).join("")}</div><div><span className="eyebrow">Active patient</span><strong>{patient.label}</strong></div><ChevronDown size={15} /></button>{showPatientMenu && <div className="patient-menu">{patients.map((item, index) => <button key={item.id} className={index === patientIndex ? "selected" : ""} onClick={() => selectPatient(index)}><div className={`mini-status mini-status-${item.health}`} /><div><strong>{item.id}</strong><span>{item.diagnosisLabel}</span></div>{index === patientIndex && <Check size={15} />}</button>)}</div>}</div><button className="ghost-action" onClick={() => handleAction("Trajectory focus enabled.")}><Activity size={15} /> View trajectory</button></div>
            </div>
            <div className="hero-diagnosis">
              <AnimatePresence mode="wait">
                <motion.div key={patient.id} className="diagnosis-panel" initial={{ opacity: 0, scale: 0.97, x: 16 }} animate={{ opacity: 1, scale: 1, x: 0 }} exit={{ opacity: 0, scale: 0.98, x: -10 }} transition={{ duration: 0.45, ease: [0.23, 1, 0.32, 1] }}>
                  <div className="diagnosis-panel-top"><span className="eyebrow">AD risk / current status</span><div className={`diagnosis-status diagnosis-status-${patient.health}`}><span className="status-dot" /> {patient.health === "stable" ? "stable signal" : patient.health === "watch" ? "watch signal" : "escalate signal"}</div></div>
                  <div className="diagnosis-main"><div><span className={`diagnosis-code diagnosis-code-${patient.diagnosis}`}>{patient.diagnosis}</span><h2>{patient.diagnosisLabel}</h2><p>Multimodal posterior from {patient.visits} longitudinal visits.</p></div><ConfidenceRing confidence={patient.confidence} diagnosis={patient.diagnosis} /></div>
                  <div className="confidence-strip"><div className="strip-label"><span>Uncertainty range</span><strong>± {patient.uncertainty.toFixed(1)}%</strong></div><div className="uncertainty-meter"><motion.div className={`uncertainty-fill uncertainty-fill-${patient.health}`} initial={{ width: 0 }} animate={{ width: `${Math.max(18, 100 - patient.uncertainty * 3)}%` }} transition={{ duration: 0.8, delay: 0.25 }} /><span className="uncertainty-marker" style={{ left: `${Math.max(18, 100 - patient.uncertainty * 3)}%` }} /></div><div className="strip-foot"><span>lower confidence</span><span>higher confidence</span></div></div>
                  <div className="diagnosis-footer"><span><Clock3 size={14} /> last visit {patient.lastSeen}</span><button onClick={() => handleAction("Diagnostic rationale opened.")}>View rationale <ArrowUpRight size={14} /></button></div>
                </motion.div>
              </AnimatePresence>
            </div>
            <div className="hero-scroll-cue"><span className="scroll-line" /><span>Scroll to read the signal</span><ArrowDownRight size={14} /></div>
          </section>

          <section className="signal-strip" aria-label="Patient signal summary">
            <MetricChip label="Trajectory score" value={`${patient.score.toFixed(1)} / 100`} note={`${Math.abs(patient.delta).toFixed(1)} pts over 12 mo`} icon={Activity} tone={patient.delta < -4 ? "amber" : "teal"} />
            <MetricChip label="Model confidence" value={`${patient.confidence.toFixed(1)}%`} note={`${patient.uncertainty.toFixed(1)}% uncertainty`} icon={Gauge} tone="indigo" />
            <MetricChip label="Visits observed" value={`${patient.visits}`} note="longitudinal context" icon={CalendarDays} tone="teal" />
            <MetricChip label="Data coverage" value={patient.diagnosis === "CN" ? "4 / 4" : "3 / 4"} note="modalities available" icon={ClipboardCheck} tone="amber" />
          </section>

          <SectionReveal className="section-anchor" delay={0.04}><div className="chapter-heading"><div><span className="section-index">01 / OVERVIEW</span><h2>The signal, in context.</h2></div><p>One surface for the current state, the evidence beneath it, and the next decision that would make the picture clearer.</p></div></SectionReveal>

          <SectionReveal className="overview-layout" delay={0.08}>
            <div id="trajectory" ref={(node) => { sectionRefs.current.trajectory = node; }} className="panel trajectory-panel clinical-plate">
              <div className="panel-heading"><div><span className="eyebrow"><span className="signal-pulse" /> Disease trajectory</span><h3>Projected cognitive trajectory</h3></div><div className="panel-heading-meta"><span className="image-texture-tag">48 month horizon</span><span className="plate-tag">calibrated / 90% CI</span><button className="icon-button small" onClick={() => handleAction("Trajectory view expanded.")}><MoreHorizontal size={17} /></button></div></div>
              <div className="trajectory-intro"><p>Observed performance is mapped against a calibrated forward projection. The uncertainty field widens as the model looks further ahead.</p><div className="trajectory-score"><span className="eyebrow">Current score</span><strong><AnimatedNumber value={patient.score} decimals={1} /></strong><span className={patient.delta < 0 ? "negative" : "positive"}><ArrowDownRight size={14} /> {Math.abs(patient.delta).toFixed(1)} in 12 mo</span></div></div>
              <TrajectoryChart patient={patient} />
              <div className="chart-footnote"><span><Info size={13} /> Score is a composite cognitive index normalized to the study cohort.</span><button onClick={() => handleAction("Model notes opened.")}>Model notes <ArrowUpRight size={13} /></button></div>
            </div>
            <div className="overview-side">
              <div className="panel readout-panel"><div className="panel-heading compact"><div><span className="eyebrow">Signal readout</span><h3>At a glance</h3></div><span className="live-label"><span className="status-dot" /> live</span></div><div className="readout-stack"><div className="readout-row"><div className="readout-label"><span className="readout-icon readout-icon-teal"><ShieldCheck size={15} /></span><span>Confidence</span></div><strong>{patient.confidence.toFixed(1)}%</strong><span className="readout-trend positive">high</span></div><div className="readout-row"><div className="readout-label"><span className="readout-icon readout-icon-amber"><CircleHelp size={15} /></span><span>Uncertainty</span></div><strong>{patient.uncertainty.toFixed(1)}%</strong><span className="readout-trend">moderate</span></div><div className="readout-row"><div className="readout-label"><span className="readout-icon readout-icon-indigo"><Zap size={15} /></span><span>Information gain</span></div><strong>+0.21</strong><span className="readout-trend positive">actionable</span></div></div><div className="readout-note"><Sparkles size={15} /><p>Signal convergence is {patient.diagnosis === "CN" ? "reassuring" : patient.diagnosis === "MCI" ? "incomplete" : "strong"}. The next best action is selected below.</p></div></div>
              <div className="panel insight-panel" style={{ backgroundImage: `linear-gradient(135deg, rgba(13,17,28,.08), rgba(13,17,28,.8)), url(${NEURAL_TEXTURE})` }}><div className="insight-mark"><Sparkles size={16} /></div><span className="eyebrow">AI interpretation</span><h3>{patient.diagnosis === "CN" ? "No convergent disease pattern detected." : patient.diagnosis === "MCI" ? "The pattern is changing before the diagnosis is." : "The pattern is convergent across modalities."}</h3><p>{patient.pathway.rationale}</p><button className="text-action" onClick={() => scrollToSection("pathway")}>Open pathway <ArrowUpRight size={14} /></button></div>
            </div>
          </SectionReveal>

          <SectionReveal className="evidence-section" delay={0.1}><div className="evidence-header"><div><span className="section-index">EVIDENCE LAYERS</span><h2>What the model is reading.</h2></div><span className="evidence-count"><strong>09</strong> measures active</span></div><div className="evidence-grid"><div className="panel biomarker-panel clinical-plate"><div className="panel-heading compact"><div><span className="eyebrow">Multimodal inputs</span><h3>Biomarker state</h3></div><button className="text-action" onClick={() => handleAction("All 9 active measures shown.")}>View all <ArrowUpRight size={13} /></button></div><div className="biomarker-list">{patient.biomarkers.map((biomarker, index) => <motion.div key={biomarker.name} className="biomarker-row" initial={{ opacity: 0, x: -10 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true }} transition={{ delay: index * 0.07 }}><div className={`biomarker-icon biomarker-icon-${biomarker.tone}`}>{index === 0 ? <Brain size={15} /> : index === 1 ? <Orbit size={15} /> : index === 2 ? <Activity size={15} /> : <ScanLine size={15} />}</div><div className="biomarker-copy"><strong>{biomarker.name}</strong><span>{biomarker.note}</span></div><div className={`biomarker-value biomarker-value-${biomarker.tone}`}>{biomarker.value}</div><Sparkline values={[35 + index * 4, 41 + index * 3, 38 + index * 5, 52 + index * 2, 47 + index * 4, 62 + index * 3]} tone={biomarker.tone === "coral" ? "amber" : biomarker.tone as "indigo" | "teal" | "amber"} /></motion.div>)}</div></div><div className="panel visit-panel"><div className="panel-heading compact"><div><span className="eyebrow">Longitudinal record</span><h3>Visit cadence</h3></div><span className="visit-total"><AnimatedNumber value={patient.visits} /> visits</span></div><div className="visit-timeline">{["Baseline", "6 mo", "12 mo", "18 mo", "Current"].slice(0, patient.visits).map((visit, index) => <div key={visit} className={`visit-node ${index === patient.visits - 1 ? "current" : ""}`}><span className="visit-dot" /><div><strong>{visit}</strong><span>{index === patient.visits - 1 ? patient.lastSeen : `${18 - index * 3} months ago`}</span></div>{index < patient.visits - 1 && <i className="visit-connector" />}</div>)}</div><div className="visit-callout"><CalendarDays size={16} /><span>Next review window</span><strong>{patient.diagnosis === "CN" ? "Feb 2027" : patient.diagnosis === "MCI" ? "Nov 2026" : "Oct 2026"}</strong></div></div></div></SectionReveal>

          <section id="pathway" ref={(node) => { sectionRefs.current.pathway = node; }} className="full-section pathway-section">
            <SectionReveal><div className="chapter-heading chapter-heading-inline"><div><span className="section-index">02 / DECISION PATHWAY</span><h2>Make the next decision count.</h2></div><p>Recommendations are ranked by expected information gain, burden, and current uncertainty — not by habit.</p></div></SectionReveal>
            <SectionReveal className="pathway-layout" delay={0.06}><div className="panel pathway-panel clinical-plate"><div className="pathway-top"><div><span className="eyebrow">Active pathway</span><h3>{patient.pathway.current}</h3></div><div className="recommendation-badge"><span className="status-dot" /> recommendation ready</div></div><div className="pathway-flow"><div className="flow-node flow-current"><span className="flow-node-index">01</span><div className="flow-node-icon"><Brain size={18} /></div><span className="eyebrow">Current state</span><strong>{patient.diagnosis}</strong><small>{patient.confidence.toFixed(1)}% confidence</small></div><div className="flow-connector"><span className="connector-line" /><motion.span className="connector-dot" animate={{ x: [0, 34, 68], opacity: [0, 1, 0] }} transition={{ duration: 2.4, repeat: Infinity, ease: "easeInOut" }} /></div><div className="flow-node flow-recommendation"><span className="flow-node-index">02</span><div className="flow-node-icon"><Sparkles size={18} /></div><span className="eyebrow">AI recommendation</span><strong>{patient.pathway.recommendation.replace("_", " ")}</strong><small>+{patient.pathway.expectedBenefit.replace("+", "")} expected gain</small></div><div className="flow-connector"><span className="connector-line" /><motion.span className="connector-dot" animate={{ x: [0, 34, 68], opacity: [0, 1, 0] }} transition={{ duration: 2.4, repeat: Infinity, delay: 0.7, ease: "easeInOut" }} /></div><div className="flow-actions">{pathwayOptions.slice(0, 4).map((option) => { const Icon = option.icon; return <button key={option.id} className={`flow-action ${pathwayAction === option.id ? "selected" : ""} flow-action-${option.color}`} onClick={() => setPathwayAction(option.id)}><Icon size={17} /><strong>{option.label}</strong><span>{option.detail}</span>{pathwayAction === option.id && <Check size={14} className="action-check" />}</button>; })}</div></div><div className="pathway-bottom"><div className="pathway-rationale"><span className="eyebrow">Why this action</span><p>{patient.pathway.rationale}</p></div><AnimatePresence mode="wait"><motion.div key={selectedPathway.id} className="pathway-metrics" initial={{ opacity: 0, y: 7 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -5 }}><div><span>Expected benefit</span><strong className="text-teal">{selectedPathway.id === "blood" ? "+0.21" : selectedPathway.id === "mri" ? "+0.16" : selectedPathway.id === "pet" ? "+0.08" : selectedPathway.id === "diagnose" ? "+0.04" : "+0.08"}</strong></div><div><span>Resource cost</span><strong>{selectedPathway.id === "blood" ? "1.0 pts" : selectedPathway.id === "mri" ? "5.0 pts" : selectedPathway.id === "pet" ? "12.0 pts" : "0.0 pts"}</strong></div><button className="primary-button" onClick={() => handleAction(`${selectedPathway.label} pathway marked for review.`)}>Review action <ArrowUpRight size={14} /></button></motion.div></AnimatePresence></div></div><div className="pathway-side panel"><div className="pathway-side-top"><span className="eyebrow">Recommendation trace</span><GitBranch size={18} /></div><div className="trace-list"><div className="trace-item active"><span className="trace-point" /><div><strong>Posterior uncertainty</strong><span>{patient.uncertainty.toFixed(1)}% spread</span></div></div><div className="trace-item"><span className="trace-point" /><div><strong>Modalities available</strong><span>{patient.diagnosis === "CN" ? "4 of 4" : "3 of 4"} connected</span></div></div><div className="trace-item"><span className="trace-point" /><div><strong>Decision threshold</strong><span>0.70 posterior</span></div></div></div><div className="trace-meter"><div className="trace-meter-head"><span>Evidence convergence</span><strong>{patient.diagnosis === "AD" ? "92%" : patient.diagnosis === "MCI" ? "64%" : "81%"}</strong></div><div className="progress-track"><motion.div initial={{ width: 0 }} whileInView={{ width: patient.diagnosis === "AD" ? "92%" : patient.diagnosis === "MCI" ? "64%" : "81%" }} viewport={{ once: true }} transition={{ duration: 1 }} className="progress-value progress-value-teal" /></div></div></div></SectionReveal>
          </section>

          <section className="full-section simulation-section">
            <SectionReveal><div className="chapter-heading chapter-heading-inline"><div><span className="section-index">03 / COUNTERFACTUALS</span><h2>Explore the intervention space.</h2></div><p>Move from “what is” to “what if” without losing the observed baseline.</p></div></SectionReveal>
            <SectionReveal className="simulation-layout" delay={0.06}><div className="panel simulation-panel"><div className="panel-heading"><div><span className="eyebrow">Counterfactual engine</span><h3>What happens if we intervene?</h3></div><div className="segmented-control"><button className={scenario === "observed" ? "active" : ""} onClick={() => setScenario("observed")}>Observed</button><button className={scenario === "intervene" ? "active" : ""} onClick={() => setScenario("intervene")}>Intervene</button></div></div><div className="simulation-controls"><div className="sim-control"><div className="sim-control-head"><span><span className="control-icon control-icon-coral">↘</span> p-tau181 reduction</span><strong>{interventionStrength}%</strong></div><input type="range" min="0" max="60" value={interventionStrength} onChange={(event) => { setInterventionStrength(Number(event.target.value)); setScenario("intervene"); }} style={{ "--range-progress": `${(interventionStrength / 60) * 100}%` } as CSSProperties} /><div className="range-foot"><span>0%</span><span>60%</span></div></div><div className="sim-control"><div className="sim-control-head"><span><span className="control-icon control-icon-indigo">↗</span> cognitive reserve</span><strong>+2 yrs</strong></div><div className="static-range"><span style={{ width: "34%" }} /></div><div className="range-foot"><span>baseline</span><span>modeled</span></div></div></div><div className="simulation-result"><div className="sim-result-main"><span className="eyebrow">Projected AD probability</span><AnimatePresence mode="wait"><motion.strong key={`${scenario}-${interventionStrength}`} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -7 }}>{(scenario === "observed" ? observedRisk : interventionRisk).toFixed(1)}%</motion.strong></AnimatePresence><div className="sim-delta"><ArrowDownRight size={14} /> <AnimatedNumber value={Math.max(0, observedRisk - interventionRisk)} decimals={1} suffix=" pts" /> lower with intervention</div></div><div className="sim-bars"><div className="sim-bar-row"><span>Observed</span><div className="progress-track"><motion.div className="progress-value progress-value-coral" animate={{ width: `${observedRisk}%` }} transition={{ duration: 0.7 }} /></div><strong>{observedRisk.toFixed(1)}%</strong></div><div className="sim-bar-row"><span>Intervene</span><div className="progress-track"><motion.div className="progress-value progress-value-teal" animate={{ width: `${interventionRisk}%` }} transition={{ duration: 0.7 }} /></div><strong>{interventionRisk.toFixed(1)}%</strong></div></div></div><div className="simulation-foot"><span><Sparkles size={14} /> Scenario uses a calibrated observational estimate, not a treatment guarantee.</span><button onClick={() => handleAction("Counterfactual scenario saved to the patient record.")}>Save scenario <ArrowUpRight size={13} /></button></div></div><div className="counterfactual-aside" style={{ backgroundImage: `linear-gradient(180deg, rgba(13,17,28,.05), rgba(13,17,28,.92)), url(${TRAJECTORY_TEXTURE})` }}><div className="aside-orbit"><Orbit size={18} /></div><span className="eyebrow">Modeled outcome</span><h3>{scenario === "observed" ? "Baseline held constant." : "The future bends."}</h3><p>{scenario === "observed" ? "Compare the observed state with an intervention hypothesis to understand where action has the most leverage." : "At this intensity, the model projects a meaningful reduction in risk slope over the 48-month horizon."}</p><div className="aside-stat"><span>Risk slope</span><strong>{scenario === "observed" ? "−4.8% / yr" : "−2.9% / yr"}</strong><small>{scenario === "observed" ? "current trajectory" : "modeled change"}</small></div></div></SectionReveal>
          </section>

          <section id="explainability" ref={(node) => { sectionRefs.current.explainability = node; }} className="full-section explainability-section">
            <SectionReveal><div className="chapter-heading chapter-heading-inline"><div><span className="section-index">04 / EXPLAINABILITY</span><h2>Every score has a shape.</h2></div><p>The strongest inputs are visible, ranked, and interpretable at a glance.</p></div></SectionReveal>
            <SectionReveal className="explainability-layout" delay={0.06}><div className="panel shap-panel clinical-plate"><div className="panel-heading"><div><span className="eyebrow">Local feature attribution</span><h3>What moved the prediction</h3></div><div className="shap-key"><span><i className="key-swatch key-risk" /> risk</span><span><i className="key-swatch key-protective" /> protective</span></div></div><div className="shap-list">{patient.factors.map((factor, index) => <motion.div key={factor.label} className={`shap-row ${factor.direction}`} initial={{ opacity: 0, x: -12 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true, amount: 0.2 }} transition={{ delay: index * 0.08 }}><div className="shap-label"><strong>{factor.label}</strong><span>{factor.contribution}</span></div><div className="shap-bar-area"><span className="shap-zero" /><motion.div className="shap-bar" initial={{ scaleX: 0 }} whileInView={{ scaleX: factor.value / 100 }} viewport={{ once: true }} transition={{ delay: index * 0.08 + 0.2, duration: 0.6, ease: [0.23, 1, 0.32, 1] }} style={{ originX: 0 }} /><span className="shap-detail">{factor.detail}</span></div><strong className="shap-value">{factor.direction === "positive" ? "+" : factor.direction === "negative" ? "−" : "·"}{Math.round(factor.value * 0.013 * 10) / 10}</strong></motion.div>)}</div><div className="shap-foot"><span><Info size={13} /> Local attribution from the frozen fusion model.</span><button onClick={() => handleAction("Full feature attribution report opened.")}>Open report <ArrowUpRight size={13} /></button></div></div><div className="panel confidence-panel"><div className="panel-heading compact"><div><span className="eyebrow">Calibration</span><h3>How sure is sure?</h3></div><CircleHelp size={17} /></div><div className="calibration-graphic"><div className="calibration-ring"><span><strong>{patient.confidence.toFixed(0)}%</strong><small>reliability</small></span></div><div className="calibration-axis"><span>low</span><div className="axis-line"><i style={{ left: `${patient.confidence}%` }} /></div><span>high</span></div></div><div className="calibration-notes"><div><span className="note-bar note-bar-teal" /><p><strong>Calibrated</strong> on held-out cohort</p></div><div><span className="note-bar note-bar-amber" /><p><strong>{patient.uncertainty.toFixed(1)}%</strong> posterior spread</p></div></div></div></SectionReveal>
          </section>

          <section id="causal" ref={(node) => { sectionRefs.current.causal = node; }} className="full-section causal-section">
            <SectionReveal><div className="chapter-heading chapter-heading-inline"><div><span className="section-index">05 / CAUSAL ANALYSIS</span><h2>Separate signal from confounding.</h2></div><p>Adjusted associations show which relationships survive age and education controls.</p></div></SectionReveal>
            <SectionReveal className="causal-layout" delay={0.06}><div className="panel causal-panel clinical-plate"><div className="panel-heading"><div><span className="eyebrow">Confounder-adjusted view</span><h3>Causal relationship map</h3></div><span className="adjustment-pill"><ShieldCheck size={14} /> FWL adjusted</span></div><div className="causal-map"><div className="causal-column causal-column-left"><div className="causal-node node-age"><span className="node-symbol">A</span><div><strong>Age</strong><small>confounder</small></div></div><div className="causal-node node-education"><span className="node-symbol">E</span><div><strong>Education</strong><small>confounder</small></div></div><div className="causal-node node-amyloid"><span className="node-symbol">β</span><div><strong>Amyloid</strong><small>biomarker</small></div></div></div><div className="causal-flow-visual"><svg viewBox="0 0 420 220" preserveAspectRatio="none" aria-hidden="true"><motion.path d="M 8 30 C 120 30, 130 78, 210 92 S 310 98, 412 110" className="causal-path path-amber" initial={{ pathLength: 0 }} whileInView={{ pathLength: 1 }} viewport={{ once: true }} transition={{ duration: 1.2 }} /><motion.path d="M 8 184 C 120 184, 130 135, 210 128 S 310 120, 412 110" className="causal-path path-teal" initial={{ pathLength: 0 }} whileInView={{ pathLength: 1 }} viewport={{ once: true }} transition={{ duration: 1.2, delay: 0.2 }} /><motion.path d="M 8 108 C 120 108, 130 108, 210 110 S 310 110, 412 110" className="causal-path path-coral" initial={{ pathLength: 0 }} whileInView={{ pathLength: 1 }} viewport={{ once: true }} transition={{ duration: 1.2, delay: 0.4 }} /></svg><div className="causal-center-label"><span className="eyebrow">adjusted outcome</span><strong>cognition</strong><span>score movement</span></div></div><div className="causal-column causal-column-right"><div className="causal-node node-cognition"><span className="node-symbol">C</span><div><strong>MMSE / ADAS13</strong><small>observed outcome</small></div></div></div></div><div className="causal-legend"><span><i className="legend-dot dot-amber" /> age adjustment</span><span><i className="legend-dot dot-teal" /> education adjustment</span><span><i className="legend-dot dot-coral" /> biological path</span><strong>n = 2,107 visits</strong></div></div><div className="panel causal-table-panel"><div className="panel-heading compact"><div><span className="eyebrow">Adjusted effects</span><h3>Relationships that persist</h3></div><Info size={16} /></div><div className="effect-list">{patient.causal.map((item) => <div key={`${item.from}-${item.to}`} className="effect-row"><div><strong>{item.from}</strong><span>→ {item.to}</span></div><div className="effect-meter"><span className={`effect-fill effect-fill-${item.tone}`} style={{ width: `${Math.abs(item.strength) * 100}%` }} /></div><strong className={item.strength > 0 ? "effect-positive" : "effect-negative"}>{item.strength > 0 ? "+" : "−"}{Math.abs(item.strength).toFixed(2)}</strong></div>)}</div><div className="causal-note"><Sparkles size={15} /><p>Associations are not treatment effects. Use the pathway to identify the next measure that can reduce uncertainty.</p></div></div></SectionReveal>
          </section>

          <section id="optimization" ref={(node) => { sectionRefs.current.optimization = node; }} className="full-section optimization-section">
            <SectionReveal><div className="chapter-heading chapter-heading-inline"><div><span className="section-index">06 / OPTIMIZATION</span><h2>Allocate the next hour wisely.</h2></div><p>Capacity-aware prioritization surfaces the candidates where one test changes the decision most.</p></div></SectionReveal>
            <SectionReveal className="optimization-layout" delay={0.06}><div className="panel optimization-panel clinical-plate"><div className="optimization-top"><div><span className="eyebrow">Resource allocation</span><h3>Population screening queue</h3></div><div className="capacity-control"><span>Capacity</span><input type="range" min="20" max="100" step="10" value={capacity} onChange={(event) => setCapacity(Number(event.target.value))} style={{ "--range-progress": `${((capacity - 20) / 80) * 100}%` } as CSSProperties} /><strong>{capacity} pts</strong></div></div><div className="optimization-metrics"><div><span className="eyebrow">Selected candidates</span><strong><AnimatedNumber value={selectedCount} /></strong><small>/ 248 total</small></div><div><span className="eyebrow">Expected yield</span><strong><AnimatedNumber value={Number(expectedYield)} decimals={2} /></strong><small>information gain</small></div><div><span className="eyebrow">Capacity utilization</span><strong><AnimatedNumber value={utilization} suffix="%" /></strong><small>solver confidence high</small></div></div><div className="optimization-table"><div className="optimization-table-head"><span>Candidate</span><span>Missing modalities</span><span>Expected gain</span><span>Priority</span></div>{["SUBJ2-0015 · m24", "SUBJ2-0028 · m24", "SUBJ2-0073 · m6", "SUBJ2-0117 · m0"].map((candidate, index) => <motion.div key={candidate} className="optimization-row" layout><div className="candidate-name"><span className="rank">0{index + 1}</span><strong>{candidate}</strong></div><span className="missing-test">{index === 0 ? "PET" : index === 1 ? "Blood + PET" : index === 2 ? "Blood" : "MRI + PET"}</span><strong className="gain-value">+{(0.31 - index * 0.045 + capacity / 1000).toFixed(2)}</strong><div className="priority-bar"><span style={{ width: `${91 - index * 13}%` }} /></div></motion.div>)}</div><div className="optimization-foot"><span><Zap size={14} /> Dynamic programming fallback · last solved 09:41</span><button onClick={() => handleAction("Allocation queue exported.")}><Download size={13} /> Export queue</button></div></div><div className="optimizer-aside"><div className="aside-orbit"><Target size={18} /></div><span className="eyebrow">Why this matters</span><h3>Find the patient where the next result changes the story.</h3><p>Prioritization balances clinical yield against test cost, so a constrained capacity still creates a useful next move.</p><div className="aside-stat"><span>Marginal value at {capacity} pts</span><strong>+{(0.18 + capacity / 1000).toFixed(2)}</strong><small>above cohort median</small></div></div></SectionReveal>
          </section>

          <footer className="app-footer"><div><img src={BRAND_MARK} alt="" className="footer-mark" /><span>sentinel / clinical intelligence</span></div><span>Research demo · synthetic cohort · v0.9.4</span><div><button onClick={() => handleAction("Documentation opened in a new workspace.")}>Documentation</button><button onClick={() => handleAction("Feedback channel opened.")}>Feedback</button></div></footer>
        </div>
      </main>
      <AnimatePresence>{notice && <motion.div className="toast-notice" initial={{ opacity: 0, y: 16, x: 8 }} animate={{ opacity: 1, y: 0, x: 0 }} exit={{ opacity: 0, y: 10 }}><Check size={15} /><span>{notice}</span><button onClick={() => setNotice(null)} aria-label="Dismiss notification"><X size={14} /></button></motion.div>}</AnimatePresence>
    </div>
  );
}
