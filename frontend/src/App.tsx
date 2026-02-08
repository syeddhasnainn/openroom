import React, { useRef, useState, useEffect, useCallback, useMemo } from "react";
import init, {
  adjust_image,
  apply_preset,
  adjust_hsl,
  apply_sharpening,
  apply_noise_reduction,
  apply_grain,
  apply_distortion,
  apply_vignette,
  apply_defringe,
  apply_tone_curve,
} from "./wasm";

interface SliderConfig {
  label: string;
  key: string;
  min: number;
  max: number;
  step: number;
  defaultValue: number;
}

const SLIDERS: SliderConfig[] = [
  { label: "Exposure", key: "exposure", min: -5, max: 5, step: 0.1, defaultValue: 0 },
  { label: "Contrast", key: "contrast", min: -100, max: 100, step: 1, defaultValue: 0 },
  { label: "Highlights", key: "highlights", min: -100, max: 100, step: 1, defaultValue: 0 },
  { label: "Shadows", key: "shadows", min: -100, max: 100, step: 1, defaultValue: 0 },
  { label: "Whites", key: "whites", min: -100, max: 100, step: 1, defaultValue: 0 },
  { label: "Blacks", key: "blacks", min: -100, max: 100, step: 1, defaultValue: 0 },
];

interface Adjustments {
  exposure: number;
  contrast: number;
  highlights: number;
  shadows: number;
  whites: number;
  blacks: number;
}

const defaultAdjustments: Adjustments = {
  exposure: 0,
  contrast: 0,
  highlights: 0,
  shadows: 0,
  whites: 0,
  blacks: 0,
};

// HSL Color Channels
const HSL_CHANNELS = [
  { key: "red", label: "Red", color: "#ef4444" },
  { key: "orange", label: "Orange", color: "#f97316" },
  { key: "yellow", label: "Yellow", color: "#eab308" },
  { key: "green", label: "Green", color: "#22c55e" },
  { key: "aqua", label: "Aqua", color: "#06b6d4" },
  { key: "blue", label: "Blue", color: "#3b82f6" },
  { key: "purple", label: "Purple", color: "#a855f7" },
  { key: "magenta", label: "Magenta", color: "#ec4899" },
] as const;

type HSLChannel = (typeof HSL_CHANNELS)[number]["key"];

interface HSLAdjustments {
  red: { hue: number; saturation: number; luminance: number };
  orange: { hue: number; saturation: number; luminance: number };
  yellow: { hue: number; saturation: number; luminance: number };
  green: { hue: number; saturation: number; luminance: number };
  aqua: { hue: number; saturation: number; luminance: number };
  blue: { hue: number; saturation: number; luminance: number };
  purple: { hue: number; saturation: number; luminance: number };
  magenta: { hue: number; saturation: number; luminance: number };
}

const defaultHSLChannel = { hue: 0, saturation: 0, luminance: 0 };

const defaultHSLAdjustments: HSLAdjustments = {
  red: { ...defaultHSLChannel },
  orange: { ...defaultHSLChannel },
  yellow: { ...defaultHSLChannel },
  green: { ...defaultHSLChannel },
  aqua: { ...defaultHSLChannel },
  blue: { ...defaultHSLChannel },
  purple: { ...defaultHSLChannel },
  magenta: { ...defaultHSLChannel },
};

// Detail adjustments (Sharpening & Noise Reduction & Grain)
interface DetailAdjustments {
  // Sharpening
  sharpAmount: number;
  sharpRadius: number;
  sharpDetail: number;
  sharpMasking: number;
  // Noise Reduction
  noiseLuminance: number;
  noiseDetail: number;
  noiseContrast: number;
  noiseColor: number;
  noiseColorDetail: number;
  noiseSmoothness: number;
  // Grain
  grainAmount: number;
  grainSize: number;
  grainRoughness: number;
}

const defaultDetailAdjustments: DetailAdjustments = {
  sharpAmount: 0,
  sharpRadius: 1.0,
  sharpDetail: 25,
  sharpMasking: 0,
  noiseLuminance: 0,
  noiseDetail: 50,
  noiseContrast: 50,
  noiseColor: 0,
  noiseColorDetail: 50,
  noiseSmoothness: 50,
  grainAmount: 0,
  grainSize: 25,
  grainRoughness: 50,
};

// Lens Corrections
interface LensCorrections {
  distortion: number;
  vignetteAmount: number;
  vignetteMidpoint: number;
  vignetteRoundness: number;
  vignetteFeather: number;
  defringePurple: number;
  defringePurpleHue: number;
  defringeGreen: number;
  defringeGreenHue: number;
}

const defaultLensCorrections: LensCorrections = {
  distortion: 0,
  vignetteAmount: 0,
  vignetteMidpoint: 50,
  vignetteRoundness: 0,
  vignetteFeather: 50,
  defringePurple: 0,
  defringePurpleHue: 0,
  defringeGreen: 0,
  defringeGreenHue: 0,
};

// Tone Curve - control points for each channel
type CurvePoint = { x: number; y: number };
type CurveChannel = "rgb" | "red" | "green" | "blue";

interface ToneCurveState {
  rgb: CurvePoint[];
  red: CurvePoint[];
  green: CurvePoint[];
  blue: CurvePoint[];
}

const defaultCurvePoints: CurvePoint[] = [
  { x: 0, y: 0 },
  { x: 1, y: 1 },
];

const defaultToneCurve: ToneCurveState = {
  rgb: [...defaultCurvePoints],
  red: [...defaultCurvePoints],
  green: [...defaultCurvePoints],
  blue: [...defaultCurvePoints],
};

// Film presets - processed in Rust with per-channel color grading
const FILM_PRESETS = [
  { id: 0, name: "Kodak Portra 400", color: "#e8a87c" },
  { id: 1, name: "Fuji Velvia 50", color: "#e85d75" },
  { id: 2, name: "Kodak Gold 200", color: "#d4a03c" },
  { id: 3, name: "Cinestill 800T", color: "#5da0e8" },
  { id: 4, name: "Kodak Ektar 100", color: "#e87c5d" },
  { id: 5, name: "Fuji Pro 400H", color: "#7ce8a8" },
  { id: 6, name: "Ilford HP5", color: "#888888" },
  { id: 7, name: "Kodak T-Max 400", color: "#666666" },
  { id: 8, name: "Matte Film", color: "#a88cd4" },
  { id: 9, name: "Faded Vintage", color: "#c9b896" },
  { id: 10, name: "High Contrast", color: "#ffffff" },
  { id: 11, name: "Soft & Dreamy", color: "#a8d4e8" },
];

// Image item in library
interface ImageItem {
  id: string;
  name: string;
  image: HTMLImageElement;
  originalImageData: ImageData;
  thumbnailUrl: string;
  adjustments: Adjustments;
  hslAdjustments: HSLAdjustments;
  detailAdjustments: DetailAdjustments;
  lensCorrections: LensCorrections;
  toneCurve: ToneCurveState;
  activePreset: number | null;
}

// Tone Curve Component - optimized with useMemo and useCallback
interface ToneCurveEditorProps {
  points: CurvePoint[];
  channel: CurveChannel;
  onChange: (points: CurvePoint[]) => void;
}

const CURVE_SIZE = 200;
const POINT_RADIUS = 6;
const GRID_LINES = 4;

const channelColors: Record<CurveChannel, string> = {
  rgb: "#ffffff",
  red: "#ef4444",
  green: "#22c55e",
  blue: "#3b82f6",
};

function ToneCurveEditor({ points, channel, onChange }: ToneCurveEditorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [draggingIndex, setDraggingIndex] = useState<number | null>(null);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  // Generate spline curve points for smooth rendering
  const curvePoints = useMemo(() => {
    if (points.length < 2) return [];

    const result: { x: number; y: number }[] = [];
    const sorted = [...points].sort((a, b) => a.x - b.x);

    for (let i = 0; i <= CURVE_SIZE; i++) {
      const x = i / CURVE_SIZE;

      // Find segment
      let seg = 0;
      for (let j = 0; j < sorted.length - 1; j++) {
        if (x >= sorted[j].x && x <= sorted[j + 1].x) {
          seg = j;
          break;
        }
        if (j === sorted.length - 2) seg = j;
      }

      // Handle edges
      if (x <= sorted[0].x) {
        result.push({ x: i, y: (1 - sorted[0].y) * CURVE_SIZE });
        continue;
      }
      if (x >= sorted[sorted.length - 1].x) {
        result.push({ x: i, y: (1 - sorted[sorted.length - 1].y) * CURVE_SIZE });
        continue;
      }

      const x0 = sorted[seg].x;
      const x1 = sorted[seg + 1].x;
      const y0 = sorted[seg].y;
      const y1 = sorted[seg + 1].y;

      // Catmull-Rom interpolation
      const t = x1 !== x0 ? (x - x0) / (x1 - x0) : 0;

      const m0 = seg > 0 ? ((y1 - sorted[seg - 1].y) / (x1 - sorted[seg - 1].x)) * (x1 - x0) : y1 - y0;

      const m1 =
        seg < sorted.length - 2 ? ((sorted[seg + 2].y - y0) / (sorted[seg + 2].x - x0)) * (x1 - x0) : y1 - y0;

      const t2 = t * t;
      const t3 = t2 * t;
      const h00 = 2 * t3 - 3 * t2 + 1;
      const h10 = t3 - 2 * t2 + t;
      const h01 = -2 * t3 + 3 * t2;
      const h11 = t3 - t2;

      const y = Math.max(0, Math.min(1, h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1));
      result.push({ x: i, y: (1 - y) * CURVE_SIZE });
    }

    return result;
  }, [points]);

  // Draw the curve
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d")!;
    const dpr = window.devicePixelRatio || 1;

    canvas.width = CURVE_SIZE * dpr;
    canvas.height = CURVE_SIZE * dpr;
    ctx.scale(dpr, dpr);

    // Clear
    ctx.fillStyle = "#171717";
    ctx.fillRect(0, 0, CURVE_SIZE, CURVE_SIZE);

    // Grid
    ctx.strokeStyle = "#2a2a2a";
    ctx.lineWidth = 1;
    for (let i = 1; i < GRID_LINES; i++) {
      const pos = (i / GRID_LINES) * CURVE_SIZE;
      ctx.beginPath();
      ctx.moveTo(pos, 0);
      ctx.lineTo(pos, CURVE_SIZE);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, pos);
      ctx.lineTo(CURVE_SIZE, pos);
      ctx.stroke();
    }

    // Diagonal reference line
    ctx.strokeStyle = "#333";
    ctx.beginPath();
    ctx.moveTo(0, CURVE_SIZE);
    ctx.lineTo(CURVE_SIZE, 0);
    ctx.stroke();

    // Draw curve
    if (curvePoints.length > 1) {
      ctx.strokeStyle = channelColors[channel];
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(curvePoints[0].x, curvePoints[0].y);
      for (let i = 1; i < curvePoints.length; i++) {
        ctx.lineTo(curvePoints[i].x, curvePoints[i].y);
      }
      ctx.stroke();
    }

    // Draw control points
    const sorted = [...points].sort((a, b) => a.x - b.x);
    sorted.forEach((point, idx) => {
      const px = point.x * CURVE_SIZE;
      const py = (1 - point.y) * CURVE_SIZE;

      ctx.beginPath();
      ctx.arc(px, py, POINT_RADIUS, 0, Math.PI * 2);
      ctx.fillStyle = hoveredIndex === idx || draggingIndex === idx ? channelColors[channel] : "#171717";
      ctx.fill();
      ctx.strokeStyle = channelColors[channel];
      ctx.lineWidth = 2;
      ctx.stroke();
    });
  }, [points, curvePoints, channel, hoveredIndex, draggingIndex]);

  const getMousePos = useCallback((e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width)),
      y: Math.max(0, Math.min(1, 1 - (e.clientY - rect.top) / rect.height)),
    };
  }, []);

  const findPointIndex = useCallback(
    (pos: { x: number; y: number }) => {
      const threshold = (POINT_RADIUS * 2) / CURVE_SIZE;
      for (let i = 0; i < points.length; i++) {
        const dx = points[i].x - pos.x;
        const dy = points[i].y - pos.y;
        if (Math.sqrt(dx * dx + dy * dy) < threshold) {
          return i;
        }
      }
      return -1;
    },
    [points],
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      const pos = getMousePos(e);
      const idx = findPointIndex(pos);

      if (idx >= 0) {
        setDraggingIndex(idx);
      } else {
        // Add new point
        const newPoints = [...points, { x: pos.x, y: pos.y }].sort((a, b) => a.x - b.x);
        onChange(newPoints);
        // Find the new index after sorting
        const newIdx = newPoints.findIndex((p) => p.x === pos.x && p.y === pos.y);
        setDraggingIndex(newIdx);
      }
    },
    [points, onChange, getMousePos, findPointIndex],
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      const pos = getMousePos(e);

      if (draggingIndex !== null) {
        const newPoints = [...points];
        // Don't allow moving first/last point horizontally past bounds
        const isFirst = draggingIndex === 0 || points[draggingIndex].x === 0;
        const isLast = draggingIndex === points.length - 1 || points[draggingIndex].x === 1;

        newPoints[draggingIndex] = {
          x: isFirst ? 0 : isLast ? 1 : Math.max(0.01, Math.min(0.99, pos.x)),
          y: Math.max(0, Math.min(1, pos.y)),
        };
        onChange(newPoints);
      } else {
        const idx = findPointIndex(pos);
        setHoveredIndex(idx >= 0 ? idx : null);
      }
    },
    [draggingIndex, points, onChange, getMousePos, findPointIndex],
  );

  const handleMouseUp = useCallback(() => {
    setDraggingIndex(null);
  }, []);

  const handleDoubleClick = useCallback(
    (e: React.MouseEvent) => {
      const pos = getMousePos(e);
      const idx = findPointIndex(pos);

      // Don't delete endpoints
      if (idx > 0 && idx < points.length - 1) {
        const newPoints = points.filter((_, i) => i !== idx);
        onChange(newPoints);
      }
    },
    [points, onChange, getMousePos, findPointIndex],
  );

  return (
    <canvas
      ref={canvasRef}
      width={CURVE_SIZE}
      height={CURVE_SIZE}
      style={{
        width: CURVE_SIZE,
        height: CURVE_SIZE,
        cursor: hoveredIndex !== null ? "pointer" : "crosshair",
      }}
      className="rounded border border-neutral-700"
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onDoubleClick={handleDoubleClick}
    />
  );
}

function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [images, setImages] = useState<ImageItem[]>([]);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [activeImageId, setActiveImageId] = useState<string | null>(null);
  const [wasmReady, setWasmReady] = useState(false);
  const [basicPanelOpen, setBasicPanelOpen] = useState(true);
  const [hslPanelOpen, setHslPanelOpen] = useState(false);
  const [detailPanelOpen, setDetailPanelOpen] = useState(false);
  const [lensPanelOpen, setLensPanelOpen] = useState(false);
  const [toneCurvePanelOpen, setToneCurvePanelOpen] = useState(false);
  const [activeCurveChannel, setActiveCurveChannel] = useState<CurveChannel>("rgb");
  const [activeHSLMode, setActiveHSLMode] = useState<"hue" | "saturation" | "luminance">("hue");
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Get the active image for display
  const activeImage = images.find((img) => img.id === activeImageId);

  // Get current adjustments (from active image or defaults)
  const currentAdjustments = activeImage?.adjustments ?? defaultAdjustments;
  const currentHslAdjustments = activeImage?.hslAdjustments ?? defaultHSLAdjustments;
  const currentDetailAdjustments = activeImage?.detailAdjustments ?? defaultDetailAdjustments;
  const currentLensCorrections = activeImage?.lensCorrections ?? defaultLensCorrections;
  const currentToneCurve = activeImage?.toneCurve ?? defaultToneCurve;
  const currentActivePreset = activeImage?.activePreset ?? null;

  // Initialize WASM on mount
  useEffect(() => {
    init().then(() => setWasmReady(true));
  }, []);

  const generateId = () => `img_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    Array.from(files).forEach((file) => {
      const img = new Image();
      img.onload = () => {
        // Create a temporary canvas to get image data
        const tempCanvas = document.createElement("canvas");
        tempCanvas.width = img.width;
        tempCanvas.height = img.height;
        const tempCtx = tempCanvas.getContext("2d")!;
        tempCtx.drawImage(img, 0, 0);
        const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);

        // Generate thumbnail
        const thumbCanvas = document.createElement("canvas");
        const thumbSize = 160;
        const scale = Math.min(thumbSize / img.width, thumbSize / img.height);
        thumbCanvas.width = img.width * scale;
        thumbCanvas.height = img.height * scale;
        const thumbCtx = thumbCanvas.getContext("2d")!;
        thumbCtx.drawImage(img, 0, 0, thumbCanvas.width, thumbCanvas.height);
        const thumbnailUrl = thumbCanvas.toDataURL("image/jpeg", 0.7);

        const newImage: ImageItem = {
          id: generateId(),
          name: file.name,
          image: img,
          originalImageData: imageData,
          thumbnailUrl,
          adjustments: { ...defaultAdjustments },
          hslAdjustments: JSON.parse(JSON.stringify(defaultHSLAdjustments)),
          detailAdjustments: { ...defaultDetailAdjustments },
          lensCorrections: { ...defaultLensCorrections },
          toneCurve: JSON.parse(JSON.stringify(defaultToneCurve)),
          activePreset: null,
        };

        setImages((prev) => {
          const updated = [...prev, newImage];
          // Auto-select first image if none selected
          if (prev.length === 0) {
            setActiveImageId(newImage.id);
            setSelectedIds(new Set([newImage.id]));
          }
          return updated;
        });
      };
      img.src = URL.createObjectURL(file);
    });

    // Reset file input so the same files can be selected again
    e.target.value = "";
  };

  // Handle thumbnail click with multi-select support
  const handleThumbnailClick = (id: string, e: React.MouseEvent) => {
    if (e.ctrlKey || e.metaKey) {
      // Ctrl/Cmd + click: toggle selection
      setSelectedIds((prev) => {
        const next = new Set(prev);
        if (next.has(id)) {
          next.delete(id);
          // If we're deselecting the active image, set a new active
          if (activeImageId === id && next.size > 0) {
            setActiveImageId([...next][0]);
          }
        } else {
          next.add(id);
        }
        return next;
      });
    } else if (e.shiftKey && activeImageId) {
      // Shift + click: range selection
      const currentIndex = images.findIndex((img) => img.id === activeImageId);
      const clickedIndex = images.findIndex((img) => img.id === id);
      const start = Math.min(currentIndex, clickedIndex);
      const end = Math.max(currentIndex, clickedIndex);
      const rangeIds = images.slice(start, end + 1).map((img) => img.id);
      setSelectedIds(new Set(rangeIds));
    } else {
      // Regular click: single selection
      setSelectedIds(new Set([id]));
      setActiveImageId(id);
    }
  };

  // Check if HSL has any adjustments
  const hasHSLAdjustments = (hsl: HSLAdjustments) => {
    return HSL_CHANNELS.some(
      (ch) => hsl[ch.key].hue !== 0 || hsl[ch.key].saturation !== 0 || hsl[ch.key].luminance !== 0,
    );
  };

  // Check if detail has any adjustments
  const hasDetailAdjustments = (detail: DetailAdjustments) => {
    return (
      detail.sharpAmount !== 0 ||
      detail.noiseLuminance !== 0 ||
      detail.noiseColor !== 0 ||
      detail.grainAmount !== 0
    );
  };

  // Check if lens corrections have any adjustments
  const hasLensCorrections = (lens: LensCorrections) => {
    return (
      lens.distortion !== 0 ||
      lens.vignetteAmount !== 0 ||
      lens.defringePurple !== 0 ||
      lens.defringeGreen !== 0
    );
  };

  // Check if tone curve has any adjustments (not default linear)
  const hasToneCurveAdjustments = (curve: ToneCurveState) => {
    const isDefault = (pts: CurvePoint[]) =>
      pts.length === 2 &&
      Math.abs(pts[0].x) < 0.001 &&
      Math.abs(pts[0].y) < 0.001 &&
      Math.abs(pts[1].x - 1) < 0.001 &&
      Math.abs(pts[1].y - 1) < 0.001;

    return (
      !isDefault(curve.rgb) || !isDefault(curve.red) || !isDefault(curve.green) || !isDefault(curve.blue)
    );
  };

  // Process image with adjustments
  const processImage = useCallback(
    (
      originalData: ImageData,
      adjustments: Adjustments,
      hslAdjustments: HSLAdjustments,
      detailAdjustments: DetailAdjustments,
      lensCorrections: LensCorrections,
      toneCurve: ToneCurveState,
      preset: number | null,
    ): Uint8Array | Uint8ClampedArray => {
      if (!wasmReady) return originalData.data;

      const width = originalData.width;
      const height = originalData.height;

      // If preset is active, just apply preset
      if (preset !== null) {
        return apply_preset(new Uint8Array(originalData.data), preset);
      }

      const { exposure, contrast, highlights, shadows, whites, blacks } = adjustments;
      const isBasicDefault =
        exposure === 0 && contrast === 0 && highlights === 0 && shadows === 0 && whites === 0 && blacks === 0;
      const isHSLDefault = !hasHSLAdjustments(hslAdjustments);
      const isDetailDefault = !hasDetailAdjustments(detailAdjustments);
      const isLensDefault = !hasLensCorrections(lensCorrections);
      const isToneCurveDefault = !hasToneCurveAdjustments(toneCurve);

      if (isBasicDefault && isHSLDefault && isDetailDefault && isLensDefault && isToneCurveDefault) {
        return originalData.data;
      }

      let processedData: Uint8Array = new Uint8Array(originalData.data);

      // 1. Apply lens distortion first (geometric transformation)
      if (lensCorrections.distortion !== 0) {
        processedData = apply_distortion(processedData, width, height, lensCorrections.distortion);
      }

      // 2. Apply basic adjustments
      if (!isBasicDefault) {
        processedData = adjust_image(processedData, exposure, contrast, highlights, shadows, whites, blacks);
      }

      // 3. Apply HSL adjustments
      if (!isHSLDefault) {
        processedData = adjust_hsl(
          processedData,
          hslAdjustments.red.hue,
          hslAdjustments.red.saturation,
          hslAdjustments.red.luminance,
          hslAdjustments.orange.hue,
          hslAdjustments.orange.saturation,
          hslAdjustments.orange.luminance,
          hslAdjustments.yellow.hue,
          hslAdjustments.yellow.saturation,
          hslAdjustments.yellow.luminance,
          hslAdjustments.green.hue,
          hslAdjustments.green.saturation,
          hslAdjustments.green.luminance,
          hslAdjustments.aqua.hue,
          hslAdjustments.aqua.saturation,
          hslAdjustments.aqua.luminance,
          hslAdjustments.blue.hue,
          hslAdjustments.blue.saturation,
          hslAdjustments.blue.luminance,
          hslAdjustments.purple.hue,
          hslAdjustments.purple.saturation,
          hslAdjustments.purple.luminance,
          hslAdjustments.magenta.hue,
          hslAdjustments.magenta.saturation,
          hslAdjustments.magenta.luminance,
        );
      }

      // 4. Apply defringe (chromatic aberration)
      if (lensCorrections.defringePurple > 0 || lensCorrections.defringeGreen > 0) {
        processedData = apply_defringe(
          processedData,
          width,
          height,
          lensCorrections.defringePurple,
          lensCorrections.defringePurpleHue,
          lensCorrections.defringeGreen,
          lensCorrections.defringeGreenHue,
        );
      }

      // 5. Apply noise reduction (before sharpening)
      if (detailAdjustments.noiseLuminance > 0 || detailAdjustments.noiseColor > 0) {
        processedData = apply_noise_reduction(
          processedData,
          width,
          height,
          detailAdjustments.noiseLuminance,
          detailAdjustments.noiseDetail,
          detailAdjustments.noiseContrast,
          detailAdjustments.noiseColor,
          detailAdjustments.noiseColorDetail,
          detailAdjustments.noiseSmoothness,
        );
      }

      // 6. Apply sharpening
      if (detailAdjustments.sharpAmount > 0) {
        processedData = apply_sharpening(
          processedData,
          width,
          height,
          detailAdjustments.sharpAmount,
          detailAdjustments.sharpRadius,
          detailAdjustments.sharpDetail,
          detailAdjustments.sharpMasking,
        );
      }

      // 7. Apply grain
      if (detailAdjustments.grainAmount > 0) {
        processedData = apply_grain(
          processedData,
          width,
          height,
          detailAdjustments.grainAmount,
          detailAdjustments.grainSize,
          detailAdjustments.grainRoughness,
        );
      }

      // 8. Apply tone curve (very fast - LUT based)
      if (!isToneCurveDefault) {
        const flattenPoints = (pts: CurvePoint[]) => {
          const sorted = [...pts].sort((a, b) => a.x - b.x);
          return new Float32Array(sorted.flatMap((p) => [p.x, p.y]));
        };

        processedData = apply_tone_curve(
          processedData,
          flattenPoints(toneCurve.rgb),
          flattenPoints(toneCurve.red),
          flattenPoints(toneCurve.green),
          flattenPoints(toneCurve.blue),
        );
      }

      // 9. Apply vignette (last, as it's a creative effect)
      if (lensCorrections.vignetteAmount !== 0) {
        processedData = apply_vignette(
          processedData,
          width,
          height,
          lensCorrections.vignetteAmount,
          lensCorrections.vignetteMidpoint,
          lensCorrections.vignetteRoundness,
          lensCorrections.vignetteFeather,
        );
      }

      return processedData;
    },
    [wasmReady],
  );

  // Render active image to canvas
  useEffect(() => {
    if (!activeImage || !canvasRef.current || !wasmReady) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d")!;
    canvas.width = activeImage.originalImageData.width;
    canvas.height = activeImage.originalImageData.height;

    const processedData = processImage(
      activeImage.originalImageData,
      activeImage.adjustments,
      activeImage.hslAdjustments,
      activeImage.detailAdjustments,
      activeImage.lensCorrections,
      activeImage.toneCurve,
      activeImage.activePreset,
    );

    ctx.putImageData(new ImageData(new Uint8ClampedArray(processedData), canvas.width, canvas.height), 0, 0);
  }, [activeImage, wasmReady, processImage]);

  // Update adjustments for all selected images
  const updateSelectedAdjustments = (updater: (adj: Adjustments) => Adjustments) => {
    setImages((prev) =>
      prev.map((img) => {
        if (selectedIds.has(img.id)) {
          return {
            ...img,
            adjustments: updater(img.adjustments),
            activePreset: null, // Clear preset when manually adjusting
          };
        }
        return img;
      }),
    );
  };

  // Update HSL for all selected images
  const updateSelectedHSL = (updater: (hsl: HSLAdjustments) => HSLAdjustments) => {
    setImages((prev) =>
      prev.map((img) => {
        if (selectedIds.has(img.id)) {
          return {
            ...img,
            hslAdjustments: updater(img.hslAdjustments),
            activePreset: null,
          };
        }
        return img;
      }),
    );
  };

  const handleSliderChange = (key: string, value: number) => {
    updateSelectedAdjustments((adj) => ({ ...adj, [key]: value }));
  };

  const handleHSLChange = (
    channel: HSLChannel,
    property: "hue" | "saturation" | "luminance",
    value: number,
  ) => {
    updateSelectedHSL((hsl) => ({
      ...hsl,
      [channel]: { ...hsl[channel], [property]: value },
    }));
  };

  const resetHSL = () => {
    updateSelectedHSL(() => JSON.parse(JSON.stringify(defaultHSLAdjustments)));
  };

  const resetAll = () => {
    setImages((prev) =>
      prev.map((img) => {
        if (selectedIds.has(img.id)) {
          return {
            ...img,
            adjustments: { ...defaultAdjustments },
            hslAdjustments: JSON.parse(JSON.stringify(defaultHSLAdjustments)),
            detailAdjustments: { ...defaultDetailAdjustments },
            lensCorrections: { ...defaultLensCorrections },
            toneCurve: JSON.parse(JSON.stringify(defaultToneCurve)),
            activePreset: null,
          };
        }
        return img;
      }),
    );
  };

  // Update tone curve for all selected images
  const updateSelectedToneCurve = (updater: (curve: ToneCurveState) => ToneCurveState) => {
    setImages((prev) =>
      prev.map((img) => {
        if (selectedIds.has(img.id)) {
          return {
            ...img,
            toneCurve: updater(img.toneCurve),
            activePreset: null,
          };
        }
        return img;
      }),
    );
  };

  const handleToneCurveChange = (channel: CurveChannel, points: CurvePoint[]) => {
    updateSelectedToneCurve((curve) => ({
      ...curve,
      [channel]: points,
    }));
  };

  const resetToneCurve = () => {
    updateSelectedToneCurve(() => JSON.parse(JSON.stringify(defaultToneCurve)));
  };

  // Update detail adjustments for all selected images
  const updateSelectedDetail = (updater: (detail: DetailAdjustments) => DetailAdjustments) => {
    setImages((prev) =>
      prev.map((img) => {
        if (selectedIds.has(img.id)) {
          return {
            ...img,
            detailAdjustments: updater(img.detailAdjustments),
            activePreset: null,
          };
        }
        return img;
      }),
    );
  };

  // Update lens corrections for all selected images
  const updateSelectedLens = (updater: (lens: LensCorrections) => LensCorrections) => {
    setImages((prev) =>
      prev.map((img) => {
        if (selectedIds.has(img.id)) {
          return {
            ...img,
            lensCorrections: updater(img.lensCorrections),
            activePreset: null,
          };
        }
        return img;
      }),
    );
  };

  const handleDetailChange = (key: keyof DetailAdjustments, value: number) => {
    updateSelectedDetail((detail) => ({ ...detail, [key]: value }));
  };

  const handleLensChange = (key: keyof LensCorrections, value: number) => {
    updateSelectedLens((lens) => ({ ...lens, [key]: value }));
  };

  const resetDetail = () => {
    updateSelectedDetail(() => ({ ...defaultDetailAdjustments }));
  };

  const resetLens = () => {
    updateSelectedLens(() => ({ ...defaultLensCorrections }));
  };

  const handlePresetClick = (presetId: number) => {
    setImages((prev) =>
      prev.map((img) => {
        if (selectedIds.has(img.id)) {
          return { ...img, activePreset: presetId };
        }
        return img;
      }),
    );
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  // Select all images
  const selectAll = () => {
    setSelectedIds(new Set(images.map((img) => img.id)));
  };

  // Remove selected images
  const removeSelected = () => {
    setImages((prev) => {
      const remaining = prev.filter((img) => !selectedIds.has(img.id));
      if (remaining.length > 0 && selectedIds.has(activeImageId || "")) {
        setActiveImageId(remaining[0].id);
      } else if (remaining.length === 0) {
        setActiveImageId(null);
      }
      setSelectedIds(new Set(remaining.length > 0 ? [remaining[0]?.id].filter(Boolean) : []));
      return remaining;
    });
  };

  // Export images
  const exportImages = async () => {
    if (!wasmReady || selectedIds.size === 0) return;

    const imagesToExport = images.filter((img) => selectedIds.has(img.id));

    for (const img of imagesToExport) {
      // Process the image with all current adjustments
      const processedData = processImage(
        img.originalImageData,
        img.adjustments,
        img.hslAdjustments,
        img.detailAdjustments,
        img.lensCorrections,
        img.toneCurve,
        img.activePreset,
      );

      // Create a canvas for the full-size export
      const exportCanvas = document.createElement("canvas");
      exportCanvas.width = img.originalImageData.width;
      exportCanvas.height = img.originalImageData.height;
      const ctx = exportCanvas.getContext("2d")!;

      // Put the processed image data on the canvas
      ctx.putImageData(
        new ImageData(new Uint8ClampedArray(processedData), exportCanvas.width, exportCanvas.height),
        0,
        0,
      );

      // Generate filename: remove extension and add suffix
      const nameWithoutExt = img.name.replace(/\.[^/.]+$/, "");
      const filename = `${nameWithoutExt}_edited.png`;

      // Convert to blob and trigger download
      exportCanvas.toBlob(
        (blob) => {
          if (blob) {
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
          }
        },
        "image/png",
        1.0,
      );

      // Small delay between downloads to prevent browser blocking
      if (imagesToExport.length > 1) {
        await new Promise((resolve) => setTimeout(resolve, 300));
      }
    }
  };

  return (
    <div className="flex flex-col h-screen bg-neutral-900 text-neutral-300 font-sans selection:bg-blue-500 selection:text-white">
      {/* Header */}
      <header className="flex items-center justify-between h-12 px-4 border-b border-neutral-800 bg-neutral-900 shrink-0">
        <div className="flex items-center gap-4">
          <span className="font-bold text-lg text-white">OpenRoom</span>
          <nav className="flex gap-2 text-sm font-medium">
            <button className="px-3 py-1 hover:bg-neutral-800 rounded transition-colors text-neutral-400 hover:text-white">
              Library
            </button>
            <button className="px-3 py-1 bg-neutral-800 text-white rounded transition-colors">Develop</button>
          </nav>
        </div>
        <div className="flex items-center gap-2">
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFile}
            className="hidden"
            accept="image/*"
            multiple
          />
          <button
            onClick={triggerFileInput}
            className="px-3 py-1.5 text-xs font-semibold bg-neutral-700 hover:bg-neutral-600 text-white rounded transition-colors"
          >
            Import Photos
          </button>
          <button
            onClick={exportImages}
            disabled={selectedIds.size === 0}
            className={`px-3 py-1.5 text-xs font-semibold rounded transition-colors ${
              selectedIds.size > 0
                ? "bg-blue-600 hover:bg-blue-500 text-white"
                : "bg-neutral-700 text-neutral-500 cursor-not-allowed"
            }`}
          >
            Export{selectedIds.size > 1 ? ` (${selectedIds.size})` : ""}
          </button>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left Sidebar */}
        <aside className="hidden md:flex flex-col w-64 bg-neutral-900 border-r border-neutral-800 shrink-0">
          <div className="p-3 border-b border-neutral-800">
            <h3 className="text-xs font-bold uppercase tracking-wider text-neutral-500">Navigator</h3>
            <div className="mt-2 h-32 bg-neutral-800 rounded border border-neutral-700 flex items-center justify-center overflow-hidden relative">
              {activeImage ? (
                <img
                  src={activeImage.image.src}
                  alt="Nav"
                  className="w-full h-full object-contain opacity-70"
                />
              ) : (
                <span className="text-xs text-neutral-600">No Image</span>
              )}
            </div>
          </div>
          <div className="flex-1 overflow-y-auto">
            <div className="p-3">
              <h3 className="text-xs font-bold uppercase tracking-wider text-neutral-500 mb-2">
                Film Presets
              </h3>
              <ul className="space-y-0.5 text-sm text-neutral-400">
                <li
                  onClick={resetAll}
                  className={`px-2 py-1.5 hover:bg-neutral-800 hover:text-white rounded cursor-pointer transition-colors flex items-center gap-2 ${
                    currentActivePreset === null &&
                    currentAdjustments.exposure === 0 &&
                    currentAdjustments.contrast === 0
                      ? "bg-neutral-800 text-white"
                      : ""
                  }`}
                >
                  <span className="w-2 h-2 rounded-full bg-neutral-500"></span>
                  None (Reset)
                </li>
                {FILM_PRESETS.map((preset) => (
                  <li
                    key={preset.id}
                    onClick={() => handlePresetClick(preset.id)}
                    className={`px-2 py-1.5 hover:bg-neutral-800 hover:text-white rounded cursor-pointer transition-colors flex items-center gap-2 ${
                      currentActivePreset === preset.id ? "bg-neutral-800 text-white" : ""
                    }`}
                  >
                    <span className="w-2 h-2 rounded-full" style={{ background: preset.color }}></span>
                    {preset.name}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </aside>

        {/* Center - Canvas */}
        <main className="flex-1 bg-neutral-950 flex flex-col relative overflow-hidden">
          <div className="flex-1 overflow-auto flex items-center justify-center p-8">
            <canvas
              ref={canvasRef}
              className="max-w-full max-h-full shadow-2xl border border-neutral-900 object-contain"
              style={{ display: activeImage ? "block" : "none" }}
            ></canvas>
            {!activeImage && (
              <div className="text-center text-neutral-600">
                <button
                  onClick={triggerFileInput}
                  className="w-16 h-16 mx-auto bg-neutral-900 rounded-full flex items-center justify-center mb-4 border border-neutral-800 hover:bg-neutral-800 hover:border-neutral-700 transition-colors cursor-pointer"
                >
                  <span className="text-2xl">+</span>
                </button>
                <p className="text-sm font-medium mb-1 text-neutral-500">No photo selected</p>
                <button
                  onClick={triggerFileInput}
                  className="text-xs text-blue-500 hover:text-blue-400 hover:underline"
                >
                  Click to import
                </button>
              </div>
            )}
          </div>

          {/* Filmstrip */}
          <div className="h-28 bg-neutral-900 border-t border-neutral-800 flex flex-col shrink-0">
            {/* Filmstrip toolbar */}
            {images.length > 0 && (
              <div className="flex items-center justify-between px-4 py-1 border-b border-neutral-800 bg-neutral-900/80">
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-neutral-500 uppercase tracking-wide">
                    {selectedIds.size} of {images.length} selected
                  </span>
                </div>
                <div className="flex items-center gap-1">
                  <button
                    onClick={selectAll}
                    className="px-2 py-0.5 text-[10px] text-neutral-400 hover:text-white hover:bg-neutral-700 rounded transition-colors"
                  >
                    Select All
                  </button>
                  {selectedIds.size > 0 && (
                    <button
                      onClick={removeSelected}
                      className="px-2 py-0.5 text-[10px] text-red-400 hover:text-red-300 hover:bg-neutral-700 rounded transition-colors"
                    >
                      Remove
                    </button>
                  )}
                </div>
              </div>
            )}

            {/* Thumbnails */}
            <div className="flex-1 flex items-center px-4 overflow-x-auto gap-2 filmstrip-scroll">
              {images.length > 0 ? (
                images.map((img) => (
                  <div
                    key={img.id}
                    onClick={(e) => handleThumbnailClick(img.id, e)}
                    className={`relative w-20 h-16 shrink-0 rounded overflow-hidden cursor-pointer group transition-all ${
                      activeImageId === img.id
                        ? "ring-2 ring-white ring-offset-1 ring-offset-neutral-900"
                        : selectedIds.has(img.id)
                        ? "ring-2 ring-blue-500 ring-offset-1 ring-offset-neutral-900"
                        : "ring-1 ring-neutral-700 hover:ring-neutral-500"
                    }`}
                  >
                    <img src={img.thumbnailUrl} alt={img.name} className="w-full h-full object-cover" />
                    {/* Selection indicator */}
                    {selectedIds.has(img.id) && selectedIds.size > 1 && (
                      <div className="absolute top-1 right-1 w-4 h-4 bg-blue-500 rounded-full flex items-center justify-center">
                        <span className="text-[8px] text-white font-bold">✓</span>
                      </div>
                    )}
                    {/* Filename tooltip on hover */}
                    <div className="absolute inset-x-0 bottom-0 bg-black/70 px-1 py-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                      <span className="text-[9px] text-neutral-300 truncate block">{img.name}</span>
                    </div>
                  </div>
                ))
              ) : (
                <span className="text-xs text-neutral-700 w-full text-center">
                  Import photos to start editing
                </span>
              )}
            </div>
          </div>
        </main>

        {/* Right Sidebar - Tools */}
        <aside className="w-80 bg-neutral-900 border-l border-neutral-800 flex flex-col shrink-0">
          <div className="p-2 border-b border-neutral-800 flex justify-between items-center bg-neutral-900">
            <h3 className="text-xs font-bold uppercase tracking-wider text-neutral-500 pl-2">Histogram</h3>
            {selectedIds.size > 1 && (
              <span className="text-[10px] text-blue-400 pr-2">Editing {selectedIds.size} images</span>
            )}
          </div>
          <div className="h-32 bg-neutral-800 border-b border-neutral-800 shrink-0 flex items-end justify-center pb-2 gap-0.5 px-4 pt-4">
            {[...Array(40)].map((_, i) => (
              <div
                key={i}
                className="flex-1 bg-neutral-600 rounded-t-sm opacity-50"
                style={{ height: `${Math.random() * 80 + 20}%` }}
              ></div>
            ))}
          </div>

          <div className="flex-1 overflow-y-auto custom-scrollbar">
            <div className="border-b border-neutral-800">
              <button
                onClick={() => setBasicPanelOpen(!basicPanelOpen)}
                className="w-full text-left px-4 py-2 bg-neutral-800 text-xs font-bold uppercase flex justify-between items-center text-neutral-300"
              >
                <span>Basic</span>
                <div className="flex items-center gap-2">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      resetAll();
                    }}
                    className="text-[10px] text-neutral-500 hover:text-white px-2 py-0.5 rounded hover:bg-neutral-700 transition-colors"
                  >
                    Reset
                  </button>
                  <span className={`transition-transform ${basicPanelOpen ? "rotate-90" : ""}`}>▶</span>
                </div>
              </button>
              {basicPanelOpen && (
                <div className="p-4 space-y-5">
                  {SLIDERS.map((slider) => (
                    <div key={slider.key} className="space-y-1.5">
                      <div className="flex justify-between text-[11px] font-medium text-neutral-400 uppercase tracking-wide">
                        <span>{slider.label}</span>
                        <span className="text-neutral-500 tabular-nums">
                          {slider.key === "exposure"
                            ? currentAdjustments[slider.key as keyof Adjustments].toFixed(1)
                            : currentAdjustments[slider.key as keyof Adjustments]}
                        </span>
                      </div>
                      <input
                        type="range"
                        min={slider.min}
                        max={slider.max}
                        step={slider.step}
                        value={currentAdjustments[slider.key as keyof Adjustments]}
                        onChange={(e) => handleSliderChange(slider.key, parseFloat(e.target.value))}
                        className="w-full h-1 bg-neutral-700 rounded-lg appearance-none cursor-pointer accent-neutral-400 hover:accent-white focus:outline-none focus:accent-blue-500"
                      />
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="border-b border-neutral-800">
              <button
                onClick={() => setToneCurvePanelOpen(!toneCurvePanelOpen)}
                className="w-full text-left px-4 py-2 bg-neutral-800 text-xs font-bold uppercase flex justify-between items-center text-neutral-300"
              >
                <span>Tone Curve</span>
                <div className="flex items-center gap-2">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      resetToneCurve();
                    }}
                    className="text-[10px] text-neutral-500 hover:text-white px-2 py-0.5 rounded hover:bg-neutral-700 transition-colors"
                  >
                    Reset
                  </button>
                  <span className={`transition-transform ${toneCurvePanelOpen ? "rotate-90" : ""}`}>▶</span>
                </div>
              </button>
              {toneCurvePanelOpen && (
                <div className="p-4 space-y-4">
                  {/* Channel Tabs */}
                  <div className="flex gap-1 p-1 bg-neutral-800 rounded-lg">
                    {(["rgb", "red", "green", "blue"] as const).map((ch) => (
                      <button
                        key={ch}
                        onClick={() => setActiveCurveChannel(ch)}
                        className={`flex-1 px-2 py-1.5 text-[10px] font-semibold uppercase rounded transition-colors flex items-center justify-center gap-1 ${
                          activeCurveChannel === ch
                            ? "bg-neutral-700 text-white"
                            : "text-neutral-500 hover:text-neutral-300"
                        }`}
                      >
                        {ch !== "rgb" && (
                          <span
                            className="w-2 h-2 rounded-full"
                            style={{ backgroundColor: channelColors[ch] }}
                          />
                        )}
                        {ch === "rgb" ? "RGB" : ch.charAt(0).toUpperCase()}
                      </button>
                    ))}
                  </div>

                  {/* Curve Editor */}
                  <div className="flex justify-center">
                    <ToneCurveEditor
                      points={currentToneCurve[activeCurveChannel]}
                      channel={activeCurveChannel}
                      onChange={(pts) => handleToneCurveChange(activeCurveChannel, pts)}
                    />
                  </div>

                  {/* Instructions */}
                  <div className="text-[10px] text-neutral-500 space-y-1">
                    <p>• Click to add points</p>
                    <p>• Drag to adjust</p>
                    <p>• Double-click to remove</p>
                  </div>

                  {/* Quick presets */}
                  <div className="flex gap-2">
                    <button
                      onClick={() =>
                        handleToneCurveChange(activeCurveChannel, [
                          { x: 0, y: 0 },
                          { x: 0.25, y: 0.2 },
                          { x: 0.75, y: 0.8 },
                          { x: 1, y: 1 },
                        ])
                      }
                      className="flex-1 px-2 py-1.5 text-[10px] font-medium bg-neutral-800 hover:bg-neutral-700 rounded transition-colors"
                    >
                      S-Curve
                    </button>
                    <button
                      onClick={() =>
                        handleToneCurveChange(activeCurveChannel, [
                          { x: 0, y: 0.05 },
                          { x: 1, y: 0.95 },
                        ])
                      }
                      className="flex-1 px-2 py-1.5 text-[10px] font-medium bg-neutral-800 hover:bg-neutral-700 rounded transition-colors"
                    >
                      Fade
                    </button>
                    <button
                      onClick={() =>
                        handleToneCurveChange(activeCurveChannel, [
                          { x: 0, y: 0 },
                          { x: 1, y: 1 },
                        ])
                      }
                      className="flex-1 px-2 py-1.5 text-[10px] font-medium bg-neutral-800 hover:bg-neutral-700 rounded transition-colors"
                    >
                      Linear
                    </button>
                  </div>
                </div>
              )}
            </div>
            <div className="border-b border-neutral-800">
              <button
                onClick={() => setHslPanelOpen(!hslPanelOpen)}
                className="w-full text-left px-4 py-2 bg-neutral-800 text-xs font-bold uppercase flex justify-between items-center text-neutral-300"
              >
                <span>HSL / Color</span>
                <div className="flex items-center gap-2">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      resetHSL();
                    }}
                    className="text-[10px] text-neutral-500 hover:text-white px-2 py-0.5 rounded hover:bg-neutral-700 transition-colors"
                  >
                    Reset
                  </button>
                  <span className={`transition-transform ${hslPanelOpen ? "rotate-90" : ""}`}>▶</span>
                </div>
              </button>
              {hslPanelOpen && (
                <div className="p-4 space-y-4">
                  {/* HSL Mode Tabs */}
                  <div className="flex gap-1 p-1 bg-neutral-800 rounded-lg">
                    {(["hue", "saturation", "luminance"] as const).map((mode) => (
                      <button
                        key={mode}
                        onClick={() => setActiveHSLMode(mode)}
                        className={`flex-1 px-2 py-1.5 text-[10px] font-semibold uppercase rounded transition-colors ${
                          activeHSLMode === mode
                            ? "bg-neutral-700 text-white"
                            : "text-neutral-500 hover:text-neutral-300"
                        }`}
                      >
                        {mode === "hue" ? "Hue" : mode === "saturation" ? "Sat" : "Lum"}
                      </button>
                    ))}
                  </div>

                  {/* Color Channel Sliders */}
                  <div className="space-y-3">
                    {HSL_CHANNELS.map((channel) => (
                      <div key={channel.key} className="space-y-1">
                        <div className="flex justify-between items-center">
                          <div className="flex items-center gap-2">
                            <span
                              className="w-2.5 h-2.5 rounded-full"
                              style={{ backgroundColor: channel.color }}
                            />
                            <span className="text-[11px] font-medium text-neutral-400">{channel.label}</span>
                          </div>
                          <span className="text-[10px] text-neutral-500 tabular-nums w-8 text-right">
                            {activeHSLMode === "hue"
                              ? currentHslAdjustments[channel.key].hue > 0
                                ? `+${currentHslAdjustments[channel.key].hue}`
                                : currentHslAdjustments[channel.key].hue
                              : activeHSLMode === "saturation"
                              ? currentHslAdjustments[channel.key].saturation > 0
                                ? `+${currentHslAdjustments[channel.key].saturation}`
                                : currentHslAdjustments[channel.key].saturation
                              : currentHslAdjustments[channel.key].luminance > 0
                              ? `+${currentHslAdjustments[channel.key].luminance}`
                              : currentHslAdjustments[channel.key].luminance}
                          </span>
                        </div>
                        <input
                          type="range"
                          min={activeHSLMode === "hue" ? -180 : -100}
                          max={activeHSLMode === "hue" ? 180 : 100}
                          step={1}
                          value={currentHslAdjustments[channel.key][activeHSLMode]}
                          onChange={(e) =>
                            handleHSLChange(channel.key, activeHSLMode, parseFloat(e.target.value))
                          }
                          className="w-full h-1 bg-neutral-700 rounded-lg appearance-none cursor-pointer accent-neutral-400 hover:accent-white focus:outline-none"
                          style={{
                            accentColor: channel.color,
                          }}
                        />
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
            {/* Detail Panel */}
            <div className="border-b border-neutral-800">
              <button
                onClick={() => setDetailPanelOpen(!detailPanelOpen)}
                className="w-full text-left px-4 py-2 bg-neutral-800 text-xs font-bold uppercase flex justify-between items-center text-neutral-300"
              >
                <span>Detail</span>
                <div className="flex items-center gap-2">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      resetDetail();
                    }}
                    className="text-[10px] text-neutral-500 hover:text-white px-2 py-0.5 rounded hover:bg-neutral-700 transition-colors"
                  >
                    Reset
                  </button>
                  <span className={`transition-transform ${detailPanelOpen ? "rotate-90" : ""}`}>▶</span>
                </div>
              </button>
              {detailPanelOpen && (
                <div className="p-4 space-y-6">
                  {/* Sharpening Section */}
                  <div className="space-y-4">
                    <h4 className="text-[10px] font-semibold uppercase tracking-wider text-neutral-500 border-b border-neutral-700 pb-1">
                      Sharpening
                    </h4>
                    <div className="space-y-3">
                      {[
                        { key: "sharpAmount", label: "Amount", min: 0, max: 150, step: 1 },
                        { key: "sharpRadius", label: "Radius", min: 0.5, max: 3, step: 0.1 },
                        { key: "sharpDetail", label: "Detail", min: 0, max: 100, step: 1 },
                        { key: "sharpMasking", label: "Masking", min: 0, max: 100, step: 1 },
                      ].map((slider) => (
                        <div key={slider.key} className="space-y-1">
                          <div className="flex justify-between items-center">
                            <span className="text-[11px] font-medium text-neutral-400">{slider.label}</span>
                            <span className="text-[10px] text-neutral-500 tabular-nums">
                              {slider.key === "sharpRadius"
                                ? currentDetailAdjustments[slider.key as keyof DetailAdjustments].toFixed(1)
                                : currentDetailAdjustments[slider.key as keyof DetailAdjustments]}
                            </span>
                          </div>
                          <input
                            type="range"
                            min={slider.min}
                            max={slider.max}
                            step={slider.step}
                            value={currentDetailAdjustments[slider.key as keyof DetailAdjustments]}
                            onChange={(e) =>
                              handleDetailChange(
                                slider.key as keyof DetailAdjustments,
                                parseFloat(e.target.value),
                              )
                            }
                            className="w-full h-1 bg-neutral-700 rounded-lg appearance-none cursor-pointer accent-neutral-400 hover:accent-white"
                          />
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Noise Reduction Section */}
                  <div className="space-y-4">
                    <h4 className="text-[10px] font-semibold uppercase tracking-wider text-neutral-500 border-b border-neutral-700 pb-1">
                      Noise Reduction
                    </h4>
                    <div className="space-y-3">
                      {[
                        { key: "noiseLuminance", label: "Luminance", min: 0, max: 100, step: 1 },
                        { key: "noiseDetail", label: "Detail", min: 0, max: 100, step: 1 },
                        { key: "noiseContrast", label: "Contrast", min: 0, max: 100, step: 1 },
                      ].map((slider) => (
                        <div key={slider.key} className="space-y-1">
                          <div className="flex justify-between items-center">
                            <span className="text-[11px] font-medium text-neutral-400">{slider.label}</span>
                            <span className="text-[10px] text-neutral-500 tabular-nums">
                              {currentDetailAdjustments[slider.key as keyof DetailAdjustments]}
                            </span>
                          </div>
                          <input
                            type="range"
                            min={slider.min}
                            max={slider.max}
                            step={slider.step}
                            value={currentDetailAdjustments[slider.key as keyof DetailAdjustments]}
                            onChange={(e) =>
                              handleDetailChange(
                                slider.key as keyof DetailAdjustments,
                                parseFloat(e.target.value),
                              )
                            }
                            className="w-full h-1 bg-neutral-700 rounded-lg appearance-none cursor-pointer accent-neutral-400 hover:accent-white"
                          />
                        </div>
                      ))}
                    </div>

                    {/* Color Noise Reduction */}
                    <h4 className="text-[10px] font-semibold uppercase tracking-wider text-neutral-500 border-b border-neutral-700 pb-1 mt-4">
                      Color Noise Reduction
                    </h4>
                    <div className="space-y-3">
                      {[
                        { key: "noiseColor", label: "Color", min: 0, max: 100, step: 1 },
                        { key: "noiseColorDetail", label: "Detail", min: 0, max: 100, step: 1 },
                        { key: "noiseSmoothness", label: "Smoothness", min: 0, max: 100, step: 1 },
                      ].map((slider) => (
                        <div key={slider.key} className="space-y-1">
                          <div className="flex justify-between items-center">
                            <span className="text-[11px] font-medium text-neutral-400">{slider.label}</span>
                            <span className="text-[10px] text-neutral-500 tabular-nums">
                              {currentDetailAdjustments[slider.key as keyof DetailAdjustments]}
                            </span>
                          </div>
                          <input
                            type="range"
                            min={slider.min}
                            max={slider.max}
                            step={slider.step}
                            value={currentDetailAdjustments[slider.key as keyof DetailAdjustments]}
                            onChange={(e) =>
                              handleDetailChange(
                                slider.key as keyof DetailAdjustments,
                                parseFloat(e.target.value),
                              )
                            }
                            className="w-full h-1 bg-neutral-700 rounded-lg appearance-none cursor-pointer accent-neutral-400 hover:accent-white"
                          />
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Grain Section */}
                  <div className="space-y-4">
                    <h4 className="text-[10px] font-semibold uppercase tracking-wider text-neutral-500 border-b border-neutral-700 pb-1">
                      Grain
                    </h4>
                    <div className="space-y-3">
                      {[
                        { key: "grainAmount", label: "Amount", min: 0, max: 100, step: 1 },
                        { key: "grainSize", label: "Size", min: 0, max: 100, step: 1 },
                        { key: "grainRoughness", label: "Roughness", min: 0, max: 100, step: 1 },
                      ].map((slider) => (
                        <div key={slider.key} className="space-y-1">
                          <div className="flex justify-between items-center">
                            <span className="text-[11px] font-medium text-neutral-400">{slider.label}</span>
                            <span className="text-[10px] text-neutral-500 tabular-nums">
                              {currentDetailAdjustments[slider.key as keyof DetailAdjustments]}
                            </span>
                          </div>
                          <input
                            type="range"
                            min={slider.min}
                            max={slider.max}
                            step={slider.step}
                            value={currentDetailAdjustments[slider.key as keyof DetailAdjustments]}
                            onChange={(e) =>
                              handleDetailChange(
                                slider.key as keyof DetailAdjustments,
                                parseFloat(e.target.value),
                              )
                            }
                            className="w-full h-1 bg-neutral-700 rounded-lg appearance-none cursor-pointer accent-neutral-400 hover:accent-white"
                          />
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Lens Corrections Panel */}
            <div className="border-b border-neutral-800">
              <button
                onClick={() => setLensPanelOpen(!lensPanelOpen)}
                className="w-full text-left px-4 py-2 bg-neutral-800 text-xs font-bold uppercase flex justify-between items-center text-neutral-300"
              >
                <span>Lens Corrections</span>
                <div className="flex items-center gap-2">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      resetLens();
                    }}
                    className="text-[10px] text-neutral-500 hover:text-white px-2 py-0.5 rounded hover:bg-neutral-700 transition-colors"
                  >
                    Reset
                  </button>
                  <span className={`transition-transform ${lensPanelOpen ? "rotate-90" : ""}`}>▶</span>
                </div>
              </button>
              {lensPanelOpen && (
                <div className="p-4 space-y-6">
                  {/* Distortion Section */}
                  <div className="space-y-4">
                    <h4 className="text-[10px] font-semibold uppercase tracking-wider text-neutral-500 border-b border-neutral-700 pb-1">
                      Manual Corrections
                    </h4>
                    <div className="space-y-3">
                      <div className="space-y-1">
                        <div className="flex justify-between items-center">
                          <span className="text-[11px] font-medium text-neutral-400">Distortion</span>
                          <span className="text-[10px] text-neutral-500 tabular-nums">
                            {currentLensCorrections.distortion > 0 ? "+" : ""}
                            {currentLensCorrections.distortion}
                          </span>
                        </div>
                        <input
                          type="range"
                          min={-100}
                          max={100}
                          step={1}
                          value={currentLensCorrections.distortion}
                          onChange={(e) => handleLensChange("distortion", parseFloat(e.target.value))}
                          className="w-full h-1 bg-neutral-700 rounded-lg appearance-none cursor-pointer accent-neutral-400 hover:accent-white"
                        />
                      </div>
                    </div>
                  </div>

                  {/* Vignette Section */}
                  <div className="space-y-4">
                    <h4 className="text-[10px] font-semibold uppercase tracking-wider text-neutral-500 border-b border-neutral-700 pb-1">
                      Vignette
                    </h4>
                    <div className="space-y-3">
                      {[
                        {
                          key: "vignetteAmount",
                          label: "Amount",
                          min: -100,
                          max: 100,
                          step: 1,
                          showSign: true,
                        },
                        {
                          key: "vignetteMidpoint",
                          label: "Midpoint",
                          min: 0,
                          max: 100,
                          step: 1,
                          showSign: false,
                        },
                        {
                          key: "vignetteRoundness",
                          label: "Roundness",
                          min: -100,
                          max: 100,
                          step: 1,
                          showSign: true,
                        },
                        {
                          key: "vignetteFeather",
                          label: "Feather",
                          min: 0,
                          max: 100,
                          step: 1,
                          showSign: false,
                        },
                      ].map((slider) => (
                        <div key={slider.key} className="space-y-1">
                          <div className="flex justify-between items-center">
                            <span className="text-[11px] font-medium text-neutral-400">{slider.label}</span>
                            <span className="text-[10px] text-neutral-500 tabular-nums">
                              {slider.showSign &&
                              currentLensCorrections[slider.key as keyof LensCorrections] > 0
                                ? "+"
                                : ""}
                              {currentLensCorrections[slider.key as keyof LensCorrections]}
                            </span>
                          </div>
                          <input
                            type="range"
                            min={slider.min}
                            max={slider.max}
                            step={slider.step}
                            value={currentLensCorrections[slider.key as keyof LensCorrections]}
                            onChange={(e) =>
                              handleLensChange(
                                slider.key as keyof LensCorrections,
                                parseFloat(e.target.value),
                              )
                            }
                            className="w-full h-1 bg-neutral-700 rounded-lg appearance-none cursor-pointer accent-neutral-400 hover:accent-white"
                          />
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Defringe Section */}
                  <div className="space-y-4">
                    <h4 className="text-[10px] font-semibold uppercase tracking-wider text-neutral-500 border-b border-neutral-700 pb-1">
                      Defringe
                    </h4>
                    <div className="space-y-3">
                      {/* Purple Fringe */}
                      <div className="space-y-2">
                        <div className="flex items-center gap-2">
                          <span className="w-2 h-2 rounded-full bg-purple-500"></span>
                          <span className="text-[10px] font-medium text-neutral-500 uppercase">
                            Purple Haze
                          </span>
                        </div>
                        <div className="space-y-1">
                          <div className="flex justify-between items-center">
                            <span className="text-[11px] font-medium text-neutral-400">Amount</span>
                            <span className="text-[10px] text-neutral-500 tabular-nums">
                              {currentLensCorrections.defringePurple}
                            </span>
                          </div>
                          <input
                            type="range"
                            min={0}
                            max={20}
                            step={1}
                            value={currentLensCorrections.defringePurple}
                            onChange={(e) => handleLensChange("defringePurple", parseFloat(e.target.value))}
                            className="w-full h-1 bg-neutral-700 rounded-lg appearance-none cursor-pointer accent-purple-500 hover:accent-purple-400"
                          />
                        </div>
                        <div className="space-y-1">
                          <div className="flex justify-between items-center">
                            <span className="text-[11px] font-medium text-neutral-400">Hue</span>
                            <span className="text-[10px] text-neutral-500 tabular-nums">
                              {currentLensCorrections.defringePurpleHue > 0 ? "+" : ""}
                              {currentLensCorrections.defringePurpleHue}
                            </span>
                          </div>
                          <input
                            type="range"
                            min={-100}
                            max={100}
                            step={1}
                            value={currentLensCorrections.defringePurpleHue}
                            onChange={(e) =>
                              handleLensChange("defringePurpleHue", parseFloat(e.target.value))
                            }
                            className="w-full h-1 bg-neutral-700 rounded-lg appearance-none cursor-pointer accent-purple-500 hover:accent-purple-400"
                          />
                        </div>
                      </div>

                      {/* Green Fringe */}
                      <div className="space-y-2 mt-3">
                        <div className="flex items-center gap-2">
                          <span className="w-2 h-2 rounded-full bg-green-500"></span>
                          <span className="text-[10px] font-medium text-neutral-500 uppercase">
                            Green Haze
                          </span>
                        </div>
                        <div className="space-y-1">
                          <div className="flex justify-between items-center">
                            <span className="text-[11px] font-medium text-neutral-400">Amount</span>
                            <span className="text-[10px] text-neutral-500 tabular-nums">
                              {currentLensCorrections.defringeGreen}
                            </span>
                          </div>
                          <input
                            type="range"
                            min={0}
                            max={20}
                            step={1}
                            value={currentLensCorrections.defringeGreen}
                            onChange={(e) => handleLensChange("defringeGreen", parseFloat(e.target.value))}
                            className="w-full h-1 bg-neutral-700 rounded-lg appearance-none cursor-pointer accent-green-500 hover:accent-green-400"
                          />
                        </div>
                        <div className="space-y-1">
                          <div className="flex justify-between items-center">
                            <span className="text-[11px] font-medium text-neutral-400">Hue</span>
                            <span className="text-[10px] text-neutral-500 tabular-nums">
                              {currentLensCorrections.defringeGreenHue > 0 ? "+" : ""}
                              {currentLensCorrections.defringeGreenHue}
                            </span>
                          </div>
                          <input
                            type="range"
                            min={-100}
                            max={100}
                            step={1}
                            value={currentLensCorrections.defringeGreenHue}
                            onChange={(e) => handleLensChange("defringeGreenHue", parseFloat(e.target.value))}
                            className="w-full h-1 bg-neutral-700 rounded-lg appearance-none cursor-pointer accent-green-500 hover:accent-green-400"
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}

export default App;
