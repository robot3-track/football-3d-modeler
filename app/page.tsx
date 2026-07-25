'use client';

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';

// --- INTERFACES ---
interface Point { x: number; y: number; label?: string; }
interface PlayerDetection { 
  id: string;
  x: number; 
  z: number; 
  team: 'Dark' | 'Light'; 
  weight?: number;
  box?: { minX: number; minY: number; maxX: number; maxY: number };
}
interface FrameData { 
  players: Record<string, PlayerDetection>; 
  ball: { x: number; z: number } | null;
  debugBoxes: Array<{ minX: number; minY: number; maxX: number; maxY: number; team: string; weight: number }>;
}

// --- MATH HELPERS & LENS CORRECTION ---
function undistortCoordinate(x: number, y: number, width: number, height: number, k1: number) {
  if (k1 === 0) return { x, y };
  const cx = width / 2;
  const cy = height / 2;
  const nx = (x - cx) / cx;
  const ny = (y - cy) / cy;
  const r2 = nx * nx + ny * ny;
  const undistortedNx = nx * (1 + k1 * r2);
  const undistortedNy = ny * (1 + k1 * r2);
  return {
    x: undistortedNx * cx + cx,
    y: undistortedNy * cy + cy
  };
}

function sortFourCorners(pts: Point[]) {
  if (pts.length !== 4) return null;
  const sortedByY = [...pts].sort((a, b) => a.y - b.y);
  const topTwo = sortedByY.slice(0, 2).sort((a, b) => a.x - b.x);
  const botTwo = sortedByY.slice(2, 4).sort((a, b) => a.x - b.x);
  return [
    topTwo[0], // Top-Left
    topTwo[1], // Top-Right
    botTwo[1], // Bottom-Right
    botTwo[0]  // Bottom-Left
  ];
}

function solveHomography(srcPts: Point[], dstPts: Point[]) {
  try {
    const M: number[][] = [];
    const B: number[] = [];
    for (let i = 0; i < 4; i++) {
      const u = srcPts[i].x; const v = srcPts[i].y;
      const X = dstPts[i].x; const Y = dstPts[i].y;
      M.push([u, v, 1, 0, 0, 0, -u * X, -v * X]); B.push(X);
      M.push([0, 0, 0, u, v, 1, -u * Y, -v * Y]); B.push(Y);
    }
    const h = gaussianElimination8x8(M, B);
    if (!h || h.some(val => isNaN(val) || !isFinite(val))) throw new Error("Matrix singular");
    return [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]];
  } catch (e) {
    const w = Math.abs(srcPts[1].x - srcPts[0].x) || 1;
    const h = Math.abs(srcPts[3].y - srcPts[0].y) || 1;
    return [[105 / w, 0, -srcPts[0].x * (105 / w)], [0, 68 / h, -srcPts[0].y * (68 / h)], [0, 0, 1.0]];
  }
}

function gaussianElimination8x8(A: number[][], b: number[]): number[] | null {
  const n = 8;
  const mat = A.map((row, i) => [...row, b[i]]);
  for (let i = 0; i < n; i++) {
    let maxEl = Math.abs(mat[i][i]), maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(mat[k][i]) > maxEl) { maxEl = Math.abs(mat[k][i]); maxRow = k; }
    }
    for (let k = i; k < n + 1; k++) {
      const tmp = mat[maxRow][k]; mat[maxRow][k] = mat[i][k]; mat[i][k] = tmp;
    }
    if (Math.abs(mat[i][i]) < 1e-12) return null;
    for (let k = i + 1; k < n; k++) {
      const c = -mat[k][i] / mat[i][i];
      for (let j = i; j < n + 1; j++) { if (i === j) mat[k][j] = 0; else mat[k][j] += c * mat[i][j]; }
    }
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    if (Math.abs(mat[i][i]) < 1e-12) return null;
    let sum = mat[i][n];
    for (let j = i + 1; j < n; j++) sum -= mat[i][j] * x[j];
    x[i] = sum / mat[i][i];
  }
  return x;
}

function projectPoint(H: number[][], u: number, v: number) {
  const W = H[2][0] * u + H[2][1] * v + H[2][2];
  if (Math.abs(W) < 1e-6) return { x: 52.5, z: 34 };
  const rawX = (H[0][0] * u + H[0][1] * v + H[0][2]) / W;
  const rawZ = (H[1][0] * u + H[1][1] * v + H[1][2]) / W;
  // Strict clamping to prevent out-of-bounds projections stacking on edges
  return {
    x: Math.max(2, Math.min(103, rawX)),
    z: Math.max(2, Math.min(66, rawZ))
  };
}

function isInsidePolygon(pt: { x: number; y: number }, polygon: Point[]) {
  if (polygon.length < 3) return false;
  let x = pt.x, y = pt.y, inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    let xi = polygon[i].x, yi = polygon[i].y, xj = polygon[j].x, yj = polygon[j].y;
    if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / ((yj - yi) || 0.00001) + xi)) inside = !inside;
  }
  return inside;
}

// --- 3D RENDER COMPONENTS ---
function RealisticPitchMarkings() {
  const stripes = Array.from({ length: 15 }).map((_, i) => (
    <mesh key={i} position={[3.5 + i * 7, -0.01, 34]} rotation={[-Math.PI / 2, 0, 0]}>
      <planeGeometry args={[7, 68]} />
      <meshStandardMaterial color={i % 2 === 0 ? "#2a4d36" : "#2f573d"} roughness={0.9} />
    </mesh>
  ));

  return (
    <group>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[52.5, -0.02, 34]}>
        <planeGeometry args={[135, 98]} />
        <meshStandardMaterial color="#1a2e21" roughness={1.0} />
      </mesh>
      {stripes}
      <group position={[0, 0.02, 0]}>
        <mesh position={[52.5, 0, 0]}><boxGeometry args={[105, 0.05, 0.2]} /><meshBasicMaterial color="white" /></mesh>
        <mesh position={[52.5, 0, 68]}><boxGeometry args={[105, 0.05, 0.2]} /><meshBasicMaterial color="white" /></mesh>
        <mesh position={[0, 0, 34]}><boxGeometry args={[0.2, 0.05, 68]} /><meshBasicMaterial color="white" /></mesh>
        <mesh position={[105, 0, 34]}><boxGeometry args={[0.2, 0.05, 68]} /><meshBasicMaterial color="white" /></mesh>
        <mesh position={[52.5, 0, 34]}><boxGeometry args={[0.2, 0.05, 68]} /><meshBasicMaterial color="white" /></mesh>
        <mesh position={[52.5, 0, 34]} rotation={[-Math.PI / 2, 0, 0]}><ringGeometry args={[9.05, 9.25, 64]} /><meshBasicMaterial color="white" /></mesh>
        <mesh position={[52.5, 0, 34]}><cylinderGeometry args={[0.3, 0.3, 0.05, 16]} /><meshBasicMaterial color="white" /></mesh>
        
        {/* Left Box */}
        <mesh position={[16.5, 0, 34]}><boxGeometry args={[0.2, 0.05, 40.32]} /><meshBasicMaterial color="white" /></mesh>
        <mesh position={[8.25, 0, 13.84]}><boxGeometry args={[16.5, 0.05, 0.2]} /><meshBasicMaterial color="white" /></mesh>
        <mesh position={[8.25, 0, 54.16]}><boxGeometry args={[16.5, 0.05, 0.2]} /><meshBasicMaterial color="white" /></mesh>

        {/* Right Box */}
        <mesh position={[105 - 16.5, 0, 34]}><boxGeometry args={[0.2, 0.05, 40.32]} /><meshBasicMaterial color="white" /></mesh>
        <mesh position={[105 - 8.25, 0, 13.84]}><boxGeometry args={[16.5, 0.05, 0.2]} /><meshBasicMaterial color="white" /></mesh>
        <mesh position={[105 - 8.25, 0, 54.16]}><boxGeometry args={[16.5, 0.05, 0.2]} /><meshBasicMaterial color="white" /></mesh>
      </group>
    </group>
  );
}

function LiveSoccerPitch3D({ liveFrameData }: any) {
  const players = liveFrameData?.players || {};

  return (
    <>
      <ambientLight intensity={0.7} />
      <directionalLight position={[52.5, 60, 34]} intensity={1.3} />
      <RealisticPitchMarkings />

      {/* Goal 1 Structure */}
      <group position={[0, 0, 34]}>
        <mesh position={[0, 1.22, -3.66]}><cylinderGeometry args={[0.06, 0.06, 2.44]} /><meshBasicMaterial color="#ffffff" /></mesh>
        <mesh position={[0, 1.22, 3.66]}><cylinderGeometry args={[0.06, 0.06, 2.44]} /><meshBasicMaterial color="#ffffff" /></mesh>
        <mesh position={[0, 2.44, 0]} rotation={[Math.PI / 2, 0, 0]}><cylinderGeometry args={[0.06, 0.06, 7.32]} /><meshBasicMaterial color="#ffffff" /></mesh>
      </group>

      {/* Goal 2 Structure */}
      <group position={[105, 0, 34]}>
        <mesh position={[0, 1.22, -3.66]}><cylinderGeometry args={[0.06, 0.06, 2.44]} /><meshBasicMaterial color="#ffffff" /></mesh>
        <mesh position={[0, 1.22, 3.66]}><cylinderGeometry args={[0.06, 0.06, 2.44]} /><meshBasicMaterial color="#ffffff" /></mesh>
        <mesh position={[0, 2.44, 0]} rotation={[Math.PI / 2, 0, 0]}><cylinderGeometry args={[0.06, 0.06, 7.32]} /><meshBasicMaterial color="#ffffff" /></mesh>
      </group>

      {Object.entries(players).map(([id, player]: [string, any]) => (
        <group key={id} position={[player.x, 0.9, player.z]}>
          <mesh>
            <cylinderGeometry args={[0.4, 0.4, 1.8, 12]} />
            <meshStandardMaterial color={player.team === 'Dark' ? '#18181b' : '#f8fafc'} roughness={0.4} metalness={0.1} />
          </mesh>
          <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.89, 0]}>
            <ringGeometry args={[0.55, 0.7, 16]} />
            <meshBasicMaterial color={player.team === 'Dark' ? '#ef4444' : '#3b82f6'} />
          </mesh>
        </group>
      ))}
    </>
  );
}

// --- MAIN UI DASHBOARD ---
export default function Home() {
  const [videoSrc, setVideoSrc] = useState<string | null>(null);
  const [videoAspect, setVideoAspect] = useState<number>(16 / 9);
  
  // CALIBRATION STATES (Strictly 4 Pitch Corners)
  const [calibrationStep, setCalibrationStep] = useState<'corners' | 'ready'>('corners');
  const [pitchCorners, setPitchCorners] = useState<Point[]>([]);
  const [showDebugBoxes, setShowDebugBoxes] = useState<boolean>(true);
  
  const [homographyMatrix, setHomographyMatrix] = useState<number[][] | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [liveFrameData, setLiveFrameData] = useState<FrameData>({ players: {}, ball: null, debugBoxes: [] });
  
  const [k1Distortion, setK1Distortion] = useState<number>(0.0);
  const [luminanceThreshold, setLuminanceThreshold] = useState<number>(110);
  const [clusteringRadius, setClusteringRadius] = useState<number>(3.5);
  const [minPixelWeight, setMinPixelWeight] = useState<number>(3);
  const [motionSensitivity, setMotionSensitivity] = useState<number>(25);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const backgroundCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const animFrameRef = useRef<number | null>(null);

  useEffect(() => {
    if (!canvasRef.current) canvasRef.current = document.createElement('canvas');
    if (!backgroundCanvasRef.current) backgroundCanvasRef.current = document.createElement('canvas');
  }, []);

  const handleVideoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      setVideoSrc(URL.createObjectURL(e.target.files[0]));
      resetCalibration();
    }
  };

  const resetCalibration = () => {
    if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    setPitchCorners([]);
    setCalibrationStep('corners');
    setHomographyMatrix(null);
    setIsPlaying(false);
  };

  const handleLoadedMetadata = () => {
    if (videoRef.current && videoRef.current.videoWidth) {
      setVideoAspect(videoRef.current.videoWidth / videoRef.current.videoHeight);
    }
  };

  // Capture background frame reference on start for motion differencing
  const captureBackgroundFrame = () => {
    if (!videoRef.current || !backgroundCanvasRef.current) return;
    const bgCanvas = backgroundCanvasRef.current;
    const video = videoRef.current;
    bgCanvas.width = video.videoWidth;
    bgCanvas.height = video.videoHeight;
    const bgCtx = bgCanvas.getContext('2d', { willReadFrequently: true });
    if (bgCtx) {
      bgCtx.drawImage(video, 0, 0, bgCanvas.width, bgCanvas.height);
    }
  };

  // Dedicated Robust Motion + Vision Loop via requestAnimationFrame
  const runVisionLoop = useCallback(() => {
    if (!isPlaying || !videoRef.current || !homographyMatrix || pitchCorners.length !== 4 || videoRef.current.paused) {
      animFrameRef.current = requestAnimationFrame(runVisionLoop);
      return;
    }

    const canvas = canvasRef.current;
    const bgCanvas = backgroundCanvasRef.current;
    const video = videoRef.current;
    if (!canvas || !bgCanvas || video.videoWidth === 0) {
      animFrameRef.current = requestAnimationFrame(runVisionLoop);
      return;
    }

    if (canvas.width !== video.videoWidth) {
      canvas.width = video.videoWidth; 
      canvas.height = video.videoHeight;
    }

    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const bgCtx = bgCanvas.getContext('2d', { willReadFrequently: true });
    if (!ctx || !bgCtx) {
      animFrameRef.current = requestAnimationFrame(runVisionLoop);
      return;
    }

    try {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const currImgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const bgImgData = bgCtx.getImageData(0, 0, canvas.width, canvas.height);
      const currData = currImgData.data;
      const bgData = bgImgData.data;

      interface ClusterInternal {
        x: number; z: number; team: 'Dark' | 'Light'; weight: number;
        minX: number; maxX: number; minY: number; maxY: number;
      }

      const rawClusters: { Dark: ClusterInternal[], Light: ClusterInternal[] } = { Dark: [], Light: [] };
      const stepX = 6, stepY = 6; 

      for (let y = 0; y < canvas.height; y += stepY) {
        for (let x = 0; x < canvas.width; x += stepX) {
          if (!isInsidePolygon({ x, y }, pitchCorners)) continue;

          const idx = (y * canvas.width + x) * 4;
          const r = currData[idx], g = currData[idx + 1], b = currData[idx + 2];
          const bgR = bgData[idx], bgG = bgData[idx + 1], bgB = bgData[idx + 2];

          // MOTION DIFFERENCING: Check if pixel differs from static background (ignores grass & buildings!)
          const diff = Math.abs(r - bgR) + Math.abs(g - bgG) + Math.abs(b - bgB);

          if (diff > motionSensitivity) {
            const luminance = 0.299 * r + 0.587 * g + 0.114 * b;
            const undistorted = undistortCoordinate(x, y, canvas.width, canvas.height, k1Distortion);
            const projected = projectPoint(homographyMatrix, undistorted.x, undistorted.y);

            const team = luminance >= luminanceThreshold ? 'Light' : 'Dark';
            let merged = false;

            for (let c of rawClusters[team]) {
              const dist = Math.hypot(c.x - projected.x, c.z - projected.z);
              if (dist < clusteringRadius) {
                c.x = ((c.x * c.weight) + projected.x) / (c.weight + 1);
                c.z = ((c.z * c.weight) + projected.z) / (c.weight + 1);
                c.weight += 1;
                c.minX = Math.min(c.minX, x);
                c.maxX = Math.max(c.maxX, x);
                c.minY = Math.min(c.minY, y);
                c.maxY = Math.max(c.maxY, y);
                merged = true;
                break;
              }
            }
            
            if (!merged) {
              rawClusters[team].push({ 
                x: projected.x, 
                z: projected.z, 
                team, 
                weight: 1,
                minX: x, maxX: x, minY: y, maxY: y 
              });
            }
          }
        }
      }

      const applyNMS = (clusters: ClusterInternal[]) => {
        const sorted = [...clusters].filter(c => c.weight >= minPixelWeight).sort((a, b) => b.weight - a.weight);
        const kept: ClusterInternal[] = [];
        for (const c of sorted) {
          let overlap = false;
          for (const k of kept) {
            if (Math.hypot(k.x - c.x, k.z - c.z) < (clusteringRadius * 1.5)) {
              overlap = true;
              break;
            }
          }
          if (!overlap) kept.push(c);
        }
        return kept.slice(0, 11);
      };

      const validDark = applyNMS(rawClusters.Dark);
      const validLight = applyNMS(rawClusters.Light);

      const players: Record<string, PlayerDetection> = {};
      const debugBoxes: Array<{ minX: number; minY: number; maxX: number; maxY: number; team: string; weight: number }> = [];
      let pCount = 1;

      [...validDark, ...validLight].forEach(c => {
        const id = `p_${pCount++}`;
        players[id] = { 
          id, x: Number(c.x.toFixed(2)), z: Number(c.z.toFixed(2)), team: c.team,
          box: { minX: c.minX, minY: c.minY, maxX: c.maxX, maxY: c.maxY }
        };
        debugBoxes.push({ minX: c.minX, minY: c.minY, maxX: c.maxX, maxY: c.maxY, team: c.team, weight: c.weight });
      });

      setLiveFrameData({ players, ball: null, debugBoxes });
    } catch (e) {
      console.error("Vision Processing Error:", e);
    }

    animFrameRef.current = requestAnimationFrame(runVisionLoop);
  }, [isPlaying, homographyMatrix, pitchCorners, k1Distortion, luminanceThreshold, clusteringRadius, minPixelWeight, motionSensitivity]);

  useEffect(() => {
    if (isPlaying) {
      animFrameRef.current = requestAnimationFrame(runVisionLoop);
    } else if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
    }
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [isPlaying, runVisionLoop]);

  const handleVideoClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (homographyMatrix || !containerRef.current || !videoRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.round((e.clientX - rect.left) * (videoRef.current.videoWidth / rect.width));
    const y = Math.round((e.clientY - rect.top) * (videoRef.current.videoHeight / rect.height));
    
    if (calibrationStep === 'corners') {
      const newCorners = [...pitchCorners, { x, y }];
      setPitchCorners(newCorners);
      if (newCorners.length === 4) {
        const sorted = sortFourCorners(newCorners);
        if (sorted) {
          setPitchCorners(sorted);
          const undistortedSrcPoints = sorted.map(pt => 
            undistortCoordinate(pt.x, pt.y, videoRef.current!.videoWidth, videoRef.current!.videoHeight, k1Distortion)
          );
          const dstPoints = [
            { x: 0, y: 0 },    // Top-Left
            { x: 105, y: 0 },  // Top-Right
            { x: 105, y: 68 }, // Bottom-Right
            { x: 0, y: 68 }    // Bottom-Left
          ];
          setHomographyMatrix(solveHomography(undistortedSrcPoints, dstPoints));
          captureBackgroundFrame();
          setCalibrationStep('ready');
          setIsPlaying(true);
          videoRef.current.play();
        }
      }
    }
  };

  const getScale = () => {
    if (!containerRef.current || !videoRef.current) return { scaleX: 1, scaleY: 1 };
    const rect = containerRef.current.getBoundingClientRect();
    return {
      scaleX: rect.width / videoRef.current.videoWidth,
      scaleY: rect.height / videoRef.current.videoHeight
    };
  };

  const { scaleX, scaleY } = getScale();

  return (
    <main className="h-screen w-screen bg-neutral-950 text-neutral-200 flex flex-col font-mono select-none antialiased overflow-hidden">
      
      <div className="w-full border-b border-neutral-800 px-6 py-3 bg-neutral-900 flex justify-between items-center z-50">
        <div className="flex items-center gap-3">
          <div className={`w-2 h-2 ${isPlaying ? 'bg-emerald-500 animate-pulse' : 'bg-neutral-500'}`}></div>
          <span className="text-xs font-bold tracking-widest text-white uppercase">TACTICAL RADAR V14.0 [MOTION SUBTRACTION + CLEAN HOMOGRAPHY]</span>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-[11px] text-neutral-400 font-semibold uppercase tracking-wider">SOURCE FEED:</span>
          <input type="file" accept="video/*" onChange={handleVideoUpload} className="text-xs text-neutral-300 border border-neutral-700 px-3 py-1 bg-neutral-950 cursor-pointer" />
        </div>
      </div>

      <div className="flex flex-grow overflow-hidden w-full h-full">
        
        {/* LEFT PANEL */}
        <div className="w-1/2 h-full border-r border-neutral-800 p-6 flex flex-col gap-5 bg-neutral-950 overflow-y-auto">
          {videoSrc ? (
            <div className="flex flex-col gap-4 w-full">
              
              <div className="bg-neutral-900 border border-neutral-800 p-4 rounded flex flex-col gap-5 shadow-inner">
                <span className="text-xs font-black tracking-widest uppercase text-white border-b border-neutral-800 pb-2">Vision Engine Parameters</span>
                
                <div className="flex flex-col gap-2">
                  <div className="flex justify-between items-center">
                    <span className="text-[10px] uppercase font-bold text-cyan-400">Lens Un-Bender (k1)</span>
                    <span className="text-[10px] text-neutral-400">{k1Distortion.toFixed(6)}</span>
                  </div>
                  <input type="range" min="-0.00005" max="0.00005" step="0.000001" value={k1Distortion} onChange={(e) => setK1Distortion(parseFloat(e.target.value))} className="w-full cursor-pointer accent-cyan-500" disabled={homographyMatrix !== null}/>
                </div>

                <div className="flex flex-col gap-2">
                  <div className="flex justify-between items-center">
                    <span className="text-[10px] uppercase font-bold text-teal-400">Motion Sensitivity (Ignore Grass)</span>
                    <span className="text-[10px] text-neutral-400">{motionSensitivity}</span>
                  </div>
                  <input type="range" min="10" max="80" step="1" value={motionSensitivity} onChange={(e) => setMotionSensitivity(parseInt(e.target.value))} className="w-full cursor-pointer accent-teal-500"/>
                </div>

                <div className="flex flex-col gap-2">
                  <div className="flex justify-between items-center">
                    <span className="text-[10px] uppercase font-bold text-amber-400">Kit Luminance Split</span>
                    <span className="text-[10px] text-neutral-400">{luminanceThreshold}</span>
                  </div>
                  <input type="range" min="30" max="200" step="1" value={luminanceThreshold} onChange={(e) => setLuminanceThreshold(parseInt(e.target.value))} className="w-full cursor-pointer accent-amber-500"/>
                </div>

                <div className="flex flex-col gap-2">
                  <div className="flex justify-between items-center">
                    <span className="text-[10px] uppercase font-bold text-fuchsia-400">Cluster Radius (Meters)</span>
                    <span className="text-[10px] text-neutral-400">{clusteringRadius.toFixed(1)}m</span>
                  </div>
                  <input type="range" min="1.0" max="8.0" step="0.1" value={clusteringRadius} onChange={(e) => setClusteringRadius(parseFloat(e.target.value))} className="w-full cursor-pointer accent-fuchsia-500"/>
                </div>

                <div className="flex flex-col gap-2">
                  <div className="flex justify-between items-center">
                    <span className="text-[10px] uppercase font-bold text-red-400">Ghost Filter (Min Pixel Mass)</span>
                    <span className="text-[10px] text-neutral-400">{minPixelWeight} px</span>
                  </div>
                  <input type="range" min="1" max="25" step="1" value={minPixelWeight} onChange={(e) => setMinPixelWeight(parseInt(e.target.value))} className="w-full cursor-pointer accent-red-500"/>
                </div>

                <div className="flex items-center justify-between border-t border-neutral-800 pt-3">
                  <span className="text-[10px] uppercase font-bold text-emerald-400">Show Tracker Bounding Boxes (Debug)</span>
                  <input type="checkbox" checked={showDebugBoxes} onChange={(e) => setShowDebugBoxes(e.target.checked)} className="accent-emerald-500 w-4 h-4 cursor-pointer"/>
                </div>
              </div>

              {/* CALIBRATION GUIDE */}
              <div className="flex flex-col gap-2 border border-neutral-800 p-3 bg-black rounded">
                <div className="flex justify-between items-center">
                  <span className="text-[11px] font-bold tracking-wider uppercase text-amber-500">
                    {calibrationStep === 'corners' && `CLICK 4 PITCH CORNERS IN ORDER (${pitchCorners.length}/4): TL -> TR -> BR -> BL`}
                    {calibrationStep === 'ready' && 'STATUS: MATRIX LOCKED - LIVE MOTION TRACKING'}
                  </span>
                  <button onClick={resetCalibration} className="text-[9px] text-neutral-400 hover:text-white underline">RESET</button>
                </div>
              </div>

              <div ref={containerRef} onClick={handleVideoClick} style={{ aspectRatio: videoAspect }} className="relative border-2 border-neutral-800 bg-black cursor-crosshair w-full overflow-hidden shrink-0">
                <video ref={videoRef} src={videoSrc} onLoadedMetadata={handleLoadedMetadata} className="w-full h-full object-fill opacity-90 pointer-events-none" muted playsInline />
                
                {/* Calibration Polygon Area */}
                {pitchCorners.length > 0 && (
                  <svg className="absolute top-0 left-0 w-full h-full pointer-events-none z-10">
                    <polygon 
                      points={pitchCorners.map(p => `${p.x * scaleX},${p.y * scaleY}`).join(" ")} 
                      fill="rgba(6, 182, 212, 0.15)" 
                      stroke="#06b6d4" 
                      strokeWidth="2" 
                    />
                  </svg>
                )}

                {/* Pitch Corner Markers */}
                {pitchCorners.map((pt, idx) => (
                  <div key={`corner-${idx}`} style={{ left: (pt.x * scaleX) - 6, top: (pt.y * scaleY) - 6 }} className="absolute w-3 h-3 border-2 border-amber-500 bg-black z-20 flex items-center justify-center rounded-sm">
                    <span className="text-[8px] text-amber-400 font-bold absolute -top-3">{idx + 1}</span>
                  </div>
                ))}

                {/* DEBUG BOUNDING BOXES OVERLAY */}
                {showDebugBoxes && isPlaying && liveFrameData.debugBoxes.map((box, idx) => {
                  const width = (box.maxX - box.minX) * scaleX;
                  const height = (box.maxY - box.minY) * scaleY;
                  const left = box.minX * scaleX;
                  const top = box.minY * scaleY;
                  const color = box.team === 'Dark' ? '#ef4444' : '#3b82f6';
                  return (
                    <div 
                      key={`debug-box-${idx}`}
                      style={{ left, top, width: Math.max(width, 12), height: Math.max(height, 12), borderColor: color }}
                      className="absolute border border-dashed bg-white/10 z-30 pointer-events-none flex items-start p-0.5"
                    >
                      <span className="text-[8px] text-white bg-black/80 px-1 font-bold">{box.weight}px</span>
                    </div>
                  );
                })}
              </div>

            </div>
          ) : (
             <div className="h-full flex items-center justify-center border border-dashed border-neutral-800 text-[11px] text-neutral-500 uppercase tracking-widest">Awaiting Video File</div>
          )}
        </div>

        {/* RIGHT PANEL: 3D ENGINE */}
        <div className="w-1/2 h-full relative bg-neutral-950 flex flex-col border-l border-neutral-900">
          <div className="absolute top-4 left-4 z-20 pointer-events-none">
            {homographyMatrix && (
              <div className="bg-black/80 border border-neutral-800 p-3 text-[10px] text-neutral-400 tracking-wider uppercase backdrop-blur-sm rounded w-48 shadow-lg">
                <span className="font-bold text-white block border-b border-neutral-800 pb-1 mb-2">Live Telemetry (Max 11)</span>
                <div className="flex justify-between items-center mb-1">
                  <span className="text-red-400">Dark Kit Active:</span>
                  <span className="text-white font-mono">{Object.values(liveFrameData.players).filter(p => p.team === 'Dark').length}/11</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-blue-400">Light Kit Active:</span>
                  <span className="text-white font-mono">{Object.values(liveFrameData.players).filter(p => p.team === 'Light').length}/11</span>
                </div>
              </div>
            )}
          </div>

          <div className="flex-grow w-full h-full cursor-move">
            <Canvas camera={{ position: [52.5, 55, 100], fov: 38 }}>
              <color attach="background" args={['#050505']} />
              <LiveSoccerPitch3D 
                liveFrameData={liveFrameData} 
              />
              <OrbitControls target={[52.5, 0, 34]} maxPolarAngle={Math.PI / 2.15} enableDamping dampingFactor={0.05} />
            </Canvas>
          </div>

          {homographyMatrix && (
            <div className="bg-black border-t border-neutral-900 p-4 flex items-center justify-between z-30 shadow-[0_-10px_30px_rgba(0,0,0,0.5)]">
              <button onClick={() => { if(videoRef.current){ isPlaying ? videoRef.current.pause() : videoRef.current.play(); setIsPlaying(!isPlaying); } }} className="px-6 py-2 bg-neutral-900 text-white text-[11px] font-bold border border-neutral-700 uppercase tracking-widest rounded-sm hover:bg-neutral-800 transition">
                {isPlaying ? 'PAUSE STREAM' : 'RESUME STREAM'}
              </button>
            </div>
          )}
        </div>

      </div>
    </main>
  );
}