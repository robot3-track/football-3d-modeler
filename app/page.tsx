'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';

// --- INTERFACES ---
interface Point { x: number; y: number; label?: string; }
interface PlayerDetection { x: number; z: number; team: 'Dark' | 'Light'; weight?: number; }
interface FrameData { players: Record<string, PlayerDetection>; ball: { x: number; z: number } | null; }

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

/**
 * Sorts exactly 4 points into Top-Left, Top-Right, Bottom-Right, Bottom-Left
 * based on standard broadcast camera orientations.
 */
function sortFourCorners(pts: Point[]) {
  if (pts.length !== 4) return null;
  const sortedByY = [...pts].sort((a, b) => a.y - b.y);
  const topTwo = sortedByY.slice(0, 2).sort((a, b) => a.x - b.x);
  const botTwo = sortedByY.slice(2, 4).sort((a, b) => a.x - b.x);
  return [
    topTwo[0], // Top-Left (Far left)
    topTwo[1], // Top-Right (Far right)
    botTwo[1], // Bottom-Right (Near right)
    botTwo[0]  // Bottom-Left (Near left)
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
    // Fallback scaling if exact homography fails
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
  return { x: (H[0][0] * u + H[0][1] * v + H[0][2]) / W, z: (H[1][0] * u + H[1][1] * v + H[1][2]) / W };
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
      {/* Base Grass */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[52.5, -0.02, 34]}>
        <planeGeometry args={[135, 98]} />
        <meshStandardMaterial color="#1a2e21" roughness={1.0} />
      </mesh>
      
      {stripes}
      
      <group position={[0, 0.02, 0]}>
        {/* Bounds */}
        <mesh position={[52.5, 0, 0]}><boxGeometry args={[105, 0.05, 0.2]} /><meshBasicMaterial color="white" /></mesh>
        <mesh position={[52.5, 0, 68]}><boxGeometry args={[105, 0.05, 0.2]} /><meshBasicMaterial color="white" /></mesh>
        <mesh position={[0, 0, 34]}><boxGeometry args={[0.2, 0.05, 68]} /><meshBasicMaterial color="white" /></mesh>
        <mesh position={[105, 0, 34]}><boxGeometry args={[0.2, 0.05, 68]} /><meshBasicMaterial color="white" /></mesh>
        
        {/* Center */}
        <mesh position={[52.5, 0, 34]}><boxGeometry args={[0.2, 0.05, 68]} /><meshBasicMaterial color="white" /></mesh>
        <mesh position={[52.5, 0, 34]} rotation={[-Math.PI / 2, 0, 0]}><ringGeometry args={[9.05, 9.25, 64]} /><meshBasicMaterial color="white" /></mesh>
        <mesh position={[52.5, 0, 34]}><cylinderGeometry args={[0.3, 0.3, 0.05, 16]} /><meshBasicMaterial color="white" /></mesh>
        
        {/* Left Boxes */}
        <mesh position={[16.5, 0, 34]}><boxGeometry args={[0.2, 0.05, 40.32]} /><meshBasicMaterial color="white" /></mesh>
        <mesh position={[8.25, 0, 13.84]}><boxGeometry args={[16.5, 0.05, 0.2]} /><meshBasicMaterial color="white" /></mesh>
        <mesh position={[8.25, 0, 54.16]}><boxGeometry args={[16.5, 0.05, 0.2]} /><meshBasicMaterial color="white" /></mesh>
        <mesh position={[5.5, 0, 34]}><boxGeometry args={[0.2, 0.05, 18.32]} /><meshBasicMaterial color="white" /></mesh>
        <mesh position={[2.75, 0, 24.84]}><boxGeometry args={[5.5, 0.05, 0.2]} /><meshBasicMaterial color="white" /></mesh>
        <mesh position={[2.75, 0, 43.16]}><boxGeometry args={[5.5, 0.05, 0.2]} /><meshBasicMaterial color="white" /></mesh>

        {/* Right Boxes */}
        <mesh position={[105 - 16.5, 0, 34]}><boxGeometry args={[0.2, 0.05, 40.32]} /><meshBasicMaterial color="white" /></mesh>
        <mesh position={[105 - 8.25, 0, 13.84]}><boxGeometry args={[16.5, 0.05, 0.2]} /><meshBasicMaterial color="white" /></mesh>
        <mesh position={[105 - 8.25, 0, 54.16]}><boxGeometry args={[16.5, 0.05, 0.2]} /><meshBasicMaterial color="white" /></mesh>
        <mesh position={[105 - 5.5, 0, 34]}><boxGeometry args={[0.2, 0.05, 18.32]} /><meshBasicMaterial color="white" /></mesh>
        <mesh position={[105 - 2.75, 0, 24.84]}><boxGeometry args={[5.5, 0.05, 0.2]} /><meshBasicMaterial color="white" /></mesh>
        <mesh position={[105 - 2.75, 0, 43.16]}><boxGeometry args={[5.5, 0.05, 0.2]} /><meshBasicMaterial color="white" /></mesh>
      </group>
    </group>
  );
}

// --- CORE COMPUTER VISION PIPELINE ---
function LiveSoccerPitch3D({ 
  videoRef, homographyMatrix, isPlaying, maskPoints, liveFrameData, setLiveFrameData, 
  k1Distortion, luminanceThreshold, clusteringRadius, minPixelWeight 
}: any) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (!canvasRef.current) canvasRef.current = document.createElement('canvas');
  }, []);

  useFrame(() => {
    if (!isPlaying || !videoRef.current || !homographyMatrix || maskPoints.length < 3 || videoRef.current.paused) return;

    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || video.videoWidth === 0) return;
    if (canvas.width !== video.videoWidth) {
      canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    }

    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    try {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imgData.data;

      const rawClusters: { Dark: PlayerDetection[], Light: PlayerDetection[] } = { Dark: [], Light: [] };
      const stepX = 6, stepY = 6;

      for (let y = 0; y < canvas.height; y += stepY) {
        for (let x = 0; x < canvas.width; x += stepX) {
          // Verify pixel is inside the Mask Boundary
          if (!isInsidePolygon({ x, y }, maskPoints)) continue;

          const idx = (y * canvas.width + x) * 4;
          const r = data[idx], g = data[idx + 1], b = data[idx + 2];

          // Turf exclusion heuristics
          const isStandardGreen = (g > r * 0.85 && g > b * 0.85);
          const isDryTurf = (r > 100 && g > 90 && b < r && g > b);
          const isDirtPatch = (r > 120 && g > 110 && b > 80 && Math.abs(r - g) < 30);
          
          if (!isStandardGreen && !isDryTurf && !isDirtPatch) {
            const luminance = 0.299 * r + 0.587 * g + 0.114 * b;

            if (luminance > 20 && luminance < 245) {
              const undistorted = undistortCoordinate(x, y, canvas.width, canvas.height, k1Distortion);
              const projected = projectPoint(homographyMatrix, undistorted.x, undistorted.y);

              // 105x68 dimension check
              if (projected.x >= 0 && projected.x <= 105 && projected.z >= 0 && projected.z <= 68) {
                const team = luminance >= luminanceThreshold ? 'Light' : 'Dark';
                let merged = false;

                for (let c of rawClusters[team]) {
                  const dist = Math.hypot(c.x - projected.x, c.z - projected.z);
                  if (dist < clusteringRadius) {
                    c.x = ((c.x * (c.weight || 1)) + projected.x) / ((c.weight || 1) + 1);
                    c.z = ((c.z * (c.weight || 1)) + projected.z) / ((c.weight || 1) + 1);
                    c.weight = (c.weight || 1) + 1;
                    merged = true;
                    break;
                  }
                }
                
                if (!merged) {
                  rawClusters[team].push({ x: projected.x, z: projected.z, team, weight: 1 });
                }
              }
            }
          }
        }
      }

      // Filter noise by mass, limit to 11 heaviest clusters per team
      const validDark = rawClusters.Dark
        .filter(c => (c.weight || 0) >= minPixelWeight)
        .sort((a, b) => (b.weight || 0) - (a.weight || 0))
        .slice(0, 11);

      const validLight = rawClusters.Light
        .filter(c => (c.weight || 0) >= minPixelWeight)
        .sort((a, b) => (b.weight || 0) - (a.weight || 0))
        .slice(0, 11);

      const players: Record<string, PlayerDetection> = {};
      let pCount = 1;
      [...validDark, ...validLight].forEach(c => {
        players[`p_${pCount++}`] = { x: Number(c.x.toFixed(2)), z: Number(c.z.toFixed(2)), team: c.team };
      });

      setLiveFrameData({ players, ball: null });
    } catch (e) { console.error(e) }
  });

  const players = liveFrameData?.players || {};

  return (
    <>
      <ambientLight intensity={0.7} />
      <directionalLight position={[52.5, 60, 34]} intensity={1.3} />
      <RealisticPitchMarkings />

      <group position={[0, 0, 34]}>
        <mesh position={[0, 1.22, -3.66]}><cylinderGeometry args={[0.06, 0.06, 2.44]} /><meshBasicMaterial color="#ffffff" /></mesh>
        <mesh position={[0, 1.22, 3.66]}><cylinderGeometry args={[0.06, 0.06, 2.44]} /><meshBasicMaterial color="#ffffff" /></mesh>
        <mesh position={[0, 2.44, 0]} rotation={[Math.PI / 2, 0, 0]}><cylinderGeometry args={[0.06, 0.06, 7.32]} /><meshBasicMaterial color="#ffffff" /></mesh>
      </group>

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
  
  // DUAL STATE CALIBRATION
  const [pitchCorners, setPitchCorners] = useState<Point[]>([]); // Strict 4 for Homography orientation
  const [maskPoints, setMaskPoints] = useState<Point[]>([]);     // Freeform N points for bounds
  const [isDrawingMask, setIsDrawingMask] = useState(false);
  
  const [homographyMatrix, setHomographyMatrix] = useState<number[][] | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [liveFrameData, setLiveFrameData] = useState<FrameData>({ players: {}, ball: null });
  
  const [k1Distortion, setK1Distortion] = useState<number>(0.0);
  const [luminanceThreshold, setLuminanceThreshold] = useState<number>(110);
  const [clusteringRadius, setClusteringRadius] = useState<number>(3.5);
  const [minPixelWeight, setMinPixelWeight] = useState<number>(4);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleVideoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      setVideoSrc(URL.createObjectURL(e.target.files[0]));
      resetCalibration();
    }
  };

  const resetCalibration = () => {
    setPitchCorners([]);
    setMaskPoints([]);
    setIsDrawingMask(false);
    setHomographyMatrix(null);
    setIsPlaying(false);
  };

  const handleLoadedMetadata = () => {
    if (videoRef.current && videoRef.current.videoWidth) {
      setVideoAspect(videoRef.current.videoWidth / videoRef.current.videoHeight);
    }
  };

  const handleVideoClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (homographyMatrix || !containerRef.current || !videoRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.round((e.clientX - rect.left) * (videoRef.current.videoWidth / rect.width));
    const y = Math.round((e.clientY - rect.top) * (videoRef.current.videoHeight / rect.height));
    
    if (!isDrawingMask) {
      // Step 1: Collect up to 4 exact orientation corners
      if (pitchCorners.length < 4) {
        const newCorners = [...pitchCorners, { x, y }];
        setPitchCorners(newCorners);
        // Auto-fill mask with corners once 4 are reached
        if (newCorners.length === 4) {
          const sorted = sortFourCorners(newCorners);
          if (sorted) setMaskPoints(sorted);
        }
      }
    } else {
      // Step 2: Custom Freeform Masking
      setMaskPoints([...maskPoints, { x, y }]);
    }
  };

  const startLiveTracking = () => {
    if (pitchCorners.length !== 4 || !videoRef.current) return;
    
    const sortedCorners = sortFourCorners(pitchCorners);
    if (!sortedCorners) return;

    // Fix matrix distortion based on lens setting
    const undistortedSrcPoints = sortedCorners.map(pt => 
      undistortCoordinate(pt.x, pt.y, videoRef.current!.videoWidth, videoRef.current!.videoHeight, k1Distortion)
    );

    // Hardcode 3D dimension mappings (X = 0..105, Z = 0..68)
    const dstPoints = [
      { x: 0, y: 0 },    // Top-Left (Far left)
      { x: 105, y: 0 },  // Top-Right (Far right)
      { x: 105, y: 68 }, // Bottom-Right (Near right)
      { x: 0, y: 68 }    // Bottom-Left (Near left)
    ];

    setHomographyMatrix(solveHomography(undistortedSrcPoints, dstPoints));
    setIsPlaying(true);
    videoRef.current.play();
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
          <span className="text-xs font-bold tracking-widest text-white uppercase">TACTICAL RADAR V10.0 [DUAL CALIBRATION MATRIX]</span>
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
              </div>

              {/* CALIBRATION INSTRUCTIONS */}
              <div className="flex flex-col gap-2 border border-neutral-800 p-3 bg-black rounded">
                <div className="flex justify-between items-center">
                  <span className={`text-[11px] font-bold tracking-wider uppercase ${pitchCorners.length < 4 ? 'text-amber-500' : 'text-emerald-500'}`}>
                    {pitchCorners.length < 4 ? `STEP 1: SELECT 4 CORNERS (${pitchCorners.length}/4)` : 'ORIENTATION LOCKED'}
                  </span>
                  {(pitchCorners.length > 0 && !homographyMatrix) && (
                    <button onClick={resetCalibration} className="text-[9px] text-neutral-400 hover:text-white underline">RESET</button>
                  )}
                </div>
                
                {pitchCorners.length === 4 && !homographyMatrix && (
                  <div className="mt-2 flex items-center justify-between border-t border-neutral-800 pt-2">
                     <span className="text-[10px] text-neutral-400">Mask bounds auto-generated.</span>
                     <button onClick={() => { setMaskPoints([]); setIsDrawingMask(true); }} className={`text-[10px] px-2 py-1 border rounded transition ${isDrawingMask ? 'bg-cyan-900/50 border-cyan-500 text-cyan-300' : 'border-neutral-700 text-neutral-400 hover:border-cyan-500 hover:text-cyan-400'}`}>
                       DRAW CUSTOM MASK
                     </button>
                  </div>
                )}
              </div>

              <div ref={containerRef} onClick={handleVideoClick} style={{ aspectRatio: videoAspect }} className="relative border-2 border-neutral-800 bg-black cursor-crosshair w-full overflow-hidden shrink-0">
                <video ref={videoRef} src={videoSrc} onLoadedMetadata={handleLoadedMetadata} className="w-full h-full object-fill opacity-90 pointer-events-none" muted playsInline />
                
                {/* 1. Draw Freeform Mask Polygon */}
                {maskPoints.length > 0 && (
                  <svg className="absolute top-0 left-0 w-full h-full pointer-events-none z-10">
                    <polygon 
                      points={maskPoints.map(p => `${p.x * scaleX},${p.y * scaleY}`).join(" ")} 
                      fill="rgba(6, 182, 212, 0.1)" 
                      stroke="#06b6d4" 
                      strokeWidth="1.5" 
                      strokeDasharray={isDrawingMask && maskPoints.length < 3 ? "5 5" : "0"} 
                    />
                  </svg>
                )}

                {/* 2. Draw Pitch Anchor Corners (Persistent) */}
                {pitchCorners.map((pt, idx) => (
                  <div 
                    key={`anchor-${idx}`} 
                    style={{ left: (pt.x * scaleX) - 6, top: (pt.y * scaleY) - 6 }} 
                    className="absolute w-3 h-3 border-2 border-amber-500 bg-black/50 z-20 flex items-center justify-center rounded-sm"
                  >
                    <div className="w-1 h-1 bg-amber-500"></div>
                  </div>
                ))}

                {/* 3. Draw Mask Nodes if actively drawing custom mask */}
                {isDrawingMask && maskPoints.map((pt, idx) => (
                  <div key={`mask-${idx}`} style={{ left: (pt.x * scaleX) - 4, top: (pt.y * scaleY) - 4 }} className="absolute w-2 h-2 bg-cyan-500 rounded-full z-30"></div>
                ))}
              </div>

              {pitchCorners.length === 4 && !homographyMatrix && (
                <button onClick={startLiveTracking} className="w-full py-3 bg-emerald-600 hover:bg-emerald-500 text-black font-black text-[11px] tracking-[0.2em] uppercase rounded-sm transition">
                  INITIATE LIVE TRACKING
                </button>
              )}
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
                videoRef={videoRef} 
                homographyMatrix={homographyMatrix} 
                isPlaying={isPlaying} 
                maskPoints={maskPoints} 
                liveFrameData={liveFrameData} 
                setLiveFrameData={setLiveFrameData} 
                k1Distortion={k1Distortion} 
                luminanceThreshold={luminanceThreshold} 
                clusteringRadius={clusteringRadius} 
                minPixelWeight={minPixelWeight} 
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