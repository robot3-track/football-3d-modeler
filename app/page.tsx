'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';

interface Point { x: number; y: number; label: string; }

const CLICK_LABELS = [
  "1. Top-Left Corner (Far Left Touchline / Goal Line)",
  "2. Top-Right Corner (Far Right Touchline / Goal Line)",
  "3. Bottom-Right Corner (Near Right Touchline / Sideline)",
  "4. Bottom-Left Corner (Near Left Touchline / Sideline)"
];

function solveHomography(srcPts: {x: number, y: number}[], dstPts: {x: number, y: number}[]) {
  try {
    const M: number[][] = [];
    const B: number[] = [];

    for (let i = 0; i < 4; i++) {
      const u = srcPts[i].x;
      const v = srcPts[i].y;
      const X = dstPts[i].x;
      const Y = dstPts[i].y;

      M.push([u, v, 1, 0, 0, 0, -u * X, -v * X]);
      B.push(X);

      M.push([0, 0, 0, u, v, 1, -u * Y, -v * Y]);
      B.push(Y);
    }

    const h = gaussianElimination8x8(M, B);
    if (!h || h.some(val => isNaN(val) || !isFinite(val))) {
      throw new Error("Matrix singular");
    }

    return [
      [h[0], h[1], h[2]],
      [h[3], h[4], h[5]],
      [h[6], h[7], 1.0]
    ];
  } catch (e) {
    const w = Math.abs(srcPts[1].x - srcPts[0].x) || 1;
    const h = Math.abs(srcPts[3].y - srcPts[0].y) || 1;
    return [
      [105 / w, 0, -srcPts[0].x * (105 / w)],
      [0, 68 / h, -srcPts[0].y * (68 / h)],
      [0, 0, 1.0]
    ];
  }
}

function gaussianElimination8x8(A: number[][], b: number[]): number[] | null {
  const n = 8;
  const mat = A.map((row, i) => [...row, b[i]]);

  for (let i = 0; i < n; i++) {
    let maxEl = Math.abs(mat[i][i]);
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(mat[k][i]) > maxEl) {
        maxEl = Math.abs(mat[k][i]);
        maxRow = k;
      }
    }

    for (let k = i; k < n + 1; k++) {
      const tmp = mat[maxRow][k];
      mat[maxRow][k] = mat[i][k];
      mat[i][k] = tmp;
    }

    if (Math.abs(mat[i][i]) < 1e-12) return null;

    for (let k = i + 1; k < n; k++) {
      const c = -mat[k][i] / mat[i][i];
      for (let j = i; j < n + 1; j++) {
        if (i === j) mat[k][j] = 0;
        else mat[k][j] += c * mat[i][j];
      }
    }
  }

  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    if (Math.abs(mat[i][i]) < 1e-12) return null;
    let sum = mat[i][n];
    for (let j = i + 1; j < n; j++) {
      sum -= mat[i][j] * x[j];
    }
    x[i] = sum / mat[i][i];
  }
  return x;
}

function projectPoint(H: number[][], u: number, v: number) {
  const W = H[2][0] * u + H[2][1] * v + H[2][2];
  if (Math.abs(W) < 1e-6) return { x: 52.5, z: 34 };
  const X = (H[0][0] * u + H[0][1] * v + H[0][2]) / W;
  const Y = (H[1][0] * u + H[1][1] * v + H[1][2]) / W;
  return { x: X, z: Y };
}

function isInsidePolygon(pt: { x: number; y: number }, polygon: Point[]) {
  if (polygon.length < 4) return false;
  let x = pt.x, y = pt.y;
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    let xi = polygon[i].x, yi = polygon[i].y;
    let xj = polygon[j].x, yj = polygon[j].y;
    let intersect = ((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / ((yj - yi) || 0.00001) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

function LiveSoccerPitch3D({ videoRef, homographyMatrix, isPlaying, clickedPoints, liveFrameData, setLiveFrameData }: any) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (!canvasRef.current) {
      canvasRef.current = document.createElement('canvas');
    }
  }, []);

  useFrame(() => {
    if (!isPlaying || !videoRef.current || !homographyMatrix || clickedPoints.length < 4 || videoRef.current.paused) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!canvas || video.videoWidth === 0) return;

    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    try {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imgData.data;

      const rawDetections: { x: number; z: number; team: string }[] = [];
      const stepX = 6; 
      const stepY = 6;

      for (let y = 0; y < canvas.height; y += stepY) {
        for (let x = 0; x < canvas.width; x += stepX) {
          if (!isInsidePolygon({ x, y }, clickedPoints)) continue;

          const idx = (y * canvas.width + x) * 4;
          const r = data[idx];
          const g = data[idx + 1];
          const b = data[idx + 2];

          // Adaptive green/brown field color threshold
          const isDryOrGreenTurf = (g > r * 0.92 && g > b * 0.95) || (r > 100 && g > 95 && b < r && g > b);
          
          if (!isDryOrGreenTurf && (r + g + b > 90) && (r + g + b < 680)) {
            const projected = projectPoint(homographyMatrix, x, y);
            if (projected.x >= 0 && projected.x <= 105 && projected.z >= 0 && projected.z <= 68) {
              const team = r > b ? 'A' : 'B';
              rawDetections.push({
                x: Number(projected.x.toFixed(2)),
                z: Number(projected.z.toFixed(2)),
                team
              });
            }
          }
        }
      }

      const players: any = {};
      let pCount = 1;
      
      for (const det of rawDetections) {
        let merged = false;
        for (const key of Object.keys(players)) {
          const p = players[key];
          const dist = Math.hypot(p.x - det.x, p.z - det.z);
          
          if (dist < 3.5) { 
            merged = true;
            break;
          }
        }
        if (!merged && Object.keys(players).length < 24) {
          players[`p_${pCount++}`] = { x: det.x, z: det.z, team: det.team };
        }
      }

      let ball = null;
      let foundBall = false;
      for (let y = 0; y < canvas.height && !foundBall; y += 4) {
        for (let x = 0; x < canvas.width; x += 4) {
          if (!isInsidePolygon({ x, y }, clickedPoints)) continue;
          const idx = (y * canvas.width + x) * 4;
          if (data[idx] > 220 && data[idx + 1] > 220 && data[idx + 2] > 220) {
            const bp = projectPoint(homographyMatrix, x, y);
            if (bp.x > 2 && bp.x < 103 && bp.z > 2 && bp.z < 66) {
              ball = { x: Number(bp.x.toFixed(2)), z: Number(bp.z.toFixed(2)) };
              foundBall = true;
              break;
            }
          }
        }
      }

      setLiveFrameData({ players, ball });
    } catch (e) {}
  });

  const players = liveFrameData?.players || {};
  const ball = liveFrameData?.ball || null;

  return (
    <>
      <ambientLight intensity={0.7} />
      <directionalLight position={[52.5, 60, 34]} intensity={1.3} />

      {/* Outer Field Boundary Rim */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[52.5, -0.02, 34]}>
        <planeGeometry args={[135, 98]} />
        <meshStandardMaterial color="#0d0d0d" roughness={1.0} />
      </mesh>

      {/* Core Playable Surface Area */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[52.5, -0.01, 34]}>
        <planeGeometry args={[105, 68]} />
        <meshStandardMaterial color="#1b3325" roughness={0.8} />
      </mesh>

      <gridHelper args={[105, 14, '#ffffff', '#284d37']} position={[52.5, 0.01, 34]} />

      {/* Left Goal Box Infrastructure */}
      <group position={[0, 0, 34]}>
        <mesh position={[0, 1.22, -3.66]}><cylinderGeometry args={[0.06, 0.06, 2.44]} /><meshBasicMaterial color="#ffffff" /></mesh>
        <mesh position={[0, 1.22, 3.66]}><cylinderGeometry args={[0.06, 0.06, 2.44]} /><meshBasicMaterial color="#ffffff" /></mesh>
        <mesh position={[0, 2.44, 0]} rotation={[Math.PI / 2, 0, 0]}><cylinderGeometry args={[0.06, 0.06, 7.32]} /><meshBasicMaterial color="#ffffff" /></mesh>
        <mesh position={[8.25, 0.01, 0]} rotation={[-Math.PI / 2, 0, 0]}><planeGeometry args={[16.5, 40.3]} /><meshBasicMaterial color="#ffffff" wireframe /></mesh>
      </group>

      {/* Right Goal Box Infrastructure */}
      <group position={[105, 0, 34]}>
        <mesh position={[0, 1.22, -3.66]}><cylinderGeometry args={[0.06, 0.06, 2.44]} /><meshBasicMaterial color="#ffffff" /></mesh>
        <mesh position={[0, 1.22, 3.66]}><cylinderGeometry args={[0.06, 0.06, 2.44]} /><meshBasicMaterial color="#ffffff" /></mesh>
        <mesh position={[0, 2.44, 0]} rotation={[Math.PI / 2, 0, 0]}><cylinderGeometry args={[0.06, 0.06, 7.32]} /><meshBasicMaterial color="#ffffff" /></mesh>
        <mesh position={[-8.25, 0.01, 0]} rotation={[-Math.PI / 2, 0, 0]}><planeGeometry args={[16.5, 40.3]} /><meshBasicMaterial color="#ffffff" wireframe /></mesh>
      </group>

      {/* Dynamic Player Model Anchors */}
      {Object.entries(players).map(([id, player]: [string, any]) => {
        const teamColor = player.team === 'A' ? '#e11d48' : '#2563eb';
        return (
          <group key={id} position={[player.x, 0.9, player.z]}>
            <mesh>
              <cylinderGeometry args={[0.4, 0.4, 1.8, 12]} />
              <meshStandardMaterial color={teamColor} roughness={0.3} metalness={0.1} />
            </mesh>
            <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.89, 0]}>
              <ringGeometry args={[0.55, 0.7, 16]} />
              <meshBasicMaterial color={teamColor} />
            </mesh>
          </group>
        );
      })}

      {ball && (
        <mesh position={[ball.x, 0.35, ball.z]}>
          <sphereGeometry args={[0.3, 16, 16]} />
          <meshStandardMaterial color="#ffffff" roughness={0.2} emissive="#ffffff" emissiveIntensity={0.4} />
        </mesh>
      )}
    </>
  );
}

export default function Home() {
  const [videoSrc, setVideoSrc] = useState<string | null>(null);
  const [videoAspect, setVideoAspect] = useState<number>(16 / 9);
  const [clickedPoints, setClickedPoints] = useState<Point[]>([]);
  const [homographyMatrix, setHomographyMatrix] = useState<number[][] | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [liveFrameData, setLiveFrameData] = useState<any>({ players: {}, ball: null });
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleVideoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setVideoSrc(URL.createObjectURL(file));
      setClickedPoints([]);
      setHomographyMatrix(null);
      setIsPlaying(false);
    }
  };

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      const w = videoRef.current.videoWidth;
      const h = videoRef.current.videoHeight;
      if (w && h) setVideoAspect(w / h);
    }
  };

  const handleVideoClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (clickedPoints.length >= 4 || !containerRef.current || !videoRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const scaleX = videoRef.current.videoWidth / rect.width;
    const scaleY = videoRef.current.videoHeight / rect.height;

    const clickX = (e.clientX - rect.left) * scaleX;
    const clickY = (e.clientY - rect.top) * scaleY;

    setClickedPoints([...clickedPoints, { 
      x: Math.round(clickX), 
      y: Math.round(clickY),
      label: CLICK_LABELS[clickedPoints.length]
    }]);
  };

  const startLiveTracking = () => {
    if (clickedPoints.length !== 4) return;
    
    // FIXED: Corrected destination array configuration to align with the 3D grid layout coordinates
    const dstPoints = [
      { x: 0, y: 68 },   // 1. Top-Left Corner (Far Line) -> X=0, Y=68
      { x: 105, y: 68 }, // 2. Top-Right Corner (Far Line) -> X=105, Y=68
      { x: 105, y: 0 },  // 3. Bottom-Right Corner (Near Line) -> X=105, Y=0
      { x: 0, y: 0 }     // 4. Bottom-Left Corner (Near Line) -> X=0, Y=0
    ];

    const H = solveHomography(clickedPoints, dstPoints);
    setHomographyMatrix(H);
    setIsPlaying(true);
    if (videoRef.current) {
      videoRef.current.play();
    }
  };

  const togglePlay = () => {
    if (!videoRef.current) return;
    if (isPlaying) {
      videoRef.current.pause();
      setIsPlaying(false);
    } else {
      videoRef.current.play();
      setIsPlaying(true);
    }
  };

  return (
    <main className="h-screen w-screen bg-neutral-950 text-neutral-200 flex flex-col font-mono select-none antialiased">
      
      <div className="w-full border-b border-neutral-800 px-6 py-3 bg-neutral-900 flex justify-between items-center">
        <div className="flex items-center gap-3">
          <div className={`w-2 h-2 ${isPlaying ? 'bg-emerald-500 animate-pulse' : 'bg-neutral-500'}`}></div>
          <span className="text-xs font-bold tracking-widest text-white uppercase">TACTICAL COMPUTER VISION RADAR SUITE v3.4 [CLUSTERED DETECTION]</span>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-[11px] text-neutral-400 font-semibold uppercase tracking-wider">SOURCE FEED:</span>
          <input type="file" accept="video/*" onChange={handleVideoUpload} className="text-xs text-neutral-300 border border-neutral-700 px-3 py-1.5 bg-neutral-950 cursor-pointer hover:border-neutral-500 transition" />
        </div>
      </div>

      <div className="flex flex-grow overflow-hidden w-full h-full">
        
        <div className="w-1/2 h-full border-r border-neutral-800 p-6 flex flex-col gap-5 bg-neutral-950 overflow-y-auto">
          {videoSrc ? (
            <div className="flex flex-col gap-4 w-full">
              
              <div className="flex items-center justify-between border-b border-neutral-800 pb-3">
                <span className="text-xs uppercase tracking-widest font-bold text-amber-400">
                  {clickedPoints.length < 4 ? `CALIBRATION STEP [${clickedPoints.length + 1}/4]: SELECT PITCH CORNERS` : "STATUS: MATRIX LOCKED - LIVE READY"}
                </span>
                {clickedPoints.length > 0 && !homographyMatrix && (
                  <button onClick={() => setClickedPoints([])} className="text-[11px] text-neutral-400 hover:text-white underline uppercase tracking-wider transition">
                    Reset Points
                  </button>
                )}
              </div>

              <div 
                ref={containerRef} 
                onClick={handleVideoClick} 
                style={{ aspectRatio: videoAspect }}
                className="relative border border-neutral-800 bg-neutral-900 cursor-crosshair w-full overflow-hidden shadow-2xl"
              >
                <video 
                  ref={videoRef} 
                  src={videoSrc} 
                  onLoadedMetadata={handleLoadedMetadata}
                  className="w-full h-full object-fill" 
                  muted 
                  playsInline
                />
                {clickedPoints.map((pt, idx) => {
                  if (!containerRef.current || !videoRef.current) return null;
                  const rect = containerRef.current.getBoundingClientRect();
                  const scaleX = rect.width / videoRef.current.videoWidth;
                  const scaleY = rect.height / videoRef.current.videoHeight;
                  return (
                    <div key={idx} style={{ left: (pt.x * scaleX) - 10, top: (pt.y * scaleY) - 10 }} className="absolute w-5 h-5 bg-white border-2 border-black text-black text-[10px] font-black flex items-center justify-center shadow-xl">
                      {idx + 1}
                    </div>
                  );
                })}
              </div>

              <div className="bg-neutral-900 border border-neutral-800 p-4 text-[11px] space-y-2">
                <span className="font-bold text-neutral-400 uppercase tracking-wider block mb-1">Calibration Target Progress:</span>
                {CLICK_LABELS.map((lbl, idx) => (
                  <div key={idx} className={`flex items-center gap-2.5 ${clickedPoints[idx] ? 'text-emerald-400 font-bold' : 'text-neutral-500'}`}>
                    <span className="font-mono">[{clickedPoints[idx] ? '✓' : ' '}]</span>
                    <span>{lbl}</span>
                  </div>
                ))}
              </div>

              {clickedPoints.length === 4 && !homographyMatrix && (
                <button 
                  onClick={startLiveTracking} 
                  className="w-full py-3.5 bg-white hover:bg-neutral-200 text-black font-bold text-xs tracking-widest uppercase transition border border-neutral-300 shadow-lg cursor-pointer"
                >
                  START LIVE REAL-TIME TRACKING
                </button>
              )}
            </div>
          ) : (
            <div className="h-full flex flex-col items-center justify-center border border-dashed border-neutral-800 text-xs text-neutral-500 uppercase tracking-widest gap-2">
              <span className="text-neutral-400 font-bold">No match feed uploaded</span>
              <span>Upload live match file to begin calibration</span>
            </div>
          )}
        </div>

        <div className="w-1/2 h-full relative bg-neutral-950 flex flex-col">
          <div className="absolute top-4 left-4 z-20 bg-neutral-900/90 border border-neutral-800 px-3 py-1.5 text-[11px] text-neutral-400 tracking-wider">
            3D TACTICAL RADAR [LIVE ON-THE-GO]
          </div>

          <div className="flex-grow w-full h-full">
            <Canvas camera={{ position: [52.5, 55, 100], fov: 38 }}>
              <color attach="background" args={['#070707']} />
              <LiveSoccerPitch3D 
                videoRef={videoRef} 
                homographyMatrix={homographyMatrix} 
                isPlaying={isPlaying} 
                clickedPoints={clickedPoints}
                liveFrameData={liveFrameData} 
                setLiveFrameData={setLiveFrameData} 
              />
              <OrbitControls target={[52.5, 0, 34]} maxPolarAngle={Math.PI / 2.15} />
            </Canvas>
          </div>

          {homographyMatrix && (
            <div className="bg-neutral-900 border-t border-neutral-800 p-4 flex items-center gap-4 z-30">
              <button onClick={togglePlay} className="px-4 py-2 bg-neutral-950 hover:bg-neutral-800 text-white text-xs font-bold border border-neutral-700 uppercase tracking-widest cursor-pointer">
                {isPlaying ? 'PAUSE LIVE STREAM' : 'RESUME LIVE STREAM'}
              </button>
              <span className="text-xs font-mono text-emerald-400 tracking-widest uppercase">
                {isPlaying ? '● LIVE TRACKING ACTIVE (Adaptive Scan)' : '■ STREAM PAUSED'}
              </span>
            </div>
          )}
        </div>

      </div>
    </main>
  );
}