import { NextResponse } from 'next/server';
import { exec } from 'child_process';
import fs from 'fs';
import path from 'path';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { points } = body;

    if (!points || points.length !== 4) {
      return NextResponse.json({ success: false, error: "Invalid alignment points payload." }, { status: 400 });
    }

    // Format click coordinates string passed into the python runtime script arguments
    const ptsArg = `${points[0].x},${points[0].y},${points[1].x},${points[1].y},${points[2].x},${points[2].y},${points[3].x},${points[3].y}`;
    
    // Explicit absolute tracking script route paths
    const scriptPath = path.join(process.cwd(), 'track_onnx.py');

    console.log(`Starting processing pipeline execution with coordinates: ${ptsArg}`);

    // Execute tracking pipeline directly using your local OpenCV 5 environment script
    await new Promise((resolve, reject) => {
      exec(`python "${scriptPath}" --points ${ptsArg}`, (error, stdout, stderr) => {
        if (error) {
          console.error(`Execution error context: ${stderr || error.message}`);
          reject(new Error(stderr || error.message));
        } else {
          resolve(stdout);
        }
      });
    });

    // Read back the compiled tracking telemetry output json file
    const dataPath = path.join(process.cwd(), 'public', 'tracking_data.json');
    if (!fs.existsSync(dataPath)) {
      return NextResponse.json({ success: false, error: "Tracking script ran but output matrix file was not created." }, { status: 500 });
    }

    const trackingJson = JSON.parse(fs.readFileSync(dataPath, 'utf8'));
    return NextResponse.json({ success: true, tracking: trackingJson });

  } catch (error: any) {
    console.error("Internal Worker Error Stack:", error.message);
    return NextResponse.json({ success: false, error: error.message }, { status: 500 });
  }
}