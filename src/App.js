import React, { useRef, useEffect, useState } from 'react';
import './App.css';
import Webcam from 'react-webcam';

import * as tf from '@tensorflow/tfjs';
import * as facemesh from '@tensorflow-models/face-landmarks-detection';
import * as posenet from '@tensorflow-models/posenet';
import * as cocossd from '@tensorflow-models/coco-ssd';

import { drawMesh, drawKeypoints, drawSkeleton, drawRect } from './utilities';

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [label, setLabel] = useState('');

  const run = async () => {
    const faceNet = await facemesh.load(
      facemesh.SupportedPackages.mediapipeFacemesh,
    );
    const poseNet = await posenet.load({
      inputResolution: { width: 640, height: 480 },
      scale: 0.8,
    });
    const cocoNet = await cocossd.load();

    const testdata = await fetch(process.env.PUBLIC_URL + '/my-model/weights.bin')
    .then(res => res.blob())
    .then(blob => console.log('Weights file size (bytes):', blob.size));

    console.log('my data test', testdata);

    // Load your custom Teachable Machine model
    const model = await tf.loadLayersModel('my-model/model.json');

    console.log('Custom model loaded');

    setInterval(() => {
      detect(faceNet, poseNet, cocoNet, model);
    }, 200);
  };

  const detect = async (faceNet, poseNet, cocoNet, model) => {
    if (webcamRef.current && webcamRef.current.video.readyState === 4) {
      const video = webcamRef.current.video;
      const videoWidth = video.videoWidth;
      const videoHeight = video.videoHeight;
  
      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;
  
      // Detect face
      const face = await faceNet.estimateFaces({ input: video });


      console.log('Face detection result:', face);
  
      // If no face found, clear label and skip prediction
      if (!face || face.length === 0) {
        setLabel('');
        const pose = await poseNet.estimateSinglePose(video);
        const obj = await cocoNet.detect(video);
        drawCanvas([], pose, obj, canvasRef, '', 0);
        return;
      }
  
      // Run full prediction only if face is detected
      const pose = await poseNet.estimateSinglePose(video);
      const obj = await cocoNet.detect(video);
  
      const input = tf.browser
        .fromPixels(video)
        .resizeNearestNeighbor([224, 224])
        .toFloat()
        .expandDims();
  
      const prediction = await model.predict(input);
      const predictionData = await prediction.data();
      const maxIdx = predictionData.indexOf(Math.max(...predictionData));
  
      const classLabels = ['Rayhan Al Mim', 'Unknown'];
      const className = classLabels[maxIdx];
  
      if (className === 'Rayhan Al Mim' && predictionData[maxIdx] > 0.9) {
        setLabel('Rayhan Al Mim');
      } else {
        setLabel('');
      }
  
      drawCanvas(face, pose, obj, canvasRef, className, predictionData[maxIdx]);
    }
  };
  

  const drawCanvas = (face, pose, obj, canvas, label, confidence) => {
    const ctx = canvas.current.getContext('2d');

    requestAnimationFrame(() => {
      drawMesh(face, ctx);
      drawKeypoints(pose.keypoints, 0.6, ctx);
      drawSkeleton(pose.keypoints, 0.7, ctx);
      drawRect(obj, ctx);

      if (label === 'Rayhan Al Mim') {
        ctx.fillStyle = 'lime';
        ctx.font = '24px Arial';
        ctx.fillText(`${label} (${(confidence * 100).toFixed(1)}%)`, 20, 40);
      }
    });
  };

  useEffect(() => {
    run();
  }, []);

  return (
    <div className='App'>
      <header className='App-header'>
        <Webcam
          ref={webcamRef}
          style={{
            position: 'absolute',
            marginLeft: 'auto',
            marginRight: 'auto',
            left: 0,
            right: 0,
            textAlign: 'center',
            zIndex: 9,
            width: 640,
            height: 480,
          }}
        />
        <canvas
          ref={canvasRef}
          style={{
            position: 'absolute',
            marginLeft: 'auto',
            marginRight: 'auto',
            left: 0,
            right: 0,
            textAlign: 'center',
            zIndex: 9,
            width: 640,
            height: 480,
          }}
        />
      </header>
    </div>
  );
}

export default App;
