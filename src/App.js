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

    // Clear the canvas before drawing new content
    ctx.clearRect(0, 0, canvas.current.width, canvas.current.height);

    requestAnimationFrame(() => {
      drawMesh(face, ctx);
      drawKeypoints(pose.keypoints, 0.6, ctx);
      drawSkeleton(pose.keypoints, 0.7, ctx);
      drawRect(obj, ctx);

      if (label === 'Rayhan Al Mim') {
        // Set text properties once
        ctx.fillStyle = 'lime';
        ctx.font = '24px Arial';
        ctx.textBaseline = 'top';
        ctx.textAlign = 'left';

        // Draw text with a background for better visibility
        const text = `${label} (${(confidence * 100).toFixed(1)}%)`;
        const textWidth = ctx.measureText(text).width;

        // Draw semi-transparent background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.fillRect(15, 15, textWidth + 10, 30);

        // Draw text
        ctx.fillStyle = 'lime';
        ctx.fillText(text, 20, 20);
      }
    });
  };

  useEffect(() => {
    run();
  }, []);

  return (
    <div className='App'>
      <div className="header">
        <div className="header-content">
          <div className="university-info">
            <img 
              src='https://res.cloudinary.com/deqkxg249/image/upload/v1746868354/idgnx23u9eiihzbfcyi8.png'
              alt="Green University of Bangladesh Logo"
            />
            <div className="university-text">
              <h1>Green University of Bangladesh</h1>
              <h1>Artificial Intelligence</h1>
            </div>
          </div>
          <div className="project-title">
            <h2>Face Recognition & Detection System</h2>
          </div>
        </div>
      </div>

      <div className="main-container">
        <div className="camera-section">
          <div className="camera-container">
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
                borderRadius: '10px',
                boxShadow: '0 4px 8px rgba(0,0,0,0.2)',
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
                borderRadius: '10px',
              }}
            />
          </div>
        </div>

        <div className="team-section">
          <h3>Project Team Members</h3>
          <div className="team-members">
            <div className="member-card">
              <div className="member-info">
                <h4>Rayhan Al Mim</h4>
                <p>ID: 222015010</p>
                <p>Team Lead</p>
              </div>
            </div>
            <div className="member-card">
              <div className="member-info">
                <h4>Nasrin Jahan Fateen</h4>
                <p>ID: 222015015</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <footer className="footer">
        <p>© 2024 University AI Lab - All Rights Reserved</p>
      </footer>
    </div>
  );
}

export default App;
