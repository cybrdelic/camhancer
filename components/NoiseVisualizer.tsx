import React, { useEffect, useRef, useCallback, useState } from 'react';

// --- WebGPU Type Polyfills ---
type GPUDevice = any;
type GPUCanvasContext = any;
type GPURenderPipeline = any;
type GPUBuffer = any;
type GPUSampler = any;
type GPUBindGroup = any;
type GPUExternalTexture = any;
type GPUTexture = any;

const GPUBufferUsage = {
  UNIFORM: 64,
  COPY_DST: 8,
};

const GPUTextureUsage = {
  COPY_SRC: 0x01,
  COPY_DST: 0x02,
  TEXTURE_BINDING: 0x04,
  RENDER_ATTACHMENT: 0x10,
};

// --- WGSL Shaders ---

const FULLSCREEN_VERTEX_SHADER = `
  struct VertexOutput {
    @builtin(position) Position : vec4<f32>,
    @location(0) uv : vec2<f32>,
  }

  @vertex
  fn main(@builtin(vertex_index) VertexIndex : u32) -> VertexOutput {
    var pos = array<vec2<f32>, 6>(
      vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
      vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
    );

    var output : VertexOutput;
    output.Position = vec4<f32>(pos[VertexIndex], 0.0, 1.0);
    output.uv = pos[VertexIndex] * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    return output;
  }
`;

/**
 * ALGORITHM: Professional ISP Engine
 * 1. Lens Correction (Chromatic Aberration + Vignette)
 * 2. Perceptual OKLab Color Transformation
 * 3. Variance-Clipped Temporal Stability (TAA-style)
 * 4. Tensor-Guided Anisotropic Luma Filter
 * 5. Luma-Guided Joint Bilateral Chroma Filter
 * 6. Coherence-Guided Structural Sharpening
 * 7. Tensor-Flow Temporal Anti-Aliasing (TFAA)
 * 8. Dithering
 */
const RESEARCH_GRADE_FRAGMENT_SHADER = `
  @group(0) @binding(0) var mySampler : sampler;
  @group(0) @binding(1) var myTexture : texture_external;
  @group(0) @binding(2) var<uniform> uniforms : Uniforms;
  @group(0) @binding(3) var historyTexture : texture_2d<f32>;
  
  struct Uniforms {
    resolution: vec2<f32>, // Source Resolution (Video)
    spatialSigma: f32,    
    rangeSigma: f32,      
    anisotropy: f32,      
    chromaBoost: f32,     
    mode: f32, 
    comparisonPos: f32,   
    temporalStrength: f32,
    sharpness: f32,
    lensDistortion: f32,
    vignetteStrength: f32,
    aaStrength: f32,
    temporalGamma: f32    // Variance Clip Width
  };

  const PI: f32 = 3.14159265;

  // --- BICUBIC INTERPOLATION (CATMULL-ROM) ---
  // Calculates weights for cubic interpolation
  fn w0(a: f32) -> f32 { return (1.0/6.0) * (a*(a*(-a + 3.0) - 3.0) + 1.0); }
  fn w1(a: f32) -> f32 { return (1.0/6.0) * (a*a*(3.0*a - 6.0) + 4.0); }
  fn w2(a: f32) -> f32 { return (1.0/6.0) * (a*(a*(-3.0*a + 3.0) + 3.0) + 1.0); }
  fn w3(a: f32) -> f32 { return (1.0/6.0) * (a*a*a); }

  fn sampleBicubic(uv: vec2<f32>) -> vec4<f32> {
      let texSize = uniforms.resolution;
      let invTexSize = 1.0 / texSize;
      
      let tc = uv * texSize - 0.5;
      let tcIndex = floor(tc);
      let f = fract(tc);
      
      // We rely on 4 bilinear samples to approximate 16 bicubic samples.
      // This is a standard GPU optimization trick.
      
      let w0_x = w0(f.x); let w1_x = w1(f.x); let w2_x = w2(f.x); let w3_x = w3(f.x);
      let w0_y = w0(f.y); let w1_y = w1(f.y); let w2_y = w2(f.y); let w3_y = w3(f.y);

      let g0_x = w0_x + w1_x; 
      let g1_x = w2_x + w3_x;
      let h0_x = (w1_x / g0_x) - 0.5 + tcIndex.x;
      let h1_x = (w3_x / g1_x) + 1.5 + tcIndex.x;
      
      let g0_y = w0_y + w1_y;
      let g1_y = w2_y + w3_y;
      let h0_y = (w1_y / g0_y) - 0.5 + tcIndex.y;
      let h1_y = (w3_y / g1_y) + 1.5 + tcIndex.y;
      
      let uv00 = vec2<f32>(h0_x, h0_y) * invTexSize;
      let uv10 = vec2<f32>(h1_x, h0_y) * invTexSize;
      let uv01 = vec2<f32>(h0_x, h1_y) * invTexSize;
      let uv11 = vec2<f32>(h1_x, h1_y) * invTexSize;
      
      let c00 = textureSampleBaseClampToEdge(myTexture, mySampler, uv00);
      let c10 = textureSampleBaseClampToEdge(myTexture, mySampler, uv10);
      let c01 = textureSampleBaseClampToEdge(myTexture, mySampler, uv01);
      let c11 = textureSampleBaseClampToEdge(myTexture, mySampler, uv11);
      
      let colA = mix(c00, c10, g1_x / (g0_x + g1_x));
      let colB = mix(c01, c11, g1_x / (g0_x + g1_x));
      
      return mix(colA, colB, g1_y / (g0_y + g1_y));
  }

  // --- COLOR SPACE UTILS (OKLAB) ---
  
  fn sRGBToLinear(color: vec3<f32>) -> vec3<f32> {
    let safeColor = max(color, vec3<f32>(0.0));
    return pow(safeColor, vec3<f32>(2.2));
  }

  fn linearToSRGB(color: vec3<f32>) -> vec3<f32> {
    let safeColor = max(color, vec3<f32>(0.0));
    return pow(safeColor, vec3<f32>(1.0 / 2.2));
  }

  // OKLab is perceptually uniform. 
  // L = Lightness, a = Green-Red, b = Blue-Yellow
  fn linearToOklab(c: vec3<f32>) -> vec3<f32> {
      let l = 0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b;
      let m = 0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b;
      let s = 0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b;
      let l_ = pow(max(l, 0.0), 1.0/3.0);
      let m_ = pow(max(m, 0.0), 1.0/3.0);
      let s_ = pow(max(s, 0.0), 1.0/3.0);
      return vec3<f32>(
          0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
          1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
          0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
      );
  }

  fn oklabToLinear(c: vec3<f32>) -> vec3<f32> {
      let l_ = c.x + 0.3963377774 * c.y + 0.2158037573 * c.z;
      let m_ = c.x - 0.1055613458 * c.y - 0.0638541728 * c.z;
      let s_ = c.x - 0.0894841775 * c.y - 1.2914855480 * c.z;
      let l = l_ * l_ * l_;
      let m = m_ * m_ * m_;
      let s = s_ * s_ * s_;
      return vec3<f32>(
          4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
          -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
          -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
      );
  }

  fn gaussian(x: f32, sigma: f32) -> f32 {
    return exp(-(x * x) / (2.0 * sigma * sigma));
  }

  fn rand(co: vec2<f32>) -> f32 {
      return fract(sin(dot(co, vec2<f32>(12.9898, 78.233))) * 43758.5453);
  }

  // --- ISP: Lens Correction & Super Res Sampling ---
  fn sampleLensCorrected(uv: vec2<f32>) -> vec3<f32> {
    let dist = distance(uv, vec2<f32>(0.5));
    let aberration = uniforms.lensDistortion * dist * 0.01;
    
    // Chromatic Aberration with Bicubic Interpolation
    let r = sampleBicubic(uv - aberration * (uv - 0.5)).r;
    let g = sampleBicubic(uv).g;
    let b = sampleBicubic(uv + aberration * (uv - 0.5)).b;
    
    var color = vec3<f32>(r, g, b);

    // Vignette Correction
    let vignette = 1.0 - (uniforms.vignetteStrength * dist * dist);
    color = color / max(vignette, 0.4); 

    return sRGBToLinear(color);
  }

  @fragment
  fn main(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
    // Standard Bilinear for Raw view
    let rawColorSRGB = textureSampleBaseClampToEdge(myTexture, mySampler, uv).rgb;
    
    // Bicubic Super-Res for processed pipeline
    let ispColorLinear = sampleLensCorrected(uv);
    let ispColorOklab = linearToOklab(ispColorLinear); 
    
    // NOTE: onePixel refers to the SOURCE sensor texel size.
    let onePixel = vec2<f32>(1.0) / uniforms.resolution;

    // --- 0. Variance-Clipped Temporal Accumulation ---
    // Instead of min/max box, we use statistical variance.
    // This allows for "softer" clamping that preserves sub-pixel detail better.
    
    var m1 = vec3<f32>(0.0); // First Moment (Mean)
    var m2 = vec3<f32>(0.0); // Second Moment (Variance)
    
    // 3x3 Block Statistics (Sampling source texels)
    for (var i = -1; i <= 1; i++) {
        for (var j = -1; j <= 1; j++) {
            let sampleUV = uv + vec2<f32>(f32(i), f32(j)) * onePixel;
            let val = linearToOklab(sampleLensCorrected(sampleUV));
            m1 += val;
            m2 += val * val;
        }
    }
    let mean = m1 / 9.0;
    let sigma = sqrt(abs((m2 / 9.0) - (mean * mean)));
    
    // Variance Clipping: Mean +/- Gamma * Sigma
    let gamma = uniforms.temporalGamma;
    let minBox = mean - gamma * sigma;
    let maxBox = mean + gamma * sigma;

    // History Sample (OKLab)
    let histColorSample = textureSampleLevel(historyTexture, mySampler, uv, 0.0);
    let histColorLinear = sRGBToLinear(histColorSample.rgb);
    let histColorOklab = linearToOklab(histColorLinear);
    
    // Clip History to Statistical Box
    let clippedHist = clamp(histColorOklab, minBox, maxBox);
    
    // Motion Detection (Luma-based in OKLab)
    let diff = abs(mean.x - clippedHist.x);
    let motionFactor = smoothstep(0.002, 0.03, diff); 
    
    let blend = uniforms.temporalStrength * (1.0 - motionFactor);
    let accumulatedOklab = mix(ispColorOklab, clippedHist, blend);

    if (uniforms.mode < 0.1) {
       return vec4<f32>(linearToSRGB(oklabToLinear(accumulatedOklab)), 1.0); 
    }

    // --- 1. Structure Tensor (Calculated on OKLab Luma) ---
    
    let tl = linearToOklab(sampleLensCorrected(uv + vec2<f32>(-1.0, -1.0) * onePixel)).x;
    let t  = linearToOklab(sampleLensCorrected(uv + vec2<f32>( 0.0, -1.0) * onePixel)).x;
    let tr = linearToOklab(sampleLensCorrected(uv + vec2<f32>( 1.0, -1.0) * onePixel)).x;
    let l  = linearToOklab(sampleLensCorrected(uv + vec2<f32>(-1.0,  0.0) * onePixel)).x;
    let r  = linearToOklab(sampleLensCorrected(uv + vec2<f32>( 1.0,  0.0) * onePixel)).x;
    let bl = linearToOklab(sampleLensCorrected(uv + vec2<f32>(-1.0,  1.0) * onePixel)).x;
    let b  = linearToOklab(sampleLensCorrected(uv + vec2<f32>( 0.0,  1.0) * onePixel)).x;
    let br = linearToOklab(sampleLensCorrected(uv + vec2<f32>( 1.0,  1.0) * onePixel)).x;

    let Ix = (tr + 2.0*r + br) - (tl + 2.0*l + bl);
    let Iy = (bl + 2.0*b + br) - (tl + 2.0*t + tr);

    let J11 = Ix * Ix;
    let J12 = Ix * Iy;
    let J22 = Iy * Iy;

    let trace = J11 + J22;
    let det = J11 * J22 - J12 * J12;
    let D = sqrt(max(0.0, trace * trace - 4.0 * det));
    let lambda1 = (trace + D) / 2.0; 
    let lambda2 = (trace - D) / 2.0; 
    let coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + 0.00001); 

    let gradLen = sqrt(Ix*Ix + Iy*Iy) + 0.00001;
    let v1 = vec2<f32>(Ix, Iy) / gradLen; 
    let v2 = vec2<f32>(-Iy, Ix) / gradLen; 

    // --- 2. Advanced Spatial Filter (Separated Luma vs Chroma) ---
    
    // Signal Model based on OKLab Lightness
    let signalNoiseModel = mix(2.0, 0.8, sqrt(max(accumulatedOklab.x, 0.0))); 
    let spatialBoost = (1.0 + (motionFactor * 3.0)) * signalNoiseModel; 
    
    let A = clamp(coherence * uniforms.anisotropy, 0.0, 0.98); 
    let sigmaX = uniforms.spatialSigma * spatialBoost * (1.0 - A); 
    let sigmaY = uniforms.spatialSigma * spatialBoost * (1.0 + A); 

    // Accumulators
    var sumL = 0.0; var weightL = 0.0001;
    var sumA = 0.0; var weightA = 0.0001;
    var sumB = 0.0; var weightB = 0.0001;

    let centerOklab = accumulatedOklab;
    let K = 4; 

    for (var i = -K; i <= K; i++) {
      for (var j = -K; j <= K; j++) {
        let offset = vec2<f32>(f32(i), f32(j));
        if (dot(offset, offset) > f32(K*K + 1)) { continue; }

        let sampleUV = uv + offset * onePixel;
        let sampleOklab = linearToOklab(sampleLensCorrected(sampleUV));

        // Spatial Weight (Anisotropic for Luma, Isotropic for Chroma)
        let dNormal  = dot(offset, v1);
        let dTangent = dot(offset, v2);
        let wSpatialL = gaussian(dNormal, max(0.5, sigmaX)) * gaussian(dTangent, max(0.5, sigmaY));
        let wSpatialC = gaussian(length(offset), max(0.5, sigmaX * 2.0)); // Chroma is blurred more

        // Range Weights
        // Guided Filter: Chroma weights depend on LUMA differences
        let lumaDiff = abs(centerOklab.x - sampleOklab.x);
        
        let wRangeL = gaussian(lumaDiff, uniforms.rangeSigma * signalNoiseModel);
        
        // Chroma Guide: If Luma is similar, we blur chroma aggressively. 
        // If Luma is different (edge), we stop chroma blur to prevent bleeding.
        let wRangeC = gaussian(lumaDiff, uniforms.rangeSigma * 0.5); 

        // Luma Accumulation
        let wL = wSpatialL * wRangeL;
        sumL += sampleOklab.x * wL;
        weightL += wL;

        // Chroma Accumulation
        let wC = wSpatialC * wRangeC * uniforms.chromaBoost;
        sumA += sampleOklab.y * wC;
        weightA += wC;
        
        sumB += sampleOklab.z * wC;
        weightB += wC;
      }
    }

    let finalL = sumL / weightL;
    let finalA = sumA / weightA;
    let finalB = sumB / weightB;
    
    var finalOklab = vec3<f32>(finalL, finalA, finalB);

    // --- 3. Coherence-Guided Structural Sharpening ---

    let lumaAccum = accumulatedOklab.x;
    let lumaFinal = finalOklab.x;
    let detailVal = lumaAccum - lumaFinal;

    let structuralMask = smoothstep(0.15, 0.6, coherence);
    let shadowProtection = smoothstep(0.0, 0.15, finalL);

    let smartDetail = detailVal * structuralMask * shadowProtection * uniforms.sharpness;

    // Add detail back to Luma only
    finalOklab.x += smartDetail;

    // Anti-Halo: Clamp to statistical neighborhood
    let haloRelax = 0.02; 
    finalOklab.x = clamp(finalOklab.x, minBox.x - haloRelax, maxBox.x + haloRelax);

    // --- 4. Tensor-Flow Temporal Anti-Aliasing (TFAA) ---
    if (uniforms.aaStrength > 0.0 && motionFactor < 0.15) {
        // Use textureSampleLevel to allow gradient-free sampling inside conditional
        let histP = linearToOklab(sRGBToLinear(textureSampleLevel(historyTexture, mySampler, uv + v2 * onePixel * 0.75, 0.0).rgb));
        let histN = linearToOklab(sRGBToLinear(textureSampleLevel(historyTexture, mySampler, uv - v2 * onePixel * 0.75, 0.0).rgb));
        
        let tangentAvg = (histP + histN) * 0.5;
        finalOklab = mix(finalOklab, tangentAvg, coherence * uniforms.aaStrength);
    }

    // --- 5. Output ---
    let finalLinear = oklabToLinear(finalOklab);
    let ditherNoise = (rand(uv) - 0.5) / 255.0; 
    let finalSRGB = linearToSRGB(max(vec3<f32>(0.0), finalLinear)) + ditherNoise;

    if (uniforms.mode == 1.0) {
        return vec4<f32>(coherence, coherence, coherence, 1.0);
    }
    else if (uniforms.mode == 2.0) {
        let split = uniforms.comparisonPos;
        let lineThickness = 2.0 * onePixel.x;
        if (abs(uv.x - split) < lineThickness) { return vec4<f32>(1.0, 1.0, 1.0, 1.0); }
        if (uv.x < split) { return vec4<f32>(rawColorSRGB, 1.0); } 
        return vec4<f32>(finalSRGB, 1.0); 
    }
    
    return vec4<f32>(finalSRGB, 1.0);
  }
`;

interface NoiseVisualizerProps {
  spatialSigma: number;
  rangeSigma: number;
  anisotropy: number;
  chromaBoost: number;
  temporalStrength: number;
  sharpness: number;
  lensDistortion: number;
  vignetteStrength: number;
  aaStrength: number;
  temporalGamma: number;
  resolutionScale: number;
  displayZoom: number; // New: For pixel peeping
  mode: number;
  onStatsUpdate: (fps: number) => void;
  isPaused: boolean;
}

const NoiseVisualizer: React.FC<NoiseVisualizerProps> = ({ 
  spatialSigma,
  rangeSigma,
  anisotropy,
  chromaBoost,
  temporalStrength,
  sharpness,
  lensDistortion,
  vignetteStrength,
  aaStrength,
  temporalGamma,
  resolutionScale,
  displayZoom,
  mode, 
  onStatsUpdate, 
  isPaused,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const deviceRef = useRef<GPUDevice | null>(null);
  const contextRef = useRef<GPUCanvasContext | null>(null);
  const pipelineRef = useRef<GPURenderPipeline | null>(null);
  const uniformBufferRef = useRef<GPUBuffer | null>(null);
  const samplerRef = useRef<GPUSampler | null>(null);
  const historyTextureRef = useRef<GPUTexture | null>(null);
  
  const requestRef = useRef<number>(0);
  const lastTimeRef = useRef<number>(0);
  const frameCountRef = useRef<number>(0);

  const comparisonPosRef = useRef<number>(0.5);
  const motionCheckFrameCount = useRef(0);

  // Parameter Ref Pattern (Fixes Stuttering)
  const paramsRef = useRef({
    spatialSigma,
    rangeSigma,
    anisotropy,
    chromaBoost,
    temporalStrength,
    sharpness,
    lensDistortion,
    vignetteStrength,
    aaStrength,
    temporalGamma,
    resolutionScale,
    mode,
    isPaused
  });

  // Sync props to ref
  useEffect(() => {
    paramsRef.current = {
        spatialSigma,
        rangeSigma,
        anisotropy,
        chromaBoost,
        temporalStrength,
        sharpness,
        lensDistortion,
        vignetteStrength,
        aaStrength,
        temporalGamma,
        resolutionScale,
        mode,
        isPaused
    };
  }, [spatialSigma, rangeSigma, anisotropy, chromaBoost, temporalStrength, sharpness, lensDistortion, vignetteStrength, aaStrength, temporalGamma, resolutionScale, mode, isPaused]);

  // Pan/Zoom State
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const isDraggingRef = useRef(false);
  const lastMousePosRef = useRef({ x: 0, y: 0 });

  const initWebGPU = useCallback(async () => {
    if (!(navigator as any).gpu) {
      console.error("WebGPU not supported");
      return;
    }

    const adapter = await (navigator as any).gpu.requestAdapter();
    if (!adapter) return;

    const device = await adapter.requestDevice();
    deviceRef.current = device;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const context = canvas.getContext('webgpu') as unknown as GPUCanvasContext;
    contextRef.current = context;

    const format = (navigator as any).gpu.getPreferredCanvasFormat();
    
    context.configure({
      device,
      format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      alphaMode: 'premultiplied',
    });

    const pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module: device.createShaderModule({ code: FULLSCREEN_VERTEX_SHADER }),
        entryPoint: 'main',
      },
      fragment: {
        module: device.createShaderModule({ code: RESEARCH_GRADE_FRAGMENT_SHADER }),
        entryPoint: 'main',
        targets: [{ format }],
      },
      primitive: {
        topology: 'triangle-list',
      },
    });
    pipelineRef.current = pipeline;

    const uniformBuffer = device.createBuffer({
      size: 64, 
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    uniformBufferRef.current = uniformBuffer;

    const sampler = device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
    });
    samplerRef.current = sampler;
  }, []);

  useEffect(() => {
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { 
            width: { ideal: 1920 },
            height: { ideal: 1080 },
            facingMode: "user"
          } 
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }
        await initWebGPU();
      } catch (err) {
        console.error("Error accessing camera:", err);
      }
    };

    startCamera();

    return () => {
      if (videoRef.current?.srcObject) {
        const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
        tracks.forEach(track => track.stop());
      }
      cancelAnimationFrame(requestRef.current);
      // Clean up texture on unmount
      if (historyTextureRef.current) {
          historyTextureRef.current.destroy();
      }
    };
  }, [initWebGPU]);

  const render = useCallback((time: number) => {
    const params = paramsRef.current;
    
    if (params.isPaused) {
      requestRef.current = requestAnimationFrame(render);
      return;
    }

    const device = deviceRef.current;
    const context = contextRef.current;
    const pipeline = pipelineRef.current;
    const video = videoRef.current;
    const uniformBuffer = uniformBufferRef.current;
    const sampler = samplerRef.current;

    if (!device || !context || !pipeline || !video || !uniformBuffer || !sampler) {
       requestRef.current = requestAnimationFrame(render);
       return;
    }

    if (video.readyState < 2) {
        requestRef.current = requestAnimationFrame(render);
        return;
    }

    const delta = time - lastTimeRef.current;
    frameCountRef.current++;
    motionCheckFrameCount.current++;
    
    if (delta > 1000) {
      onStatsUpdate(Math.round((frameCountRef.current * 1000) / delta));
      lastTimeRef.current = time;
      frameCountRef.current = 0;
    }

    // Canvas Sizing Logic (Super Resolution)
    const targetWidth = Math.floor(video.videoWidth * params.resolutionScale);
    const targetHeight = Math.floor(video.videoHeight * params.resolutionScale);

    if (canvasRef.current && (canvasRef.current.width !== targetWidth || canvasRef.current.height !== targetHeight)) {
      canvasRef.current.width = targetWidth;
      canvasRef.current.height = targetHeight;
      // Force recreation of history texture if size changes, cleaning up old one
      if (historyTextureRef.current) {
          historyTextureRef.current.destroy();
      }
      historyTextureRef.current = null;
    }

    if (!historyTextureRef.current) {
        historyTextureRef.current = device.createTexture({
            size: [targetWidth, targetHeight, 1], // History matches output resolution
            format: (navigator as any).gpu.getPreferredCanvasFormat(), 
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
        });
    }
    const historyTexture = historyTextureRef.current;

    const uniformData = new Float32Array([
      video.videoWidth, 
      video.videoHeight, 
      params.spatialSigma,
      params.rangeSigma,
      params.anisotropy,
      params.chromaBoost,
      params.mode,
      comparisonPosRef.current,
      params.temporalStrength,
      params.sharpness, 
      params.lensDistortion, 
      params.vignetteStrength,
      params.aaStrength,
      params.temporalGamma
    ]);
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    let externalTexture: GPUExternalTexture;
    try {
        externalTexture = device.importExternalTexture({ source: video });
    } catch (e) {
        requestRef.current = requestAnimationFrame(render);
        return;
    }

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: sampler },
        { binding: 1, resource: externalTexture },
        { binding: 2, resource: { buffer: uniformBuffer } },
        { binding: 3, resource: historyTexture.createView() },
      ],
    });

    const commandEncoder = device.createCommandEncoder();
    const currentTexture = context.getCurrentTexture();
    const textureView = currentTexture.createView();

    const passEncoder = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: textureView,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });

    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.draw(6);
    passEncoder.end();

    // Copy the high-res output back to history
    commandEncoder.copyTextureToTexture(
        { texture: currentTexture },
        { texture: historyTexture },
        [targetWidth, targetHeight, 1]
    );

    device.queue.submit([commandEncoder.finish()]);

    requestRef.current = requestAnimationFrame(render);
  }, [onStatsUpdate]); // Minimal dependencies to ensure loop stability

  useEffect(() => {
    requestRef.current = requestAnimationFrame(render);
    return () => cancelAnimationFrame(requestRef.current);
  }, [render]);

  const handleInteraction = useCallback((clientX: number) => {
    // Access latest mode from ref
    if (paramsRef.current.mode !== 2 || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    comparisonPosRef.current = x;
  }, []);

  // Mouse Drag for Pan (when zoomed)
  const onMouseDown = (e: React.MouseEvent) => {
      if (displayZoom > 1.0) {
          isDraggingRef.current = true;
          lastMousePosRef.current = { x: e.clientX, y: e.clientY };
      } else {
          handleInteraction(e.clientX);
      }
  };

  const onMouseMove = (e: React.MouseEvent) => {
    if (displayZoom > 1.0 && isDraggingRef.current) {
        const dx = e.clientX - lastMousePosRef.current.x;
        const dy = e.clientY - lastMousePosRef.current.y;
        setPan(prev => ({ x: prev.x + dx, y: prev.y + dy }));
        lastMousePosRef.current = { x: e.clientX, y: e.clientY };
    } else if (e.buttons > 0) {
        handleInteraction(e.clientX);
    }
  };
  
  const onMouseUp = () => {
      isDraggingRef.current = false;
  };

  const onTouchMove = (e: React.TouchEvent) => {
     handleInteraction(e.touches[0].clientX);
  };
  
  // Reset pan when scale changes
  useEffect(() => {
      setPan({x: 0, y: 0});
  }, [displayZoom]);

  return (
    <div 
      ref={containerRef}
      className={`relative w-full h-full bg-black flex items-center justify-center overflow-hidden group 
        ${mode === 2 ? 'cursor-col-resize' : ''} 
        ${displayZoom > 1 ? 'cursor-move' : ''}
      `}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      onMouseLeave={onMouseUp}
      onTouchMove={onTouchMove}
    >
      <video ref={videoRef} muted playsInline className="hidden" />
      
      {/* Canvas Container with Transform for Zoom */}
      <div 
        style={{ 
            transform: `scale(${displayZoom}) translate(${pan.x / displayZoom}px, ${pan.y / displayZoom}px)`,
            transformOrigin: 'center center',
            transition: isDraggingRef.current ? 'none' : 'transform 0.2s cubic-bezier(0.25, 0.46, 0.45, 0.94)'
        }}
        className="will-change-transform shadow-2xl"
      >
          <canvas ref={canvasRef} className="block" style={{ width: '100%', height: 'auto' }} />
      </div>

      {/* Zoom Indicator Grid */}
      {displayZoom >= 2.0 && (
          <div className="absolute inset-0 pointer-events-none opacity-20 bg-[url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMCIgaGVpZ2h0PSIyMCI+PHBhdGggZD0iTTEgMGgwLTV2MjBoMXptMCAxaDIwdjEwMCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIvPjwvc3ZnPg==')]"></div>
      )}

      {/* Comparison Overlay Labels */}
      {mode === 2 && (
         <div className="absolute inset-x-0 bottom-8 px-8 flex justify-between pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-300">
             <div className="bg-black/80 text-white text-[10px] font-bold tracking-widest px-3 py-1 uppercase">
                 Source
             </div>
             <div className="bg-white text-black text-[10px] font-bold tracking-widest px-3 py-1 uppercase">
                 Processed
             </div>
         </div>
      )}
    </div>
  );
};

export default NoiseVisualizer;