import React, { useState } from 'react';
import NoiseVisualizer from './components/NoiseVisualizer';

const App: React.FC = () => {
  // Advanced Filter Parameters
  const [spatialSigma, setSpatialSigma] = useState<number>(2.0); // Filter Radius
  const [rangeSigma, setRangeSigma] = useState<number>(0.20);    // Edge sensitivity
  const [anisotropy, setAnisotropy] = useState<number>(0.75);    // Edge flow
  const [chromaBoost, setChromaBoost] = useState<number>(2.5);   // Chroma denoise
  const [temporalStrength, setTemporalStrength] = useState<number>(0.90); 
  const [sharpness, setSharpness] = useState<number>(0.4); 
  
  // Optical Corrections
  const [lensDistortion, setLensDistortion] = useState<number>(1.5); 
  const [vignetteStrength, setVignetteStrength] = useState<number>(0.3); 
  
  // Anti-Aliasing (New)
  const [aaStrength, setAaStrength] = useState<number>(0.5); 

  // Temporal Stability (New)
  const [temporalGamma, setTemporalGamma] = useState<number>(1.25); // Variance Clip Width

  // Super Resolution & Inspection
  const [resolutionScale, setResolutionScale] = useState<number>(1.0);
  const [displayZoom, setDisplayZoom] = useState<number>(1.0); // Visual Zoom (Fit, 1x, 2x)

  const [mode, setMode] = useState<number>(3); 
  const [fps, setFps] = useState<number>(0);
  const [isPaused, setIsPaused] = useState<boolean>(false);

  const SliderGroup = ({ label, value, min, max, step, onChange, unit = "" }: any) => {
    const percentage = Math.min(100, Math.max(0, ((value - min) / (max - min)) * 100));

    return (
      <div className="group py-3">
        <div className="flex justify-between items-end mb-3">
          <label className="text-[10px] font-black text-zinc-500 uppercase tracking-widest group-hover:text-white transition-colors duration-300">
            {label}
          </label>
          <div className="text-[10px] font-mono text-zinc-500 group-hover:text-white transition-colors duration-300 bg-zinc-900/50 px-1.5 py-0.5 rounded">
            {value.toFixed(step < 0.1 ? 2 : 1)}{unit}
          </div>
        </div>
        
        <div className="relative h-6 flex items-center w-full">
           <input 
            type="range" 
            min={min} 
            max={max} 
            step={step}
            value={value}
            onChange={(e) => onChange(parseFloat(e.target.value))}
            style={{
              background: `linear-gradient(to right, white 0%, white ${percentage}%, #27272a ${percentage}%, #27272a 100%)`
            }}
            className="w-full h-[2px] appearance-none cursor-pointer focus:outline-none bg-zinc-800 rounded-none
              [&::-webkit-slider-thumb]:appearance-none 
              [&::-webkit-slider-thumb]:w-4 
              [&::-webkit-slider-thumb]:h-4 
              [&::-webkit-slider-thumb]:bg-white 
              [&::-webkit-slider-thumb]:border-[4px]
              [&::-webkit-slider-thumb]:border-transparent
              [&::-webkit-slider-thumb]:ring-1
              [&::-webkit-slider-thumb]:ring-black
              [&::-webkit-slider-thumb]:shadow-lg
              [&::-webkit-slider-thumb]:transition-transform
              [&::-webkit-slider-thumb]:hover:scale-110
              
              [&::-moz-range-thumb]:w-4
              [&::-moz-range-thumb]:h-4
              [&::-moz-range-thumb]:bg-white
              [&::-moz-range-thumb]:border-[4px]
              [&::-moz-range-thumb]:border-transparent
              [&::-moz-range-thumb]:ring-1
              [&::-moz-range-thumb]:ring-black
              [&::-moz-range-thumb]:rounded-none
              [&::-moz-range-thumb]:hover:scale-110
            "
          />
        </div>
      </div>
    );
  };

  return (
    <div className="h-screen w-full bg-black text-white font-sans overflow-hidden flex">
      
      {/* LEFT COLUMN: VIEWPORT */}
      <div className="flex-1 relative bg-black flex flex-col min-w-0">
        {/* Header Overlay */}
        <div className="absolute top-0 left-0 p-8 z-10 pointer-events-none mix-blend-difference">
           <h1 className="text-5xl font-black tracking-tighter text-white leading-none">GRAINSCOPE</h1>
           <p className="text-[10px] font-mono text-zinc-400 mt-2 tracking-[0.2em] uppercase">Pro Signal Processing Pipeline v2.0</p>
        </div>

        {/* Stats Overlay (Bottom Left) */}
        <div className="absolute bottom-0 left-0 p-8 z-10 pointer-events-none mix-blend-difference">
          <div className="flex flex-col space-y-1">
             <div className="flex space-x-6 text-[10px] font-bold tracking-widest uppercase text-white">
                <span>{fps} FPS</span>
                <span>{(resolutionScale * 100).toFixed(0)}% SCALE</span>
                <span>{displayZoom < 1.0 ? 'FIT' : `${displayZoom}X ZOOM`}</span>
             </div>
             <div className="h-[1px] w-full bg-white/20"></div>
             <div className="text-[9px] font-mono text-zinc-500 uppercase tracking-wider">
               Internal Resolution: {(1920 * resolutionScale).toFixed(0)}x{(1080 * resolutionScale).toFixed(0)}
             </div>
          </div>
        </div>
        
        {/* Canvas Area */}
        <div className="flex-1 flex items-center justify-center overflow-hidden bg-zinc-950">
             <NoiseVisualizer 
                spatialSigma={spatialSigma}
                rangeSigma={rangeSigma}
                anisotropy={anisotropy}
                chromaBoost={chromaBoost}
                temporalStrength={temporalStrength}
                sharpness={sharpness}
                lensDistortion={lensDistortion}
                vignetteStrength={vignetteStrength}
                aaStrength={aaStrength}
                temporalGamma={temporalGamma}
                resolutionScale={resolutionScale}
                displayZoom={displayZoom}
                mode={mode}
                onStatsUpdate={setFps}
                isPaused={isPaused}
              />
        </div>
      </div>

      {/* RIGHT COLUMN: CONTROLS */}
      <div className="w-[380px] bg-black border-l border-zinc-900 h-full flex flex-col shrink-0 z-20 shadow-2xl">
         
         <div className="p-8 border-b border-zinc-900">
            <div className="flex items-center justify-between mb-4">
               <span className="text-[10px] font-black text-white uppercase tracking-widest">Signal Path</span>
            </div>
            <div className="flex bg-black border border-zinc-800 p-0.5">
                {[
                  { id: 0, label: 'RAW' },
                  { id: 2, label: 'SPLIT' },
                  { id: 3, label: 'PROCESSED' }
                ].map(m => (
                  <button
                    key={m.id}
                    onClick={() => setMode(m.id)}
                    className={`flex-1 py-3 text-[10px] font-black tracking-widest transition-all ${
                      mode === m.id ? 'bg-white text-black' : 'text-zinc-500 hover:text-zinc-300 hover:bg-zinc-900'
                    }`}
                  >
                    {m.label}
                  </button>
                ))}
            </div>
         </div>

         <div className="flex-1 overflow-y-auto px-8 py-4 space-y-12 custom-scrollbar">
            
            {/* SIGNAL PROCESSING */}
            <section>
              <h3 className="text-[10px] font-black text-zinc-400 mb-6 uppercase tracking-widest flex items-center">
                <span className="w-2 h-2 bg-white mr-2"></span>
                Denoising Engine
              </h3>
              <div className="space-y-2">
                <SliderGroup label="Luma Filter Strength" value={spatialSigma} min={0.5} max={5.0} step={0.1} onChange={setSpatialSigma} />
                <SliderGroup label="Chroma Suppression" value={chromaBoost} min={1.0} max={5.0} step={0.1} onChange={setChromaBoost} unit="x" />
                <SliderGroup label="Edge Sensitivity" value={rangeSigma} min={0.01} max={0.5} step={0.01} onChange={setRangeSigma} />
                <SliderGroup label="Tensor Flow" value={anisotropy} min={0.0} max={1.0} step={0.05} onChange={setAnisotropy} />
              </div>
            </section>

            {/* TEMPORAL */}
            <section>
              <h3 className="text-[10px] font-black text-zinc-400 mb-6 uppercase tracking-widest flex items-center">
                 <span className="w-2 h-2 bg-white mr-2"></span>
                 Temporal Stability
              </h3>
              <div className="space-y-2">
                <SliderGroup label="Frame Accumulation" value={temporalStrength} min={0.0} max={0.99} step={0.01} onChange={setTemporalStrength} unit="%" />
                <SliderGroup label="Variance Gamma" value={temporalGamma} min={0.1} max={3.0} step={0.05} onChange={setTemporalGamma} />
                <SliderGroup label="Temporal AA" value={aaStrength} min={0.0} max={2.0} step={0.1} onChange={setAaStrength} />
              </div>
            </section>

             {/* OPTICAL & TEXTURE */}
             <section>
              <h3 className="text-[10px] font-black text-zinc-400 mb-6 uppercase tracking-widest flex items-center">
                 <span className="w-2 h-2 bg-white mr-2"></span>
                 Optics & Detail
              </h3>
              <div className="space-y-2">
                <SliderGroup label="Lens Aberration" value={lensDistortion} min={0.0} max={5.0} step={0.1} onChange={setLensDistortion} />
                <SliderGroup label="Vignette Falloff" value={vignetteStrength} min={0.0} max={1.0} step={0.05} onChange={setVignetteStrength} />
                <SliderGroup label="Structure Sharpen" value={sharpness} min={0.0} max={2.0} step={0.05} onChange={setSharpness} />
              </div>
            </section>

            {/* RESOLUTION */}
            <section className="pb-8">
              <h3 className="text-[10px] font-black text-zinc-400 mb-6 uppercase tracking-widest flex items-center">
                 <span className="w-2 h-2 bg-white mr-2"></span>
                 Output Geometry
              </h3>
              <div className="space-y-8">
                 <div>
                    <div className="flex justify-between items-baseline mb-3">
                       <label className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">Internal Upscale</label>
                       <span className="text-[10px] font-mono text-zinc-400">{resolutionScale}x</span>
                    </div>
                    <div className="grid grid-cols-4 gap-1">
                      {[1.0, 1.25, 1.5, 2.0].map(s => (
                        <button 
                          key={s} 
                          onClick={() => setResolutionScale(s)}
                          className={`py-2 text-[10px] font-black border transition-all ${resolutionScale === s ? 'border-white bg-white text-black' : 'border-zinc-800 text-zinc-600 hover:border-zinc-600 hover:text-white'}`}
                        >
                          {s}x
                        </button>
                      ))}
                    </div>
                 </div>

                 <div>
                    <div className="flex justify-between items-baseline mb-3">
                       <label className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">Viewport Zoom</label>
                       <span className="text-[10px] font-mono text-zinc-400">{displayZoom}x</span>
                    </div>
                    <div className="grid grid-cols-4 gap-1">
                      {[0.5, 1.0, 2.0, 4.0].map(s => (
                        <button 
                          key={s} 
                          onClick={() => setDisplayZoom(s)}
                          className={`py-2 text-[10px] font-black border transition-all ${displayZoom === s ? 'border-white bg-white text-black' : 'border-zinc-800 text-zinc-600 hover:border-zinc-600 hover:text-white'}`}
                        >
                          {s}x
                        </button>
                      ))}
                    </div>
                 </div>
              </div>
            </section>
         </div>

         {/* FOOTER */}
         <div className="p-8 border-t border-zinc-900 bg-black">
             <button 
               onClick={() => setIsPaused(!isPaused)}
               className={`w-full py-4 text-[10px] font-black tracking-[0.2em] uppercase transition-all border ${
                 isPaused ? 'bg-zinc-800 border-zinc-700 text-white animate-pulse' : 'bg-white border-white text-black hover:bg-zinc-200'
               }`}
             >
               {isPaused ? 'System Halted' : 'Freeze System'}
             </button>
         </div>
      </div>
    </div>
  );
};

export default App;