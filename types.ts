export enum AnalysisStatus {
  IDLE = 'IDLE',
  ANALYZING = 'ANALYZING',
  COMPLETED = 'COMPLETED',
  ERROR = 'ERROR'
}

export interface WebGPUStats {
  fps: number;
  frameTime: number;
  resolution: string;
}

export interface AnalysisResult {
  markdown: string;
}
