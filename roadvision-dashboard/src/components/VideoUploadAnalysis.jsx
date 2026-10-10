import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Upload, Download, Play, Pause, RotateCcw, Maximize, Minimize,
  Volume2, VolumeX, Gauge, Check, FileVideo, Sparkles,
  Cpu, Activity, Layers, ShieldAlert, CheckCircle2, AlertCircle
} from 'lucide-react';

export default function VideoUploadAnalysis({ onAnalysisComplete }) {
  const [videoSrc, setVideoSrc] = useState(null);
  const [videoName, setVideoName] = useState('');
  const [fileSize, setFileSize] = useState('');
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1.0);
  const [showSpeedMenu, setShowSpeedMenu] = useState(false);
  const [volume, setVolume] = useState(1.0);
  const [isMuted, setIsMuted] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // AI Pipeline Processing State
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [processingStatus, setProcessingStatus] = useState('Preparing survey footage for analysis...');
  const [analysisError, setAnalysisError] = useState(null);
  const [hasAnalyzed, setHasAnalyzed] = useState(false);

  const videoRef = useRef(null);
  const containerRef = useRef(null);

  // Monitor fullscreen state changes
  useEffect(() => {
    const handleFullscreenChange = () => {
      const isFs = Boolean(
        document.fullscreenElement ||
        document.webkitFullscreenElement ||
        document.mozFullScreenElement ||
        document.msFullscreenElement
      );
      setIsFullscreen(isFs);
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
      document.removeEventListener('webkitfullscreenchange', handleFullscreenChange);
    };
  }, []);

  // Keyboard Shortcuts: Space = Play/Pause, F = Fullscreen, Arrows = Seek
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.target.tagName === 'INPUT') return;

      if (e.code === 'Space') {
        e.preventDefault();
        togglePlay();
      } else if (e.code === 'KeyF') {
        e.preventDefault();
        toggleFullscreen();
      } else if (e.code === 'ArrowRight') {
        e.preventDefault();
        seekRelative(5);
      } else if (e.code === 'ArrowLeft') {
        e.preventDefault();
        seekRelative(-5);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isPlaying, duration]);

  // Meaningful processing phase progression while awaiting backend response
  useEffect(() => {
    if (!isAnalyzing) return;

    setProcessingStatus('Preparing survey footage for perception pipeline...');

    const timer1 = setTimeout(() => {
      setProcessingStatus('Analyzing road frames with YOLOv8 & MiDaS monocular depth...');
    }, 2400);

    const timer2 = setTimeout(() => {
      setProcessingStatus('Associating defect trajectories and calibrating metric severity...');
    }, 6000);

    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
    };
  }, [isAnalyzing]);

  // Handle Video Upload: Send to real backend → run model → display analyzed video
  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const sizeFormatted = file.size
      ? (file.size / (1024 * 1024)).toFixed(1) + ' MB'
      : '';

    setVideoName(file.name);
    setFileSize(sizeFormatted);
    setVideoSrc(null);
    setIsPlaying(false);
    setIsAnalyzing(true);
    setAnalysisError(null);
    setHasAnalyzed(false);
    setProcessingStatus('Preparing survey footage for perception pipeline...');

    const formData = new FormData();
    formData.append('video', file);

    try {
      const res = await fetch('http://localhost:5000/api/detect', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();

      if (data.success && data.videoUrl) {
        setProcessingStatus('Analysis complete. Loading annotated survey...');

        setTimeout(() => {
          const finalUrl = data.videoUrl.startsWith('http') ? data.videoUrl : `http://localhost:5000${data.videoUrl}`;
          setVideoSrc(finalUrl);
          setIsAnalyzing(false);
          setHasAnalyzed(true);

          if (onAnalysisComplete) {
            onAnalysisComplete({
              stats: data.stats,
              auditRecords: data.auditRecords,
              videoUrl: data.videoUrl,
              filename: data.filename
            });
          }

          setTimeout(() => {
            if (videoRef.current) {
              videoRef.current.currentTime = 0;
              videoRef.current.play().then(() => setIsPlaying(true)).catch(() => {});
            }
          }, 300);
        }, 500);
      } else {
        setIsAnalyzing(false);
        setAnalysisError(data.error || 'Server returned an error during video processing.');
      }
    } catch (err) {
      console.error('Backend connection error:', err);
      setIsAnalyzing(false);
      setAnalysisError(
        'Could not connect to the backend server. Make sure server.py is running on port 5000.'
      );
    }
  };

  const togglePlay = () => {
    if (!videoRef.current) return;
    if (isPlaying) {
      videoRef.current.pause();
      setIsPlaying(false);
    } else {
      videoRef.current.play().then(() => setIsPlaying(true)).catch(() => {});
    }
  };

  const handleRestart = () => {
    if (!videoRef.current) return;
    videoRef.current.currentTime = 0;
    videoRef.current.play().then(() => setIsPlaying(true)).catch(() => {});
    setCurrentTime(0);
  };

  const seekRelative = (seconds) => {
    if (!videoRef.current) return;
    const nextTime = Math.max(0, Math.min(duration || 100, videoRef.current.currentTime + seconds));
    videoRef.current.currentTime = nextTime;
    setCurrentTime(nextTime);
  };

  const handleScrubberChange = (e) => {
    const time = parseFloat(e.target.value);
    setCurrentTime(time);
    if (videoRef.current) {
      videoRef.current.currentTime = time;
    }
  };

  const changeSpeed = (rate) => {
    setPlaybackRate(rate);
    setShowSpeedMenu(false);
    if (videoRef.current) {
      videoRef.current.playbackRate = rate;
    }
  };

  const handleVolumeChange = (e) => {
    const val = parseFloat(e.target.value);
    setVolume(val);
    setIsMuted(val === 0);
    if (videoRef.current) {
      videoRef.current.volume = val;
      videoRef.current.muted = (val === 0);
    }
  };

  const toggleMute = () => {
    if (!videoRef.current) return;
    if (isMuted) {
      videoRef.current.muted = false;
      setIsMuted(false);
      if (volume === 0) {
        setVolume(1.0);
        videoRef.current.volume = 1.0;
      }
    } else {
      videoRef.current.muted = true;
      setIsMuted(true);
    }
  };

  const toggleFullscreen = () => {
    const el = containerRef.current;
    if (!el) return;

    if (!document.fullscreenElement && !document.webkitFullscreenElement) {
      if (el.requestFullscreen) {
        el.requestFullscreen();
      } else if (el.webkitRequestFullscreen) {
        el.webkitRequestFullscreen();
      }
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen();
      } else if (document.webkitExitFullscreen) {
        document.webkitExitFullscreen();
      }
    }
  };

  const formatTime = (secs) => {
    if (!secs || isNaN(secs)) return '0:00';
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m}:${s < 10 ? '0' : ''}${s}`;
  };

  // Download the processed, annotated video (actual backend output URL)
  const handleDownloadVideo = async () => {
    if (!videoSrc) return;
    try {
      const response = await fetch(videoSrc);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const blob = await response.blob();
      const blobUrl = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = blobUrl;
      link.download = 'roadvision_annotated_survey.mp4';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      setTimeout(() => URL.revokeObjectURL(blobUrl), 1000);
    } catch (err) {
      console.warn('Direct blob download failed, falling back to server attachment endpoint:', err);
      const sep = videoSrc.includes('?') ? '&' : '?';
      const fallbackUrl = `${videoSrc}${sep}download=1&name=roadvision_annotated_survey.mp4`;
      const link = document.createElement('a');
      link.href = fallbackUrl;
      link.download = 'roadvision_annotated_survey.mp4';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  return (
    <div className="video-studio-viewport">

      {/* ── 1. COMPACT FILE-INPUT UPLOAD SECTION (When no video is loaded) ── */}
      {!videoSrc && !isAnalyzing && (
        <div className="compact-upload-panel">
          <div className="compact-upload-header">
            <div className="compact-upload-icon-badge">
              <Upload size={20} />
            </div>
            <div className="compact-upload-text">
              <h2 className="compact-upload-title">Road Survey Video Input</h2>
              <p className="compact-upload-desc">
                Select dashcam or inspection video footage for automated pavement defect detection and metric depth profiling.
              </p>
            </div>
          </div>

          <div className="compact-upload-action-row">
            <label className="btn-compact-choose" title="Choose road survey video file from your computer">
              <Upload size={15} />
              <span>Choose road survey video</span>
              <input type="file" accept="video/*" onChange={handleFileUpload} hidden />
            </label>
            <span className="compact-format-tag">Supported formats: MP4, AVI, MOV</span>
            {videoName && (
              <span className="compact-file-selected">
                <FileVideo size={14} />
                <span>{videoName}</span>
                {fileSize && <span className="file-size-badge">({fileSize})</span>}
              </span>
            )}
          </div>

          {analysisError && (
            <div className="compact-error-banner" role="alert">
              <AlertCircle size={16} className="error-icon" />
              <span>{analysisError}</span>
            </div>
          )}
        </div>
      )}

      {/* ── 2. RESTRAINED PROCESSING STATE ── */}
      {isAnalyzing && (
        <div className="compact-processing-card" role="status" aria-live="polite">
          <div className="processing-card-header">
            <div className="pulse-indicator" />
            <span className="processing-tag mono">PERCEPTION PIPELINE ACTIVE</span>
            {videoName && (
              <span className="processing-file-meta">
                <FileVideo size={12} />
                <span>{videoName} {fileSize ? `(${fileSize})` : ''}</span>
              </span>
            )}
          </div>
          <h3 className="processing-title">Analyzing Road Survey Video</h3>
          <p className="processing-subtext">{processingStatus}</p>

          <div className="indeterminate-scanning-track" aria-hidden="true">
            <div className="indeterminate-scanning-bar" />
          </div>

          <div className="processing-footnote-row">
            <span className="processing-footnote">Running PyTorch YOLOv8 defect localization and MiDaS monocular metric depth.</span>
          </div>
        </div>
      )}

      {/* ── 3. PROCESSED VIDEO PLAYER (When output is available) ── */}
      {videoSrc && (
        <div className="processed-video-section">
          {/* Top Options Bar */}
          <div className="video-options-bar">
            <div className="options-left">
              <label className="btn-upload-reselect" title="Upload a different road survey video">
                <Upload size={14} />
                <span>Choose another video</span>
                <input type="file" accept="video/*" onChange={handleFileUpload} hidden />
              </label>
              {videoName && (
                <div className="video-filename-tag">
                  <FileVideo size={14} />
                  <span>{videoName}</span>
                  {fileSize && <span className="file-size-meta">({fileSize})</span>}
                </div>
              )}
            </div>

            <div className="options-right">
              <button
                className="btn-download-prominent"
                onClick={handleDownloadVideo}
                title="Download processed and annotated MP4 video (roadvision_annotated_survey.mp4)"
                aria-label="Download Annotated Video"
              >
                <Download size={14} />
                <span>Download Annotated Video</span>
              </button>
            </div>
          </div>

          {/* Dedicated Player Container */}
          <div
            className={`video-player-container ${isFullscreen ? 'is-fullscreen' : ''}`}
            ref={containerRef}
          >
            <div className="video-screen-wrap" onClick={togglePlay} onDoubleClick={toggleFullscreen}>
              <video
                ref={videoRef}
                src={videoSrc}
                className="main-display-video"
                onLoadedMetadata={() => {
                  if (videoRef.current) {
                    setDuration(videoRef.current.duration);
                    videoRef.current.playbackRate = playbackRate;
                  }
                }}
                onTimeUpdate={() => {
                  if (videoRef.current) {
                    setCurrentTime(videoRef.current.currentTime);
                  }
                }}
                onPlay={() => setIsPlaying(true)}
                onPause={() => setIsPlaying(false)}
                onEnded={() => setIsPlaying(false)}
                playsInline
              />

              {!isPlaying && (
                <div className="center-play-overlay">
                  <div className="play-glass-disc" aria-label="Play video">
                    <Play size={24} fill="currentColor" />
                  </div>
                </div>
              )}
            </div>

            {/* Custom Video Controls Bar */}
            <div className="player-controls-strip">
              <button
                className="ctrl-btn-action play-btn"
                onClick={togglePlay}
                title={isPlaying ? 'Pause (Space)' : 'Play (Space)'}
                aria-label={isPlaying ? 'Pause' : 'Play'}
              >
                {isPlaying ? <Pause size={16} fill="currentColor" /> : <Play size={16} fill="currentColor" />}
              </button>
              <button
                className="ctrl-btn-action"
                onClick={handleRestart}
                title="Restart Video (0:00)"
                aria-label="Restart Video"
              >
                <RotateCcw size={15} />
              </button>
              <div className="time-code mono">
                <span className="time-curr">{formatTime(currentTime)}</span>
                <span className="time-sep">/</span>
                <span className="time-total">{formatTime(duration)}</span>
              </div>
              <div className="scrubber-bar-container">
                <input
                  type="range"
                  min="0"
                  max={duration || 100}
                  step="0.05"
                  value={currentTime}
                  onChange={handleScrubberChange}
                  className="player-timeline-slider"
                  aria-label="Seek video position"
                />
              </div>
              <div className="speed-dropdown-wrapper">
                <button
                  className={`ctrl-btn-action speed-btn ${playbackRate !== 1.0 ? 'active' : ''}`}
                  onClick={() => setShowSpeedMenu(!showSpeedMenu)}
                  title="Playback Speed"
                  aria-label={`Playback Speed ${playbackRate}x`}
                >
                  <Gauge size={14} />
                  <span>{playbackRate}x</span>
                </button>
                {showSpeedMenu && (
                  <div className="speed-options-card">
                    <div className="speed-menu-header">Playback Speed</div>
                    {[0.5, 0.75, 1.0, 1.25, 1.5, 2.0].map((rate) => (
                      <button
                        key={rate}
                        className={`speed-option-item ${playbackRate === rate ? 'selected' : ''}`}
                        onClick={() => changeSpeed(rate)}
                      >
                        <span>{rate}x {rate === 1.0 ? '(Normal)' : ''}</span>
                        {playbackRate === rate && <Check size={13} />}
                      </button>
                    ))}
                  </div>
                )}
              </div>
              <div className="volume-control-cluster">
                <button
                  className="ctrl-btn-action volume-btn"
                  onClick={toggleMute}
                  title={isMuted ? 'Unmute' : 'Mute'}
                  aria-label={isMuted ? 'Unmute' : 'Mute'}
                >
                  {isMuted || volume === 0 ? <VolumeX size={16} /> : <Volume2 size={16} />}
                </button>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={isMuted ? 0 : volume}
                  onChange={handleVolumeChange}
                  className="player-volume-slider"
                  title="Volume"
                  aria-label="Volume slider"
                />
              </div>
              <button
                className="ctrl-btn-action download-ctrl-btn"
                onClick={handleDownloadVideo}
                title="Download Annotated Video (roadvision_annotated_survey.mp4)"
                aria-label="Download Annotated Video"
              >
                <Download size={15} />
              </button>
              <button
                className="ctrl-btn-action fullscreen-btn"
                onClick={toggleFullscreen}
                title={isFullscreen ? 'Exit Fullscreen (Esc or F)' : 'Fullscreen (F)'}
                aria-label={isFullscreen ? 'Exit Fullscreen' : 'Enter Fullscreen'}
              >
                {isFullscreen ? <Minimize size={16} /> : <Maximize size={16} />}
              </button>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
