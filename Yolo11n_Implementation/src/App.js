import { useEffect, useRef, useState } from 'react';
import './App.css';

const API_BASE = process.env.REACT_APP_API_BASE || 'http://127.0.0.1:8000';
const EMPTY_LANE_DISTRIBUTION = { N: 0, E: 0, S: 0, W: 0 };
const EMPTY_METRICS = {
  totalVehicles: 0,
  ambulanceCount: 0,
  emergencyVehicleCount: 0,
  haltingVehicleCount: 0,
  movingVehicleCount: 0,
  averageWaitTimeSec: 0,
  averageHaltingWaitTimeSec: 0,
  laneDistribution: EMPTY_LANE_DISTRIBUTION,
  laneStats: {},
  currentGreenLane: null,
  signalCycleTimeSec: 0,
};

function formatSeconds(value) {
  return `${Number(value || 0).toFixed(2)}s`;
}

function buildPolylinePoints(values, width, height) {
  if (!values.length) {
    return '';
  }

  const maxValue = Math.max(...values, 1);
  const step = values.length === 1 ? width : width / (values.length - 1);

  return values
    .map((value, index) => {
      const x = index * step;
      const y = height - (value / maxValue) * height;
      return `${x},${y.toFixed(2)}`;
    })
    .join(' ');
}

function MetricSparkline({ values, stroke, fill }) {
  const width = 320;
  const height = 88;
  const points = buildPolylinePoints(values, width, height);

  if (!points) {
    return (
      <svg className="sparkline" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
        <line x1="0" y1={height - 2} x2={width} y2={height - 2} className="sparkline-empty" />
      </svg>
    );
  }

  const fillPoints = `${points} ${width},${height} 0,${height}`;

  return (
    <svg className="sparkline" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
      <polygon points={fillPoints} fill={fill} />
      <polyline points={points} fill="none" stroke={stroke} strokeWidth="4" strokeLinecap="round" />
    </svg>
  );
}

function MetricCard({ label, value, accent, caption }) {
  return (
    <article className="metric-card">
      <span className="metric-label">{label}</span>
      <strong className="metric-value" style={{ color: accent }}>
        {value}
      </strong>
      <span className="metric-caption">{caption}</span>
    </article>
  );
}

function LaneCard({ lane, stats, totalVehicles, active }) {
  const laneVehicles = stats?.vehicles || 0;
  const laneShare = totalVehicles ? Math.round((laneVehicles / totalVehicles) * 100) : 0;

  return (
    <article className={`lane-card ${active ? 'lane-card-active' : ''}`}>
      <header className="lane-card-header">
        <span className="lane-token">{lane}</span>
        <span className="lane-share">{laneShare}% share</span>
      </header>
      <strong className="lane-count">{laneVehicles}</strong>
      <div className="lane-meta">
        <span>Halting {stats?.halting || 0}</span>
        <span>Ambulances {stats?.ambulances || 0}</span>
      </div>
      <div className="lane-progress">
        <div className="lane-progress-bar" style={{ width: `${laneShare}%` }} />
      </div>
      <span className="lane-wait">Avg wait {formatSeconds(stats?.averageWaitTimeSec || 0)}</span>
    </article>
  );
}

function BoundingBoxOverlay({ vehicleDetails, video }) {
  const frameWidth = Number(video?.width || 0);
  const frameHeight = Number(video?.height || 0);

  if (!frameWidth || !frameHeight || !vehicleDetails.length) {
    return null;
  }

  return (
    <div className="video-overlay" aria-hidden="true">
      {vehicleDetails.map((vehicle) => {
        if (!Array.isArray(vehicle.bbox) || vehicle.bbox.length !== 4) {
          return null;
        }

        const [x1, y1, x2, y2] = vehicle.bbox;
        const width = Math.max(0, x2 - x1);
        const height = Math.max(0, y2 - y1);
        const toneClass = vehicle.isEmergency ? 'bbox-emergency' : vehicle.isHalting ? 'bbox-halting' : '';

        return (
          <div
            key={vehicle.id}
            className={`bbox ${toneClass}`.trim()}
            style={{
              left: `${(x1 / frameWidth) * 100}%`,
              top: `${(y1 / frameHeight) * 100}%`,
              width: `${(width / frameWidth) * 100}%`,
              height: `${(height / frameHeight) * 100}%`,
            }}
          >
            <span className="bbox-label">
              {vehicle.className} #{vehicle.id}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function parseVideoPaths(rawValue) {
  return String(rawValue || '')
    .split(/[\r\n,]+/)
    .map((line) => line.trim().replace(/^['"]|['"]$/g, ''))
    .filter(Boolean);
}

function createVideoEntry(videoPath, sequenceIndex, sequenceTotal) {
  return {
    sequenceIndex,
    sequenceTotal,
    videoPath,
    video: null,
    history: [],
    currentFrame: null,
    finalFrame: null,
    emittedCount: 0,
    status: 'pending',
  };
}

function updateSequenceEntry(videos, sequenceIndex, updater) {
  return videos.map((video) =>
    video.sequenceIndex === sequenceIndex ? updater(video) : video
  );
}

function buildOverallSummary(videos, queuedCount = 0) {
  const inspectedVideos = videos.filter((video) => video.finalFrame || video.currentFrame);
  const completedVideos = videos.filter((video) => video.status === 'complete').length;
  const busiestVideo = inspectedVideos.reduce((currentBest, video) => {
    const currentVehicles =
      video.finalFrame?.metrics?.totalVehicles || video.currentFrame?.metrics?.totalVehicles || 0;
    const bestVehicles =
      currentBest?.finalFrame?.metrics?.totalVehicles ||
      currentBest?.currentFrame?.metrics?.totalVehicles ||
      0;
    return currentVehicles > bestVehicles ? video : currentBest;
  }, null);

  return {
    totalVideos: videos.length || queuedCount,
    completedVideos,
    totalSnapshots: videos.reduce((total, video) => total + (video.emittedCount || 0), 0),
    averageVehiclesPerVideo: inspectedVideos.length
      ? Number(
          (
            inspectedVideos.reduce(
              (total, video) =>
                total +
                (video.finalFrame?.metrics?.totalVehicles ||
                  video.currentFrame?.metrics?.totalVehicles ||
                  0),
              0
            ) / inspectedVideos.length
          ).toFixed(2)
        )
      : 0,
    averageWaitTimeSec: inspectedVideos.length
      ? Number(
          (
            inspectedVideos.reduce(
              (total, video) =>
                total +
                (video.finalFrame?.metrics?.averageWaitTimeSec ||
                  video.currentFrame?.metrics?.averageWaitTimeSec ||
                  0),
              0
            ) / inspectedVideos.length
          ).toFixed(2)
        )
      : 0,
    busiestVideoPath: busiestVideo?.videoPath || null,
    busiestVideoVehicles:
      busiestVideo?.finalFrame?.metrics?.totalVehicles ||
      busiestVideo?.currentFrame?.metrics?.totalVehicles ||
      0,
  };
}

function formatPathStatus(status) {
  if (status === 'streaming') {
    return 'Running';
  }
  if (status === 'complete') {
    return 'Complete';
  }
  return 'Queued';
}

function App() {
  const streamRef = useRef(null);
  const previewVideoRef = useRef(null);
  const [videos, setVideos] = useState([]);
  const [selectedVideoIndex, setSelectedVideoIndex] = useState(0);
  const [overallSummary, setOverallSummary] = useState(null);
  const [error, setError] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [status, setStatus] = useState({
    tone: 'idle',
    title: 'Backend required',
    message: 'Start the Python server to stream YOLO analysis into the dashboard.',
  });
  const [form, setForm] = useState({
    videoPaths: '',
    processFps: '10',
    emitIntervalSec: '1',
    realtime: true,
  });

  useEffect(() => {
    let isCancelled = false;

    async function loadConfig() {
      try {
        const response = await fetch(`${API_BASE}/api/config`);
        if (!response.ok) {
          throw new Error(`Backend responded with ${response.status}`);
        }

        const payload = await response.json();
        if (isCancelled) {
          return;
        }

        setForm({
          videoPaths: payload.defaultVideoPath || '',
          processFps: String(payload.defaults?.processFps || 10),
          emitIntervalSec: String(payload.defaults?.emitIntervalSec || 1),
          realtime: Boolean(payload.defaults?.realtime),
        });
        setStatus({
          tone: 'ready',
          title: 'Backend online',
          message: 'Model bundle detected. Queue one or more video paths and start the sequence.',
        });
      } catch (loadError) {
        if (isCancelled) {
          return;
        }

        setError(loadError.message);
        setStatus({
          tone: 'error',
          title: 'Backend unavailable',
          message: 'Run the Python backend first. The React dashboard is ready to connect.',
        });
      }
    }

    loadConfig();

    return () => {
      isCancelled = true;
      if (streamRef.current) {
        streamRef.current.close();
        streamRef.current = null;
      }
    };
  }, []);

  function updateForm(field, value) {
    setForm((previous) => ({
      ...previous,
      [field]: value,
    }));
  }

  function stopStream(nextStatus) {
    if (streamRef.current) {
      streamRef.current.close();
      streamRef.current = null;
    }
    setIsStreaming(false);
    if (nextStatus) {
      setStatus(nextStatus);
    }
  }

  function startStream() {
    const videoPaths = parseVideoPaths(form.videoPaths);
    const processFps = Number(form.processFps);
    const emitIntervalSec = Number(form.emitIntervalSec);

    if (!videoPaths.length) {
      setError('At least one video path is required.');
      return;
    }
    if (!Number.isFinite(processFps) || processFps <= 0) {
      setError('Process FPS must be greater than 0.');
      return;
    }
    if (!Number.isFinite(emitIntervalSec) || emitIntervalSec <= 0) {
      setError('Emit interval must be greater than 0.');
      return;
    }

    const preparedVideos = videoPaths.map((videoPath, index) =>
      createVideoEntry(videoPath, index + 1, videoPaths.length)
    );

    setError('');
    setVideos(preparedVideos);
    setSelectedVideoIndex(0);
    setOverallSummary(null);
    stopStream();

    const params = new URLSearchParams();
    videoPaths.forEach((videoPath) => {
      params.append('videoPath', videoPath);
    });
    params.set('processFps', String(processFps));
    params.set('emitIntervalSec', String(emitIntervalSec));
    params.set('realtime', String(form.realtime));
    const source = new EventSource(`${API_BASE}/api/stream?${params.toString()}`);
    streamRef.current = source;
    setIsStreaming(true);
    setStatus({
      tone: 'streaming',
      title: 'Streaming sequence',
      message: `Queued ${videoPaths.length} video paths for sequential analysis.`,
    });

    source.onmessage = (event) => {
      if (streamRef.current !== source) {
        return;
      }

      const payload = JSON.parse(event.data);
      setError('');

      if (payload.type === 'session') {
        setSelectedVideoIndex(Math.max(0, (payload.sequenceIndex || 1) - 1));
        setVideos((previous) =>
          updateSequenceEntry(previous, payload.sequenceIndex, (video) => ({
            ...video,
            videoPath: payload.videoPath || video.videoPath,
            video: payload.video || video.video,
            status: 'streaming',
          }))
        );
        setStatus({
          tone: 'streaming',
          title: 'Sequence running',
          message: `Analyzing video ${payload.sequenceIndex} of ${payload.sequenceTotal}.`,
        });
        return;
      }

      if (payload.type === 'analysis') {
        setSelectedVideoIndex(Math.max(0, (payload.sequenceIndex || 1) - 1));
        setVideos((previous) =>
          updateSequenceEntry(previous, payload.sequenceIndex, (video) => ({
            ...video,
            videoPath: payload.videoPath || video.videoPath,
            video: payload.video || video.video,
            currentFrame: payload,
            emittedCount: payload.emittedIndex || video.emittedCount,
            history: [...video.history.slice(-23), payload],
            status: 'streaming',
          }))
        );
        return;
      }

      if (payload.type === 'videoComplete') {
        setSelectedVideoIndex(Math.max(0, (payload.sequenceIndex || 1) - 1));
        setVideos((previous) =>
          updateSequenceEntry(previous, payload.sequenceIndex, (video) => ({
            ...video,
            videoPath: payload.videoPath || video.videoPath,
            video: payload.video || video.video,
            finalFrame: payload.finalFrame || video.finalFrame,
            currentFrame: payload.finalFrame || video.currentFrame,
            emittedCount: payload.emittedCount || video.emittedCount,
            status: 'complete',
          }))
        );
        setStatus({
          tone: 'streaming',
          title:
            payload.sequenceIndex === payload.sequenceTotal
              ? 'Final video complete'
              : 'Video complete',
          message:
            payload.sequenceIndex === payload.sequenceTotal
              ? 'Final path finished. Waiting for overall summary.'
              : `Finished video ${payload.sequenceIndex} of ${payload.sequenceTotal}.`,
        });
        return;
      }

      if (payload.type === 'complete') {
        if (Array.isArray(payload.videoSummaries)) {
          setVideos((previous) =>
            previous.map((video) => {
              const summary = payload.videoSummaries.find(
                (entry) => entry.sequenceIndex === video.sequenceIndex
              );
              if (!summary) {
                return video;
              }
              return {
                ...video,
                emittedCount: summary.emittedCount || video.emittedCount,
                video: summary.video || video.video,
              };
            })
          );
        }
        setOverallSummary(payload.overallSummary || null);
        stopStream({
          tone: 'ready',
          title: 'Sequence complete',
          message: `Processed ${payload.overallSummary?.completedVideos || 0} of ${
            payload.overallSummary?.totalVideos || payload.sequenceTotal || 0
          } videos across ${payload.emittedCount || 0} snapshots.`,
        });
        return;
      }

      if (payload.type === 'error') {
        setError(payload.message || 'The backend reported an error.');
        stopStream({
          tone: 'error',
          title: 'Stream failed',
          message: payload.message || 'The backend reported an error.',
        });
      }
    };

    source.onerror = () => {
      if (streamRef.current !== source) {
        return;
      }

      setError('The SSE connection closed unexpectedly.');
      stopStream({
        tone: 'error',
        title: 'Connection dropped',
        message: 'The stream disconnected before completion.',
      });
    };
  }

  const queuedVideoPaths = parseVideoPaths(form.videoPaths);
  const queue = videos.length
    ? videos
    : queuedVideoPaths.map((videoPath, index) =>
        createVideoEntry(videoPath, index + 1, queuedVideoPaths.length)
      );
  const activeVideoIndex = Math.min(selectedVideoIndex, Math.max(queue.length - 1, 0));
  const selectedVideo = queue[activeVideoIndex] || null;
  const displayedFrame = selectedVideo?.finalFrame || selectedVideo?.currentFrame || null;
  const metrics = displayedFrame?.metrics || EMPTY_METRICS;
  const laneStats = metrics.laneStats || {};
  const totalVehicles = metrics.totalVehicles || 0;
  const history = selectedVideo?.history || [];
  const recentVehicleCounts = history.map((entry) => entry.metrics.totalVehicles || 0);
  const recentWaitTimes = history.map((entry) => entry.metrics.averageWaitTimeSec || 0);
  const recentEmergencyCounts = history.map((entry) => entry.metrics.emergencyVehicleCount || 0);
  const latestVehicles = displayedFrame?.vehicleDetails || [];
  const recentFrames = [...history].reverse();
  const overall = overallSummary || buildOverallSummary(queue, queuedVideoPaths.length);
  const videoUrl = selectedVideo?.videoPath
    ? `${API_BASE}/api/video?path=${encodeURIComponent(selectedVideo.videoPath)}`
    : '';

  useEffect(() => {
    const videoElement = previewVideoRef.current;
    if (!videoElement || !selectedVideo?.videoPath) {
      return;
    }

    if (displayedFrame?.timestampSec != null) {
      const drift = Math.abs((videoElement.currentTime || 0) - displayedFrame.timestampSec);
      if (drift > 0.35) {
        try {
          videoElement.currentTime = displayedFrame.timestampSec;
        } catch (error) {
          void error;
        }
      }
    }

    if (isStreaming && form.realtime && selectedVideo?.status === 'streaming') {
      const playPromise = videoElement.play();
      if (playPromise && typeof playPromise.catch === 'function') {
        playPromise.catch(() => {});
      }
      return;
    }

    if (!isStreaming || selectedVideo?.status !== 'streaming' || !form.realtime) {
      videoElement.pause();
    }
  }, [
    displayedFrame?.timestampSec,
    form.realtime,
    isStreaming,
    selectedVideo?.status,
    selectedVideo?.videoPath,
  ]);

  return (
    <div className="app-shell">
      <div className="app-glow app-glow-one" />
      <div className="app-glow app-glow-two" />

      <main className="dashboard">
        <section className="hero panel">
          <div className="hero-copy">
            <span className="eyebrow">YOLO Traffic Intelligence</span>
            <h1>Interactive command dashboard for four-way traffic flow analysis.</h1>
            <p className="hero-text">
              Queues multiple local video paths, analyzes them one after another, and keeps the
              selected path in focus while the dashboard builds an overall traffic summary.
            </p>
            <div className="hero-tags">
              <span>Queued paths {queue.length}</span>
              <span>
                Active {selectedVideo ? `${selectedVideo.sequenceIndex}/${selectedVideo.sequenceTotal}` : '0/0'}
              </span>
            </div>
          </div>

          <div className={`status-card status-${status.tone}`}>
            <span className="status-kicker">{status.title}</span>
            <p>{status.message}</p>
            <div className="status-actions">
              <button className="primary-button" onClick={startStream} disabled={isStreaming}>
                Start sequence
              </button>
              <button
                className="secondary-button"
                onClick={() =>
                  stopStream({
                    tone: 'ready',
                    title: 'Sequence stopped',
                    message: 'The sequence was closed from the dashboard.',
                  })
                }
                disabled={!isStreaming}
              >
                Stop stream
              </button>
            </div>
          </div>
        </section>

        <section className="top-grid">
          <article className="panel video-panel">
            <div className="panel-heading">
              <div>
                <span className="panel-kicker">Video Source</span>
                <h2>Selected path preview</h2>
              </div>
              <span className="panel-badge">
                {displayedFrame?.timestampLabel || selectedVideo?.video?.durationSec || '00:00'}
              </span>
            </div>

            {videoUrl ? (
              <div className="video-stage">
                <video
                  key={videoUrl}
                  ref={previewVideoRef}
                  className="video-player"
                  controls
                  muted
                  playsInline
                  preload="metadata"
                  src={videoUrl}
                />
                <BoundingBoxOverlay vehicleDetails={latestVehicles} video={selectedVideo?.video || displayedFrame?.video} />
              </div>
            ) : (
              <div className="video-placeholder">Enter one or more local video paths to preview a source.</div>
            )}

            <div className="video-meta-grid">
              <div>
                <span>Selected path</span>
                <strong>{selectedVideo ? `Video ${selectedVideo.sequenceIndex}` : 'Pending'}</strong>
              </div>
              <div>
                <span>Current green lane</span>
                <strong>{metrics.currentGreenLane || 'Pending'}</strong>
              </div>
              <div>
                <span>Signal cycle</span>
                <strong>{formatSeconds(metrics.signalCycleTimeSec)}</strong>
              </div>
            </div>
          </article>

          <article className="panel control-panel">
            <div className="panel-heading">
              <div>
                <span className="panel-kicker">Controls</span>
                <h2>Sequence setup</h2>
              </div>
            </div>

            <label className="field">
              <span>Video paths</span>
              <textarea
                value={form.videoPaths}
                onChange={(event) => updateForm('videoPaths', event.target.value)}
                placeholder={
                  'C:\\Users\\seera\\Downloads\\traffic-01.mp4, C:\\Users\\seera\\Downloads\\traffic-02.mp4'
                }
              />
            </label>
            <p className="field-help">
              Enter local paths separated by commas. New lines also work. The backend processes them in sequence.
            </p>

            <div className="field-row">
              <label className="field">
                <span>Process FPS</span>
                <input
                  type="number"
                  min="1"
                  step="1"
                  value={form.processFps}
                  onChange={(event) => updateForm('processFps', event.target.value)}
                />
              </label>

              <label className="field">
                <span>Emit interval (sec)</span>
                <input
                  type="number"
                  min="1"
                  step="1"
                  value={form.emitIntervalSec}
                  onChange={(event) => updateForm('emitIntervalSec', event.target.value)}
                />
              </label>
            </div>

            <label className="toggle-row">
              <span>Wall-clock pacing</span>
              <button
                type="button"
                className={`toggle-chip ${form.realtime ? 'toggle-chip-on' : ''}`}
                onClick={() => updateForm('realtime', !form.realtime)}
              >
                {form.realtime ? 'Real time' : 'Fast as possible'}
              </button>
            </label>

            {error ? <p className="error-text">{error}</p> : null}
          </article>
        </section>

        <section className="sequence-grid">
          <article className="panel sequence-panel">
            <div className="panel-heading">
              <div>
                <span className="panel-kicker">Path Navigator</span>
                <h2>Switch between queued videos</h2>
              </div>
              <span className="panel-badge">{queue.length} paths</span>
            </div>

            <div className="path-nav-list">
              {queue.length ? (
                queue.map((video, index) => (
                  <button
                    key={`${video.sequenceIndex}-${video.videoPath}`}
                    type="button"
                    className={`path-nav-button ${
                      index === activeVideoIndex ? 'path-nav-button-active' : ''
                    }`}
                    onClick={() => setSelectedVideoIndex(index)}
                  >
                    <div className="path-nav-top">
                      <span className="path-nav-index">Video {video.sequenceIndex}</span>
                      <span className={`path-nav-state path-state-${video.status}`}>
                        {formatPathStatus(video.status)}
                      </span>
                    </div>
                    <strong className="path-nav-path" title={video.videoPath}>
                      {video.videoPath}
                    </strong>
                    <div className="path-nav-meta">
                      <span>{video.emittedCount || 0} snapshots</span>
                      <span>
                        {video.finalFrame?.timestampLabel ||
                          video.currentFrame?.timestampLabel ||
                          'Awaiting analysis'}
                      </span>
                    </div>
                  </button>
                ))
              ) : (
                <div className="feed-empty">Add local paths above to build the analysis queue.</div>
              )}
            </div>
          </article>

          <article className="panel summary-panel">
            <div className="panel-heading">
              <div>
                <span className="panel-kicker">Overall Analysis</span>
                <h2>Sequence-wide summary</h2>
              </div>
              <span className="panel-badge">{overall.completedVideos}/{overall.totalVideos} complete</span>
            </div>

            <div className="summary-grid">
              <div>
                <span>Total videos</span>
                <strong>{overall.totalVideos}</strong>
              </div>
              <div>
                <span>Total snapshots</span>
                <strong>{overall.totalSnapshots}</strong>
              </div>
              <div>
                <span>Avg vehicles / video</span>
                <strong>{overall.averageVehiclesPerVideo}</strong>
              </div>
              <div>
                <span>Avg wait</span>
                <strong>{formatSeconds(overall.averageWaitTimeSec)}</strong>
              </div>
            </div>

            <div className="summary-highlight">
              <span>Busiest path</span>
              <strong title={overall.busiestVideoPath || ''}>
                {overall.busiestVideoPath || 'No completed path yet'}
              </strong>
              <p>{overall.busiestVideoVehicles} vehicles in the busiest final frame.</p>
            </div>
          </article>
        </section>

        <section className="metric-grid">
          <MetricCard
            label="Vehicles"
            value={metrics.totalVehicles}
            accent="#ffd166"
            caption={`${metrics.movingVehicleCount} moving right now`}
          />
          <MetricCard
            label="Ambulances"
            value={metrics.ambulanceCount}
            accent="#ff6b6b"
            caption={`${metrics.emergencyVehicleCount} emergency vehicles`}
          />
          <MetricCard
            label="Average wait"
            value={formatSeconds(metrics.averageWaitTimeSec)}
            accent="#5eead4"
            caption={`Halting avg ${formatSeconds(metrics.averageHaltingWaitTimeSec)}`}
          />
          <MetricCard
            label="Halting vehicles"
            value={metrics.haltingVehicleCount}
            accent="#8ec5ff"
            caption={`Cycle ${formatSeconds(metrics.signalCycleTimeSec)}`}
          />
        </section>

        <section className="middle-grid">
          <article className="panel lane-board">
            <div className="panel-heading">
              <div>
                <span className="panel-kicker">Lane Distribution</span>
                <h2>Compass view</h2>
              </div>
              <span className="panel-badge">{totalVehicles} vehicles tracked</span>
            </div>

            <div className="compass-grid">
              <div className="compass-slot compass-north">
                <LaneCard
                  lane="N"
                  stats={laneStats.N}
                  totalVehicles={totalVehicles}
                  active={metrics.currentGreenLane === 'N'}
                />
              </div>
              <div className="compass-slot compass-west">
                <LaneCard
                  lane="W"
                  stats={laneStats.W}
                  totalVehicles={totalVehicles}
                  active={metrics.currentGreenLane === 'W'}
                />
              </div>
              <div className="compass-slot compass-center">
                <div className="signal-core">
                  <span className="signal-kicker">Signal focus</span>
                  <strong>{metrics.currentGreenLane || 'No active lane'}</strong>
                  <p>{metrics.haltingVehicleCount} vehicles are currently halting.</p>
                </div>
              </div>
              <div className="compass-slot compass-east">
                <LaneCard
                  lane="E"
                  stats={laneStats.E}
                  totalVehicles={totalVehicles}
                  active={metrics.currentGreenLane === 'E'}
                />
              </div>
              <div className="compass-slot compass-south">
                <LaneCard
                  lane="S"
                  stats={laneStats.S}
                  totalVehicles={totalVehicles}
                  active={metrics.currentGreenLane === 'S'}
                />
              </div>
            </div>
          </article>

          <article className="panel trend-panel">
            <div className="panel-heading">
              <div>
                <span className="panel-kicker">Recent Trend</span>
                <h2>Last 24 emitted snapshots</h2>
              </div>
            </div>

            <div className="trend-stack">
              <div className="trend-card">
                <div className="trend-meta">
                  <span>Vehicle count</span>
                  <strong>{recentVehicleCounts[recentVehicleCounts.length - 1] || 0}</strong>
                </div>
                <MetricSparkline values={recentVehicleCounts} stroke="#ffd166" fill="rgba(255, 209, 102, 0.16)" />
              </div>

              <div className="trend-card">
                <div className="trend-meta">
                  <span>Average wait</span>
                  <strong>{formatSeconds(recentWaitTimes[recentWaitTimes.length - 1] || 0)}</strong>
                </div>
                <MetricSparkline values={recentWaitTimes} stroke="#5eead4" fill="rgba(94, 234, 212, 0.16)" />
              </div>

              <div className="trend-card">
                <div className="trend-meta">
                  <span>Emergency traffic</span>
                  <strong>{recentEmergencyCounts[recentEmergencyCounts.length - 1] || 0}</strong>
                </div>
                <MetricSparkline values={recentEmergencyCounts} stroke="#ff6b6b" fill="rgba(255, 107, 107, 0.16)" />
              </div>
            </div>

            <div className="recent-feed">
              {recentFrames.length ? (
                recentFrames.slice(0, 6).map((frame) => (
                  <div key={frame.emittedIndex} className="feed-row">
                    <span>{frame.timestampLabel}</span>
                    <strong>{frame.metrics.totalVehicles} vehicles</strong>
                    <span>
                      N {frame.metrics.laneDistribution.N} / E {frame.metrics.laneDistribution.E} / S{' '}
                      {frame.metrics.laneDistribution.S} / W {frame.metrics.laneDistribution.W}
                    </span>
                  </div>
                ))
              ) : (
                <div className="feed-empty">Select a processed path to inspect its frame history.</div>
              )}
            </div>
          </article>
        </section>

        <section className="bottom-grid">
          <article className="panel table-panel">
            <div className="panel-heading">
              <div>
                <span className="panel-kicker">Tracked Vehicles</span>
                <h2>Current frame detail</h2>
              </div>
              <span className="panel-badge">{latestVehicles.length} rows</span>
            </div>

            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Lane</th>
                    <th>Class</th>
                    <th>Wait</th>
                    <th>Halting</th>
                    <th>Emergency</th>
                  </tr>
                </thead>
                <tbody>
                  {latestVehicles.length ? (
                    latestVehicles.slice(0, 14).map((vehicle) => (
                      <tr key={vehicle.id}>
                        <td>{vehicle.id}</td>
                        <td>{vehicle.lane}</td>
                        <td>{vehicle.className}</td>
                        <td>{formatSeconds(vehicle.waitTimeSec)}</td>
                        <td>{vehicle.isHalting ? 'Yes' : 'No'}</td>
                        <td>{vehicle.isEmergency ? 'Yes' : 'No'}</td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan="6" className="empty-row">
                        Start the sequence and select a processed path to populate the vehicle table.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </article>

          <article className="panel json-panel">
            <div className="panel-heading">
              <div>
                <span className="panel-kicker">Per-second JSON</span>
                <h2>Latest raw vehicle payload</h2>
              </div>
              <span className="panel-badge">{displayedFrame?.timestampLabel || 'Idle'}</span>
            </div>

            <pre className="json-viewer">{JSON.stringify(displayedFrame?.vehicles || [], null, 2)}</pre>
          </article>
        </section>
      </main>
    </div>
  );
}

export default App;
