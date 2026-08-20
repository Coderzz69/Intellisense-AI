const appState = {
  manifest: null,
  currentSecond: 0,
  pollTimer: null,
  videos: [],
};

const elements = {
  uploadForm: document.getElementById("upload-form"),
  jobStatus: document.getElementById("job-status"),
  globalSignal: document.getElementById("global-signal"),
  globalMeta: document.getElementById("global-meta"),
  junctionLabel: document.getElementById("junction-label"),
  timelineSlider: document.getElementById("timeline-slider"),
  secondLabel: document.getElementById("second-label"),
  qValues: document.getElementById("q-values"),
  laneMetrics: document.getElementById("lane-metrics"),
  feedGrid: document.getElementById("feed-grid"),
  playAll: document.getElementById("play-all"),
  pauseAll: document.getElementById("pause-all"),
};

function getEntry(second) {
  if (!appState.manifest || !appState.manifest.timeline.length) {
    return null;
  }
  return appState.manifest.timeline.find((entry) => entry.second === second)
    || appState.manifest.timeline[appState.manifest.timeline.length - 1];
}

function renderBoard(second) {
  const entry = getEntry(second);
  if (!entry) {
    return;
  }

  appState.currentSecond = second;
  elements.globalSignal.textContent = entry.globalSignal;
  elements.globalMeta.textContent = `Source: ${entry.decisionSource}`;
  elements.junctionLabel.textContent = entry.globalSignal;
  elements.secondLabel.textContent = `Second ${entry.second}`;
  elements.qValues.textContent = `q = [${entry.qValues.join(", ")}]`;
  elements.timelineSlider.value = String(entry.second);

  document.querySelectorAll(".road").forEach((node) => {
    const road = node.dataset.road;
    const signal = entry.lights[road] || "red";
    node.classList.remove("is-green", "is-red", "is-yellow");
    node.classList.add(`is-${signal}`);
  });

  elements.laneMetrics.innerHTML = "";
  Object.entries(entry.roads).forEach(([road, payload]) => {
    const card = document.createElement("article");
    card.className = "metric";
    card.innerHTML = `
      <h3>${road}</h3>
      <p>Signal: <strong>${payload.signal}</strong></p>
      <p>Vehicles: ${payload.vehicleCount}</p>
      <p>Queue: ${payload.queue}</p>
      <p>Wait: ${payload.wait}s</p>
      <p>Emergency: ${payload.emergency ? "Yes" : "No"}</p>
    `;
    elements.laneMetrics.appendChild(card);
  });
}

function syncAllVideos(second) {
  appState.videos.forEach((video) => {
    if (Number.isFinite(video.duration) && video.duration > second) {
      video.currentTime = second;
    }
  });
}

function renderFeeds(feeds) {
  elements.feedGrid.innerHTML = "";
  appState.videos = [];

  feeds.forEach((feed, index) => {
    const card = document.createElement("article");
    card.className = "feed-card";
    card.innerHTML = `
      <div class="feed-head">
        <div>
          <p class="feed-lane">${feed.lane}</p>
          <h3>${feed.road}</h3>
        </div>
        <div class="feed-actions">
          <span class="feed-badge">Annotated Output</span>
          <a href="${feed.annotatedVideo}" target="_blank" rel="noreferrer">Open annotated</a>
          <a href="${feed.sourceVideo}" target="_blank" rel="noreferrer">Original</a>
        </div>
      </div>
      <video controls muted playsinline preload="metadata" src="${feed.annotatedVideo}"></video>
    `;
    const video = card.querySelector("video");
    if (index === 0) {
      video.addEventListener("timeupdate", () => {
        const second = Math.max(0, Math.round(video.currentTime));
        renderBoard(second);
      });
      video.addEventListener("play", () => {
        appState.videos.slice(1).forEach((peer) => {
          peer.currentTime = video.currentTime;
          peer.play().catch(() => {});
        });
      });
      video.addEventListener("pause", () => {
        appState.videos.slice(1).forEach((peer) => peer.pause());
      });
    }
    appState.videos.push(video);
    elements.feedGrid.appendChild(card);
  });
}

function renderManifest(manifest) {
  appState.manifest = manifest;
  elements.timelineSlider.max = String(manifest.maxSecond || 0);
  renderFeeds(manifest.feeds);
  renderBoard(1);
}

async function pollJob(jobId) {
  if (appState.pollTimer) {
    clearInterval(appState.pollTimer);
  }

  appState.pollTimer = setInterval(async () => {
    const response = await fetch(`/api/jobs/${jobId}`);
    const payload = await response.json();
    if (!response.ok) {
      elements.jobStatus.textContent = payload.error || "Could not fetch job status.";
      return;
    }

    elements.jobStatus.textContent = payload.message || payload.status;
    if (payload.previewFeeds && payload.previewFeeds.length) {
      renderFeeds(payload.previewFeeds);
    }
    if (payload.status === "completed") {
      clearInterval(appState.pollTimer);
      appState.pollTimer = null;
      renderManifest(payload.manifest);
    }
    if (payload.status === "failed") {
      clearInterval(appState.pollTimer);
      appState.pollTimer = null;
      elements.jobStatus.textContent = payload.error || "Processing failed.";
    }
  }, 3000);
}

async function submitFeeds(event) {
  event.preventDefault();
  const formData = new FormData(elements.uploadForm);
  elements.jobStatus.textContent = "Uploading feeds...";

  const response = await fetch("/api/jobs", {
    method: "POST",
    body: formData,
  });
  const payload = await response.json();
  if (!response.ok) {
    elements.jobStatus.textContent = payload.error || "Upload failed.";
    return;
  }

  elements.jobStatus.textContent = "Upload complete. Waiting for processing worker.";
  pollJob(payload.jobId);
}

elements.uploadForm.addEventListener("submit", submitFeeds);
elements.timelineSlider.addEventListener("input", (event) => {
  const second = Number(event.target.value);
  syncAllVideos(second);
  renderBoard(second);
});
elements.playAll.addEventListener("click", () => {
  if (!appState.videos.length) {
    return;
  }
  const master = appState.videos[0];
  appState.videos.forEach((video) => {
    video.currentTime = master.currentTime;
    video.play().catch(() => {});
  });
});
elements.pauseAll.addEventListener("click", () => {
  appState.videos.forEach((video) => video.pause());
});
