const appState = {
  manifest: null,
  currentSecond: 0,
  pollTimer: null,
  syncingMedia: false,
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

function videoMimeType(path) {
  if (path.endsWith(".webm")) {
    return "video/webm";
  }
  if (path.endsWith(".mp4")) {
    return "video/mp4";
  }
  return "video/mp4";
}

function feedBadge(feed, preferPreview) {
  if (preferPreview && feed.previewImage) {
    return "Live Annotated Preview";
  }
  return "Annotated Output";
}

function getEntry(second) {
  if (!appState.manifest || !appState.manifest.timeline.length) {
    return null;
  }
  return appState.manifest.timeline.find((entry) => entry.second === second)
    || appState.manifest.timeline[appState.manifest.timeline.length - 1];
}

function renderBoardEntry(entry, maxSecond = null) {
  if (!entry) {
    return;
  }

  appState.currentSecond = entry.second;
  elements.globalSignal.textContent = entry.globalSignal;
  elements.globalMeta.textContent = `Source: ${entry.decisionSource}`;
  elements.junctionLabel.textContent = entry.globalSignal;
  elements.secondLabel.textContent = `Second ${entry.second}`;
  elements.qValues.textContent = `q = [${entry.qValues.join(", ")}]`;
  elements.timelineSlider.value = String(entry.second);
  if (maxSecond !== null) {
    elements.timelineSlider.max = String(maxSecond);
  }

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
      <p>Stopped vehicles: ${payload.stoppedCount}</p>
      <p>Visible tracks: ${payload.visibleVehicleCount}</p>
      <p>Halt wait: ${payload.wait}s</p>
      <p>Emergency memory: ${payload.emergency ? "Yes" : "No"}</p>
    `;
    elements.laneMetrics.appendChild(card);
  });
}

function renderBoard(second) {
  const entry = getEntry(second);
  if (!entry) {
    return;
  }
  renderBoardEntry(entry, appState.manifest ? appState.manifest.maxSecond || entry.second : entry.second);
}

function syncAllVideos(second) {
  appState.videos.forEach((video) => {
    if (Number.isFinite(video.duration) && video.duration > second) {
      video.currentTime = second;
    }
  });
}

function syncPeerVideos(masterVideo, { play = false, pause = false } = {}) {
  if (appState.syncingMedia) {
    return;
  }

  appState.syncingMedia = true;
  appState.videos.forEach((video) => {
    if (video === masterVideo) {
      return;
    }
    if (Number.isFinite(masterVideo.currentTime)) {
      video.currentTime = masterVideo.currentTime;
    }
    if (play) {
      video.play().catch(() => {});
    }
    if (pause) {
      video.pause();
    }
  });
  appState.syncingMedia = false;
}

function playAllVideos() {
  if (!appState.videos.length) {
    return;
  }
  const master = appState.videos[0];
  syncPeerVideos(master);
  appState.videos.forEach((video) => {
    video.play().catch(() => {});
  });
}

function pauseAllVideos() {
  appState.videos.forEach((video) => video.pause());
}

function buildFeedMedia(feed, preferPreview) {
  if (preferPreview && feed.previewImage) {
    return `<img class="feed-preview" src="${feed.previewImage}" alt="${feed.road} live annotated preview">`;
  }
  if (feed.annotatedVideo) {
    return `
      <video controls muted playsinline preload="metadata">
        <source src="${feed.annotatedVideo}" type="${videoMimeType(feed.annotatedVideo)}">
      </video>
    `;
  }
  return `<div class="feed-placeholder">Preparing annotated output...</div>`;
}

function renderFeeds(feeds, { preferPreview = false } = {}) {
  elements.feedGrid.innerHTML = "";
  appState.videos = [];

  feeds.forEach((feed) => {
    const card = document.createElement("article");
    card.className = "feed-card";
    card.innerHTML = `
      <div class="feed-head">
        <div>
          <p class="feed-lane">${feed.lane}</p>
          <h3>${feed.road}</h3>
        </div>
        <div class="feed-actions">
          <span class="feed-badge">${feedBadge(feed, preferPreview)}</span>
          ${feed.annotatedVideo ? `<a href="${feed.annotatedVideo}" target="_blank" rel="noreferrer">Open annotated</a>` : ""}
          <a href="${feed.sourceVideo}" target="_blank" rel="noreferrer">Original</a>
        </div>
      </div>
      ${buildFeedMedia(feed, preferPreview)}
    `;
    const video = card.querySelector("video");
    if (video) {
      video.addEventListener("timeupdate", () => {
        if (appState.syncingMedia) {
          return;
        }
        const second = Math.max(0, Math.round(video.currentTime));
        renderBoard(second);
        if (!video.paused) {
          syncPeerVideos(video);
        }
      });
      video.addEventListener("play", () => {
        syncPeerVideos(video, { play: true });
      });
      video.addEventListener("pause", () => {
        syncPeerVideos(video, { pause: true });
      });
      video.addEventListener("seeked", () => {
        if (appState.syncingMedia) {
          return;
        }
        syncPeerVideos(video);
        renderBoard(Math.max(0, Math.round(video.currentTime)));
      });
      appState.videos.push(video);
    }
    elements.feedGrid.appendChild(card);
  });
}

function renderManifest(manifest) {
  appState.manifest = manifest;
  elements.timelineSlider.max = String(manifest.maxSecond || 0);
  renderFeeds(manifest.feeds, { preferPreview: false });
  renderBoard(1);
  elements.jobStatus.textContent = "Processing complete. Annotated outputs are ready below.";
  elements.feedGrid.scrollIntoView({ behavior: "smooth", block: "start" });
  if (appState.videos.length) {
    appState.videos[0].addEventListener("loadeddata", () => {
      playAllVideos();
    }, { once: true });
  }
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
      renderFeeds(payload.previewFeeds, { preferPreview: true });
    }
    if (payload.currentEntry) {
      renderBoardEntry(
        payload.currentEntry,
        payload.progress ? payload.progress.total || payload.currentEntry.second : payload.currentEntry.second,
      );
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
  }, 1000);
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
  playAllVideos();
});
elements.pauseAll.addEventListener("click", () => {
  pauseAllVideos();
});
