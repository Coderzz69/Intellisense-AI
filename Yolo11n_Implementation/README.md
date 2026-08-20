# Intellient_Traffic_management

Interactive traffic management dashboard built with React and a local YOLO-based analysis backend.

## Demo Video

<p align="center">
  <a href="https://www.youtube.com/watch?v=ndheRcY9i64">
    <img src="https://img.youtube.com/vi/ndheRcY9i64/maxresdefault.jpg" width="700">
  </a>
</p>

## Features

- Streams per-second vehicle JSON from a local video source
- Tracks total vehicles, ambulances, halting vehicles, and wait times
- Breaks lane distribution down across north, east, south, and west
- Serves a local video preview and SSE analysis feed
- Renders a Mapbox traffic density map with heatmap and marker modes

## Mapbox Setup

1. Install frontend dependencies:

```powershell
npm install
```

2. Create a `.env` file from `.env.example` and set your Mapbox token:

```powershell
Copy-Item .env.example .env
```

3. Set `REACT_APP_MAPBOX_ACCESS_TOKEN` in `.env`.

Notes:

- Browser geolocation works on `localhost` and HTTPS origins.
- The current MVP projects the live `N/E/S/W` lane telemetry onto map nodes around the user's location.
- If your backend starts returning real node coordinates, pass them as `trafficNodes` and the heatmap renderer will use them directly.

## Local Run

```powershell
npm run start:backend
npm start
```

