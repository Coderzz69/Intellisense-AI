import { useEffect, useRef, useState } from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import './TrafficMap.css';
import { buildTrafficNodesGeoJson, FALLBACK_TRAFFIC_CENTER } from '../utils/trafficNodes';

const MAPBOX_STYLE = 'mapbox://styles/mapbox/dark-v11';
const SOURCE_ID = 'traffic-nodes';
const HEATMAP_LAYER_ID = 'traffic-nodes-heatmap';
const MARKER_LAYER_ID = 'traffic-nodes-markers';
const MARKER_LABEL_LAYER_ID = 'traffic-nodes-labels';
const DEFAULT_ZOOM = 11.2;

function formatCoordinate(value) {
  return Number(value).toFixed(5);
}

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function buildPopupMarkup(feature) {
  const properties = feature?.properties ?? {};
  const [longitude, latitude] = feature?.geometry?.coordinates ?? [0, 0];
  const title = escapeHtml(properties.title || 'Traffic node');
  const description = escapeHtml(properties.description || 'No description available.');
  const vehicleCount = Number(properties.vehicleCount ?? 0);
  const intensity = Number(properties.intensity ?? 1);

  return `
    <div class="traffic-map-popup-card">
      <strong>${title}</strong>
      <span>${formatCoordinate(latitude)}, ${formatCoordinate(longitude)}</span>
      <span>Vehicles: ${vehicleCount}</span>
      <span>Intensity: ${intensity.toFixed(2)}</span>
      <p>${description}</p>
    </div>
  `;
}

function buildMarkerOpacity(viewMode) {
  if (viewMode === 'markers') {
    return 1;
  }

  return [
    'interpolate',
    ['linear'],
    ['zoom'],
    8,
    0,
    10,
    0.08,
    11.5,
    0.62,
    13,
    0.92,
    15,
    1,
  ];
}

function applyViewMode(map, viewMode) {
  if (map.getLayer(HEATMAP_LAYER_ID)) {
    map.setLayoutProperty(HEATMAP_LAYER_ID, 'visibility', viewMode === 'heatmap' ? 'visible' : 'none');
  }

  if (map.getLayer(MARKER_LAYER_ID)) {
    map.setPaintProperty(MARKER_LAYER_ID, 'circle-opacity', buildMarkerOpacity(viewMode));
    map.setPaintProperty(MARKER_LAYER_ID, 'circle-stroke-opacity', buildMarkerOpacity(viewMode));
  }

  if (map.getLayer(MARKER_LABEL_LAYER_ID)) {
    map.setLayoutProperty(MARKER_LABEL_LAYER_ID, 'visibility', viewMode === 'markers' ? 'visible' : 'none');
  }
}

function addTrafficLayers(map, geoJson) {
  map.addSource(SOURCE_ID, {
    type: 'geojson',
    data: geoJson,
  });

  map.addLayer({
    id: HEATMAP_LAYER_ID,
    type: 'heatmap',
    source: SOURCE_ID,
    maxzoom: 15,
    paint: {
      // Weight each point by its traffic intensity so dense lanes produce a hotter field.
      'heatmap-weight': [
        'interpolate',
        ['linear'],
        ['coalesce', ['get', 'intensity'], 1],
        1,
        0.12,
        20,
        0.55,
        60,
        1,
      ],
      'heatmap-intensity': [
        'interpolate',
        ['linear'],
        ['zoom'],
        5,
        0.9,
        9,
        1.6,
        13,
        2.6,
      ],
      // The color ramp moves from blue to red as density increases.
      'heatmap-color': [
        'interpolate',
        ['linear'],
        ['heatmap-density'],
        0,
        'rgba(37, 99, 235, 0)',
        0.18,
        '#2563eb',
        0.38,
        '#22c55e',
        0.62,
        '#fde047',
        0.8,
        '#fb923c',
        1,
        '#ef4444',
      ],
      'heatmap-radius': [
        'interpolate',
        ['linear'],
        ['zoom'],
        5,
        16,
        8,
        28,
        11,
        42,
        14,
        64,
      ],
      'heatmap-opacity': [
        'interpolate',
        ['linear'],
        ['zoom'],
        7,
        0.74,
        11,
        0.64,
        13,
        0.38,
        15,
        0,
      ],
    },
  });

  map.addLayer({
    id: MARKER_LAYER_ID,
    type: 'circle',
    source: SOURCE_ID,
    paint: {
      'circle-color': [
        'case',
        ['boolean', ['get', 'isDerived'], false],
        '#ff6b6b',
        '#ef4444',
      ],
      'circle-radius': [
        'interpolate',
        ['linear'],
        ['zoom'],
        7,
        5,
        11,
        7,
        14,
        10,
        17,
        13,
      ],
      'circle-stroke-color': '#ffe7dd',
      'circle-stroke-width': 1.5,
      'circle-opacity': buildMarkerOpacity('heatmap'),
      'circle-stroke-opacity': buildMarkerOpacity('heatmap'),
    },
  });

  map.addLayer({
    id: MARKER_LABEL_LAYER_ID,
    type: 'symbol',
    source: SOURCE_ID,
    minzoom: 12,
    layout: {
      'text-field': ['coalesce', ['get', 'lane'], ['get', 'title']],
      'text-size': 11,
      'text-offset': [0, 1.4],
      'text-anchor': 'top',
      'text-font': ['Open Sans Semibold', 'Arial Unicode MS Bold'],
    },
    paint: {
      'text-color': '#f8fafc',
      'text-halo-color': 'rgba(3, 7, 18, 0.9)',
      'text-halo-width': 1,
    },
  });
}

function TrafficMap({ trafficNodes, onUserLocationChange, timestampLabel, totalVehicles }) {
  const mapContainerRef = useRef(null);
  const mapRef = useRef(null);
  const popupRef = useRef(null);
  const [isMapReady, setIsMapReady] = useState(false);
  const viewMode = 'markers';
  const [locationMessage, setLocationMessage] = useState('Resolving your current location...');
  const accessToken = process.env.REACT_APP_MAPBOX_ACCESS_TOKEN;
  const geoJson = buildTrafficNodesGeoJson(trafficNodes);

  useEffect(() => {
    if (!accessToken || mapRef.current || !mapContainerRef.current) {
      return undefined;
    }

    mapboxgl.accessToken = accessToken;

    const map = new mapboxgl.Map({
      container: mapContainerRef.current,
      style: MAPBOX_STYLE,
      center: [FALLBACK_TRAFFIC_CENTER.longitude, FALLBACK_TRAFFIC_CENTER.latitude],
      zoom: DEFAULT_ZOOM,
      attributionControl: false,
    });

    mapRef.current = map;
    map.addControl(new mapboxgl.NavigationControl(), 'top-right');
    map.addControl(
      new mapboxgl.GeolocateControl({
        positionOptions: {
          enableHighAccuracy: true,
        },
        trackUserLocation: true,
        showUserHeading: true,
      }),
      'top-right'
    );

    function handleLoad() {
      addTrafficLayers(map, geoJson);
      applyViewMode(map, viewMode);
      setIsMapReady(true);
    }

    function handleMarkerClick(event) {
      const feature = event.features?.[0];
      if (!feature) {
        return;
      }

      popupRef.current?.remove();
      popupRef.current = new mapboxgl.Popup({
        closeButton: true,
        closeOnClick: true,
        className: 'traffic-map-popup',
      })
        .setLngLat(event.lngLat)
        .setHTML(buildPopupMarkup(feature))
        .addTo(map);
    }

    function handleMouseEnter() {
      map.getCanvas().style.cursor = 'pointer';
    }

    function handleMouseLeave() {
      map.getCanvas().style.cursor = '';
    }

    map.on('load', handleLoad);
    map.on('click', MARKER_LAYER_ID, handleMarkerClick);
    map.on('mouseenter', MARKER_LAYER_ID, handleMouseEnter);
    map.on('mouseleave', MARKER_LAYER_ID, handleMouseLeave);

    return () => {
      popupRef.current?.remove();
      popupRef.current = null;
      map.remove();
      mapRef.current = null;
      setIsMapReady(false);
    };
  }, [accessToken, geoJson, viewMode]);

  useEffect(() => {
    if (!isMapReady || !mapRef.current) {
      return;
    }

    const source = mapRef.current.getSource(SOURCE_ID);
    if (source) {
      source.setData(geoJson);
    }
  }, [geoJson, isMapReady]);

  useEffect(() => {
    if (!isMapReady || !mapRef.current) {
      return;
    }

    applyViewMode(mapRef.current, viewMode);
  }, [isMapReady, viewMode]);

  useEffect(() => {
    if (!isMapReady || !mapRef.current || !navigator.geolocation) {
      return;
    }

    navigator.geolocation.getCurrentPosition(
      (position) => {
        const nextLocation = {
          latitude: position.coords.latitude,
          longitude: position.coords.longitude,
        };
        mapRef.current?.flyTo({
          center: [nextLocation.longitude, nextLocation.latitude],
          zoom: DEFAULT_ZOOM + 0.8,
          essential: true,
        });
        onUserLocationChange?.(nextLocation);
        setLocationMessage('Centered on your live location.');
      },
      () => {
        setLocationMessage('Location access denied. Showing the fallback intersection center.');
      },
      {
        enableHighAccuracy: true,
        timeout: 12000,
        maximumAge: 60000,
      }
    );
  }, [isMapReady, onUserLocationChange]);

  if (!accessToken) {
    return (
      <section className="panel traffic-map-panel">
        <div className="traffic-map-toolbar">
        <div>
          <span className="panel-kicker">Traffic Map</span>
          <h2>Mapbox token required</h2>
        </div>
      </div>
      <div className="traffic-map-empty">
        Set <code>REACT_APP_MAPBOX_ACCESS_TOKEN</code> in <code>.env</code> and restart the
          frontend to enable the map.
      </div>
    </section>
  );
}

  return (
    <section className="panel traffic-map-panel">
      <div className="traffic-map-toolbar">
        <div>
          <span className="panel-kicker">Traffic Map</span>
          <h2>Live marker overlay</h2>
        </div>
      </div>

      <div className="traffic-map-stage">
        <div ref={mapContainerRef} className="traffic-map-canvas" />

        <div className="traffic-map-floating-card traffic-map-summary">
          <span>Snapshot</span>
          <strong>{timestampLabel || 'Idle'}</strong>
          <span>{trafficNodes.length} traffic nodes</span>
          <span>{totalVehicles || 0} vehicles in the latest stream frame</span>
          <span>{locationMessage}</span>
        </div>
      </div>
    </section>
  );
}

export default TrafficMap;
