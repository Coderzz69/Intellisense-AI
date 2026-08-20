const FALLBACK_TRAFFIC_CENTER = {
  latitude: 17.385,
  longitude: 78.4867,
};

const LANE_ORDER = ['N', 'E', 'S', 'W'];
const LANE_OFFSETS = {
  N: { latitude: 0.0072, longitude: 0 },
  E: { latitude: 0, longitude: 0.0084 },
  S: { latitude: -0.0072, longitude: 0 },
  W: { latitude: 0, longitude: -0.0084 },
};

const LANE_LABELS = {
  N: 'North Approach',
  E: 'East Approach',
  S: 'South Approach',
  W: 'West Approach',
};

function roundCoordinate(value) {
  return Number(value.toFixed(6));
}

function toPositiveNumber(value, fallback = 0) {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue) || numericValue < 0) {
    return fallback;
  }
  return numericValue;
}

function toIntensity(value) {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue) || numericValue <= 0) {
    return 1;
  }
  return Number(numericValue.toFixed(2));
}

function normaliseExternalNodes(nodes) {
  if (!Array.isArray(nodes)) {
    return [];
  }

  return nodes
    .map((node, index) => {
      const latitude = Number(node?.latitude ?? node?.lat);
      const longitude = Number(node?.longitude ?? node?.lng ?? node?.lon);

      if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) {
        return null;
      }

      const vehicleCount = toPositiveNumber(
        node?.vehicleCount ?? node?.vehicle_count ?? node?.count ?? node?.density ?? node?.intensity,
        0
      );

      return {
        id: node?.id ?? `traffic-node-${index + 1}`,
        latitude: roundCoordinate(latitude),
        longitude: roundCoordinate(longitude),
        title: node?.title ?? node?.label ?? `Traffic node ${index + 1}`,
        description: node?.description ?? '',
        intensity: toIntensity(node?.intensity ?? vehicleCount),
        vehicleCount,
        source: node?.source ?? 'dataset',
        isDerived: false,
      };
    })
    .filter(Boolean);
}

function buildLaneNode(lane, laneStats, baseLocation, timestampLabel) {
  const offset = LANE_OFFSETS[lane];
  const vehicles = toPositiveNumber(laneStats?.vehicles, 0);
  const halting = toPositiveNumber(laneStats?.halting, 0);
  const emergencyVehicles = toPositiveNumber(laneStats?.emergencyVehicles, 0);
  const ambulances = toPositiveNumber(laneStats?.ambulances, 0);
  const averageWaitTimeSec = toPositiveNumber(laneStats?.averageWaitTimeSec, 0);

  const intensity = toIntensity(
    vehicles + halting * 1.5 + emergencyVehicles * 2.5 + ambulances * 2 + averageWaitTimeSec / 4
  );

  return {
    id: `lane-${lane.toLowerCase()}`,
    latitude: roundCoordinate(baseLocation.latitude + offset.latitude),
    longitude: roundCoordinate(baseLocation.longitude + offset.longitude),
    title: LANE_LABELS[lane],
    description: `Live lane density snapshot${timestampLabel ? ` at ${timestampLabel}` : ''}.`,
    intensity,
    vehicleCount: vehicles,
    source: 'live-lane',
    isDerived: true,
    lane,
  };
}

function buildDerivedNodes(frame, baseLocation) {
  const metrics = frame?.metrics ?? {};
  const laneStats = metrics.laneStats ?? {};
  const haltingVehicleCount = toPositiveNumber(metrics.haltingVehicleCount, 0);
  const emergencyVehicleCount = toPositiveNumber(metrics.emergencyVehicleCount, 0);
  const totalVehicles = toPositiveNumber(metrics.totalVehicles, 0);
  const timestampLabel = frame?.timestampLabel ?? '';

  const intersectionNode = {
    id: 'intersection-core',
    latitude: roundCoordinate(baseLocation.latitude),
    longitude: roundCoordinate(baseLocation.longitude),
    title: 'Intersection Core',
    description: frame
      ? `Live congestion summary captured at ${timestampLabel}.`
      : 'Waiting for live traffic data from the dashboard stream.',
    intensity: toIntensity(totalVehicles + haltingVehicleCount * 1.5 + emergencyVehicleCount * 3),
    vehicleCount: totalVehicles,
    source: 'live-summary',
    isDerived: true,
  };

  const laneNodes = LANE_ORDER.map((lane) =>
    buildLaneNode(lane, laneStats[lane], baseLocation, timestampLabel)
  );

  return [intersectionNode, ...laneNodes];
}

export function buildTrafficNodes(frame, userLocation = null) {
  const explicitNodes = normaliseExternalNodes(
    frame?.trafficNodes ?? frame?.traffic_nodes ?? frame?.nodes ?? []
  );
  if (explicitNodes.length) {
    return explicitNodes;
  }

  const baseLocation = userLocation ?? FALLBACK_TRAFFIC_CENTER;
  return buildDerivedNodes(frame, baseLocation);
}

export function buildTrafficNodesGeoJson(nodes) {
  return {
    type: 'FeatureCollection',
    features: nodes.map((node) => ({
      type: 'Feature',
      geometry: {
        type: 'Point',
        coordinates: [node.longitude, node.latitude],
      },
      // Mapbox heatmaps consume GeoJSON points, so each traffic node is flattened
      // into a feature with a numeric `intensity` property for weight and color.
      properties: {
        id: node.id,
        title: node.title ?? 'Traffic node',
        description: node.description ?? '',
        intensity: toIntensity(node.intensity ?? node.vehicleCount ?? 1),
        vehicleCount: toPositiveNumber(node.vehicleCount, 0),
        source: node.source ?? 'dataset',
        isDerived: Boolean(node.isDerived),
        lane: node.lane ?? '',
      },
    })),
  };
}

export { FALLBACK_TRAFFIC_CENTER };
