import type { RoadGraph } from '../api/types';

/**
 * Nodes reachable from `start` over the road graph. Roads are two-way (docs/IMPLEMENTATION_SPEC.md section 4), so
 * edges are treated as undirected. This only drives which map nodes are selectable; the backend plans the route and
 * rejects unreachable destinations itself (422 destination_unreachable).
 */
export function reachableFrom(graph: RoadGraph, start: string | null): Set<string> {
  const reachable = new Set<string>();
  if (!start || !graph.nodes.some((node) => node.id === start)) return reachable;
  const adjacency = new Map<string, string[]>();
  for (const edge of graph.edges) {
    adjacency.set(edge.from, [...(adjacency.get(edge.from) ?? []), edge.to]);
    adjacency.set(edge.to, [...(adjacency.get(edge.to) ?? []), edge.from]);
  }
  const queue = [start];
  reachable.add(start);
  while (queue.length > 0) {
    const current = queue.shift() as string;
    for (const next of adjacency.get(current) ?? []) {
      if (!reachable.has(next)) {
        reachable.add(next);
        queue.push(next);
      }
    }
  }
  return reachable;
}

/** Nodes that have at least one road (isolated nodes cannot start a route). */
export function connectedNodes(graph: RoadGraph): Set<string> {
  const connected = new Set<string>();
  for (const edge of graph.edges) {
    connected.add(edge.from);
    connected.add(edge.to);
  }
  return connected;
}

export interface Bounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

export function graphBounds(graph: RoadGraph, extra: number[][] = []): Bounds {
  const xs: number[] = [];
  const ys: number[] = [];
  const add = (point: number[]) => {
    const [x, y] = point;
    if (typeof x === 'number' && typeof y === 'number' && Number.isFinite(x) && Number.isFinite(y)) {
      xs.push(x);
      ys.push(y);
    }
  };
  graph.nodes.forEach((node) => add([node.x, node.y]));
  graph.edges.forEach((edge) => edge.polyline.forEach(add));
  extra.forEach(add);
  if (xs.length === 0) return { minX: 0, maxX: 1, minY: 0, maxY: 1 };
  return { minX: Math.min(...xs), maxX: Math.max(...xs), minY: Math.min(...ys), maxY: Math.max(...ys) };
}

/** SVG path for a polyline in ENU metres, with y flipped so north is up. */
export function polylinePath(points: number[][]): string {
  return points
    .filter((point) => point.length >= 2)
    .map((point, index) => `${index === 0 ? 'M' : 'L'}${(point[0] ?? 0).toFixed(2)} ${(-(point[1] ?? 0)).toFixed(2)}`)
    .join(' ');
}
