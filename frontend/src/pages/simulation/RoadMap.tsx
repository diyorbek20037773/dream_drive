import type { KeyboardEvent, ReactNode } from 'react';
import { useEffect, useMemo, useRef, useState } from 'react';

import type { RoadGraph, RunLive } from '../../api/types';
import { useI18n } from '../../i18n';
import { graphBounds, polylinePath } from '../../lib/graph';

export type MapPickMode = 'start' | 'destination';

/**
 * Top-down virtual road graph (ENU metres, north up). Node selection is the only input; the map is a UI/navigation
 * view of the static topology, not a perception input.
 */
export function RoadMap({
  graph,
  startId,
  destinationId,
  startCandidates,
  destinationCandidates,
  mode,
  onPick,
  route,
  ego,
  interactive,
}: {
  graph: RoadGraph;
  startId: string | null;
  destinationId: string | null;
  startCandidates: Set<string>;
  destinationCandidates: Set<string>;
  mode: MapPickMode;
  onPick: (nodeId: string) => void;
  route: number[][] | null;
  ego: RunLive['ego'] | null;
  interactive: boolean;
}) {
  const { t, lang } = useI18n();
  const svgRef = useRef<SVGSVGElement>(null);
  const [renderedWidth, setRenderedWidth] = useState(720);
  useEffect(() => {
    const element = svgRef.current;
    if (!element || typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver((entries) => {
      const width = entries[0]?.contentRect.width;
      if (width && width > 0) setRenderedWidth(width);
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const bounds = useMemo(() => graphBounds(graph, route ?? []), [graph, route]);
  const extent = Math.max(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY, 20);
  const pad = extent * 0.08 + 6;
  const viewWidth = bounds.maxX - bounds.minX + 2 * pad;
  const viewBox = `${bounds.minX - pad} ${-bounds.maxY - pad} ${viewWidth} ${bounds.maxY - bounds.minY + 2 * pad}`;
  // Metres per screen pixel: labels and markers keep a readable on-screen size at every viewport width,
  // while roads keep their physical width.
  const mPerPx = viewWidth / renderedWidth;
  const roadWidth = graph.lane_width_m > 0 ? graph.lane_width_m * 2 : 7;
  const nodeR = Math.max(2.2, 7 * mPerPx);
  const fontSize = Math.max(2.5, 12 * mPerPx);
  const egoSize = Math.max(3.5, 11 * mPerPx);

  const candidates = mode === 'start' ? startCandidates : destinationCandidates;

  const keyHandler = (nodeId: string, enabled: boolean) => (event: KeyboardEvent<SVGGElement>) => {
    if (!enabled) return;
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      onPick(nodeId);
    }
  };

  return (
    <svg
      ref={svgRef}
      viewBox={viewBox}
      className="h-auto max-h-[460px] w-full rounded-control bg-[#eef3ec]"
      role="group"
      aria-label={t.sim.mapTitle}
      data-testid="road-map"
    >
      {/* Roads */}
      <g aria-hidden>
        {graph.edges.map((edge) => (
          <path
            key={`road-${edge.id}`}
            d={polylinePath(edge.polyline)}
            fill="none"
            stroke="#c7d0db"
            strokeWidth={roadWidth}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        ))}
        {graph.edges.map((edge) => (
          <path
            key={`center-${edge.id}`}
            d={polylinePath(edge.polyline)}
            fill="none"
            stroke="#eab308"
            strokeWidth={Math.max(0.25, roadWidth / 28)}
            strokeDasharray={`${roadWidth * 0.5} ${roadWidth * 0.5}`}
          />
        ))}
      </g>
      {/* Planned route (lane-centre path from the backend) */}
      {route && route.length > 1 && (
        <path
          d={polylinePath(route)}
          fill="none"
          stroke="#2563eb"
          strokeWidth={Math.max(1.2, roadWidth / 3.2)}
          strokeLinecap="round"
          strokeLinejoin="round"
          opacity={0.9}
          data-testid="map-route"
          aria-hidden
        />
      )}
      {/* Nodes */}
      {graph.nodes.map((node) => {
        const enabled = interactive && candidates.has(node.id);
        const isStart = node.id === startId;
        const isDestination = node.id === destinationId;
        const unreachable = !destinationCandidates.has(node.id) && !startCandidates.has(node.id);
        const label = lang === 'uz' ? node.label_uz : node.label_en;
        const role = isStart ? t.sim.legendStart : isDestination ? t.sim.legendDestination : unreachable ? t.sim.nodeUnreachable : '';
        const fill = isStart ? '#15803d' : isDestination ? '#2563eb' : unreachable ? '#e2e8f0' : '#ffffff';
        return (
          <g
            key={node.id}
            role="button"
            tabIndex={enabled ? 0 : -1}
            aria-disabled={!enabled}
            aria-pressed={isStart || isDestination}
            aria-label={t.sim.nodeAria(label, role)}
            data-testid={`map-node-${node.id}`}
            data-node-id={node.id}
            data-enabled={enabled}
            className={enabled ? 'cursor-pointer' : 'cursor-not-allowed'}
            onClick={() => enabled && onPick(node.id)}
            onKeyDown={keyHandler(node.id, enabled)}
          >
            <title>{unreachable ? `${label} — ${t.sim.nodeUnreachable}` : label}</title>
            {/* Larger transparent hit target for touch input. */}
            <circle cx={node.x} cy={-node.y} r={nodeR * 2.2} fill="transparent" />
            <circle
              cx={node.x}
              cy={-node.y}
              r={nodeR}
              fill={fill}
              stroke={unreachable ? '#94a3b8' : '#12233b'}
              strokeWidth={nodeR * 0.35}
              strokeDasharray={unreachable ? `${nodeR * 0.6} ${nodeR * 0.5}` : undefined}
              opacity={!enabled && !isStart && !isDestination && interactive ? 0.55 : 1}
            />
            <text
              x={node.x}
              y={-node.y - nodeR - fontSize * 0.45}
              textAnchor="middle"
              fontSize={fontSize}
              fontWeight={600}
              fill={unreachable ? '#64748b' : '#12233b'}
              paintOrder="stroke"
              stroke="#eef3ec"
              strokeWidth={fontSize * 0.25}
            >
              {label}
            </text>
          </g>
        );
      })}
      {/* Ego pose from simulator telemetry (allowed localization input). */}
      {ego && (
        <g
          transform={`translate(${ego.x_m} ${-ego.y_m}) rotate(${(-ego.yaw_rad * 180) / Math.PI})`}
          data-testid="map-ego"
          aria-hidden
        >
          <path
            d={`M ${egoSize} 0 L ${-egoSize * 0.7} ${egoSize * 0.6} L ${-egoSize * 0.35} 0 L ${-egoSize * 0.7} ${-egoSize * 0.6} Z`}
            fill="#0891b2"
            stroke="#ffffff"
            strokeWidth={egoSize * 0.18}
          />
        </g>
      )}
    </svg>
  );
}

export function MapLegend() {
  const { t } = useI18n();
  const item = (swatch: ReactNode, label: string) => (
    <li className="flex items-center gap-1.5">
      {swatch}
      <span>{label}</span>
    </li>
  );
  return (
    <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted" data-testid="map-legend">
      {item(<span className="size-3 rounded-full border-2 border-ink bg-success" />, t.sim.legendStart)}
      {item(<span className="size-3 rounded-full border-2 border-ink bg-accent" />, t.sim.legendDestination)}
      {item(<span className="h-1 w-5 rounded bg-accent" />, t.sim.legendRoute)}
      {item(
        <svg viewBox="-6 -5 12 10" className="size-3.5" aria-hidden>
          <path d="M 5 0 L -4 4 L -2 0 L -4 -4 Z" fill="#0891b2" />
        </svg>,
        t.sim.legendEgo,
      )}
      {item(<span className="size-3 rounded-full border-2 border-dashed border-slate-400 bg-slate-200" />, t.sim.legendUnreachable)}
    </ul>
  );
}
