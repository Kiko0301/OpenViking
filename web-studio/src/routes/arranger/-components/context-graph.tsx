import { useMemo } from 'react'
import dagre from '@dagrejs/dagre'
import {
  Background,
  Controls,
  Handle,
  Position,
  ReactFlow,
} from '@xyflow/react'
import type { NodeProps } from '@xyflow/react'
import { CableIcon, FileTextIcon, FolderIcon } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'

import '@xyflow/react/dist/style.css'

import type { VikingFsEntry } from '#/routes/resources/-types/viking-fm'
import { cn } from '#/lib/utils'

type GraphData = {
  entry: VikingFsEntry
  isLive: boolean
  label: string
}

type ContextGraphProps = {
  entries: VikingFsEntry[]
  isLive: (uri: string) => boolean
  onSelect: (entry: VikingFsEntry) => void
  root: VikingFsEntry
  selectedUri?: string
}

function ResourceNode({ data, selected }: NodeProps) {
  const value = data as GraphData
  const Icon: LucideIcon = value.isLive
    ? CableIcon
    : value.entry.isDir
      ? FolderIcon
      : FileTextIcon
  return (
    <div
      className={cn(
        'flex h-11 w-44 items-center gap-2 rounded-md border bg-background px-3 text-xs shadow-sm',
        selected && 'border-primary ring-2 ring-primary/15',
      )}
    >
      <Handle type="target" position={Position.Left} className="opacity-0" />
      <Icon
        className={cn(
          'size-4 shrink-0',
          value.isLive ? 'text-emerald-600' : 'text-muted-foreground',
        )}
      />
      <span className="truncate">{value.label}</span>
      <Handle type="source" position={Position.Right} className="opacity-0" />
    </div>
  )
}

const nodeTypes = { resource: ResourceNode }

function parentUri(uri: string): string {
  return uri.slice(0, uri.lastIndexOf('/'))
}

export function ContextGraph({
  entries,
  isLive,
  onSelect,
  root,
  selectedUri,
}: ContextGraphProps) {
  const { edges, nodes } = useMemo(() => {
    const graph = new dagre.graphlib.Graph()
    graph.setDefaultEdgeLabel(() => ({}))
    graph.setGraph({ rankdir: 'LR', nodesep: 24, ranksep: 64 })
    const allEntries = [root, ...entries]
    for (const entry of allEntries) {
      graph.setNode(entry.uri, { height: 44, width: 176 })
    }
    const nextEdges = entries
      .filter((entry) => graph.hasNode(parentUri(entry.uri)))
      .map((entry) => {
        const source = parentUri(entry.uri)
        graph.setEdge(source, entry.uri)
        return {
          id: `${source}->${entry.uri}`,
          source,
          target: entry.uri,
          type: 'smoothstep',
        }
      })
    dagre.layout(graph)
    return {
      edges: nextEdges,
      nodes: allEntries.map((entry) => {
        const position = graph.node(entry.uri) as { x: number; y: number }
        return {
          id: entry.uri,
          data: {
            entry,
            isLive: isLive(entry.uri),
            label: entry.name || entry.uri,
          } satisfies GraphData,
          position: { x: position.x - 88, y: position.y - 22 },
          selected: entry.uri === selectedUri,
          type: 'resource',
        }
      }),
    }
  }, [entries, isLive, root, selectedUri])

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      fitView
      minZoom={0.35}
      maxZoom={1.6}
      nodesDraggable={false}
      onNodeClick={(_, node) => onSelect((node.data as GraphData).entry)}
      proOptions={{ hideAttribution: true }}
      className="bg-background"
    >
      <Background gap={24} size={1} />
      <Controls showInteractive={false} />
    </ReactFlow>
  )
}
