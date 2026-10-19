import type { VikingFsEntry } from '#/routes/resources/-types/viking-fm'

export type ArrangerView = 'graph' | 'list'

export type LiveContextMount = {
  node_uri: string
  tool_uri: string
  mcp: string
  tool: string
  arguments: Record<string, unknown>
  revision?: string
  created_at?: string
  title?: string
  description?: string
  timeout_seconds?: number
  tags?: string[]
}

export type JsonSchema = {
  type?: string | string[]
  anyOf?: JsonSchema[]
  oneOf?: JsonSchema[]
  allOf?: JsonSchema[]
  title?: string
  description?: string
  default?: unknown
  enum?: unknown[]
  properties?: Record<string, JsonSchema>
  required?: string[]
  items?: JsonSchema
  minimum?: number
  maximum?: number
  minLength?: number
  maxLength?: number
  examples?: unknown[]
}

export type McpToolDefinition = {
  name: string
  title?: string
  description?: string
  inputSchema: JsonSchema
  annotations?: Record<string, unknown>
}

export type McpCatalogTool = {
  server: string
  uri: string
  revision: string
  tool: McpToolDefinition
}

export type McpCatalog = {
  complete: boolean
  error?: string
  servers: Array<{ name: string; tools: McpCatalogTool[] }>
  tools: McpCatalogTool[]
}

export type LiveMountIndex = {
  complete: boolean
  error?: string
  mounts: Map<string, LiveContextMount>
}

export type LoadedDirectory = {
  entries: VikingFsEntry[]
  hasMore: boolean
  nextOffset: number
}

export type TextDocument = {
  bytes: Uint8Array
  content: string
  sha256: string
}

export type MountInput = {
  tool_uri: string
  node_uri: string
  title: string
  description?: string
  timeout_seconds: number
  arguments: Record<string, unknown>
  tags?: string[]
}

export type MutationKind = 'copy' | 'delete' | 'move'
