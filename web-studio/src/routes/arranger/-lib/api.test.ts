import { describe, expect, it } from 'vitest'

import {
  isLiveContextDocument,
  parseLiveContextMetadata,
  parseMcpMounts,
  parseMcpServers,
  parseMcpTools,
} from './api'

describe('MCP response parsing', () => {
  it('accepts result objects and direct arrays', () => {
    expect(parseMcpServers({ servers: [{ name: 'k8s' }] })).toEqual([
      { name: 'k8s' },
    ])
    expect(parseMcpServers([{ name: 'database' }])).toEqual([
      { name: 'database' },
    ])
    expect(
      parseMcpMounts({
        mounts: [
          {
            arguments: {},
            mcp: 'k8s',
            node_uri: 'viking://resources/arrange/status.md',
            tool: 'status',
            tool_uri: 'viking://agent/tools/mcp/k8s/status.mcp',
          },
        ],
      }),
    ).toHaveLength(1)
  })

  it('reports a connection error when Vite HTML is returned', () => {
    expect(() => parseMcpServers('<!doctype html><html></html>')).toThrow(
      /Studio HTML page/,
    )
    expect(() => parseMcpServers(undefined)).toThrow(/Invalid MCP server list/)
  })

  it('normalizes tool documents into selectable catalog entries', () => {
    expect(
      parseMcpTools('demo', {
        tools: [
          {
            revision: 'r1',
            tool: {
              description: 'Add two numbers',
              inputSchema: {
                properties: {
                  a: { type: 'number' },
                  b: { type: 'number' },
                },
                required: ['a', 'b'],
                type: 'object',
              },
              name: 'get-sum',
              title: 'Get sum',
            },
          },
        ],
      }),
    ).toEqual([
      expect.objectContaining({
        revision: 'r1',
        server: 'demo',
        uri: 'viking://agent/tools/demo/get-sum.json',
        tool: expect.objectContaining({
          name: 'get-sum',
          title: 'Get sum',
        }),
      }),
    ])
  })
})

describe('live context frontmatter detection', () => {
  it('parses mount fields without executing the document', () => {
    const content = `---
type: OpenViking Live Context
title: Cluster health
description: Current cluster state
x-openviking:
  type: mcp
  definition: viking://agent/tools/mcp/k8s/status.mcp
  timeout_seconds: 30
  args:
    cluster: production
---
`
    expect(isLiveContextDocument(content)).toBe(true)
    expect(parseLiveContextMetadata(content)).toMatchObject({
      arguments: { cluster: 'production' },
      description: 'Current cluster state',
      timeout_seconds: 30,
      title: 'Cluster health',
      tool_uri: 'viking://agent/tools/mcp/k8s/status.mcp',
    })
  })

  it('does not classify ordinary Markdown as live context', () => {
    expect(isLiveContextDocument('# Notes')).toBe(false)
    expect(
      isLiveContextDocument(`---
title: Notes
---
`),
    ).toBe(false)
  })

  it('fails closed for malformed live frontmatter', () => {
    expect(
      isLiveContextDocument(`---
type: OpenViking Live Context
broken: [
---
`),
    ).toBe(true)
  })
})
