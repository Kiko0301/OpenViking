import { describe, expect, it } from 'vitest'

import {
  argumentsToInputValues,
  initialSchemaArguments,
  schemaFields,
  validateSchemaArguments,
} from './schema'

const schema = {
  type: 'object',
  properties: {
    a: { type: 'number', description: 'First number' },
    b: { type: 'number', description: 'Second number' },
    note: { type: 'string', default: 'sum' },
  },
  required: ['a', 'b'],
} as const

describe('MCP input schema forms', () => {
  it('preserves field order and required state', () => {
    expect(
      schemaFields(schema).map(({ name, required }) => ({ name, required })),
    ).toEqual([
      { name: 'a', required: true },
      { name: 'b', required: true },
      { name: 'note', required: false },
    ])
  })

  it('applies defaults and converts values to inputs', () => {
    const values = initialSchemaArguments(schema, { a: 1 })
    expect(values).toEqual({ a: 1, note: 'sum' })
    expect(argumentsToInputValues(schema, values)).toEqual({
      a: '1',
      note: 'sum',
    })
  })

  it('validates required values and parses numeric arguments', () => {
    expect(validateSchemaArguments(schema, { a: '1.5', b: '2' })).toEqual({
      arguments: { a: 1.5, b: 2 },
      errors: {},
    })
    expect(validateSchemaArguments(schema, { a: '', b: 'bad' })).toEqual({
      arguments: {},
      errors: {
        a: 'required',
        b: 'Expected a number',
      },
    })
  })
})
