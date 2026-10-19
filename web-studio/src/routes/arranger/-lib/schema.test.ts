import { describe, expect, it } from 'vitest'

import {
  argumentsToInputValues,
  initialSchemaArguments,
  resolveSchema,
  schemaFields,
  schemaType,
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

  it('parses nullable anyOf values using their non-null type', () => {
    const nullableSchema = {
      properties: {
        limit: {
          anyOf: [{ type: 'integer' }, { type: 'null' }],
          default: 10,
        },
        start_time: {
          anyOf: [{ type: 'integer' }, { type: 'null' }],
          default: null,
        },
      },
      type: 'object',
    } as const

    expect(schemaType(nullableSchema.properties.start_time)).toBe('integer')
    expect(resolveSchema(nullableSchema.properties.start_time)).toEqual({
      default: null,
      type: 'integer',
    })
    expect(
      validateSchemaArguments(nullableSchema, {
        limit: '10',
        start_time: '1792310712773',
      }),
    ).toEqual({
      arguments: {
        limit: 10,
        start_time: 1792310712773,
      },
      errors: {},
    })
  })
})
