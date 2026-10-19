import type { JsonSchema } from './model'

export type SchemaField = {
  name: string
  schema: JsonSchema
  required: boolean
}

export function schemaFields(schema: JsonSchema): SchemaField[] {
  const required = new Set(schema.required || [])
  return Object.entries(schema.properties || {}).map(([name, property]) => ({
    name,
    schema: property,
    required: required.has(name),
  }))
}

export function resolveSchema(schema: JsonSchema): JsonSchema {
  const alternatives = schema.anyOf || schema.oneOf
  if (!alternatives) return schema
  const nonNull = alternatives.filter((candidate) => {
    const type = candidate.type
    return type !== 'null' && !(Array.isArray(type) && type.includes('null'))
  })
  if (nonNull.length !== 1) return schema
  const parent = { ...schema }
  delete parent.anyOf
  delete parent.oneOf
  return { ...nonNull[0], ...parent }
}

export function schemaType(schema: JsonSchema): string {
  const resolved = resolveSchema(schema)
  if (Array.isArray(resolved.type)) {
    return resolved.type.find((type) => type !== 'null') || 'string'
  }
  if (resolved.type) return resolved.type
  if (resolved.properties) return 'object'
  if (resolved.items) return 'array'
  return 'string'
}

export function initialSchemaArguments(
  schema: JsonSchema,
  current: Record<string, unknown> = {},
): Record<string, unknown> {
  const result = { ...current }
  for (const field of schemaFields(schema)) {
    if (field.name in result || field.schema.default === undefined) continue
    result[field.name] = field.schema.default
  }
  return result
}

export function parseSchemaInput(raw: string, schema: JsonSchema): unknown {
  const resolved = resolveSchema(schema)
  const type = schemaType(resolved)
  if (type === 'number' || type === 'integer') {
    const value = Number(raw)
    if (!Number.isFinite(value)) throw new Error('Expected a number')
    if (type === 'integer' && !Number.isInteger(value)) {
      throw new Error('Expected an integer')
    }
    return value
  }
  if (type === 'boolean') {
    if (raw === 'true') return true
    if (raw === 'false') return false
    throw new Error('Expected true or false')
  }
  if (type === 'object' || type === 'array') {
    const value = JSON.parse(raw) as unknown
    if (type === 'array' && !Array.isArray(value)) {
      throw new Error('Expected a JSON array')
    }
    if (
      type === 'object' &&
      (!value || typeof value !== 'object' || Array.isArray(value))
    ) {
      throw new Error('Expected a JSON object')
    }
    return value
  }
  return raw
}

export function validateSchemaArguments(
  schema: JsonSchema,
  values: Record<string, string>,
): {
  arguments: Record<string, unknown>
  errors: Record<string, string>
} {
  const result: Record<string, unknown> = {}
  const errors: Record<string, string> = {}
  for (const field of schemaFields(schema)) {
    const raw = values[field.name]
    if (!(field.name in values) || raw === '') {
      if (field.required) errors[field.name] = 'required'
      continue
    }
    try {
      result[field.name] = parseSchemaInput(raw, field.schema)
    } catch (error) {
      errors[field.name] =
        error instanceof Error ? error.message : String(error)
    }
  }
  return { arguments: result, errors }
}

export function argumentsToInputValues(
  schema: JsonSchema,
  values: Record<string, unknown>,
): Record<string, string> {
  const result: Record<string, string> = {}
  for (const field of schemaFields(schema)) {
    const value = values[field.name]
    if (value === undefined || value === null) continue
    const type = schemaType(field.schema)
    result[field.name] =
      type === 'object' || type === 'array'
        ? JSON.stringify(value, null, 2)
        : String(value)
  }
  return result
}
