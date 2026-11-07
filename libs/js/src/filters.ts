/** Translation of LangChain-style filters to a RushDB `where` (spec: Filters). */
import { isScalar } from './connection.js'

const COMPARISON = new Set(['$eq', '$ne', '$gt', '$gte', '$lt', '$lte'])
const MEMBERSHIP = new Set(['$in', '$nin'])
const STRING_MATCH = new Set(['$contains', '$startsWith', '$endsWith'])

export type Filter = Record<string, unknown>

/** A filter that cannot be translated; never silently ignored. */
export class FilterError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'FilterError'
  }
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/** Rename every field to `<prefix>.<field>` and validate the operators. */
export function translateFilter(filter: Filter | undefined | null, prefix: string): Record<string, unknown> {
  if (filter === undefined || filter === null) return {}
  if (!isPlainObject(filter)) throw new FilterError(`A filter is an object, got ${typeof filter}.`)
  const where: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(filter)) {
    if (key === '$and' || key === '$or') {
      if (!Array.isArray(value) || !value.length) {
        throw new FilterError(`${key} takes a non-empty list of filters.`)
      }
      where[key] = value.map((item) => translateFilter(item as Filter, prefix))
    } else if (key === '$not') {
      where[key] = translateFilter(value as Filter, prefix)
    } else if (key.startsWith('$')) {
      throw new FilterError(`Unsupported logic operator '${key}'.`)
    } else {
      where[`${prefix}.${key}`] = condition(key, value)
    }
  }
  return where
}

function condition(field: string, value: unknown): unknown {
  if (isScalar(value)) return value
  if (!isPlainObject(value) || !Object.keys(value).length) {
    throw new FilterError(
      `The condition on '${field}' must be a string, number, boolean or an operator object, ` +
        `got ${JSON.stringify(value)}.`
    )
  }
  const out: Record<string, unknown> = {}
  for (const [op, operand] of Object.entries(value)) {
    if (COMPARISON.has(op)) {
      if (!isScalar(operand)) throw new FilterError(`${op} on '${field}' takes a string, number or boolean.`)
    } else if (MEMBERSHIP.has(op)) {
      if (!Array.isArray(operand) || !operand.every(isScalar)) {
        throw new FilterError(`${op} on '${field}' takes a list of scalars.`)
      }
    } else if (op === '$exists') {
      if (typeof operand !== 'boolean') throw new FilterError(`$exists on '${field}' takes a boolean.`)
    } else if (STRING_MATCH.has(op)) {
      if (typeof operand !== 'string') throw new FilterError(`${op} on '${field}' takes a string.`)
    } else {
      throw new FilterError(`Unsupported operator '${op}' on '${field}'.`)
    }
    out[op] = operand
  }
  return out
}
