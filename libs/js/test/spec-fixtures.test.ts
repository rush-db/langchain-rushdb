/** The shared spec fixtures; the Python package runs the same cases. */
import { readFileSync } from 'node:fs'

import { describe, expect, it } from 'vitest'

import { mirror } from '../src/connection.js'
import { FilterError, translateFilter } from '../src/filters.js'
import { namespaceString, prefixWhere } from '../src/store.js'
import { type Similarity, relevanceFn } from '../src/vectorstores.js'

const load = (name: string) =>
  JSON.parse(readFileSync(new URL(`../../../spec/fixtures/${name}`, import.meta.url), 'utf8'))

const FILTERS = load('filters.json')
const MIRRORS = load('mirrors.json')
const NAMESPACES = load('namespaces.json')

describe('filters.json', () => {
  it.each(FILTERS.cases as any[])('$name', (c) => {
    expect(translateFilter(c.filter, FILTERS.prefix)).toEqual(c.where)
  })
  it.each(FILTERS.errors as any[])('error: $name', (c) => {
    expect(() => translateFilter(c.filter, FILTERS.prefix)).toThrow(FilterError)
  })
})

describe('mirrors.json', () => {
  it.each(MIRRORS.cases as any[])('$name', (c) => {
    expect(mirror(MIRRORS.prefix, c.object)).toEqual(c.properties)
  })
})

describe('namespaces.json', () => {
  it.each(NAMESPACES.cases as any[])('$ns', (c) => {
    expect(namespaceString(c.namespace)).toBe(c.ns)
  })
  it.each(NAMESPACES.prefixWhere as any[])('prefix $prefix', (c) => {
    expect(prefixWhere(c.prefix)).toEqual(c.where)
  })
  it.each(NAMESPACES.relevance as any[])('$similarity $score', (c) => {
    expect(relevanceFn(c.similarity as Similarity)(c.score)).toBeCloseTo(c.relevance)
  })
})
