import { describe, expect, it } from 'vitest'

import {
  assertMutationBoundary,
  isWithinRoot,
  joinVikingUri,
  normalizeVikingUri,
} from './uri'

describe('arranger Viking URI boundaries', () => {
  it('normalizes trailing separators and rejects relative segments', () => {
    expect(normalizeVikingUri(' viking://resources/team/ ')).toBe(
      'viking://resources/team',
    )
    expect(() => normalizeVikingUri('viking://resources/../secret')).toThrow()
  })

  it('compares path segments instead of string prefixes', () => {
    expect(
      isWithinRoot(
        'viking://resources/arrange/devops',
        'viking://resources/arrange',
      ),
    ).toBe(true)
    expect(
      isWithinRoot('viking://resources/arranger', 'viking://resources/arrange'),
    ).toBe(false)
  })

  it('keeps targets in the root and out of the source subtree', () => {
    expect(() =>
      assertMutationBoundary(
        'viking://resources/arrange/a',
        'viking://resources/arrange/a/child',
        'viking://resources/arrange',
      ),
    ).toThrow()
    expect(() =>
      assertMutationBoundary(
        'viking://resources/arrange/a',
        'viking://resources/other/a',
        'viking://resources/arrange',
      ),
    ).toThrow()
    expect(joinVikingUri('viking://resources/arrange', 'notes.md')).toBe(
      'viking://resources/arrange/notes.md',
    )
  })
})
