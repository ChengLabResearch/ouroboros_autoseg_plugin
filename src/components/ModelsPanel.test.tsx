import { useEffect } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { afterEach, describe, expect, it, vi } from 'vitest';
import ModelsPanel from './ModelsPanel';

vi.mock('react', async (importOriginal) => {
    const actual = await importOriginal<typeof import('react')>();
    return { ...actual, useEffect: vi.fn() };
});

afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
});

describe('model setup before inference readiness', () => {
    it('mounts model-status polling when the inference backend is not yet ready', async () => {
        const effects: Array<Parameters<typeof useEffect>[0]> = [];
        vi.mocked(useEffect).mockImplementation((effect) => { effects.push(effect); });
        const fetcher = vi.fn<typeof fetch>().mockImplementation(async (_url, options) => new Promise((_resolve, reject) => {
            options?.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
        }));
        vi.stubGlobal('fetch', fetcher);

        // SSR supplies React's real state/ref hooks. Run its captured mount effects
        // explicitly to test initial setup without adding a DOM test dependency.
        renderToStaticMarkup(<ModelsPanel connected={false} />);
        const cleanups = effects.map((effect) => effect());
        expect(fetcher).toHaveBeenCalledTimes(1);
        expect(String(fetcher.mock.calls[0][0])).toMatch(/\/model-status$/);
        for (const cleanup of cleanups) cleanup?.();
        await Promise.resolve();
        expect(fetcher.mock.calls[0][1]?.signal?.aborted).toBe(true);
    });
});
