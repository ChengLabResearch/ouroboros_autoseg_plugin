import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ModelStatusError, requestModelDownload, watchModelDownloads } from './modelDownloads';
import type { ModelDownloadStatus, ModelStatusSnapshot } from './modelDownloads';

function download(state: ModelDownloadStatus['state'], overrides: Partial<ModelDownloadStatus> = {}): ModelDownloadStatus {
    return { state, bytes_received: 0, total_bytes: null, error: null, ...overrides };
}

function body(status = download('idle')) {
    return {
        checkpoint_dir: '/models',
        models: { sam3: true, medical_sam3: true },
        downloads: { sam3: download('idle'), medical_sam3: status },
    };
}

function response(status = download('idle')): Response {
    return new Response(JSON.stringify(body(status)));
}

function deferred<T>() {
    let resolve!: (value: T) => void;
    const promise = new Promise<T>((done) => { resolve = done; });
    return { promise, resolve };
}

const watchers: ReturnType<typeof watchModelDownloads>[] = [];

function watch(fetcher: typeof fetch, options: Pick<Parameters<typeof watchModelDownloads>[0], 'hasPendingRequest'> = {}) {
    const onStatus = vi.fn<(snapshot: ModelStatusSnapshot) => void>();
    const onError = vi.fn<(error: ModelStatusError) => void>();
    const watcher = watchModelDownloads({ fetcher, backendUrl: '/api', onStatus, onError, ...options });
    watchers.push(watcher);
    return { ...watcher, onStatus, onError };
}

describe('model download status polling', () => {
    beforeEach(() => vi.useFakeTimers());
    afterEach(() => {
        for (const watcher of watchers.splice(0)) watcher.stop();
        vi.useRealTimers();
    });

    it('polls immediately, ignores contradictory legacy booleans, and waits five seconds while idle', async () => {
        const fetcher = vi.fn<typeof fetch>().mockImplementation(async () => response());
        const { onStatus } = watch(fetcher);
        await vi.advanceTimersByTimeAsync(0);
        expect(fetcher).toHaveBeenCalledTimes(1);
        expect(onStatus.mock.calls[0][0]).toEqual({ downloads: body().downloads });
        expect(onStatus.mock.calls[0][0]).not.toHaveProperty('models');
        await vi.advanceTimersByTimeAsync(4999);
        expect(fetcher).toHaveBeenCalledTimes(1);
        await vi.advanceTimersByTimeAsync(1);
        expect(fetcher).toHaveBeenCalledTimes(2);
    });

    it.each([
        ['downloading', false],
        ['idle', true],
    ] as const)('polls each second while %s with POST pending=%s, without resetting state', async (state, pending) => {
        const fetcher = vi.fn<typeof fetch>().mockImplementation(async () => response(download(state)));
        const { onStatus } = watch(fetcher, { hasPendingRequest: () => pending });
        await vi.advanceTimersByTimeAsync(5100);
        expect(fetcher).toHaveBeenCalledTimes(6);
        expect(onStatus.mock.calls.every(([snapshot]) => snapshot.downloads.medical_sam3.state === state)).toBe(true);
    });

    it('serializes refreshes and rejects an older idle response after POST reconciliation starts', async () => {
        const first = deferred<Response>();
        const fetcher = vi.fn<typeof fetch>()
            .mockReturnValueOnce(first.promise)
            .mockImplementation(async () => response(download('downloading')));
        const watcher = watch(fetcher);
        watcher.refresh();
        watcher.refresh();
        expect(fetcher).toHaveBeenCalledTimes(1);
        first.resolve(response());
        await vi.advanceTimersByTimeAsync(0);
        expect(fetcher).toHaveBeenCalledTimes(2);
        expect(watcher.onStatus).toHaveBeenCalledTimes(1);
        expect(watcher.onStatus.mock.calls[0][0].downloads.medical_sam3.state).toBe('downloading');
    });

    it('preserves the last snapshot on contact loss and restores current progress on recovery', async () => {
        const active = download('downloading', { bytes_received: 64 });
        const fetcher = vi.fn<typeof fetch>()
            .mockResolvedValueOnce(response(active))
            .mockRejectedValueOnce(new Error('Connection lost'))
            .mockImplementation(async () => response(download('available')));
        const { onStatus, onError } = watch(fetcher);
        await vi.advanceTimersByTimeAsync(1000);
        expect(onStatus).toHaveBeenCalledExactlyOnceWith({ downloads: body(active).downloads });
        expect(onError).toHaveBeenCalledWith(expect.objectContaining({ kind: 'unavailable', message: 'Connection lost' }));
        await vi.advanceTimersByTimeAsync(1000);
        expect(onStatus).toHaveBeenLastCalledWith({ downloads: body(download('available')).downloads });
    });

    it('times out a stalled status request, reports stale contact, and resumes polling', async () => {
        const fetcher = vi.fn<typeof fetch>()
            .mockResolvedValueOnce(response(download('downloading')))
            .mockImplementationOnce(async (_url, options) => new Promise((_resolve, reject) => {
                options?.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
            }))
            .mockImplementation(async () => response(download('available')));
        const { onStatus, onError } = watch(fetcher);
        await vi.advanceTimersByTimeAsync(11000);
        expect(onError).toHaveBeenCalledWith(expect.objectContaining({
            kind: 'unavailable', message: 'Model status request timed out. Retrying…',
        }));
        expect(onStatus).toHaveBeenCalledTimes(1);
        await vi.advanceTimersByTimeAsync(1000);
        expect(onStatus.mock.calls[onStatus.mock.calls.length - 1]?.[0].downloads.medical_sam3.state).toBe('available');
    });

    it('aborts on unmount and restores progress on remount, ignoring a late response', async () => {
        const oldRequest = deferred<Response>();
        const fetcher = vi.fn<typeof fetch>()
            .mockReturnValueOnce(oldRequest.promise)
            .mockImplementation(async () => response(download('finalizing', { bytes_received: 100, total_bytes: 100 })));
        const previous = watch(fetcher);
        previous.stop();
        expect(fetcher.mock.calls[0][1]?.signal?.aborted).toBe(true);
        const current = watch(fetcher);
        await vi.advanceTimersByTimeAsync(0);
        oldRequest.resolve(response());
        await vi.advanceTimersByTimeAsync(0);
        expect(previous.onStatus).not.toHaveBeenCalled();
        expect(previous.onError).not.toHaveBeenCalled();
        expect(current.onStatus).toHaveBeenCalledTimes(1);
        expect(current.onStatus.mock.calls[0][0].downloads.medical_sam3.state).toBe('finalizing');
        current.stop();
        await vi.advanceTimersByTimeAsync(20000);
        expect(fetcher).toHaveBeenCalledTimes(2);
    });

    it.each([
        ['legacy-only schema', JSON.stringify({ models: { sam3: true } })],
        ['invalid download state', JSON.stringify(body(download('unknown' as ModelDownloadStatus['state'])))],
        ['malformed JSON', '<html>Wrong endpoint</html>'],
    ])('classifies an initial %s response as invalid and recovers on a valid snapshot', async (_name, payload) => {
        const fetcher = vi.fn<typeof fetch>()
            .mockResolvedValueOnce(new Response(payload))
            .mockImplementation(async () => response());
        const { onStatus, onError } = watch(fetcher);
        await vi.advanceTimersByTimeAsync(0);
        expect(onStatus).not.toHaveBeenCalled();
        expect(onError.mock.calls[0][0]).toBeInstanceOf(ModelStatusError);
        expect(onError.mock.calls[0][0]).toMatchObject({
            kind: 'invalid-response', message: expect.stringContaining('Check that the plugin and backend versions match.'),
        });
        await vi.advanceTimersByTimeAsync(1000);
        expect(onStatus).toHaveBeenCalledExactlyOnceWith({ downloads: body().downloads });
        expect(onError).toHaveBeenCalledTimes(1);
    });

    it.each([
        [401, 'http'],
        [404, 'http'],
        [408, 'unavailable'],
        [429, 'unavailable'],
        [503, 'unavailable'],
    ] as const)('classifies HTTP %i as %s without inventing a snapshot', async (status, kind) => {
        const fetcher = vi.fn<typeof fetch>().mockResolvedValue(new Response('', { status }));
        const { onStatus, onError } = watch(fetcher);
        await vi.advanceTimersByTimeAsync(0);
        expect(onStatus).not.toHaveBeenCalled();
        expect(onError).toHaveBeenCalledWith(expect.objectContaining({
            kind,
            message: `Model status request failed (HTTP ${status}). ${kind === 'unavailable' ? 'Retrying…' : 'Check the backend address and logs.'}`,
        }));
    });

    it.each(['connect', 'read body'])('classifies an initial failure to %s as unavailable and resumes polling', async (stage) => {
        const fetcher = vi.fn<typeof fetch>().mockImplementationOnce(async () => {
            if (stage === 'connect') throw new TypeError('Failed to fetch');
            return new Response(new ReadableStream({
                start(controller) { controller.error(new TypeError('Body transfer failed')); },
            }));
        }).mockImplementation(async () => response());
        const { onStatus, onError } = watch(fetcher);
        await vi.advanceTimersByTimeAsync(0);
        expect(onStatus).not.toHaveBeenCalled();
        expect(onError.mock.calls[0][0]).toMatchObject({ kind: 'unavailable' });
        await vi.advanceTimersByTimeAsync(1000);
        expect(onStatus).toHaveBeenCalledExactlyOnceWith({ downloads: body().downloads });
    });

    it('continues backend progress after a lost POST response without inventing a download failure', async () => {
        let received = 42;
        const fetcher = vi.fn<typeof fetch>().mockImplementation(async (url) => {
            if (String(url).endsWith('/download-model')) throw new Error('Response lost');
            return response(download('downloading', { bytes_received: received }));
        });
        const watcher = watch(fetcher);
        await vi.advanceTimersByTimeAsync(0);
        await expect(requestModelDownload(fetcher, '/api', 'medical_sam3', undefined, new AbortController().signal))
            .rejects.toThrow('Response lost');
        received = 84;
        watcher.refresh();
        await vi.advanceTimersByTimeAsync(0);
        expect(watcher.onStatus.mock.calls.map(([snapshot]) => snapshot.downloads.medical_sam3)).toEqual([
            download('downloading', { bytes_received: 42 }),
            download('downloading', { bytes_received: 84 }),
        ]);
        expect(watcher.onError).not.toHaveBeenCalled();
    });
});

describe('model download request feedback', () => {
    it('sends credentials only in the POST body and does not derive availability from success', async () => {
        const fetcher = vi.fn<typeof fetch>().mockResolvedValue(new Response('{"status":"success"}'));
        await expect(requestModelDownload(fetcher, '/api', 'sam3', 'secret-token', new AbortController().signal)).resolves.toBeUndefined();
        expect(fetcher.mock.calls[0][0]).toBe('/api/download-model');
        expect(JSON.parse(String(fetcher.mock.calls[0][1]?.body))).toEqual({ model_type: 'sam3', hf_token: 'secret-token' });
    });

    it('reports an already-running request as feedback to reconcile via status', async () => {
        const fetcher = vi.fn<typeof fetch>().mockResolvedValue(new Response('', { status: 409 }));
        await expect(requestModelDownload(fetcher, '/api', 'medical_sam3', undefined, new AbortController().signal))
            .rejects.toThrow('A download is already running. Refreshing its progress.');
    });
});
