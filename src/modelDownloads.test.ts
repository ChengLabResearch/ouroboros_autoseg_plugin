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

describe('model download status polling', () => {
    beforeEach(() => vi.useFakeTimers());
    afterEach(() => vi.useRealTimers());

    it('polls immediately, ignores contradictory legacy booleans, and waits five seconds while idle', async () => {
        const fetcher = vi.fn<typeof fetch>().mockImplementation(async () => response());
        const onStatus = vi.fn();
        const watcher = watchModelDownloads({ fetcher, backendUrl: '/api', onStatus, onError: vi.fn() });
        await vi.advanceTimersByTimeAsync(0);
        expect(fetcher).toHaveBeenCalledTimes(1);
        expect(onStatus.mock.calls[0][0]).toEqual({ downloads: body().downloads });
        expect(onStatus.mock.calls[0][0]).not.toHaveProperty('models');
        await vi.advanceTimersByTimeAsync(4999);
        expect(fetcher).toHaveBeenCalledTimes(1);
        await vi.advanceTimersByTimeAsync(1);
        expect(fetcher).toHaveBeenCalledTimes(2);
        watcher.stop();
    });

    it('keeps active downloads authoritative across the former five-second reset', async () => {
        const active = download('downloading', { bytes_received: 25, total_bytes: 100 });
        const fetcher = vi.fn<typeof fetch>().mockImplementation(async () => response(active));
        const onStatus = vi.fn();
        const watcher = watchModelDownloads({ fetcher, backendUrl: '/api', onStatus, onError: vi.fn() });
        await vi.advanceTimersByTimeAsync(5100);
        expect(fetcher).toHaveBeenCalledTimes(6);
        expect(onStatus.mock.calls.every(([snapshot]) => snapshot.downloads.medical_sam3.state === 'downloading')).toBe(true);
        watcher.stop();
    });

    it('polls each second while a POST is pending even before the server reports active work', async () => {
        const fetcher = vi.fn<typeof fetch>().mockImplementation(async () => response());
        const watcher = watchModelDownloads({
            fetcher, backendUrl: '/api', onStatus: vi.fn(), onError: vi.fn(), hasPendingRequest: () => true,
        });
        await vi.advanceTimersByTimeAsync(1000);
        expect(fetcher).toHaveBeenCalledTimes(2);
        watcher.stop();
    });

    it('serializes refreshes and rejects an older idle response after POST reconciliation starts', async () => {
        const first = deferred<Response>();
        const fetcher = vi.fn<typeof fetch>()
            .mockReturnValueOnce(first.promise)
            .mockImplementation(async () => response(download('downloading')));
        const onStatus = vi.fn();
        const watcher = watchModelDownloads({ fetcher, backendUrl: '/api', onStatus, onError: vi.fn() });
        watcher.refresh();
        watcher.refresh();
        expect(fetcher).toHaveBeenCalledTimes(1);
        first.resolve(response());
        await vi.advanceTimersByTimeAsync(0);
        expect(fetcher).toHaveBeenCalledTimes(2);
        expect(onStatus).toHaveBeenCalledTimes(1);
        expect(onStatus.mock.calls[0][0].downloads.medical_sam3.state).toBe('downloading');
        watcher.stop();
    });

    it('preserves the last snapshot on contact loss and restores current progress on recovery', async () => {
        const active = download('downloading', { bytes_received: 64 });
        const fetcher = vi.fn<typeof fetch>()
            .mockResolvedValueOnce(response(active))
            .mockRejectedValueOnce(new Error('Connection lost'))
            .mockImplementation(async () => response(download('available')));
        let snapshot: ModelStatusSnapshot | null = null;
        const onError = vi.fn();
        const watcher = watchModelDownloads({
            fetcher, backendUrl: '/api', onStatus: (value) => { snapshot = value; }, onError,
        });
        await vi.advanceTimersByTimeAsync(1000);
        expect(snapshot).toEqual({ downloads: body(active).downloads });
        expect(onError).toHaveBeenCalledWith(expect.objectContaining({ kind: 'unavailable', message: 'Connection lost' }));
        await vi.advanceTimersByTimeAsync(1000);
        expect(snapshot).toEqual({ downloads: body(download('available')).downloads });
        watcher.stop();
    });

    it('times out a stalled status request, reports stale contact, and resumes polling', async () => {
        const fetcher = vi.fn<typeof fetch>()
            .mockResolvedValueOnce(response(download('downloading')))
            .mockImplementationOnce(async (_url, options) => new Promise((_resolve, reject) => {
                options?.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
            }))
            .mockImplementation(async () => response(download('available')));
        const onStatus = vi.fn();
        const onError = vi.fn();
        const watcher = watchModelDownloads({ fetcher, backendUrl: '/api', onStatus, onError });
        await vi.advanceTimersByTimeAsync(11000);
        expect(onError).toHaveBeenCalledWith(expect.objectContaining({
            kind: 'unavailable', message: 'Model status request timed out. Retrying…',
        }));
        expect(onStatus).toHaveBeenCalledTimes(1);
        await vi.advanceTimersByTimeAsync(1000);
        expect(onStatus.mock.calls[onStatus.mock.calls.length - 1]?.[0].downloads.medical_sam3.state).toBe('available');
        watcher.stop();
    });

    it('ignores a late response after unmount even when the transport ignores abort', async () => {
        const pending = deferred<Response>();
        const fetcher = vi.fn<typeof fetch>().mockReturnValue(pending.promise);
        const onStatus = vi.fn();
        const onError = vi.fn();
        const watcher = watchModelDownloads({ fetcher, backendUrl: '/api', onStatus, onError });
        watcher.stop();
        expect(fetcher.mock.calls[0][1]?.signal?.aborted).toBe(true);
        pending.resolve(response(download('available')));
        await vi.advanceTimersByTimeAsync(20000);
        expect(onStatus).not.toHaveBeenCalled();
        expect(onError).not.toHaveBeenCalled();
        expect(fetcher).toHaveBeenCalledTimes(1);
    });

    it('restores backend progress on remount without accepting the previous connection response', async () => {
        const oldRequest = deferred<Response>();
        const fetcher = vi.fn<typeof fetch>()
            .mockReturnValueOnce(oldRequest.promise)
            .mockImplementation(async () => response(download('finalizing', { bytes_received: 100, total_bytes: 100 })));
        const onStatus = vi.fn();
        const options = { fetcher, backendUrl: '/api', onStatus, onError: vi.fn() };
        watchModelDownloads(options).stop();
        const current = watchModelDownloads(options);
        await vi.advanceTimersByTimeAsync(0);
        oldRequest.resolve(response());
        await vi.advanceTimersByTimeAsync(0);
        expect(onStatus).toHaveBeenCalledTimes(1);
        expect(onStatus.mock.calls[0][0].downloads.medical_sam3.state).toBe('finalizing');
        current.stop();
    });

    it.each([
        ['legacy-only schema', JSON.stringify({ models: { sam3: true } })],
        ['invalid download state', JSON.stringify(body(download('unknown' as ModelDownloadStatus['state'])))],
        ['malformed JSON', '<html>Wrong endpoint</html>'],
    ])('classifies an initial %s response as invalid and recovers on a valid snapshot', async (_name, payload) => {
        const fetcher = vi.fn<typeof fetch>()
            .mockResolvedValueOnce(new Response(payload))
            .mockImplementation(async () => response());
        const onStatus = vi.fn();
        const onError = vi.fn();
        const watcher = watchModelDownloads({ fetcher, backendUrl: '/api', onStatus, onError });
        await vi.advanceTimersByTimeAsync(0);
        expect(onStatus).not.toHaveBeenCalled();
        expect(onError.mock.calls[0][0]).toBeInstanceOf(ModelStatusError);
        expect(onError.mock.calls[0][0]).toMatchObject({
            kind: 'invalid-response', message: expect.stringContaining('Check that the plugin and backend versions match.'),
        });
        await vi.advanceTimersByTimeAsync(1000);
        expect(onStatus).toHaveBeenCalledExactlyOnceWith({ downloads: body().downloads });
        expect(onError).toHaveBeenCalledTimes(1);
        watcher.stop();
    });

    it.each([
        [401, 'http'],
        [404, 'http'],
        [408, 'unavailable'],
        [429, 'unavailable'],
        [503, 'unavailable'],
    ] as const)('classifies HTTP %i as %s without inventing a snapshot', async (status, kind) => {
        const fetcher = vi.fn<typeof fetch>().mockResolvedValue(new Response('', { status }));
        const onStatus = vi.fn();
        const onError = vi.fn();
        const watcher = watchModelDownloads({ fetcher, backendUrl: '/api', onStatus, onError });
        await vi.advanceTimersByTimeAsync(0);
        expect(onStatus).not.toHaveBeenCalled();
        expect(onError).toHaveBeenCalledWith(expect.objectContaining({
            kind,
            message: `Model status request failed (HTTP ${status}). ${kind === 'unavailable' ? 'Retrying…' : 'Check the backend address and logs.'}`,
        }));
        watcher.stop();
    });

    it.each(['connect', 'read body'])('classifies an initial failure to %s as unavailable and resumes polling', async (stage) => {
        const fetcher = vi.fn<typeof fetch>().mockImplementationOnce(async () => {
            if (stage === 'connect') throw new TypeError('Failed to fetch');
            return new Response(new ReadableStream({
                start(controller) { controller.error(new TypeError('Body transfer failed')); },
            }));
        }).mockImplementation(async () => response());
        const onStatus = vi.fn();
        const onError = vi.fn();
        const watcher = watchModelDownloads({ fetcher, backendUrl: '/api', onStatus, onError });
        await vi.advanceTimersByTimeAsync(0);
        expect(onStatus).not.toHaveBeenCalled();
        expect(onError.mock.calls[0][0]).toMatchObject({ kind: 'unavailable' });
        await vi.advanceTimersByTimeAsync(1000);
        expect(onStatus).toHaveBeenCalledExactlyOnceWith({ downloads: body().downloads });
        watcher.stop();
    });

    it('continues backend progress after a lost POST response without inventing a download failure', async () => {
        let received = 42;
        const fetcher = vi.fn<typeof fetch>().mockImplementation(async (url) => {
            if (String(url).endsWith('/download-model')) throw new Error('Response lost');
            return response(download('downloading', { bytes_received: received }));
        });
        const onStatus = vi.fn();
        const onError = vi.fn();
        const watcher = watchModelDownloads({ fetcher, backendUrl: '/api', onStatus, onError });
        await vi.advanceTimersByTimeAsync(0);
        await expect(requestModelDownload(fetcher, '/api', 'medical_sam3', undefined, new AbortController().signal))
            .rejects.toThrow('Response lost');
        received = 84;
        watcher.refresh();
        await vi.advanceTimersByTimeAsync(0);
        expect(onStatus.mock.calls.map(([snapshot]) => snapshot.downloads.medical_sam3)).toEqual([
            download('downloading', { bytes_received: 42 }),
            download('downloading', { bytes_received: 84 }),
        ]);
        expect(onError).not.toHaveBeenCalled();
        watcher.stop();
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
