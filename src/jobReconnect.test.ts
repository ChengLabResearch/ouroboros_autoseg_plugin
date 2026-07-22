import { describe, expect, it, vi } from 'vitest';
import { findRunningJob } from './jobReconnect';

const running = {
    status: 'running',
    steps: [
        { name: 'Transferring', progress: 100 },
        { name: 'Inference', progress: 37 },
        { name: 'Saving', progress: 0 },
    ],
};

const waitingForHotstartCallbacks = {
    ...running,
    steps: [
        { name: 'Transferring', progress: 100 },
        { name: 'Inference', progress: 0 },
        { name: 'Saving', progress: 0 },
    ],
};

function response(body: unknown, ok = true) {
    return { ok, json: async () => body };
}

describe('findRunningJob', () => {
    it('discovers and restores a running job for a fresh client', async () => {
        const fetcher = vi.fn(async (url: string) => (
            url.endsWith('/latest-job')
                ? response({ job_id: 'remote-job' })
                : response(running)
        ));
        await expect(findRunningJob(fetcher, '/api', null)).resolves.toEqual({
            jobId: 'remote-job',
            record: running,
            discoveredFromLatest: true,
        });
        expect(fetcher).toHaveBeenCalledWith('/api/status/remote-job');
    });

    it('restores the latest committed progress during the initial hotstart delay', async () => {
        const hotstartDelay = 4;
        expect(hotstartDelay).toBeGreaterThan(0);
        const fetcher = vi.fn(async (url: string) => (
            url.endsWith('/latest-job')
                ? response({ job_id: 'delayed-job' })
                : response(waitingForHotstartCallbacks)
        ));

        await expect(findRunningJob(fetcher, '/api', null)).resolves.toEqual({
            jobId: 'delayed-job',
            record: waitingForHotstartCallbacks,
            discoveredFromLatest: true,
        });
        expect(fetcher).toHaveBeenCalledWith('/api/status/delayed-job');
    });

    it('falls back to latest-job when a persisted job is stale', async () => {
        const fetcher = vi.fn(async (url: string) => {
            if (url.endsWith('/latest-job')) return response({ job_id: 'active-job' });
            if (url.endsWith('/status/stale-job')) return response({}, false);
            return response(running);
        });
        const result = await findRunningJob(fetcher, '/api', 'stale-job');
        expect(result?.jobId).toBe('active-job');
    });

    it('does not reconnect terminal jobs', async () => {
        const fetcher = vi.fn(async (url: string) => (
            url.endsWith('/latest-job')
                ? response({ job_id: 'done-job' })
                : response({ ...running, status: 'completed' })
        ));
        await expect(findRunningJob(fetcher, '/api', null)).resolves.toBeNull();
    });
});
