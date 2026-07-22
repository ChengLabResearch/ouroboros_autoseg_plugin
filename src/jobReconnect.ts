import type { ProgressItem } from './components/ProgressPanel';

export type JobStatusRecord = {
    status: 'running' | 'completed' | 'error' | string;
    steps: ProgressItem[];
    active_phase?: string;
    error?: string;
};

export type ReconnectedJob = {
    jobId: string;
    record: JobStatusRecord;
    discoveredFromLatest: boolean;
};

type FetchResponse = {
    ok: boolean;
    json(): Promise<unknown>;
};

export async function findRunningJob(
    fetcher: (url: string) => Promise<FetchResponse>,
    backendUrl: string,
    storedJobId: string | null,
): Promise<ReconnectedJob | null> {
    const candidates: Array<{ jobId: string; discoveredFromLatest: boolean }> = [];
    if (storedJobId) candidates.push({ jobId: storedJobId, discoveredFromLatest: false });

    try {
        const latestResponse = await fetcher(`${backendUrl}/latest-job`);
        if (latestResponse.ok) {
            const latest = await latestResponse.json() as { job_id?: unknown };
            if (
                typeof latest.job_id === 'string'
                && latest.job_id.length > 0
                && latest.job_id !== storedJobId
            ) {
                candidates.push({ jobId: latest.job_id, discoveredFromLatest: true });
            }
        }
    } catch {
        // A stored job can still reconnect while latest-job is temporarily unavailable.
    }

    for (const candidate of candidates) {
        try {
            const statusResponse = await fetcher(`${backendUrl}/status/${candidate.jobId}`);
            if (!statusResponse.ok) continue;
            const record = await statusResponse.json() as JobStatusRecord;
            if (record.status === 'running' && Array.isArray(record.steps)) {
                return { ...candidate, record };
            }
        } catch {
            // Try the next candidate; polling will retry if neither is reachable.
        }
    }
    return null;
}
