export type ModelType = 'sam3' | 'medical_sam3';

export type ModelDownloadStatus = {
    state: 'idle' | 'connecting' | 'downloading' | 'finalizing' | 'available' | 'failed';
    bytes_received: number;
    total_bytes: number | null;
    error: string | null;
};

export type ModelStatusSnapshot = {
    downloads: Record<ModelType, ModelDownloadStatus>;
};

export type DownloadRequestFeedback = { pending: boolean; error: string | null };

export class ModelStatusError extends Error {
    constructor(public readonly kind: 'unavailable' | 'http' | 'invalid-response', message: string) {
        super(message);
        this.name = 'ModelStatusError';
    }
}

const modelTypes: ModelType[] = ['sam3', 'medical_sam3'];
const downloadStates = ['idle', 'connecting', 'downloading', 'finalizing', 'available', 'failed'];

export function isActiveDownload(status: ModelDownloadStatus): boolean {
    return status.state === 'connecting' || status.state === 'downloading' || status.state === 'finalizing';
}

function parseModelStatus(value: unknown): ModelStatusSnapshot {
    const downloads = (value as Partial<ModelStatusSnapshot> | null)?.downloads;
    if (!downloads || !modelTypes.every((model) => {
        const status = downloads[model];
        return status && downloadStates.includes(status.state)
            && Number.isFinite(status.bytes_received) && status.bytes_received >= 0
            && (status.total_bytes === null || (Number.isFinite(status.total_bytes) && status.total_bytes >= 0))
            && (status.error === null || typeof status.error === 'string');
    })) {
        throw new ModelStatusError('invalid-response',
            'The backend did not return model download progress. Check that the plugin and backend versions match.');
    }
    // Legacy availability booleans are deliberately not carried into the UI.
    return { downloads };
}

export function watchModelDownloads({
    fetcher,
    backendUrl,
    onStatus,
    onError,
    hasPendingRequest = () => false,
}: {
    fetcher: typeof fetch;
    backendUrl: string;
    onStatus: (snapshot: ModelStatusSnapshot) => void;
    onError: (error: ModelStatusError) => void;
    hasPendingRequest?: () => boolean;
}): { refresh: () => void; stop: () => void } {
    let stopped = false;
    let controller: AbortController | null = null;
    let timer: ReturnType<typeof setTimeout> | null = null;
    let deadline: ReturnType<typeof setTimeout> | null = null;
    let generation = 0;

    const poll = async (): Promise<void> => {
        if (stopped || controller) return;
        if (timer) clearTimeout(timer);
        timer = null;
        const requestGeneration = generation;
        const requestController = new AbortController();
        controller = requestController;
        let timedOut = false;
        deadline = setTimeout(() => {
            timedOut = true;
            requestController.abort();
        }, 10000);
        let delay = 1000;
        try {
            const response = await fetcher(`${backendUrl}/model-status`, { signal: requestController.signal });
            if (!response.ok) {
                // These responses can accompany startup or temporary unavailability.
                const unavailable = response.status === 408 || response.status === 429 || response.status >= 500;
                throw new ModelStatusError(unavailable ? 'unavailable' : 'http',
                    `Model status request failed (HTTP ${response.status}). ${unavailable ? 'Retrying…' : 'Check the backend address and logs.'}`);
            }
            let payload: unknown;
            try {
                payload = await response.json();
            } catch (error) {
                if (error instanceof SyntaxError) {
                    throw new ModelStatusError('invalid-response',
                        'The backend returned invalid JSON for model download progress. Check that the plugin and backend versions match.');
                }
                // A failed body transfer is still a connection problem.
                throw error;
            }
            const snapshot = parseModelStatus(payload);
            if (!stopped && requestGeneration === generation) {
                onStatus(snapshot);
                delay = Object.values(snapshot.downloads).some(isActiveDownload) ? 1000 : 5000;
            }
        } catch (error) {
            if (!stopped && requestGeneration === generation) {
                onError(timedOut
                    ? new ModelStatusError('unavailable', 'Model status request timed out. Retrying…')
                    : error instanceof ModelStatusError ? error
                        : new ModelStatusError('unavailable', error instanceof Error ? error.message : String(error)));
            }
        } finally {
            if (deadline) clearTimeout(deadline);
            deadline = null;
            controller = null;
            if (!stopped) {
                timer = setTimeout(() => { void poll(); },
                    requestGeneration !== generation ? 0 : hasPendingRequest() ? 1000 : delay);
            }
        }
    };

    void poll();
    return {
        refresh: () => {
            // An older response must not undo reconciliation after a POST settles.
            generation += 1;
            void poll();
        },
        stop: () => {
            stopped = true;
            if (timer) clearTimeout(timer);
            if (deadline) clearTimeout(deadline);
            controller?.abort();
        },
    };
}

export async function requestModelDownload(
    fetcher: typeof fetch,
    backendUrl: string,
    model: ModelType,
    token: string | undefined,
    signal: AbortSignal,
): Promise<void> {
    const response = await fetcher(`${backendUrl}/download-model`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_type: model, hf_token: token ?? '' }),
        signal,
    });
    if (!response.ok) {
        if (response.status === 409) {
            throw new Error('A download is already running. Refreshing its progress.');
        }
        const detail = await response.text().catch(() => '');
        throw new Error(`Download request failed (HTTP ${response.status})${detail ? `: ${detail.slice(0, 200)}` : '.'}`);
    }
    // A POST response is request feedback; only a status snapshot establishes availability.
}

export function formatDownloadBytes(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    const units = ['KiB', 'MiB', 'GiB', 'TiB'];
    let value = bytes / 1024;
    let unit = 0;
    while (value >= 1024 && unit < units.length - 1) {
        value /= 1024;
        unit += 1;
    }
    return `${value.toFixed(1)} ${units[unit]}`;
}
