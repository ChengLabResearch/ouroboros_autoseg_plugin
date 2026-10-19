import { useEffect, useRef, useState } from 'react';
import styles from '../assets/styles.module.css';
import { BACKEND_URL } from '../config';
import { requestModelDownload, watchModelDownloads } from '../modelDownloads';
import type { ModelStatusSnapshot, ModelType } from '../modelDownloads';
import ModelDownload from './ModelDownload';

export default function ModelsPanel({ connected }: { connected: boolean }) {
    const [token, setToken] = useState('');
    const [snapshot, setSnapshot] = useState<ModelStatusSnapshot | null>(null);
    const [fresh, setFresh] = useState(false);
    const [statusError, setStatusError] = useState<string | null>(null);
    const [requestErrors, setRequestErrors] = useState<Record<ModelType, string | null>>({
        sam3: null,
        medical_sam3: null,
    });
    const polling = useRef<ReturnType<typeof watchModelDownloads> | null>(null);
    const requestControllers = useRef(new Map<ModelType, AbortController>());

    useEffect(() => {
        setFresh(false);
        // Inference readiness can be false until a model exists. Model downloads
        // use their own endpoint's reachability, including during initial setup.
        const watcher = watchModelDownloads({
            fetcher: fetch,
            backendUrl: BACKEND_URL,
            hasPendingRequest: () => requestControllers.current.size > 0,
            onStatus: (status) => {
                setSnapshot(status);
                setFresh(true);
                setStatusError(null);
                setRequestErrors((previous) => ({
                    sam3: status.downloads.sam3.state === 'idle' ? previous.sam3 : null,
                    medical_sam3: status.downloads.medical_sam3.state === 'idle' ? previous.medical_sam3 : null,
                }));
            },
            onError: (error) => {
                setFresh(false);
                setStatusError(error);
            },
        });
        polling.current = watcher;
        return () => {
            watcher.stop();
            polling.current = null;
        };
    }, [connected]);

    useEffect(() => {
        const controllers = requestControllers.current;
        return () => {
            for (const controller of controllers.values()) controller.abort();
            controllers.clear();
        };
    }, []);

    const downloadModel = async (model: ModelType, hfToken?: string): Promise<void> => {
        if (requestControllers.current.has(model)) return;
        const controller = new AbortController();
        requestControllers.current.set(model, controller);
        setRequestErrors((previous) => ({ ...previous, [model]: null }));
        polling.current?.refresh();
        let error: string | null = null;
        try {
            await requestModelDownload(fetch, BACKEND_URL, model, hfToken, controller.signal);
        } catch (cause) {
            error = `${cause instanceof Error ? cause.message : String(cause)} Checking the backend for the current download status.`;
        } finally {
            if (requestControllers.current.get(model) === controller) {
                requestControllers.current.delete(model);
                setRequestErrors((previous) => ({ ...previous, [model]: error }));
                // Even a lost response can leave the backend worker running.
                // Reconcile before an older idle snapshot can enable a retry.
                setFresh(false);
                polling.current?.refresh();
            }
        }
    };

    return (
        <div className={styles.container}>
            <div className={styles.headerRow}>
                <span className={styles.headerTitle}>MODELS</span>
            </div>
            <div className={styles.scrollContent}>
                <div className={styles.section}>
                    {statusError && (
                        <div className={styles.inlineError} role="alert">
                            Unable to refresh model downloads. {snapshot && 'Last reported progress is shown. '}{statusError}
                        </div>
                    )}
                    <ModelDownload
                        label="SAM3 (Official; Authenticated)"
                        status={snapshot?.downloads.sam3}
                        fresh={fresh}
                        request={{ pending: requestControllers.current.has('sam3'), error: requestErrors.sam3 }}
                        requiresToken
                        token={token}
                        onDownload={() => { void downloadModel('sam3', token); }}
                    >
                        <input
                            className={styles.tokenInput}
                            type="password"
                            value={token}
                            onChange={(event) => setToken(event.target.value)}
                            placeholder="Paste HF Token"
                            aria-label="Hugging Face token for official SAM3"
                        />
                    </ModelDownload>
                    <ModelDownload
                        label="SAM3 (Medical)"
                        status={snapshot?.downloads.medical_sam3}
                        fresh={fresh}
                        request={{ pending: requestControllers.current.has('medical_sam3'), error: requestErrors.medical_sam3 }}
                        onDownload={() => { void downloadModel('medical_sam3'); }}
                    />
                </div>
            </div>
        </div>
    );
}
