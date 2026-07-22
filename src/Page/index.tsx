import { useState, useEffect, useRef } from 'react';
import styles from '../assets/styles.module.css';
import OptionsPanel from '../components/OptionsPanel';
import ProgressPanel, { BackendStatus, ErrorEntry, ProgressItem, VolumeServerState } from '../components/ProgressPanel';
import ModelsPanel from '../components/ModelsPanel';
import { BACKEND_URL } from '../config';
import { findRunningJob } from '../jobReconnect';

const MAX_ERRORS = 5;
const DEDUP_WINDOW_MS = 5000;
const VOLUME_SERVER_URL = 'http://localhost:3001';

type RunOptions = {
    filePath: string;
    outputFile: string;
    modelType: string;
    predictor_type: string;
    overlayAnnotationPoints: boolean;
};

export default function SAM3Page() {
    const [prog, setProg] = useState<ProgressItem[]>([]);
    const [run, setRun] = useState(false);
    const [jobId, setJobId] = useState<string | null>(null);
    const [connected, setConnected] = useState(false);
    const [backendStatus, setBackendStatus] = useState<BackendStatus | null>(null);
    const [reconnected, setReconnected] = useState(false);
    const [errors, setErrors] = useState<ErrorEntry[]>([]);
    const [volumeServer, setVolumeServer] = useState<VolumeServerState>('unchecked');
    const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
    const errorIdRef = useRef(0);
    const lastErrorRef = useRef<{ message: string; at: number } | null>(null);
    const storageKey = 'ouroboros.autoseg.jobId';

    const addError = (message: string): void => {
        const now = Date.now();
        const last = lastErrorRef.current;
        if (last && last.message === message && now - last.at < DEDUP_WINDOW_MS) return;
        lastErrorRef.current = { message, at: now };
        const id = `err-${++errorIdRef.current}`;
        setErrors((prev) => [...prev.slice(-(MAX_ERRORS - 1)), { id, message }]);
    };

    const dismissError = (id: string): void => {
        setErrors((prev) => prev.filter((e) => e.id !== id));
    };

    // Once Docker is ready, probe the volume server directly from the plugin.
    // The volume server has no /-status endpoint yet; a HEAD to the root that
    // resolves (2xx / 3xx / 404) is enough to prove it's listening.
    useEffect(() => {
        if (!connected) {
            setVolumeServer('unchecked');
            return;
        }
        let cancelled = false;
        const probe = async (): Promise<void> => {
            try {
                await fetch(VOLUME_SERVER_URL, { method: 'HEAD', mode: 'no-cors' });
                if (!cancelled) setVolumeServer('ok');
            } catch {
                if (!cancelled) setVolumeServer('unreachable');
            }
        };
        probe();
        const interval = setInterval(probe, 10000);
        return () => {
            cancelled = true;
            clearInterval(interval);
        };
    }, [connected]);

    const getStoredJobId = () => {
        try {
            return localStorage.getItem(storageKey);
        } catch {
            return null;
        }
    };

    const setStoredJobId = (id: string) => {
        try {
            localStorage.setItem(storageKey, id);
        } catch (error) {
            console.warn('Failed to persist job id in localStorage', error);
        }
    };

    const clearStoredJobId = () => {
        try {
            localStorage.removeItem(storageKey);
        } catch (error) {
            console.warn('Failed to clear job id in localStorage', error);
        }
    };

    const scheduleReconnectBannerClear = () => {
        if (reconnectTimer.current) {
            clearTimeout(reconnectTimer.current);
        }
        setReconnected(true);
        reconnectTimer.current = setTimeout(() => setReconnected(false), 6000);
    };

    useEffect(() => {
        let cancelled = false;
        let timer: ReturnType<typeof setTimeout> | null = null;
        const statusUrls = ['/docker-status', `${BACKEND_URL}/startup-status`];

        const check = async (): Promise<void> => {
            for (const statusUrl of statusUrls) {
                try {
                    const res = await fetch(statusUrl);
                    if (!res.ok) continue;
                    const data: BackendStatus = await res.json();
                    if (!cancelled) {
                        setConnected(Boolean(data.is_ready));
                        setBackendStatus(data);
                    }
                    return;
                } catch {
                    // Try next source.
                }
            }

            if (!cancelled) {
                setConnected(false);
                setBackendStatus(null);
            }
        };

        const poll = async (delayMs: number) => {
            if (cancelled) return;
            timer = setTimeout(async () => {
                await check();
                await poll(5000);
            }, delayMs);
        };

        // Give backend container a small startup grace period in dev.
        poll(2500);

        return () => {
            cancelled = true;
            if (timer) clearTimeout(timer);
        };
    }, []);

    const handleRun = async (opts: RunOptions) => {
        setRun(true);
        setProg([
            {name: 'Transferring', progress: 0},
            {name: 'Inference', progress: 0},
            {name: 'Saving', progress: 0}
        ]);
        clearStoredJobId();
        try {
            const res = await fetch(`${BACKEND_URL}/process-stack`, {
                method:'POST', 
                headers:{'Content-Type':'application/json'}, 
                body:JSON.stringify({
                    file_path: opts.filePath, 
                    output_file: opts.outputFile,
                    model_type: opts.modelType,
                    predictor_type: opts.predictor_type,
                    overlay_annotation_points: opts.overlayAnnotationPoints
                })
            });
            if (!res.ok) {
                const detail = await res.text().catch(() => '');
                addError(`Failed to start job (HTTP ${res.status})${detail ? `: ${detail.slice(0, 200)}` : ''}`);
                setRun(false);
                return;
            }
            const data = await res.json();
            if (data.job_id) {
                setJobId(data.job_id);
                setStoredJobId(data.job_id);
            } else {
                addError('Backend accepted the request but did not return a job id.');
                setRun(false);
            }
        } catch (error) {
            addError(`Failed to start job: ${error instanceof Error ? error.message : String(error)}`);
            setRun(false);
        }
    };

    useEffect(() => {
        if (!connected || run || jobId) return;

        let cancelled = false;

        const tryReconnect = async () => {
            const storedJobId = getStoredJobId();
            const restored = await findRunningJob(fetch, BACKEND_URL, storedJobId);
            if (cancelled) return;
            if (!restored) {
                if (storedJobId) clearStoredJobId();
                return;
            }
            setProg(restored.record.steps);
            setJobId(restored.jobId);
            setStoredJobId(restored.jobId);
            setRun(true);
            scheduleReconnectBannerClear();
        };

        tryReconnect();
        return () => { cancelled = true; };
    }, [connected, run, jobId]);

    useEffect(() => {
        let interval: ReturnType<typeof setInterval> | null = null;
        if(run && jobId) {
            interval = setInterval(async () => {
                try {
                    const res = await fetch(`${BACKEND_URL}/status/${jobId}`);
                    if(res.ok) {
                        const data = await res.json();
                        setProg(data.steps);
                        if(data.status === 'completed' || data.status === 'error') {
                            if (data.status === 'error') {
                                addError(data.error ?? `${data.active_phase ?? 'Unknown'} phase failed`);
                            }
                            setRun(false);
                            setJobId(null);
                            clearStoredJobId();
                        }
                    } else if (res.status === 404) {
                        addError('Job disappeared from the backend before it completed.');
                        setRun(false);
                        setJobId(null);
                        clearStoredJobId();
                    }
                } catch (error) {
                    addError(
                        `Lost contact with the backend while polling job status: ${
                            error instanceof Error ? error.message : String(error)
                        }`
                    );
                }
            }, 500);
        }
        return () => {
            if (interval) clearInterval(interval);
        };
    }, [run, jobId]);

    useEffect(() => {
        return () => {
            if (reconnectTimer.current) {
                clearTimeout(reconnectTimer.current);
            }
        };
    }, []);

    return (
        <div className={styles.pageLayout}>
            <div className={styles.centerArea}>
                <div className={styles.progressArea}>
                    <ProgressPanel
                        items={prog}
                        connected={connected}
                        backendStatus={backendStatus}
                        reconnected={reconnected}
                        errors={errors}
                        onDismissError={dismissError}
                        volumeServer={volumeServer}
                    />
                </div>
			</div>
            <div className={styles.rightSidebar}>
                <div className={styles.optionsArea}>
                    <OptionsPanel
                        onSubmit={handleRun}
                        isRunning={run}
                    />
                </div>
                <div className={styles.modelsArea}>
                    <ModelsPanel connected={connected} />
                </div>
            </div>
        </div>
    );
}
