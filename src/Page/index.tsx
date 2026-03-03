import { useState, useEffect, useRef } from 'react';
import styles from '../assets/styles.module.css';
import OptionsPanel from '../components/OptionsPanel';
import ProgressPanel, { BackendStatus, ProgressItem } from '../components/ProgressPanel';
import VisualizePanel from '../components/VisualizePanel';
import { BACKEND_URL } from '../config';

export default function SAM3Page() {
    const [prog, setProg] = useState<ProgressItem[]>([]);
    const [run, setRun] = useState(false);
    const [jobId, setJobId] = useState<string | null>(null);
    const [connected, setConnected] = useState(false);
    const [backendStatus, setBackendStatus] = useState<BackendStatus | null>(null);
    const [reconnected, setReconnected] = useState(false);
    const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
    const storageKey = 'ouroboros.autoseg.jobId';

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
        } catch {}
    };

    const clearStoredJobId = () => {
        try {
            localStorage.removeItem(storageKey);
        } catch {}
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

        const check = async () => {
            try {
                const res = await fetch(`${BACKEND_URL}/startup-status`);
                if(res.ok) {
                    const data: BackendStatus = await res.json();
                    if (!cancelled) {
                        setConnected(true);
                        setBackendStatus(data);
                    }
                } else {
                    if (!cancelled) {
                        setConnected(false);
                        setBackendStatus(null);
                    }
                }
            } catch(e) {
                if (!cancelled) {
                    setConnected(false);
                    setBackendStatus(null);
                }
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

    const handleRun = async (opts: any) => {
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
                    predictor_type: opts.predictor_type
                })
            });
            const data = await res.json();
            if (data.job_id) {
                setJobId(data.job_id);
                setStoredJobId(data.job_id);
            } else {
                setRun(false);
            }
        } catch(e) { console.error(e); setRun(false); }
    };

    useEffect(() => {
        if (!connected || run || jobId) return;

        let cancelled = false;

        const tryReconnect = async () => {
            const storedJobId = getStoredJobId();
            let candidateJobId = storedJobId;

            if (!candidateJobId) {
                try {
                    const latestRes = await fetch(`${BACKEND_URL}/latest-job`);
                    if (latestRes.ok) {
                        const latestData = await latestRes.json();
                        const latestJobId = latestData?.job_id;
                        if (typeof latestJobId === 'string' && latestJobId.length > 0) {
                            candidateJobId = latestJobId;
                            setStoredJobId(latestJobId);
                        }
                    }
                } catch {
                    // Ignore; we'll retry later when the backend is stable.
                }
            }

            if (!candidateJobId) return;

            try {
                const res = await fetch(`${BACKEND_URL}/status/${candidateJobId}`);
                if (!res.ok) {
                    clearStoredJobId();
                    return;
                }
                const data = await res.json();
                if (cancelled) return;
                if (Array.isArray(data.steps)) {
                    setProg(data.steps);
                }
                if (data.status === 'completed' || data.status === 'error') {
                    clearStoredJobId();
                    return;
                }
                setJobId(candidateJobId);
                setRun(true);
                scheduleReconnectBannerClear();
            } catch {
                // Keep stored job id for a later reconnect attempt.
            }
        };

        tryReconnect();
        return () => { cancelled = true; };
    }, [connected, run, jobId]);

    useEffect(() => {
        let interval: any;
        if(run && jobId) {
            interval = setInterval(async () => {
                try {
                    const res = await fetch(`${BACKEND_URL}/status/${jobId}`);
                    if(res.ok) {
                        const data = await res.json();
                        setProg(data.steps);
                        if(data.status === 'completed' || data.status === 'error') {
                            setRun(false);
                            setJobId(null);
                            clearStoredJobId();
                        }
                    } else if (res.status === 404) {
                        setRun(false);
                        setJobId(null);
                        clearStoredJobId();
                    }
                } catch(e) {}
            }, 500);
        }
        return () => clearInterval(interval);
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
				<div className={styles.vizArea}>
					<VisualizePanel><div>SAM3 Visualization</div></VisualizePanel>
				</div>
                <div className={styles.progressArea}>
                    <ProgressPanel items={prog} connected={connected} backendStatus={backendStatus} reconnected={reconnected} />
                </div>
			</div>
            <div className={styles.rightSidebar}>
                <div className={styles.optionsArea}>
                    <OptionsPanel onSubmit={handleRun} isRunning={run} connected={connected} />
                </div>
            </div>
        </div>
    );
}
