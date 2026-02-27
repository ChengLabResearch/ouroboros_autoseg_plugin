import { useState, useEffect } from 'react';
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
            if(data.job_id) setJobId(data.job_id);
        } catch(e) { console.error(e); setRun(false); }
    };

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
                        }
                    }
                } catch(e) {}
            }, 500);
        }
        return () => clearInterval(interval);
    }, [run, jobId]);

    return (
        <div className={styles.pageLayout}>
            <div className={styles.centerArea}>
				<div className={styles.vizArea}>
					<VisualizePanel><div>SAM3 Visualization</div></VisualizePanel>
				</div>
                <div className={styles.progressArea}>
                    <ProgressPanel items={prog} connected={connected} backendStatus={backendStatus} />
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
