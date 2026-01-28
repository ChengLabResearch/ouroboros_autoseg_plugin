import { useState, useEffect } from 'react';
import styles from '../assets/styles.module.css';
import OptionsPanel from '../components/OptionsPanel';
import ProgressPanel, { ProgressItem } from '../components/ProgressPanel';
import VisualizePanel from '../components/VisualizePanel';

const BACKEND_URL = "http://localhost:8686";

export default function SAM3Page() {
    const [prog, setProg] = useState<ProgressItem[]>([]);
    const [run, setRun] = useState(false);
    const [jobId, setJobId] = useState<string | null>(null);
    const [connected, setConnected] = useState(false);

    useEffect(() => {
        const check = async () => {
            try {
                const res = await fetch(`${BACKEND_URL}/`);
                if(res.ok) setConnected(true);
                else setConnected(false);
            } catch(e) { setConnected(false); }
        };
        check();
        const i = setInterval(check, 5000);
        return () => clearInterval(i);
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
                    <ProgressPanel items={prog} connected={connected} />
                </div>
			</div>
            <div className={styles.rightSidebar}>
                <div className={styles.optionsArea}>
                    <OptionsPanel onSubmit={handleRun} isRunning={run} />
                </div>
            </div>
        </div>
    );
}