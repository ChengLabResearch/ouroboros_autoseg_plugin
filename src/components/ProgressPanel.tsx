import styles from '../assets/styles.module.css';
export type ProgressItem = { name: string; progress: number; };
export type BackendStep = { name: string; status: string; };
export type BackendStatus = {
    is_ready: boolean;
    initialization_steps: BackendStep[];
    start_time: number | null;
    ready_time: number | null;
};

function statusClass(status: string): string {
    if (status === 'completed') return styles.statusCompleted;
    if (status === 'warning' || status === 'error') return styles.statusWarning;
    if (status === 'in_progress') return styles.statusInProgress;
    return styles.statusPending;
}

export default function ProgressPanel({
    items,
    connected,
    backendStatus
}: {
    items: ProgressItem[],
    connected: boolean,
    backendStatus: BackendStatus | null
}) {
    const backendLabel = !connected
        ? 'Disconnected'
        : backendStatus?.is_ready
            ? 'Ready'
            : 'Starting';

    return (
        <div className={styles.container}>
            <div className={styles.headerRow}>
                <span className={styles.headerTitle}>PROGRESS</span>
                <div 
                    className={`${styles.connectionDot} ${connected ? styles.connected : ''}`} 
                    title={connected ? "Backend Connected" : "Backend Disconnected"} 
                />
            </div>
            <div className={styles.progressContent}>
                <div className={styles.backendStatusBlock}>
                    <div className={styles.backendStatusHeader}>
                        <span>Docker Backend</span>
                        <span className={`${styles.backendStatusBadge} ${connected ? styles.statusCompleted : styles.statusWarning}`}>
                            {backendLabel}
                        </span>
                    </div>
                    {connected && backendStatus && (
                        <div className={styles.backendSteps}>
                            {backendStatus.initialization_steps.map((step) => (
                                <div key={step.name} className={styles.backendStepRow}>
                                    <span>{step.name}</span>
                                    <span className={`${styles.backendStatusBadge} ${statusClass(step.status)}`}>
                                        {step.status.replace('_', ' ')}
                                    </span>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
                {items.map((i, idx) => (
                    <div key={idx} className={styles.progressItem}>
                        <div className={styles.progressLabelRow}><span>{i.name}</span><span>{i.progress}%</span></div>
                        <div className={styles.progressBarBg}>
                            <div className={styles.progressBarFill} style={{ width: `${i.progress}%` }} />
                        </div>
                    </div>
                ))}
                {items.length === 0 && <div style={{color:'#666', fontStyle:'italic', fontSize: '0.9rem'}}>No active tasks.</div>}
            </div>
        </div>
    );
}
