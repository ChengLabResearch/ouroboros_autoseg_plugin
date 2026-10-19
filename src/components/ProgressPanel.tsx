import styles from '../assets/styles.module.css';
import { presentInitialization } from '../statusPresentation';
import type { BackendStatus, VolumeServerState } from '../statusPresentation';
import StatusLabel from './StatusLabel';
export type { BackendStatus, BackendStep, VolumeServerState } from '../statusPresentation';
export type ProgressItem = { name: string; progress: number; };
export type ErrorEntry = { id: string; message: string; };

export default function ProgressPanel({
    items,
    connected,
    backendStatus,
    reconnected,
    errors = [],
    onDismissError,
    volumeServer = 'unchecked'
}: {
    items: ProgressItem[],
    connected: boolean,
    backendStatus: BackendStatus | null,
    reconnected?: boolean,
    errors?: ErrorEntry[],
    onDismissError?: (id: string) => void,
    volumeServer?: VolumeServerState
}) {
    const initialization = presentInitialization(backendStatus, connected ? volumeServer : undefined);

    return (
        <div className={styles.container}>
            <div className={styles.headerRow}>
                <span className={styles.headerTitle}>PROGRESS</span>
            </div>
            <div className={styles.progressContent}>
                <section className={styles.backendStatusBlock} aria-label="Backend initialization">
                    <div className={styles.backendStatusHeader}>
                        <strong>Backend</strong>
                        <StatusLabel {...initialization.summary} />
                    </div>
                    {reconnected && (
                        <div className={styles.reconnectBanner}>
                            Reconnected to an active run.
                        </div>
                    )}
                    {backendStatus && (
                        <dl className={styles.backendSteps}>
                            {initialization.steps.map((step) => (
                                <div key={step.name}>
                                    <dt>{step.name}</dt>
                                    <dd><StatusLabel status={step.status} label={step.label} /></dd>
                                </div>
                            ))}
                        </dl>
                    )}
                </section>
                {errors.length > 0 && (
                    <div className={styles.errorStack}>
                        {errors.map((err) => (
                            <div key={err.id} className={styles.errorBanner} role="alert">
                                <span className={styles.errorText}>{err.message}</span>
                                {onDismissError && (
                                    <button
                                        type="button"
                                        onClick={() => onDismissError(err.id)}
                                        className={styles.errorDismiss}
                                        aria-label="Dismiss error"
                                    >
                                        ×
                                    </button>
                                )}
                            </div>
                        ))}
                    </div>
                )}
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
