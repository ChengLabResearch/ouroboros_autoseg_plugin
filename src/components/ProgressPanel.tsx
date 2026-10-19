import styles from '../assets/styles.module.css';
import { presentInitialization, STATUS_LABELS } from '../statusPresentation';
import type { BackendStatus, VolumeServerState } from '../statusPresentation';
import { presentJob } from '../jobStatus';
import type { JobObservation } from '../jobStatus';
import StatusLabel from './StatusLabel';
export type { BackendStatus, BackendStep, VolumeServerState } from '../statusPresentation';
export type ErrorEntry = { id: string; message: string; };

export default function ProgressPanel({
    jobObservation,
    isRunning,
    jobId,
    connected,
    backendStatus,
    reconnected,
    errors = [],
    onDismissError,
    volumeServer = 'unchecked'
}: {
    jobObservation: JobObservation,
    isRunning: boolean,
    jobId: string | null,
    connected: boolean,
    backendStatus: BackendStatus | null,
    reconnected?: boolean,
    errors?: ErrorEntry[],
    onDismissError?: (id: string) => void,
    volumeServer?: VolumeServerState
}) {
    const initialization = presentInitialization(backendStatus, connected ? volumeServer : undefined);
    const task = presentJob(jobObservation, isRunning, jobId);

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
                <section aria-label="Segmentation task">
                    {task.summary && (
                        <div className={styles.taskSummary} role="status">
                            <strong>Segmentation</strong>
                            <StatusLabel {...task.summary} />
                        </div>
                    )}
                    {task.detail && <div className={styles.taskDetail}>{task.detail}</div>}
                    {task.steps.map((step) => (
                        <div key={step.name} className={styles.taskProgressItem}>
                            <div className={styles.taskProgressHeading}>
                                <span>{step.name}</span>
                                <StatusLabel status={step.status} />
                            </div>
                            <div className={styles.taskProgressRow}>
                                <div
                                    className={styles.taskProgressBar}
                                    role="progressbar"
                                    aria-label={step.name}
                                    aria-valuemin={0}
                                    aria-valuemax={100}
                                    aria-valuenow={step.progress}
                                    aria-valuetext={STATUS_LABELS[step.status]}
                                >
                                    <div className={styles.taskProgressFill} style={{ width: `${step.progress}%` }} />
                                </div>
                                <span className={styles.taskProgressPercent}>{step.progress}%</span>
                            </div>
                        </div>
                    ))}
                    {!task.summary && <div className={styles.taskDetail}>No active tasks.</div>}
                </section>
            </div>
        </div>
    );
}
