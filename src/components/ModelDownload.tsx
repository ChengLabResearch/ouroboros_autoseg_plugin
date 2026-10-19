import type { ReactNode } from 'react';
import styles from '../assets/styles.module.css';
import { formatDownloadBytes, isActiveDownload } from '../modelDownloads';
import type { DownloadRequestFeedback, ModelDownloadStatus } from '../modelDownloads';
import Notice from './Notice';

const stateLabels: Record<ModelDownloadStatus['state'], string> = {
    idle: 'Ready to download',
    connecting: 'Connecting to download',
    downloading: 'Downloading',
    finalizing: 'Finishing download',
    available: 'Downloaded',
    failed: 'Download failed',
};

export default function ModelDownload({
    label,
    status,
    fresh,
    request,
    requiresToken = false,
    token = '',
    onDownload,
    children,
}: {
    label: string;
    status: ModelDownloadStatus | undefined;
    fresh: boolean;
    request: DownloadRequestFeedback;
    requiresToken?: boolean;
    token?: string;
    onDownload: () => void;
    children?: ReactNode;
}) {
    const active = Boolean(status && isActiveDownload(status));
    const available = status?.state === 'available';
    const failed = status?.state === 'failed';
    const showProgress = active || failed;
    const total = status?.total_bytes;
    const measured = (status?.state === 'downloading' || failed) && total !== null && total !== undefined && total > 0;
    const percent = measured ? Math.min(100, Math.floor(status.bytes_received / total * 100)) : null;
    const animate = active && fresh && percent === null;
    const buttonLabel = available ? 'Downloaded'
        : active ? 'Downloading…'
            : request.pending ? 'Request pending…'
                : failed ? 'Retry' : 'Download';
    const buttonClass = available ? `${styles.downloadBtn} ${styles.downloadBtnSuccess}`
        : failed ? `${styles.downloadBtn} ${styles.downloadBtnError}` : styles.downloadBtn;

    return (
        <div className={styles.sam3Container}>
            <div className={styles.row}>
                <span className={styles.label}>{label}</span>
            </div>
            <div className={styles.row}>
                {children}
                <button
                    className={buttonClass}
                    onClick={onDownload}
                    disabled={!fresh || !status || active || available || request.pending || (requiresToken && !token.trim())}
                >
                    {buttonLabel}
                </button>
            </div>
            <div className={styles.modelDownloadStatus} aria-live="polite">
                {status ? `${fresh ? '' : 'Last reported: '}${stateLabels[status.state]}` : 'Checking model status…'}
            </div>
            {showProgress && status && (
                <div className={styles.modelDownloadProgress}>
                    <div className={styles.progressLabelRow}>
                        <span>
                            {formatDownloadBytes(status.bytes_received)}
                            {total !== null && total !== undefined && ` / ${formatDownloadBytes(total)}`}
                        </span>
                        {percent !== null && <span>{percent}%</span>}
                    </div>
                    <div
                        className={styles.progressBarBg}
                        role="progressbar"
                        aria-label={`${label} download`}
                        aria-valuemin={0}
                        aria-valuemax={100}
                        aria-valuenow={percent ?? undefined}
                        aria-valuetext={`${fresh ? '' : 'Last reported: '}${stateLabels[status.state]}`}
                        aria-busy={animate}
                    >
                        <div
                            className={`${styles.progressBarFill} ${animate ? styles.modelDownloadIndeterminate : ''}`}
                            style={{ width: percent === null ? active ? '35%' : '0%' : `${percent}%` }}
                        />
                    </div>
                </div>
            )}
            {failed && status?.error && <Notice type="error" message={status.error} />}
            {request.error && !failed && <Notice type="error" message={request.error} />}
        </div>
    );
}
