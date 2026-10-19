import { STATUS_LABELS } from '../statusPresentation';
import type { PresentationStatus } from '../statusPresentation';
import styles from './StatusLabel.module.css';

export default function StatusLabel({ status, label = STATUS_LABELS[status] }: {
    status: PresentationStatus;
    label?: string;
}) {
    return <span className={styles.status} data-status={status}>{label}</span>;
}
