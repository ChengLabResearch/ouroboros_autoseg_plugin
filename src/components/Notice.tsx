import styles from './Notice.module.css';

export default function Notice({ message, type, onDismiss }: {
    message: string;
    type: 'error' | 'info';
    onDismiss?: () => void;
}) {
    return (
        <div className={styles.notice} data-type={type} role={type === 'error' ? 'alert' : 'status'}>
            <div className={styles.message}>{message}</div>
            {onDismiss && (
                <button type="button" className={styles.dismiss} onClick={onDismiss}>
                    Dismiss
                </button>
            )}
        </div>
    );
}
