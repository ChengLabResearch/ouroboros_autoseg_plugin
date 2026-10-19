import styles from '../assets/styles.module.css';
import type { ModelStatusError, ModelStatusSnapshot } from '../modelDownloads';

export default function ModelStatusNotice({ error, snapshot }: {
    error: ModelStatusError | null;
    snapshot: ModelStatusSnapshot | null;
}) {
    if (!error || (!snapshot && error.kind === 'unavailable')) return null;

    return (
        <div className={styles.inlineError} role="alert">
            {snapshot
                ? 'Unable to refresh model downloads. Last reported progress is shown. '
                : 'Unable to read model download status. '}
            {error.message}
        </div>
    );
}
