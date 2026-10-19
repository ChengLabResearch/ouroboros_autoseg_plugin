import Notice from './Notice';
import type { ModelStatusError, ModelStatusSnapshot } from '../modelDownloads';

export default function ModelStatusNotice({ error, snapshot }: {
    error: ModelStatusError | null;
    snapshot: ModelStatusSnapshot | null;
}) {
    if (!error) return null;

    if (!snapshot && error.kind === 'unavailable') {
        return <Notice type="info" message={`Model status unavailable. Retrying automatically.\n${error.message}`} />;
    }

    const context = snapshot
        ? 'Unable to refresh model downloads. Last reported progress is shown. '
        : 'Unable to read model download status. ';
    return <Notice type="error" message={`${context}${error.message}`} />;
}
