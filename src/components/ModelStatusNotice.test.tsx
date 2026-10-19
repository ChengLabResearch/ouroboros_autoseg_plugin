import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { ModelStatusError } from '../modelDownloads';
import type { ModelStatusSnapshot } from '../modelDownloads';
import ModelStatusNotice from './ModelStatusNotice';

const snapshot: ModelStatusSnapshot = {
    downloads: {
        sam3: { state: 'idle', bytes_received: 0, total_bytes: null, error: null },
        medical_sam3: { state: 'downloading', bytes_received: 1024, total_bytes: 4096, error: null },
    },
};

function markup(error: ModelStatusError | null, status: ModelStatusSnapshot | null): string {
    return renderToStaticMarkup(<ModelStatusNotice error={error} snapshot={status} />);
}

describe('model status notices', () => {
    it.each(['unavailable', 'http', 'invalid-response'] as const)(
        'presents %s according to snapshot availability', (kind) => {
            // Message contents do not determine visibility; the watcher owns classification.
            const error = new ModelStatusError(kind, 'Details from the status request.\nCheckpoint path: /models/checkpoint.pt');
            const initial = markup(error, null);
            if (kind === 'unavailable') {
                expect(initial).toContain('data-type="info" role="status"');
                expect(initial).toContain('Model status unavailable. Retrying automatically.');
            } else {
                expect(initial).toContain('data-type="error" role="alert"');
                expect(initial).toContain('Unable to read model download status.');
            }
            expect(initial).toContain(error.message);
            expect(initial).not.toContain('Last reported progress');
            const stale = markup(error, snapshot);
            expect(stale).toContain('data-type="error" role="alert"');
            expect(stale).toContain('Last reported progress is shown.');
            expect(stale).toContain(error.message);
            expect(stale).not.toContain('Dismiss');
        },
    );

    it('has no notice before the first response or after recovery', () => {
        expect(markup(null, null)).toBe('');
        expect(markup(null, snapshot)).toBe('');
    });
});
