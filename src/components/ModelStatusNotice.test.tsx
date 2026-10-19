import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { ModelStatusError } from '../modelDownloads';
import type { ModelStatusSnapshot } from '../modelDownloads';
import ModelDownload from './ModelDownload';
import ModelStatusNotice from './ModelStatusNotice';

const snapshot: ModelStatusSnapshot = {
    downloads: {
        sam3: { state: 'idle', bytes_received: 0, total_bytes: null, error: null },
        medical_sam3: { state: 'downloading', bytes_received: 1024, total_bytes: 4096, error: null },
    },
};

function markup(error: ModelStatusError | null, status: ModelStatusSnapshot | null): string {
    return renderToStaticMarkup(<>
        <ModelStatusNotice error={error} snapshot={status} />
        <ModelDownload
            label="SAM3 (Medical)"
            status={status?.downloads.medical_sam3}
            fresh={Boolean(status && !error)}
            request={{ pending: false, error: null }}
            onDownload={() => {}}
        />
    </>);
}

describe('model status notices', () => {
    it.each([
        'Failed to fetch',
        'Model status request timed out. Retrying…',
        'Model status request failed (HTTP 503). Retrying…',
    ])('keeps initial unavailability quiet: %s', (message) => {
        const html = markup(new ModelStatusError('unavailable', message), null);
        expect(html).not.toContain('role="alert"');
        expect(html).not.toContain(message);
        expect(html).toContain('Checking model status…');
        expect(html).toContain('disabled=""');
    });

    it.each([
        ['invalid-response', 'The backend returned invalid JSON for model download progress. Check that the plugin and backend versions match.'],
        ['invalid-response', 'The backend did not return model download progress. Check that the plugin and backend versions match.'],
        ['http', 'Model status request failed (HTTP 404). Check the backend address and logs.'],
    ] as const)('shows an initial %s failure without claiming retained progress', (kind, message) => {
        const html = markup(new ModelStatusError(kind, message), null);
        expect(html).toContain('role="alert"');
        expect(html).toContain('Unable to read model download status.');
        expect(html).toContain(message);
        expect(html).not.toContain('Last reported progress is shown');
        expect(html).not.toContain('Last reported:');
        expect(html).toContain('disabled=""');
    });

    it.each(['unavailable', 'http', 'invalid-response'] as const)(
        'retains measured progress and warns after a %s failure', (kind) => {
            const html = markup(new ModelStatusError(kind, 'Status request failed.'), snapshot);
            expect(html).toContain('role="alert"');
            expect(html).toContain('Unable to refresh model downloads. Last reported progress is shown.');
            expect(html).toContain('Status request failed.');
            expect(html).toContain('Last reported: Downloading');
            expect(html).toContain('1.0 KiB / 4.0 KiB');
            expect(html).toContain('aria-valuenow="25"');
            expect(html).toContain('disabled=""');
        },
    );

    it('removes the notice and stale label when a valid snapshot recovers', () => {
        expect(markup(new ModelStatusError('unavailable', 'Failed to fetch'), snapshot)).toContain('role="alert"');

        const html = markup(null, snapshot);
        expect(html).not.toContain('role="alert"');
        expect(html).not.toContain('Last reported');
        expect(html).not.toContain('Failed to fetch');
        expect(html).toContain('1.0 KiB / 4.0 KiB');
        expect(html).toContain('aria-valuenow="25"');
    });

    it('has no error before the first response', () => {
        const html = markup(null, null);
        expect(html).not.toContain('role="alert"');
        expect(html).toContain('Checking model status…');
    });
});
