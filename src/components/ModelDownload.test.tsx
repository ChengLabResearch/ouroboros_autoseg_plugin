import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import ModelDownload from './ModelDownload';
import type { ModelDownloadStatus } from '../modelDownloads';

function markup(status: ModelDownloadStatus | undefined, props: Partial<Parameters<typeof ModelDownload>[0]> = {}): string {
    return renderToStaticMarkup(<ModelDownload
        label="SAM3 (Medical)"
        status={status}
        fresh
        request={{ pending: false, error: null }}
        onDownload={() => {}}
        {...props}
    />);
}

function download(state: ModelDownloadStatus['state'], overrides: Partial<ModelDownloadStatus> = {}): ModelDownloadStatus {
    return { state, bytes_received: 0, total_bytes: null, error: null, ...overrides };
}

describe('model download progress', () => {
    it('shows measured bytes and percentage with accessible progress values', () => {
        const html = markup(download('downloading', { bytes_received: 1024, total_bytes: 4096 }));
        expect(html).toContain('1.0 KiB / 4.0 KiB');
        expect(html).toContain('25%');
        expect(html).toContain('aria-valuenow="25"');
        expect(html).toContain('disabled=""');
    });

    it('shows bytes and an indeterminate bar for an unknown content length', () => {
        const html = markup(download('downloading', { bytes_received: 1024 }));
        expect(html).toContain('1.0 KiB');
        expect(html).not.toContain('aria-valuenow');
        expect(html).toContain('aria-busy="true"');
        expect(html).toContain('modelDownloadIndeterminate');
    });

    it('keeps finalization active after all bytes arrive and never labels it downloaded', () => {
        const html = markup(download('finalizing', { bytes_received: 4096, total_bytes: 4096 }));
        expect(html).toContain('Finishing download');
        expect(html).not.toContain('Downloaded');
        expect(html).not.toContain('aria-valuenow');
        expect(html).toContain('disabled=""');
    });

    it('does not enable a retry merely because a measured transfer reaches 100 percent', () => {
        const html = markup(download('downloading', { bytes_received: 4096, total_bytes: 4096 }));
        expect(html).toContain('aria-valuenow="100"');
        expect(html).toContain('disabled=""');
        expect(html).not.toContain('Downloaded');
    });

    it('labels stale progress and stops its animation during contact loss', () => {
        const html = markup(download('downloading', { bytes_received: 20 }), { fresh: false });
        expect(html).toContain('Last reported: Downloading');
        expect(html).not.toContain('modelDownloadIndeterminate');
        expect(html).toContain('aria-busy="false"');
        expect(html).toContain('disabled=""');
    });

    it('keeps a backend download active despite a lost POST response', () => {
        const html = markup(download('downloading'), { request: { pending: false, error: 'Response lost. Checking the backend.' } });
        expect(html).toContain('Downloading');
        expect(html).toContain('Response lost');
        expect(html).toContain('disabled=""');
        expect(html).not.toContain('Download failed');
    });

    it('shows actual backend failure with retry and its error', () => {
        const html = markup(download('failed', { error: 'Disk is full', bytes_received: 1024, total_bytes: 4096 }));
        expect(html).toContain('Download failed');
        expect(html).toContain('Disk is full');
        expect(html).toContain('Retry');
        expect(html).not.toContain('disabled=""');
        expect(html).toContain('aria-valuenow="25"');
        expect(html).not.toContain('modelDownloadIndeterminate');
    });

    it('requires a nonblank token only for official SAM3', () => {
        expect(markup(download('idle'), { requiresToken: true, token: '  ' })).toContain('disabled=""');
        expect(markup(download('idle'), { requiresToken: true, token: 'token' })).not.toContain('disabled=""');
        expect(markup(download('idle'))).not.toContain('disabled=""');
    });

    it('disables downloads until a current snapshot exists and while a POST is pending', () => {
        expect(markup(undefined)).toContain('disabled=""');
        expect(markup(download('idle'), { fresh: false })).toContain('disabled=""');
        const pending = markup(download('idle'), { request: { pending: true, error: null } });
        expect(pending).toContain('Request pending');
        expect(pending).toContain('disabled=""');
        expect(pending).not.toContain('Downloading');
    });

    it('shows downloaded only from backend availability', () => {
        const html = markup(download('available'));
        expect(html).toContain('Downloaded');
        expect(html).toContain('disabled=""');
        expect(html).not.toContain('role="progressbar"');
    });
});
