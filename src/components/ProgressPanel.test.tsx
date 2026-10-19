import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import ProgressPanel from './ProgressPanel';
import type { BackendStatus } from './ProgressPanel';

function snapshot(steps: BackendStatus['initialization_steps'], is_ready = false): BackendStatus {
    return { is_ready, initialization_steps: steps, start_time: null, ready_time: null };
}

function markup(backendStatus: BackendStatus | null, props: Partial<Parameters<typeof ProgressPanel>[0]> = {}): string {
    return renderToStaticMarkup(<ProgressPanel items={[]} connected={false} backendStatus={backendStatus} {...props} />);
}

describe('initialization presentation', () => {
    it('waits for a startup observation without asserting a disconnected API', () => {
        const html = markup(null);
        expect(html).toContain('aria-label="Backend initialization"');
        expect(html).toContain('data-status="waiting">Waiting for startup status');
        expect(html).not.toMatch(/Disconnected|Connected|connectionDot/);
    });

    it.each([
        ['pending', 'waiting', 'Waiting'],
        ['in_progress', 'running', 'Running'],
        ['completed', 'complete', 'Complete'],
        ['warning', 'warning', 'Warning'],
        ['error', 'failed', 'Failed'],
        ['unrecognized', 'unknown', 'Unknown'],
    ])('renders an existing %s step using the shared vocabulary', (status, display, label) => {
        expect(markup(snapshot([{ name: 'Docker Compose', status }])))
            .toContain(`data-status="${display}">${label}</span>`);
    });

    it('uses reported readiness rather than inferring it from completed steps', () => {
        const steps = [{ name: 'Building Docker Image', status: 'completed' }];
        expect(markup(snapshot(steps))).toContain('data-status="waiting">Waiting for backend');
        expect(markup(snapshot(steps, true))).toContain('data-status="complete">Ready');
    });

    it('presents a lazy ML runtime neutrally and names the Rust service fact accurately', () => {
        const html = markup(snapshot([
            { name: 'Building Docker Image', status: 'completed' },
            { name: 'Connecting to Volume Server', status: 'completed' },
            { name: 'Initializing ML Models', status: 'warning' },
        ]));
        expect(html).toContain('Backend service');
        expect(html).toContain('Volume server (backend)');
        expect(html).toContain('data-status="waiting">Loads on first run');
        expect(html).not.toMatch(/Building Docker Image|data-status="warning"|data-status="failed"|Disconnected/);
        expect(html).not.toContain('Volume server (browser)');
    });

    it('keeps backend and browser volume observations distinct when they disagree', () => {
        const html = markup(snapshot([
            { name: 'Connecting to Volume Server', status: 'completed' },
        ], true), { connected: true, volumeServer: 'unreachable' });
        expect(html).toContain('data-status="warning">Startup needs attention');
        expect(html).toMatch(/Volume server \(backend\)<\/dt><dd><span[^>]+data-status="complete">Reachable/);
        expect(html).toMatch(/Volume server \(browser\)<\/dt><dd><span[^>]+data-status="warning">Unreachable/);
    });

    it('does not hide a reported startup failure when the container is running', () => {
        const status = snapshot([
            { name: 'Docker Compose', status: 'error' },
            { name: 'Container Startup', status: 'completed' },
        ], true);
        const html = markup(status, { connected: true, volumeServer: 'ok' });
        expect(html).toContain('data-status="warning">Startup needs attention');
        expect(html).toContain('data-status="failed">Failed');
        expect(html).toContain('Container Startup');
        expect(html).not.toContain('Connected');

        const running = markup(snapshot([
            { name: 'Docker Compose', status: 'completed' },
            { name: 'Container Startup', status: 'completed' },
        ], true), { connected: true });
        expect(running).toContain('data-status="complete">Container running');
        expect(running).toContain('data-status="waiting">Checking');
        expect(running).not.toContain('Connected');
    });
});
