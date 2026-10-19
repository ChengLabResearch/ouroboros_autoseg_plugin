import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import ProgressPanel from './ProgressPanel';
import type { BackendStatus } from './ProgressPanel';
import type { JobStatusRecord } from '../jobStatus';

function snapshot(steps: BackendStatus['initialization_steps'], is_ready = false): BackendStatus {
    return { is_ready, initialization_steps: steps, start_time: null, ready_time: null };
}

function markup(backendStatus: BackendStatus | null, props: Partial<Parameters<typeof ProgressPanel>[0]> = {}): string {
    return renderToStaticMarkup(<ProgressPanel jobObservation={null} isRunning={false} jobId={null} connected={false} backendStatus={backendStatus} {...props} />);
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

    it('keeps the existing reconnect message informational and nondismissible', () => {
        const html = markup(null, { reconnected: true });
        expect(html).toContain('data-type="info" role="status"');
        expect(html).toContain('Reconnected to an active run.');
        expect(html).not.toContain('Dismiss');
        expect(markup(null)).not.toContain('Reconnected to an active run.');
    });
});

const running: JobStatusRecord = {
    status: 'running',
    active_phase: 'inference',
    steps: [
        { name: 'Transferring', progress: 100 },
        { name: 'Inference', progress: 0 },
        { name: 'Saving', progress: 0 },
    ],
};

describe('task presentation', () => {
    it('waits honestly for submission and then the first status response', () => {
        expect(markup(null, { isRunning: true })).toContain('Starting job…');
        const accepted = markup(null, { isRunning: true, jobId: 'accepted-job' });
        expect(accepted).toContain('Waiting for job status…');
        expect(accepted.match(/aria-valuenow="0"/g)).toHaveLength(3);
        expect(accepted).not.toContain('data-status="running"');
        expect(markup(null)).toContain('No active tasks.');
    });

    it('uses the reported phase and order even before the first positive percentage', () => {
        const html = markup(null, { jobObservation: { kind: 'status', record: running } });
        expect(html).toContain('Running — Inference');
        expect(html).toMatch(/aria-label="Transferring"[^>]+aria-valuetext="Complete"/);
        expect(html).toMatch(/aria-label="Inference"[^>]+aria-valuenow="0"[^>]+aria-valuetext="Running"/);
        expect(html).toMatch(/aria-label="Saving"[^>]+aria-valuetext="Waiting"/);
    });

    it('requires a terminal response rather than treating 100 percent as completion', () => {
        const finalBars = running.steps.map((step) => ({ ...step, progress: 100 }));
        const active = markup(null, { jobObservation: { kind: 'status', record: {
            ...running, active_phase: 'saving', steps: finalBars,
        } } });
        expect(active).toContain('Running — Saving');
        expect(active).toMatch(/aria-label="Saving"[^>]+aria-valuenow="100"[^>]+aria-valuetext="Running"/);

        const completed = markup(null, { jobObservation: { kind: 'status', record: {
            status: 'completed', steps: finalBars,
        } } });
        expect(completed).toMatch(/<strong>Segmentation<\/strong><span[^>]+data-status="complete">Complete/);
        expect(completed).not.toContain('No active tasks.');
    });

    it('keeps a failed outcome independently of its one dismissible diagnostic', () => {
        const jobObservation = { kind: 'status' as const, record: {
            ...running, status: 'error', error: 'Inference: Device unavailable',
        } };
        const html = markup(null, { jobObservation, errors: [{ id: 'failure', message: jobObservation.record.error }], onDismissError: () => {} });
        expect(html.match(/Inference: Device unavailable/g)).toHaveLength(1);
        expect(html).toMatch(/aria-label="Segmentation task".*data-type="error" role="alert"/);
        expect(html).toMatch(/<button[^>]+>Dismiss<\/button>/);
        expect(html).toContain('Failed during Inference');
        expect(html).toMatch(/aria-label="Inference"[^>]+aria-valuetext="Failed"/);
        const dismissed = markup(null, { jobObservation });
        expect(dismissed).toContain('Failed during Inference');
        expect(dismissed).not.toContain('Device unavailable');
        expect(dismissed).not.toContain('Dismiss');
    });

    it.each([null, running])('shows unknown outcome after a missing job with prior record %j', (lastRecord) => {
        const html = markup(null, { jobObservation: { kind: 'missing', lastRecord } });
        expect(html).toContain('Outcome unknown');
        expect(html).not.toMatch(/data-status="running"|data-status="failed"|No active tasks/);
        if (lastRecord) {
            expect(html).toContain('Last reported progress.');
            expect(html).toMatch(/aria-label="Transferring"[^>]+aria-valuenow="100"[^>]+aria-valuetext="Unknown"/);
        } else {
            expect(html).toContain('No job status was received.');
            expect(html).not.toContain('role="progressbar"');
        }
    });

    it('keeps unknown status and phase details neutral without guessing from percentages', () => {
        const unknownPhase = markup(null, { jobObservation: { kind: 'status', record: {
            ...running, active_phase: 'unrecognized',
        } } });
        expect(unknownPhase.match(/aria-valuetext="Unknown"/g)).toHaveLength(3);
        expect(unknownPhase).not.toContain('Running —');
        const unknownState = markup(null, { jobObservation: { kind: 'status', record: {
            ...running, status: 'unrecognized',
        } } });
        expect(unknownState).not.toContain('data-status="running"');
        expect(unknownState.match(/aria-valuetext="Unknown"/g)).toHaveLength(3);
    });
});
