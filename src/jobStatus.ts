import type { PresentationStatus } from './statusPresentation';

export type ProgressItem = { name: string; progress: number; };
export type JobStatusRecord = {
    status: string;
    steps: ProgressItem[];
    active_phase?: string;
    error?: string;
};

// One observation retains both reported job facts and the existing missing-job
// response. A missing response is not a reported execution failure.
export type JobObservation =
    | { kind: 'status'; record: JobStatusRecord; }
    | { kind: 'missing'; lastRecord: JobStatusRecord | null; }
    | null;

const PHASE_NAMES = ['Transferring', 'Inference', 'Saving'] as const;
type JobDisplay = {
    summary: { status: PresentationStatus; label?: string; } | null;
    detail?: string;
    steps: (ProgressItem & { status: PresentationStatus; })[];
};

// Render-time display only: percentages, phase, and outcome remain response facts.
export function presentJob(observation: JobObservation, isRunning: boolean, jobId: string | null): JobDisplay {
    if (observation?.kind === 'missing') {
        return {
            summary: { status: 'unknown', label: 'Outcome unknown' },
            detail: observation.lastRecord ? 'Last reported progress.' : 'No job status was received.',
            steps: observation.lastRecord?.steps.map((step) => ({ ...step, status: 'unknown' })) ?? [],
        };
    }
    if (!observation) {
        return isRunning ? {
            summary: { status: 'waiting', label: jobId ? 'Waiting for job status…' : 'Starting job…' },
            steps: PHASE_NAMES.map((name) => ({ name, progress: 0, status: 'waiting' })),
        } : { summary: null, steps: [] };
    }

    const record = observation.record;
    const phaseName = PHASE_NAMES.find((name) => name.toLowerCase() === record.active_phase);
    const phaseIndex = record.steps.findIndex((step) => step.name === phaseName);
    const knownState = record.status === 'running' || record.status === 'error';
    const summary: JobDisplay['summary'] = record.status === 'completed'
        ? { status: 'complete' }
        : record.status === 'error'
            ? { status: 'failed', label: phaseName ? `Failed during ${phaseName}` : 'Failed' }
            : record.status === 'running'
                ? { status: 'running', label: phaseName ? `Running — ${phaseName}` : 'Running' }
                : { status: 'unknown' };

    return {
        summary,
        steps: record.steps.map((step, index) => {
            let status: PresentationStatus = 'unknown';
            if (record.status === 'completed') status = 'complete';
            else if (knownState && phaseIndex >= 0) {
                status = index < phaseIndex ? 'complete'
                    : index > phaseIndex ? 'waiting'
                        : record.status === 'error' ? 'failed' : 'running';
            }
            return { ...step, status };
        }),
    };
}
