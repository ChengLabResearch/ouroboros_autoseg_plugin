import { isValidElement, useState } from 'react';
import type { ReactNode } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import OptionsPanel from './OptionsPanel';

vi.mock('react', async (importOriginal) => {
    const actual = await importOriginal<typeof import('react')>();
    return {
        ...actual,
        useState: vi.fn(),
        useRef: () => ({ current: null }),
        useEffect: vi.fn(),
    };
});

type Control = {
    id?: string;
    title?: string;
    placeholder?: string;
    checked?: boolean;
    value?: string;
    disabled?: boolean;
    'aria-invalid'?: boolean;
    onChange?: (event: { target: { value: string; checked: boolean; validity: { badInput: boolean } } }) => void;
    onClick?: () => void;
    children?: ReactNode;
};

let states: unknown[];
let cursor: number;

beforeEach(() => {
    states = [];
    cursor = 0;
    // Retain hook state between renders so event handlers exercise the actual
    // form without adding a browser environment dependency to the test suite.
    vi.mocked(useState).mockImplementation(((initial: unknown) => {
        const index = cursor++;
        if (!(index in states)) states[index] = initial;
        return [states[index], (value: unknown) => { states[index] = value; }];
    }) as typeof useState);
});

function form() {
    const onSubmit = vi.fn();
    let tree: ReactNode;
    const render = () => {
        cursor = 0;
        tree = OptionsPanel({ onSubmit, isRunning: false });
    };
    const find = (match: (control: Control) => boolean): Control | undefined => {
        const visit = (node: ReactNode): Control | undefined => {
            if (Array.isArray(node)) {
                for (const child of node) {
                    const found = visit(child);
                    if (found) return found;
                }
            } else if (isValidElement<Control>(node)) {
                if (match(node.props)) return node.props;
                return visit(node.props.children);
            }
            return undefined;
        };
        return visit(tree);
    };
    const control = (id: string) => {
        const result = find((props) => props.id === id || props.placeholder === id || props.title === id || props.value === id);
        if (!result) throw new Error(`Missing control: ${id}`);
        return result;
    };
    const change = (id: string, value: string | boolean, badInput = false) => {
        control(id).onChange?.({ target: {
            value: typeof value === 'string' ? value : '',
            checked: typeof value === 'boolean' ? value : false,
            validity: { badInput },
        } });
        render();
    };
    render();
    change('Drag file here...', '/input.tif');
    change('/path/to/output.tif', '/output.tif');
    return { onSubmit, control, change, find, html: () => renderToStaticMarkup(tree) };
}

describe('negative border point options', () => {
    it('starts disabled and submits null without displaying numeric controls', () => {
        const panel = form();
        expect(panel.control('negative-border-points').checked).toBe(false);
        expect(panel.find((props) => props.id === 'negative-border-count')).toBeUndefined();
        expect(panel.find((props) => props.id === 'negative-border-inset')).toBeUndefined();
        expect(panel.control('Run Segmentation').disabled).toBe(false);
        panel.control('Run Segmentation').onClick?.();
        expect(panel.onSubmit).toHaveBeenCalledWith(expect.objectContaining({ negativeBorderPoints: null }));
    });

    it('enables the documented defaults and submits fractional insets without rounding', () => {
        const panel = form();
        panel.change('negative-border-points', true);
        expect(panel.control('negative-border-count').value).toBe('15');
        expect(panel.control('negative-border-inset').value).toBe('12');
        panel.control('Run Segmentation').onClick?.();
        expect(panel.onSubmit).toHaveBeenLastCalledWith(expect.objectContaining({
            negativeBorderPoints: { count: 15, inset_px: 12 },
        }));
        panel.change('negative-border-count', '20');
        panel.change('negative-border-inset', '0.5');
        expect(panel.control('Run Segmentation').disabled).toBe(false);
        panel.control('Run Segmentation').onClick?.();
        expect(panel.onSubmit).toHaveBeenLastCalledWith(expect.objectContaining({
            negativeBorderPoints: { count: 20, inset_px: 0.5 },
        }));
    });

    it.each(['', '0', '-1', '1.5', 'Infinity'])('blocks an invalid count of %s only while enabled', (count) => {
        const panel = form();
        panel.change('negative-border-points', true);
        panel.change('negative-border-count', count);
        expect(panel.control('negative-border-count')['aria-invalid']).toBe(true);
        expect(panel.control('Run Segmentation').disabled).toBe(true);
        expect(panel.html()).toContain('Enter a positive whole number of border points.');
        panel.change('negative-border-points', false);
        expect(panel.control('Run Segmentation').disabled).toBe(false);
        panel.control('Run Segmentation').onClick?.();
        expect(panel.onSubmit).toHaveBeenLastCalledWith(expect.objectContaining({ negativeBorderPoints: null }));
    });

    it.each(['', '-0.5', 'Infinity', 'NaN'])('blocks an invalid inset of %s', (inset) => {
        const panel = form();
        panel.change('negative-border-points', true);
        panel.change('negative-border-inset', inset);
        expect(panel.control('negative-border-inset')['aria-invalid']).toBe(true);
        expect(panel.control('Run Segmentation').disabled).toBe(true);
        expect(panel.html()).toContain('Enter a finite, nonnegative inset in pixels.');
        panel.change('negative-border-inset', '0');
        expect(panel.control('Run Segmentation').disabled).toBe(false);
    });

    it('blocks browser bad-input values even when the string looks valid', () => {
        const panel = form();
        panel.change('negative-border-points', true);
        panel.change('negative-border-count', '15', true);
        expect(panel.control('Run Segmentation').disabled).toBe(true);
        panel.change('negative-border-count', '15');
        panel.change('negative-border-inset', '12', true);
        expect(panel.control('Run Segmentation').disabled).toBe(true);
    });
});

describe('inconsistent frame repair options', () => {
    it('starts disabled and submits null without displaying repair settings', () => {
        const panel = form();
        expect(panel.control('repair-inconsistent-frames').checked).toBe(false);
        expect(panel.find((props) => props.id === 'repair-max-area-ratio')).toBeUndefined();
        expect(panel.find((props) => props.id === 'repair-min-iou')).toBeUndefined();
        expect(panel.find((props) => props.id === 'repair-keep-unrepaired')).toBeUndefined();
        expect(panel.control('Run Segmentation').disabled).toBe(false);
        panel.control('Run Segmentation').onClick?.();
        expect(panel.onSubmit).toHaveBeenCalledWith(expect.objectContaining({ repairInconsistentFrames: null }));
    });

    it.each(['ImagePredictor', 'VideoPredictor'])('submits defaults and edited settings for %s', (predictor) => {
        const panel = form();
        panel.change('ImagePredictor', predictor);
        panel.change('repair-inconsistent-frames', true);
        expect(panel.control('repair-max-area-ratio').value).toBe('1.5');
        expect(panel.control('repair-min-iou').value).toBe('0.5');
        expect(panel.control('repair-keep-unrepaired').checked).toBe(false);
        expect(panel.control('Run Segmentation').disabled).toBe(false);
        panel.control('Run Segmentation').onClick?.();
        expect(panel.onSubmit).toHaveBeenLastCalledWith(expect.objectContaining({
            predictor_type: predictor,
            repairInconsistentFrames: { max_area_ratio: 1.5, min_iou: 0.5, keep_unrepaired: false },
        }));
        panel.change('repair-max-area-ratio', '2.25');
        panel.change('repair-min-iou', '0.75');
        panel.change('repair-keep-unrepaired', true);
        expect(panel.control('Run Segmentation').disabled).toBe(false);
        panel.control('Run Segmentation').onClick?.();
        expect(panel.onSubmit).toHaveBeenLastCalledWith(expect.objectContaining({
            repairInconsistentFrames: { max_area_ratio: 2.25, min_iou: 0.75, keep_unrepaired: true },
        }));
    });

    it.each(['0', '1'])('accepts an area ratio of 1 and inclusive IoU boundary %s', (iou) => {
        const panel = form();
        panel.change('repair-inconsistent-frames', true);
        panel.change('repair-max-area-ratio', '1');
        panel.change('repair-min-iou', iou);
        expect(panel.control('Run Segmentation').disabled).toBe(false);
        panel.control('Run Segmentation').onClick?.();
        expect(panel.onSubmit).toHaveBeenLastCalledWith(expect.objectContaining({
            repairInconsistentFrames: { max_area_ratio: 1, min_iou: Number(iou), keep_unrepaired: false },
        }));
    });

    it.each(['', '0', '0.99', 'Infinity', 'NaN'])('blocks invalid area ratio %s only while enabled', (ratio) => {
        const panel = form();
        panel.change('repair-inconsistent-frames', true);
        panel.change('repair-max-area-ratio', ratio);
        expect(panel.control('repair-max-area-ratio')['aria-invalid']).toBe(true);
        expect(panel.control('Run Segmentation').disabled).toBe(true);
        expect(panel.html()).toContain('Enter a finite area ratio of at least 1.');
        panel.change('repair-inconsistent-frames', false);
        expect(panel.control('Run Segmentation').disabled).toBe(false);
        panel.control('Run Segmentation').onClick?.();
        expect(panel.onSubmit).toHaveBeenLastCalledWith(expect.objectContaining({ repairInconsistentFrames: null }));
    });

    it.each(['', '-0.01', '1.01', 'Infinity', 'NaN'])('blocks invalid IoU %s', (iou) => {
        const panel = form();
        panel.change('repair-inconsistent-frames', true);
        panel.change('repair-min-iou', iou);
        expect(panel.control('repair-min-iou')['aria-invalid']).toBe(true);
        expect(panel.control('Run Segmentation').disabled).toBe(true);
        expect(panel.html()).toContain('Enter a finite IoU between 0 and 1.');
        panel.change('repair-min-iou', '0.5');
        expect(panel.control('Run Segmentation').disabled).toBe(false);
    });

    it('blocks browser bad-input thresholds even when their strings look valid', () => {
        const panel = form();
        panel.change('repair-inconsistent-frames', true);
        panel.change('repair-max-area-ratio', '1.5', true);
        expect(panel.control('Run Segmentation').disabled).toBe(true);
        panel.change('repair-max-area-ratio', '1.5');
        panel.change('repair-min-iou', '0.5', true);
        expect(panel.control('Run Segmentation').disabled).toBe(true);
    });
});
