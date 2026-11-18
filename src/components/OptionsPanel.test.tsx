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

describe('inconsistent frame repair options', () => {
    it('starts disabled and submits both options as false', () => {
        const panel = form();
        expect(panel.control('repair-inconsistent-frames').checked).toBe(false);
        expect(panel.find((props) => props.id === 'repair-keep-unrepaired')).toBeUndefined();
        expect(panel.control('Run Segmentation').disabled).toBe(false);
        panel.control('Run Segmentation').onClick?.();
        expect(panel.onSubmit).toHaveBeenCalledWith(expect.objectContaining({
            repairInconsistentFrames: false,
            keepUnrepaired: false,
        }));
    });

    it.each(['ImagePredictor', 'VideoPredictor'])('submits repair and optional raw output for %s', (predictor) => {
        const panel = form();
        panel.change('ImagePredictor', predictor);
        panel.change('repair-inconsistent-frames', true);
        expect(panel.control('repair-keep-unrepaired').checked).toBe(false);
        expect(panel.control('Run Segmentation').disabled).toBe(false);
        panel.control('Run Segmentation').onClick?.();
        expect(panel.onSubmit).toHaveBeenLastCalledWith(expect.objectContaining({
            predictor_type: predictor,
            repairInconsistentFrames: true,
            keepUnrepaired: false,
        }));
        panel.change('repair-keep-unrepaired', true);
        panel.control('Run Segmentation').onClick?.();
        expect(panel.onSubmit).toHaveBeenLastCalledWith(expect.objectContaining({
            repairInconsistentFrames: true,
            keepUnrepaired: true,
        }));
        panel.change('repair-keep-unrepaired', false);
        panel.control('Run Segmentation').onClick?.();
        expect(panel.onSubmit).toHaveBeenLastCalledWith(expect.objectContaining({
            repairInconsistentFrames: true,
            keepUnrepaired: false,
        }));
    });

    it('clears the raw output option when repair is disabled', () => {
        const panel = form();
        panel.change('repair-inconsistent-frames', true);
        panel.change('repair-keep-unrepaired', true);
        panel.change('repair-inconsistent-frames', false);
        expect(panel.find((props) => props.id === 'repair-keep-unrepaired')).toBeUndefined();
        panel.control('Run Segmentation').onClick?.();
        expect(panel.onSubmit).toHaveBeenLastCalledWith(expect.objectContaining({
            repairInconsistentFrames: false,
            keepUnrepaired: false,
        }));
        panel.change('repair-inconsistent-frames', true);
        expect(panel.control('repair-keep-unrepaired').checked).toBe(false);
    });
});
