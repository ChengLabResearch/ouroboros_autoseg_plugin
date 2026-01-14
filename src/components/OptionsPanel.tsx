import React, { useState } from 'react';

type Props = { 
    onSubmit: (d: any) => void; 
    isRunning: boolean; 
};

const styles = {
    container: {
        backgroundColor: 'var(--panel-background)',
        height: '100%',
        display: 'flex',
        flexDirection: 'column' as const,
        borderTop: '1px solid #111'
    },
    headerRow: {
        padding: '10px 15px',
        backgroundColor: 'var(--panel-background)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        borderBottom: '1px solid #222',
        marginTop: '10px'
    },
    headerTitle: {
        color: 'var(--panel-header-text)',
        fontSize: '1rem',
        fontWeight: 600,
        letterSpacing: '0.05em'
    },
    section: {
        padding: '15px',
        borderBottom: '1px solid #222'
    },
    playBtn: (disabled: boolean) => ({
        background: 'none',
        border: 'none',
        color: disabled ? 'var(--inactive-menu-text)' : 'var(--option-highlight-color)',
        cursor: disabled ? 'default' : 'pointer',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '5px'
    }),
    downloadBtn: (disabled: boolean) => ({
        padding: '6px 12px',
        backgroundColor: disabled ? '#444' : 'var(--option-highlight-color)',
        color: disabled ? '#888' : 'var(--dark-text)',
        border: 'none',
        borderRadius: '4px',
        fontSize: '0.8rem',
        fontWeight: 'bold',
        cursor: disabled ? 'default' : 'pointer',
        width: '100px',
        textAlign: 'center' as const
    }),
    row: {
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '10px',
        minHeight: '30px'
    },
    label: {
        color: 'var(--primary-text)',
        fontSize: '0.9rem',
        flex: 1
    },
    inputContainer: {
        flex: 1.5,
        display: 'flex',
        justifyContent: 'flex-end',
        alignItems: 'center',
        gap: '10px'
    },
    input: {
        width: '100%',
        background: 'transparent',
        border: 'none',
        borderBottom: '1px solid var(--inactive-menu-text)',
        color: 'var(--primary-text)',
        textAlign: 'right' as const,
        padding: '4px',
        fontSize: '0.9rem',
        outline: 'none'
    },
    select: {
        width: '100%',
        background: 'transparent',
        border: 'none',
        color: 'var(--primary-text)',
        textAlign: 'right' as const,
        fontSize: '0.9rem',
        appearance: 'none' as const,
        cursor: 'pointer'
    }
};

const BACKEND_URL = "http://localhost:8686";

export default function OptionsPanel({ onSubmit, isRunning }: Props) {
    const [fp, setFp] = useState('');
    const [outFp, setOutFp] = useState('');
    const [model, setModel] = useState('sam2_hiera_base_plus');
	const [predictor, setPredictor] = useState('ImagePredictor')
    const [token, setToken] = useState('');

    // Download States
    const [sam2Status, setSam2Status] = useState('Download');
    const [sam3Status, setSam3Status] = useState('Download');

    const handleDrop = (e: React.DragEvent, setFn: (s:string)=>void) => {
        e.preventDefault();
        e.stopPropagation();
        const data = e.dataTransfer.getData("text/plain");
        if (data) setFn(data);
    };

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
    };

    const downloadModel = async (modelType: string, hfToken: string, setStatus: (s:string)=>void) => {
        setStatus('Downloading...');
        try {
            const res = await fetch(`${BACKEND_URL}/download-model`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_type: modelType, hf_token: hfToken })
            });
            if (res.ok) setStatus('Done');
            else {
                console.error(await res.text());
                setStatus('Error');
            }
        } catch (e) {
            console.error(e);
            setStatus('Error');
        }
    };

    const canRun = !isRunning && fp && outFp;

    return (
        <div style={styles.container}>
            <div style={{overflowY: 'auto', flex: 1}}>
                
                {/* --- OPTIONS SECTION --- */}
                <div style={styles.headerRow}>
                    <span style={styles.headerTitle}>OPTIONS</span>
                    <button 
                        style={styles.playBtn(!canRun)} 
                        onClick={() => onSubmit({filePath: fp, outputFile: outFp, modelType: model, predictor_type: predictor})} 
                        disabled={!canRun}
                        title="Run Segmentation"
                    >
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z" /></svg>
                    </button>
                </div>

                <div style={styles.section}>
                    <div style={styles.row}>
                        <span style={styles.label}>Input File</span>
                        <div style={styles.inputContainer}>
                            <input style={styles.input} value={fp} onChange={e=>setFp(e.target.value)} placeholder="Drag file here..." onDrop={(e)=>handleDrop(e, setFp)} onDragOver={handleDragOver} />
                        </div>
                    </div>

                    <div style={styles.row}>
                        <span style={styles.label}>Output File</span>
                        <div style={styles.inputContainer}>
                            <input style={styles.input} value={outFp} onChange={e=>setOutFp(e.target.value)} placeholder="/path/to/output.tif" onDrop={(e)=>handleDrop(e, setOutFp)} onDragOver={handleDragOver} />
                        </div>
                    </div>

                    <div style={styles.row}>
                        <span style={styles.label}>Predictor</span>
                        <div style={styles.inputContainer}>
                            <select style={styles.select} value={predictor} onChange={e=>setPredictor(e.target.value)}>
								<option value="ImagePredictor">Image Predictor</option>
								<option value="VideoPredictor">Video Predictor</option>
                            </select>
                        </div>
                    </div>

                    <div style={styles.row}>
                        <span style={styles.label}>Model</span>
                        <div style={styles.inputContainer}>
                            <select style={styles.select} value={model} onChange={e=>setModel(e.target.value)}>
                                <optgroup label="SAM 2">
                                    <option value="sam2_hiera_base_plus">SAM2 Base+</option>
                                    <option value="sam2_hiera_large">SAM2 Large</option>
                                </optgroup>
                                <optgroup label="SAM 3">
                                    <option value="sam3">SAM3</option>
                                </optgroup>
                            </select>
                        </div>
                    </div>
                </div>

                {/* --- MODELS SECTION --- */}
                <div style={styles.headerRow}>
                    <span style={styles.headerTitle}>MODELS</span>
                </div>

                <div style={styles.section}>
                    {/* SAM 2 */}
                    <div style={styles.row}>
                        <span style={styles.label}>SAM 2 (Official)</span>
                        <button 
                            style={styles.downloadBtn(sam2Status !== 'Download')} 
                            onClick={() => downloadModel('sam2_hiera_base_plus', '', setSam2Status)}
                            disabled={sam2Status !== 'Download'}
                        >
                            {sam2Status}
                        </button>
                    </div>

                    {/* SAM 3 */}
                    <div style={{marginTop: '20px'}}>
                        <div style={styles.row}>
                            <span style={styles.label}>SAM 3 (Authenticated)</span>
                        </div>
                        <div style={styles.row}>
                            <input 
                                style={{...styles.input, textAlign: 'left', marginRight: '10px'}} 
                                type="password" 
                                value={token} 
                                onChange={e=>setToken(e.target.value)} 
                                placeholder="Paste HF Token" 
                            />
                            <button 
                                style={styles.downloadBtn(sam3Status !== 'Download' || !token)} 
                                onClick={() => downloadModel('sam3_hiera_base', token, setSam3Status)}
                                disabled={sam3Status !== 'Download' || !token}
                            >
                                {sam3Status}
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}