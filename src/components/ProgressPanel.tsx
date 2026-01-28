import styles from '../assets/styles.module.css';
export type ProgressItem = { name: string; progress: number; };

export default function ProgressPanel({ items, connected }: { items: ProgressItem[], connected: boolean }) {
    return (
        <div className={styles.container}>
            <div className={styles.headerRow}>
                <span className={styles.headerTitle}>PROGRESS</span>
                <div 
                    className={`${styles.connectionDot} ${connected ? styles.connected : ''}`} 
                    title={connected ? "Backend Connected" : "Backend Disconnected"} 
                />
            </div>
            <div className={styles.progressContent}>
                {items.map((i, idx) => (
                    <div key={idx} className={styles.progressItem}>
                        <div className={styles.progressLabelRow}><span>{i.name}</span><span>{i.progress}%</span></div>
                        <div className={styles.progressBarBg}>
                            <div className={styles.progressBarFill} style={{ width: `${i.progress}%` }} />
                        </div>
                    </div>
                ))}
                {items.length === 0 && <div style={{color:'#666', fontStyle:'italic', fontSize: '0.9rem'}}>No active tasks.</div>}
            </div>
        </div>
    );
}