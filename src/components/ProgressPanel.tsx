export type ProgressItem = { name: string; progress: number; };

const styles = {
    container: {
        backgroundColor: 'var(--panel-background)',
        height: '100%',
        display: 'flex',
        flexDirection: 'column' as const,
        overflow: 'hidden'
    },
    headerRow: {
        padding: '10px 15px',
        borderBottom: '1px solid #222',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
    },
    headerTitle: {
        color: 'var(--panel-header-text)',
        fontSize: '1rem',
        fontWeight: 600,
        letterSpacing: '0.05em'
    },
    connectionDot: (connected: boolean) => ({
        width: '10px',
        height: '10px',
        borderRadius: '50%',
        backgroundColor: connected ? '#4caf50' : '#f44336',
        boxShadow: connected ? '0 0 5px #4caf50' : 'none',
        transition: 'background-color 0.3s'
    }),
    content: {
        padding: '15px',
        overflowY: 'auto' as const,
        flex: 1
    },
    item: { marginBottom: '15px' },
    labelRow: { display:'flex', justifyContent:'space-between', color:'var(--primary-text)', fontSize:'0.9rem', marginBottom:'5px' },
    barBg: { height:'6px', background:'var(--light-background)', borderRadius:'3px', overflow:'hidden' },
    barFill: (val: number) => ({ width: `${val}%`, height:'100%', background:'var(--option-highlight-color)', transition: 'width 0.3s ease-out' })
};

export default function ProgressPanel({ items, connected }: { items: ProgressItem[], connected: boolean }) {
    return (
        <div style={styles.container}>
            <div style={styles.headerRow}>
                <span style={styles.headerTitle}>PROGRESS</span>
                <div style={styles.connectionDot(connected)} title={connected ? "Backend Connected" : "Backend Disconnected"} />
            </div>
            <div style={styles.content}>
                {items.map((i, idx) => (
                    <div key={idx} style={styles.item}>
                        <div style={styles.labelRow}><span>{i.name}</span><span>{i.progress}%</span></div>
                        <div style={styles.barBg}><div style={styles.barFill(i.progress)} /></div>
                    </div>
                ))}
                {items.length === 0 && <div style={{color:'#666', fontStyle:'italic', fontSize: '0.9rem'}}>No active tasks.</div>}
            </div>
        </div>
    );
}