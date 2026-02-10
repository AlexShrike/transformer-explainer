/* Transformer Explainer — Frontend */

const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);

let currentData = null;
let modelInfo = null;

// ─── Color Utils ─────────────────────────────────────────────
const TOKEN_COLORS = [
    '#4f8fff', '#ff6b6b', '#51cf66', '#ffd43b', '#cc5de8',
    '#ff922b', '#20c997', '#f06595', '#7950f2', '#15aabf',
];

function valToColor(val, min, max) {
    const t = Math.max(0, Math.min(1, (val - min) / (max - min + 1e-9)));
    const r = Math.round(15 + t * 64);
    const g = Math.round(23 + t * 80);
    const b = Math.round(85 + t * 170);
    return `rgb(${r},${g},${b})`;
}

function heatColor(val) {
    // 0=dark blue, 0.5=purple, 1=bright red
    const t = Math.max(0, Math.min(1, val));
    if (t < 0.5) {
        const s = t * 2;
        return `rgb(${Math.round(s * 160)}, ${Math.round(30 + s * 20)}, ${Math.round(120 + s * 80)})`;
    } else {
        const s = (t - 0.5) * 2;
        return `rgb(${Math.round(160 + s * 95)}, ${Math.round(50 - s * 30)}, ${Math.round(200 - s * 160)})`;
    }
}

// ─── Init ────────────────────────────────────────────────────
async function init() {
    try {
        const res = await fetch('/api/model-info');
        modelInfo = await res.json();
        renderModelInfo(modelInfo);
    } catch(e) {
        console.error('Failed to load model info', e);
    }
    
    $('#btn-process').addEventListener('click', doProcess);
    $('#btn-generate').addEventListener('click', doGenerate);
    $('#input-text').addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); doProcess(); }
    });
}

function renderModelInfo(info) {
    const grid = $('#model-info');
    grid.innerHTML = [
        { label: 'Vocab', value: info.vocab_size },
        { label: 'd_model', value: info.d_model },
        { label: 'Heads', value: info.n_heads },
        { label: 'Layers', value: info.n_layers },
        { label: 'd_ff', value: info.d_ff },
        { label: 'Max Len', value: info.max_len },
    ].map(i => `<div class="info-item"><span class="label">${i.label}</span><span class="value">${i.value}</span></div>`).join('');
}

// ─── Process ─────────────────────────────────────────────────
async function doProcess() {
    const text = $('#input-text').value.trim();
    if (!text) return;
    
    $('#btn-process').textContent = '⏳ Processing…';
    try {
        const res = await fetch('/api/process', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
        });
        currentData = await res.json();
        renderTokens(currentData);
        renderFlow(currentData);
        showInspector('overview', currentData);
    } catch(e) {
        console.error(e);
    }
    $('#btn-process').textContent = '⚡ Process';
}

// ─── Tokens ──────────────────────────────────────────────────
function renderTokens(data) {
    const el = $('#token-display');
    el.innerHTML = data.tokens.map((t, i) => {
        const color = TOKEN_COLORS[i % TOKEN_COLORS.length];
        const display = t === ' ' ? '␣' : t;
        return `<span class="token-chip" style="background:${color}">${display}<span class="tid">${data.token_ids[i]}</span></span>`;
    }).join('');
    $('#tokenization').classList.remove('hidden');
}

// ─── Flow Diagram ────────────────────────────────────────────
function renderFlow(data) {
    const container = $('#flow-diagram');
    container.innerHTML = '';
    
    // Input tokens
    addFlowNode(container, '📝 Input Tokens', `${data.tokens.length} characters`, 'tokens', data);
    addArrow(container);
    
    // Embedding
    addFlowNode(container, '🔢 Embedding + Positional Encoding', `${modelInfo.d_model}-dim vectors`, 'embedding', data);
    addArrow(container);
    
    // Transformer blocks
    data.blocks.forEach((block, i) => {
        const group = document.createElement('div');
        group.className = 'flow-group';
        group.innerHTML = `<div class="flow-group-title">Transformer Block ${i + 1}</div>`;
        
        const attnNode = makeFlowNode('🎯 Multi-Head Attention', `${modelInfo.n_heads} heads`, () => showInspector('attention', { block, blockIdx: i, data }));
        // Mini attention preview
        const preview = document.createElement('div');
        preview.className = 'node-preview';
        const canvas = createAttentionHeatmap(block.attention.attn_weights, data.tokens, 120);
        if (canvas) preview.appendChild(canvas);
        attnNode.appendChild(preview);
        group.appendChild(attnNode);
        
        addArrow(group);
        group.appendChild(makeFlowNode('➕ Add & LayerNorm', 'Residual connection', () => showInspector('residual1', { block, blockIdx: i, data })));
        addArrow(group);
        group.appendChild(makeFlowNode('⚙️ Feed Forward', `${modelInfo.d_model} → ${modelInfo.d_ff} → ${modelInfo.d_model}`, () => showInspector('ffn', { block, blockIdx: i, data })));
        addArrow(group);
        group.appendChild(makeFlowNode('➕ Add & LayerNorm', 'Residual connection', () => showInspector('residual2', { block, blockIdx: i, data })));
        
        container.appendChild(group);
        addArrow(container);
    });
    
    // Output
    addFlowNode(container, '📊 Output Linear + Softmax', `Top: "${data.top_predictions[0]?.char}" (${(data.top_predictions[0]?.prob * 100).toFixed(1)}%)`, 'output', data);
}

function makeFlowNode(title, subtitle, onClick) {
    const node = document.createElement('div');
    node.className = 'flow-node';
    node.innerHTML = `<div class="node-title">${title}</div><div class="node-subtitle">${subtitle}</div>`;
    if (onClick) node.addEventListener('click', () => {
        $$('.flow-node').forEach(n => n.classList.remove('active'));
        node.classList.add('active');
        onClick();
    });
    return node;
}

function addFlowNode(container, title, subtitle, type, data) {
    const node = makeFlowNode(title, subtitle, () => showInspector(type, data));
    container.appendChild(node);
}

function addArrow(container) {
    const arrow = document.createElement('div');
    arrow.className = 'flow-arrow';
    container.appendChild(arrow);
}

// ─── Attention Heatmap ───────────────────────────────────────
function createAttentionHeatmap(attnWeights, tokens, size) {
    // attnWeights: array of heads, each head is [seq x seq]
    // Show first head by default
    if (!attnWeights || !attnWeights.length) return null;
    
    const head = attnWeights[0]; // first head
    const seq = head.length;
    if (seq === 0) return null;
    
    const cellSize = Math.min(Math.floor(size / seq), 20);
    const canvas = document.createElement('canvas');
    canvas.className = 'heatmap';
    canvas.width = cellSize * seq;
    canvas.height = cellSize * seq;
    canvas.style.width = canvas.width + 'px';
    canvas.style.height = canvas.height + 'px';
    
    const ctx = canvas.getContext('2d');
    for (let i = 0; i < seq; i++) {
        for (let j = 0; j < seq; j++) {
            ctx.fillStyle = heatColor(head[i][j]);
            ctx.fillRect(j * cellSize, i * cellSize, cellSize - 1, cellSize - 1);
        }
    }
    return canvas;
}

// ─── Inspector ───────────────────────────────────────────────
const EXPLANATIONS = {
    overview: `<h3>How it works</h3><p>The transformer processes your input text character by character. Each character is converted to a number (token ID), then to a vector (embedding). These vectors flow through transformer blocks, where <b>attention</b> lets each position look at all others, and <b>feed-forward</b> layers transform the representations. Finally, a linear layer + softmax predicts the next character.</p>`,
    tokens: `<h3>Tokenization</h3><p>The input text is split into individual characters. Each character is mapped to a unique ID from the vocabulary. This is the simplest form of tokenization — real LLMs use subword tokenization (BPE), but characters make the process fully transparent.</p>`,
    embedding: `<h3>Embedding + Positional Encoding</h3><p>Each token ID is looked up in an <b>embedding table</b> — a learned matrix where each row is a vector representing a character. Since transformers have no notion of order, we add <b>sinusoidal positional encodings</b> — fixed patterns that encode each position.</p>`,
    output: `<h3>Output Projection</h3><p>The final hidden state is projected through a linear layer to produce <b>logits</b> (one score per vocabulary item). <b>Softmax</b> converts these to probabilities. The highest probability character is the model's prediction for the next character.</p>`,
};

function showInspector(type, data) {
    const el = $('#inspector');
    
    if (type === 'attention') {
        const { block, blockIdx, data: fullData } = data;
        const attn = block.attention;
        let html = `<h3>🎯 Multi-Head Attention — Block ${blockIdx + 1}</h3>`;
        html += `<p>Each head learns to attend to different patterns. The attention matrix shows how much each position "looks at" every other position. Bright = high attention.</p>`;
        
        // Show each head
        const nHeads = attn.attn_weights.length;
        for (let h = 0; h < nHeads; h++) {
            html += `<h3>Head ${h + 1}</h3>`;
            html += `<div id="attn-head-${h}" class="attn-heatmap-container"></div>`;
        }
        
        // Q, K, V info
        html += `<h3>Dimensions</h3>`;
        html += `<p>Q, K, V: [${attn.Q.length} × ${attn.Q[0].length}] → per head: [${attn.Q_heads[0].length} × ${attn.Q_heads[0][0].length}]</p>`;
        
        el.innerHTML = html;
        
        // Render heatmaps after DOM update
        for (let h = 0; h < nHeads; h++) {
            const container = $(`#attn-head-${h}`);
            if (container) {
                const canvas = createDetailedAttentionHeatmap(attn.attn_weights[h], fullData.tokens);
                if (canvas) container.appendChild(canvas);
            }
        }
        return;
    }
    
    if (type === 'ffn') {
        const { block, blockIdx } = data;
        const ff = block.feed_forward;
        let html = `<h3>⚙️ Feed Forward — Block ${blockIdx + 1}</h3>`;
        html += `<p>Two linear transformations with ReLU activation in between. This is where the model does most of its "thinking" — transforming the attention-mixed representations into richer features.</p>`;
        html += `<p>Formula: FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂</p>`;
        
        // Show activation heatmap
        html += `<h3>Activation Heatmap (post-ReLU)</h3>`;
        html += `<div id="ffn-heatmap"></div>`;
        el.innerHTML = html;
        
        const heatmapEl = $('#ffn-heatmap');
        const activations = ff.post_activation;
        if (activations && heatmapEl) {
            const canvas = createMatrixHeatmap(activations, 200);
            heatmapEl.appendChild(canvas);
        }
        return;
    }
    
    if (type === 'residual1' || type === 'residual2') {
        const { block, blockIdx } = data;
        const key = type === 'residual1' ? 'residual_after_attn' : 'residual_after_ff';
        el.innerHTML = `<h3>➕ Add & LayerNorm — Block ${blockIdx + 1}</h3>
            <p><b>Residual connection:</b> The input is added to the output of the sub-layer. This helps gradients flow during training and lets the model preserve information.</p>
            <p><b>Layer normalization:</b> The result is normalized across the feature dimension to stabilize training.</p>
            <p>Formula: LayerNorm(x + SubLayer(x))</p>`;
        return;
    }
    
    if (type === 'output' && data.top_predictions) {
        let html = EXPLANATIONS.output;
        html += `<h3>Top Predictions</h3><div class="bar-chart">`;
        data.top_predictions.forEach(p => {
            const pct = (p.prob * 100).toFixed(1);
            const display = p.char === ' ' ? '␣' : (p.char === '\n' ? '↵' : p.char);
            html += `<div class="bar-row">
                <span class="bar-label">'${display}'</span>
                <div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:var(--accent)">${pct}%</div></div>
            </div>`;
        });
        html += `</div>`;
        
        // Show logits
        html += `<h3>Raw Logits (top 10)</h3><div class="matrix-view"><code>`;
        const logits = data.final_logits;
        if (logits) {
            const sorted = logits.map((v, i) => [i, v]).sort((a, b) => b[1] - a[1]).slice(0, 10);
            sorted.forEach(([i, v]) => {
                html += `[${i}] = ${v.toFixed(3)}<br>`;
            });
        }
        html += `</code></div>`;
        el.innerHTML = html;
        return;
    }
    
    if (type === 'embedding' && data.embedding) {
        let html = EXPLANATIONS.embedding;
        html += `<h3>Token Embeddings Heatmap</h3><div id="emb-heatmap"></div>`;
        html += `<h3>Positional Encoding Heatmap</h3><div id="pe-heatmap"></div>`;
        html += `<h3>Combined (Embedding + PE)</h3><div id="combined-heatmap"></div>`;
        el.innerHTML = html;
        
        const emb = data.embedding;
        appendHeatmap('#emb-heatmap', emb.token_embeddings);
        appendHeatmap('#pe-heatmap', emb.positional_encoding);
        appendHeatmap('#combined-heatmap', emb.combined);
        return;
    }
    
    // Default
    el.innerHTML = EXPLANATIONS[type] || `<div class="inspector-placeholder">Click any component to inspect</div>`;
}

function appendHeatmap(selector, matrix) {
    const el = $(selector);
    if (el && matrix) {
        const canvas = createMatrixHeatmap(matrix, 250);
        el.appendChild(canvas);
    }
}

function createDetailedAttentionHeatmap(head, tokens) {
    const seq = head.length;
    if (seq === 0) return null;
    
    const labelW = 30;
    const cellSize = Math.min(28, Math.floor(250 / seq));
    const w = labelW + cellSize * seq;
    const h = labelW + cellSize * seq;
    
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    canvas.className = 'heatmap';
    
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#0f1729';
    ctx.fillRect(0, 0, w, h);
    
    // Labels
    ctx.fillStyle = '#8899bb';
    ctx.font = `${Math.min(11, cellSize - 2)}px monospace`;
    ctx.textAlign = 'center';
    
    for (let i = 0; i < seq; i++) {
        const display = tokens[i] === ' ' ? '␣' : tokens[i];
        ctx.fillText(display, labelW + i * cellSize + cellSize / 2, labelW - 4);
        
        ctx.save();
        ctx.translate(labelW - 4, labelW + i * cellSize + cellSize / 2 + 4);
        ctx.fillText(display, 0, 0);
        ctx.restore();
    }
    
    // Cells
    for (let i = 0; i < seq; i++) {
        for (let j = 0; j < seq; j++) {
            ctx.fillStyle = heatColor(head[i][j]);
            ctx.fillRect(labelW + j * cellSize, labelW + i * cellSize, cellSize - 1, cellSize - 1);
            
            // Show value
            if (cellSize >= 20) {
                ctx.fillStyle = head[i][j] > 0.5 ? '#fff' : '#aaa';
                ctx.font = `${Math.min(9, cellSize - 8)}px monospace`;
                ctx.textAlign = 'center';
                ctx.fillText(head[i][j].toFixed(1), labelW + j * cellSize + cellSize / 2, labelW + i * cellSize + cellSize / 2 + 3);
            }
        }
    }
    
    return canvas;
}

function createMatrixHeatmap(matrix, maxW) {
    const rows = matrix.length;
    const cols = matrix[0].length;
    
    // Find min/max
    let min = Infinity, max = -Infinity;
    for (const row of matrix) {
        for (const v of row) {
            if (v < min) min = v;
            if (v > max) max = v;
        }
    }
    
    const cellW = Math.max(2, Math.min(8, Math.floor(maxW / cols)));
    const cellH = Math.max(2, Math.min(12, Math.floor(100 / rows)));
    
    const canvas = document.createElement('canvas');
    canvas.width = cellW * cols;
    canvas.height = cellH * rows;
    canvas.style.width = Math.min(maxW, cellW * cols) + 'px';
    canvas.style.height = cellH * rows + 'px';
    canvas.className = 'heatmap';
    
    const ctx = canvas.getContext('2d');
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            const t = (matrix[i][j] - min) / (max - min + 1e-9);
            ctx.fillStyle = heatColor(t);
            ctx.fillRect(j * cellW, i * cellH, cellW, cellH);
        }
    }
    
    return canvas;
}

// ─── Generation ──────────────────────────────────────────────
async function doGenerate() {
    const text = $('#input-text').value.trim();
    if (!text) return;
    
    $('#btn-generate').textContent = '⏳ Generating…';
    try {
        const res = await fetch('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, max_tokens: 30 }),
        });
        const data = await res.json();
        
        const genEl = $('#generation-output');
        genEl.classList.remove('hidden');
        
        const genText = $('#gen-text');
        genText.innerHTML = `<span class="gen-original">${escHtml(data.input)}</span><span class="gen-new">${escHtml(data.output.slice(data.input.length))}</span>`;
        
        // Show step details
        const stepsEl = $('#gen-steps');
        stepsEl.innerHTML = '<h3>Generation Steps</h3>';
        data.steps.forEach((step, i) => {
            const div = document.createElement('div');
            div.className = 'info-item';
            div.style.marginBottom = '4px';
            const topPreds = step.top_predictions.map(p => `'${p.char === ' ' ? '␣' : p.char}' ${(p.prob * 100).toFixed(0)}%`).join(', ');
            div.innerHTML = `<span class="label">Step ${i + 1}</span> → <b>${step.predicted_char === ' ' ? '␣' : step.predicted_char}</b> <span style="color:var(--text-dim);font-size:0.7rem">(${topPreds})</span>`;
            stepsEl.appendChild(div);
        });
    } catch(e) {
        console.error(e);
    }
    $('#btn-generate').textContent = '🔄 Generate';
}

function escHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ─── Boot ────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', init);
