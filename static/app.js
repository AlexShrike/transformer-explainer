/* Transformer Explainer — Frontend (Redesigned, No Emojis, High Contrast) */

const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);

let currentData = null;
let modelInfo = null;
let stepMode = false;
let currentStep = 0;
let totalSteps = 0;
let flowNodesList = [];

// ─── Token Colors (higher contrast) ─────────────────────────
const TOKEN_COLORS = [
    '#2b6cb0', '#c0392b', '#1a8a5a', '#c57d0a', '#7c3aed',
    '#c44536', '#0e7c7b', '#a83279', '#5b21b6', '#1a6fa0',
];

// ─── Heatmap Colors (higher contrast blue-orange) ────────────
function heatColor(val) {
    const t = Math.max(0, Math.min(1, val));
    if (t < 0.5) {
        const s = t * 2;
        const r = Math.round(220 - s * 80);
        const g = Math.round(225 - s * 80);
        const b = Math.round(245 - s * 40);
        return `rgb(${r},${g},${b})`;
    } else {
        const s = (t - 0.5) * 2;
        const r = Math.round(140 + s * 100);
        const g = Math.round(145 - s * 110);
        const b = Math.round(205 - s * 180);
        return `rgb(${r},${g},${b})`;
    }
}

// ─── Color Legend HTML ───────────────────────────────────────
function colorLegendHTML(lowLabel, highLabel) {
    return `<div class="color-legend">
        <span class="color-legend-label">${lowLabel}</span>
        <canvas class="color-legend-bar" width="120" height="12" id="legend-${Math.random().toString(36).slice(2,8)}"></canvas>
        <span class="color-legend-label">${highLabel}</span>
    </div>`;
}

function paintLegend(canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    for (let x = 0; x < 120; x++) {
        ctx.fillStyle = heatColor(x / 119);
        ctx.fillRect(x, 0, 1, 12);
    }
}

// ─── Dynamic Explanations ────────────────────────────────────
function flowExplanation(type, data) {
    if (!data || !modelInfo) return '';
    const input = data.tokens.join('');
    const n = data.tokens.length;
    const tokenList = data.tokens.map((t, i) => `'${t === ' ' ? '(space)' : t}' (ID ${data.token_ids[i]})`).join(', ');
    const d = modelInfo.d_model;
    const dk = Math.floor(d / modelInfo.n_heads);
    const lastChar = data.tokens[n - 1] === ' ' ? '(space)' : data.tokens[n - 1];
    const topPred = data.top_predictions && data.top_predictions[0];

    switch (type) {
        case 'tokens':
            return `We split '${escHtml(input)}' into ${n} tokens: ${tokenList}. Each character gets a unique number -- the model only understands numbers.`;
        case 'embedding':
            return `Each of the ${n} token IDs is converted into a vector of ${d} numbers (d_model = ${d}: the size of each token's vector -- like how many 'features' describe each token). We also add positional encoding -- a mathematical pattern that tells the model '${data.tokens[0]}' is at position 1, '${data.tokens[Math.min(1, n-1)]}' is at position 2, etc. Without this, the model would have no sense of order.`;
        case 'attention':
            return `Attention with ${modelInfo.n_heads} heads (n_heads = ${modelInfo.n_heads}: number of parallel attention mechanisms) means we run ${modelInfo.n_heads} SEPARATE attention mechanisms in parallel (at the same time). Each head can learn different relationships between tokens. For each token, we compute Query ('What am I looking for?', ${dk} numbers), Key ('What do I contain?', ${dk} numbers), and Value ('If selected, here is my information', ${dk} numbers). d_k = ${dk} (dimension per head = d_model / n_heads = ${d} / ${modelInfo.n_heads}). The 'parallel' part means all heads compute simultaneously, then their results are combined.`;
        case 'residual1':
        case 'residual2':
            return `Residual connection: we ADD the input of this sublayer back to its output. If the attention/FFN learned nothing useful for '${escHtml(input)}', the original signal still passes through unchanged. LayerNorm: normalizes values to prevent them from growing too large or too small -- keeps training stable.`;
        case 'ffn':
            return `Each token passes through a 2-layer neural network independently. It expands from d_model (${d}) to d_ff (${modelInfo.d_ff}: the inner width -- temporarily expanded for more computational power), applies ReLU (sets negatives to zero), then compresses back to ${d}. This is where much of the model's 'thinking' happens for each token in '${escHtml(input)}'.`;
        case 'output': {
            const topChar = topPred ? (topPred.char === ' ' ? '(space)' : topPred.char) : '?';
            const topProb = topPred ? (topPred.prob * 100).toFixed(1) : '?';
            const top3 = (data.top_predictions || []).slice(0, 3).map(p => `'${p.char === ' ' ? '(space)' : p.char}' ${(p.prob*100).toFixed(1)}%`).join(', ');
            return `For the last token '${lastChar}', we project its ${d}-dimensional vector to ${modelInfo.vocab_size} scores (vocab_size = ${modelInfo.vocab_size}: one per character the model knows). Softmax converts raw scores into probabilities that sum to 1.0. Top predictions: ${top3}. The model predicts '${topChar}' at ${topProb}%.`;
        }
        default:
            return '';
    }
}

const INSPECTOR_EXPLANATIONS = {
    overview: {
        title: 'How Transformers Work',
        body: `<p>A transformer processes text in a series of steps. Each step transforms the representation of every token, building up richer understanding.</p>
        <p>The key innovation is <strong>attention</strong> -- a mechanism that lets each token look at all other tokens to determine relevance. It lets the model consider relationships between ALL positions simultaneously, unlike older models that read left-to-right.</p>
        <p><strong>Click any component</strong> in the flow diagram to see its details with your actual input, or use the <strong>step-by-step controls</strong> to walk through one piece at a time.</p>`
    }
};

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
    $('#btn-next-step').addEventListener('click', () => navigateStep(1));
    $('#btn-prev-step').addEventListener('click', () => navigateStep(-1));
    $('#btn-show-all').addEventListener('click', showAllSteps);
    $('#btn-close-gen').addEventListener('click', () => $('#generation-banner').classList.add('hidden'));
}

function renderModelInfo(info) {
    const grid = $('#model-info');
    const items = [
        { label: 'Vocab Size', value: info.vocab_size, tip: 'Number of unique tokens (characters) the model knows' },
        { label: 'Embedding Dim', value: info.d_model, tip: 'Size of each token\'s vector representation' },
        { label: 'Attn Heads', value: info.n_heads, tip: 'Number of parallel attention mechanisms per layer' },
        { label: 'Layers', value: info.n_layers, tip: 'Number of transformer blocks stacked together' },
        { label: 'FFN Width', value: info.d_ff, tip: 'Width of the feed-forward hidden layer' },
        { label: 'Max Length', value: info.max_len, tip: 'Maximum number of tokens the model can process at once' },
    ];
    grid.innerHTML = items.map(i =>
        `<div class="info-item tooltip" data-tip="${i.tip}"><span class="label">${i.label}</span><span class="value">${i.value}</span></div>`
    ).join('');
}

// ─── Process ─────────────────────────────────────────────────
async function doProcess() {
    const text = $('#input-text').value.trim();
    if (!text) return;

    $('#btn-process').textContent = 'Processing...';
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
        enableStepMode();
    } catch(e) {
        console.error(e);
    }
    $('#btn-process').textContent = 'Process';
}

// ─── Tokens ──────────────────────────────────────────────────
function renderTokens(data) {
    const el = $('#token-display');
    el.innerHTML = data.tokens.map((t, i) => {
        const color = TOKEN_COLORS[i % TOKEN_COLORS.length];
        const display = t === ' ' ? '(space)' : t;
        return `<span class="token-chip" style="background:${color}" title="Token ID ${data.token_ids[i]}: This character is represented as number ${data.token_ids[i]} in the vocabulary">${display}<span class="tid">ID: ${data.token_ids[i]}</span></span>`;
    }).join('');
    $('#tokenization').classList.remove('hidden');
}

// ─── Flow Diagram ────────────────────────────────────────────
function renderFlow(data) {
    const container = $('#flow-diagram');
    container.innerHTML = '';
    flowNodesList = [];

    addFlowNode(container, 'Step 1', 'Input Tokens', `${data.tokens.length} tokens`, flowExplanation('tokens', data), 'tokens', data);
    addArrow(container);
    addFlowNode(container, 'Step 2', 'Embedding + Positional Encoding', `${modelInfo.d_model}-dim vectors`, flowExplanation('embedding', data), 'embedding', data);
    addArrow(container);

    let stepNum = 3;
    data.blocks.forEach((block, i) => {
        const group = document.createElement('div');
        group.className = 'flow-group';
        group.setAttribute('data-step-group', '');
        group.innerHTML = `<div class="flow-group-title">Transformer Block ${i + 1}</div>`;

        const attnNode = makeFlowNode(`Step ${stepNum}`, 'Multi-Head Attention', `${modelInfo.n_heads} heads — each token decides how relevant every other token is`, flowExplanation('attention', data),
            () => showInspector('attention', { block, blockIdx: i, data }));
        const preview = document.createElement('div');
        preview.className = 'node-preview';
        const canvas = createAttentionHeatmap(block.attention.attn_weights, data.tokens, 140);
        if (canvas) preview.appendChild(canvas);
        attnNode.appendChild(preview);
        group.appendChild(attnNode);
        flowNodesList.push(attnNode);
        stepNum++;

        addArrow(group);
        const r1 = makeFlowNode(`Step ${stepNum}`, 'Add & LayerNorm', 'Residual connection -- preserves original signal', flowExplanation('residual1', data),
            () => showInspector('residual1', { block, blockIdx: i, data }));
        group.appendChild(r1);
        flowNodesList.push(r1);
        stepNum++;

        addArrow(group);
        const ffn = makeFlowNode(`Step ${stepNum}`, 'Feed Forward Network', `${modelInfo.d_model} -> ${modelInfo.d_ff} -> ${modelInfo.d_model} (expand then compress)`, flowExplanation('ffn', data),
            () => showInspector('ffn', { block, blockIdx: i, data }));
        group.appendChild(ffn);
        flowNodesList.push(ffn);
        stepNum++;

        addArrow(group);
        const r2 = makeFlowNode(`Step ${stepNum}`, 'Add & LayerNorm', 'Residual connection -- preserves original signal', flowExplanation('residual2', data),
            () => showInspector('residual2', { block, blockIdx: i, data }));
        group.appendChild(r2);
        flowNodesList.push(r2);
        stepNum++;

        container.appendChild(group);
        addArrow(container);
    });

    addFlowNode(container, `Step ${stepNum}`, 'Output Linear + Softmax', `Top prediction: "${data.top_predictions[0]?.char}" (${(data.top_predictions[0]?.prob * 100).toFixed(1)}%)`, flowExplanation('output', data), 'output', data);
}

function makeFlowNode(stepLabel, title, subtitle, explanation, onClick) {
    const node = document.createElement('div');
    node.className = 'flow-node';
    node.innerHTML = `
        <div class="node-header"><span class="node-step-label">${stepLabel}</span><span class="node-title">${title}</span></div>
        <div class="node-subtitle">${subtitle}</div>
        <div class="node-explanation">${explanation}</div>`;
    if (onClick) node.addEventListener('click', () => {
        $$('.flow-node').forEach(n => n.classList.remove('active'));
        node.classList.add('active');
        onClick();
    });
    return node;
}

function addFlowNode(container, stepLabel, title, subtitle, explanation, type, data) {
    const node = makeFlowNode(stepLabel, title, subtitle, explanation, () => showInspector(type, data));
    container.appendChild(node);
    flowNodesList.push(node);
}

function addArrow(container) {
    const arrow = document.createElement('div');
    arrow.className = 'flow-arrow animated';
    container.appendChild(arrow);
}

// ─── Step-by-Step Mode ───────────────────────────────────────
function enableStepMode() {
    totalSteps = flowNodesList.length;
    currentStep = totalSteps;
    stepMode = false;
    $('#step-controls').classList.remove('hidden');
    showAllSteps();
}

function navigateStep(delta) {
    if (!stepMode) {
        stepMode = true;
        currentStep = delta > 0 ? 0 : totalSteps - 1;
    } else {
        currentStep = Math.max(0, Math.min(totalSteps - 1, currentStep + delta));
    }
    updateStepView();
}

function showAllSteps() {
    stepMode = false;
    currentStep = totalSteps;
    flowNodesList.forEach(n => {
        n.classList.remove('step-hidden', 'step-current');
    });
    $$('.flow-group').forEach(g => g.classList.remove('step-hidden'));
    updateStepIndicator();
}

function updateStepView() {
    flowNodesList.forEach((n, i) => {
        n.classList.remove('step-current', 'step-hidden');
        if (i > currentStep) n.classList.add('step-hidden');
        if (i === currentStep) n.classList.add('step-current');
    });
    $$('.flow-group').forEach(group => {
        const children = Array.from(group.querySelectorAll('.flow-node'));
        const allHidden = children.every(c => c.classList.contains('step-hidden'));
        group.classList.toggle('step-hidden', allHidden);
    });
    updateStepIndicator();

    if (flowNodesList[currentStep]) {
        flowNodesList[currentStep].click();
        flowNodesList[currentStep].scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
}

function updateStepIndicator() {
    $('#step-indicator').textContent = stepMode ? `Step ${currentStep + 1} of ${totalSteps}` : `All ${totalSteps} steps`;
    $('#btn-prev-step').disabled = stepMode && currentStep === 0;
    $('#btn-next-step').disabled = stepMode && currentStep === totalSteps - 1;
}

// ─── Attention Heatmap ───────────────────────────────────────
function createAttentionHeatmap(attnWeights, tokens, size) {
    if (!attnWeights || !attnWeights.length) return null;
    const head = attnWeights[0];
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
function showInspector(type, data) {
    const el = $('#inspector');

    if (type === 'attention') {
        const { block, blockIdx, data: fullData } = data;
        const attn = block.attention;
        const tokens = fullData.tokens;
        const input = tokens.join('');
        const dk = Math.floor(modelInfo.d_model / modelInfo.n_heads);
        let html = `<h3>Multi-Head Attention -- Block ${blockIdx + 1}</h3>`;
        html += `<p><strong>Multi-Head Attention</strong> means we run ${modelInfo.n_heads} SEPARATE attention mechanisms (called "heads") in parallel (at the same time) on '${escHtml(input)}'. n_heads (${modelInfo.n_heads}): number of parallel attention mechanisms. Each head can learn different relationships between the ${tokens.length} tokens.</p>`;
        html += `<p><strong>How it works for '${escHtml(input)}':</strong> Each cell in the heatmap shows how much one token 'pays attention to' another. Row = the token asking, Column = the token being attended to. Each row sums to 1.0 (probabilities via softmax).</p>`;
        html += `<p><strong>Q, K, V for each token:</strong></p><ul style="font-size:0.82rem;margin:0.3rem 0;">`;
        tokens.forEach((t, i) => {
            const display = t === ' ' ? '(space)' : t;
            html += `<li>Token '${display}' at position ${i}: Query = 'What am I looking for?' (${dk} numbers), Key = 'What do I contain?' (${dk} numbers), Value = 'If selected, here is my info' (${dk} numbers)</li>`;
        });
        html += `</ul>`;
        html += `<p>d_k (${dk}): dimension per attention head = d_model / n_heads = ${modelInfo.d_model} / ${modelInfo.n_heads}.</p>`;
        html += `<div class="formula">Attention(Q,K,V) = softmax(QK^T / sqrt(${dk})) * V</div>`;
        html += colorLegendHTML('Low attention', 'High attention');

        const nHeads = attn.attn_weights.length;
        for (let h = 0; h < nHeads; h++) {
            html += `<h3>Head ${h + 1}</h3>`;
            // Show a notable attention pair for this head
            const headW = attn.attn_weights[h];
            let maxVal = 0, maxI = 0, maxJ = 0;
            for (let i = 0; i < headW.length; i++) for (let j = 0; j < headW[i].length; j++) {
                if (headW[i][j] > maxVal) { maxVal = headW[i][j]; maxI = i; maxJ = j; }
            }
            const fromT = tokens[maxI] === ' ' ? '(space)' : tokens[maxI];
            const toT = tokens[maxJ] === ' ' ? '(space)' : tokens[maxJ];
            html += `<p style="font-size:0.78rem;color:var(--text-dim)">Strongest connection: '${fromT}' (pos ${maxI}) attends to '${toT}' (pos ${maxJ}) at ${(maxVal*100).toFixed(1)}%.</p>`;
            html += `<div id="attn-head-${h}" class="attn-heatmap-container"></div>`;
        }
        html += `<h3>Matrix Dimensions</h3>`;
        html += `<p>Q, K, V: [${attn.Q.length} tokens x ${attn.Q[0].length} total dims], split per head: [${attn.Q_heads[0].length} tokens x ${attn.Q_heads[0][0].length} dims (d_k=${dk})]</p>`;

        el.innerHTML = html;

        // Paint legends
        el.querySelectorAll('.color-legend-bar').forEach(c => {
            const ctx = c.getContext('2d');
            for (let x = 0; x < 120; x++) { ctx.fillStyle = heatColor(x / 119); ctx.fillRect(x, 0, 1, 12); }
        });

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
        const { block, blockIdx, data: fullData } = data;
        const ff = block.feed_forward;
        const input = fullData ? fullData.tokens.join('') : '';
        let html = `<h3>Feed Forward Network -- Block ${blockIdx + 1}</h3>`;
        html += `<p>Each of the ${fullData ? fullData.tokens.length : '?'} tokens in '${escHtml(input)}' passes through a 2-layer neural network independently (tokens do not interact here).</p>`;
        html += `<p><strong>Layer 1:</strong> Expands from d_model (${modelInfo.d_model}: the size of each token's vector) to d_ff (${modelInfo.d_ff}: the inner width -- temporarily expanded for more computational power). That is a ${(modelInfo.d_ff / modelInfo.d_model).toFixed(1)}x expansion.</p>`;
        html += `<p><strong>ReLU:</strong> Sets all negative values to zero. This non-linearity lets the network learn complex patterns.</p>`;
        html += `<p><strong>Layer 2:</strong> Compresses back from ${modelInfo.d_ff} to ${modelInfo.d_model}.</p>`;
        html += `<div class="formula">FFN(x) = ReLU(x * W1 + b1) * W2 + b2</div>`;
        html += `<h3>Activation Heatmap (post-ReLU)</h3>`;
        html += `<p>This shows which neurons activated. Green = positive (activated neuron), dark = zero (filtered out by ReLU). The pattern of activations encodes learned features.</p>`;
        html += colorLegendHTML('Zero (filtered)', 'Strong activation');
        html += `<div id="ffn-heatmap"></div>`;
        el.innerHTML = html;

        el.querySelectorAll('.color-legend-bar').forEach(c => {
            const ctx = c.getContext('2d');
            for (let x = 0; x < 120; x++) { ctx.fillStyle = heatColor(x / 119); ctx.fillRect(x, 0, 1, 12); }
        });

        const heatmapEl = $('#ffn-heatmap');
        if (ff.post_activation && heatmapEl) {
            heatmapEl.appendChild(createMatrixHeatmap(ff.post_activation, 280));
        }
        return;
    }

    if (type === 'residual1' || type === 'residual2') {
        const { block, blockIdx, data: fullData } = data;
        const sub = type === 'residual1' ? 'Attention' : 'Feed-Forward';
        const input = fullData ? fullData.tokens.join('') : '';
        el.innerHTML = `<h3>Add & LayerNorm -- Block ${blockIdx + 1}</h3>
            <p>After the ${sub} sublayer processes '${escHtml(input)}':</p>
            <p><strong>Residual connection:</strong> We ADD the sublayer's input back to its output. If the ${sub} learned nothing useful for the tokens in '${escHtml(input)}', the original signal still passes through unchanged. This creates a "shortcut" that helps information and gradients flow during training.</p>
            <p><strong>Layer normalization (LayerNorm):</strong> Normalizes values across the ${modelInfo.d_model} feature dimensions to have zero mean and unit variance. This prevents values from growing too large or too small, keeping training stable.</p>
            <div class="formula">output = LayerNorm(x + SubLayer(x))</div>
            <p>Without residual connections, deep transformers would be nearly impossible to train -- gradients would vanish in deeper layers.</p>`;
        return;
    }

    if (type === 'output' && data.top_predictions) {
        const lastChar = data.tokens[data.tokens.length - 1] === ' ' ? '(space)' : data.tokens[data.tokens.length - 1];
        let html = `<h3>Output Projection + Softmax</h3>`;
        html += `<p>For the last token '${lastChar}' in '${escHtml(data.tokens.join(''))}', we project its ${modelInfo.d_model}-dimensional vector (d_model = ${modelInfo.d_model}) through a linear layer to produce ${modelInfo.vocab_size} scores called <strong>logits</strong> (raw, unnormalized scores -- one per character in the vocabulary). vocab_size (${modelInfo.vocab_size}): total number of unique characters the model knows.</p>`;
        html += `<p><strong>Softmax</strong> converts these raw scores into probabilities that sum to 1.0. The highest probability character is the model's prediction for the next character.</p>`;
        html += `<div class="formula">P(next_token = i) = exp(logit_i) / sum of exp(logit_j)</div>`;
        html += `<h3>Top Predictions</h3>`;
        html += `<p style="font-size:0.82rem;color:var(--text-secondary)">These are the model's predictions for the next character. Higher bar = more confident.</p>`;
        html += `<div class="bar-chart">`;
        data.top_predictions.forEach(p => {
            const pct = (p.prob * 100).toFixed(1);
            const display = p.char === ' ' ? '(space)' : (p.char === '\n' ? '(newline)' : p.char);
            const color = p === data.top_predictions[0] ? 'var(--accent)' : '#78909c';
            html += `<div class="bar-row">
                <span class="bar-label">'${display}'</span>
                <div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${color}">${pct}%</div></div>
            </div>`;
        });
        html += `</div>`;

        const logits = data.final_logits;
        if (logits) {
            html += `<h3>Raw Logits (top 10)</h3>`;
            html += `<p style="font-size:0.78rem;color:var(--text-dim)">Logits are the raw scores before softmax converts them to probabilities. Higher logit = higher probability.</p>`;
            html += `<div class="matrix-view">`;
            const sorted = logits.map((v, i) => [i, v]).sort((a, b) => b[1] - a[1]).slice(0, 10);
            sorted.forEach(([i, v]) => {
                const color = v > 0 ? 'var(--positive)' : 'var(--negative)';
                html += `<span style="color:${color}">[${i}] = ${v.toFixed(3)}</span><br>`;
            });
            html += `</div>`;
        }
        el.innerHTML = html;
        return;
    }

    if (type === 'embedding' && data.embedding) {
        const input = data.tokens.join('');
        const firstTok = data.tokens[0] === ' ' ? '(space)' : data.tokens[0];
        let html = `<h3>Embedding + Positional Encoding</h3>`;
        html += `<p>Each of the ${data.tokens.length} token IDs from '${escHtml(input)}' is looked up in an embedding table -- a learned matrix where row i is the vector for token i. Think of it as converting the simple number ${data.token_ids[0]} for '${firstTok}' into a rich description with d_model (${modelInfo.d_model}: the size of each token's vector -- like how many 'features' describe each token) dimensions.</p>`;
        html += `<p><strong>Positional encoding:</strong> These values encode WHERE the token is in the sequence. Since transformers process all tokens in parallel (no left-to-right!), they have no built-in sense of order. The sinusoidal encoding gives each position a unique signature: '${data.tokens[0]}' at position 0, '${data.tokens[Math.min(1, data.tokens.length-1)]}' at position 1, etc.</p>`;
        html += `<div class="formula">combined[i] = embedding[token_id] + positional_encoding[position]</div>`;
        html += `<h3>Token Embeddings Heatmap</h3>`;
        html += `<p>Each row is one token from '${escHtml(input)}'. Each cell is one of the ${modelInfo.d_model} dimensions. Together, these numbers represent the token's meaning. Similar patterns = similar meanings.</p>`;
        html += colorLegendHTML('Low value', 'High value');
        html += `<div id="emb-heatmap"></div>`;
        html += `<h3>Positional Encoding Heatmap</h3>`;
        html += `<p>These values encode WHERE the token is in the sequence. Notice the wave-like pattern — this is a sinusoidal encoding. Each position has a unique signature.</p>`;
        html += colorLegendHTML('Negative', 'Positive');
        html += `<div id="pe-heatmap"></div>`;
        html += `<h3>Combined (Embedding + Position)</h3>`;
        html += `<p>The final input to the transformer: token meaning + position information combined.</p>`;
        html += colorLegendHTML('Low', 'High');
        html += `<div id="combined-heatmap"></div>`;
        el.innerHTML = html;

        el.querySelectorAll('.color-legend-bar').forEach(c => {
            const ctx = c.getContext('2d');
            for (let x = 0; x < 120; x++) { ctx.fillStyle = heatColor(x / 119); ctx.fillRect(x, 0, 1, 12); }
        });

        appendHeatmap('#emb-heatmap', data.embedding.token_embeddings);
        appendHeatmap('#pe-heatmap', data.embedding.positional_encoding);
        appendHeatmap('#combined-heatmap', data.embedding.combined);
        return;
    }

    if (type === 'tokens' && data && data.tokens) {
        const input = data.tokens.join('');
        const tokenList = data.tokens.map((t, i) => `'${t === ' ' ? '(space)' : t}' = ID ${data.token_ids[i]}`).join(', ');
        el.innerHTML = `<h3>Tokenization</h3>
            <p>We split '${escHtml(input)}' into ${data.tokens.length} individual character tokens: ${tokenList}.</p>
            <p><strong>Token ID:</strong> Each character is mapped to a unique number from the vocabulary (vocab_size = ${modelInfo.vocab_size}: total number of unique characters the model knows). The model only works with numbers, not letters. These IDs are indices into the vocabulary table.</p>
            <p>This is the simplest form of tokenization. Real LLMs like GPT use subword tokenization (BPE -- Byte-Pair Encoding groups frequent character pairs into single tokens, e.g. 'th' becomes one token), but characters make the process fully transparent.</p>`;
        return;
    }

    // Default: use inspector explanations
    const exp = INSPECTOR_EXPLANATIONS[type];
    if (exp) {
        el.innerHTML = `<h3>${exp.title}</h3>${exp.body}`;
    } else {
        el.innerHTML = `<div class="inspector-placeholder"><p>Click any component to inspect</p></div>`;
    }
}

function appendHeatmap(selector, matrix) {
    const el = $(selector);
    if (el && matrix) el.appendChild(createMatrixHeatmap(matrix, 280));
}

function createDetailedAttentionHeatmap(head, tokens) {
    const seq = head.length;
    if (seq === 0) return null;

    const labelW = 30;
    const cellSize = Math.min(28, Math.floor(280 / seq));
    const w = labelW + cellSize * seq;
    const h = labelW + cellSize * seq;

    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    canvas.className = 'heatmap';
    canvas.style.cursor = 'crosshair';

    const ctx = canvas.getContext('2d');

    function draw(highlightRow, highlightCol) {
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, w, h);

        ctx.fillStyle = '#3d4654';
        ctx.font = `${Math.min(11, cellSize - 2)}px Inter, sans-serif`;
        ctx.textAlign = 'center';

        for (let i = 0; i < seq; i++) {
            const display = tokens[i] === ' ' ? '_' : tokens[i];
            const isHighlightCol = highlightRow !== undefined && i === highlightCol;
            const isHighlightRow = highlightRow !== undefined && i === highlightRow;
            ctx.fillStyle = isHighlightCol || isHighlightRow ? '#2b6cb0' : '#3d4654';
            ctx.font = isHighlightCol || isHighlightRow ? `bold ${Math.min(11, cellSize - 2)}px Inter, sans-serif` : `${Math.min(11, cellSize - 2)}px Inter, sans-serif`;
            ctx.fillText(display, labelW + i * cellSize + cellSize / 2, labelW - 5);
            ctx.save();
            ctx.translate(labelW - 5, labelW + i * cellSize + cellSize / 2 + 4);
            ctx.fillText(display, 0, 0);
            ctx.restore();
        }

        for (let i = 0; i < seq; i++) {
            for (let j = 0; j < seq; j++) {
                const val = head[i][j];
                const isHighlight = (highlightRow === i);
                ctx.fillStyle = heatColor(val);
                if (highlightRow !== undefined && !isHighlight) {
                    ctx.globalAlpha = 0.25;
                }
                ctx.fillRect(labelW + j * cellSize, labelW + i * cellSize, cellSize - 1, cellSize - 1);
                ctx.globalAlpha = 1;

                if (cellSize >= 20) {
                    ctx.fillStyle = val > 0.5 ? '#fff' : '#333';
                    ctx.font = `bold ${Math.min(9, cellSize - 8)}px monospace`;
                    ctx.textAlign = 'center';
                    ctx.fillText(val.toFixed(1), labelW + j * cellSize + cellSize / 2, labelW + i * cellSize + cellSize / 2 + 3);
                }
            }
        }
    }

    draw();

    canvas.addEventListener('mousemove', e => {
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        const col = Math.floor((mx - labelW) / cellSize);
        const row = Math.floor((my - labelW) / cellSize);
        if (row >= 0 && row < seq && col >= 0 && col < seq) {
            draw(row, col);
            let tip = canvas.parentElement.querySelector('.attn-tooltip');
            if (!tip) { tip = document.createElement('div'); tip.className = 'attn-tooltip'; canvas.parentElement.appendChild(tip); }
            const fromTok = tokens[row] === ' ' ? '(space)' : tokens[row];
            const toTok = tokens[col] === ' ' ? '(space)' : tokens[col];
            tip.textContent = `'${fromTok}' attends to '${toTok}': ${(head[row][col] * 100).toFixed(1)}%`;
            tip.style.left = (mx + 12) + 'px';
            tip.style.top = (my - 20) + 'px';
            tip.style.display = 'block';
        } else {
            draw();
            const tip = canvas.parentElement.querySelector('.attn-tooltip');
            if (tip) tip.style.display = 'none';
        }
    });
    canvas.addEventListener('mouseleave', () => {
        draw();
        const tip = canvas.parentElement.querySelector('.attn-tooltip');
        if (tip) tip.style.display = 'none';
    });

    return canvas;
}

function createMatrixHeatmap(matrix, maxW) {
    const rows = matrix.length;
    const cols = matrix[0].length;
    let min = Infinity, max = -Infinity;
    for (const row of matrix) for (const v of row) { if (v < min) min = v; if (v > max) max = v; }

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

    $('#btn-generate').textContent = 'Generating...';
    try {
        const res = await fetch('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, max_tokens: 30 }),
        });
        const data = await res.json();

        const banner = $('#generation-banner');
        banner.classList.remove('hidden');
        $('#gen-text').innerHTML = `<span class="gen-original">${escHtml(data.input)}</span><span class="gen-new">${escHtml(data.output.slice(data.input.length))}</span>`;

        const stepsCard = $('#gen-steps-card');
        stepsCard.classList.remove('hidden');
        const stepsEl = $('#gen-steps');
        stepsEl.innerHTML = '';
        data.steps.forEach((step, i) => {
            const topPreds = step.top_predictions.map(p => `'${p.char === ' ' ? '(space)' : p.char}' ${(p.prob * 100).toFixed(0)}%`).join(', ');
            stepsEl.innerHTML += `<div class="gen-step-item">Step ${i + 1}: predicted <b>${step.predicted_char === ' ' ? '(space)' : step.predicted_char}</b> <span style="color:var(--text-dim);font-size:0.72rem">(${topPreds})</span></div>`;
        });
    } catch(e) {
        console.error(e);
    }
    $('#btn-generate').textContent = 'Generate 30 chars';
}

function escHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ─── Boot ────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', init);
