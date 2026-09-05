// GPT-2 Interactive Studio JavaScript Client

document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initSliders();
    fetchStatus();
});

// Tab navigation
function initTabs() {
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabPanes = document.querySelectorAll(".tab-pane");

    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const target = btn.dataset.tab;
            tabBtns.forEach(b => b.classList.remove("active"));
            tabPanes.forEach(p => p.classList.remove("active"));

            btn.classList.add("active");
            const activePane = document.getElementById(`pane-${target}`);
            if (activePane) activePane.classList.add("active");
        });
    });
}

// Sliders value updates
function initSliders() {
    const bindSlider = (sliderId, labelId) => {
        const slider = document.getElementById(sliderId);
        const label = document.getElementById(labelId);
        if (slider && label) {
            slider.addEventListener("input", () => {
                label.textContent = slider.value;
            });
        }
    };

    bindSlider("instruct-tokens", "val-instruct-tokens");
    bindSlider("gen-tokens", "val-gen-tokens");
    bindSlider("gen-temp", "val-gen-temp");
    bindSlider("gen-topk", "val-gen-topk");
}

// Fetch device status
async function fetchStatus() {
    const dot = document.querySelector(".status-dot");
    const text = document.getElementById("device-text");
    try {
        const res = await fetch("/api/status");
        if (res.ok) {
            const data = await res.json();
            text.textContent = `${data.device.toUpperCase()} (${data.gpu_name || 'Ready'})`;
            if (dot) dot.style.backgroundColor = "#10b981";
        } else {
            text.textContent = "Server Ready";
        }
    } catch (e) {
        text.textContent = "Offline";
        if (dot) dot.style.backgroundColor = "#ef4444";
    }
}

// Preset Helpers
function setInstruct(instruction, context) {
    document.getElementById("instruct-input").value = instruction;
    document.getElementById("instruct-context").value = context;
}

function setClassify(text) {
    document.getElementById("classify-input").value = text;
}

function setPrompt(prompt) {
    document.getElementById("generate-prompt").value = prompt;
}

// 1. Run Instruction Following
async function runInstruction() {
    const instruction = document.getElementById("instruct-input").value.trim();
    const inputContext = document.getElementById("instruct-context").value.trim();
    const maxTokens = parseInt(document.getElementById("instruct-tokens").value, 10);
    const btn = document.getElementById("btn-instruct");
    const card = document.getElementById("card-instruct");
    const responseBox = document.getElementById("instruct-response");
    const metricsBox = document.getElementById("instruct-metrics");

    if (!instruction) {
        alert("Please enter an instruction first.");
        return;
    }

    btn.disabled = true;
    btn.innerHTML = "<span>Generating...</span>";
    card.style.display = "block";
    responseBox.textContent = "Thinking...";
    metricsBox.textContent = "";

    try {
        const res = await fetch("/api/instruct", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                instruction: instruction,
                input: inputContext,
                max_new_tokens: maxTokens,
                temperature: 0.0
            })
        });

        if (!res.ok) throw new Error(`Server returned ${res.status}`);
        const data = await res.json();
        responseBox.textContent = data.response || "(Empty response generated)";
        metricsBox.textContent = `${data.tokens_generated} tokens | ${data.latency_seconds}s (${data.tokens_per_second} tok/s)`;
    } catch (err) {
        responseBox.textContent = `Error: ${err.message}`;
    } finally {
        btn.disabled = false;
        btn.innerHTML = "<span>Send Request</span>";
    }
}

// 2. Run Classification
async function runClassification() {
    const text = document.getElementById("classify-input").value.trim();
    const btn = document.getElementById("btn-classify");
    const card = document.getElementById("card-classify");
    const badge = document.getElementById("verdict-badge");
    const icon = document.getElementById("verdict-icon");
    const title = document.getElementById("verdict-title");
    const confVal = document.getElementById("confidence-value");
    const confFill = document.getElementById("confidence-fill");
    const metrics = document.getElementById("classify-metrics");

    if (!text) {
        alert("Please enter a message to classify.");
        return;
    }

    btn.disabled = true;
    btn.innerHTML = "<span>Analyzing...</span>";
    card.style.display = "block";
    metrics.textContent = "Processing...";

    try {
        const res = await fetch("/api/classify", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: text })
        });

        if (!res.ok) throw new Error(`Server returned ${res.status}`);
        const data = await res.json();

        badge.className = "verdict-badge " + (data.is_spam ? "verdict-spam" : "verdict-ham");
        icon.textContent = data.is_spam ? "🚨" : "✅";
        title.textContent = data.is_spam ? "SPAM DETECTED" : "NOT SPAM (HAM)";
        confVal.textContent = `${data.confidence}%`;
        confFill.style.width = `${data.confidence}%`;
        confFill.style.backgroundColor = data.is_spam ? "#ef4444" : "#10b981";
        metrics.textContent = `Latency: ${data.latency_seconds}s`;
    } catch (err) {
        alert(`Error: ${err.message}`);
    } finally {
        btn.disabled = false;
        btn.innerHTML = "<span>Classify Message</span>";
    }
}

// 3. Run Text Generation
async function runGeneration() {
    const prompt = document.getElementById("generate-prompt").value.trim();
    const maxTokens = parseInt(document.getElementById("gen-tokens").value, 10);
    const temperature = parseFloat(document.getElementById("gen-temp").value);
    const topK = parseInt(document.getElementById("gen-topk").value, 10);
    const btn = document.getElementById("btn-generate");
    const card = document.getElementById("card-generate");
    const outputBox = document.getElementById("generate-output");
    const metricsBox = document.getElementById("gen-metrics");

    if (!prompt) {
        alert("Please enter a prompt first.");
        return;
    }

    btn.disabled = true;
    btn.innerHTML = "<span>Generating...</span>";
    card.style.display = "block";
    outputBox.textContent = "Generating continuation...";
    metricsBox.textContent = "";

    try {
        const res = await fetch("/api/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                prompt: prompt,
                max_new_tokens: maxTokens,
                temperature: temperature,
                top_k: topK
            })
        });

        if (!res.ok) throw new Error(`Server returned ${res.status}`);
        const data = await res.json();
        outputBox.textContent = data.output;
        metricsBox.textContent = `${data.tokens_generated} tokens | ${data.latency_seconds}s (${data.tokens_per_second} tok/s)`;
    } catch (err) {
        outputBox.textContent = `Error: ${err.message}`;
    } finally {
        btn.disabled = false;
        btn.innerHTML = "<span>Generate Text</span>";
    }
}

// Clipboard Helper
function copyOutput(elementId) {
    const text = document.getElementById(elementId).textContent;
    navigator.clipboard.writeText(text).then(() => {
        alert("Copied to clipboard!");
    }).catch(() => {
        alert("Failed to copy text.");
    });
}
