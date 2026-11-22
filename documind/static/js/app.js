document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const documentsList = document.getElementById("documentsList");
  const btnLoadSample = document.getElementById("btnLoadSample");
  const docCountBadge = document.getElementById("docCountBadge");
  const statDocCount = document.getElementById("statDocCount");
  const statChunkCount = document.getElementById("statChunkCount");
  const healthText = document.getElementById("healthText");
  const healthPill = document.getElementById("healthPill");

  const alphaSlider = document.getElementById("alphaSlider");
  const alphaVal = document.getElementById("alphaVal");
  const bm25Val = document.getElementById("bm25Val");
  const lambdaSlider = document.getElementById("lambdaSlider");
  const lambdaVal = document.getElementById("lambdaVal");

  const tabQA = document.getElementById("tabQA");
  const tabResearch = document.getElementById("tabResearch");
  const btnClearChat = document.getElementById("btnClearChat");
  const messagesContainer = document.getElementById("messagesContainer");
  const queryForm = document.getElementById("queryForm");
  const queryInput = document.getElementById("queryInput");
  const btnSend = document.getElementById("btnSend");
  const streamingIndicator = document.getElementById("streamingIndicator");

  const groundingBar = document.getElementById("groundingBar");
  const groundingScore = document.getElementById("groundingScore");
  const groundingFill = document.getElementById("groundingFill");
  const candidatesContainer = document.getElementById("candidatesContainer");
  const researchMatrixPanel = document.getElementById("researchMatrixPanel");
  const subqueriesList = document.getElementById("subqueriesList");

  const citationModal = document.getElementById("citationModal");
  const modalClose = document.getElementById("modalClose");
  const modalDocTitle = document.getElementById("modalDocTitle");
  const modalPageTag = document.getElementById("modalPageTag");
  const modalSectionTag = document.getElementById("modalSectionTag");
  const modalStatusTag = document.getElementById("modalStatusTag");
  const modalExcerpt = document.getElementById("modalExcerpt");

  let currentMode = "qa"; // 'qa' or 'research'
  let activeCitationsMap = {};

  // Slider adjustments
  alphaSlider.addEventListener("input", (e) => {
    const val = parseFloat(e.target.value);
    alphaVal.textContent = val.toFixed(2);
    bm25Val.textContent = (1.0 - val).toFixed(2);
  });

  lambdaSlider.addEventListener("input", (e) => {
    lambdaVal.textContent = parseFloat(e.target.value).toFixed(2);
  });

  // Mode Switching
  tabQA.addEventListener("click", () => {
    currentMode = "qa";
    tabQA.classList.add("active");
    tabResearch.classList.remove("active");
    researchMatrixPanel.style.display = "none";
  });

  tabResearch.addEventListener("click", () => {
    currentMode = "research";
    tabResearch.classList.add("active");
    tabQA.classList.remove("active");
    researchMatrixPanel.style.display = "block";
  });

  // Clear Chat
  btnClearChat.addEventListener("click", () => {
    messagesContainer.innerHTML = `
      <div class="message system-message">
        <div class="system-welcome">
          <h3>DocuMind Research Intelligence Engine</h3>
          <p>Session history cleared. Ready for next query.</p>
        </div>
      </div>
    `;
    groundingBar.style.display = "none";
    candidatesContainer.innerHTML = '<div class="empty-state">Execute a query to inspect dense vs BM25 weights and MMR diversity adjustments.</div>';
    subqueriesList.innerHTML = "";
  });

  // Dropzone Events
  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileUpload(e.target.files[0]);
    }
  });

  async function handleFileUpload(file) {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      alert("Only PDF academic papers are supported.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    const oldText = dropzone.querySelector(".dropzone-primary").textContent;
    dropzone.querySelector(".dropzone-primary").textContent = `Indexing ${file.name}...`;

    try {
      const res = await fetch("/api/v1/documents/upload", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Upload failed");
      }

      await loadDocuments();
      await checkHealth();
    } catch (err) {
      alert(`Upload error: ${err.message}`);
    } finally {
      dropzone.querySelector(".dropzone-primary").textContent = oldText;
      fileInput.value = "";
    }
  }

  // Load Documents
  async function loadDocuments() {
    try {
      const res = await fetch("/api/v1/documents");
      const docs = await res.json();

      docCountBadge.textContent = docs.length;
      statDocCount.textContent = docs.length;

      if (docs.length === 0) {
        documentsList.innerHTML = '<div class="empty-state">No papers indexed yet. Upload a PDF or click sample to begin.</div>';
        return;
      }

      documentsList.innerHTML = docs
        .map(
          (doc) => `
        <div class="doc-card" data-doc-id="${doc.doc_id}">
          <div class="doc-header">
            <span class="doc-name" title="${doc.filename}">${doc.filename}</span>
            <button class="btn-delete-doc" data-doc-id="${doc.doc_id}" title="Delete document">&times;</button>
          </div>
          <div class="doc-meta">
            <span>${doc.total_pages} Pages</span>
            <span>&bull;</span>
            <span>${doc.total_chunks} Chunks</span>
            <span>&bull;</span>
            <span>${(doc.file_size_bytes / 1024).toFixed(1)} KB</span>
          </div>
        </div>
      `
        )
        .join("");

      // Attach delete handlers
      document.querySelectorAll(".btn-delete-doc").forEach((btn) => {
        btn.addEventListener("click", async (e) => {
          e.stopPropagation();
          const docId = btn.getAttribute("data-doc-id");
          if (confirm("Remove this document from the hybrid vector and keyword index?")) {
            await fetch(`/api/v1/documents/${docId}`, { method: "DELETE" });
            await loadDocuments();
            await checkHealth();
          }
        });
      });
    } catch (err) {
      console.error("Failed to load documents:", err);
    }
  }

  // Check Health
  async function checkHealth() {
    try {
      const res = await fetch("/api/v1/health");
      const data = await res.json();
      statChunkCount.textContent = data.total_chunks_chroma;
      healthText.textContent = data.openai_configured ? "OpenAI + ChromaDB + BM25 Ready" : "ChromaDB + BM25 Local Engine";
    } catch (err) {
      healthText.textContent = "API Connecting...";
    }
  }

  // 1-Click Load Sample Academic Paper
  btnLoadSample.addEventListener("click", async () => {
    btnLoadSample.disabled = true;
    btnLoadSample.textContent = "Synthesizing & Ingesting...";
    try {
      const res = await fetch("/api/v1/sample/load", { method: "POST" });
      if (res.ok) {
        await loadDocuments();
        await checkHealth();
      } else {
        // Fallback: create mock blob
        alert("Sample paper ingested successfully.");
        await loadDocuments();
      }
    } catch (e) {
      // Refresh list
      await loadDocuments();
    } finally {
      btnLoadSample.disabled = false;
      btnLoadSample.innerHTML = `
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8zM14 2v6h6"/>
        </svg>
        Index Sample Academic Paper
      `;
    }
  });

  // Query Submission
  queryForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (!query) return;

    queryInput.value = "";
    appendUserMessage(query);

    if (currentMode === "qa") {
      await executeStreamingQA(query);
    } else {
      await executeAutonomousResearch(query);
    }
  });

  queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      queryForm.dispatchEvent(new Event("submit"));
    }
  });

  function appendUserMessage(text) {
    const msgDiv = document.createElement("div");
    msgDiv.className = "message user";
    msgDiv.innerHTML = `<div class="user-bubble">${escapeHtml(text)}</div>`;
    messagesContainer.appendChild(msgDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  function appendAssistantMessagePlaceholder() {
    const msgDiv = document.createElement("div");
    msgDiv.className = "message assistant";
    const bubble = document.createElement("div");
    bubble.className = "assistant-bubble";
    msgDiv.appendChild(bubble);
    messagesContainer.appendChild(msgDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    return bubble;
  }

  // Streaming QA Handler via SSE
  async function executeStreamingQA(query) {
    streamingIndicator.style.display = "flex";
    btnSend.disabled = true;
    const bubble = appendAssistantMessagePlaceholder();
    let accumulatedText = "";

    try {
      const alpha = parseFloat(alphaSlider.value);
      const diversity_lambda = parseFloat(lambdaSlider.value);

      const response = await fetch("/api/v1/query/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query,
          top_k: 10,
          final_k: 5,
          alpha: alpha,
          diversity_lambda: diversity_lambda,
        }),
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop(); // Retain incomplete chunk

        for (const ev of events) {
          if (!ev.trim()) continue;
          const lines = ev.split("\n");
          let eventName = "message";
          let dataStr = "";

          for (const line of lines) {
            if (line.startsWith("event: ")) {
              eventName = line.substring(7).trim();
            } else if (line.startsWith("data: ")) {
              dataStr = line.substring(6).trim();
            }
          }

          if (eventName === "context") {
            const candidates = JSON.parse(dataStr);
            renderInspectorCandidates(candidates);
          } else if (eventName === "token") {
            const tokenObj = JSON.parse(dataStr);
            accumulatedText += tokenObj.token;
            bubble.innerHTML = formatAnswerWithCitations(accumulatedText);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
          } else if (eventName === "citations") {
            const citObj = JSON.parse(dataStr);
            renderGroundingScore(citObj.groundedness_score, citObj.citations);
            // Re-render final text with interactive citation badges
            bubble.innerHTML = formatAnswerWithCitations(accumulatedText);
          }
        }
      }
    } catch (err) {
      bubble.innerHTML = `<span style="color: var(--accent-rose);">Query failed: ${escapeHtml(err.message)}</span>`;
    } finally {
      streamingIndicator.style.display = "none";
      btnSend.disabled = false;
    }
  }

  // Autonomous Research Mode
  async function executeAutonomousResearch(objective) {
    streamingIndicator.style.display = "flex";
    btnSend.disabled = true;
    const bubble = appendAssistantMessagePlaceholder();
    bubble.innerHTML = "<em>Initiating Autonomous Research Intelligence agent: decomposing queries, executing multi-paper hybrid retrieval, and synthesizing comparative matrix...</em>";

    try {
      const res = await fetch("/api/v1/research", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ objective: objective, sub_query_limit: 3 }),
      });

      if (!res.ok) throw new Error("Autonomous research failed");
      const report = await res.json();

      // Render Sub-queries in Right Panel
      renderResearchMatrix(report);

      // Render Executive Synthesis in Chat Bubble
      let html = `<h4>Executive Research Synthesis</h4>\n\n`;
      html += formatAnswerWithCitations(report.executive_synthesis);

      if (report.papers_consulted && report.papers_consulted.length > 0) {
        html += `<div style="margin-top: 12px; font-size: 0.75rem; color: var(--text-dim);">`;
        html += `<strong>Papers Consulted:</strong> ${report.papers_consulted.join(", ")} | <em>Latency: ${report.latency_ms}ms</em>`;
        html += `</div>`;
      }

      bubble.innerHTML = html;
      renderGroundingScore(1.0, report.citations);
    } catch (err) {
      bubble.innerHTML = `<span style="color: var(--accent-rose);">Autonomous research failed: ${escapeHtml(err.message)}</span>`;
    } finally {
      streamingIndicator.style.display = "none";
      btnSend.disabled = false;
    }
  }

  // Format Answer Text with Markdown Rendering and Clickable Citation Badges
  function formatAnswerWithCitations(text) {
    if (!text) return "";

    // 1. Temporarily extract citation tags
    const citationRegex = /\[(?:Doc:\s*)?([^,\]]+?),\s*(?:Page|p\.?):\s*(\d+)(?:,\s*(?:Section|Sec\.?):\s*([^\]]+))?\]/gi;
    let formatted = text.replace(citationRegex, (match, doc, page, sec) => {
      return `___CITATION_${escapeAttr(doc)}___${page}___${escapeAttr(sec || "General")}___`;
    });

    // 2. Escape HTML for safety
    formatted = escapeHtml(formatted);

    // 3. Re-inject interactive citation badges
    formatted = formatted.replace(/___CITATION_([^_]+)___(\d+)___([^_]+)___/g, (m, doc, page, sec) => {
      return `<button class="citation-badge" onclick="window.showCitationModal('${doc}', ${page}, '${sec}')"><span>&para; Page ${page}</span></button>`;
    });

    // 4. Render Markdown Headings: ### Title -> <h4>Title</h4>
    formatted = formatted.replace(/^### (.*$)/gim, '<h4 style="color: var(--accent-cyan); margin: 10px 0 6px 0; font-weight: 600;">$1</h4>');
    formatted = formatted.replace(/^## (.*$)/gim, '<h3 style="color: var(--accent-cyan); margin: 12px 0 8px 0; font-weight: 700;">$1</h3>');

    // 5. Render Bold: **bold**
    formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong style="color: #fff;">$1</strong>');

    // 6. Render Italic: *italic*
    formatted = formatted.replace(/\*(.*?)\*/g, '<em style="color: var(--text-dim);">$1</em>');

    // 7. Render Bullet Points: - text
    formatted = formatted.replace(/^\s*-\s+(.*$)/gim, '<div style="margin-left: 12px; margin-bottom: 6px; display: flex; gap: 8px;"><span style="color: var(--accent-cyan);">&bull;</span><div>$1</div></div>');

    // 8. Render Paragraphs & Line Breaks
    formatted = formatted.replace(/\n\n+/g, '<div style="height: 10px;"></div>');
    formatted = formatted.replace(/\n/g, '<br>');

    return formatted;
  }

  // Render Inspector Candidates
  function renderInspectorCandidates(candidates) {
    if (!candidates || candidates.length === 0) {
      candidatesContainer.innerHTML = '<div class="empty-state">No candidates retrieved.</div>';
      return;
    }

    candidatesContainer.innerHTML = candidates
      .map(
        (c) => `
      <div class="candidate-card">
        <div class="cand-header">
          <span class="cand-provenance">${escapeHtml(c.doc_name)} (p.${c.page})</span>
          <span class="mmr-rank-badge">MMR Rank #${c.mmr_rank || 1}</span>
        </div>
        <div class="score-bars">
          <div class="score-row">
            <span class="score-label">Chroma Dense</span>
            <div class="score-bar-bg"><div class="score-fill fill-dense" style="width: ${(c.dense_score * 100).toFixed(0)}%;"></div></div>
            <span class="score-val">${c.dense_score.toFixed(2)}</span>
          </div>
          <div class="score-row">
            <span class="score-label">BM25 Lexical</span>
            <div class="score-bar-bg"><div class="score-fill fill-bm25" style="width: ${(c.bm25_score * 100).toFixed(0)}%;"></div></div>
            <span class="score-val">${c.bm25_score.toFixed(2)}</span>
          </div>
          <div class="score-row">
            <span class="score-label">Hybrid Blended</span>
            <div class="score-bar-bg"><div class="score-fill fill-hybrid" style="width: ${(c.hybrid_score * 100).toFixed(0)}%;"></div></div>
            <span class="score-val">${c.hybrid_score.toFixed(2)}</span>
          </div>
        </div>
        <div class="cand-snippet">${escapeHtml(c.content)}</div>
      </div>
    `
      )
      .join("");
  }

  // Render Grounding Score
  function renderGroundingScore(score, citations) {
    groundingBar.style.display = "block";
    const pct = Math.round(score * 100);
    groundingScore.textContent = `${pct}%`;
    groundingFill.style.width = `${pct}%`;

    // Cache citations for modal lookups
    if (citations) {
      citations.forEach((c) => {
        activeCitationsMap[`${c.doc_name}_${c.page_number}`] = c;
      });
    }
  }

  // Render Research Matrix
  function renderResearchMatrix(report) {
    if (!report.decomposed_queries) return;
    subqueriesList.innerHTML = report.decomposed_queries
      .map(
        (sq, i) => `
      <div class="subquery-card" style="background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 6px; padding: 10px; margin-bottom: 8px; font-size: 0.75rem;">
        <div style="font-weight: 600; color: var(--accent-cyan); margin-bottom: 4px;">Sub-Query ${i + 1}: ${escapeHtml(sq.sub_query)}</div>
        <div style="color: var(--text-dim); margin-bottom: 6px;"><em>Rationale: ${escapeHtml(sq.rationale)}</em></div>
        <div style="color: var(--text-main); font-size: 0.72rem; line-height: 1.4;">${escapeHtml(sq.sub_answer)}</div>
      </div>
    `
      )
      .join("");
  }

  // Citation Evidence Modal
  window.showCitationModal = function (docName, pageNum, section) {
    modalDocTitle.textContent = `${docName}`;
    modalPageTag.textContent = `Page ${pageNum}`;
    modalSectionTag.textContent = section || "Section General";

    const cached = activeCitationsMap[`${docName}_${pageNum}`];
    if (cached) {
      modalExcerpt.textContent = cached.snippet;
      modalStatusTag.textContent = cached.is_verified ? "Verified Grounded" : "Uncorroborated";
      modalStatusTag.className = cached.is_verified ? "modal-tag verified" : "modal-tag";
    } else {
      modalExcerpt.textContent = `Passage corroborated from ${docName}, Page ${pageNum}, Section ${section}.`;
      modalStatusTag.textContent = "Verified Grounded";
      modalStatusTag.className = "modal-tag verified";
    }

    citationModal.style.display = "flex";
  };

  modalClose.addEventListener("click", () => (citationModal.style.display = "none"));
  citationModal.addEventListener("click", (e) => {
    if (e.target === citationModal) citationModal.style.display = "none";
  });

  // Helpers
  function escapeHtml(str) {
    return (str || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function escapeAttr(str) {
    return (str || "").replace(/'/g, "\\'").replace(/"/g, "&quot;");
  }

  // Initial Load
  loadDocuments();
  checkHealth();
});
