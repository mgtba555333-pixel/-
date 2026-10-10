(() => {
  "use strict";

  let MAX_MB = 12;
  const HISTORY_KEY = "pulmosight.sessionHistory.v1";
  const $ = (id) => document.getElementById(id);
  const fileInput = $("fileInput");
  const uploadZone = $("uploadZone");
  const browseBtn = $("browseBtn");
  const previewCard = $("previewCard");
  const previewImage = $("previewImage");
  const previewName = $("previewName");
  const previewInfo = $("previewInfo");
  const removeFileBtn = $("removeFileBtn");
  const analyzeBtn = $("analyzeBtn");
  const actionHint = $("actionHint");
  const uploadWarning = $("uploadWarning");
  const resultEmpty = $("resultEmpty");
  const resultContent = $("resultContent");
  const toastRegion = $("toastRegion");

  let selectedFile = null;
  let objectUrl = null;
  let modelReady = false;
  let apiReachable = false;
  let history = readHistory();

  function readHistory() {
    try {
      const value = JSON.parse(sessionStorage.getItem(HISTORY_KEY) || "[]");
      return Array.isArray(value) ? value.slice(0, 25) : [];
    } catch (_) {
      return [];
    }
  }

  function saveHistory() {
    // Store only result summaries, never image bytes, paths, or patient identifiers.
    try { sessionStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, 25))); } catch (_) {}
  }

  function toast(title, message, type = "info") {
    const item = document.createElement("div");
    item.className = "toast" + (type === "error" ? " error" : "");
    const mark = document.createElement("span");
    mark.className = "toast-mark";
    mark.textContent = type === "error" ? "!" : "✓";
    const copy = document.createElement("div");
    const heading = document.createElement("strong");
    heading.textContent = title;
    const body = document.createElement("p");
    body.textContent = message;
    copy.append(heading, body);
    item.append(mark, copy);
    toastRegion.append(item);
    window.setTimeout(() => item.remove(), 4800);
  }

  function showWarning(message) {
    uploadWarning.textContent = message;
    uploadWarning.hidden = false;
  }

  function hideWarning() {
    uploadWarning.textContent = "";
    uploadWarning.hidden = true;
  }

  function formatBytes(bytes) {
    return bytes < 1024 * 1024
      ? Math.max(1, Math.round(bytes / 1024)) + " KB"
      : (bytes / (1024 * 1024)).toFixed(2) + " MB";
  }

  function setSelectedFile(file) {
    hideWarning();
    if (!file) return;
    const validTypes = ["image/jpeg", "image/png"];
    const validExtension = /\.(jpe?g|png)$/i.test(file.name || "");
    if (!validTypes.includes(file.type) && !validExtension) {
      showWarning("صيغة الملف غير مدعومة. اختر صورة PNG أو JPG أو JPEG.");
      toast("صيغة غير مدعومة", "اختر صورة بصيغة PNG أو JPEG.", "error");
      return;
    }
    if (file.size <= 0) {
      showWarning("الملف فارغ؛ اختر صورة أخرى.");
      return;
    }
    if (file.size > MAX_MB * 1024 * 1024) {
      showWarning("حجم الملف يتجاوز " + MAX_MB + " MB. قلّل الحجم واختره مجدداً.");
      toast("حجم الصورة كبير", "الحد الأعلى المسموح هو " + MAX_MB + " MB.", "error");
      return;
    }

    selectedFile = file;
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(file);
    previewImage.src = objectUrl;
    previewName.textContent = file.name || "chest-xray-image";
    previewInfo.textContent = formatBytes(file.size) + " · جاهزة للإرسال";
    previewCard.hidden = false;
    uploadZone.hidden = true;
    analyzeBtn.disabled = false;
    actionHint.textContent = modelReady
      ? "النموذج جاهز. أرسل الصورة لعرض المخرجات."
      : "الصورة جاهزة، لكن يلزم تحميل النموذج قبل تنفيذ التحليل الحقيقي.";
  }

  function clearSelectedFile() {
    selectedFile = null;
    fileInput.value = "";
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = null;
    previewImage.removeAttribute("src");
    previewCard.hidden = true;
    uploadZone.hidden = false;
    analyzeBtn.disabled = true;
    hideWarning();
    actionHint.textContent = "اختر صورة أولاً. يتطلب التحليل ربط ملف النموذج الصحيح.";
  }

  browseBtn.addEventListener("click", (event) => {
    event.stopPropagation();
    fileInput.click();
  });
  uploadZone.addEventListener("click", (event) => {
    if (!event.target.closest("button")) fileInput.click();
  });
  uploadZone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      fileInput.click();
    }
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files[0]) setSelectedFile(fileInput.files[0]);
  });
  ["dragenter", "dragover"].forEach((name) => uploadZone.addEventListener(name, (event) => {
    event.preventDefault();
    uploadZone.classList.add("dragover");
  }));
  ["dragleave", "drop"].forEach((name) => uploadZone.addEventListener(name, (event) => {
    event.preventDefault();
    uploadZone.classList.remove("dragover");
  }));
  uploadZone.addEventListener("drop", (event) => {
    const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
    if (file) setSelectedFile(file);
  });
  removeFileBtn.addEventListener("click", clearSelectedFile);

  function setStatusDot(id, state) {
    const element = $(id);
    if (!element) return;
    element.classList.remove("ready", "error");
    if (state === "ready") element.classList.add("ready");
    if (state === "error") element.classList.add("error");
  }

  function setText(id, value) {
    const element = $(id);
    if (element) element.textContent = value;
  }

  async function refreshHealth() {
    setText("metricApiStatus", "تتحقق...");
    try {
      const response = await fetch("/api/health", { headers: { "Accept": "application/json" }, cache: "no-store" });
      if (!response.ok) throw new Error("تعذر الاتصال بالخدمة.");
      const data = await response.json();
      apiReachable = true;
      modelReady = Boolean(data.model_ready);
      MAX_MB = Number(data.max_upload_mb) || 12;
      setText("metricApiStatus", "متصل");
      setText("metricApiFoot", "خادم التحليل يستجيب");
      setStatusDot("metricApiDot", "ready");

      setText("metricModelStatus", modelReady ? "جاهز" : "غير جاهز");
      setText("metricModelFoot", modelReady ? "تم تحميل الأوزان" : "يلزم تجهيز ملف النموذج");
      setStatusDot("metricModelDot", modelReady ? "ready" : "error");
      setText("metricClassCount", String((data.supported_labels || []).length).padStart(2, "0") + " فئات");
      setText("modelMaxSize", MAX_MB + " MB");
      setText("modelPreprocessing", (data.preprocessing || "raw") + " · " + (data.color_order || "rgb").toUpperCase());
      setText("modelStateTitle", modelReady ? "النموذج جاهز" : "النموذج غير محمّل");
      setText("modelStateSubtitle", modelReady ? "Weights loaded · API ready" : "Model setup required");
      setText("modelStateMessage", data.message || (modelReady
        ? "النموذج محمّل. تحقق من طريقة المعالجة وتوافق الفئات قبل اعتماد المخرجات."
        : "أضف ملف الأوزان الصحيح إلى مجلد model ثم أعد تشغيل الخادم."));
      $("modelStatusCard").classList.toggle("ready", modelReady);
      $("modelRing").classList.toggle("ready", modelReady);
      $("modelRing").querySelector("span").textContent = modelReady ? "✓" : "!";
      setStatusDot("sidebarStatusOrb", modelReady ? "ready" : "error");
      setText("sidebarStatusTitle", modelReady ? "النموذج متاح" : "النموذج يحتاج إعداداً");
      setText("sidebarStatusText", modelReady ? "جاهز للتحليل البحثي" : "الأوزان غير محمّلة");
      if (selectedFile) {
        actionHint.textContent = modelReady
          ? "النموذج جاهز. أرسل الصورة لعرض المخرجات."
          : "الصورة محفوظة في المتصفح فقط؛ التحليل غير متاح حتى إضافة النموذج.";
      }
    } catch (_) {
      apiReachable = false;
      modelReady = false;
      setText("metricApiStatus", "غير متصل");
      setText("metricApiFoot", "تحقق من تشغيل الخادم");
      setStatusDot("metricApiDot", "error");
      setText("metricModelStatus", "غير متاح");
      setText("metricModelFoot", "تعذر الوصول إلى حالة النموذج");
      setStatusDot("metricModelDot", "error");
      setText("modelStateTitle", "الخادم غير متصل");
      setText("modelStateSubtitle", "API connection unavailable");
      setText("modelStateMessage", "تعذر الوصول إلى واجهة API. شغّل خادم FastAPI ثم حدّث الحالة.");
      $("modelStatusCard").classList.remove("ready");
      $("modelRing").classList.remove("ready");
      $("modelRing").querySelector("span").textContent = "!";
      setStatusDot("sidebarStatusOrb", "error");
      setText("sidebarStatusTitle", "الخادم غير متصل");
      setText("sidebarStatusText", "تحقق من تشغيل الموقع");
    }
  }

  function translateLabel(label) {
    const upper = String(label || "").toUpperCase();
    if (upper.includes("PNEUMONIA")) return "Pneumonia · التهاب رئوي";
    if (upper.includes("NORMAL")) return "Normal · طبيعي وفق تصنيف النموذج";
    return label || "غير محدد";
  }

  function renderBreakdown(scores) {
    const target = $("scoreBreakdown");
    target.replaceChildren();
    (Array.isArray(scores) ? scores : []).forEach((entry) => {
      const row = document.createElement("div");
      row.className = "score-row";
      const label = document.createElement("div");
      label.className = "score-row-label";
      label.textContent = translateLabel(entry.label);
      const track = document.createElement("div");
      track.className = "score-track";
      const fill = document.createElement("div");
      fill.className = "score-fill";
      fill.style.width = Math.max(0, Math.min(100, Number(entry.score) * 100)) + "%";
      track.append(fill);
      const pct = document.createElement("div");
      pct.className = "score-percent";
      pct.textContent = (Number(entry.score) * 100).toFixed(1) + "%";
      row.append(label, track, pct);
      target.append(row);
    });
  }

  function formatTime(dateValue) {
    const date = dateValue ? new Date(dateValue) : new Date();
    try {
      return new Intl.DateTimeFormat("ar-IQ", { hour: "2-digit", minute: "2-digit", day: "2-digit", month: "short" }).format(date);
    } catch (_) {
      return date.toLocaleString();
    }
  }

  function renderResult(data) {
    resultEmpty.hidden = true;
    resultContent.hidden = false;
    const label = String(data.label || "غير محدد");
    const isPneumonia = label.toUpperCase().includes("PNEUMONIA");
    setText("resultLabel", translateLabel(label));
    setText("resultScore", (Number(data.score) * 100).toFixed(1) + "%");
    setText("resultTime", "اكتمل التحليل " + formatTime(new Date().toISOString()));
    $("resultSymbol").textContent = "✓";
    $("resultSymbol").style.color = isPneumonia ? "var(--amber)" : "var(--mint)";
    $("resultSymbol").style.borderColor = isPneumonia ? "rgba(240,194,125,.25)" : "rgba(106,224,188,.2)";
    setText("resultNotice", data.notice || "مخرج بحثي آلي وليس تشخيصاً طبياً. يجب أن يفسر اختصاصي الأشعة الصورة ضمن السياق السريري.");
    renderBreakdown(data.scores);
    const record = {
      id: Date.now(),
      label,
      score: Number(data.score),
      createdAt: new Date().toISOString()
    };
    history.unshift(record);
    history = history.slice(0, 25);
    saveHistory();
    renderHistory();
    setText("metricScanCount", String(history.length).padStart(2, "0"));
    toast("اكتمل التحليل", "وصلت نتيجة من النموذج المحمّل. راجع تنبيه الاستخدام البحثي.", "info");
    $("results").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function renderHistory() {
    const body = $("historyBody");
    body.replaceChildren();
    setText("historyCount", history.length + " نتائج");
    if (!history.length) {
      const row = document.createElement("tr");
      row.className = "history-empty-row";
      const cell = document.createElement("td");
      cell.colSpan = 5;
      cell.textContent = "لا توجد نتائج مكتملة في هذه الجلسة بعد.";
      row.append(cell);
      body.append(row);
      setText("metricScanCount", "00");
      return;
    }
    history.forEach((entry, index) => {
      const row = document.createElement("tr");
      const idCell = document.createElement("td");
      const id = document.createElement("span");
      id.className = "case-id";
      id.textContent = "PS-" + String(history.length - index).padStart(4, "0");
      idCell.append(id);
      const labelCell = document.createElement("td");
      const label = document.createElement("span");
      const pneumonia = String(entry.label).toUpperCase().includes("PNEUMONIA");
      label.className = "case-label" + (pneumonia ? " pneumonia" : "");
      label.textContent = translateLabel(entry.label);
      labelCell.append(label);
      const scoreCell = document.createElement("td");
      const score = document.createElement("span");
      score.className = "case-score";
      score.textContent = (Number(entry.score) * 100).toFixed(1) + "%";
      scoreCell.append(score);
      const timeCell = document.createElement("td");
      timeCell.textContent = formatTime(entry.createdAt);
      const stateCell = document.createElement("td");
      const state = document.createElement("span");
      state.className = "case-state";
      state.textContent = "مكتمل";
      stateCell.append(state);
      row.append(idCell, labelCell, scoreCell, timeCell, stateCell);
      body.append(row);
    });
    setText("metricScanCount", String(history.length).padStart(2, "0"));
  }

  analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) {
      toast("لم تختر صورة", "اختر ملف أشعة قبل بدء التحليل.", "error");
      return;
    }
    if (!apiReachable) {
      toast("الخادم غير متصل", "شغّل خادم FastAPI ثم حدّث حالة الاتصال.", "error");
      await refreshHealth();
      return;
    }
    analyzeBtn.disabled = true;
    analyzeBtn.classList.add("is-loading");
    const spinner = document.createElement("span");
    spinner.className = "spinner";
    analyzeBtn.prepend(spinner);
    actionHint.textContent = "يتم إرسال الصورة إلى النموذج...";
    try {
      const form = new FormData();
      form.append("file", selectedFile, selectedFile.name || "xray.png");
      const response = await fetch("/api/predict", { method: "POST", body: form });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = data.detail || {};
        const message = typeof detail === "string" ? detail : (detail.message || "تعذر إكمال التحليل.");
        if (response.status === 503) {
          actionHint.textContent = "النموذج غير جاهز؛ لم تُنشأ أي نتيجة.";
          setText("modelStateMessage", message);
          toast("التحليل غير متاح", message, "error");
        } else {
          toast("تعذر التحليل", message, "error");
        }
        return;
      }
      if (!data.label || !Number.isFinite(Number(data.score)) || !Array.isArray(data.scores)) {
        throw new Error("صيغة استجابة النموذج غير مكتملة؛ لم تُضف نتيجة إلى السجل.");
      }
      renderResult(data);
    } catch (error) {
      toast("تعذر الاتصال", error.message || "حدث خطأ غير متوقع أثناء التحليل.", "error");
    } finally {
      spinner.remove();
      analyzeBtn.classList.remove("is-loading");
      analyzeBtn.disabled = !selectedFile;
      actionHint.textContent = modelReady
        ? "النموذج جاهز. أرسل الصورة لعرض المخرجات."
        : "التحليل يتطلب نموذجاً محمّلاً ومتوافقاً؛ لا يتم إنشاء تنبؤات تجريبية.";
    }
  });

  $("refreshStatusBtn").addEventListener("click", refreshHealth);
  $("refreshModelBtn").addEventListener("click", refreshHealth);
  $("clearHistoryBtn").addEventListener("click", () => {
    history = [];
    saveHistory();
    renderHistory();
    resultContent.hidden = true;
    resultEmpty.hidden = false;
    toast("تم مسح السجل", "حُذفت ملخصات النتائج من جلسة المتصفح.");
  });
  document.querySelectorAll("[data-nav]").forEach((link) => {
    link.addEventListener("click", () => {
      document.querySelectorAll("[data-nav]").forEach((item) => item.classList.remove("active"));
      link.classList.add("active");
    });
  });

  renderHistory();
  refreshHealth();
})();