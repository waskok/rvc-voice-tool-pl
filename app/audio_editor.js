() => {
  /**
   * Pasek ikon przy nożyczkach Gradio: Przytnij | Wytnij | Cofnij | Zapisz | ×
   * Cut: odczyt regionu z fali → ukryty trigger Gradio (czas przez js→outputs).
   */
  const AUDIO_ID = "rvc_edit_audio";
  const CUT_BTN_ID = "rvc_cut_btn";
  const TRIM_BTN_ID = "rvc_trim_btn";
  const UNDO_BTN_ID = "rvc_undo_btn";
  const SAVE_BTN_ID = "rvc_save_btn";

  const ICON = {
    trim: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M6 3v6"/><path d="M18 3v6"/><path d="M6 9h12"/><path d="M8 9l4 12 4-12"/></svg>`,
    cut: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><circle cx="6" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M20 4L8.12 15.88"/><path d="M14.47 14.48L20 20"/><path d="M8.12 8.12L12 12"/></svg>`,
    undo: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M9 14L4 9l5-5"/><path d="M4 9h10a5 5 0 010 10h-1"/></svg>`,
    save: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M12 3v12"/><path d="M7 10l5 5 5-5"/><path d="M5 21h14"/></svg>`,
    close: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg>`,
  };

  function parseTime(text) {
    if (!text) return 0;
    const parts = String(text)
      .trim()
      .split(":")
      .map((p) => parseFloat(p));
    if (parts.some((n) => Number.isNaN(n))) return 0;
    if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
    if (parts.length === 2) return parts[0] * 60 + parts[1];
    return parts[0] || 0;
  }

  function walkDeep(node, visit) {
    if (!node) return;
    visit(node);
    if (node.shadowRoot) walkDeep(node.shadowRoot, visit);
    const children = node.children || [];
    for (let i = 0; i < children.length; i++) walkDeep(children[i], visit);
  }

  function findRegions(root) {
    const found = [];
    walkDeep(root, (el) => {
      if (!el.getAttribute) return;
      const part = el.getAttribute("part") || "";
      if (
        part.includes("region") &&
        !part.includes("handle") &&
        !part.includes("marker")
      ) {
        found.push(el);
      }
    });
    // Fallback: rodzic uchwytu resize (WaveSurfer)
    if (!found.length) {
      walkDeep(root, (el) => {
        if (el.getAttribute && el.getAttribute("data-resize") === "left") {
          const parent = el.parentElement;
          if (parent && !found.includes(parent)) found.push(parent);
        }
      });
    }
    return found;
  }

  window.rvcGetSelection = function rvcGetSelection(elemId) {
    const root = document.getElementById(elemId || AUDIO_ID);
    if (!root) return null;
    const durationEl = root.querySelector("#duration");
    const duration = parseTime(durationEl && durationEl.textContent);
    if (!(duration > 0)) return null;
    const regions = findRegions(root);
    if (!regions.length) return null;
    const el = regions[0];
    const left = parseFloat(el.style.left) || 0;
    const right = parseFloat(el.style.right) || 0;
    const start = (left / 100) * duration;
    const end = duration - (right / 100) * duration;
    if (!(end > start + 0.01)) return null;
    return { start, end, duration };
  };

  // Element z elem_id bywa samym <button> (Button/DownloadButton) albo
  // divem-wrapperem z <button> w środku (starsze/inne komponenty) -
  // obsłuż obie postacie.
  function findButton(id) {
    const el = document.getElementById(id);
    if (!el) return null;
    if (el.tagName === "BUTTON") return el;
    return el.querySelector("button");
  }

  function clickHidden(id) {
    const btn = findButton(id);
    if (btn) {
      btn.click();
      return true;
    }
    console.error("RVC: brak ukrytego przycisku", id);
    return false;
  }

  // Wpisuje wartość do gr.Number (Svelte bind:value) - ustawia .value na
  // realnym <input> i wysyła 'input' + 'change', żeby Gradio zauważyło zmianę
  // PRZED kliknięciem prawdziwego przycisku (inaczej fn dostanie stare dane).
  function setNumberInput(elemId, value) {
    const wrap = document.getElementById(elemId);
    const input = wrap && (wrap.tagName === "INPUT" ? wrap : wrap.querySelector("input"));
    if (!input) {
      console.error("RVC: brak inputu dla", elemId);
      return false;
    }
    input.focus();
    input.value = String(value);
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
    input.blur();
    return true;
  }

  // Po Trim/Cut Gradio zostaje w trybie "edit" (Trim/Cancel), ale stary
  // region (zaznaczenie) jest usunięty razem ze starą falą - bez reset do
  // idle scissors by się nie pokazały, więc kolejna edycja byłaby niemożliwa.
  function resetToIdleAfterEdit() {
    if (window.__rvcCancelFn) window.__rvcCancelFn();
  }

  // Trim i Cut to własne, prawdziwe funkcje Python (nie wbudowany trim
  // Gradio) - obie idą tą samą ścieżką: wpisz zakres do Number, kliknij
  // odpowiedni ukryty przycisk, poczekaj na zmianę fali, wróć do idle.
  function submitEdit(sel, btnId, label) {
    if (!setNumberInput("rvc_cut_start", sel.start) || !setNumberInput("rvc_cut_end", sel.end)) {
      alert("Nie udało się przygotować zakresu. Odśwież stronę.");
      return;
    }
    const root = document.getElementById(AUDIO_ID);
    const durationEl = root && root.querySelector("#duration");
    const before = durationEl ? durationEl.textContent : null;
    // Gradio potrzebuje chwili, żeby zarejestrować input/change ze Svelte.
    setTimeout(() => {
      if (!clickHidden(btnId)) {
        alert(`Nie znaleziono przycisku ${label} w aplikacji. Odśwież stronę.`);
        return;
      }
      if (!durationEl) return;
      const obs = new MutationObserver(() => {
        if (durationEl.textContent !== before) {
          obs.disconnect();
          setTimeout(resetToIdleAfterEdit, 150);
        }
      });
      obs.observe(durationEl, { characterData: true, childList: true, subtree: true });
      setTimeout(() => obs.disconnect(), 8000);
    }, 50);
  }

  function saveCurrentAudio() {
    if (!clickHidden(SAVE_BTN_ID)) {
      alert("Brak audio do zapisu - najpierw wgraj plik.");
    }
  }

  function styleIconBtn(btn) {
    btn.style.display = "inline-flex";
    btn.style.alignItems = "center";
    btn.style.justifyContent = "center";
    btn.style.minWidth = "28px";
    btn.style.height = "28px";
    btn.style.padding = "2px 6px";
    btn.style.marginLeft = "4px";
    btn.style.lineHeight = "1";
    btn.style.cursor = "pointer";
  }

  function makeIcon(aria, title, svg, onClick) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "action icon";
    btn.setAttribute("aria-label", aria);
    btn.title = title;
    btn.innerHTML = svg;
    styleIconBtn(btn);
    btn.addEventListener("click", (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      onClick();
    });
    return btn;
  }

  function ensureToolbar(root) {
    const settings = root.querySelector(".settings-wrapper");
    if (!settings) return;

    settings.querySelectorAll("[data-rvc-cut-icon]").forEach((el) => el.remove());

    const textButtons = Array.from(
      settings.querySelectorAll("button.text-button")
    );
    const trimConfirm = textButtons.find((b) => {
      const t = b.textContent.trim();
      return t === "Trim" || b.dataset.rvcRole === "gradio-trim";
    });
    const cancelBtn = textButtons.find((b) => {
      const t = b.textContent.trim();
      return t === "Cancel" || b.dataset.rvcRole === "gradio-cancel";
    });

    // Idle: nożyczki + Undo + Save
    if (!trimConfirm) {
      const old = settings.querySelector("[data-rvc-edit-toolbar]");
      if (old) old.remove();
      const existingIdleBar = settings.querySelector("[data-rvc-idle-bar]");
      if (existingIdleBar) {
        // Wracamy z trybu edycji - poprzednio ukryte na czas Trim/Cut.
        existingIdleBar.style.display = "inline-flex";
      }
      const scissors = settings.querySelector(
        'button[aria-label="Trim audio to selection"]'
      );
      if (scissors && !existingIdleBar) {
        const bar = document.createElement("span");
        bar.setAttribute("data-rvc-idle-bar", "1");
        bar.style.display = "inline-flex";
        bar.style.alignItems = "center";
        bar.appendChild(
          makeIcon("Cofnij", "Cofnij ostatnią zmianę (Ctrl+Z)", ICON.undo, () => {
            clickHidden(UNDO_BTN_ID);
          })
        );
        bar.appendChild(
          makeIcon("Zapisz", "Zapisz - pobierz bieżące nagranie jako WAV", ICON.save, () => {
            saveCurrentAudio();
          })
        );
        scissors.insertAdjacentElement("afterend", bar);
      }
      window.__rvcCancelFn = null;
      return;
    }

    // W trybie edycji: zapamiętaj jak wrócić do idle (używane po cutcie).
    window.__rvcCancelFn = cancelBtn ? () => cancelBtn.click() : null;

    settings.querySelectorAll("[data-rvc-idle-bar]").forEach((el) => {
      el.style.display = "none";
    });

    trimConfirm.dataset.rvcRole = "gradio-trim";
    if (cancelBtn) cancelBtn.dataset.rvcRole = "gradio-cancel";
    trimConfirm.style.display = "none";
    if (cancelBtn) cancelBtn.style.display = "none";

    if (settings.querySelector("[data-rvc-edit-toolbar]")) return;

    const bar = document.createElement("div");
    bar.setAttribute("data-rvc-edit-toolbar", "1");
    bar.style.display = "inline-flex";
    bar.style.alignItems = "center";
    bar.style.gap = "2px";
    bar.style.marginLeft = "4px";

    bar.appendChild(
      makeIcon("Przytnij", "Przytnij - zostaw tylko zaznaczony fragment", ICON.trim, () => {
        const sel = window.rvcGetSelection(AUDIO_ID);
        if (!sel) {
          alert("Najpierw przesuń zaznaczenie na fragment, który ma zostać.");
          return;
        }
        submitEdit(sel, TRIM_BTN_ID, "Przytnij");
      })
    );
    bar.appendChild(
      makeIcon("Wytnij", "Wytnij - usuń zaznaczony fragment", ICON.cut, () => {
        const sel = window.rvcGetSelection(AUDIO_ID);
        if (!sel) {
          alert("Najpierw przesuń zaznaczenie na fragment do usunięcia.");
          return;
        }
        submitEdit(sel, CUT_BTN_ID, "Wytnij");
      })
    );
    bar.appendChild(
      makeIcon("Cofnij", "Cofnij ostatnią zmianę (Ctrl+Z)", ICON.undo, () => {
        clickHidden(UNDO_BTN_ID);
      })
    );
    bar.appendChild(
      makeIcon("Zapisz", "Zapisz - pobierz bieżące nagranie jako WAV", ICON.save, () => {
        saveCurrentAudio();
      })
    );
    bar.appendChild(
      makeIcon("Anuluj", "Anuluj zaznaczenie", ICON.close, () => {
        if (cancelBtn) cancelBtn.click();
        bar.remove();
      })
    );

    settings.appendChild(bar);
  }

  // Przyciski odtwarzacza Gradio mają tylko angielskie aria-label (bez i18n);
  // aria-label zostaje (po nim szukamy nożyczek), dokładamy polski dymek.
  const PLAYER_TITLES = [
    ["Trim audio to selection", "Edytuj: zaznacz fragment na fali (przytnij / wytnij)"],
    ["Reset audio", "Przywróć nagranie sprzed edycji"],
    ["Adjust volume", "Głośność odsłuchu"],
    ["Adjust playback speed", "Prędkość odtwarzania"],
    ["Skip backwards", "Przewiń do tyłu"],
    ["Skip forward", "Przewiń do przodu"],
  ];

  function translatePlayerTitles() {
    document
      .querySelectorAll("#" + AUDIO_ID + " button[aria-label], #rvc_out_audio button[aria-label]")
      .forEach((btn) => {
        const aria = btn.getAttribute("aria-label") || "";
        const hit = PLAYER_TITLES.find(([prefix]) => aria.startsWith(prefix));
        if (hit && btn.title !== hit[1]) btn.title = hit[1];
      });
  }

  function tick() {
    const root = document.getElementById(AUDIO_ID);
    if (root) ensureToolbar(root);
    translatePlayerTitles();
  }

  // Nie obserwujemy całego document - Gradio mutuje DOM bardzo często
  // (logi, progress). Observer + DOM write = ryzyko pętli / wysokiego RAM.
  if (!window.__rvcAudioEditorInit) {
    window.__rvcAudioEditorInit = true;
    tick();
    setInterval(tick, 800);
  }
}
