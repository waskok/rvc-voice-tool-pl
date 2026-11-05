() => {
  // Ikonki „?” przy etykietach pól + jeden wspólny dymek (position: fixed).
  if (window.__rvcHelpInit) return;
  window.__rvcHelpInit = true;

  const FIELD_HELP = __RVC_FIELD_HELP__;
  const BUTTON_HELP = __RVC_BUTTON_HELP__;

  // Pierwszy pasujący element w kolejności dokumentu: tytuł sekcji rozwijanej,
  // tytuł bloku HTML (log, wykres strat), tytuł pola (slider/lista/tekst/
  // liczba/radio), etykieta pliku/audio albo tekst checkboxa.
  const LABEL_SELECTOR = [
    ".rvc-block-title",
    ".label-wrap > span:first-child",
    '[data-testid="block-info"]',
    '[data-testid="block-label"]',
    'label > input[type="checkbox"] + span',
  ].join(", ");

  const tip = document.createElement("div");
  tip.className = "rvc-tip";
  tip.setAttribute("role", "tooltip");
  tip.id = "rvc-tip";
  document.body.appendChild(tip);

  let current = null;

  const place = (icon) => {
    const r = icon.getBoundingClientRect();
    const margin = 8;
    tip.style.left = "0px";
    tip.style.top = "0px";
    const w = tip.offsetWidth;
    const h = tip.offsetHeight;
    let left = r.left + r.width / 2 - w / 2;
    left = Math.max(margin, Math.min(left, window.innerWidth - w - margin));
    let top = r.bottom + 8;
    if (top + h > window.innerHeight - margin) top = Math.max(margin, r.top - h - 8);
    tip.style.left = `${Math.round(left)}px`;
    tip.style.top = `${Math.round(top)}px`;
  };

  const show = (icon) => {
    current = icon;
    tip.textContent = icon.dataset.tip || "";
    tip.classList.add("is-on");
    icon.setAttribute("aria-describedby", "rvc-tip");
    place(icon);
  };

  const hide = () => {
    if (current) current.removeAttribute("aria-describedby");
    current = null;
    tip.classList.remove("is-on");
  };

  const iconFrom = (ev) =>
    ev.target instanceof Element ? ev.target.closest(".rvc-help") : null;

  document.addEventListener("mouseover", (ev) => {
    const icon = iconFrom(ev);
    if (icon && icon !== current) show(icon);
    else if (!icon && current && !tip.contains(ev.target)) hide();
  });
  document.addEventListener("focusin", (ev) => {
    const icon = iconFrom(ev);
    if (icon) show(icon);
  });
  document.addEventListener("focusout", (ev) => {
    if (iconFrom(ev)) hide();
  });
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") hide();
  });
  window.addEventListener("scroll", hide, true);

  // Faza capture: ikonka siedzi w <label> / przycisku akordeonu, więc klik
  // nie może przełączyć checkboxa ani zwinąć sekcji. Na dotyku klik pokazuje dymek.
  document.addEventListener(
    "click",
    (ev) => {
      const icon = iconFrom(ev);
      if (!icon) return;
      ev.preventDefault();
      ev.stopPropagation();
      if (current === icon) hide();
      else show(icon);
    },
    true
  );

  const ensureIcon = (id, text) => {
    const root = document.getElementById(id);
    if (!root) return;
    const label = root.querySelector(LABEL_SELECTOR);
    if (!label) return;
    const existing = label.querySelector(":scope > .rvc-help");
    if (existing) {
      if (existing.dataset.tip !== text) existing.dataset.tip = text;
      return;
    }
    const icon = document.createElement("span");
    icon.className = "rvc-help";
    icon.tabIndex = 0;
    icon.setAttribute("role", "button");
    icon.setAttribute("aria-label", "Pomoc: co robi ta opcja");
    icon.dataset.tip = text;
    icon.textContent = "?";
    label.appendChild(icon);
  };

  const ensureTitle = (id, text) => {
    const root = document.getElementById(id);
    if (!root) return;
    const btn = root.tagName === "BUTTON" ? root : root.querySelector("button");
    if (btn && btn.title !== text) btn.title = text;
  };

  // Polling zamiast MutationObserver na body (patrz _AUDIO_LOADING_JS):
  // zapis do DOM tylko, gdy ikonki brakuje (np. po przerysowaniu pola).
  const tick = () => {
    for (const [id, text] of Object.entries(FIELD_HELP)) ensureIcon(id, text);
    for (const [id, text] of Object.entries(BUTTON_HELP)) ensureTitle(id, text);
    if (current && !current.isConnected) hide();
  };

  tick();
  setInterval(tick, 1000);
}
