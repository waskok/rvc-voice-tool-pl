() => {
  // Przycisk w rogu: jasne tło i czarny tekst albo motyw ciemny.
  // Klasę .dark na body stawia Gradio (ustawienie systemu albo ?__theme=).
  // Zmiana klasy przełącza zmienne CSS motywu bez przeładowania strony.
  if (window.__rvcThemeInit) return;
  window.__rvcThemeInit = true;

  const KEY = "rvc-theme";
  const button = document.createElement("button");
  button.id = "rvc-theme-btn";
  button.type = "button";

  const readSaved = () => {
    try {
      const value = localStorage.getItem(KEY);
      return value === "light" || value === "dark" ? value : null;
    } catch (e) {
      return null;
    }
  };

  const writeSaved = (mode) => {
    try {
      localStorage.setItem(KEY, mode);
    } catch (e) {}
  };

  const modeNow = () => (document.body.classList.contains("dark") ? "dark" : "light");

  const paint = (mode) => {
    const dark = mode === "dark";
    document.body.classList.toggle("dark", dark);
    document.body.style.background = "var(--body-background-fill)";
    document.body.style.color = "var(--body-text-color)";
    document.documentElement.style.colorScheme = dark ? "dark" : "light";
    button.textContent = dark ? "Jasny motyw" : "Ciemny motyw";
    button.setAttribute(
      "aria-label",
      dark ? "Włącz jasny motyw, lepszy na rzutnik" : "Włącz ciemny motyw"
    );
  };

  button.addEventListener("click", () => {
    const next = modeNow() === "dark" ? "light" : "dark";
    writeSaved(next);
    paint(next);
  });

  const mount = () => {
    if (!document.body) return;
    if (!button.isConnected) document.body.appendChild(button);
    const saved = readSaved();
    paint(saved || modeNow());
  };

  mount();
  // Gradio po starcie może jeszcze raz nałożyć motyw systemu. Zapisany wybór
  // wraca, dopóki użytkownik go nie zmieni.
  setInterval(mount, 1000);
}
