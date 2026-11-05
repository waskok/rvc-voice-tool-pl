# Konfiguracja silnika Applio (Zakładka 2 - trening, Zakładka 3 - konwersja)

Zakładka 2 (`2. Trening modelu`) i Zakładka 3 (`3. Nałożenie modelu na głos`)
wołają silnik **Applio** przez `python core.py …` **w osobnym procesie**, z
**osobnym środowiskiem Pythona (venv)** niż to, w którym działa nasze UI
Gradio. Konfiguracja (`APPLIO_DIR` / `APPLIO_PYTHON`) jest **wspólna** dla
obu zakładek - ustawiasz ją raz.

**Dlaczego osobny venv, a nie jedno `pip install -r requirements.txt`?**
Applio wymaga `gradio==6.20.0`, a nasze UI stoi na `gradio>=5.7,<6`. Instalacja
obu w jednym środowisku psuje jedno albo drugie. Trzymając Applio w osobnym
venv i wołając go jako proces (`subprocess`), oba mogą istnieć obok siebie bez
konfliktu wersji Gradio.

---

## 1. Sklonuj Applio (przypięty tag, nie `main`)

**Windows (PowerShell):**

```powershell
git clone --branch 3.6.5 https://github.com/IAHispano/Applio.git C:\Applio
cd C:\Applio
```

**Linux / macOS / Colab:**

```bash
git clone --branch 3.6.5 https://github.com/IAHispano/Applio.git ~/Applio
cd ~/Applio
```

> Tag `3.6.5` to wersja, z którą działa to narzędzie. Użycie `main` może w każdej chwili
> zmienić zależności i polecenia bez ostrzeżenia.

## 2. Stwórz **osobny** venv i zainstaluj zależności Applio

**Windows (PowerShell), w katalogu `C:\Applio`:**

```powershell
py -m venv C:\applio-venv
C:\applio-venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

**Linux / macOS / Colab, w katalogu `~/Applio`:**

```bash
python3 -m venv ~/applio-venv
source ~/applio-venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

To jest **osobny** venv od tego, w którym uruchamiasz nasze `app/main.py`
(tam zostaje `gradio>=5.7,<6` z naszego `requirements.txt`). Nie mieszaj ich.

## 3. Pobierz modele pomocnicze Applio (RMVPE itd.)

Nadal w aktywnym venv Applio:

```bash
python core.py prerequisites
```

Ten krok pobiera m.in. model RMVPE (estymacja wysokości głosu) i inne pliki
wymagane do inferencji. Zajmuje chwilę przy pierwszym uruchomieniu.

## 4. Ustaw zmienne środowiskowe **przed** uruchomieniem naszego UI

Wróć do katalogu naszego projektu (`rvc-voice-tool-pl`) i **aktywuj nasz** venv
(ten z `gradio>=5.7,<6`), a potem ustaw dwie zmienne wskazujące na Applio:

**Windows (PowerShell):**

```powershell
$env:APPLIO_DIR = "C:\Applio"
$env:APPLIO_PYTHON = "C:\applio-venv\Scripts\python.exe"
python -m app.main
```

**Linux / macOS / Colab:**

```bash
export APPLIO_DIR=~/Applio
export APPLIO_PYTHON=~/applio-venv/bin/python
python -m app.main
```

Jeśli te zmienne nie są ustawione (albo ścieżki są złe), Zakładka 3 pokaże
czytelny komunikat po polsku zamiast się wywalić - sprawdź go, zanim zgłosisz
błąd.

> **Pomoc w UI:** przy każdej opcji jest ikonka **„?”** - najechanie (albo
> kliknięcie / Tab na klawiaturze) pokazuje, co opcja robi i co zwykle
> warto ustawić. Treści są w `app/help_texts.py`; przyciski mają dymki
> po najechaniu. Wbudowane napisy Gradio („Upuść plik tutaj”, „Wyczyść”…)
> są wymuszone po polsku niezależnie od języka przeglądarki.

## 5. Test w UI (Zakładka 3)

1. Otwórz `3. Nałożenie modelu na głos`.
2. **Model** - wgraj:
   - ZIP zawierający `.pth` i (opcjonalnie) `.index`, **albo**
   - sam plik `.pth` (wtedy w polu **„Plik .index”** wgraj osobno swój
     `.index`, jeśli go masz - bez indeksu konwersja i tak zadziała, ale
     jakość / barwa mogą być gorsze).

   **Ważne - który `.pth`:** do konwersji potrzebny jest **eksportowany model
   inferencyjny** (plik **bez** prefiksu `G_`/`D_`, zwykle
   `{nazwa}_15e_225s.pth` - ma w środku klucz `config`).  
   **`G_….pth` / `D_….pth` to checkpointy treningowe** - wrzucenie ich do
   Zakładki 3 kończy się `KeyError: 'config'`. Zakładka 2 po poprawce pakuje
   do ZIP tylko właściwy mały `.pth` (+ `.index`).
3. **Głos / wokal źródłowy** - nagranie, na które ma zostać nałożony model.
4. Ustaw **Zmianę wysokości głosu**, jeśli trzeba (np. `-12`…`+12` półtonów przy różnicy
   głosu męski/żeński).
5. Wybierz **Metodę F0** (domyślnie `rmvpe`; `fcpe` / `crepe-tiny` zwykle
   szybsze; `crepe` jak w klasycznym RVC GUI).
6. Kliknij **Konwertuj**.
7. W polu **Log / status konwersji** zobaczysz m.in. `urządzenie: GPU/CPU`,
   wybraną metodę F0 oraz dokładne polecenie, które poszło do Applio
   (`core.py infer …`) - przydatne przy zgłaszaniu błędów.
8. Wynik pojawi się w komponencie **„Wynik (odsłuch + pobieranie)”** - można
   odsłuchać w przeglądarce i pobrać jako WAV (ikona pobierania na
   komponencie audio).

> GPU vs CPU: Applio używa CUDA automatycznie, jeśli `torch` w venv Applio
> widzi kartę NVIDIA. Nie ma przełącznika w UI (w Colabie i tak będzie GPU).
> Sprawdzenie: w logu po konwersji linia `urządzenie: …`.

## 6. Test treningu (Zakładka 2)

1. Otwórz `2. Trening modelu`.
2. **Nazwa modelu** - np. `moj_glos` (polskie znaki są automatycznie
   zamieniane na ASCII: „Mój Głos” → `Moj_Glos`).
3. **Próbki treningowe** - można wgrać kilka plików naraz, np. 2 ZIP-y
   z Zakładki 1 + pojedynczy WAV (powtarzające się nazwy próbek dostają
   prefiks z nazwy archiwum, nic się nie nadpisuje). Pod polem pojawia się
   **podsumowanie**: liczba plików, łączna długość, ostrzeżenia (za mało /
   za dużo materiału) i szacowany czas treningu na T4 (z własnego pomiaru:
   13 min audio → ~1:11/epokę, 100 epok ≈ 2 h 19 min).
4. **Tryb Prosty** (rekomendowane na start) - wybierz profil ustawień
   „Mowa” / „Śpiew” (parametry z `configs/profiles.json`). **Zaawansowany** -
   ręcznie epoki, rozmiar paczki (batch), metoda F0, model cech mowy
   (embedder).
5. **Zapisuj checkpoint co N epok** (suwak, domyślnie 25) - co tyle epok
   Applio nadpisuje parę `G_2333333.pth` / `D_2333333.pth` w `logs/<model>/`
   (`--save-every-epoch N --save-only-latest`). Mały model do konwersji
   powstaje na końcu treningu (albo po „Zatrzymaj trening” - z ostatniego G).
6. **Automatyczna kopia G/D na Drive** (domyślnie włączona, tylko Colab po
   komórce „Podłącz Google Drive”) - po każdym zapisie para trafia do
   `MyDrive/RVC_Colab_Models/<nazwa>/checkpoints/G_<nazwa>_e<N>.pth` (+ `D_…`);
   zostają 2 ostatnie pary. Kopia startuje dopiero, gdy Applio skończy
   zapisywać pliki (5 s bez zmian), przez plik tymczasowy `.part`. Każda
   kopia - udana albo nie - trafia do **„Status zapisu”** w kroku 5.
7. **Rozpocznij trening** - po prawej: **pasek postępu** (epoka, krok
   w epoce), **tempo** (mediana 5 ostatnich epok - pierwsze epoki są ~2×
   wolniejsze), **szacowany czas do końca**, ostatni checkpoint i status
   auto-kopii, **wykres strat** (`loss_avg_50/g/total` i `…/g/mel`
   z plików TensorBoard Applio w `logs/<model>/eval/`, rysowany jako
   statyczny SVG - `gr.LinePlot` zawieszał kartę przeglądarki przy
   pierwszych punktach) oraz przewijany log (stała wysokość; przewinięty
   w górę zostaje w miejscu, na dole podąża za nowymi liniami). UI
   odświeża się najwyżej raz na sekundę; stdout Applio czyta osobny wątek
   (`python -u`), więc log nie odstaje od faktycznego treningu.
8. **Zatrzymaj trening** - guzik (`queue=False`) tylko zgłasza zatrzymanie
   i od razu wraca; pętla logu kończy drzewo procesów Applio w ~1 s, a jeśli
   akurat trwa zapis G/D - po jego końcu (pasek następnej epoki = zapis
   gotowy). Potem buduje `.index` i ZIP z ostatniego checkpointu i zapisuje
   go na Dysku tak samo jak po pełnym treningu. Zamknięcie
   karty / rozłączenie też kończy trening (nic nie zostaje „w tle” bez UI).
   Mid-run zapisuje tylko G/D (bez ciężkiego `--save-every-weights` co N
   epok); DataLoader Applio bez zmian (4 workery).
9. Wynik: **ZIP z modelem inferencyjnym** (`{nazwa}_*e_*s.pth` + `.index`) -
   **nie** `G_*.pth`. Ten ZIP wgrywasz w Zakładce 3.
10. **Krok 5 „Zapisz model”** - przy podłączonym Dysku ZIP modelu zapisuje
    się **automatycznie** po treningu (i po „Zatrzymaj trening”) do
    `MyDrive/RVC_Colab_Models/<nazwa>_<data>.zip`, a checkpointy G/D -
    w trakcie treningu do `MyDrive/RVC_Colab_Models/<nazwa>/checkpoints/`.
    Przyciski **„Pobierz plik ZIP”**, **„Pobierz checkpoint G”**,
    **„Pobierz checkpoint D”** pobierają pliki na komputer (G/D pod nazwą
    `G_<nazwa>_e<N>.pth`, twardy link do `logs/<model>/`, bez kopii setek MB).
    **„Zapisz model na Dysku ponownie”** ponawia zapis ZIP-a, np. gdy Dysk
    podłączono dopiero po treningu. **„Status zapisu”** to dziennik z godziną
    (czas polski) wszystkich zapisów w sesji UI - automatycznych i ręcznych,
    udanych i nieudanych; wpisy się nie czyszczą, także po odświeżeniu
    strony. Mount z samego Gradio nie działa (brak jądra komórki Colaba).
11. **Wznowienie** - zaznacz **„Wznów trening z checkpointu”**; **Epoki** to
    łączny cel (checkpoint z epoki 100 + 200 epok → dotrenuje 101-200).
    W tej samej sesji wystarczy ta sama nazwa modelu (preprocess/extract są
    pomijane). W nowej sesji wgraj **te same** próbki i wybierz parę z listy
    **„Para checkpointów G i D z Dysku Google”** (albo wgraj G i D ręcznie); ten sam SR
    i batch size. Applio samo ładuje najnowsze G/D z `logs/<model>/`
    (`rvc/train/train.py`, blok „Load checkpoint if available”).

> Trening na CPU (bez GPU) jest bardzo wolny - praktycznie tylko do krótkich
> testów. Docelowo: GPU w Google Colab.

### Typowe problemy

| Objaw | Prawdopodobna przyczyna |
| --- | --- |
| „Brak zmiennej środowiskowej APPLIO_DIR / APPLIO_PYTHON” | Nie ustawiłeś zmiennych w **tym samym** terminalu, w którym odpalasz `python -m app.main` (zmienne `$env:` / `export` nie przechodzą między osobnymi oknami terminala). |
| „Nie znaleziono … core.py” | Zła ścieżka `APPLIO_DIR` - sprawdź, czy w tym katalogu jest plik `core.py`. |
| „Nie znaleziono interpretera Pythona” | Zła ścieżka `APPLIO_PYTHON` - powinna wskazywać na plik `python.exe` / `python` **wewnątrz** venv Applio, nie na sam folder. |
| Applio zwraca błąd o brakującym module | Prawdopodobnie `python core.py prerequisites` nie zostało uruchomione, albo `pip install -r requirements.txt` nie przeszło do końca w venv Applio. |
| Długi czas / timeout (inferencja) | Bardzo długie nagranie źródłowe albo wolny CPU/GPU - zwiększ `APPLIO_TIMEOUT_S` (domyślnie 1800 s = 30 min) albo skróć nagranie. |
| „Model … ma już checkpoint w tej sesji” | Nowy trening z nazwą, która ma już G/D w `logs/` - zaznacz „Wznów trening” albo zmień nazwę (wcześniej Applio po cichu kontynuowało stary trening). |
| Długi czas / timeout (trening) | Zwiększ `APPLIO_TRAIN_TIMEOUT_S` (domyślnie 43200 s = 12 h) albo zmniejsz liczbę epok. |
| Lista „Para checkpointów G i D z Dysku Google” pusta | Inna nazwa modelu niż w poprzedniej sesji, Drive niepodłączony albo auto-kopia była wyłączona - „Odśwież listę” po wpisaniu nazwy albo wgraj G/D ręcznie. |
| Trening: „nie znaleziono pliku wynikowego .pth” | Preprocessing/ekstrakcja nie wyprodukowały danych (zbyt mało/złe próbki) - sprawdź log kroków 1-2 wyżej. |
| Inferencja: `KeyError: 'config'` / „checkpoint treningowy G_” | Wgrano `G_*.pth` zamiast małego modelu. Użyj `{nazwa}_*e_*s.pth` z `logs/<model>/` albo komórki recovery (ekstrakcja z G_). |
| Trening kończy się, ale w `logs/<model>/` jest tylko `G_*.pth`/`D_*.pth` (brak małego `.pth`) | Applio sam wywołuje ekstrakcję na końcu treningu, ale **cicho** ją przerywa, jeśli w klonie Applio nie ma `assets/config.json` - Zakładka 2 od tego fixu tworzy ten plik automatycznie przed treningiem; jeśli trenowałeś na starszej wersji kodu, użyj komórki recovery (ekstrakcja z `G_*` w Colabie). |
| „Status zapisu”: „zapisany tylko lokalnie” / „dysk nie jest podłączony” / nic w MyDrive | W Colabie trzeba **najpierw** uruchomić komórkę montującą Drive w notatniku (przed startem UI). UI tylko kopiuje - nie otwiera okna autoryzacji. Po podłączeniu Dysku i restarcie UI w tej samej sesji wpisz tę samą nazwę modelu i kliknij „Zapisz model na Dysku ponownie” - ZIP odbuduje się z `logs/<model>/`. Lokalnie zapis na Drive jest niedostępny; ZIP i G/D i tak są do pobrania w kroku 5. |

---

## Źródła

- Applio - repozytorium (MIT): https://github.com/IAHispano/Applio
- Applio - CLI `core.py` (argumenty `infer`/`preprocess`/`extract`/`train`): https://docs.applio.org/reference/cli/
- Applio - Terms of Use: https://github.com/IAHispano/Applio/blob/main/TERMS_OF_USE.md
- Google Colab - mount Google Drive (`google.colab.drive`): https://colab.research.google.com/notebooks/io.ipynb
