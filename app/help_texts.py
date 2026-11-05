"""Treści dymków „?” przy opcjach interfejsu (klucz = ``elem_id`` komponentu).

``FIELD_HELP`` - ikonka „?” doklejana do etykiety pola (suwaki, listy,
pola wyboru, pliki, sekcje rozwijane). ``BUTTON_HELP`` - natywny dymek
(atrybut ``title``) po najechaniu na przycisk.
"""

from __future__ import annotations

from rvc_pipeline.train_monitor import AUTO_BACKUP_KEEP

FIELD_HELP: dict[str, str] = {
    # --- Zakładka 1: przygotowanie nagrań ---
    "rvc_edit_audio": (
        "Surowe nagranie głosu, z którego powstaną próbki do treningu: "
        "wav, mp3, flac, ogg lub m4a. Możesz też nagrać się mikrofonem.\n\n"
        "Najlepiej: 10-30 minut czystego głosu jednej osoby, bez muzyki, "
        "echa i innych mówców.\n\n"
        "Edycja: ikona nożyczek → przeciągnij zaznaczenie na fali → "
        "„Przytnij” zostawia tylko zaznaczenie, „Wytnij” je usuwa. "
        "„Cofnij” lub Ctrl+Z wraca o krok, „Zapisz” pobiera WAV."
    ),
    "rvc_prep_sr": (
        "Z jaką częstotliwością próbkowania zapisać próbki.\n\n"
        "40000 Hz - zalecane, dobry kompromis jakości i czasu treningu.\n"
        "48000 Hz - odrobinę więcej góry pasma, ale tylko jeśli nagranie "
        "naprawdę jest 48 kHz.\n"
        "32000 Hz - najszybciej, słabsza jakość.\n\n"
        "Ustaw tę samą wartość w zakładce 2."
    ),
    "rvc_do_slice": (
        "Dzieli nagranie na krótsze kawałki w miejscach ciszy między "
        "zdaniami i wyrzuca same pauzy.\n\n"
        "Zalecane: włączone. Wyłącz tylko, gdy nagranie jest już pocięte "
        "albo cisza jest w nim bardzo krótka - wtedy zostanie podzielone "
        "na równe okna o maksymalnej długości."
    ),
    "rvc_do_normalize": (
        "Wyrównuje głośność wszystkich próbek do jednego poziomu "
        "-23 LUFS według EBU R128, żeby model nie uczył się "
        "„raz cicho, raz głośno”.\n\n"
        "Zalecane: włączone."
    ),
    "rvc_do_denoise": (
        "Usuwa stały szum tła: szum mikrofonu, wentylator, buczenie.\n\n"
        "Zalecane: włączone dla nagrań z telefonu lub zwykłego mikrofonu. "
        "Wyłącz przy nagraniach studyjnych - mocne odszumianie może lekko "
        "zmatowić głos."
    ),
    "rvc_len_acc": (
        "Granice długości pojedynczej próbki. Domyślne wartości "
        "1-12 s pasują do większości nagrań - zwykle nie trzeba ich ruszać."
    ),
    "rvc_min_dur": (
        "Fragmenty krótsze niż ta wartość są odrzucane - oddechy, kliknięcia, "
        "pojedyncze sylaby.\n\n"
        "Zalecane: 1 s. Podnieś do 2 s, jeśli w próbkach jest dużo śmieci; "
        "obniż, gdy nagrania są bardzo krótkie i tracisz za dużo materiału."
    ),
    "rvc_max_dur": (
        "Fragmenty dłuższe niż ta wartość są dzielone na okna tej długości "
        "- nic nie jest wyrzucane.\n\n"
        "Zalecane: 10-12 s. Dłuższe próbki zużywają więcej pamięci GPU "
        "w treningu."
    ),
    "rvc_out_zip": (
        "Gotowy zestaw - wszystkie próbki w jednym ZIP-ie. Pobierz go "
        "i wgraj w zakładce 2 jako próbki treningowe."
    ),
    "rvc_out_info": (
        "Podsumowanie przetwarzania: ile próbek powstało, ile odrzucono "
        "i jaka jest łączna długość. Do dobrego modelu warto mieć "
        "co najmniej 10 minut próbek."
    ),
    "rvc_out_files_acc": (
        "Te same próbki co w ZIP-ie, jako osobne pliki WAV - przydatne, "
        "żeby odsłuchać kilka i sprawdzić jakość."
    ),
    # --- Zakładka 2: trening ---
    "rvc_model_name": (
        "Nazwa Twojego modelu - trafi do nazwy pliku .pth/.index i folderu "
        "na Google Drive. Dozwolone: litery, cyfry, _ i -.\n\n"
        "Ta sama nazwa jest potrzebna do wznowienia treningu."
    ),
    "rvc_samples": (
        "Próbki do treningu - można wgrać kilka plików naraz: ZIP-y "
        "z zakładki 1 i/lub pojedyncze nagrania wav, mp3, flac, ogg, m4a.\n\n"
        "Wszystko powinno być głosem tej samej osoby. Zalecane: "
        "10-30 minut łącznie; poniżej 5 minut model zwykle brzmi słabo. "
        "Pod polem pojawi się podsumowanie i szacowany czas treningu."
    ),
    "rvc_train_sr": (
        "Częstotliwość próbkowania modelu - musi być taka sama jak "
        "w zakładce 1.\n\n"
        "Zalecane: 40000 Hz. Przy wznawianiu treningu musi się zgadzać "
        "z checkpointem."
    ),
    "rvc_mode": (
        "Prosty - wybierasz gotowy profil ustawień: mowa albo śpiew, resztę "
        "ustawiamy za Ciebie. Zalecane na start.\n\n"
        "Zaawansowany - sam ustawiasz liczbę epok, rozmiar paczki, "
        "metodę F0 i model embeddingu."
    ),
    "rvc_preset": (
        "Gotowy profil ustawień treningu.\n\n"
        "Mowa - do nagrań mówionych: podcast, lektor, rozmowa.\n"
        "Śpiew - do wokalu; trochę więcej epok, bo śpiew ma większą "
        "rozpiętość wysokości.\n\n"
        "Wybierz ten, który pasuje do tego, co później chcesz zamieniać."
    ),
    "rvc_preset_info": "Podgląd parametrów, które ustawia wybrany profil.",
    "rvc_epochs": (
        "Ile razy model przejdzie przez wszystkie próbki. Więcej epok = "
        "dokładniejsza barwa, ale dłuższy trening i ryzyko przeuczenia - "
        "metalicznego, „sztucznego” głosu.\n\n"
        "Zalecane: 150-300 dla 10-30 min próbek; przy bardzo małej "
        "ilości próbek, poniżej 5 min, raczej 100-200. Obserwuj wykres strat - gdy "
        "przestaje spadać, dalszy trening niewiele daje.\n\n"
        "Przy wznawianiu to łączny cel: checkpoint z epoki 100 "
        "+ 200 epok → dotrenuje 101-200."
    ),
    "rvc_batch": (
        "Ile próbek model przetwarza naraz w jednym kroku.\n\n"
        "Większy = szybciej, ale więcej pamięci GPU. Na Colab T4: "
        "4-8. Przy mniej niż 10 min próbek lepiej 4. Jeśli trening "
        "kończy się błędem „CUDA out of memory” - zmniejsz.\n\n"
        "Przy wznawianiu ustaw taki sam jak w pierwszym treningu."
    ),
    "rvc_f0": (
        "Algorytm, który odczytuje wysokość głosu, czyli melodię, z próbek.\n\n"
        "rmvpe - zalecany, najlepszy kompromis jakości i szybkości.\n"
        "fcpe - szybki, podobna jakość.\n"
        "crepe-tiny - szybki, lżejszy wariant crepe.\n"
        "crepe - dokładny przy śpiewie, ale wolny.\n\n"
        "Hybrid jest niedostępny - w Applio 3.6.5 nie działa."
    ),
    "rvc_embedder": (
        "Model, który zamienia głos na „treść” - to, co jest mówione - "
        "oddzieloną od barwy.\n\n"
        "contentvec - zalecany, domyślny w Applio, działa dla polskiego.\n"
        "spin / spin-v2 - nowsze warianty, czasem czystsza artykulacja.\n"
        "chinese/japanese/korean-hubert-base - tylko dla tych języków.\n\n"
        "Zakładka 3 domyślnie bierze ten sam model z pliku .pth. "
        "Inny wybór przy konwersji psuje artykulację."
    ),
    "rvc_save_every": (
        "Checkpoint to zapis stanu treningu - pliki G i D, zwykle po kilkaset "
        "MB. Pozwala wznowić trening po przerwaniu. Suwak ustala, co ile "
        "epok je zapisywać i kopiować na Dysk Google.\n\n"
        "W trakcie treningu zapisujemy tylko G/D - mały model do konwersji "
        "powstaje na końcu albo po „Zatrzymaj trening”, z ostatniego G.\n\n"
        "Zalecane: 25 przy 100-300 epokach. Przy krótkim teście ustaw N "
        "równe liczbie epok, żeby nie zapisywać w połowie. Liczba epok "
        "najlepiej podzielna przez N."
    ),
    "rvc_auto_backup": (
        "Po każdym zapisie checkpointu kopiuje parę G/D na Dysk Google, do "
        "MyDrive/RVC_Colab_Models/<nazwa>/checkpoints/. Zostają "
        f"{AUTO_BACKUP_KEEP} ostatnie pary, starsze są usuwane. Każda kopia "
        "pojawia się w „Status zapisu” w kroku 5.\n\n"
        "Zalecane: włączone - gdy Colab rozłączy sesję, wznowisz trening "
        "z Dysku zamiast od zera. Wymaga komórki „Podłącz Google Drive” "
        "w notatniku."
    ),
    "rvc_resume": (
        "Kontynuuje trening od ostatniego checkpointu zamiast od zera.\n\n"
        "Ta sama sesja Colaba: wystarczy ta sama nazwa modelu.\n"
        "Nowa sesja: wgraj te same próbki i wybierz parę G/D z Dysku "
        "albo wgraj pliki ręcznie.\n\n"
        "„Epoki” oznaczają łączny cel - musi być większy niż epoka "
        "checkpointu."
    ),
    "rvc_drive_ckpt": (
        "Pary checkpointów G + D tego modelu znalezione na Dysku Google, "
        "najnowsza na górze. Wybierz, od której epoki kontynuować - "
        "zwykle najnowszą.\n\n"
        "Lista pusta? Sprawdź nazwę modelu i kliknij „Odśwież listę”."
    ),
    "rvc_manual_ckpt_acc": (
        "Gdy checkpointy masz na dysku komputera, a nie na Google Drive - "
        "wgraj tu oba pliki z tej samej epoki."
    ),
    "rvc_resume_g": (
        "Plik G_….pth - generator, czyli właściwa sieć tworząca głos. "
        "Musi pochodzić z tej samej epoki co plik D."
    ),
    "rvc_resume_d": (
        "Plik D_….pth - dyskryminator, sieć „oceniająca” generator "
        "w trakcie treningu. Bez niego wznowienie zaczęłoby się gorzej."
    ),
    "rvc_loss_title": (
        "Jak bardzo model się jeszcze myli - średnia z 50 kroków, "
        "niżej = lepiej. Każda seria ma własną skalę, żeby było widać "
        "spadek.\n\n"
        "Mel - różnica widma między wygenerowanym a prawdziwym głosem; "
        "najlepszy wskaźnik jakości.\n"
        "Generator - strata łączna generatora; może się wahać, liczy się "
        "trend.\n\n"
        "Gdy krzywe od dłuższego czasu stoją w miejscu albo rosną, możesz "
        "zatrzymać trening."
    ),
    "rvc_train_log_title": (
        "Szczegółowy log z silnika Applio - przydaje się, gdy coś pójdzie "
        "nie tak. Komunikaty silnika są po angielsku.\n\n"
        "Można go przewijać: przewinięty w górę zostaje w miejscu, a na "
        "samym dole sam podąża za nowymi liniami."
    ),
    "rvc_save_log_title": (
        "Dziennik wszystkich zapisów w tej sesji z godziną: automatyczne "
        "kopie checkpointów G/D, automatyczny zapis modelu po treningu "
        "i ręczne ponowienia. Wpisy nie znikają - widać, co i kiedy się "
        "zapisało, a co się nie udało."
    ),
    # --- Zakładka 3: konwersja ---
    "rvc_infer_model": (
        "Wytrenowany model: ZIP z zakładki 2 z plikami .pth i .index "
        "albo sam plik .pth.\n\n"
        "Zalecane: ZIP - z indeksem barwa jest zwykle wierniejsza."
    ),
    "rvc_index_acc": (
        "Tylko gdy wgrywasz sam plik .pth, a indeks masz osobno. "
        "ZIP z zakładki 2 ma już indeks w środku."
    ),
    "rvc_index_file": (
        "Plik .index pasujący do modelu - poprawia podobieństwo barwy. "
        "Bez niego konwersja działa, ale głos może być mniej podobny."
    ),
    "rvc_source": (
        "Nagranie mowy albo wokalu, któremu zmieniamy głos.\n\n"
        "Najlepiej czysty głos bez muzyki - przy piosence najpierw "
        "oddziel wokal od podkładu, np. w UVR, a po konwersji zmiksuj "
        "z powrotem."
    ),
    "rvc_pitch": (
        "Przesunięcie wysokości w półtonach, 12 = jedna oktawa.\n\n"
        "0 - bez zmian, gdy głos źródłowy i model są podobne.\n"
        "Męski → żeński: zwykle +12, czasem +8…+12.\n"
        "Żeński → męski: zwykle -12.\n\n"
        "Gdy wynik brzmi „chrapliwie” albo piskliwie - popraw o kilka "
        "półtonów."
    ),
    "rvc_f0_acc": (
        "Rzadziej zmieniane ustawienia konwersji: metoda wysokości, "
        "udział indeksu i model cech. Przy profilu „Mowa” albo „Śpiew” "
        "wystarczą wartości domyślne."
    ),
    "rvc_index_rate": (
        "Jak mocno wynik ma się trzymać barwy z próbek treningowych, "
        "zapisanej w pliku .index.\n\n"
        "0 - sam model, bez podpowiedzi z indeksu.\n"
        "0,75 - zalecane: barwa bliższa nagraniom, głos nadal naturalny.\n"
        "1 - maksymalnie jak próbki. Przy słabym nagraniu źródłowym "
        "może brzmieć sztucznie.\n\n"
        "Działa tylko, gdy model ma plik .index. Bez indeksu suwak "
        "jest pomijany."
    ),
    "rvc_infer_embedder": (
        "Model, który przy konwersji odczytuje treść nagrania. Musi być "
        "ten sam, którym trenowano.\n\n"
        "Z modelu - odczyt z pliku .pth. Zakładka 2 zapisuje tam wybór "
        "z pola „Model cech mowy”, więc po treningu nie trzeba go "
        "ustawiać drugi raz.\n"
        "contentvec - domyślny w Applio, właściwy dla polskiego.\n"
        "spin / spin-v2 - nowsze. Użyj tylko, jeśli tak trenowano.\n"
        "chinese/japanese/korean-hubert-base - tylko dla tych języków.\n\n"
        "Inny model niż przy treningu psuje artykulację, nawet gdy "
        "sam trening zakończył się poprawnie."
    ),
    "rvc_infer_f0": (
        "Algorytm odczytu wysokości, czyli melodii, nagrania źródłowego.\n\n"
        "rmvpe - zalecany, dobra jakość, odporny na szum.\n"
        "fcpe / crepe-tiny - zwykle szybsze.\n"
        "crepe - jak w klasycznym RVC GUI, dokładny przy śpiewie, wolniejszy.\n\n"
        "Fałszuje lub „pęka” na wysokich dźwiękach? Spróbuj crepe.\n"
        "Hybrid jest niedostępny - w Applio 3.6.5 nie działa."
    ),
    "rvc_out_audio": "Odsłuch wyniku konwersji w przeglądarce.",
    "rvc_out_download": "Wynik jako plik WAV do pobrania.",
    "rvc_infer_log_title": (
        "Szczegóły konwersji: użyty model, indeks, urządzenie GPU lub CPU "
        "i czas trwania."
    ),
}

BUTTON_HELP: dict[str, str] = {
    "rvc_prep_btn": "Tnie, czyści i normalizuje nagranie, a potem pakuje próbki do ZIP-a.",
    "rvc_train_btn": "Uruchamia trening albo wznowienie z ustawieniami powyżej.",
    "rvc_stop_btn": (
        "Przerywa trening w ciągu około sekundy. Jeśli akurat trwa zapis "
        "G/D, najpierw czeka na jego koniec, żeby checkpoint nie był "
        "uszkodzony. Potem pakuje model z ostatniego checkpointu do ZIP "
        "i zapisuje go na Dysku, jeśli jest podłączony."
    ),
    "rvc_ckpt_refresh": "Ponownie przeszukuje Google Drive w poszukiwaniu checkpointów tego modelu.",
    "rvc_zip_dl": (
        "Gotowy model: plik .pth z siecią + .index z indeksem barwy. "
        "Pobiera ZIP na Twój komputer - wgrasz go w zakładce 3."
    ),
    "rvc_g_dl": (
        "Ostatni checkpoint G - generator, sieć tworząca głos. Potrzebny "
        "tylko do wznowienia treningu w nowej sesji, zawsze razem z D."
    ),
    "rvc_d_dl": (
        "Ostatni checkpoint D - dyskryminator, sieć oceniająca generator. "
        "Potrzebny tylko do wznowienia treningu, zawsze razem z G."
    ),
    "rvc_drive_model_btn": (
        "Ponawia zapis ZIP-a modelu w MyDrive/RVC_Colab_Models/ - np. gdy "
        "Dysk podłączono dopiero po treningu. Wynik pojawi się w „Status zapisu”."
    ),
    "rvc_infer_btn": "Nakłada barwę modelu na nagranie źródłowe.",
}
