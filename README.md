# Strategia Z‑Hedge sui Futures del Cacao con Integrazione di Dati Climatici

## Perché esistono due futures sul cacao (C e CC)
- **ICE Europe Cocoa (ticker C)**: scambiato a Londra, storicamente quotato in **GBP** per tonnellata metrica, consegna fisica presso magazzini/porti approvati in Europa/UK.
- **ICE US Cocoa (ticker CC)**: scambiato a New York, quotato in **USD** per tonnellata metrica, consegna fisica presso magazzini/porti approvati negli USA.
- **In comune**: sottostante (fave di cacao consegnabili secondo specifiche), consegna fisica via documenti (ricevute di magazzino), scadenze standard (mesi di consegna), taglia contrattuale simile (10 t), meccaniche di marginazione e variazione giornaliera.
- **Differenze chiave**: valuta di quotazione (GBP vs USD), sedi/orari di negoziazione e chiusura, calendari e sedi di consegna, tick size/tick value, convenzioni di regolamento. Queste differenze generano uno “spread” strutturale e opportunità di relative value tra i due mercati.

## Idea della strategia
- **Pairs trading (mean‑reversion)** tra **CC (USD)** e **C (GBP→USD)**. Stimiamo un hedge ratio (beta) via OLS su prezzi, costruiamo lo spread = CC − (alpha + beta·C_USD) e il relativo z‑score su una finestra fissa; apriamo/chiudiamo posizioni quando lo z‑score supera/sottoscende soglie predefinite.
- **Estensione climatica**: integriamo dati **NASA POWER** per la Costa d’Avorio (principale area produttiva), in particolare le precipitazioni mensili in mm, i livelli di umidità del terreno e la temperatura massima dell'aria, per verificare se alcune condizioni climatiche aiutano a prevedere movimenti dello spread.

## Risultati principali (riassunto)
- Backtest standard (non “causale”, non inclinato/tilted):
  - Sharpe di test (standard, base): **1.252** (campo `test_sharpe` da `results/backtests/standard/base/metrics.csv`).
  - Nota importante: questa versione ignora il disallineamento orario (Londra vs New York). Nella pratica non si possono eseguire le due leg “allo stesso close” come nel modello standard.
- Strategia **causale**: risolve il problema dell’allineamento temporale costruendo il segnale in modo causale (C_USD al tempo t con FX_{t−1} confrontato con CC_{t−1}) ed eseguendo al tempo t+1. Sharpe di test (causal, base): **0.349**.
- Con dati climatici (tilting delle soglie d’ingresso in funzione della pioggia): il Sharpe “causale” migliora (causal tilted, `test_sharpe`): **0.430**. Il miglioramento è significativo rispetto alla versione causale non tilted nello stesso periodo e con gli stessi costi.

## Nota su correlazioni e robustezza
- È presente una correlazione tra la pioggia mensile (mm) e lo spread con un **lag di 2 mesi**. Tuttavia, dato il numero di feature/lag testati, è normale attendersi talvolta p‑value “significativi” per puro caso (**multiple testing**). Per questo motivo l’uso operativo del segnale climatico è implementato come **tilt delle soglie** (più facile aprire la direzione favorita, più difficile la sfavorita).

## Modello Z‑Hedge (sintesi)
- Stima hedge: $alpha, beta = OLS(CC, C_USD)$ su una finestra di training (o sull’intero storico).
- Spread: `S_t = CC_t − (alpha + beta·C_USD_t)`.
- Segnale: `z_t = (S_t − media_rolling) / devst_rolling` (finestra fissa **252** giorni).
- Regole di trading: ingresso quando $|z|$ supera la soglia di entry; uscita quando $|z|$ rientra sotto la soglia di exit o colpisce la soglia di stop; **costi** fissati a **12 bps per leg** (per cambio posizione su notional totale $|1|+|beta|$).
- Variante **causale**: decisioni basate su `C_USD_t` calcolato con `FX_{t−1}` e `CC_{t−1}`; esecuzione al `t+1` per evitare look‑ahead e disallineamenti.

## Definizione di “evento meteo estremo”
- Nelle analisi su segnali di pioggia definiamo estremi come i bucket di percentile **≤20** e **≥80** all’interno del mese di calendario (climatologia 1981–2020). Questi bucket guidano il **tilting** delle soglie: si allenta la soglia nella direzione favorita e si irrigidisce nell’altra, mantenendo causale l’allineamento temporale (lag mensile scelto su training).

## Struttura del progetto (cartelle principali)
- `src/`: codice modulare (IO dati, FX via yfinance, preprocess, OLS/spread/z‑score, motore di backtest, reporting, integrazione POWER e segnali di pioggia, ricerca parametri).
- `scripts/`: comandi CLI per costruire dataset, pannelli climatici, correlazioni, backtest unificato (standard/causale, con/senza tilting), tuning parametri e report.
- `results/`: output organizzati
  - `backtests/standard/{base,tilted}/` e `backtests/causal/{base,tilted}/` con `{metrics,equity,pnl,positions}.csv`
  - `backtests/tuning/` con i `best_params_{standard,causal}.csv` e i riassunti
  - `tables/power/lagN/` con `correlations.csv` (+ `correlations_meta.json`)
  - `tables/rain_signals/` con tabelle su segnali e regressioni
  - `REPORT.md` con link rapidi agli artifact
- `data/`: `raw/`, `external/`, `processed/`, `derived/` (creati dagli script alla prima esecuzione)
- `notebooks/`: analisi esplorativa (la logica di produzione è in `src/`)

## Requisiti e installazione
- Python 3.9+ e pacchetti da `requirements.txt`.
- Installazione rapida:

```bash
pip install -r requirements.txt
```

## Dati e fonti (input locali)
- Excel prezzi: posizionare `data/raw/dati_cacao.xlsx` (fogli con prezzi storici per `CC` e `C`).
  - Costruzione CSV: `python scripts/build_datasets.py` → crea `data/processed/*` e scarica FX GBPUSD in `data/external/gbpusd.csv` (via yfinance).
- Dati climatici POWER (point, CSV locali): copiare i file `POWER_Point_Monthly_*.csv` in `data/raw/`.
  - Pannello climatico: `python scripts/build_power_point_panel.py` → `data/derived/power_monthly_civ_gha.csv`, `results/signals/power_point_panel.csv`.
  - Correlazioni: `python scripts/eval_power_point_correlations.py --lags 0 1 2` → tabelle sotto `results/tables/power/lag{N}/`.
  - Segnali pioggia: `python scripts/eval_rain_signals.py --lag 15 --horizon 20` → tabelle in `results/tables/rain_signals/`.

## Come si esegue (sintesi)
- Pipeline completa (rigenerazione, tuning, backtest standard+causale, report):

```bash
scripts/run_full_pipeline.sh \
  --clean --rebuild-data \
  --lags "0 1 2" \
  --grid-n 11 \
  --train-start 2011-01-01 --train-end 2017-12-31 \
  --test-start 2018-01-01  --test-end 2022-12-31
```

- Il backtest legge automaticamente i best params salvati in `results/backtests/tuning/`.

### Output principali (dopo la pipeline)
- `results/tables/backtest_compare.csv`
- `results/backtests/standard/base/{metrics,equity,pnl,positions}.csv`
- `results/backtests/causal/base/{metrics,equity,pnl,positions}.csv`
- `results/backtests/standard/tilted/{metrics,equity,pnl,positions}.csv`
- `results/backtests/causal/tilted/{metrics,equity,pnl,positions}.csv`
- `results/REPORT.md`

## Conclusioni
- I futures sul cacao di Londra (**C**) e New York (**CC**) offrono una base naturale per una strategia market‑neutral. Lo Z‑Hedge standard mostra un buono Sharpe di test (**≈1.25**), ma non è eseguibile “così com’è” per via dei diversi orari di chiusura dei mercati.
- La variante **causale** risolve questo limite allineando le decisioni in modo non sincrono; l’integrazione dei dati **POWER** (precipitazioni) tramite tilting delle soglie migliora ulteriormente lo Sharpe causale (**≈0.35 → ≈0.43** nel periodo analizzato), pur con le dovute cautele statistiche sul multiple testing.
- Il progetto è riproducibile e modulare.
