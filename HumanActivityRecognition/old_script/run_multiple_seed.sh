#!/bin/bash

# Array dei seed da utilizzare
SEEDS=("43" "44" "45" "46" "47")

# Directory di log (opzionale)
LOG_DIR="logs"
mkdir -p $LOG_DIR

echo "Avvio esecuzioni multiple con diversi seed..."
echo "Seeds da utilizzare: ${SEEDS[@]}"
echo "=================================="

# Loop attraverso tutti i seed
for seed in "${SEEDS[@]}"
do
    echo ""
    echo "--- Esecuzione con seed: $seed ---"
    echo "Inizio: $(date)"
    
    # Esegui lo script Python con il seed corrente
    # Salva output e errori in file di log separati
    python main_without_anything.py --seed $seed > "$LOG_DIR/output_seed_$seed.log" 2> "$LOG_DIR/error_seed_$seed.log"
    
    # Controlla il codice di uscita
    if [ $? -eq 0 ]; then
        echo "✓ Seed $seed completato con successo"
    else
        echo "✗ Errore con seed $seed - controlla $LOG_DIR/error_seed_$seed.log"
    fi
    
    echo "Fine: $(date)"
done

echo ""
echo "=================================="
echo "Tutte le esecuzioni completate!"
echo "Log salvati in: $LOG_DIR/"
echo "File generati:"
ls -la $LOG_DIR/