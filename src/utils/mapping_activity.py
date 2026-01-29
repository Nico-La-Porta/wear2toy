import json
import os
from app_config import REPORTS_DIR

# Mapping hardcodato per la palla
BALL_ACTION_MAPPING = {
    "original_to_name": {
        11: "throw",
        19: "grab", 
        21: "stack",
        41: "move"
    },
    "original_to_encoded": {
        11: 0,  # throw
        19: 1,  # grab
        21: 2,  # stack
        41: 3   # move
    },
    "encoded_to_name": {
        0: "throw",
        1: "grab",
        2: "stack", 
        3: "move"
    },
    "name_to_encoded": {
        "throw": 0,
        "grab": 1,
        "stack": 2,
        "move": 3
    }
}

# Mapping hardcodato per il cucchiaio
SPOON_ACTION_MAPPING = {
    "original_to_name": {
        10: "hold",
        11: "throw",
        41: "move"
    },
    "original_to_encoded": {
        10: 0,  # hold
        11: 1,  # throw
        41: 2   # move
    },
    "encoded_to_name": {
        0: "hold",
        1: "throw",
        2: "move"
    },
    "name_to_encoded": {
        "hold": 0,
        "throw": 1,
        "move": 2
    }
}

CAR_ACTION_MAPPING = {
    "original_to_name": {
        4: "put down",
        6: "lift",
        7: "lower",
        8: "turn",
        10:"hold",
        13:"flip",
        21:"stack",
        23:"wheel turn",
        25:"drive",
        41:"move"
    },
    "original_to_encoded": {
        4: 0,  # put down
        6: 1,  # lift
        7: 2,  # lower
        8: 3,  # turn
        10: 4, # hold
        13: 5, # flip
        21: 6, # stack
        23: 7, # wheel turn
        25: 8, # drive
        41: 9  # move
    },
    "encoded_to_name": {
        0: "put down",
        1: "lift",
        2: "lower",
        3: "turn",
        4: "hold",
        5: "flip",
        6: "stack",
        7: "wheel turn",
        8: "drive",
        9: "move"
    },
    "name_to_encoded": {
        "put down": 0,
        "lift": 1,
        "lower": 2,
        "turn": 3,
        "hold": 4,
        "flip": 5,
        "stack": 6,
        "wheel turn": 7,
        "drive": 8,
        "move": 9
    }
}



ELEPHANT_ACTION_MAPPING = {
    "original_to_name": {
        9: "pull",
        11: "throw",
        6: "lift",
        18: "offer",
        27: "beat",
        13: "flip",
        36: "align",
        4: "put down",
        3: "touch",
        10: "hold",
        8: "turn",
        21: "stack",
        41: "move",
        12: "drop",
        32: "bring close",
        16: "hit"
    },
    "original_to_encoded": {
        3: 0,   # touch
        4: 1,   # put down
        6: 2,   # lift
        8: 3,   # turn
        9: 4,   # pull
        10: 5,  # hold
        11: 6,  # throw
        12: 7,  # drop
        13: 8,  # flip
        16: 9,  # hit
        18: 10, # offer
        21: 11, # stack
        27: 12, # beat
        32: 13, # bring close
        36: 14, # align
        41: 15  # move
    },
    "encoded_to_name": {
        0: "touch",
        1: "put down",
        2: "lift",
        3: "turn",
        4: "pull",
        5: "hold",
        6: "throw",
        7: "drop",
        8: "flip",
        9: "hit",
        10: "offer",
        11: "stack",
        12: "hit",
        13: "bring close",
        14: "align",
        15: "move"
    },
    "name_to_encoded": {
        "touch": 0,
        "put down": 1,
        "lift": 2,
        "turn": 3,
        "pull": 4,
        "hold": 5,
        "throw": 6,
        "drop": 7,
        "flip": 8,
        "hit": 9,
        "offer": 10,
        "stack": 11,
        "hit": 12,
        "bring close": 13,
        "align": 14,
        "move": 15
    }
}


DOLL_ACTION_MAPPING = {
    "original_to_name": {
        11: "throw",
        6: "lift",
        18: "offer",
        10: "hold",
        41: "move",
        16: "hit",
        30: "sit",
        32: "bring close",
        34:"feed"
    },
    "original_to_encoded": {
        6: 0,    # lift
        10: 1,   # hold
        11: 2,   # throw
        16: 3,   # hit
        18: 4,   # offer
        30: 5,   # sit
        32: 6,   # bring close
        34: 7,   # feed
        41: 8    # move
    },
    "encoded_to_name": {
        0: "lift",
        1: "hold",
        2: "throw",
        3: "hit",
        4: "offer",
        5: "sit",
        6: "bring close",
        7: "feed",
        8: "move"
    },
    "name_to_encoded": {
    "lift": 0,
    "hold": 1,
    "throw": 2,
    "hit": 3,
    "offer": 4,
    "sit": 5,
    "bring close": 6,
    "feed": 7,
    "move": 8
    }
}
# Mapping completo per tutti i giocattoli
TOY_MAPPINGS = {
    "BALL": BALL_ACTION_MAPPING,
    "SPOON": SPOON_ACTION_MAPPING,
    "CAR": CAR_ACTION_MAPPING,
    "ELEPHANT": ELEPHANT_ACTION_MAPPING,
    "DOLL": DOLL_ACTION_MAPPING
}

def save_all_mappings():
    """Salva tutti i mapping in un file JSON"""
    json_path = os.path.join(REPORTS_DIR, "toy_action_mappings.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(TOY_MAPPINGS, f, indent=2, ensure_ascii=False)
    print(f"Mapping salvato in: {json_path}")
    return json_path

def load_all_mappings():
    """Carica tutti i mapping dal file JSON"""
    json_path = os.path.join(REPORTS_DIR, "toy_action_mappings.json")
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            mappings = json.load(f)
        
        # Converte le chiavi string in int dove necessario per tutti i giocattoli
        for toy_name, toy_mapping in mappings.items():
            toy_mapping["original_to_name"] = {int(k): v for k, v in toy_mapping["original_to_name"].items()}
            toy_mapping["original_to_encoded"] = {int(k): v for k, v in toy_mapping["original_to_encoded"].items()}
        
        return mappings
    else:
        # Se non esiste, lo crea
        save_all_mappings()
        return TOY_MAPPINGS

def get_toy_mapping(toy_name):
    """Ottiene il mapping per un giocattolo specifico"""
    mappings = load_all_mappings()
    return mappings.get(toy_name.upper(), {})

def convert_original_to_names(Y_original, toy_name):
    """Converte da action_id originali a nomi per un giocattolo specifico"""
    mapping = get_toy_mapping(toy_name)
    if not mapping:
        raise ValueError(f"Mapping non trovato per {toy_name}")
    return [mapping["original_to_name"][y] for y in Y_original]

def convert_original_to_encoded(Y_original, toy_name):
    """Converte da action_id originali a encoded per un giocattolo specifico"""
    mapping = get_toy_mapping(toy_name)
    if not mapping:
        raise ValueError(f"Mapping non trovato per {toy_name}")
    return [mapping["original_to_encoded"][y] for y in Y_original]

def convert_encoded_to_names(Y_encoded, toy_name):
    """Converte da encoded a nomi per un giocattolo specifico"""
    mapping = get_toy_mapping(toy_name)
    if not mapping:
        raise ValueError(f"Mapping non trovato per {toy_name}")
    return [mapping["encoded_to_name"][y] for y in Y_encoded]

def get_class_names(toy_name):
    """Ritorna la lista ordinata dei nomi delle classi per un giocattolo"""
    mapping = get_toy_mapping(toy_name)
    if not mapping:
        raise ValueError(f"Mapping non trovato per {toy_name}")
    
    num_classes = len(mapping["encoded_to_name"])
    return [mapping["encoded_to_name"][i] for i in range(num_classes)]

def get_num_classes(toy_name):
    """Ritorna il numero di classi per un giocattolo"""
    mapping = get_toy_mapping(toy_name)
    return len(mapping.get("encoded_to_name", {}))

# Funzioni backward compatibility per BALL
def save_ball_mapping():
    """Backward compatibility - salva solo il mapping BALL"""
    return save_all_mappings()

def load_ball_mapping():
    """Backward compatibility - carica solo il mapping BALL"""
    return get_toy_mapping("BALL")

if __name__ == "__main__":
    # Salva tutti i mapping
    save_all_mappings()
    
    # Test per BALL
    print("Mapping per BALL:")
    ball_mapping = get_toy_mapping("BALL")
    for original, name in ball_mapping["original_to_name"].items():
        encoded = ball_mapping["original_to_encoded"][original]
        print(f"  {original} -> {name} -> {encoded}")
    
    # Test per SPOON
    print("\nMapping per SPOON:")
    spoon_mapping = get_toy_mapping("SPOON")
    for original, name in spoon_mapping["original_to_name"].items():
        encoded = spoon_mapping["original_to_encoded"][original]
        print(f"  {original} -> {name} -> {encoded}")
    
    # Test delle funzioni di conversione
    print("\nTest conversioni SPOON:")
    example_Y_spoon = [10, 11, 41]  # Esempio di action_id per SPOON
    Y_names_spoon = convert_original_to_names(example_Y_spoon, "SPOON")
    Y_encoded_spoon = convert_original_to_encoded(example_Y_spoon, "SPOON")
    
    print(f"Original IDs: {example_Y_spoon}")
    print(f"Names: {Y_names_spoon}")
    print(f"Encoded: {Y_encoded_spoon}")
    
    print(f"\nClass names per SPOON: {get_class_names('SPOON')}")
    print(f"Numero classi SPOON: {get_num_classes('SPOON')}")