TONE_KEYS = {"s", "f", "r", "x", "j"}

VOWEL_TONES = {
    "a": {"": "a", "s": "\u00e1", "f": "\u00e0", "r": "\u1ea3", "x": "\u00e3", "j": "\u1ea1"},
    "\u0103": {"": "\u0103", "s": "\u1eaf", "f": "\u1eb1", "r": "\u1eb3", "x": "\u1eb5", "j": "\u1eb7"},
    "\u00e2": {"": "\u00e2", "s": "\u1ea5", "f": "\u1ea7", "r": "\u1ea9", "x": "\u1eab", "j": "\u1ead"},
    "e": {"": "e", "s": "\u00e9", "f": "\u00e8", "r": "\u1ebb", "x": "\u1ebd", "j": "\u1eb9"},
    "\u00ea": {"": "\u00ea", "s": "\u1ebf", "f": "\u1ec1", "r": "\u1ec3", "x": "\u1ec5", "j": "\u1ec7"},
    "i": {"": "i", "s": "\u00ed", "f": "\u00ec", "r": "\u1ec9", "x": "\u0129", "j": "\u1ecb"},
    "o": {"": "o", "s": "\u00f3", "f": "\u00f2", "r": "\u1ecf", "x": "\u00f5", "j": "\u1ecd"},
    "\u00f4": {"": "\u00f4", "s": "\u1ed1", "f": "\u1ed3", "r": "\u1ed5", "x": "\u1ed7", "j": "\u1ed9"},
    "\u01a1": {"": "\u01a1", "s": "\u1edb", "f": "\u1edd", "r": "\u1edf", "x": "\u1ee1", "j": "\u1ee3"},
    "u": {"": "u", "s": "\u00fa", "f": "\u00f9", "r": "\u1ee7", "x": "\u0169", "j": "\u1ee5"},
    "\u01b0": {"": "\u01b0", "s": "\u1ee9", "f": "\u1eeb", "r": "\u1eed", "x": "\u1eef", "j": "\u1ef1"},
    "y": {"": "y", "s": "\u00fd", "f": "\u1ef3", "r": "\u1ef7", "x": "\u1ef9", "j": "\u1ef5"},
}

CHAR_TO_VOWEL = {}
for base, tones in VOWEL_TONES.items():
    for tone, char in tones.items():
        CHAR_TO_VOWEL[char] = (base, tone, False)
        CHAR_TO_VOWEL[char.upper()] = (base, tone, True)


def compose_vowel(base, tone="", uppercase=False):
    char = VOWEL_TONES[base][tone]
    return char.upper() if uppercase else char


def split_vowel(char):
    return CHAR_TO_VOWEL.get(char)


def transform_last_char(char, key):
    if char.lower() == "d" and key == "d":
        return "\u0110" if char.isupper() else "\u0111"

    vowel = split_vowel(char)
    if vowel is None:
        return None

    base, tone, uppercase = vowel
    transforms = {
        ("a", "a"): "\u00e2",
        ("a", "w"): "\u0103",
        ("e", "e"): "\u00ea",
        ("o", "o"): "\u00f4",
        ("o", "w"): "\u01a1",
        ("u", "w"): "\u01b0",
    }
    new_base = transforms.get((base, key))
    if new_base is None:
        return None
    return compose_vowel(new_base, tone, uppercase)


class VietnameseTelexComposer:
    def __init__(self, uppercase_output=False):
        self.text = ""
        self.uppercase_output = uppercase_output

    def clear(self):
        self.text = ""

    def backspace(self):
        self.text = self.text[:-1]

    def feed_label(self, label):
        if label == "Space":
            self.text += " "
            return self.text
        if label == "Nothing" or len(label) != 1 or not label.isalpha():
            return self.text

        key = label.lower()
        self.feed_key(key)
        return self.text

    def feed_key(self, key):
        if not self.text:
            self.text = key.upper() if self.uppercase_output else key
            return

        transformed = transform_last_char(self.text[-1], key)
        if transformed is not None:
            self.text = self.text[:-1] + transformed
            return

        if key in TONE_KEYS and self.apply_tone(key):
            return

        self.text += key.upper() if self.uppercase_output else key

    def apply_tone(self, tone_key):
        for index in range(len(self.text) - 1, -1, -1):
            if self.text[index].isspace():
                return False
            vowel = split_vowel(self.text[index])
            if vowel is None:
                continue
            base, current_tone, uppercase = vowel
            new_tone = "" if current_tone == tone_key else tone_key
            self.text = self.text[:index] + compose_vowel(base, new_tone, uppercase) + self.text[index + 1 :]
            return True
        return False
