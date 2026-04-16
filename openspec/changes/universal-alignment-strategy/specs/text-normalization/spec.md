## ADDED Requirements

### Requirement: Text normalization prepares ASR output for phoneme-based alignment

The system SHALL provide a pure-function text normalizer that transforms an ASR text string into a form safe for MMS-FA alignment. The normalizer SHALL not depend on any ML model and SHALL be deterministic for a given `(text, language)` input.

#### Scenario: Normalizing a simple English sentence

- **WHEN** the normalizer is called with `"Hello, world!"` and language `"en"`
- **THEN** it returns `"hello world"` (lowercased, punctuation stripped, whitespace collapsed)

#### Scenario: Determinism across calls

- **WHEN** the normalizer is called twice with the same input
- **THEN** both calls return byte-identical output

### Requirement: Numbers and currency are expanded to spoken form

The normalizer SHALL expand bare numbers, decimals, and currency amounts into their spoken-word equivalents before stripping symbols. Expansion SHALL use `num2words` or an equivalent multilingual library and SHALL honor the supplied language code.

#### Scenario: Expanding currency

- **WHEN** the normalizer is called with `"$100"` and language `"en"`
- **THEN** it returns a string containing `"one hundred dollars"`

#### Scenario: Expanding bare numbers

- **WHEN** the normalizer is called with `"I have 3 apples"` and language `"en"`
- **THEN** it returns `"i have three apples"`

#### Scenario: Expanding a number in a different language

- **WHEN** the normalizer is called with `"Tengo 3 manzanas"` and language `"es"`
- **THEN** it returns `"tengo tres manzanas"`

### Requirement: Symbols that crash phoneme models are removed

The normalizer SHALL strip characters that are known to crash or degrade phoneme-based aligners, including `$`, `%`, `#`, `@`, emoji, and other non-letter, non-space, non-apostrophe characters. Intra-word apostrophes (e.g., `"don't"`) SHALL be preserved.

#### Scenario: Stripping an emoji

- **WHEN** the normalizer is called with `"great work 🎉"` and language `"en"`
- **THEN** the output contains only alphabetic characters, spaces, and apostrophes, and does not contain the emoji

#### Scenario: Preserving intra-word apostrophes

- **WHEN** the normalizer is called with `"don't stop"` and language `"en"`
- **THEN** it returns `"don't stop"` with the apostrophe intact

### Requirement: Normalizer failures degrade gracefully

If number or currency expansion fails for a specific token (e.g., an unusual format `num2words` cannot parse), the normalizer SHALL strip the offending symbols and continue rather than raising, and SHALL log a warning identifying the token.

#### Scenario: Unparseable currency expression

- **WHEN** the normalizer is called with a currency token that `num2words` cannot expand
- **THEN** the token's symbols are stripped, the remaining text is returned, and a warning is logged identifying the token that could not be expanded
