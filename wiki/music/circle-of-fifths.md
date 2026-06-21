---
name: circle-of-fifths
description: Circle of Fifths for guitar — why fifths form a closed loop of 12 keys, how to read key signatures (sharps clockwise, flats counter-clockwise), find a key's I-IV-V-vi-ii-iii chords, transpose songs, and understand relative minors. Includes CAGED-friendly chord shapes and fretboard intuition.
created: 2026-06-21
updated: 2026-06-21
tags: [music-theory, guitar, harmony, chord-progressions]
---

# Circle of Fifths (Guitar)

## Why This Exists

**Problem.** You can play chords but don't know *why* a song moves G → C → D, why "Wonderwall" is in Em but the chords look like G/D/Am, or how to transpose a song up two steps for a singer. Theory feels like a separate universe from the fretboard.

**Key insight.** Stack perfect fifths — the most consonant interval after the octave — and after 12 jumps you land back where you started. That single loop encodes: every key signature, every diatonic chord family, every smooth modulation, every relative minor, and the V→I pull that drives almost all Western music.

**Reach for this when:**
- You want to know what chords belong in a key (and which ones don't).
- A capo conversation comes up ("what key is capo 3 in G shape?").
- You're transposing a song or writing a chord progression.
- You see roman numerals (I, IV, V, vi) and want them to click.
- You're memorizing the order of sharps/flats for sight-reading.
- You're improvising and need to find the "home note" by ear.

**Don't reach for this when:**
- You're learning a specific song's mechanics — tab and ear are faster.
- You're working in modal or jazz harmony (modal interchange, tritone subs) — the circle is the *starting* map, not the full territory.
- You just need a chord shape — that's a chord chart, not theory.

## Diagrams

### 1. The wheel — major key / relative minor / key signature

<svg viewBox="0 0 480 480" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:560px;display:block;margin:0 auto;border-radius:12px">
  <defs>
    <radialGradient id="bgWheel1" cx="0.5" cy="0.5" r="0.6">
      <stop offset="0%" stop-color="#ffffff"/>
      <stop offset="100%" stop-color="#f5f2e8"/>
    </radialGradient>
  </defs>
  <rect width="480" height="480" rx="12" fill="url(#bgWheel1)"/>
  <rect width="480" height="480" rx="12" fill="none" stroke="#d8d2c0" stroke-width="1.5"/>

  <!-- two reference rings -->
  <circle cx="240" cy="240" r="200" fill="none" stroke="#bfb8a3" stroke-width="1.5"/>
  <circle cx="240" cy="240" r="140" fill="none" stroke="#d4cdb8" stroke-width="1.2"/>
  <circle cx="240" cy="240" r="60"  fill="#f0eada" stroke="#bfb8a3" stroke-width="1.2"/>

  <!-- 12 spokes between the two main rings -->
  <g stroke="#cfc8b3" stroke-width="0.8">
    <line x1="240"   y1="100"  x2="240"   y2="40"/>
    <line x1="310"   y1="118.7" x2="362.5" y2="65"/>
    <line x1="361.2" y1="170"  x2="415"   y2="117.5"/>
    <line x1="380"   y1="240"  x2="440"   y2="240"/>
    <line x1="361.2" y1="310"  x2="415"   y2="362.5"/>
    <line x1="310"   y1="361.2" x2="362.5" y2="415"/>
    <line x1="240"   y1="380"  x2="240"   y2="440"/>
    <line x1="170"   y1="361.2" x2="117.5" y2="415"/>
    <line x1="118.7" y1="310"  x2="65"    y2="362.5"/>
    <line x1="100"   y1="240"  x2="40"    y2="240"/>
    <line x1="118.7" y1="170"  x2="65"    y2="117.5"/>
    <line x1="170"   y1="118.7" x2="117.5" y2="65"/>
  </g>

  <!-- example diameter: C ↔ F♯/G♭ (tritone partners) -->
  <line x1="240" y1="80" x2="240" y2="400" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="6 4" opacity="0.7"/>

  <!-- 12 major key labels (outer ring, r ≈ 170) -->
  <g font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" fill="#1f2937">
    <text x="240"   y="80"  font-size="24" font-weight="800">C</text>
    <text x="325"   y="102.8" font-size="22" font-weight="700">G</text>
    <text x="377.2" y="155"   font-size="22" font-weight="700">D</text>
    <text x="400"   y="240"   font-size="22" font-weight="700">A</text>
    <text x="377.2" y="325"   font-size="22" font-weight="700">E</text>
    <text x="325"   y="377.2" font-size="22" font-weight="700">B</text>
    <text x="240"   y="400"   font-size="18" font-weight="800" fill="#b45309">F♯ / G♭</text>
    <text x="155"   y="377.2" font-size="22" font-weight="700">D♭</text>
    <text x="102.8" y="325"   font-size="22" font-weight="700">A♭</text>
    <text x="80"    y="240"   font-size="22" font-weight="700">E♭</text>
    <text x="102.8" y="155"   font-size="22" font-weight="700">B♭</text>
    <text x="155"   y="102.8" font-size="22" font-weight="700">F</text>
  </g>

  <!-- key signatures outside the wheel -->
  <g font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="700">
    <text x="240"   y="22"  fill="#6b7280">0</text>
    <text x="340"   y="46.5" fill="#b45309">1♯</text>
    <text x="421.5" y="125"  fill="#b45309">2♯</text>
    <text x="458"   y="240"  fill="#b45309">3♯</text>
    <text x="421.5" y="355"  fill="#b45309">4♯</text>
    <text x="340"   y="433.5" fill="#b45309">5♯</text>
    <text x="240"   y="458"  fill="#6b7280">6♯ = 6♭</text>
    <text x="140"   y="433.5" fill="#1e3a5f">5♭</text>
    <text x="58.5"  y="355"  fill="#1e3a5f">4♭</text>
    <text x="22"    y="240"  fill="#1e3a5f">3♭</text>
    <text x="58.5"  y="125"  fill="#1e3a5f">2♭</text>
    <text x="140"   y="46.5" fill="#1e3a5f">1♭</text>
  </g>

  <!-- relative-minor labels (inner ring, r ≈ 105) -->
  <g font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" fill="#3b82c4" font-size="14" font-weight="700">
    <text x="240"   y="135">Am</text>
    <text x="292.5" y="149.1">Em</text>
    <text x="330.9" y="187.5">Bm</text>
    <text x="345"   y="240">F♯m</text>
    <text x="330.9" y="292.5">C♯m</text>
    <text x="292.5" y="330.9">G♯m</text>
    <text x="240"   y="345">E♭m</text>
    <text x="187.5" y="330.9">B♭m</text>
    <text x="149.1" y="292.5">Fm</text>
    <text x="135"   y="240">Cm</text>
    <text x="149.1" y="187.5">Gm</text>
    <text x="187.5" y="149.1">Dm</text>
  </g>

  <!-- center label -->
  <g font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central">
    <text x="240" y="228" font-size="11" font-weight="700" fill="#6b7280" letter-spacing="2">CIRCLE</text>
    <text x="240" y="245" font-size="11" font-weight="700" fill="#6b7280" letter-spacing="2">OF FIFTHS</text>
    <text x="240" y="262" font-size="9"  fill="#9aa0a6">major / minor</text>
  </g>

  <!-- legend strip across the top -->
  <g font-family="ui-sans-serif, system-ui, sans-serif" font-size="11" font-weight="600">
    <text x="380" y="12" fill="#b45309" text-anchor="end">→ CW: +1 sharp</text>
    <text x="100" y="12" fill="#1e3a5f" text-anchor="start">CCW: +1 flat ←</text>
  </g>
</svg>
### 2. Interval map — same wheel, labeled from a root note (C)

<svg viewBox="0 0 560 520" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:620px;display:block;margin:0 auto;border-radius:12px">
  <defs>
    <linearGradient id="bgWheel2" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%"  stop-color="#eef3fb"/>
      <stop offset="50%" stop-color="#fdfcf7"/>
      <stop offset="100%" stop-color="#fbecdf"/>
    </linearGradient>
  </defs>
  <rect width="560" height="520" rx="12" fill="url(#bgWheel2)"/>
  <rect width="560" height="520" rx="12" fill="none" stroke="#d8d2c0" stroke-width="1.5"/>

  <!-- top banner -->
  <g font-family="ui-sans-serif, system-ui, sans-serif" dominant-baseline="central">
    <rect x="380" y="22" width="160" height="26" rx="13" fill="#d97757"/>
    <text x="460" y="36" fill="#fff" text-anchor="middle" font-size="11" font-weight="800" letter-spacing="2">MAJOR-SIDE  →  CW</text>

    <rect x="20" y="22" width="160" height="26" rx="13" fill="#3b82c4"/>
    <text x="100" y="36" fill="#fff" text-anchor="middle" font-size="11" font-weight="800" letter-spacing="2">CCW  ←  MINOR-SIDE</text>

    <text x="280" y="36" fill="#1f2937" text-anchor="middle" font-size="13" font-weight="700">intervals from root C</text>
  </g>

  <!-- two rings -->
  <circle cx="280" cy="280" r="170" fill="none" stroke="#bfb8a3" stroke-width="1.5"/>
  <circle cx="280" cy="280" r="120" fill="#f5f2e8" stroke="#cfc8b3" stroke-width="1.2"/>

  <!-- tritone diameter -->
  <line x1="280" y1="110" x2="280" y2="450" stroke="#dc2626" stroke-width="2" stroke-dasharray="6 4"/>

  <!-- 12 positions (radius 170 for dots).
       angle = -90 + i*30 deg.  cx = 280 + 170*cos(angle), cy = 280 + 170*sin(angle)  -->
  <!-- 0 (top)   = C     (root)              -->
  <circle cx="280" cy="110" r="11" fill="#1f2937" stroke="#fff" stroke-width="2"/>
  <!-- 1 (1 o'clock) = G   P5                -->
  <circle cx="365" cy="132.8" r="10" fill="#d97757" stroke="#fff" stroke-width="2"/>
  <!-- 2 (2 o'clock) = D   M2                -->
  <circle cx="427.2" cy="195" r="10" fill="#d97757" stroke="#fff" stroke-width="2"/>
  <!-- 3 (3 o'clock) = A   M6                -->
  <circle cx="450" cy="280" r="10" fill="#d97757" stroke="#fff" stroke-width="2"/>
  <!-- 4 (4 o'clock) = E   M3                -->
  <circle cx="427.2" cy="365" r="10" fill="#d97757" stroke="#fff" stroke-width="2"/>
  <!-- 5 (5 o'clock) = B   M7                -->
  <circle cx="365" cy="427.2" r="10" fill="#d97757" stroke="#fff" stroke-width="2"/>
  <!-- 6 (bottom)   = F#/Gb  tritone         -->
  <circle cx="280" cy="450" r="11" fill="#dc2626" stroke="#fff" stroke-width="2"/>
  <!-- 7 (7 o'clock) = Db  b9                -->
  <circle cx="195" cy="427.2" r="10" fill="#3b82c4" stroke="#fff" stroke-width="2"/>
  <!-- 8 (8 o'clock) = Ab  m6                -->
  <circle cx="132.8" cy="365" r="10" fill="#3b82c4" stroke="#fff" stroke-width="2"/>
  <!-- 9 (9 o'clock) = Eb  m3                -->
  <circle cx="110" cy="280" r="10" fill="#3b82c4" stroke="#fff" stroke-width="2"/>
  <!-- 10 (10 o'clock) = Bb  b7              -->
  <circle cx="132.8" cy="195" r="10" fill="#3b82c4" stroke="#fff" stroke-width="2"/>
  <!-- 11 (11 o'clock) = F   P4              -->
  <circle cx="195" cy="132.8" r="10" fill="#3b82c4" stroke="#fff" stroke-width="2"/>

  <!-- note + interval labels (placed *outside* the dots, radius ≈ 200) -->
  <g font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central">
    <!-- 0 C R -->
    <text x="280" y="78" font-size="22" font-weight="800" fill="#1f2937">C</text>
    <text x="280" y="95" font-size="11" font-weight="700" fill="#6b7280">root</text>
    <!-- 1 G P5 -->
    <text x="390" y="105" font-size="20" font-weight="800" fill="#1f2937">G</text>
    <text x="390" y="122" font-size="11" font-weight="700" fill="#b45309">P5 · +7</text>
    <!-- 2 D M2 -->
    <text x="455" y="170" font-size="20" font-weight="800" fill="#1f2937">D</text>
    <text x="455" y="187" font-size="11" font-weight="700" fill="#b45309">M2 · +2</text>
    <!-- 3 A M6 -->
    <text x="490" y="270" font-size="20" font-weight="800" fill="#1f2937">A</text>
    <text x="490" y="287" font-size="11" font-weight="700" fill="#b45309">M6 · +9</text>
    <!-- 4 E M3 -->
    <text x="455" y="370" font-size="20" font-weight="800" fill="#1f2937">E</text>
    <text x="455" y="387" font-size="11" font-weight="700" fill="#b45309">M3 · +4</text>
    <!-- 5 B M7 -->
    <text x="390" y="437" font-size="20" font-weight="800" fill="#1f2937">B</text>
    <text x="390" y="454" font-size="11" font-weight="700" fill="#b45309">M7 · +11</text>
    <!-- 6 F#/Gb TT -->
    <text x="280" y="478" font-size="20" font-weight="800" fill="#dc2626">F♯ / G♭</text>
    <text x="280" y="497" font-size="11" font-weight="800" fill="#dc2626">tritone · +6</text>
    <!-- 7 Db b9 -->
    <text x="170" y="437" font-size="20" font-weight="800" fill="#1f2937">D♭</text>
    <text x="170" y="454" font-size="11" font-weight="700" fill="#1e3a5f">♭9 · +1</text>
    <!-- 8 Ab m6 -->
    <text x="105" y="370" font-size="20" font-weight="800" fill="#1f2937">A♭</text>
    <text x="105" y="387" font-size="11" font-weight="700" fill="#1e3a5f">m6 · +8</text>
    <!-- 9 Eb m3 -->
    <text x="70" y="270" font-size="20" font-weight="800" fill="#1f2937">E♭</text>
    <text x="70" y="287" font-size="11" font-weight="700" fill="#1e3a5f">m3 · +3</text>
    <!-- 10 Bb b7 -->
    <text x="105" y="170" font-size="20" font-weight="800" fill="#1f2937">B♭</text>
    <text x="105" y="187" font-size="11" font-weight="700" fill="#1e3a5f">♭7 · +10</text>
    <!-- 11 F P4 -->
    <text x="170" y="105" font-size="20" font-weight="800" fill="#1f2937">F</text>
    <text x="170" y="122" font-size="11" font-weight="700" fill="#1e3a5f">P4 · +5</text>
  </g>

  <!-- center caption -->
  <g font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central">
    <text x="280" y="265" font-size="13" font-weight="800" fill="#1f2937" letter-spacing="1">INTERVAL MAP</text>
    <text x="280" y="285" font-size="11" fill="#4b5563">step = 7 semitones</text>
    <text x="280" y="302" font-size="11" fill="#4b5563">= one perfect 5th</text>
  </g>
</svg>

> **Read it like this:** the warm (right) half holds the major-flavored intervals (P5, M2, M6, M3, M7); the cool (left) half holds the minor-flavored ones (P4, ♭7, m3, m6, ♭9). The red diameter is the tritone — opposite the root, no side, maximum tension.

#### Four things the interval map secretly encodes

**1. Inversion = mirror across the root–tritone axis.** Every interval and its octave-complement (they sum to 12 semitones) sit at mirrored positions on the map. P5 (+1 CW) mirrors P4 (−1 CCW); M3 (+4 CW) mirrors m6 (−4 CCW); M7 (+5 CW) mirrors m2 (−5 CCW). The tritone (+6) is *self-inverse* — its own mirror — which is the structural reason it sits dead opposite the root with no side. Inversion isn't a separate concept to memorize; it's just folding the map.

**2. Brightness axis: CW = bright, CCW = dark (literally).** Each CW step from the tonic *raises* one scale note by a semitone; each CCW step *flats* one. Stacking these moves produces the seven modes in order of brightness — same set of notes, different home base:

```
brightest  →  Lydian      (raised 4)
              Ionian      (major)
              Mixolydian  (♭7)
              Dorian      (♭3, ♭7)
              Aeolian     (natural minor: ♭3, ♭6, ♭7)
              Phrygian    (♭2, ♭3, ♭6, ♭7)
darkest    →  Locrian     (♭2, ♭3, ♭5, ♭6, ♭7)
```

So **how many CW vs. CCW notes a scale contains literally measures its brightness.** A Lydian solo over a major chord sounds "lifted" because every accidental is on the CW side; a Phrygian solo sounds dark because they're all on the CCW side.

**3. The map *generates* a major scale.** A major scale is **seven consecutive CW positions** starting one step CCW of the tonic. For C major:

> F (−1) · C (0) · G (+1) · D (+2) · A (+3) · E (+4) · B (+5)

Re-sort those by pitch and you have C–D–E–F–G–A–B — the C major scale, derived purely from circle positions, no memorization. The five leftover slots {F♯, C♯, G♯, D♯, A♯} form the F♯ major pentatonic — the piano's black keys. The map isn't a lookup table; it's a *generator* for scales, modes, and key signatures.

**4. Distance on the map = how related two keys are.** Adjacent positions share 6 of 7 scale notes; opposite positions share only 2. That's why composers historically modulate to "closely related keys" — circle adjacency *is* tonal closeness:

| Steps apart  | Shared notes (out of 7) | Modulation feel |
|---|---|---|
| 1 (C ↔ G)   | 6 | smooth, near-invisible |
| 2 (C ↔ D)   | 5 | smooth                  |
| 3 (C ↔ A)   | 4 | noticeable shift        |
| 6 (C ↔ F♯)  | 2 | maximally distant — the "tritone modulation" used for shock (bridge of *Sweet Child O' Mine*, B section of *Maria* from West Side Story) |

This also predicts which **modulating chords** work as pivots: chords that exist in both the source and destination keys (because they share many notes), and those are most plentiful when the keys are neighbors.

### 3. Chord-shape matrix — which clock positions light up

<svg viewBox="0 0 720 1000" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:760px;display:block;margin:0 auto;border-radius:12px">
  <rect width="720" height="1000" rx="12" fill="#fdfcf7"/>
  <rect width="720" height="1000" rx="12" fill="none" stroke="#d8d2c0" stroke-width="1.5"/>
  <!-- title row -->
  <text x="360" y="26" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="15" font-weight="800" fill="#1f2937">Chord shapes on the circle  ·  root = C (top)</text>
  <text x="120" y="50" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="11" font-weight="700" fill="#6b7280" letter-spacing="2">TRIADS</text>
  <text x="360" y="50" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="11" font-weight="700" fill="#6b7280" letter-spacing="2">7TH CHORDS · 1</text>
  <text x="600" y="50" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="11" font-weight="700" fill="#6b7280" letter-spacing="2">7TH CHORDS · 2</text>

  <!-- =================================================================== -->
  <!-- ROW 1: maj / min / dim                                                -->
  <!-- =================================================================== -->

  <!-- C major: {0, 1, 4} - "open CW fan" -->
  <g transform="translate(120,200)">
    <circle fill="#f5f2e8" stroke="#bfb8a3" stroke-width="1.2" cx="0" cy="0" r="70"/>
    <!-- 12 empty dots -->
    <circle fill="#d1cfc8" cx="0"     cy="-70" r="3"/>
    <circle fill="#d1cfc8" cx="35"    cy="-60.62" r="3"/>
    <circle fill="#d1cfc8" cx="60.62" cy="-35" r="3"/>
    <circle fill="#d1cfc8" cx="70"    cy="0" r="3"/>
    <circle fill="#d1cfc8" cx="60.62" cy="35" r="3"/>
    <circle fill="#d1cfc8" cx="35"    cy="60.62" r="3"/>
    <circle fill="#d1cfc8" cx="0"     cy="70" r="3"/>
    <circle fill="#d1cfc8" cx="-35"   cy="60.62" r="3"/>
    <circle fill="#d1cfc8" cx="-60.62" cy="35" r="3"/>
    <circle fill="#d1cfc8" cx="-70"   cy="0" r="3"/>
    <circle fill="#d1cfc8" cx="-60.62" cy="-35" r="3"/>
    <circle fill="#d1cfc8" cx="-35"   cy="-60.62" r="3"/>
    <!-- shape edges: 0 -> 1 -> 4 -> 0 -->
    <polygon fill="rgba(31,41,55,0.10)" stroke="#1f2937" stroke-width="2.2" points="0,-70 35,-60.62 60.62,35"/>
    <!-- lit + root -->
    <circle fill="#dc2626" stroke="#fff" stroke-width="2" cx="0" cy="-70" r="5.5"/>
    <circle fill="#1f2937" stroke="#fff" stroke-width="2" cx="35" cy="-60.62" r="5"/>
    <circle fill="#1f2937" stroke="#fff" stroke-width="2" cx="60.62" cy="35" r="5"/>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="800" fill="#1f2937" paint-order="stroke" stroke="#fdfcf7" stroke-width="4" x="0" y="-94">C</text>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="800" fill="#1f2937" paint-order="stroke" stroke="#fdfcf7" stroke-width="4" x="56" y="-82">G</text>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="800" fill="#1f2937" paint-order="stroke" stroke="#fdfcf7" stroke-width="4" x="88" y="56">E</text>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="17" font-weight="800" fill="#1f2937" x="0" y="114">C major</text>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="11" font-weight="600" fill="#6b7280"  x="0" y="132">1 – 3 – 5</text>
  </g>

  <!-- C minor: {0, 1, 9} - mirror of major (3rd flips side) -->
  <g transform="translate(360,200)">
    <circle fill="#f5f2e8" stroke="#bfb8a3" stroke-width="1.2" cx="0" cy="0" r="70"/>
    <circle fill="#d1cfc8" cx="0"     cy="-70" r="3"/>
    <circle fill="#d1cfc8" cx="35"    cy="-60.62" r="3"/>
    <circle fill="#d1cfc8" cx="60.62" cy="-35" r="3"/>
    <circle fill="#d1cfc8" cx="70"    cy="0" r="3"/>
    <circle fill="#d1cfc8" cx="60.62" cy="35" r="3"/>
    <circle fill="#d1cfc8" cx="35"    cy="60.62" r="3"/>
    <circle fill="#d1cfc8" cx="0"     cy="70" r="3"/>
    <circle fill="#d1cfc8" cx="-35"   cy="60.62" r="3"/>
    <circle fill="#d1cfc8" cx="-60.62" cy="35" r="3"/>
    <circle fill="#d1cfc8" cx="-70"   cy="0" r="3"/>
    <circle fill="#d1cfc8" cx="-60.62" cy="-35" r="3"/>
    <circle fill="#d1cfc8" cx="-35"   cy="-60.62" r="3"/>
    <!-- 0 -> 1 -> 9 -->
    <polygon fill="rgba(31,41,55,0.10)" stroke="#1f2937" stroke-width="2.2" points="0,-70 35,-60.62 -70,0"/>
    <circle fill="#dc2626" stroke="#fff" stroke-width="2" cx="0" cy="-70" r="5.5"/>
    <circle fill="#1f2937" stroke="#fff" stroke-width="2" cx="35" cy="-60.62" r="5"/>
    <circle fill="#1f2937" stroke="#fff" stroke-width="2" cx="-70" cy="0" r="5"/>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="800" fill="#1f2937" paint-order="stroke" stroke="#fdfcf7" stroke-width="4" x="0" y="-94">C</text>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="800" fill="#1f2937" paint-order="stroke" stroke="#fdfcf7" stroke-width="4" x="56" y="-82">G</text>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="800" fill="#1f2937" paint-order="stroke" stroke="#fdfcf7" stroke-width="4" x="-94" y="0">E♭</text>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="17" font-weight="800" fill="#1f2937" x="0" y="114">C minor</text>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="11" font-weight="600" fill="#6b7280"  x="0" y="132">1 – ♭3 – 5</text>
  </g>

  <!-- C dim: {0, 6, 9} - has tritone -->
  <g transform="translate(600,200)">
    <circle fill="#f5f2e8" stroke="#bfb8a3" stroke-width="1.2" cx="0" cy="0" r="70"/>
    <circle fill="#d1cfc8" cx="0"     cy="-70" r="3"/>
    <circle fill="#d1cfc8" cx="35"    cy="-60.62" r="3"/>
    <circle fill="#d1cfc8" cx="60.62" cy="-35" r="3"/>
    <circle fill="#d1cfc8" cx="70"    cy="0" r="3"/>
    <circle fill="#d1cfc8" cx="60.62" cy="35" r="3"/>
    <circle fill="#d1cfc8" cx="35"    cy="60.62" r="3"/>
    <circle fill="#d1cfc8" cx="0"     cy="70" r="3"/>
    <circle fill="#d1cfc8" cx="-35"   cy="60.62" r="3"/>
    <circle fill="#d1cfc8" cx="-60.62" cy="35" r="3"/>
    <circle fill="#d1cfc8" cx="-70"   cy="0" r="3"/>
    <circle fill="#d1cfc8" cx="-60.62" cy="-35" r="3"/>
    <circle fill="#d1cfc8" cx="-35"   cy="-60.62" r="3"/>
    <!-- tritone diameter explicit (0 -> 6) -->
    <line stroke="#dc2626" stroke-width="2" stroke-dasharray="5 3" x1="0" y1="-70" x2="0" y2="70"/>
    <!-- triangle 0 -> 9 -> 6 -->
    <polygon fill="rgba(31,41,55,0.10)" stroke="#1f2937" stroke-width="2.2" points="0,-70 -70,0 0,70"/>
    <circle fill="#dc2626" stroke="#fff" stroke-width="2" cx="0" cy="-70" r="5.5"/>
    <circle fill="#1f2937" stroke="#fff" stroke-width="2" cx="-70" cy="0" r="5"/>
    <circle fill="#1f2937" stroke="#fff" stroke-width="2" cx="0" cy="70" r="5"/>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="800" fill="#1f2937" paint-order="stroke" stroke="#fdfcf7" stroke-width="4" x="0" y="-94">C</text>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="800" fill="#1f2937" paint-order="stroke" stroke="#fdfcf7" stroke-width="4" x="-94" y="0">E♭</text>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="800" fill="#1f2937" paint-order="stroke" stroke="#fdfcf7" stroke-width="4" x="0" y="94">G♭</text>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="17" font-weight="800" fill="#1f2937" x="0" y="118">C dim</text>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="11" font-weight="600" fill="#6b7280"  x="0" y="136">1 – ♭3 – ♭5</text>
  </g>

  <!-- =================================================================== -->
  <!-- ROW 2: aug / maj7 / dom7                                              -->
  <!-- =================================================================== -->

  <!-- C aug: {0, 4, 8} - equilateral triangle -->
  <g transform="translate(120,480)">
    <circle fill="#f5f2e8" stroke="#bfb8a3" stroke-width="1.2" cx="0" cy="0" r="70"/>
    <circle fill="#d1cfc8" cx="0"     cy="-70" r="3"/>
    <circle fill="#d1cfc8" cx="35"    cy="-60.62" r="3"/>
    <circle fill="#d1cfc8" cx="60.62" cy="-35" r="3"/>
    <circle fill="#d1cfc8" cx="70"    cy="0" r="3"/>
    <circle fill="#d1cfc8" cx="60.62" cy="35" r="3"/>
    <circle fill="#d1cfc8" cx="35"    cy="60.62" r="3"/>
    <circle fill="#d1cfc8" cx="0"     cy="70" r="3"/>
    <circle fill="#d1cfc8" cx="-35"   cy="60.62" r="3"/>
    <circle fill="#d1cfc8" cx="-60.62" cy="35" r="3"/>
    <circle fill="#d1cfc8" cx="-70"   cy="0" r="3"/>
    <circle fill="#d1cfc8" cx="-60.62" cy="-35" r="3"/>
    <circle fill="#d1cfc8" cx="-35"   cy="-60.62" r="3"/>
    <!-- equilateral 0 -> 4 -> 8 -->
    <polygon fill="rgba(31,41,55,0.10)" stroke="#1f2937" stroke-width="2.2" points="0,-70 60.62,35 -60.62,35"/>
    <circle fill="#dc2626" stroke="#fff" stroke-width="2" cx="0" cy="-70" r="5.5"/>
    <circle fill="#1f2937" stroke="#fff" stroke-width="2" cx="60.62" cy="35" r="5"/>
    <circle fill="#1f2937" stroke="#fff" stroke-width="2" cx="-60.62" cy="35" r="5"/>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="800" fill="#1f2937" paint-order="stroke" stroke="#fdfcf7" stroke-width="4" x="0" y="-94">C</text>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="800" fill="#1f2937" paint-order="stroke" stroke="#fdfcf7" stroke-width="4" x="88" y="56">E</text>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="800" fill="#1f2937" paint-order="stroke" stroke="#fdfcf7" stroke-width="4" x="-88" y="56">A♭</text>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="17" font-weight="800" fill="#1f2937" x="0" y="114">C aug</text>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="11" font-weight="600" fill="#6b7280"  x="0" y="132">1 – 3 – ♯5</text>
  </g>

  <!-- Cmaj7: {0, 1, 4, 5} - tight CW cluster -->
  <g transform="translate(360,480)">
    <circle fill="#f5f2e8" stroke="#bfb8a3" stroke-width="1.2" cx="0" cy="0" r="70"/>
    <circle fill="#d1cfc8" cx="0"     cy="-70" r="3"/>
    <circle fill="#d1cfc8" cx="35"    cy="-60.62" r="3"/>
    <circle fill="#d1cfc8" cx="60.62" cy="-35" r="3"/>
    <circle fill="#d1cfc8" cx="70"    cy="0" r="3"/>
    <circle fill="#d1cfc8" cx="60.62" cy="35" r="3"/>
    <circle fill="#d1cfc8" cx="35"    cy="60.62" r="3"/>
    <circle fill="#d1cfc8" cx="0"     cy="70" r="3"/>
    <circle fill="#d1cfc8" cx="-35"   cy="60.62" r="3"/>
    <circle fill="#d1cfc8" cx="-60.62" cy="35" r="3"/>
    <circle fill="#d1cfc8" cx="-70"   cy="0" r="3"/>
    <circle fill="#d1cfc8" cx="-60.62" cy="-35" r="3"/>
    <circle fill="#d1cfc8" cx="-35"   cy="-60.62" r="3"/>
    <!-- 0 -> 1 -> 4 -> 5 -->
    <polygon fill="rgba(31,41,55,0.10)" stroke="#1f2937" stroke-width="2.2" points="0,-70 35,-60.62 60.62,35 35,60.62"/>
    <circle fill="#dc2626" stroke="#fff" stroke-width="2" cx="0" cy="-70" r="5.5"/>
    <circle fill="#1f2937" stroke="#fff" stroke-width="2" cx="35" cy="-60.62" r="5"/>
    <circle fill="#1f2937" stroke="#fff" stroke-width="2" cx="60.62" cy="35" r="5"/>
    <circle fill="#1f2937" stroke="#fff" stroke-width="2" cx="35" cy="60.62" r="5"/>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="800" fill="#1f2937" paint-order="stroke" stroke="#fdfcf7" stroke-width="4" x="0" y="-94">C</text>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="800" fill="#1f2937" paint-order="stroke" stroke="#fdfcf7" stroke-width="4" x="56" y="-82">G</text>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="800" fill="#1f2937" paint-order="stroke" stroke="#fdfcf7" stroke-width="4" x="88" y="56">E</text>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="800" fill="#1f2937" paint-order="stroke" stroke="#fdfcf7" stroke-width="4" x="56" y="84">B</text>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="17" font-weight="800" fill="#1f2937" x="0" y="114">C maj7</text>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="11" font-weight="600" fill="#6b7280"  x="0" y="132">1 – 3 – 5 – 7</text>
  </g>

  <!-- C7: {0, 1, 4, 10} - has tritone 4<->10 -->
  <g transform="translate(600,480)">
    <circle fill="#f5f2e8" stroke="#bfb8a3" stroke-width="1.2" cx="0" cy="0" r="70"/>
    <circle fill="#d1cfc8" cx="0"     cy="-70" r="3"/>
    <circle fill="#d1cfc8" cx="35"    cy="-60.62" r="3"/>
    <circle fill="#d1cfc8" cx="60.62" cy="-35" r="3"/>
    <circle fill="#d1cfc8" cx="70"    cy="0" r="3"/>
    <circle fill="#d1cfc8" cx="60.62" cy="35" r="3"/>
    <circle fill="#d1cfc8" cx="35"    cy="60.62" r="3"/>
    <circle fill="#d1cfc8" cx="0"     cy="70" r="3"/>
    <circle fill="#d1cfc8" cx="-35"   cy="60.62" r="3"/>
    <circle fill="#d1cfc8" cx="-60.62" cy="35" r="3"/>
    <circle fill="#d1cfc8" cx="-70"   cy="0" r="3"/>
    <circle fill="#d1cfc8" cx="-60.62" cy="-35" r="3"/>
    <circle fill="#d1cfc8" cx="-35"   cy="-60.62" r="3"/>
    <!-- highlight the internal tritone between E (4) and Bb (10) -->
    <line stroke="#dc2626" stroke-width="2" stroke-dasharray="5 3" x1="60.62" y1="35" x2="-60.62" y2="-35"/>
    <!-- internal tritone: E (4) ↔ Bb (10) -->
    <line stroke="#dc2626" stroke-width="2" stroke-dasharray="5 3" x1="60.62" y1="35" x2="-60.62" y2="-35"/>
    <polygon fill="rgba(31,41,55,0.10)" stroke="#1f2937" stroke-width="2.2" points="0,-70 35,-60.62 60.62,35 -60.62,-35"/>
    <circle fill="#dc2626" stroke="#fff" stroke-width="2" cx="0" cy="-70" r="5.5"/>
    <circle fill="#1f2937" stroke="#fff" stroke-width="2" cx="35" cy="-60.62" r="5"/>
    <circle fill="#1f2937" stroke="#fff" stroke-width="2" cx="60.62" cy="35" r="5"/>
    <circle fill="#1f2937" stroke="#fff" stroke-width="2" cx="-60.62" cy="-35" r="5"/>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="800" fill="#1f2937" paint-order="stroke" stroke="#fdfcf7" stroke-width="4" x="0" y="-94">C</text>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="800" fill="#1f2937" paint-order="stroke" stroke="#fdfcf7" stroke-width="4" x="56" y="-82">G</text>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="800" fill="#1f2937" paint-order="stroke" stroke="#fdfcf7" stroke-width="4" x="88" y="56">E</text>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="800" fill="#1f2937" paint-order="stroke" stroke="#fdfcf7" stroke-width="4" x="-88" y="-56">B♭</text>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="17" font-weight="800" fill="#1f2937" x="0" y="114">C7  (dom)</text>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="11" font-weight="600" fill="#6b7280"  x="0" y="132">1 – 3 – 5 – ♭7</text>
  </g>

  <!-- =================================================================== -->
  <!-- ROW 3: m7 / m7b5 / dim7                                               -->
  <!-- =================================================================== -->

  <!-- Cm7: {0, 1, 9, 10} - CCW cluster -->
  <g transform="translate(120,760)">
    <circle fill="#f5f2e8" stroke="#bfb8a3" stroke-width="1.2" cx="0" cy="0" r="70"/>
    <circle fill="#d1cfc8" cx="0"     cy="-70" r="3"/>
    <circle fill="#d1cfc8" cx="35"    cy="-60.62" r="3"/>
    <circle fill="#d1cfc8" cx="60.62" cy="-35" r="3"/>
    <circle fill="#d1cfc8" cx="70"    cy="0" r="3"/>
    <circle fill="#d1cfc8" cx="60.62" cy="35" r="3"/>
    <circle fill="#d1cfc8" cx="35"    cy="60.62" r="3"/>
    <circle fill="#d1cfc8" cx="0"     cy="70" r="3"/>
    <circle fill="#d1cfc8" cx="-35"   cy="60.62" r="3"/>
    <circle fill="#d1cfc8" cx="-60.62" cy="35" r="3"/>
    <circle fill="#d1cfc8" cx="-70"   cy="0" r="3"/>
    <circle fill="#d1cfc8" cx="-60.62" cy="-35" r="3"/>
    <circle fill="#d1cfc8" cx="-35"   cy="-60.62" r="3"/>
    <!-- 0 -> 1 -> 9 -> 10 -->
    <polygon fill="rgba(31,41,55,0.10)" stroke="#1f2937" stroke-width="2.2" points="0,-70 35,-60.62 -70,0 -60.62,-35"/>
    <circle fill="#dc2626" stroke="#fff" stroke-width="2" cx="0" cy="-70" r="5.5"/>
    <circle fill="#1f2937" stroke="#fff" stroke-width="2" cx="35" cy="-60.62" r="5"/>
    <circle fill="#1f2937" stroke="#fff" stroke-width="2" cx="-70" cy="0" r="5"/>
    <circle fill="#1f2937" stroke="#fff" stroke-width="2" cx="-60.62" cy="-35" r="5"/>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="800" fill="#1f2937" paint-order="stroke" stroke="#fdfcf7" stroke-width="4" x="0" y="-94">C</text>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="800" fill="#1f2937" paint-order="stroke" stroke="#fdfcf7" stroke-width="4" x="56" y="-82">G</text>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="800" fill="#1f2937" paint-order="stroke" stroke="#fdfcf7" stroke-width="4" x="-94" y="0">E♭</text>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="800" fill="#1f2937" paint-order="stroke" stroke="#fdfcf7" stroke-width="4" x="-88" y="-56">B♭</text>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="17" font-weight="800" fill="#1f2937" x="0" y="114">C m7</text>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="11" font-weight="600" fill="#6b7280"  x="0" y="132">1 – ♭3 – 5 – ♭7</text>
  </g>

  <!-- Cm7b5: {0, 6, 9, 10} -->
  <g transform="translate(360,760)">
    <circle fill="#f5f2e8" stroke="#bfb8a3" stroke-width="1.2" cx="0" cy="0" r="70"/>
    <circle fill="#d1cfc8" cx="0"     cy="-70" r="3"/>
    <circle fill="#d1cfc8" cx="35"    cy="-60.62" r="3"/>
    <circle fill="#d1cfc8" cx="60.62" cy="-35" r="3"/>
    <circle fill="#d1cfc8" cx="70"    cy="0" r="3"/>
    <circle fill="#d1cfc8" cx="60.62" cy="35" r="3"/>
    <circle fill="#d1cfc8" cx="35"    cy="60.62" r="3"/>
    <circle fill="#d1cfc8" cx="0"     cy="70" r="3"/>
    <circle fill="#d1cfc8" cx="-35"   cy="60.62" r="3"/>
    <circle fill="#d1cfc8" cx="-60.62" cy="35" r="3"/>
    <circle fill="#d1cfc8" cx="-70"   cy="0" r="3"/>
    <circle fill="#d1cfc8" cx="-60.62" cy="-35" r="3"/>
    <circle fill="#d1cfc8" cx="-35"   cy="-60.62" r="3"/>
    <!-- tritone: 0 <-> 6 -->
    <line stroke="#dc2626" stroke-width="2" stroke-dasharray="5 3" x1="0" y1="-70" x2="0" y2="70"/>
    <!-- 0 -> 9 -> 6 -> 10 -->
    <polygon fill="rgba(31,41,55,0.10)" stroke="#1f2937" stroke-width="2.2" points="0,-70 -70,0 0,70 -60.62,-35"/>
    <circle fill="#dc2626" stroke="#fff" stroke-width="2" cx="0" cy="-70" r="5.5"/>
    <circle fill="#1f2937" stroke="#fff" stroke-width="2" cx="-70" cy="0" r="5"/>
    <circle fill="#1f2937" stroke="#fff" stroke-width="2" cx="0" cy="70" r="5"/>
    <circle fill="#1f2937" stroke="#fff" stroke-width="2" cx="-60.62" cy="-35" r="5"/>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="800" fill="#1f2937" paint-order="stroke" stroke="#fdfcf7" stroke-width="4" x="0" y="-94">C</text>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="800" fill="#1f2937" paint-order="stroke" stroke="#fdfcf7" stroke-width="4" x="-94" y="0">E♭</text>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="800" fill="#1f2937" paint-order="stroke" stroke="#fdfcf7" stroke-width="4" x="0" y="94">G♭</text>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="800" fill="#1f2937" paint-order="stroke" stroke="#fdfcf7" stroke-width="4" x="-88" y="-56">B♭</text>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="17" font-weight="800" fill="#1f2937" x="0" y="118">C m7♭5</text>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="11" font-weight="600" fill="#6b7280"  x="0" y="136">1 – ♭3 – ♭5 – ♭7</text>
  </g>

  <!-- Cdim7: {0, 3, 6, 9} - perfect square -->
  <g transform="translate(600,760)">
    <circle fill="#f5f2e8" stroke="#bfb8a3" stroke-width="1.2" cx="0" cy="0" r="70"/>
    <circle fill="#d1cfc8" cx="0"     cy="-70" r="3"/>
    <circle fill="#d1cfc8" cx="35"    cy="-60.62" r="3"/>
    <circle fill="#d1cfc8" cx="60.62" cy="-35" r="3"/>
    <circle fill="#d1cfc8" cx="70"    cy="0" r="3"/>
    <circle fill="#d1cfc8" cx="60.62" cy="35" r="3"/>
    <circle fill="#d1cfc8" cx="35"    cy="60.62" r="3"/>
    <circle fill="#d1cfc8" cx="0"     cy="70" r="3"/>
    <circle fill="#d1cfc8" cx="-35"   cy="60.62" r="3"/>
    <circle fill="#d1cfc8" cx="-60.62" cy="35" r="3"/>
    <circle fill="#d1cfc8" cx="-70"   cy="0" r="3"/>
    <circle fill="#d1cfc8" cx="-60.62" cy="-35" r="3"/>
    <circle fill="#d1cfc8" cx="-35"   cy="-60.62" r="3"/>
    <!-- square 0 -> 3 -> 6 -> 9 -->
    <!-- two crossing tritones: C↔Gb and Eb↔A -->
    <line stroke="#dc2626" stroke-width="2" stroke-dasharray="5 3" x1="0" y1="-70" x2="0" y2="70"/>
    <line stroke="#dc2626" stroke-width="2" stroke-dasharray="5 3" x1="-70" y1="0" x2="70" y2="0"/>
    <polygon fill="rgba(31,41,55,0.10)" stroke="#1f2937" stroke-width="2.2" points="0,-70 70,0 0,70 -70,0"/>
    <circle fill="#dc2626" stroke="#fff" stroke-width="2" cx="0" cy="-70" r="5.5"/>
    <circle fill="#1f2937" stroke="#fff" stroke-width="2" cx="70" cy="0" r="5"/>
    <circle fill="#1f2937" stroke="#fff" stroke-width="2" cx="0" cy="70" r="5"/>
    <circle fill="#1f2937" stroke="#fff" stroke-width="2" cx="-70" cy="0" r="5"/>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="800" fill="#1f2937" paint-order="stroke" stroke="#fdfcf7" stroke-width="4" x="0" y="-94">C</text>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="800" fill="#1f2937" paint-order="stroke" stroke="#fdfcf7" stroke-width="4" x="94" y="0">A</text>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="800" fill="#1f2937" paint-order="stroke" stroke="#fdfcf7" stroke-width="4" x="0" y="94">G♭</text>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="800" fill="#1f2937" paint-order="stroke" stroke="#fdfcf7" stroke-width="4" x="-94" y="0">E♭</text>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="17" font-weight="800" fill="#1f2937" x="0" y="118">C dim7</text>
    <text font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="11" font-weight="600" fill="#6b7280"  x="0" y="136">1 – ♭3 – ♭5 – 𝄫7</text>
  </g>

  <!-- legend -->
  <g transform="translate(60,960)" font-family="ui-sans-serif, system-ui, sans-serif" font-size="12" font-weight="700" fill="#1f2937">
    <circle cx="0"   cy="0" r="6" fill="#dc2626" stroke="#fff" stroke-width="2"/>
    <text   x="14"   y="0" text-anchor="start" dominant-baseline="central">root</text>

    <circle cx="120" cy="0" r="6" fill="#1f2937" stroke="#fff" stroke-width="2"/>
    <text   x="134"  y="0" text-anchor="start" dominant-baseline="central">chord tone</text>

    <line   x1="290" y1="0" x2="330" y2="0" stroke="#dc2626" stroke-width="2.5" stroke-dasharray="5 3"/>
    <text   x="340"  y="0" text-anchor="start" dominant-baseline="central">internal tritone (tension)</text>
  </g>
</svg>

> **Compare the shapes at a glance.** Same root (C) on top of every mini-wheel. Major / minor are the *same* triangle, mirrored across the vertical axis (3rd flips side). Augmented is an equilateral triangle, dim7 is a perfect square — those polygons explain why only 4 distinct aug chords and only 3 distinct dim7 chords exist in all of music. Any shape whose vertices straddle the diameter (red dashed line) contains a **tritone** and therefore *wants to resolve* — that's the defining feature of dominant 7, m7♭5, and the diminished family.

### 4. Tritone substitution — diameter twins resolving to the same place

<svg viewBox="0 0 640 540" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:720px;display:block;margin:0 auto;border-radius:12px">
  <defs>
    <marker id="arrowOrange" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#d97757"/>
    </marker>
    <marker id="arrowBlue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#3b82c4"/>
    </marker>
    <radialGradient id="bgWheel4" cx="0.5" cy="0.5" r="0.7">
      <stop offset="0%" stop-color="#ffffff"/>
      <stop offset="100%" stop-color="#f5f2e8"/>
    </radialGradient>
  </defs>
  <rect width="640" height="540" rx="12" fill="url(#bgWheel4)"/>
  <rect width="640" height="540" rx="12" fill="none" stroke="#d8d2c0" stroke-width="1.5"/>

  <!-- title -->
  <text x="320" y="32" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" font-size="16" font-weight="800" fill="#1f2937">Tritone Substitution: G7 and D♭7 → C</text>

  <!-- wheel centered at (320, 290), r=170 -->
  <circle cx="320" cy="290" r="180" fill="none" stroke="#bfb8a3" stroke-width="1.5"/>
  <circle cx="320" cy="290" r="130" fill="#f5f2e8" stroke="#cfc8b3" stroke-width="1.2"/>

  <!-- 12 base dots: faint sand on cream so positions are visible but recede -->
  <!-- positions (radius 180 from 320,290):
       0  top                C        (320, 110)
       1  1 o'clock          G        (410, 134.1)
       2  2 o'clock          D        (475.9, 200)
       3  3 o'clock          A        (500, 290)
       4  4 o'clock          E        (475.9, 380)
       5  5 o'clock          B        (410, 445.9)
       6  bottom             F#/Gb    (320, 470)
       7  7 o'clock          Db       (230, 445.9)
       8  8 o'clock          Ab       (164.1, 380)
       9  9 o'clock          Eb       (140, 290)
       10 10 o'clock         Bb       (164.1, 200)
       11 11 o'clock         F        (230, 134.1)
  -->
  <g fill="#d1cfc8">
    <circle cx="320"   cy="110"  r="5"/>
    <circle cx="410"   cy="134.1" r="5"/>
    <circle cx="475.9" cy="200"  r="5"/>
    <circle cx="500"   cy="290"  r="5"/>
    <circle cx="475.9" cy="380"  r="5"/>
    <circle cx="410"   cy="445.9" r="5"/>
    <circle cx="320"   cy="470"  r="5"/>
    <circle cx="230"   cy="445.9" r="5"/>
    <circle cx="164.1" cy="380"  r="5"/>
    <circle cx="140"   cy="290"  r="5"/>
    <circle cx="164.1" cy="200"  r="5"/>
    <circle cx="230"   cy="134.1" r="5"/>
  </g>

  <!-- root diameter G <-> Db (gray dashed) -->
  <line x1="410" y1="134.1" x2="230" y2="445.9" stroke="#6b7280" stroke-width="1.5" stroke-dasharray="5 4" opacity="0.7"/>

  <!-- THE shared tritone diameter B <-> F (red, solid, bold) -->
  <line x1="410" y1="445.9" x2="230" y2="134.1" stroke="#dc2626" stroke-width="3"/>

  <!-- G7 chord tones (warm orange, white border)
       G (1), B (5), D (2), F (11) — B and F are shared -->
  <circle cx="410"   cy="134.1" r="12" fill="#d97757" stroke="#fff" stroke-width="2.5"/>
  <circle cx="475.9" cy="200"   r="10" fill="#d97757" stroke="#fff" stroke-width="2"/>
  <!-- Db7 chord tones (cool blue)
       Db (7), F (11), Ab (8), Cb=B (5) — F and B shared -->
  <circle cx="230"   cy="445.9" r="12" fill="#3b82c4" stroke="#fff" stroke-width="2.5"/>
  <circle cx="164.1" cy="380"   r="10" fill="#3b82c4" stroke="#fff" stroke-width="2"/>
  <!-- SHARED B and F: red with white border on top -->
  <circle cx="410" cy="445.9" r="12" fill="#dc2626" stroke="#fff" stroke-width="2.5"/>
  <circle cx="230" cy="134.1" r="12" fill="#dc2626" stroke="#fff" stroke-width="2.5"/>
  <!-- C target on top -->
  <circle cx="320" cy="110" r="14" fill="#1f2937" stroke="#fff" stroke-width="2.5"/>

  <!-- note labels (outside dots) -->
  <g font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="14" font-weight="800" fill="#1f2937">
    <text x="320" y="82">C</text>
    <text x="436" y="116">G</text>
    <text x="504" y="184">D</text>
    <text x="404" y="464">B</text>
    <text x="200" y="464">D♭</text>
    <text x="138" y="396">A♭</text>
    <text x="200" y="116">F</text>
  </g>

  <!-- resolution arrows: from G7 root -> C and from Db7 root -> C -->
  <path d="M 410 134.1 Q 380 95 332 102" fill="none" stroke="#d97757" stroke-width="2.5" marker-end="url(#arrowOrange)"/>
  <path d="M 230 445.9 Q 220 240 308 102" fill="none" stroke="#3b82c4" stroke-width="2.5" marker-end="url(#arrowBlue)"/>

  <!-- right-side label for G7 -->
  <g font-family="ui-sans-serif, system-ui, sans-serif">
    <rect x="528" y="118" width="100" height="76" rx="8" fill="#d97757"/>
    <text x="578" y="138" text-anchor="middle" font-size="16" font-weight="800" fill="#fff">G7</text>
    <text x="578" y="156" text-anchor="middle" font-size="10" font-weight="700" fill="#fff">G · B · D · F</text>
    <text x="578" y="172" text-anchor="middle" font-size="10" font-weight="600" fill="#fff">V of C major</text>
    <text x="578" y="186" text-anchor="middle" font-size="10" font-weight="600" fill="#fff">(orig. dominant)</text>
  </g>

  <!-- left-side label for Db7 -->
  <g font-family="ui-sans-serif, system-ui, sans-serif">
    <rect x="12" y="386" width="100" height="76" rx="8" fill="#3b82c4"/>
    <text x="62" y="406" text-anchor="middle" font-size="16" font-weight="800" fill="#fff">D♭7</text>
    <text x="62" y="424" text-anchor="middle" font-size="10" font-weight="700" fill="#fff">D♭ · F · A♭ · C♭</text>
    <text x="62" y="440" text-anchor="middle" font-size="10" font-weight="600" fill="#fff">tritone sub</text>
    <text x="62" y="454" text-anchor="middle" font-size="10" font-weight="600" fill="#fff">of G7</text>
  </g>

  <!-- center callout: the shared tritone -->
  <g font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central">
    <rect x="240" y="250" width="160" height="80" rx="10" fill="#fff" stroke="#dc2626" stroke-width="1.5"/>
    <text x="320" y="266" font-size="11" font-weight="800" fill="#dc2626" letter-spacing="2">SHARED TRITONE</text>
    <text x="320" y="284" font-size="14" font-weight="800" fill="#dc2626">B  ↔  F</text>
    <text x="320" y="302" font-size="10" font-weight="600" fill="#1f2937">B → C  (leading tone, +1)</text>
    <text x="320" y="316" font-size="10" font-weight="600" fill="#1f2937">F → E  (chord ♭7, −1)</text>
  </g>

  <!-- legend strip -->
  <g font-family="ui-sans-serif, system-ui, sans-serif" font-size="12" font-weight="600" fill="#1f2937">
    <circle cx="50" cy="510" r="6" fill="#d97757" stroke="#fff" stroke-width="1.5"/>
    <text x="60" y="510" text-anchor="start" dominant-baseline="central">G7 tones</text>

    <circle cx="160" cy="510" r="6" fill="#3b82c4" stroke="#fff" stroke-width="1.5"/>
    <text x="170" y="510" text-anchor="start" dominant-baseline="central">D♭7 tones</text>

    <circle cx="290" cy="510" r="6" fill="#dc2626" stroke="#fff" stroke-width="1.5"/>
    <text x="300" y="510" text-anchor="start" dominant-baseline="central">shared (the tritone)</text>

    <line x1="460" y1="510" x2="500" y2="510" stroke="#6b7280" stroke-width="1.5" stroke-dasharray="5 4"/>
    <text x="508" y="510" text-anchor="start" dominant-baseline="central">root diameter</text>
  </g>
</svg>

> **The whole substitution in one picture.** G7 and D♭7 sit at opposite ends of the circle (their roots are a tritone apart — gray dashed diameter). But the *upper* tritone — B and F (red diameter) — belongs to both chords. That shared B–F is what resolves to C: B climbs to C, F drops to E. Either chord delivers the same payload to the tonic, which is why one can stand in for the other.

## How the Loop Works (Intuition First)

### Why fifths form a closed cycle

A "perfect fifth" up means multiplying frequency by 3/2 (or going 7 semitones up on the fretboard — the distance from the open low E string to the 7th-fret B). If you keep stacking 3/2 ratios you'd never return to a perfect octave (that's the *Pythagorean comma*), but in **12-tone equal temperament** — what every fretted guitar uses — each fret is exactly the 12th root of 2, and 7 × 12 ≡ 0 (mod 12), so 12 fifths land back on the start note. That's why the circle has exactly 12 positions.

On the fretboard, "up a fifth" has a memorable shape: same fret, **one string higher** (skipping the B string adjustment). E → A → D → G are all fourths *up* (= fifths *down*), which is why the bass strings of the guitar are tuned in fourths and trace the circle counter-clockwise. This is not coincidence — the tuning *is* a circle-of-fourths slice.

### See it on the neck

<svg viewBox="0 0 720 360" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:760px;display:block;margin:0 auto;border-radius:12px">
  <rect width="720" height="360" rx="12" fill="#fdfcf7"/>
  <rect width="720" height="360" rx="12" fill="none" stroke="#d8d2c0" stroke-width="1.5"/>
  <text x="360" y="28" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" font-size="15" font-weight="800" fill="#1f2937">A fifth = +7 frets · A fourth = same fret, next string up</text>

  <!-- Fretboard, frets 0-12, low E top -->
  <!-- nut at x=70, fret centers every 50 px: 1=110, 2=160, …, 12=620, last line x=645 -->
  <line x1="70" y1="70" x2="70" y2="220" stroke="#1f2937" stroke-width="5"/>
  <g stroke="#9aa0a6" stroke-width="1.5">
    <line x1="135" y1="70" x2="135" y2="220"/>
    <line x1="185" y1="70" x2="185" y2="220"/>
    <line x1="235" y1="70" x2="235" y2="220"/>
    <line x1="285" y1="70" x2="285" y2="220"/>
    <line x1="335" y1="70" x2="335" y2="220"/>
    <line x1="385" y1="70" x2="385" y2="220"/>
    <line x1="435" y1="70" x2="435" y2="220"/>
    <line x1="485" y1="70" x2="485" y2="220"/>
    <line x1="535" y1="70" x2="535" y2="220"/>
    <line x1="585" y1="70" x2="585" y2="220"/>
    <line x1="635" y1="70" x2="635" y2="220"/>
    <line x1="685" y1="70" x2="685" y2="220"/>
  </g>
  <!-- 6 strings -->
  <line x1="70" y1="70"  x2="685" y2="70"  stroke="#1f2937" stroke-width="2.4"/>
  <line x1="70" y1="100" x2="685" y2="100" stroke="#1f2937" stroke-width="2.2"/>
  <line x1="70" y1="130" x2="685" y2="130" stroke="#1f2937" stroke-width="2.0"/>
  <line x1="70" y1="160" x2="685" y2="160" stroke="#1f2937" stroke-width="1.8"/>
  <line x1="70" y1="190" x2="685" y2="190" stroke="#1f2937" stroke-width="1.6"/>
  <line x1="70" y1="220" x2="685" y2="220" stroke="#1f2937" stroke-width="1.4"/>
  <!-- string labels -->
  <g font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="end" dominant-baseline="central" font-size="12" font-weight="800" fill="#1f2937">
    <text x="60" y="70">E</text>
    <text x="60" y="100">A</text>
    <text x="60" y="130">D</text>
    <text x="60" y="160">G</text>
    <text x="60" y="190">B</text>
    <text x="60" y="220">e</text>
  </g>
  <!-- fret-number labels (selected) -->
  <g font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" font-size="10" font-weight="700" fill="#6b7280">
    <text x="110" y="240">1</text><text x="160" y="240">2</text>
    <text x="210" y="240">3</text><text x="260" y="240">4</text>
    <text x="310" y="240">5</text><text x="360" y="240">6</text>
    <text x="410" y="240">7</text><text x="460" y="240">8</text>
    <text x="510" y="240">9</text><text x="560" y="240">10</text>
    <text x="610" y="240">11</text><text x="660" y="240">12</text>
  </g>

  <!-- ============================================================
       PART 1 (warm orange):  +7 frets on the low E string
       open E  →  fret 7 B  = a perfect FIFTH
       ============================================================ -->
  <!-- open E -->
  <circle cx="50" cy="70" r="12" fill="#d97757" stroke="#fff" stroke-width="2"/>
  <text x="50" y="70" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="10" font-weight="800" fill="#fff">E</text>
  <!-- B at fret 7 -->
  <circle cx="410" cy="70" r="12" fill="#d97757" stroke="#fff" stroke-width="2"/>
  <text x="410" y="70" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="10" font-weight="800" fill="#fff">B</text>
  <!-- arrow over the low-E string -->
  <path d="M 65 56 Q 240 38 396 56" fill="none" stroke="#d97757" stroke-width="2.2" marker-end="url(#a5th)" opacity="0.9"/>
  <!-- caption tucked just under the arrow, above the fretboard -->
  <rect x="135" y="42" width="220" height="18" rx="9" fill="#fdfcf7"/>
  <text x="245" y="52" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="11" font-weight="800" fill="#d97757">+7 frets = perfect 5th  (1 step CW)</text>

  <!-- ============================================================
       PART 2 (cool blue): same fret 0, one string up = a fourth
       E (string 6 open)  → A (string 5 open) → D (string 4 open) → G (string 3 open)
       These are the bass strings tuned in 4ths (= CCW around the circle)
       ============================================================ -->
  <!-- emphasize open strings 5,4,3 -->
  <circle cx="50" cy="100" r="12" fill="#3b82c4" stroke="#fff" stroke-width="2"/>
  <text x="50" y="100" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="10" font-weight="800" fill="#fff">A</text>
  <circle cx="50" cy="130" r="12" fill="#3b82c4" stroke="#fff" stroke-width="2"/>
  <text x="50" y="130" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="10" font-weight="800" fill="#fff">D</text>
  <circle cx="50" cy="160" r="12" fill="#3b82c4" stroke="#fff" stroke-width="2"/>
  <text x="50" y="160" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="10" font-weight="800" fill="#fff">G</text>

  <!-- arrows down the strings -->
  <path d="M 50 84 L 50 88" stroke="#3b82c4" stroke-width="2"/>
  <path d="M 50 92 L 50 114" stroke="#3b82c4" stroke-width="2" marker-end="url(#a4th)"/>
  <path d="M 50 118 L 50 144" stroke="#3b82c4" stroke-width="2" marker-end="url(#a4th)"/>
  <path d="M 50 148 L 50 174" stroke="#3b82c4" stroke-width="2" marker-end="url(#a4th)"/>

  <text x="22" y="115" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" font-size="10" font-weight="800" fill="#3b82c4">+5 st</text>
  <text x="22" y="145" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" font-size="10" font-weight="800" fill="#3b82c4">+5 st</text>
  <text x="22" y="175" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" font-size="10" font-weight="800" fill="#3b82c4">+5 st</text>

  <!-- right-side callouts -->
  <g font-family="ui-sans-serif, system-ui, sans-serif">
    <rect x="470" y="100" width="230" height="120" rx="8" fill="#3b82c4"/>
    <text x="585" y="120" text-anchor="middle" font-size="13" font-weight="800" fill="#fff">Bass strings = circle of 4ths</text>
    <text x="585" y="140" text-anchor="middle" font-size="11" font-weight="700" fill="#fff">E → A → D → G  (each +5 st)</text>
    <text x="585" y="160" text-anchor="middle" font-size="11" font-weight="600" fill="#fff">Same direction as CCW on circle.</text>
    <text x="585" y="184" text-anchor="middle" font-size="11" font-weight="700" fill="#fff">A 4th up = a 5th down.</text>
    <text x="585" y="204" text-anchor="middle" font-size="11" font-weight="700" fill="#fff">The tuning IS the circle.</text>
  </g>

  <!-- B / e strings note -->
  <text x="585" y="245" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" font-size="10" font-weight="600" fill="#6b7280">(B string breaks the pattern — it's a 3rd up from G, not a 4th)</text>

  <defs>
    <marker id="a5th" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#d97757"/>
    </marker>
    <marker id="a4th" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#3b82c4"/>
    </marker>
  </defs>

  <!-- bottom legend -->
  <g transform="translate(70,290)" font-family="ui-sans-serif, system-ui, sans-serif" font-size="11" font-weight="700" fill="#1f2937">
    <circle cx="0"   cy="0" r="6" fill="#d97757" stroke="#fff" stroke-width="2"/>
    <text   x="14"   y="0" text-anchor="start" dominant-baseline="central">perfect 5th = +7 semitones (= one CW step on the circle)</text>
    <circle cx="0"   cy="22" r="6" fill="#3b82c4" stroke="#fff" stroke-width="2"/>
    <text   x="14"   y="22" text-anchor="start" dominant-baseline="central">perfect 4th = +5 semitones = same fret, next string up (= one CCW step)</text>
  </g>
</svg>

> **What to look at.** The orange dots span the low E string from open to fret 7 — that's the literal **+7 semitones = perfect fifth** the math demands. The blue dots are the *open* A, D, and G strings: each one is a perfect fourth above its neighbor, which is the *same thing* as a fifth below (one CCW step on the circle). The bass half of your guitar is geometrically a slice of the circle of fifths — that's why fingering patterns translate so naturally between adjacent strings.

### Why clockwise = sharps, counter-clockwise = flats

Each step clockwise raises one note of the previous key by a semitone — specifically, the 7th degree, which becomes the new key's leading tone. Starting from C major (all naturals):

| Key | Sharps added | Order |
|-----|--------------|-------|
| C   | —            | none |
| G   | F#           | 1 |
| D   | F# C#        | 2 |
| A   | F# C# G#     | 3 |
| E   | F# C# G# D#  | 4 |
| B   | + A#         | 5 |
| F#  | + E#         | 6 |

**The mnemonic: sharps appear in the order F# C# G# D# A# E# B# — which is itself fifths going up.** Flats reverse it: Bb Eb Ab Db Gb Cb Fb — fifths going down (or fourths going up).

Sharp-key trick: the last sharp is always one semitone below the tonic. Three sharps (F# C# G#) → last sharp is G# → tonic is A.
Flat-key trick: the second-to-last flat *is* the tonic. Three flats (Bb Eb Ab) → second-to-last is Eb → key is Eb.

## Diatonic Chords from the Circle

In every major key, the seven diatonic chords follow a fixed quality pattern:

```
I    ii    iii   IV    V    vi   vii°
maj  min   min   maj   maj  min  dim
```

The circle gives you the three majors (I, IV, V) **at a glance**: the I is your starting position, IV is one step *counter-clockwise*, V is one step *clockwise*. The relative minor (vi) is directly inside the wheel.

Example — key of G:
- One CCW from G → **C** = IV
- One CW from G → **D** = V
- Inner ring at G → **Em** = vi
- ii is the inner ring at IV → **Am**
- iii is the inner ring at V → **Bm**
- vii° is between V and the inner ring at IV → **F#dim**

So G major's diatonic chords are: **G, Am, Bm, C, D, Em, F#dim**. That's "Wonderwall," "Sweet Home Alabama," "Let It Be," and about 80% of campfire songs.

### What "in the key of G" actually looks like on the neck

The six main chords (I, ii, iii, IV, V, vi) drawn as open-position chord boxes. Every dot is one of the seven notes of **G major: G A B C D E F♯**. No accidentals appear anywhere — that's literally what "diatonic to G" means.

<svg viewBox="0 0 720 480" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:760px;display:block;margin:0 auto;border-radius:12px">
  <rect width="720" height="480" rx="12" fill="#fdfcf7"/>
  <rect width="720" height="480" rx="12" fill="none" stroke="#d8d2c0" stroke-width="1.5"/>
  <text x="360" y="28" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" font-size="15" font-weight="800" fill="#1f2937">Diatonic chords of G major — every note is G A B C D E F♯</text>

  <!--
    Chord-box layout: 2 rows × 3 cols.  Each box ≈ 200×190 starting at:
      Col x = 40, 250, 480
      Row y = 60, 270
    Inside each box: nut (left thick bar), 3 frets, 6 strings.
    Box internal coords:
      nut at x_box+50, fret 1 = x_box+95, fret 2 = x_box+140, fret 3 = x_box+185 (centers)
      vertical lines at x_box+50, x_box+72, x_box+117, x_box+162, x_box+207  (5 frets including nut)
      strings at y_box+30..y_box+105 step 15
      string labels at left
    Open-string circles (above the nut) at x_box+38

    Color: I/IV/V = warm orange, ii/iii/vi = cool blue (matches the circle's outer/inner ring)
  -->

  <!-- Common rendering helpers via reusable template? SVG has no for-loops, write inline. -->

  <!-- ============================ G  (I) — open: 320003 ============================ -->
  <g transform="translate(40,60)">
    <!-- frame -->
    <line x1="50" y1="30" x2="50" y2="105" stroke="#1f2937" stroke-width="4"/>
    <line x1="95" y1="30" x2="95" y2="105" stroke="#9aa0a6" stroke-width="1.6"/>
    <line x1="140" y1="30" x2="140" y2="105" stroke="#9aa0a6" stroke-width="1.6"/>
    <line x1="185" y1="30" x2="185" y2="105" stroke="#9aa0a6" stroke-width="1.6"/>
    <line x1="50" y1="30"  x2="185" y2="30"  stroke="#1f2937" stroke-width="1.4"/>
    <line x1="50" y1="45"  x2="185" y2="45"  stroke="#1f2937" stroke-width="1.4"/>
    <line x1="50" y1="60"  x2="185" y2="60"  stroke="#1f2937" stroke-width="1.4"/>
    <line x1="50" y1="75"  x2="185" y2="75"  stroke="#1f2937" stroke-width="1.4"/>
    <line x1="50" y1="90"  x2="185" y2="90"  stroke="#1f2937" stroke-width="1.4"/>
    <line x1="50" y1="105" x2="185" y2="105" stroke="#1f2937" stroke-width="1.4"/>
    <g font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="end" dominant-baseline="central" font-size="9" font-weight="700" fill="#9aa0a6">
      <text x="44" y="30">E</text><text x="44" y="45">A</text><text x="44" y="60">D</text>
      <text x="44" y="75">G</text><text x="44" y="90">B</text><text x="44" y="105">e</text>
    </g>
    <!-- G(3-2-0-0-0-3): E3(G), A2(B), D0, G0, B0, e3(G) -->
    <circle cx="162" cy="30" r="9" fill="#d97757" stroke="#fff" stroke-width="2"/>
    <text x="162" y="30" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="9" font-weight="800" fill="#fff">G</text>
    <circle cx="117" cy="45" r="9" fill="#d97757" stroke="#fff" stroke-width="2"/>
    <text x="117" y="45" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="9" font-weight="800" fill="#fff">B</text>
    <circle cx="35" cy="60" r="8" fill="none" stroke="#d97757" stroke-width="2"/>
    <text x="35" y="60" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="8" font-weight="800" fill="#d97757">D</text>
    <circle cx="35" cy="75" r="8" fill="none" stroke="#d97757" stroke-width="2"/>
    <text x="35" y="75" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="8" font-weight="800" fill="#d97757">G</text>
    <circle cx="35" cy="90" r="8" fill="none" stroke="#d97757" stroke-width="2"/>
    <text x="35" y="90" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="8" font-weight="800" fill="#d97757">B</text>
    <circle cx="162" cy="105" r="9" fill="#d97757" stroke="#fff" stroke-width="2"/>
    <text x="162" y="105" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="9" font-weight="800" fill="#fff">G</text>
    <!-- label -->
    <rect x="20" y="130" width="195" height="40" rx="6" fill="#d97757"/>
    <text x="117" y="148" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" font-size="14" font-weight="800" fill="#fff">G    (I)</text>
    <text x="117" y="164" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" font-size="9" font-weight="700" fill="#fff">G B D · tonic</text>
  </g>

  <!-- ============================ C  (IV) — open: x32010 ============================ -->
  <g transform="translate(250,60)">
    <line x1="50" y1="30" x2="50" y2="105" stroke="#1f2937" stroke-width="4"/>
    <line x1="95" y1="30" x2="95" y2="105" stroke="#9aa0a6" stroke-width="1.6"/>
    <line x1="140" y1="30" x2="140" y2="105" stroke="#9aa0a6" stroke-width="1.6"/>
    <line x1="185" y1="30" x2="185" y2="105" stroke="#9aa0a6" stroke-width="1.6"/>
    <line x1="50" y1="30"  x2="185" y2="30"  stroke="#1f2937" stroke-width="1.4"/>
    <line x1="50" y1="45"  x2="185" y2="45"  stroke="#1f2937" stroke-width="1.4"/>
    <line x1="50" y1="60"  x2="185" y2="60"  stroke="#1f2937" stroke-width="1.4"/>
    <line x1="50" y1="75"  x2="185" y2="75"  stroke="#1f2937" stroke-width="1.4"/>
    <line x1="50" y1="90"  x2="185" y2="90"  stroke="#1f2937" stroke-width="1.4"/>
    <line x1="50" y1="105" x2="185" y2="105" stroke="#1f2937" stroke-width="1.4"/>
    <g font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="end" dominant-baseline="central" font-size="9" font-weight="700" fill="#9aa0a6">
      <text x="44" y="30">E</text><text x="44" y="45">A</text><text x="44" y="60">D</text>
      <text x="44" y="75">G</text><text x="44" y="90">B</text><text x="44" y="105">e</text>
    </g>
    <!-- C(x-3-2-0-1-0): A3(C), D2(E), G0(G), B1(C), e0(E) -->
    <text x="35" y="30" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="9" font-weight="800" fill="#9aa0a6">×</text>
    <circle cx="162" cy="45" r="9" fill="#d97757" stroke="#fff" stroke-width="2"/>
    <text x="162" y="45" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="9" font-weight="800" fill="#fff">C</text>
    <circle cx="117" cy="60" r="9" fill="#d97757" stroke="#fff" stroke-width="2"/>
    <text x="117" y="60" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="9" font-weight="800" fill="#fff">E</text>
    <circle cx="35" cy="75" r="8" fill="none" stroke="#d97757" stroke-width="2"/>
    <text x="35" y="75" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="8" font-weight="800" fill="#d97757">G</text>
    <circle cx="72" cy="90" r="9" fill="#d97757" stroke="#fff" stroke-width="2"/>
    <text x="72" y="90" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="9" font-weight="800" fill="#fff">C</text>
    <circle cx="35" cy="105" r="8" fill="none" stroke="#d97757" stroke-width="2"/>
    <text x="35" y="105" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="8" font-weight="800" fill="#d97757">E</text>
    <rect x="20" y="130" width="195" height="40" rx="6" fill="#d97757"/>
    <text x="117" y="148" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" font-size="14" font-weight="800" fill="#fff">C    (IV)</text>
    <text x="117" y="164" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" font-size="9" font-weight="700" fill="#fff">C E G · subdominant</text>
  </g>

  <!-- ============================ D  (V) — open: xx0232 ============================ -->
  <g transform="translate(480,60)">
    <line x1="50" y1="30" x2="50" y2="105" stroke="#1f2937" stroke-width="4"/>
    <line x1="95" y1="30" x2="95" y2="105" stroke="#9aa0a6" stroke-width="1.6"/>
    <line x1="140" y1="30" x2="140" y2="105" stroke="#9aa0a6" stroke-width="1.6"/>
    <line x1="185" y1="30" x2="185" y2="105" stroke="#9aa0a6" stroke-width="1.6"/>
    <line x1="50" y1="30"  x2="185" y2="30"  stroke="#1f2937" stroke-width="1.4"/>
    <line x1="50" y1="45"  x2="185" y2="45"  stroke="#1f2937" stroke-width="1.4"/>
    <line x1="50" y1="60"  x2="185" y2="60"  stroke="#1f2937" stroke-width="1.4"/>
    <line x1="50" y1="75"  x2="185" y2="75"  stroke="#1f2937" stroke-width="1.4"/>
    <line x1="50" y1="90"  x2="185" y2="90"  stroke="#1f2937" stroke-width="1.4"/>
    <line x1="50" y1="105" x2="185" y2="105" stroke="#1f2937" stroke-width="1.4"/>
    <g font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="end" dominant-baseline="central" font-size="9" font-weight="700" fill="#9aa0a6">
      <text x="44" y="30">E</text><text x="44" y="45">A</text><text x="44" y="60">D</text>
      <text x="44" y="75">G</text><text x="44" y="90">B</text><text x="44" y="105">e</text>
    </g>
    <!-- D(x-x-0-2-3-2): D0, G2(A), B3(D), e2(F#) -->
    <text x="35" y="30" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="9" font-weight="800" fill="#9aa0a6">×</text>
    <text x="35" y="45" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="9" font-weight="800" fill="#9aa0a6">×</text>
    <circle cx="35" cy="60" r="8" fill="none" stroke="#d97757" stroke-width="2"/>
    <text x="35" y="60" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="8" font-weight="800" fill="#d97757">D</text>
    <circle cx="117" cy="75" r="9" fill="#d97757" stroke="#fff" stroke-width="2"/>
    <text x="117" y="75" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="9" font-weight="800" fill="#fff">A</text>
    <circle cx="162" cy="90" r="9" fill="#d97757" stroke="#fff" stroke-width="2"/>
    <text x="162" y="90" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="9" font-weight="800" fill="#fff">D</text>
    <circle cx="117" cy="105" r="9" fill="#d97757" stroke="#fff" stroke-width="2"/>
    <text x="117" y="105" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="9" font-weight="800" fill="#fff">F♯</text>
    <rect x="20" y="130" width="195" height="40" rx="6" fill="#d97757"/>
    <text x="117" y="148" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" font-size="14" font-weight="800" fill="#fff">D    (V)</text>
    <text x="117" y="164" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" font-size="9" font-weight="700" fill="#fff">D F♯ A · dominant</text>
  </g>

  <!-- ============================ Em  (vi) — open: 022000 ============================ -->
  <g transform="translate(40,270)">
    <line x1="50" y1="30" x2="50" y2="105" stroke="#1f2937" stroke-width="4"/>
    <line x1="95" y1="30" x2="95" y2="105" stroke="#9aa0a6" stroke-width="1.6"/>
    <line x1="140" y1="30" x2="140" y2="105" stroke="#9aa0a6" stroke-width="1.6"/>
    <line x1="185" y1="30" x2="185" y2="105" stroke="#9aa0a6" stroke-width="1.6"/>
    <line x1="50" y1="30"  x2="185" y2="30"  stroke="#1f2937" stroke-width="1.4"/>
    <line x1="50" y1="45"  x2="185" y2="45"  stroke="#1f2937" stroke-width="1.4"/>
    <line x1="50" y1="60"  x2="185" y2="60"  stroke="#1f2937" stroke-width="1.4"/>
    <line x1="50" y1="75"  x2="185" y2="75"  stroke="#1f2937" stroke-width="1.4"/>
    <line x1="50" y1="90"  x2="185" y2="90"  stroke="#1f2937" stroke-width="1.4"/>
    <line x1="50" y1="105" x2="185" y2="105" stroke="#1f2937" stroke-width="1.4"/>
    <g font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="end" dominant-baseline="central" font-size="9" font-weight="700" fill="#9aa0a6">
      <text x="44" y="30">E</text><text x="44" y="45">A</text><text x="44" y="60">D</text>
      <text x="44" y="75">G</text><text x="44" y="90">B</text><text x="44" y="105">e</text>
    </g>
    <!-- Em(0-2-2-0-0-0): E0, A2(B), D2(E), G0, B0, e0 -->
    <circle cx="35" cy="30" r="8" fill="none" stroke="#3b82c4" stroke-width="2"/>
    <text x="35" y="30" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="8" font-weight="800" fill="#3b82c4">E</text>
    <circle cx="117" cy="45" r="9" fill="#3b82c4" stroke="#fff" stroke-width="2"/>
    <text x="117" y="45" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="9" font-weight="800" fill="#fff">B</text>
    <circle cx="117" cy="60" r="9" fill="#3b82c4" stroke="#fff" stroke-width="2"/>
    <text x="117" y="60" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="9" font-weight="800" fill="#fff">E</text>
    <circle cx="35" cy="75" r="8" fill="none" stroke="#3b82c4" stroke-width="2"/>
    <text x="35" y="75" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="8" font-weight="800" fill="#3b82c4">G</text>
    <circle cx="35" cy="90" r="8" fill="none" stroke="#3b82c4" stroke-width="2"/>
    <text x="35" y="90" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="8" font-weight="800" fill="#3b82c4">B</text>
    <circle cx="35" cy="105" r="8" fill="none" stroke="#3b82c4" stroke-width="2"/>
    <text x="35" y="105" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="8" font-weight="800" fill="#3b82c4">E</text>
    <rect x="20" y="130" width="195" height="40" rx="6" fill="#3b82c4"/>
    <text x="117" y="148" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" font-size="14" font-weight="800" fill="#fff">Em    (vi)</text>
    <text x="117" y="164" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" font-size="9" font-weight="700" fill="#fff">E G B · relative minor</text>
  </g>

  <!-- ============================ Am  (ii) — open: x02210 ============================ -->
  <g transform="translate(250,270)">
    <line x1="50" y1="30" x2="50" y2="105" stroke="#1f2937" stroke-width="4"/>
    <line x1="95" y1="30" x2="95" y2="105" stroke="#9aa0a6" stroke-width="1.6"/>
    <line x1="140" y1="30" x2="140" y2="105" stroke="#9aa0a6" stroke-width="1.6"/>
    <line x1="185" y1="30" x2="185" y2="105" stroke="#9aa0a6" stroke-width="1.6"/>
    <line x1="50" y1="30"  x2="185" y2="30"  stroke="#1f2937" stroke-width="1.4"/>
    <line x1="50" y1="45"  x2="185" y2="45"  stroke="#1f2937" stroke-width="1.4"/>
    <line x1="50" y1="60"  x2="185" y2="60"  stroke="#1f2937" stroke-width="1.4"/>
    <line x1="50" y1="75"  x2="185" y2="75"  stroke="#1f2937" stroke-width="1.4"/>
    <line x1="50" y1="90"  x2="185" y2="90"  stroke="#1f2937" stroke-width="1.4"/>
    <line x1="50" y1="105" x2="185" y2="105" stroke="#1f2937" stroke-width="1.4"/>
    <g font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="end" dominant-baseline="central" font-size="9" font-weight="700" fill="#9aa0a6">
      <text x="44" y="30">E</text><text x="44" y="45">A</text><text x="44" y="60">D</text>
      <text x="44" y="75">G</text><text x="44" y="90">B</text><text x="44" y="105">e</text>
    </g>
    <!-- Am(x-0-2-2-1-0): A0, D2(E), G2(A), B1(C), e0(E) -->
    <text x="35" y="30" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="9" font-weight="800" fill="#9aa0a6">×</text>
    <circle cx="35" cy="45" r="8" fill="none" stroke="#3b82c4" stroke-width="2"/>
    <text x="35" y="45" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="8" font-weight="800" fill="#3b82c4">A</text>
    <circle cx="117" cy="60" r="9" fill="#3b82c4" stroke="#fff" stroke-width="2"/>
    <text x="117" y="60" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="9" font-weight="800" fill="#fff">E</text>
    <circle cx="117" cy="75" r="9" fill="#3b82c4" stroke="#fff" stroke-width="2"/>
    <text x="117" y="75" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="9" font-weight="800" fill="#fff">A</text>
    <circle cx="72" cy="90" r="9" fill="#3b82c4" stroke="#fff" stroke-width="2"/>
    <text x="72" y="90" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="9" font-weight="800" fill="#fff">C</text>
    <circle cx="35" cy="105" r="8" fill="none" stroke="#3b82c4" stroke-width="2"/>
    <text x="35" y="105" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="8" font-weight="800" fill="#3b82c4">E</text>
    <rect x="20" y="130" width="195" height="40" rx="6" fill="#3b82c4"/>
    <text x="117" y="148" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" font-size="14" font-weight="800" fill="#fff">Am    (ii)</text>
    <text x="117" y="164" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" font-size="9" font-weight="700" fill="#fff">A C E · supertonic</text>
  </g>

  <!-- ============================ Bm  (iii) — barre: x24432 ============================ -->
  <g transform="translate(480,270)">
    <line x1="50" y1="30" x2="50" y2="105" stroke="#1f2937" stroke-width="4"/>
    <line x1="95" y1="30" x2="95" y2="105" stroke="#9aa0a6" stroke-width="1.6"/>
    <line x1="140" y1="30" x2="140" y2="105" stroke="#9aa0a6" stroke-width="1.6"/>
    <line x1="185" y1="30" x2="185" y2="105" stroke="#9aa0a6" stroke-width="1.6"/>
    <line x1="50" y1="30"  x2="185" y2="30"  stroke="#1f2937" stroke-width="1.4"/>
    <line x1="50" y1="45"  x2="185" y2="45"  stroke="#1f2937" stroke-width="1.4"/>
    <line x1="50" y1="60"  x2="185" y2="60"  stroke="#1f2937" stroke-width="1.4"/>
    <line x1="50" y1="75"  x2="185" y2="75"  stroke="#1f2937" stroke-width="1.4"/>
    <line x1="50" y1="90"  x2="185" y2="90"  stroke="#1f2937" stroke-width="1.4"/>
    <line x1="50" y1="105" x2="185" y2="105" stroke="#1f2937" stroke-width="1.4"/>
    <g font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="end" dominant-baseline="central" font-size="9" font-weight="700" fill="#9aa0a6">
      <text x="44" y="30">E</text><text x="44" y="45">A</text><text x="44" y="60">D</text>
      <text x="44" y="75">G</text><text x="44" y="90">B</text><text x="44" y="105">e</text>
    </g>
    <!-- Bm(x-2-4-4-3-2): A2(B), D4(F#), G4(B), B3(D), e2(F#) -->
    <text x="35" y="30" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="9" font-weight="800" fill="#9aa0a6">×</text>
    <circle cx="72" cy="45" r="9" fill="#3b82c4" stroke="#fff" stroke-width="2"/>
    <text x="72" y="45" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="9" font-weight="800" fill="#fff">B</text>
    <circle cx="162" cy="60" r="9" fill="#3b82c4" stroke="#fff" stroke-width="2"/>
    <text x="162" y="60" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="9" font-weight="800" fill="#fff">F♯</text>
    <circle cx="162" cy="75" r="9" fill="#3b82c4" stroke="#fff" stroke-width="2"/>
    <text x="162" y="75" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="9" font-weight="800" fill="#fff">B</text>
    <circle cx="117" cy="90" r="9" fill="#3b82c4" stroke="#fff" stroke-width="2"/>
    <text x="117" y="90" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="9" font-weight="800" fill="#fff">D</text>
    <circle cx="72" cy="105" r="9" fill="#3b82c4" stroke="#fff" stroke-width="2"/>
    <text x="72" y="105" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="9" font-weight="800" fill="#fff">F♯</text>
    <text x="225" y="45" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="end" dominant-baseline="central" font-size="9" font-weight="700" fill="#6b7280">fr 2</text>
    <rect x="20" y="130" width="195" height="40" rx="6" fill="#3b82c4"/>
    <text x="117" y="148" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" font-size="14" font-weight="800" fill="#fff">Bm    (iii)</text>
    <text x="117" y="164" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" font-size="9" font-weight="700" fill="#fff">B D F♯ · mediant</text>
  </g>

  <!-- Legend / takeaway -->
  <g transform="translate(360,460)" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" font-size="11" font-weight="700" fill="#6b7280">
    <text x="0" y="0">All seven notes used (G A B C D E F♯) = the G major scale.  No accidentals appear anywhere.</text>
  </g>
</svg>

> **What to look at.** Every dot on every fretboard is one of *seven* notes — G, A, B, C, D, E, F♯. No sharps or flats outside that set show up, which is the literal meaning of "diatonic to G." Warm orange = the I–IV–V majors (the three CW-adjacent positions on the circle); cool blue = the ii–iii–vi minors (their relative-minor partners). Switch any two of these chords in any order and you've written a song in G.

### Common progressions, decoded

| Roman numerals | In G | In C | What it does |
|---|---|---|---|
| I–V–vi–IV | G–D–Em–C | C–G–Am–F | The "pop punk" / "Don't Stop Believin'" loop |
| I–IV–V | G–C–D | C–F–G | 12-bar blues, three-chord rock |
| ii–V–I | Am–D–G | Dm–G–C | The jazz cadence; strongest resolution |
| vi–IV–I–V | Em–C–G–D | Am–F–C–G | Same chords as I–V–vi–IV, starts on the minor |
| I–vi–IV–V | G–Em–C–D | C–Am–F–G | '50s doo-wop |

Every one of these is **adjacent positions on the circle**. That's the magic: smooth voice leading lives in neighboring slots.

## Chord Construction on the Circle

The previous section showed *which chords* live in a key. This one zooms in: how the *notes inside each chord* sit on the circle. Same map, different scale.

### The interval map (every interval = a circle distance)

Each CW step on the circle = 7 semitones = one perfect fifth up. Mod 12, that means every interval from the root is just "N steps clockwise":

| Interval up from root | Semitones | Steps on circle | Example from C |
|---|---|---|---|
| Perfect 5th (P5) | 7  | **+1 CW** | G |
| Perfect 4th (P4) | 5  | **−1 CCW** (= +11) | F |
| Major 2nd (M2)   | 2  | +2 CW | D |
| Major 3rd (M3)   | 4  | **+4 CW** | E |
| Minor 3rd (m3)   | 3  | **−3 CCW** | Eb |
| Major 7th (M7)   | 11 | **+5 CW** | B |
| Minor 7th (b7)   | 10 | **−2 CCW** | Bb |
| Tritone (b5/#4)  | 6  | **+6 = −6** (opposite) | F# / Gb |
| Major 6th (M6)   | 9  | +3 CW | A |
| Minor 6th (m6)   | 8  | −4 CCW | Ab |

Pattern: **major-flavored intervals fall short-CW; minor-flavored intervals fall short-CCW; the tritone sits exactly across the diameter.**

> **Quick check on the intuition that started this section:** "Go back from C = the 4th (F); go forward = the 5th (G)." That's correct as a *single-interval* claim and gives you the I–IV–V *family* (three chord roots) in a key. But a single major *chord* is 1–3–5, not 1–4–5 — to actually build C major you also need E, which is **+4 CW**, not the immediate neighbor. Chord-roots-in-a-key and chord-tones-in-a-chord are two different uses of the circle.

### Triads

Each triad = three positions on the circle. Plot them; the family resemblance pops out.

**Major (1 — 3 — 5)** → positions **{0, +1 CW, +4 CW}**
- C major = **C, G, E**.
- A "fan" opening CW: the 5th sits next-door; the 3rd jumps four CW.

**Minor (1 — b3 — 5)** → positions **{0, +1 CW, −3 CCW}**
- C minor = **C, G, Eb**.
- Same as major, **but the 3rd flips to the CCW side.** That single mirror across the root–fifth axis *is* the major/minor distinction.

On the fretboard, that "3rd flip" is one finger moving by one fret. Watch:

<svg viewBox="0 0 720 320" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:760px;display:block;margin:0 auto;border-radius:12px">
  <rect width="720" height="320" rx="12" fill="#fdfcf7"/>
  <rect width="720" height="320" rx="12" fill="none" stroke="#d8d2c0" stroke-width="1.5"/>
  <text x="360" y="28" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" font-size="15" font-weight="800" fill="#1f2937">Major ↔ Minor: the 3rd flips by one fret</text>

  <!-- Two compact fretboards, 5-fret span; positions: A-shape barre at fret 3 = C / Cm -->
  <!-- nut shown as fret 0 of A-barre (just visual); barre at fret 3.  Frets 1..5, x=110,180,250,320,390;
       nut at x=80, last vertical at x=430 -->

  <!-- == C MAJOR (A-shape, root on A string fret 3) ==
       Voicing: x 3 5 5 5 3
         A:3 (C, root)
         D:5 (G, 5th)
         G:5 (C, root)
         B:5 (E, 3rd)   <-- THIS is the 3rd that flips
         e:3 (G)
  -->
  <g transform="translate(0,0)">
    <line x1="80" y1="80" x2="80" y2="200" stroke="#1f2937" stroke-width="5"/>
    <line x1="150" y1="80" x2="150" y2="200" stroke="#9aa0a6" stroke-width="2"/>
    <line x1="220" y1="80" x2="220" y2="200" stroke="#9aa0a6" stroke-width="2"/>
    <line x1="290" y1="80" x2="290" y2="200" stroke="#9aa0a6" stroke-width="2"/>
    <line x1="360" y1="80" x2="360" y2="200" stroke="#9aa0a6" stroke-width="2"/>
    <line x1="80" y1="80"  x2="360" y2="80"  stroke="#9aa0a6" stroke-width="1.6" stroke-dasharray="4 3" opacity="0.45"/>
    <line x1="80" y1="104" x2="360" y2="104" stroke="#1f2937" stroke-width="2"/>
    <line x1="80" y1="128" x2="360" y2="128" stroke="#1f2937" stroke-width="2"/>
    <line x1="80" y1="152" x2="360" y2="152" stroke="#1f2937" stroke-width="1.9"/>
    <line x1="80" y1="176" x2="360" y2="176" stroke="#1f2937" stroke-width="1.7"/>
    <line x1="80" y1="200" x2="360" y2="200" stroke="#1f2937" stroke-width="1.5"/>
    <g font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="end" dominant-baseline="central" font-size="11" font-weight="800" fill="#9aa0a6">
      <text x="70" y="80">E×</text>
    </g>
    <g font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="end" dominant-baseline="central" font-size="12" font-weight="800" fill="#1f2937">
      <text x="70" y="104">A</text>
      <text x="70" y="128">D</text>
      <text x="70" y="152">G</text>
      <text x="70" y="176">B</text>
      <text x="70" y="200">e</text>
    </g>
    <g font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" font-size="10" font-weight="700" fill="#6b7280">
      <text x="115" y="218">3</text><text x="185" y="218">4</text><text x="255" y="218">5</text>
    </g>

    <!-- C (root) at A:3 -->
    <circle cx="115" cy="104" r="12" fill="#d97757" stroke="#fff" stroke-width="2"/>
    <text x="115" y="104" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="10" font-weight="800" fill="#fff">C</text>
    <!-- G (5th) at D:5 -->
    <circle cx="255" cy="128" r="12" fill="#d97757" stroke="#fff" stroke-width="2"/>
    <text x="255" y="128" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="10" font-weight="800" fill="#fff">G</text>
    <!-- C (root oct) at G:5 -->
    <circle cx="255" cy="152" r="12" fill="#d97757" stroke="#fff" stroke-width="2"/>
    <text x="255" y="152" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="10" font-weight="800" fill="#fff">C</text>
    <!-- E (3rd) at B:5 -- THE NOTE THAT FLIPS -->
    <circle cx="255" cy="176" r="14" fill="#dc2626" stroke="#fff" stroke-width="2.5"/>
    <text x="255" y="176" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="11" font-weight="800" fill="#fff">E</text>
    <!-- G (5th oct) at e:3 -->
    <circle cx="115" cy="200" r="12" fill="#d97757" stroke="#fff" stroke-width="2"/>
    <text x="115" y="200" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="10" font-weight="800" fill="#fff">G</text>

    <!-- label -->
    <rect x="80" y="240" width="280" height="46" rx="8" fill="#d97757"/>
    <text x="220" y="260" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" font-size="15" font-weight="800" fill="#fff">C major   (x 3 5 5 5 3)</text>
    <text x="220" y="278" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" font-size="10" font-weight="700" fill="#fff">3rd (E) on B-string, fret 5</text>
  </g>

  <!-- == C MINOR (A-shape minor, x 3 5 5 4 3) ==
       Same voicing EXCEPT B string drops to fret 4 (E♭ instead of E)
  -->
  <g transform="translate(360,0)">
    <line x1="80" y1="80" x2="80" y2="200" stroke="#1f2937" stroke-width="5"/>
    <line x1="150" y1="80" x2="150" y2="200" stroke="#9aa0a6" stroke-width="2"/>
    <line x1="220" y1="80" x2="220" y2="200" stroke="#9aa0a6" stroke-width="2"/>
    <line x1="290" y1="80" x2="290" y2="200" stroke="#9aa0a6" stroke-width="2"/>
    <line x1="360" y1="80" x2="360" y2="200" stroke="#9aa0a6" stroke-width="2"/>
    <line x1="80" y1="80"  x2="360" y2="80"  stroke="#9aa0a6" stroke-width="1.6" stroke-dasharray="4 3" opacity="0.45"/>
    <line x1="80" y1="104" x2="360" y2="104" stroke="#1f2937" stroke-width="2"/>
    <line x1="80" y1="128" x2="360" y2="128" stroke="#1f2937" stroke-width="2"/>
    <line x1="80" y1="152" x2="360" y2="152" stroke="#1f2937" stroke-width="1.9"/>
    <line x1="80" y1="176" x2="360" y2="176" stroke="#1f2937" stroke-width="1.7"/>
    <line x1="80" y1="200" x2="360" y2="200" stroke="#1f2937" stroke-width="1.5"/>
    <g font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="end" dominant-baseline="central" font-size="11" font-weight="800" fill="#9aa0a6">
      <text x="70" y="80">E×</text>
    </g>
    <g font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="end" dominant-baseline="central" font-size="12" font-weight="800" fill="#1f2937">
      <text x="70" y="104">A</text>
      <text x="70" y="128">D</text>
      <text x="70" y="152">G</text>
      <text x="70" y="176">B</text>
      <text x="70" y="200">e</text>
    </g>
    <g font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" font-size="10" font-weight="700" fill="#6b7280">
      <text x="115" y="218">3</text><text x="185" y="218">4</text><text x="255" y="218">5</text>
    </g>

    <!-- same root, fifth, octave -->
    <circle cx="115" cy="104" r="12" fill="#3b82c4" stroke="#fff" stroke-width="2"/>
    <text x="115" y="104" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="10" font-weight="800" fill="#fff">C</text>
    <circle cx="255" cy="128" r="12" fill="#3b82c4" stroke="#fff" stroke-width="2"/>
    <text x="255" y="128" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="10" font-weight="800" fill="#fff">G</text>
    <circle cx="255" cy="152" r="12" fill="#3b82c4" stroke="#fff" stroke-width="2"/>
    <text x="255" y="152" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="10" font-weight="800" fill="#fff">C</text>
    <!-- ♭3 (E♭) at B:4  -- one fret LOWER than the major -->
    <circle cx="185" cy="176" r="14" fill="#dc2626" stroke="#fff" stroke-width="2.5"/>
    <text x="185" y="176" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="11" font-weight="800" fill="#fff">E♭</text>
    <circle cx="115" cy="200" r="12" fill="#3b82c4" stroke="#fff" stroke-width="2"/>
    <text x="115" y="200" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="10" font-weight="800" fill="#fff">G</text>

    <!-- ghost: where the major's 3rd WAS -->
    <circle cx="255" cy="176" r="12" fill="none" stroke="#9aa0a6" stroke-width="1.5" stroke-dasharray="3 2"/>
    <text x="255" y="176" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="9" font-weight="700" fill="#9aa0a6">E</text>
    <!-- arrow: E (fret 5) → E♭ (fret 4) -->
    <path d="M 244 176 L 198 176" fill="none" stroke="#dc2626" stroke-width="2" marker-end="url(#arrFlip)" opacity="0.9"/>

    <!-- label -->
    <rect x="80" y="240" width="280" height="46" rx="8" fill="#3b82c4"/>
    <text x="220" y="260" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" font-size="15" font-weight="800" fill="#fff">C minor   (x 3 5 5 4 3)</text>
    <text x="220" y="278" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" font-size="10" font-weight="700" fill="#fff">♭3 (E♭) on B-string, fret 4</text>
  </g>

  <defs>
    <marker id="arrFlip" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#dc2626"/>
    </marker>
  </defs>
</svg>

> **What to look at.** Every note on the neck is identical between the two chords *except* the red one on the B string. In C major it sits on fret 5 (E, the natural 3rd); in C minor it sits on fret 4 (E♭, the flat 3rd). One finger moves one fret backward — that's the entire major-to-minor transformation. Every Western emotional shift between bright and dark, joy and grief, runs through this one-fret move.

**Diminished (1 — b3 — b5)** → positions **{0, −3 CCW, +6}**
- C° = **C, Eb, Gb**.
- The 5th is swapped for the tritone (opposite the root). No "home side" — the tritone is unstable, which is exactly why dim chords push for resolution.

**Augmented (1 — 3 — #5)** → positions **{0, +4 CW, +8 CW (= −4 CCW)}**
- C+ = **C, E, G#**.
- Three notes spaced 4 steps apart — an **equilateral triangle** inscribed in the circle. Only **four** distinct augmented chords exist, because rotating the triangle lands on the same notes.

### Seventh chords (just add the 7th degree)

**Dominant 7 (1 — 3 — 5 — b7)** → **{0, +1, +4, −2}**
- C7 = C, E, G, Bb.
- The M3 (+4) and b7 (−2) are exactly **6 steps apart** — a **tritone**. *That* tritone is the engine of dominant function: in G7 → C, the B wants to climb to C and the F wants to drop to E, both by half-step.

**Major 7 (1 — 3 — 5 — 7)** → **{0, +1, +4, +5}**
- Cmaj7 = C, E, G, B.
- The M7 (+5) sits right next to the M3 (+4). All four tones cluster on the CW side — lush, calm, **no tritone**.

**Minor 7 (1 — b3 — 5 — b7)** → **{0, +1, −3, −2}**
- Cm7 = C, Eb, G, Bb.
- Mirror image of Cmaj7's cluster on the *CCW* side, plus the same +1 fifth.

**Half-diminished / m7b5 (1 — b3 — b5 — b7)** → **{0, −3, +6, −2}**
- Cm7b5 = C, Eb, Gb, Bb.
- m7 with the 5th flattened to the tritone. Functions as the **ii in a minor key**: in A minor, Bm7b5 → E7 → Am is the classic minor ii–V–i.

**Fully diminished 7 (1 — b3 — b5 — bb7)** → **{0, −3, +6, +3}**
- C°7 = C, Eb, Gb, A (bb7 = enharmonic M6).
- **Four positions, every 3 steps apart — a perfect square inscribed in the circle.** Because of that symmetry there are only **three** distinct dim7 chords; rotating the square lands on the same four pitches.

### Shape-on-the-circle comparison

Picture the C-rooted chord with **C at 12 o'clock**, and the other tones plotted around the dial:

```
Chord     Circle positions (relative to root at 0)    Geometric shape
─────────────────────────────────────────────────────────────────────
Major     {0, +1, +4}              CW fan (root + 5th + far M3 CW)
Minor     {0, +1, −3}              CCW-mirror of major (3rd flips side)
Aug       {0, +4, +8}              equilateral triangle (4 chords total)
Dim       {0, −3, +6}              minor triad with the 5th flung to opposite side
Maj7      {0, +1, +4, +5}          tight CW cluster, no tritone
Dom7      {0, +1, +4, −2}          M3 + b7 form the tritone (the tension)
m7        {0, +1, −3, −2}          minor triad + b7 (parallel to dom7, 3rd flipped)
m7b5      {0, −3, +6, −2}          half-dim — m7 with 5th flattened to tritone
dim7      {0, −3, +6, +3}          square — only 3 distinct chords exist
```

**The whole logic in one line:** flipping the 3rd's direction (CW ↔ CCW) is major ↔ minor; replacing the 5th with the diameter point (+6) is *5 → b5* (diminished family); adding the −2 CCW slot is *adding the b7* (dominant / minor-7 / half-dim family); the +5 CW slot is the *major 7th* (only mixes with major).

### Guitar fretboard echo

The single-fret moves on the neck mirror the single-step moves on the circle:

| Chord | Open shape | What changed from C |
|---|---|---|
| C       | `x32010` | baseline (1–3–5) |
| Cm      | `x35543` | E → Eb on the 2nd string (3rd flipped CW→CCW) |
| Cmaj7   | `x32000` | high C → B (5 CW: added M7) |
| C7      | `x32310` | high C → Bb (−2 CCW: added b7) |
| Cm7     | `x31313` | Cm + the same Bb |
| Cdim    | `x3424x` | 3rd flat *and* 5th flat |
| Cm7b5   | `x3434x` | one note different from Cm7 — the 5th drops to Gb |
| C°7     | `x3424x` | symmetric — slides up 3 frets to the same chord |

## V → I: The Engine

The single most important relationship in tonal music is **V resolving to I** — a fifth dropping to its root. In G, that's D → G; the F# in D (the 3rd) is a semitone below G (the leading tone), and the ear demands that resolution. This is why nearly every song ends on the I, usually preceded by the V.

The circle is literally a *chain of V→I* resolutions. Starting at any position and going one step counter-clockwise *is* a V→I move. That's why ii–V–I works: you're walking three steps counter-clockwise (Am → D → G in G major), three V→I pulls in a row, each landing you closer to home.

### On the neck: the V→I pull as finger motion

The resolution `G7 → C` is two single-fret moves happening simultaneously. The leading tone climbs by one fret; the chord seventh releases by one fret. See it on the strings:

<svg viewBox="0 0 720 360" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:760px;display:block;margin:0 auto;border-radius:12px">
  <rect width="720" height="360" rx="12" fill="#fdfcf7"/>
  <rect width="720" height="360" rx="12" fill="none" stroke="#d8d2c0" stroke-width="1.5"/>
  <text x="360" y="28" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" font-size="15" font-weight="800" fill="#1f2937">G7 → C major: the two-finger resolution</text>

  <!-- Single fretboard, frets 0-3, low E on top -->
  <!-- nut at x=100, fret centers: 1=170, 2=260, 3=350; line ends at x=410 -->
  <line x1="100" y1="80" x2="100" y2="230" stroke="#1f2937" stroke-width="5"/>
  <line x1="220" y1="80" x2="220" y2="230" stroke="#9aa0a6" stroke-width="2"/>
  <line x1="340" y1="80" x2="340" y2="230" stroke="#9aa0a6" stroke-width="2"/>
  <line x1="460" y1="80" x2="460" y2="230" stroke="#9aa0a6" stroke-width="2"/>
  <line x1="100" y1="80"  x2="460" y2="80"  stroke="#1f2937" stroke-width="2.4"/>
  <line x1="100" y1="110" x2="460" y2="110" stroke="#1f2937" stroke-width="2.2"/>
  <line x1="100" y1="140" x2="460" y2="140" stroke="#1f2937" stroke-width="2.0"/>
  <line x1="100" y1="170" x2="460" y2="170" stroke="#1f2937" stroke-width="1.8"/>
  <line x1="100" y1="200" x2="460" y2="200" stroke="#1f2937" stroke-width="1.6"/>
  <line x1="100" y1="230" x2="460" y2="230" stroke="#1f2937" stroke-width="1.4"/>
  <g font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="end" dominant-baseline="central" font-size="12" font-weight="800" fill="#1f2937">
    <text x="84" y="80">E</text><text x="84" y="110">A</text><text x="84" y="140">D</text>
    <text x="84" y="170">G</text><text x="84" y="200">B</text><text x="84" y="230">e</text>
  </g>
  <g font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" font-size="11" font-weight="700" fill="#6b7280">
    <text x="160" y="252">1</text><text x="280" y="252">2</text><text x="400" y="252">3</text>
  </g>

  <!-- G7 voicing on left half: B at A-string fret 2; F at high-e fret 1 -->
  <!-- C voicing: C at A-string fret 3; E at high-e fret 0 (open) -->

  <!-- B (G7's leading tone) -->
  <circle cx="280" cy="110" r="13" fill="#d97757" stroke="#fff" stroke-width="2"/>
  <text x="280" y="110" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="11" font-weight="800" fill="#fff">B</text>
  <!-- C (target) on same string -->
  <circle cx="400" cy="110" r="13" fill="#3b82c4" stroke="#fff" stroke-width="2"/>
  <text x="400" y="110" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="11" font-weight="800" fill="#fff">C</text>
  <!-- arrow B -> C : climbs one fret -->
  <path d="M 296 100 Q 340 75 384 100" fill="none" stroke="#1f2937" stroke-width="2" marker-end="url(#arrUp)" opacity="0.85"/>

  <!-- F (G7 chord seventh) -->
  <circle cx="160" cy="230" r="13" fill="#d97757" stroke="#fff" stroke-width="2"/>
  <text x="160" y="230" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="11" font-weight="800" fill="#fff">F</text>
  <!-- E (target) -- open high-e string -->
  <circle cx="84" cy="230" r="11" fill="#3b82c4" stroke="#fff" stroke-width="2"/>
  <text x="84" y="230" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="10" font-weight="800" fill="#fff">E</text>
  <!-- arrow F -> E : releases one fret -->
  <path d="M 144 244 Q 120 268 100 244" fill="none" stroke="#1f2937" stroke-width="2" marker-end="url(#arrDn)" opacity="0.85"/>

  <defs>
    <marker id="arrUp" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#1f2937"/>
    </marker>
    <marker id="arrDn" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#1f2937"/>
    </marker>
  </defs>

  <!-- callouts -->
  <g font-family="ui-sans-serif, system-ui, sans-serif">
    <rect x="500" y="80" width="200" height="50" rx="8" fill="#d97757"/>
    <text x="600" y="100" text-anchor="middle" font-size="14" font-weight="800" fill="#fff">G7  (V of C)</text>
    <text x="600" y="118" text-anchor="middle" font-size="10" font-weight="700" fill="#fff">B = leading tone · F = ♭7</text>

    <rect x="500" y="155" width="200" height="50" rx="8" fill="#3b82c4"/>
    <text x="600" y="175" text-anchor="middle" font-size="14" font-weight="800" fill="#fff">C major  (I)</text>
    <text x="600" y="193" text-anchor="middle" font-size="10" font-weight="700" fill="#fff">B → C · F → E (half-steps)</text>

    <text x="600" y="240" text-anchor="middle" font-size="11" font-weight="700" fill="#1f2937">Two fingers move</text>
    <text x="600" y="256" text-anchor="middle" font-size="11" font-weight="700" fill="#1f2937">one fret each.</text>
    <text x="600" y="272" text-anchor="middle" font-size="11" font-weight="600" fill="#6b7280">That's the whole resolution.</text>
  </g>

  <!-- bottom legend -->
  <g transform="translate(100,310)" font-family="ui-sans-serif, system-ui, sans-serif" font-size="11" font-weight="700" fill="#1f2937">
    <circle cx="0"   cy="0" r="6" fill="#d97757" stroke="#fff" stroke-width="2"/>
    <text   x="14"   y="0" text-anchor="start" dominant-baseline="central">G7 chord tone</text>
    <circle cx="160" cy="0" r="6" fill="#3b82c4" stroke="#fff" stroke-width="2"/>
    <text   x="174"  y="0" text-anchor="start" dominant-baseline="central">C major chord tone (target)</text>
  </g>
</svg>

> **The lesson.** Every dominant 7 → tonic move on the guitar is *literally* two fingers each sliding by one fret in opposite directions. That's the half-step pair physical music theory talks about — and it's the exact same machinery whether you're playing folk, blues, jazz, or pop. The B–F tritone inside G7 doesn't just *want* to resolve; it gives your fingers the easiest possible path home.

## Tone Substitution (Why the Circle Predicts What Sounds Right)

Substitution = replacing one chord with another that **shares enough notes / function** to feel like the same move. The circle exposes *why* the swap works because shared-circle-distance ≈ shared-pitch-content.

### 1. Tritone substitution (the classic)

**Rule.** Any dominant 7 can be replaced by the dominant 7 a tritone away (the chord directly across the circle).

- G7 = G, B, D, F. The pulling tritone is **B–F**.
- **Db7** = Db, F, Ab, **B** (=Cb). The tritone is **F–Cb**.
- Same two notes (B and F) — *that's the same tritone, just relabeled*. Both chords beg to resolve to **C major** with B→C and F→E.

Result: `Dm7 — G7 — Cmaj7` (ii–V–I) becomes `Dm7 — Db7 — Cmaj7`. The bassline now walks **chromatically down** (D → Db → C) and the resolution still lands. This is the single most-used reharmonization in jazz.

**Circle picture.** G and Db are **exactly opposite** (+6 / −6). The two tritone-sub partners are always diameter twins.

### 2. Diatonic substitution (relative neighbors)

Within a key, chords that share two of three notes are interchangeable:

| Sub | Replaces | In C | Why |
|---|---|---|---|
| iii  | I    | Em ↔ C   | Em = E, G, B; C = C, E, G — two shared tones |
| vi   | I    | Am ↔ C   | Am = A, C, E; C = C, E, G — two shared tones |
| vi   | IV   | Am ↔ F   | Am = A, C, E; F = F, A, C — two shared tones |
| ii   | IV   | Dm ↔ F   | Dm = D, F, A; F = F, A, C — two shared tones |
| vii° | V7 (no root) | Bdim ↔ G7 | Bdim = B, D, F = the upper three notes of G7 |

**Circle picture.** Diatonic substitutes are **always adjacent** on the wheel (chord ↔ its inner-ring partner, or chord ↔ next-door slot). Adjacency on the circle = maximal note overlap.

### 3. Secondary dominants ("V of V" etc.)

Any chord can be approached by *its own* V — borrowed from outside the key. The circle shows this as **prepending one CW step** to the target.

- In C major, V = G. **V of V** = the dominant of G = **D7**. So `D7 — G — C` is "tonicizing" G before resolving home. D is one CW from G; G is one CW from C — you're just walking three positions CCW around the circle.
- In C major, **V of vi** = E7 → Am (E is one CW from A on the circle).

### 4. Backdoor dominant (the bVII7 trick)

Instead of resolving via V→I (one CCW step), resolve via **bVII7→I** (two CCW steps + an unexpected b7 swap).
- In C: Bb7 → C. Bb7 shares two notes with Fm6 and with the tritone-sub of E7 — it sneaks in *through* the subdominant side of the circle. Heard in jazz endings and a lot of Steely Dan.

### 5. Modal interchange (parallel-key borrowing)

Swap a chord for its parallel-minor equivalent — same root, opposite-side tonality.
- In C major, borrow **Fm** from C minor: `C — Fm — C` (Coldplay's "Yellow," lots of Beatles).
- On the circle: C major lives in the no-sharps/no-flats slot; C minor sits **3 steps CCW** (three flats). Borrowed chords show up where those two key-zones overlap. The **bVII (Bb), bIII (Eb), bVI (Ab), iv (Fm)** are the four common loans — all are clustered on the CCW side of C.

### 6. Why all five sub-types work for the same reason

Every substitution above is some flavor of: *replace a chord with one whose tones live nearby on the circle (so most pitches are kept) or whose tritone matches (so the pull is kept)*. The circle is literally a **pitch-proximity map**, so substitutions that work musically also work geometrically — they're either:

- **Adjacent** (diatonic, secondary dominant, modal interchange) → shared note content
- **Diameter-opposite** (tritone sub) → shared tritone, opposite roots

Once you see it that way, you don't memorize subs — you read them off the wheel.

## Relative Minor and Modal Cousins

The relative minor shares the **same key signature** but starts on the 6th degree of the major scale. Em is the relative minor of G — both have one sharp (F#), both use the same seven notes, but Em-centered music *feels* sad/dark because the tonic is now Em, and the V→i pull is now Bm → Em (or B7 → Em if you want the leading tone).

This is why songs like "Stairway to Heaven" (Am, key signature of C) and "House of the Rising Sun" (Am, also no sharps/flats) feel related to C major songs — they share an alphabet. Switching between a major key and its relative minor mid-song is one of the cheapest, most powerful moves in songwriting.

## Guitar-Specific Shortcuts

### Tuning *is* the circle (in fourths)

Standard tuning E–A–D–G–B–E: the bottom four strings go up by a perfect fourth each — which means they go *counter-clockwise* around the circle. The B string breaks the pattern (it's a third above G, not a fourth) so the high strings get a different shape, but the principle holds: many of your scale and chord fingerings *are* circle moves baked into geometry.

### CAGED and key transposition

The five CAGED shapes (C, A, G, E, D) cover the fretboard. Combined with the circle, transposing a song is mechanical: count fifths between source and target, and either capo or shift shapes.

Example: a song is in G but the singer wants it in A. A is one fifth above G (one step CW on the circle), which is 2 semitones up. **Capo 2, play G shapes** — done. The circle told you "1 step clockwise = 2 frets."

### Reading dominant 7 and its tritone sub off the fretboard

Your intuition lines up exactly. Working in **C major** as home:

- **G7** is the V7. G is a P5 above C (7 frets up, or one CW step on the circle). G7 = **G  B  D  F**. The crucial bit is the **tritone B↔F** inside it — that's the engine asking to resolve.
- The tritone substitute is **D♭7** (= C♯7 — same frets on guitar, different spelling). D♭ is a P5 above G♭, which means it sits **directly across the circle** from G — exactly 6 steps away.
- D♭7 = **D♭  F  A♭  C♭**. The C♭ is enharmonically a B. So D♭7 contains the **same B and F** as G7 — same tritone, different root. Both chords resolve to C the same way: B → C (climbs a half-step) and F → E (falls a half-step).

So yes — relative to C as root, D♭ is the **♭2** (one fret above the root). The standard label for the tritone sub is therefore **♭II7**, and the substitution turns a ii–V–I into a **ii–♭II7–I** with a chromatic bass walk:

```
Functional name:   ii   V7   I            ii    ♭II7   I
Chord (in C):      Dm7  G7   Cmaj7   →    Dm7   D♭7    Cmaj7
Bass note:          D    G    C            D     D♭     C       <- chromatic descent
```

That chromatic D → D♭ → C bassline is *the* sonic signature of a tritone sub. Jazz players reach for it because it gets you to the I chord with the smoothest possible voice leading.

### How it actually sits on the neck (open-position shapes)

Both G7 and D♭7 use the same dominant-7 **shape** — the latter is just slid up the neck:

| Chord | Shape | Notes (low → high) | Tritone pair inside |
|---|---|---|---|
| G7  | `320001` (open G + add F on string 1, fret 1) | G B D G B F | **B–F**  (strings 5↔1) |
| D♭7 | `x4342x` (5th-fret A7 shape moved down to D♭) | — D♭ A♭ B F — | **B–F**  (strings 3↔2) |

If you fret both and listen, the **B and F notes ring in both chords** — that's why they swap-in cleanly for each other. The G or D♭ in the bass is what changes color; the upper tritone is what carries the function.

Drawn on the fretboard (frets 1–5, low E at top → high E at bottom):

<svg viewBox="0 0 720 380" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:760px;display:block;margin:0 auto;border-radius:12px">
  <rect width="720" height="380" rx="12" fill="#fdfcf7"/>
  <rect width="720" height="380" rx="12" fill="none" stroke="#d8d2c0" stroke-width="1.5"/>

  <!-- title -->
  <text x="360" y="28" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" font-size="15" font-weight="800" fill="#1f2937">G7 and D♭7 share the B–F tritone (red) — that's why they substitute</text>

  <!-- ============================================================
       FRETBOARD GEOMETRY
       6 strings (E A D G B E) at y = 80, 110, 140, 170, 200, 230
       6 fret slots: nut (x=80) + 5 frets, each 100 px wide
       fret centers (for placing dots) at x = 130, 230, 330, 430, 530
       nut at x=80, last fret line at x=580
       ============================================================ -->

  <!-- nut (thick) -->
  <line x1="80" y1="80" x2="80" y2="230" stroke="#1f2937" stroke-width="5"/>
  <!-- frets -->
  <line x1="180" y1="80" x2="180" y2="230" stroke="#9aa0a6" stroke-width="2"/>
  <line x1="280" y1="80" x2="280" y2="230" stroke="#9aa0a6" stroke-width="2"/>
  <line x1="380" y1="80" x2="380" y2="230" stroke="#9aa0a6" stroke-width="2"/>
  <line x1="480" y1="80" x2="480" y2="230" stroke="#9aa0a6" stroke-width="2"/>
  <line x1="580" y1="80" x2="580" y2="230" stroke="#9aa0a6" stroke-width="2"/>

  <!-- strings (thickness from low to high) -->
  <line x1="80" y1="80"  x2="580" y2="80"  stroke="#1f2937" stroke-width="2.4"/>
  <line x1="80" y1="110" x2="580" y2="110" stroke="#1f2937" stroke-width="2.2"/>
  <line x1="80" y1="140" x2="580" y2="140" stroke="#1f2937" stroke-width="2.0"/>
  <line x1="80" y1="170" x2="580" y2="170" stroke="#1f2937" stroke-width="1.8"/>
  <line x1="80" y1="200" x2="580" y2="200" stroke="#1f2937" stroke-width="1.6"/>
  <line x1="80" y1="230" x2="580" y2="230" stroke="#1f2937" stroke-width="1.4"/>

  <!-- string labels (left) -->
  <g font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="end" dominant-baseline="central" font-size="13" font-weight="800" fill="#1f2937">
    <text x="60" y="80">E (low)</text>
    <text x="60" y="110">A</text>
    <text x="60" y="140">D</text>
    <text x="60" y="170">G</text>
    <text x="60" y="200">B</text>
    <text x="60" y="230">e (high)</text>
  </g>

  <!-- fret number labels (below) -->
  <g font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" font-size="11" font-weight="700" fill="#6b7280">
    <text x="130" y="252">1</text>
    <text x="230" y="252">2</text>
    <text x="330" y="252">3</text>
    <text x="430" y="252">4</text>
    <text x="530" y="252">5</text>
  </g>

  <!-- ===================== G7  (open-position 320001 plus F on string 1 fret 1) ===================== -->
  <!-- Voicing used here:  E:3(G)  A:2(B)  D:0(D)  G:0(G)  B:0(B)  e:1(F)
       This makes the tritone B(string 5 fret 2) ↔ F(string 1 fret 1) visible. -->

  <!-- G  on low E, fret 3 -->
  <circle cx="330" cy="80" r="14" fill="#d97757" stroke="#fff" stroke-width="2"/>
  <text x="330" y="80" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="11" font-weight="800" fill="#fff">G</text>

  <!-- B  on A, fret 2 -- TRITONE -->
  <circle cx="230" cy="110" r="14" fill="#dc2626" stroke="#fff" stroke-width="2"/>
  <text x="230" y="110" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="11" font-weight="800" fill="#fff">B</text>

  <!-- D  open string D -->
  <circle cx="50" cy="140" r="11" fill="none" stroke="#d97757" stroke-width="2.5"/>
  <text x="50" y="140" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="10" font-weight="800" fill="#d97757">D</text>

  <!-- G  open string G (use small 'o' marker style on the same string line) -->
  <circle cx="50" cy="170" r="11" fill="none" stroke="#d97757" stroke-width="2.5"/>
  <text x="50" y="170" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="10" font-weight="800" fill="#d97757">G</text>

  <!-- B  open string B -- the open-B is part of G7 (still a chord tone), but the tritone B is the fretted one at string A, fret 2 -->
  <circle cx="50" cy="200" r="11" fill="none" stroke="#d97757" stroke-width="2.5"/>
  <text x="50" y="200" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="10" font-weight="800" fill="#d97757">B</text>

  <!-- F  on high e, fret 1 -- TRITONE -->
  <circle cx="130" cy="230" r="14" fill="#dc2626" stroke="#fff" stroke-width="2"/>
  <text x="130" y="230" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="11" font-weight="800" fill="#fff">F</text>

  <!-- highlight the tritone with a connecting line -->
  <line x1="230" y1="110" x2="130" y2="230" stroke="#dc2626" stroke-width="2" stroke-dasharray="5 3" opacity="0.7"/>

  <!-- label -->
  <g font-family="ui-sans-serif, system-ui, sans-serif">
    <rect x="600" y="80" width="100" height="60" rx="8" fill="#d97757"/>
    <text x="650" y="100" text-anchor="middle" font-size="16" font-weight="800" fill="#fff">G7</text>
    <text x="650" y="118" text-anchor="middle" font-size="10" font-weight="700" fill="#fff">3 2 0 0 0 1</text>
    <text x="650" y="132" text-anchor="middle" font-size="10" font-weight="600" fill="#fff">V of C major</text>
  </g>

  <!-- ===================== D♭7  (movable shape:  x 4 3 4 2 x ) ===================== -->
  <!-- Place a second mini-fretboard below the first one? No - same fretboard with another color, but spatially overlapping is messy.
       Better: draw a second fretboard underneath. -->

  <!-- =============================================================
        FRETBOARD #2  (D♭7  voicing x 4 3 4 2 x)
        Same X mapping but Y shifted down by 100 px region:
        strings at y = 290 (skipped, mute), 300 (A), 310 (D), 320 (G), 330 (B), 340 (mute)
        Actually let me put the WHOLE second fretboard below cleanly.
       ============================================================= -->
  <line x1="80" y1="290" x2="80" y2="350" stroke="#1f2937" stroke-width="5"/>
  <line x1="180" y1="290" x2="180" y2="350" stroke="#9aa0a6" stroke-width="2"/>
  <line x1="280" y1="290" x2="280" y2="350" stroke="#9aa0a6" stroke-width="2"/>
  <line x1="380" y1="290" x2="380" y2="350" stroke="#9aa0a6" stroke-width="2"/>
  <line x1="480" y1="290" x2="480" y2="350" stroke="#9aa0a6" stroke-width="2"/>
  <line x1="580" y1="290" x2="580" y2="350" stroke="#9aa0a6" stroke-width="2"/>
  <!-- 4 inner strings for D♭7 voicing -->
  <line x1="80" y1="290" x2="580" y2="290" stroke="#1f2937" stroke-width="2.0"/>
  <line x1="80" y1="310" x2="580" y2="310" stroke="#1f2937" stroke-width="1.9"/>
  <line x1="80" y1="330" x2="580" y2="330" stroke="#1f2937" stroke-width="1.7"/>
  <line x1="80" y1="350" x2="580" y2="350" stroke="#1f2937" stroke-width="1.5"/>
  <g font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="end" dominant-baseline="central" font-size="13" font-weight="800" fill="#1f2937">
    <text x="60" y="290">A</text>
    <text x="60" y="310">D</text>
    <text x="60" y="330">G</text>
    <text x="60" y="350">B</text>
  </g>
  <!-- D♭7:  A:4(D♭)   D:3(F)   G:4(B=C♭)   B:2(C♯)  -->
  <!-- Wait — let me re-derive cleanly.
       Standard 'A7' open shape is x 0 2 0 2 0 (A C♯ G C♯ E? No.)
       The A-shape barre at fret 4 -> root on A string fret 4 = C♯ = D♭.
       Shape (relative to barre): A:4(root)  D:6(5th, often muted)  G:4(♭7)  B:5(3rd)  high-e:4(root)
       Simpler 4-string voicing of D♭7:  A:4 D:3 G:4 B:2  giving D♭ F C♭(=B) E♭?  hmm
       Cleanest small-shape D♭7 = "x 4 3 4 2 x" :
         A string fret 4 = C♯/D♭    (root)
         D string fret 3 = F        (3rd)
         G string fret 4 = B = C♭   (♭7)
         B string fret 2 = C♯/D♭    (root octave)
       So F and B (=C♭) are the shared tritone with G7.
   -->
  <!-- D♭ on A string fret 4 -->
  <circle cx="430" cy="290" r="14" fill="#3b82c4" stroke="#fff" stroke-width="2"/>
  <text x="430" y="290" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="10" font-weight="800" fill="#fff">D♭</text>
  <!-- F on D string fret 3 -- TRITONE -->
  <circle cx="330" cy="310" r="14" fill="#dc2626" stroke="#fff" stroke-width="2"/>
  <text x="330" y="310" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="11" font-weight="800" fill="#fff">F</text>
  <!-- B (=Cb) on G string fret 4 -- TRITONE -->
  <circle cx="430" cy="330" r="14" fill="#dc2626" stroke="#fff" stroke-width="2"/>
  <text x="430" y="330" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="11" font-weight="800" fill="#fff">B</text>
  <!-- D♭ on B string fret 2 -->
  <circle cx="230" cy="350" r="14" fill="#3b82c4" stroke="#fff" stroke-width="2"/>
  <text x="230" y="350" font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" dominant-baseline="central" font-size="10" font-weight="800" fill="#fff">D♭</text>

  <!-- connect the tritone -->
  <line x1="330" y1="310" x2="430" y2="330" stroke="#dc2626" stroke-width="2" stroke-dasharray="5 3" opacity="0.7"/>

  <!-- label -->
  <g font-family="ui-sans-serif, system-ui, sans-serif">
    <rect x="600" y="290" width="100" height="60" rx="8" fill="#3b82c4"/>
    <text x="650" y="310" text-anchor="middle" font-size="16" font-weight="800" fill="#fff">D♭7</text>
    <text x="650" y="328" text-anchor="middle" font-size="10" font-weight="700" fill="#fff">x 4 3 4 2 x</text>
    <text x="650" y="342" text-anchor="middle" font-size="10" font-weight="600" fill="#fff">♭II7 (tritone sub)</text>
  </g>

  <!-- fret number labels for second fretboard -->
  <g font-family="ui-sans-serif, system-ui, sans-serif" text-anchor="middle" font-size="11" font-weight="700" fill="#6b7280">
    <text x="130" y="372">1</text>
    <text x="230" y="372">2</text>
    <text x="330" y="372">3</text>
    <text x="430" y="372">4</text>
    <text x="530" y="372">5</text>
  </g>
</svg>

> **What to look at.** Red dots = the shared tritone (B and F). In **G7** it lives on string 5 fret 2 (B) and string 1 fret 1 (F). In **D♭7** the *same two notes* live on string 4 fret 3 (F) and string 3 fret 4 (B, spelled C♭). The fingerings look different but the resolution machinery is identical — that's the geometric reality of "tritone substitution" on the neck.

### Bonus: secondary dominants on the fretboard

The same logic generalizes. To "tonicize" any chord temporarily, play **its** V7 (or its tritone sub) right before it. In C major:

| You want to land on | V7 of it | Tritone sub |
|---|---|---|
| C (I)        | G7  | D♭7 |
| Dm (ii)      | A7  | E♭7 |
| Em (iii)     | B7  | F7  |
| F (IV)       | C7  | G♭7 |
| G (V)        | D7  | A♭7 |
| Am (vi)      | E7  | B♭7 |

Every row is one CW circle-step away (the V7) or six CW steps (the tritone sub). On the guitar, this means inserting one dominant-7 shape before the target chord. `C — A7 — Dm — G7 — C` ("Five Foot Two" / "Last Night I Had the Strangest Dream") is exactly this trick: A7 is the V7 of Dm, even though A natural isn't in C major — it's a one-bar visit to D-minor land before resolving back.

### Finding the key of an unknown song by ear

1. Find the "home" note — the one the song keeps returning to and ends on. That's the tonic.
2. Listen for whether the resting chord is major or minor.
3. Position on circle = key signature = list of in-key chords.

Most songs you'll encounter are in **G, C, D, A, or E major** (and their relative minors Em, Am, Bm, F#m, C#m) because those keys lay well on the guitar's open strings.

## Trade-offs

- **The circle assumes tonal harmony.** It's a perfect map for ~95% of pop, folk, rock, country, blues, and classical. It is *not* the right map for modal music (much of jazz, Indian classical, folk modal tunes), atonal music, or anything that pivots on tritones rather than fifths.
- **It encodes diatonic, not chromatic.** Borrowed chords (e.g., the bVII in rock — F major in a G song), secondary dominants (D7 leading to G in C major), and modal interchange all live *off* the circle. The circle tells you what's *expected*; departures from it are where character lives.
- **Equal temperament hides the seam.** The Pythagorean comma is real — stacked acoustic fifths don't quite close. Fretted guitars hide this by compressing every fifth slightly flat. Worth knowing if you ever play with a violinist who *isn't* compressing.

## Common Pitfalls

- **Memorizing the circle as a static picture, not as a process.** If you can't *generate* it from "stack fifths from C," you'll forget it. Practice writing it from scratch every few weeks.
- **Confusing relative and parallel minor.** Relative minor of C = Am (same key sig, vi degree). Parallel minor of C = Cm (same root, different key sig — 3 flats). Mixing them up makes modal interchange opaque.
- **Treating ii–V–I as jazz-only.** It's everywhere — "Let It Be" (Dm–G–C in C), "Hey Jude" (the verse), and most country songs all use ii–V–I or its sibling IV–V–I.
- **Forgetting F# and Gb are the same fretboard position.** When transposing, swap to the easier-spelling enharmonic — they're the same notes on a fretted instrument.
- **Counting frets instead of fifths for transposition.** Both work, but the circle tells you *why* the new key has its specific chord set; semitone counting only gives you the new tonic.

## Decision Table

| Scenario | Use this | Not that | Why |
|----------|----------|----------|-----|
| "What chords are in key X?" | Circle: I = X, IV = one CCW, V = one CW, vi = inner ring | Memorizing chord lists per key | Three positions on the wheel give you the whole diatonic set |
| "Transpose this G-major song to A" | Capo or shift up one CW step (2 frets) | Re-deriving every chord | One circle step = one transposition delta |
| "What's the relative minor of D?" | Inner ring of D = Bm (vi of D major) | Counting "down a minor third" semitone by semitone | The inner ring labels it directly |
| "Why does V→I sound final?" | Circle: V is one CW from I, so resolving CCW = "coming home" by a fifth | Treating it as arbitrary convention | The fifth is the strongest interval after the octave |
| "Should I write F major in my G-major song?" | Off-circle = borrowed (modal mixture, bVII rock move) | Calling it a "wrong" chord | Off-circle chords are intentional color, not errors |
| Sight-reading key signatures | Sharp count → key via "last sharp + semitone = tonic" trick | Memorizing all 15 key sigs | The trick is two arithmetic steps and works every time |

## References

- [Circle of Fifths — Wikipedia](https://en.wikipedia.org/wiki/Circle_of_fifths) — canonical history, including Heinichen's 1728 diagram.
- [Key Signature — Wikipedia](https://en.wikipedia.org/wiki/Key_signature) — the order-of-sharps/flats rules and mnemonics.
- [Equal Temperament — Wikipedia](https://en.wikipedia.org/wiki/Equal_temperament) — why 12 fifths close on a fretted instrument but not acoustically.
- [Roman Numeral Analysis — Wikipedia](https://en.wikipedia.org/wiki/Roman_numeral_analysis) — diatonic chord quality conventions (uppercase major, lowercase minor, ° diminished).
- [CAGED System — Wikipedia](https://en.wikipedia.org/wiki/CAGED_system) — five-shape framework that pairs naturally with circle-based transposition.

## See Also

- (placeholder for future) `[fretboard-intervals](./fretboard-intervals.md)` — the geometry of intervals across strings.
- (placeholder for future) `[modes-and-modal-interchange](./modes-and-modal-interchange.md)` — what lives *outside* the circle.
