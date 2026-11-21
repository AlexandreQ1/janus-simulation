#include "color.h"
#include <cstring>

// Static member definitions
int   Color::max_mass        = 1;
float Color::pos_palette[32][3] = {};
float Color::neg_palette[32][3] = {};
float Color::vel_palette[32][3] = {};

// ---------------------------------------------------------------------------
// Utility: linearly interpolate a single channel
// ---------------------------------------------------------------------------
static float lerp(float a, float b, float t) {
    return a + (b - a) * t;
}

// ---------------------------------------------------------------------------
// Warm palette: black -> red -> orange -> yellow -> white  (32 steps)
//
// Segment layout (indices 0-31):
//   0-7   : black  (0,0,0)  ->  dark red (0.5, 0, 0)
//   8-15  : dark red        ->  red      (1,   0, 0)
//   16-23 : red             ->  yellow   (1,   1, 0)
//   24-31 : yellow          ->  white    (1,   1, 1)
// ---------------------------------------------------------------------------
void Color::init_color_table() {
    // Keypoints
    const float keys[5][3] = {
        {0.0f, 0.0f, 0.0f},   // 0  black
        {0.5f, 0.0f, 0.0f},   // 8  dark red
        {1.0f, 0.0f, 0.0f},   // 16 red
        {1.0f, 1.0f, 0.0f},   // 24 yellow
        {1.0f, 1.0f, 1.0f},   // 32 white
    };

    for (int i = 0; i < 32; ++i) {
        // Determine which segment [0,4) this index falls into
        float fi = static_cast<float>(i);
        float seg = fi / 8.0f;             // 0..4 (exclusive)
        int   s   = static_cast<int>(seg); // segment index 0-3
        if (s > 3) s = 3;
        float t = seg - static_cast<float>(s);  // fractional position

        for (int c = 0; c < 3; ++c) {
            pos_palette[i][c] = lerp(keys[s][c], keys[s + 1][c], t);
        }
    }
}

// ---------------------------------------------------------------------------
// Negative mass palette: black -> dark fuchsia -> fuchsia -> bright -> pink
// (matches Universe: magenta/fuchsia for clear distinction from warm +)
// ---------------------------------------------------------------------------
void Color::ninit_color_table() {
    const float keys[5][3] = {
        {0.0f, 0.0f, 0.0f},   // 0  black
        {0.4f, 0.0f, 0.4f},   // 8  dark fuchsia
        {0.8f, 0.0f, 0.8f},   // 16 fuchsia
        {1.0f, 0.1f, 1.0f},   // 24 bright fuchsia
        {1.0f, 0.5f, 0.9f},   // 32 pale pink
    };

    for (int i = 0; i < 32; ++i) {
        float fi  = static_cast<float>(i);
        float seg = fi / 8.0f;
        int   s   = static_cast<int>(seg);
        if (s > 3) s = 3;
        float t = seg - static_cast<float>(s);

        for (int c = 0; c < 3; ++c) {
            neg_palette[i][c] = lerp(keys[s][c], keys[s + 1][c], t);
        }
    }
}

// ---------------------------------------------------------------------------
// Velocity palette : jet  dark blue -> blue -> cyan -> green -> yellow -> red
//   0-5   : dark blue   (0.0, 0.0, 0.5) -> blue  (0.0, 0.2, 1.0)
//   6-11  : blue        (0.0, 0.2, 1.0) -> cyan  (0.0, 1.0, 1.0)
//   12-18 : cyan        (0.0, 1.0, 1.0) -> yellow(1.0, 1.0, 0.0)
//   19-25 : yellow      (1.0, 1.0, 0.0) -> orange(1.0, 0.5, 0.0)
//   26-31 : orange      (1.0, 0.5, 0.0) -> bright red (1.0, 0.0, 0.0)
// ---------------------------------------------------------------------------
void Color::vel_init_color_table() {
    const float keys[6][3] = {
        {0.0f, 0.0f, 0.5f},  //  0  dark blue  — zero velocity
        {0.0f, 0.2f, 1.0f},  //  6  bright blue
        {0.0f, 1.0f, 1.0f},  // 12  cyan
        {1.0f, 1.0f, 0.0f},  // 19  yellow
        {1.0f, 0.5f, 0.0f},  // 25  orange
        {1.0f, 0.0f, 0.0f},  // 31  red — max velocity
    };
    // Intervals (in indices 0-31): 6, 6, 7, 6, 6
    const int breaks[6] = {0, 6, 12, 19, 25, 31};

    for (int i = 0; i < 32; ++i) {
        // Find which segment we are in
        int seg = 4;
        for (int s = 0; s < 5; ++s) {
            if (i <= breaks[s+1]) { seg = s; break; }
        }
        float span = (float)(breaks[seg+1] - breaks[seg]);
        float t    = (span > 0.f) ? (float)(i - breaks[seg]) / span : 1.0f;
        for (int c = 0; c < 3; ++c)
            vel_palette[i][c] = lerp(keys[seg][c], keys[seg+1][c], t);
    }
}

void Color::getPosPalette(float palette[32][3]) {
    std::memcpy(palette, pos_palette, sizeof(pos_palette));
}

void Color::getNegPalette(float palette[32][3]) {
    std::memcpy(palette, neg_palette, sizeof(neg_palette));
}

void Color::getVelPalette(float palette[32][3]) {
    std::memcpy(palette, vel_palette, sizeof(vel_palette));
}
