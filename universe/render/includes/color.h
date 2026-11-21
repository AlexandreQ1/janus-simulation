#pragma once

// Color palette manager for particle rendering.
//
// Two palettes of 32 RGB triplets are maintained:
//   pos_palette  - warm ramp (black -> red -> orange -> yellow -> white)
//                  used for positive-mass particles
//   neg_palette  - cool ramp (black -> blue -> cyan -> white)
//                  used for negative-mass particles
//
// Usage:
//   Color::init_color_table();           // build warm palette
//   Color::ninit_color_table();          // build cool palette
//   Color::getPosPalette(palette);       // copy to float[32][3]
//   Color::getNegPalette(palette);       // copy to float[32][3]
class Color {
public:
    static int max_mass;

    static void init_color_table();      // palette chaude masses positives
    static void ninit_color_table();     // palette froide masses négatives
    static void vel_init_color_table();  // palette vitesse (jet : bleu→cyan→jaune→rouge)

    static void getPosPalette(float palette[32][3]);
    static void getNegPalette(float palette[32][3]);
    static void getVelPalette(float palette[32][3]);  // pour Color by Velocity

private:
    static float pos_palette[32][3];
    static float neg_palette[32][3];
    static float vel_palette[32][3];
};
