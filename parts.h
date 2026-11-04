// parts.h -- which rungs of the ladder are implemented.
//
// The harness runs, checks, and profiles only the rungs switched on
// here, so you can build and measure Part 1 before Part 6 exists.
// The naive kernel is always on, and the provided reg1d starts on.
// Flip a switch to 1 when the rung is written; `make check` starts
// testing it on the next build.  HAVE_SWEEP needs HAVE_COALESCED and
// HAVE_TILED.
//
// The graded submission has every switch at 1.

#ifndef LAB2_PARTS_H
#define LAB2_PARTS_H

#define HAVE_COALESCED 0    // Part 2: sgemm_coalesced
#define HAVE_TILED     0    // Part 3: sgemm_tiled
#define HAVE_SWEEP     0    // Part 4: sgemm_coalesced_shape,
                            //         sgemm_tiled_size
#define HAVE_REG1D     1    // Part 5: sgemm_reg1d (provided)
#define HAVE_REG2D     0    // Part 6: sgemm_reg2d
#define HAVE_MINE      0    // Part 7: sgemm_mine

#endif // LAB2_PARTS_H
