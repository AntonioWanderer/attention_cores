
#include "division.h"

Tensor3D Divide(Tensor3D input, double divisor){
    size_t B = input.B;
    size_t S = input.S;
    size_t E = input.E;

    Tensor3D result = Tensor3D(B, S, E, false);

    for (size_t b = 0; b < B; b++) {
        for (size_t s = 0; s < S; s++) {
            for (size_t e = 0; e < E; e++) {
                result.at(b, s, e) = input.at(b, s, e) / divisor;
            }
        }
    }

    return result;
}