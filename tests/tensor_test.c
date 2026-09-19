#include "common.h"
#include "math/tensor.h"
#include <stdio.h>

int main(void) {
    Tensor t; usize shape[5] = {2, 4, 5, 6, 7}; 
    tensor_create(&t, 5, shape);

    printf("Size: %zu\n", t.size);
    printf("Rank: %zu\n", t.rank);

    printf("Strides: ");
    for (usize i = 0; i < t.rank; i++)
        printf("%zu ", t.strides[i]);

    printf("\nShape: ");
    for (usize i = 0; i < t.rank; i++)
        printf("%zu ", t.shape[i]);

    tensor_destroy(&t);
    
    return 0;
}
