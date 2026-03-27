// test_mil_variants.cpp — Systematically test what MIL features ANE accepts
// to find which ops/shapes can handle dynamic weights
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "core/ane_runtime.h"
#include <ane_lm/common.h>
#include <objc/objc.h>
#include <objc/runtime.h>
#include <objc/message.h>
#include <IOSurface/IOSurface.h>

extern "C" void* objc_autoreleasePoolPush(void);
extern "C" void  objc_autoreleasePoolPop(void*);

using namespace ane_lm;

// Expose ane_compile_raw by re-declaring (it's static in ane_runtime.cpp)
// Instead, let's use a wrapper that calls into the ANE compilation directly
// We'll replicate the minimal compilation path here.

// Actually, the simplest approach: add test MIL generators and compile them
// through the existing ane_compile_raw. But it's static...
// Let's just add public test functions to ane_runtime.

// For now, I'll write MIL to a temp file and use the same approach as
// ane_compile_raw but exposed. Since ane_compile_raw is in the same TU,
// let me just add the variants as new public functions.

// HACK: We can test by generating different MIL programs and passing them
// through ane_compile_dynamic_matmul's code path after modifying the MIL string.
// But cleaner: add a generic "compile MIL program" function.

// Let's do it the quick way — add test compile functions to ane_runtime

int main() {
    printf("See test_dynamic2.cpp — this file is a placeholder\n");
    return 0;
}
