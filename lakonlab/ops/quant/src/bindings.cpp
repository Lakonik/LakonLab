#include <torch/extension.h>

#include "int8_mm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int8_mm", &int8_mm);
}
