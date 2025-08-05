#include "ARBDException.h"

namespace ARBD {

std::string Exception::type_to_str(ExceptionType type) {
    switch (type) {
        case ExceptionType::UnspecifiedError:    return "Unspecified Error";
        case ExceptionType::NotImplementedError: return "Not Implemented Error";
        case ExceptionType::ValueError:          return "Value Error";
        case ExceptionType::DivideByZeroError:   return "Divide By Zero Error";
        case ExceptionType::CUDARuntimeError:    return "CUDA Runtime Error";
        case ExceptionType::SYCLRuntimeError:    return "SYCL Runtime Error";
        case ExceptionType::MetalRuntimeError:   return "Metal Runtime Error";
        case ExceptionType::FileIoError:         return "File IO Error";
        case ExceptionType::FileOpenError:       return "File Open Error";
        case ExceptionType::RuntimeError:        return "Unified Runtime Error";
        default:
            return "Unknown Error Code (" + std::to_string(static_cast<int>(type)) + ")";
    }
}

} 