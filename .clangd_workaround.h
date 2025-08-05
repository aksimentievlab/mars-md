#pragma once

// Workaround for clangd not recognizing certain C++20 functions
// This file is force-included by .clangd configuration

// Ensure __cplusplus is set to C++20
#if !defined(__cplusplus) || __cplusplus < 202002L
#undef __cplusplus
#define __cplusplus 202002L
#endif

// Ensure feature test macros are available
#ifndef __cpp_lib_is_constant_evaluated
#define __cpp_lib_is_constant_evaluated 201811L
#endif

#ifndef __cpp_impl_three_way_comparison
#define __cpp_impl_three_way_comparison 202002L
#endif

#ifndef __cpp_lib_constexpr_string
#define __cpp_lib_constexpr_string 201907L
#endif

#ifndef __cpp_constexpr_dynamic_alloc
#define __cpp_constexpr_dynamic_alloc 201907L
#endif

// Provide fallback for std::__is_constant_evaluated if not available
namespace std {
#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

#if !__has_builtin(__builtin_is_constant_evaluated) && !defined(__is_constant_evaluated)
    // Fallback implementation that always returns false for syntax analysis
    constexpr bool __is_constant_evaluated() noexcept { return false; }
#elif __has_builtin(__builtin_is_constant_evaluated) && !defined(__is_constant_evaluated)
    // Use the builtin if available
    constexpr bool __is_constant_evaluated() noexcept { 
        return __builtin_is_constant_evaluated(); 
    }
#endif

#if !defined(is_constant_evaluated) && __cpp_lib_is_constant_evaluated >= 201811L
    // Provide std::is_constant_evaluated if not available
    constexpr bool is_constant_evaluated() noexcept {
        return __is_constant_evaluated();
    }
#endif
}

// Ensure _GLIBCXX23_CONSTEXPR is defined as empty for C++20
#ifndef _GLIBCXX23_CONSTEXPR
#define _GLIBCXX23_CONSTEXPR
#endif 