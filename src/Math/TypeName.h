#pragma once

#include <cstdlib>
#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>
#if __has_include(<cxxabi.h>)
#include <cxxabi.h>
#define HAS_CXXABI 1
#else
#define HAS_CXXABI 0
#endif
#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Backend/Resource.h"

namespace ARBD {
template<typename T>
concept Demangable = requires { typeid(T).name(); };

template<typename T>
concept HasTypeid = requires {
	typename T::value_type; // Example: for container types
} || std::is_fundamental_v<T>;

namespace detail {
template<typename T>
consteval const char* basic_type_name() noexcept {
	if constexpr (std::is_same_v<T, bool>)
		return "bool";
	else if constexpr (std::is_same_v<T, char>)
		return "char";
	else if constexpr (std::is_same_v<T, signed char>)
		return "signed char";
	else if constexpr (std::is_same_v<T, unsigned char>)
		return "unsigned char";
	else if constexpr (std::is_same_v<T, short>)
		return "short";
	else if constexpr (std::is_same_v<T, unsigned short>)
		return "unsigned short";
	else if constexpr (std::is_same_v<T, int>)
		return "int";
	else if constexpr (std::is_same_v<T, unsigned int>)
		return "unsigned int";
	else if constexpr (std::is_same_v<T, long>)
		return "long";
	else if constexpr (std::is_same_v<T, unsigned long>)
		return "unsigned long";
	else if constexpr (std::is_same_v<T, long long>)
		return "long long";
	else if constexpr (std::is_same_v<T, unsigned long long>)
		return "unsigned long long";
	else if constexpr (std::is_same_v<T, float>)
		return "float";
	else if constexpr (std::is_same_v<T, double>)
		return "double";
	else if constexpr (std::is_same_v<T, long double>)
		return "long double";
	else if constexpr (std::is_same_v<T, void>)
		return "void";
	else
		return nullptr;
}

inline std::string demangle_name(const char* mangled_name) {
#if HAS_CXXABI && !defined(_MSC_VER)
	int status = 0;
	std::unique_ptr<char, decltype(&std::free)> demangled{
		abi::__cxa_demangle(mangled_name, nullptr, nullptr, &status),
		&std::free};
	return (status == 0 && demangled) ? demangled.get() : mangled_name;
#else
	return mangled_name; 
#endif
}

template<typename T>
std::string add_qualifiers(std::string base_name) {
	using TR = std::remove_reference_t<T>;

	if constexpr (std::is_const_v<TR>) {
		base_name += " const";
	}
	if constexpr (std::is_volatile_v<TR>) {
		base_name += " volatile";
	}
	if constexpr (std::is_lvalue_reference_v<T>) {
		base_name += "&";
	} else if constexpr (std::is_rvalue_reference_v<T>) {
		base_name += "&&";
	}
	if constexpr (std::is_pointer_v<TR>) {
		base_name += "*";
	}

	return base_name;
}

// Character classification utilities (migrated from Useful.h)
namespace char_utils {
constexpr bool is_real_char(char c) noexcept {
	constexpr char real_chars[] = "0123456789-eE.";
	for (char valid : real_chars) {
		if (c == valid)
			return true;
	}
	return false;
}

constexpr bool is_int_char(char c) noexcept {
	constexpr char int_chars[] = "0123456789-";
	for (char valid : int_chars) {
		if (c == valid)
			return true;
	}
	return false;
}
} // namespace char_utils
// Public character classification functions
constexpr bool is_real_character(char c) noexcept {
	return char_utils::is_real_char(c);
}

constexpr bool is_int_character(char c) noexcept {
	return char_utils::is_int_char(c);
}

constexpr bool is_whitespace(char c) noexcept {
	return c == ' ' || c == '\n' || c == '\t' || c == '\v' || c == '\b' || c == '\r' || c == '\f' ||
		   c == '\a';
}

// String utilities
inline int first_space_position(const char* s, int max_len) noexcept {
	for (int i = 0; i < max_len; ++i) {
		if (s[i] == ' ')
			return i;
	}
	return -1;
}

// IndexList moved to Simulation folder - see Simulation/IndexList.h
} // namespace detail

// Primary type name function with C++20 concepts
template<typename T, typename... Extras>
	requires Demangable<T>
std::string type_name() {
	// Try compile-time lookup for basic types first
	if constexpr (std::is_fundamental_v<std::remove_cv_t<std::remove_reference_t<T>>>) {
		using BaseType = std::remove_cv_t<std::remove_reference_t<T>>;
		constexpr const char* basic_name = detail::basic_type_name<BaseType>();
		if constexpr (basic_name != nullptr) {
			return detail::add_qualifiers<T>(std::string(basic_name));
		}
	}

	// Fall back to runtime demangling
	using TR = std::remove_reference_t<T>;
	std::string base_name = detail::demangle_name(typeid(TR).name());
	return detail::add_qualifiers<T>(std::move(base_name));
}

// Simplified version that just returns the raw mangled name
template<typename T>
	requires Demangable<T>
constexpr const char* raw_type_name() noexcept {
	return typeid(T).name();
}

// C++20 version with compile-time type name for basic types
template<typename T>
	requires std::is_fundamental_v<T>
consteval const char* constexpr_type_name() noexcept {
	return detail::basic_type_name<T>();
}

// Utility to check if two types have the same name
template<typename T, typename U>
constexpr bool same_type_name() noexcept {
	return std::is_same_v<T, U>;
}

// Pretty print type with template parameters
template<typename T>
std::string pretty_type_name() {
	std::string name = type_name<T>();

	// Add some basic prettification
	if constexpr (std::is_array_v<T>) {
		name = "Array<" + type_name<std::remove_extent_t<T>>() + ">";
	} else if constexpr (std::is_pointer_v<T>) {
		name = type_name<std::remove_pointer_t<T>>() + "*";
	}

	return name;
}

} // namespace ARBD
