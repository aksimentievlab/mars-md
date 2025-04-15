/*
 * ARBDException class handles the 
 * run-time exception.
 * Han-Yi Chou
 */

#pragma once

#include <string>
#include <cstdarg>
#include <exception>
#include "SignalManager.h"

struct ExceptionType {
public:
    explicit constexpr ExceptionType(int value) noexcept : value{value} {};
    constexpr operator int() const { return value; };
private:
    const int value;
};
// inline constexpr ExceptionType const UnspecifiedError{0};
constexpr ExceptionType const UnspecifiedError{0};
constexpr ExceptionType const NotImplementedError{1};
constexpr ExceptionType const ValueError{2};
constexpr ExceptionType const DivideByZeroError{3};
constexpr ExceptionType const CUDARuntimeError{4};
constexpr ExceptionType const FileIoError{5};
constexpr ExceptionType const FileOpenError{6};

class _ARBDException : public std::exception 
{
    
  private:
    std::string _error;
    std::string sformat(const std::string &fmt, va_list &ap);
    static std::string type_to_str(ExceptionType type) {
	switch (type) {
	case UnspecifiedError:
	    return "Error";
	case NotImplementedError:
	    return "NotImplementedError";
	}
	return "UnkownError";
    }
    
  public:
    _ARBDException(const std::string& location, const ExceptionType type, const std::string &ss, ...);
    // compile with c++11 !!!
    virtual const char* what() const noexcept;
};

// #include "common_macros.h"
#ifdef CUDACC
#define Exception(...) throw _ARBDException(LOCATION, __VA_ARGS__)
#else
#define Exception(EXCPT,...) printf("Runtime CUDA exception at %s: ", LOCATION); \
    printf("%s %s\n", #EXCPT, __VA_ARGS__);
#endif
//use illegal instruction to abort; used in functions defined both in __host__ and __device__
#if 0
#define CudaException(...) \
printf("Run-time exception occurs at %s: ", LOCATION); \
printf(__VA_ARGS__);
//TODO I want to add asm("trap;") but the compiling does not work
asm("trap;");
#endif
