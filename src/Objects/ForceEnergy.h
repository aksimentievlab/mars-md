#pragma once

#include "Backend/Resource.h" 
#include "ARBDLogger.h"
#include "ARBDException.h"
#include <string>
#include <sstream>

namespace ARBD {

/**
 * @brief A simple class to store force and energy values together
 * 
 * This class represents a force-energy pair commonly used in molecular
 * dynamics simulations. Migrated from Useful.h to the modern Types system.
 */
class ForceEnergy {
public:
    HOST DEVICE ForceEnergy() : f(0.0f), e(0.0f) {}
    
    HOST DEVICE explicit ForceEnergy(float energy) : f(energy), e(energy) {}
    
    HOST DEVICE ForceEnergy(float force, float energy) : f(force), e(energy) {}
    
    HOST DEVICE ForceEnergy(const ForceEnergy& other) : f(other.f), e(other.e) {}
    
    HOST DEVICE ForceEnergy& operator=(const ForceEnergy& other) {
        if (this != &other) {
            f = other.f;
            e = other.e;
        }
        return *this;
    }
    
    // Arithmetic operations
    HOST DEVICE ForceEnergy operator+(const ForceEnergy& other) const {
        return ForceEnergy(f + other.f, e + other.e);
    }
    
    HOST DEVICE ForceEnergy operator-(const ForceEnergy& other) const {
        return ForceEnergy(f - other.f, e - other.e);
    }
    
    HOST DEVICE ForceEnergy operator*(float scalar) const {
        return ForceEnergy(f * scalar, e * scalar);
    }
    
    HOST DEVICE ForceEnergy operator/(float scalar) const {
        return ForceEnergy(f / scalar, e / scalar);
    }
    
    HOST DEVICE ForceEnergy& operator+=(const ForceEnergy& other) {
        f += other.f;
        e += other.e;
        return *this;
    }
    
    HOST DEVICE ForceEnergy& operator-=(const ForceEnergy& other) {
        f -= other.f;
        e -= other.e;
        return *this;
    }
    
    HOST DEVICE ForceEnergy& operator*=(float scalar) {
        f *= scalar;
        e *= scalar;
        return *this;
    }
    
    HOST DEVICE ForceEnergy& operator/=(float scalar) {
        f /= scalar;
        e /= scalar;
        return *this;
    }
    
    // Comparison operators
    HOST DEVICE bool operator==(const ForceEnergy& other) const {
        return f == other.f && e == other.e;
    }
    
    HOST DEVICE bool operator!=(const ForceEnergy& other) const {
        return !(*this == other);
    }
    
    // Accessors
    HOST DEVICE float force() const { return f; }
    HOST DEVICE float energy() const { return e; }
    HOST DEVICE void set_force(float force) { f = force; }
    HOST DEVICE void set_energy(float energy) { e = energy; }
    
    // String representation (host-only)
    std::string to_string() const {
        std::ostringstream oss;
        oss << "ForceEnergy(f=" << f << ", e=" << e << ")";
        return oss.str();
    }
    
    // Reset to zero
    HOST DEVICE void reset() {
        f = 0.0f;
        e = 0.0f;
    }
    
    // Magnitude operations
    HOST DEVICE float force_magnitude() const { return f; }
    HOST DEVICE float energy_magnitude() const { return e; }
    
public:
    float f; ///< Force component
    float e; ///< Energy component
};

// Free function operators
HOST DEVICE inline ForceEnergy operator*(float scalar, const ForceEnergy& fe) {
    return fe * scalar;
}

} // namespace ARBD 