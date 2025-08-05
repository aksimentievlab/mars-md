// Author: Jeff Comer <jcomer2@illinois.edu>
// Refactored for the arbd2/cpp20 branch with on 2025
// Author: Pin-Yi Li <pinyili2@illinois.edu> with Claude 4.0 sonnet

#pragma once

#include "ARBDException.h"
#include "ARBDLogger.h"
#include "FileHandle.h"
#include "IO/DcdWriter.h"
#include "Math/Matrix3.h"
#include "Math/Types.h"
#include "Math/Vector3.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ARBD {

constexpr std::string_view PDB_TEMPLATE_LINE = 
    "ATOM      1  CA  MOL S   1      -6.210  -9.711   3.288  0.00  0.00      ION";

/**
 * @brief Modern C++20 trajectory writer supporting multiple formats
 * 
 * Supports DCD (binary), PDB (text), and TRAJ (custom text) formats.
 * Refactored for arbd2/cpp20 with:
 * - Modern C++20 features
 * - RAII resource management
 * - Exception safety
 * - Integration with ARBD2 systems
 */
class TrajectoryWriter {
public:
    enum class Format { 
        Dcd = 0,   ///< Binary DCD format (CHARMM/NAMD)
        Pdb = 1,   ///< Protein Data Bank format
        Traj = 2   ///< Custom trajectory format
    };

    /**
     * @brief Constructor - creates trajectory writer
     * @param filePrefix Base filename (extension added automatically)
     * @param format Output format
     * @param box Simulation box (Matrix3)
     * @param numAtoms Number of atoms
     * @param timestep Timestep size
     * @param outputPeriod Steps between outputs
     * @throws ARBD::Exception on initialization failure
     */
    TrajectoryWriter(std::string_view filePrefix, 
                    Format format,
                    const Matrix3& box,
                    int numAtoms,
                    float timestep,
                    int outputPeriod)
        : box_(box), 
          numAtoms_(numAtoms), 
          timestep_(timestep), 
          outputPeriod_(outputPeriod),
          format_(format) {
        
        if (numAtoms <= 0) {
            throw Exception(ExceptionType::ValueError,
                           SourceLocation(),
                           "TrajectoryWriter: Invalid number of atoms: %d", numAtoms);
        }

        // Build filename with appropriate extension
        fileName_ = std::string(filePrefix) + "." + getFormatExtension(format);
        
        // Calculate unit cell parameters
        calculateUnitCell();
        
        // Initialize format-specific writer
        initializeWriter();
        
        LOGINFO("TrajectoryWriter: Created {} writer for '{}' with {} atoms",
               getFormatName(format), fileName_, numAtoms_);
    }

    /**
     * @brief Destructor - ensures proper cleanup
     */
    ~TrajectoryWriter() {
        try {
            closeWriter();
        } catch (const std::exception& e) {
            LOGERROR("TrajectoryWriter destructor error: {}", e.what());
        }
    }

    // Non-copyable but movable
    TrajectoryWriter(const TrajectoryWriter&) = delete;
    TrajectoryWriter& operator=(const TrajectoryWriter&) = delete;
    
    TrajectoryWriter(TrajectoryWriter&&) = default;
    TrajectoryWriter& operator=(TrajectoryWriter&&) = default;

    /**
     * @brief Write initial frame and create new file
     * @param positions Array of atom positions
     * @param names Array of atom names (optional for DCD)
     * @param time Current simulation time
     */
    void writeNewFile(std::span<const Vector3> positions,
                     std::span<const std::string> names = {},
                     float time = 0.0f) {
        validateInput(positions, names);
        
        switch (format_) {
            case Format::Dcd:
                writeNewDcd(positions);
                break;
            case Format::Pdb:
                writeNewPdb(positions, names, time);
                break;
            case Format::Traj:
                writeNewTraj(positions, names, time);
                break;
        }
        
        LOGTRACE("TrajectoryWriter: Wrote new file with {} atoms", positions.size());
    }

    /**
     * @brief Write initial frame with atom IDs
     * @param positions Array of atom positions
     * @param names Array of atom names
     * @param ids Array of atom IDs
     * @param time Current simulation time
     */
    void writeNewFile(std::span<const Vector3> positions,
                     std::span<const std::string> names,
                     std::span<const int> ids,
                     float time = 0.0f) {
        validateInput(positions, names, ids);
        
        switch (format_) {
            case Format::Dcd:
                writeNewDcd(positions);
                break;
            case Format::Pdb:
                writeNewPdb(positions, names, time);
                break;
            case Format::Traj:
                writeNewTraj(positions, names, ids, time);
                break;
        }
    }

    /**
     * @brief Append frame to existing file
     * @param positions Array of atom positions
     * @param names Array of atom names (optional for DCD)
     * @param time Current simulation time
     */
    void appendFrame(std::span<const Vector3> positions,
                    std::span<const std::string> names = {},
                    float time = 0.0f) {
        validateInput(positions, names);
        
        switch (format_) {
            case Format::Dcd:
                appendDcd(positions);
                break;
            case Format::Pdb:
                appendPdb(positions, names);
                break;
            case Format::Traj:
                appendTraj(positions, names, time);
                break;
        }
    }

    /**
     * @brief Append frame with atom IDs
     * @param positions Array of atom positions
     * @param names Array of atom names
     * @param ids Array of atom IDs
     * @param time Current simulation time
     */
    void appendFrame(std::span<const Vector3> positions,
                    std::span<const std::string> names,
                    std::span<const int> ids,
                    float time = 0.0f) {
        validateInput(positions, names, ids);
        
        switch (format_) {
            case Format::Dcd:
                appendDcd(positions);
                break;
            case Format::Pdb:
                appendPdb(positions, names);
                break;
            case Format::Traj:
                appendTraj(positions, names, ids, time);
                break;
        }
    }

    /**
     * @brief Get format name as string
     * @param format Format enum
     * @return Format name
     */
    static std::string getFormatName(Format format) {
        switch (format) {
            case Format::Dcd: return "DCD";
            case Format::Pdb: return "PDB";
            case Format::Traj: return "TRAJ";
            default: return "Unknown";
        }
    }

    /**
     * @brief Get file extension for format
     * @param format Format enum
     * @return File extension (without dot)
     */
    static std::string getFormatExtension(Format format) {
        switch (format) {
            case Format::Dcd: return "dcd";
            case Format::Pdb: return "pdb";
            case Format::Traj: return "traj";
            default: return "dat";
        }
    }

    /**
     * @brief Get current filename
     * @return Filename string
     */
    [[nodiscard]] const std::string& getFileName() const noexcept {
        return fileName_;
    }

    /**
     * @brief Get format
     * @return Current format
     */
    [[nodiscard]] Format getFormat() const noexcept {
        return format_;
    }

    /**
     * @brief Get number of atoms
     * @return Number of atoms
     */
    [[nodiscard]] int getNumAtoms() const noexcept {
        return numAtoms_;
    }

private:
    Matrix3 box_;
    int numAtoms_;
    float timestep_;
    int outputPeriod_;
    Format format_;
    std::string fileName_;
    std::array<double, 6> unitCell_{};  // a, b, c, alpha, beta, gamma
    
    // Format-specific writers
    std::unique_ptr<DcdWriter> dcdWriter_;

    /**
     * @brief Calculate unit cell parameters from simulation box
     */
    void calculateUnitCell() {
        float RAD_TO_DEG = 180.0f / (4.0f * std::atan(1.0f));
        
        // Unit cell lengths
        unitCell_[0] = box_.ex().length();  // a
        unitCell_[1] = box_.ey().length();  // b
        unitCell_[2] = box_.ez().length();  // c
        
        // Unit cell angles (in degrees)
        const Vector3 ex = box_.ex();
        const Vector3 ey = box_.ey();
        const Vector3 ez = box_.ez();
        
        // Calculate angles between basis vectors
        double cosAlpha = ey.dot(ez) / (unitCell_[1] * unitCell_[2]);
        double cosBeta = ex.dot(ez) / (unitCell_[0] * unitCell_[2]);
        double cosGamma = ex.dot(ey) / (unitCell_[0] * unitCell_[1]);
        
        // Clamp to valid range [-1, 1] and convert to degrees
        cosAlpha = std::clamp(cosAlpha, -1.0, 1.0);
        cosBeta = std::clamp(cosBeta, -1.0, 1.0);
        cosGamma = std::clamp(cosGamma, -1.0, 1.0);
        
        unitCell_[3] = std::acos(cosAlpha) * RAD_TO_DEG;  // alpha
        unitCell_[4] = std::acos(cosBeta) * RAD_TO_DEG;   // beta
        unitCell_[5] = std::acos(cosGamma) * RAD_TO_DEG;  // gamma
    }

    /**
     * @brief Initialize format-specific writer
     */
    void initializeWriter() {
        if (format_ == Format::Dcd) {
            dcdWriter_ = std::make_unique<DcdWriter>(fileName_);
            dcdWriter_->writeHeader(numAtoms_, 1, 1, outputPeriod_, 0, timestep_, true);
        }
    }

    /**
     * @brief Close format-specific writer
     */
    void closeWriter() noexcept {
        dcdWriter_.reset();
    }

    /**
     * @brief Validate input arrays
     */
    void validateInput(std::span<const Vector3> positions,
                      std::span<const std::string> names = {},
                      std::span<const int> ids = {}) const {
        if (positions.size() != static_cast<size_t>(numAtoms_)) {
            throw Exception(ExceptionType::ValueError,
                           SourceLocation(),
                           "Position array size (%zu) doesn't match expected (%d)",
                           positions.size(), numAtoms_);
        }
        
        if (!names.empty() && names.size() != positions.size()) {
            throw Exception(ExceptionType::ValueError,
                           SourceLocation(),
                           "Names array size (%zu) doesn't match positions (%zu)",
                           names.size(), positions.size());
        }
        
        if (!ids.empty() && ids.size() != positions.size()) {
            throw Exception(ExceptionType::ValueError,
                           SourceLocation(),
                           "IDs array size (%zu) doesn't match positions (%zu)",
                           ids.size(), positions.size());
        }
    }

    // DCD format methods
    void writeNewDcd(std::span<const Vector3> positions) {
        appendDcd(positions);
    }

    void appendDcd(std::span<const Vector3> positions) {
        if (!dcdWriter_) {
            throw Exception(ExceptionType::ValueError,
                           SourceLocation(),
                           "DCD writer not initialized");
        }
        
        std::vector<Vector3> pos_vec(positions.begin(), positions.end());
        std::vector<double> cell_vec(unitCell_.begin(), unitCell_.end());
        dcdWriter_->writeStep(pos_vec, cell_vec);
    }

    // PDB format methods
    void writeNewPdb(std::span<const Vector3> positions,
                    std::span<const std::string> names,
                    float time) {
        FileHandle pdbFile(fileName_.c_str(), "w");
        FILE* fp = pdbFile.get();
        
        writePdbHeader(fp, time);
        writePdbAtoms(fp, positions, names);
        writePdbFooter(fp);
    }

    void appendPdb(std::span<const Vector3> positions,
                  std::span<const std::string> names) {
        FileHandle pdbFile(fileName_.c_str(), "a");
        FILE* fp = pdbFile.get();
        
        writePdbAtoms(fp, positions, names);
        writePdbFooter(fp);
    }

    void writePdbHeader(FILE* fp, float time) const {
        const Vector3 ex = box_.ex();
        const Vector3 ey = box_.ey();
        const Vector3 ez = box_.ez();
        
        std::fprintf(fp, "CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f P 1           1\n",
                    ex.length(), ey.length(), ez.length(),
                    unitCell_[3], unitCell_[4], unitCell_[5]);
        
        std::fprintf(fp, "REMARK   frameTime %.10g ns\n", time);
    }

    void writePdbAtoms(FILE* fp, std::span<const Vector3> positions,
                      std::span<const std::string> names) const {
        for (size_t i = 0; i < positions.size(); ++i) {
            const Vector3& pos = positions[i];
            const std::string& name = names.empty() ? "CA" : names[i];
            
            std::fprintf(fp, "ATOM  %5zu  %-4s MOL S%4zu    %8.3f%8.3f%8.3f  1.00  0.00      ION\n",
                        i + 1, name.c_str(), i + 1, pos.x, pos.y, pos.z);
        }
    }

    void writePdbFooter(FILE* fp) const {
        std::fprintf(fp, "END\n");
    }

    // TRAJ format methods
    void writeNewTraj(std::span<const Vector3> positions,
                     std::span<const std::string> names,
                     float time) {
        FileHandle trajFile(fileName_.c_str(), "w");
        FILE* fp = trajFile.get();
        
        for (size_t i = 0; i < positions.size(); ++i) {
            const Vector3& pos = positions[i];
            const std::string& name = names.empty() ? "ATOM" : names[i];
            
            std::fprintf(fp, "%s %.10g %.10g %.10g %.10g\n",
                        name.c_str(), time, pos.x, pos.y, pos.z);
        }
        std::fprintf(fp, "END\n");
    }

    void writeNewTraj(std::span<const Vector3> positions,
                     std::span<const std::string> names,
                     std::span<const int> ids,
                     float time) {
        FileHandle trajFile(fileName_.c_str(), "w");
        FILE* fp = trajFile.get();
        
        for (size_t i = 0; i < positions.size(); ++i) {
            const Vector3& pos = positions[i];
            const std::string& name = names.empty() ? "ATOM" : names[i];
            const int id = ids.empty() ? static_cast<int>(i) : ids[i];
            
            std::fprintf(fp, "%s %.10g %.10g %.10g %.10g %d\n",
                        name.c_str(), time, pos.x, pos.y, pos.z, id);
        }
        std::fprintf(fp, "END\n");
    }

    void appendTraj(std::span<const Vector3> positions,
                   std::span<const std::string> names,
                   float time) {
        FileHandle trajFile(fileName_.c_str(), "a");
        FILE* fp = trajFile.get();
        
        for (size_t i = 0; i < positions.size(); ++i) {
            const Vector3& pos = positions[i];
            const std::string& name = names.empty() ? "ATOM" : names[i];
            
            std::fprintf(fp, "%s %.10g %.10g %.10g %.10g\n",
                        name.c_str(), time, pos.x, pos.y, pos.z);
        }
        std::fprintf(fp, "END\n");
    }

    void appendTraj(std::span<const Vector3> positions,
                   std::span<const std::string> names,
                   std::span<const int> ids,
                   float time) {
        FileHandle trajFile(fileName_.c_str(), "a");
        FILE* fp = trajFile.get();
        
        for (size_t i = 0; i < positions.size(); ++i) {
            const Vector3& pos = positions[i];
            const std::string& name = names.empty() ? "ATOM" : names[i];
            const int id = ids.empty() ? static_cast<int>(i) : ids[i];
            
            std::fprintf(fp, "%s %.10g %.10g %.10g %.10g %d\n",
                        name.c_str(), time, pos.x, pos.y, pos.z, id);
        }
        std::fprintf(fp, "END\n");
    }
};

} // namespace ARBD