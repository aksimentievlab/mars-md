///////////////////////////////////////////////////////////////////////
// Configuration file reader
// Author: Jeff Comer <jcomer2@illinois.edu>
// Refactored for the arbd2/cpp20 branch with on 2025
// Author: Pin-Yi Li <pinyili2@illinois.edu>
#pragma once
#include "ARBDException.h"
#include "ARBDLogger.h"
#include "FileHandle.h"
#include "Types/Types.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <ranges>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ARBD {

/**
 * @brief Modern C++20 configuration file reader for ARBD2
 *
 * Features:
 * - Exception-safe file handling
 * - Modern C++20 ranges and views
 * - Efficient parameter lookup with hash maps
 * - std::string_view for zero-copy string operations
 * - Proper RAII resource management
 * - Integration with ARBD2 logging and exception systems
 * @example Basic Usage:
 * ```cpp
 * #include "IO/Reader.h"
 *
 * // Create a Reader for a configuration file
 * ARBD::Reader reader("config.txt");
 *
 * // Access the number of parameter lines
 * size_t num_params = reader.length();
 *
 * // Iterate over all parameters
 * for (size_t i = 0; i < reader.length(); ++i) {
 *     std::string param = reader.parameter(i);
 *     std::string value = reader.value(i);
 *     std::cout << param << " = " << value << std::endl;
 * }
 *
 * // Lookup a parameter by name
 * if (reader.has("temperature")) {
 *     std::string temp = reader.get("temperature");
 *     std::cout << "Temperature: " << temp << std::endl;
 * }
 *
 * // Parse a boolean parameter
 * bool enabled = ARBD::Reader::parseBool(reader.get("enabled"));
 * ```
 */

class Reader {
  public:
	using ParameterMap = std::unordered_multimap<std::string, std::string>;
	using ParameterPair = std::pair<std::string, std::string>;

	/**
	 * @brief Constructor - reads and parses configuration file
	 * @param fileName Path to configuration file
	 * @throws ARBD::Exception on file access or parsing errors
	 */
	explicit Reader(std::string_view fileName) : fileName_(fileName) {
		try {
			readAndParseFile();
			LOGINFO("Reader: Successfully parsed {} parameter lines from '{}'",
					parameters_.size(),
					fileName_);
		} catch (const std::exception& e) {
			throw Exception(ExceptionType::FileIoError,
							SourceLocation(),
							"Failed to read configuration file '%s': %s",
							fileName_.c_str(),
							e.what());
		}
	}

	/**
	 * @brief Default constructor for empty reader
	 */
	Reader() = default;

	// Rule of 5 - movable but not copyable for efficiency
	Reader(const Reader&) = delete;
	Reader& operator=(const Reader&) = delete;

	Reader(Reader&&) = default;
	Reader& operator=(Reader&&) = default;

	~Reader() = default;

	/**
	 * @brief Get number of parameter lines (excluding comments/blanks)
	 * @return Total number of valid parameter entries
	 */
	[[nodiscard]] size_t length() const noexcept {
		return parameters_.size();
	}

	/**
	 * @brief Get parameter name by index
	 * @param i Index (supports negative indexing and wrapping)
	 * @return Parameter name
	 */
	[[nodiscard]] std::string getParameter(int i) const {
		if (parameters_.empty())
			return {};

		const size_t index = normalizeIndex(i);
		auto it = parameters_.begin();
		std::advance(it, index);
		return it->first;
	}

	/**
	 * @brief Get parameter value by index
	 * @param i Index (supports negative indexing and wrapping)
	 * @return Parameter value
	 */
	[[nodiscard]] std::string getValue(int i) const {
		if (parameters_.empty())
			return {};

		const size_t index = normalizeIndex(i);
		auto it = parameters_.begin();
		std::advance(it, index);
		return it->second;
	}

	/**
	 * @brief Get parameter-value pair by index
	 * @param i Index
	 * @return Pair of {parameter, value}
	 */
	[[nodiscard]] ParameterPair getPair(int i) const {
		if (parameters_.empty())
			return {};

		const size_t index = normalizeIndex(i);
		auto it = parameters_.begin();
		std::advance(it, index);
		return *it;
	}

	/**
	 * @brief Find first value for a given parameter
	 * @param param Parameter name to search for
	 * @return Value string, or empty string if not found
	 */
	[[nodiscard]] std::string findValue(std::string_view param) const {
		auto it = parameterMap_.find(std::string(param));
		return (it != parameterMap_.end()) ? it->second : std::string{};
	}

	/**
	 * @brief Find all values for a given parameter
	 * @param param Parameter name to search for
	 * @return Vector of all matching values
	 */
	[[nodiscard]] std::vector<std::string> findAllValues(std::string_view param) const {
		std::vector<std::string> results;
		auto range = parameterMap_.equal_range(std::string(param));

		for (auto it = range.first; it != range.second; ++it) {
			results.push_back(it->second);
		}

		return results;
	}

	/**
	 * @brief Count occurrences of a parameter
	 * @param param Parameter name to count
	 * @return Number of occurrences
	 */
	[[nodiscard]] size_t countParameter(std::string_view param) const {
		return parameterMap_.count(std::string(param));
	}

	/**
	 * @brief Check if a parameter exists
	 * @param param Parameter name to check
	 * @return true if parameter exists
	 */
	[[nodiscard]] bool hasParameter(std::string_view param) const {
		return parameterMap_.contains(std::string(param));
	}

	/**
	 * @brief Get all unique parameter names
	 * @return Vector of unique parameter names
	 */
	[[nodiscard]] std::vector<std::string> getParameterNames() const {
		std::vector<std::string> names;

		for (const auto& [param, value] : parameterMap_) {
			if (std::find(names.begin(), names.end(), param) == names.end()) {
				names.push_back(param);
			}
		}

		std::sort(names.begin(), names.end());
		return names;
	}

	/**
	 * @brief Convert to string representation
	 * @return Multi-line string with all parameters
	 */
	[[nodiscard]] std::string toString() const {
		std::ostringstream oss;

		for (const auto& [param, value] : parameters_) {
			oss << param << " " << value << "\n";
		}

		return oss.str();
	}

	/**
	 * @brief Get iterator to beginning of parameters
	 */
	[[nodiscard]] auto begin() const {
		return parameters_.begin();
	}

	/**
	 * @brief Get iterator to end of parameters
	 */
	[[nodiscard]] auto end() const {
		return parameters_.end();
	}

	/**
	 * @brief Get all parameters as a view
	 * @return Span view of parameter-value pairs
	 */
	[[nodiscard]] std::span<const ParameterPair> getParameters() const {
		return std::span(parameters_);
	}

	/**
	 * @brief Get filename that was read
	 * @return Original filename
	 */
	[[nodiscard]] const std::string& getFileName() const noexcept {
		return fileName_;
	}

	/**
	 * @brief Static method to count valid parameter lines in a file
	 * @param fileName Path to file
	 * @return Number of valid (non-comment, non-blank) lines
	 * @throws ARBD::Exception on file access error
	 */
	static size_t countParameterLines(std::string_view fileName) {
		FileHandle file(fileName.data(), "r");
		FILE* fp = file.get();

		size_t count = 0;
		char* line = nullptr;
		size_t len = 0;
		ssize_t read;

		while ((read = getline(&line, &len, fp)) != -1) {
			std::string lineStr(line, read);
			if (isValidParameterLine(lineStr)) {
				++count;
			}
		}

		if (line) {
			free(line);
		}

		return count;
	}

	/**
	 * @brief Parse a value as a specific type
	 * @tparam T Target type (int, float, double, bool, etc.)
	 * @param param Parameter name to look up
	 * @param defaultValue Default value if parameter not found or conversion fails
	 * @return Parsed value or default
	 */
	template<typename T>
	[[nodiscard]] T parseValue(std::string_view param, const T& defaultValue = T{}) const {
		const std::string valueStr = findValue(param);
		if (valueStr.empty()) {
			return defaultValue;
		}

		try {
			if constexpr (std::is_same_v<T, bool>) {
				return parseBool(valueStr);
			} else if constexpr (std::is_integral_v<T>) {
				return static_cast<T>(std::stoll(valueStr));
			} else if constexpr (std::is_floating_point_v<T>) {
				return static_cast<T>(std::stod(valueStr));
			} else if constexpr (std::is_same_v<T, std::string>) {
				return valueStr;
			} else {
				static_assert(!std::is_same_v<T, T>, "Unsupported type for parseValue");
			}
		} catch (const std::exception& e) {
			LOGWARN("Reader: Failed to parse '{}' as {} for parameter '{}', using default",
					valueStr,
					type_name<T>(),
					param);
			return defaultValue;
		}
	}

	/**
	 * @brief Parse a Vector3 from three consecutive values
	 * @param param Parameter name
	 * @param defaultValue Default Vector3 if parsing fails
	 * @return Parsed Vector3
	 */
	[[nodiscard]] Vector3 parseVector3(std::string_view param,
									   const Vector3& defaultValue = Vector3{}) const {
		const std::string valueStr = findValue(param);
		if (valueStr.empty()) {
			return defaultValue;
		}

		std::istringstream iss(valueStr);
		std::vector<std::string> tokens;
		std::string token;

		while (iss >> token) {
			tokens.push_back(token);
		}

		if (tokens.size() >= 3) {
			try {
				return Vector3(std::stof(tokens[0]), std::stof(tokens[1]), std::stof(tokens[2]));
			} catch (const std::exception& e) {
				LOGWARN(
					"Reader: Failed to parse Vector3 from '{}' for parameter '{}', using default",
					valueStr,
					param);
			}
		}

		return defaultValue;
	}

  private:
	std::string fileName_;
	std::vector<ParameterPair> parameters_; // Maintains order
	ParameterMap parameterMap_;				// For fast lookup

	/**
	 * @brief Read and parse the configuration file
	 */
	void readAndParseFile() {
		FileHandle file(fileName_.c_str(), "r");
		FILE* fp = file.get();

		char* line = nullptr;
		size_t len = 0;
		ssize_t read;
		size_t lineNumber = 0;

		while ((read = getline(&line, &len, fp)) != -1) {
			++lineNumber;
			std::string lineStr(line, read);

			if (!isValidParameterLine(lineStr)) {
				continue;
			}

			try {
				auto [param, value] = parseLine(lineStr);
				if (!param.empty()) {
					parameters_.emplace_back(param, value);
					parameterMap_.emplace(param, value);
				}
			} catch (const std::exception& e) {
				LOGWARN("Reader: Skipping malformed line {} in '{}': {}",
						lineNumber,
						fileName_,
						lineStr);
			}
		}

		if (line) {
			free(line);
		}
	}

	/**
	 * @brief Check if a line contains valid parameter data
	 * @param line Line to check
	 * @return true if line should be processed
	 */
	static bool isValidParameterLine(const std::string& line) {
		// Find first non-whitespace character
		auto first_non_space = std::ranges::find_if_not(line, [](char c) {
			return std::isspace(static_cast<unsigned char>(c));
		});

		// Skip empty lines and comments
		return first_non_space != line.end() && *first_non_space != '#';
	}

	/**
	 * @brief Parse a single line into parameter and value
	 * @param line Line to parse
	 * @return Pair of {parameter, value}
	 */
	static ParameterPair parseLine(const std::string& line) {
		std::istringstream iss(line);
		std::vector<std::string> tokens;
		std::string token;

		// Split line into tokens
		while (iss >> token) {
			tokens.push_back(token);
		}

		if (tokens.empty()) {
			return {};
		}

		// First token is parameter name
		std::string param = tokens[0];

		// Remaining tokens form the value (space-separated)
		std::string value;
		for (size_t i = 1; i < tokens.size(); ++i) {
			if (i > 1)
				value += " ";
			value += tokens[i];
		}

		return {std::move(param), std::move(value)};
	}

	/**
	 * @brief Normalize index to handle negative values and wrapping
	 * @param i Input index
	 * @return Normalized index within bounds
	 */
	[[nodiscard]] size_t normalizeIndex(int i) const noexcept {
		if (parameters_.empty())
			return 0;

		const int size = static_cast<int>(parameters_.size());

		// Handle negative indices
		while (i < 0) {
			i += size;
		}

		// Handle wrapping
		return static_cast<size_t>(i % size);
	}

	/**
	 * @brief Parse boolean value from string
	 * @param str String to parse
	 * @return Boolean value
	 */
	static bool parseBool(const std::string& str) {
		std::string lower = str;
		std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

		if (lower == "true" || lower == "yes" || lower == "on" || lower == "1") {
			return true;
		} else if (lower == "false" || lower == "no" || lower == "off" || lower == "0") {
			return false;
		} else {
			throw Exception(ExceptionType::ValueError,
							SourceLocation(),
							"Invalid boolean value: %s",
							str.c_str());
		}
	}
};
} // namespace ARBD
