#pragma once
#include "ARBDLogger.h"
#include "FileHandle.h"
#include <sstream>
#include <string>
#include <vector>

namespace ARBD {

class TabulatedReader {
  public:
	explicit TabulatedReader(std::string_view fileName) {
		readFile(fileName);
	}

	const std::vector<float>& getX() const {
		return X;
	}
	const std::vector<float>& getY() const {
		return Y;
	}
	size_t size() const {
		return X.size();
	}

  private:
	std::vector<float> X;
	std::vector<float> Y;

	void readFile(std::string_view fileName) {
		FileHandle file(fileName.data(), "r");
		FILE* fp = file.get();

		char* line = nullptr;
		size_t len = 0;
		ssize_t read;
		size_t lineNumber = 0;

		while ((read = getline(&line, &len, fp)) != -1) {
			++lineNumber;
			std::string lineStr(line, read);

			// Skip empty lines and comments
			if (lineStr.empty() || lineStr[0] == '#') {
				continue;
			}

			std::istringstream iss(lineStr);
			float x, y;

			if (iss >> x >> y) {
				X.push_back(x);
				Y.push_back(y);
			} else {
				LOGWARN("TabulatedReader: Failed to parse line {}: {}", lineNumber, lineStr);
			}
		}

		if (line) {
			free(line);
		}
	}
};

} // namespace ARBD
