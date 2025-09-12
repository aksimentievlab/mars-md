#include "Configuration.h"

namespace ARBD {

Configuration::Configuration(const char* config_file) {
	Reader reader(config_file);
	params_ = reader.getParameters();
}

} // namespace ARBD
