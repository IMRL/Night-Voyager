/* 
 * Night-Voyager: Consistent and Efficient Nocturnal Vision-Aided State Estimation in Object Maps
 * Copyright (C) 2025 Night-Voyager Contributors
 * 
 * For commercial use, please contact Tianxiao Gao at <ga0.tianxiao@connect.um.edu.mo>
 * or Mingle Zhao at <zhao.mingle@connect.um.edu.mo>
 * 
 * This file is subject to the terms and conditions outlined in the 'LICENSE' file,
 * which is included as part of this source code package.
 */
#include "utils/Print.h"

namespace night_voyager {

// Need to define the static variable for everything to work
Printer::PrintLevel Printer::current_print_level = PrintLevel::INFO;

void Printer::setPrintLevel(const std::string &level) {
    if (level == "ALL")
        setPrintLevel(PrintLevel::ALL);
    else if (level == "DEBUG")
        setPrintLevel(PrintLevel::DEBUG);
    else if (level == "INFO")
        setPrintLevel(PrintLevel::INFO);
    else if (level == "WARNING")
        setPrintLevel(PrintLevel::WARNING);
    else if (level == "ERROR")
        setPrintLevel(PrintLevel::ERROR);
    else if (level == "SILENT")
        setPrintLevel(PrintLevel::SILENT);
    else {
        std::cout << "Invalid print level requested: " << level << std::endl;
        std::cout << "Valid levels are: ALL, DEBUG, INFO, WARNING, ERROR, SILENT" << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void Printer::setPrintLevel(PrintLevel level) {
    Printer::current_print_level = level;
    std::cout << "Setting printing level to: ";
    switch (current_print_level) {
    case PrintLevel::ALL:
        std::cout << "ALL";
        break;
    case PrintLevel::DEBUG:
        std::cout << "DEBUG";
        break;
    case PrintLevel::INFO:
        std::cout << "INFO";
        break;
    case PrintLevel::WARNING:
        std::cout << "WARNING";
        break;
    case PrintLevel::ERROR:
        std::cout << "ERROR";
        break;
    case PrintLevel::SILENT:
        std::cout << "SILENT";
        break;
    default:
        std::cout << std::endl;
        std::cout << "Invalid print level requested: " << level << std::endl;
        std::cout << "Valid levels are: ALL, DEBUG, INFO, WARNING, ERROR, SILENT" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::cout << std::endl;
}

void Printer::debugPrint(PrintLevel level, const char location[], const char line[], const char *format, ...) {
    // Only print for the current debug level
    if (static_cast<int>(level) < static_cast<int>(Printer::current_print_level)) {
        return;
    }

    // Print the location info first for our debug output
    // Truncate the filename to the max size for the filepath
    if (static_cast<int>(Printer::current_print_level) <= static_cast<int>(Printer::PrintLevel::DEBUG)) {
        std::string path(location);
        std::string base_filename = path.substr(path.find_last_of("/\\") + 1);
        if (base_filename.size() > MAX_FILE_PATH_LEGTH) {
            printf("%s", base_filename.substr(base_filename.size() - MAX_FILE_PATH_LEGTH, base_filename.size()).c_str());
        } else {
            printf("%s", base_filename.c_str());
        }
        printf(":%s ", line);
    }

    // Print the rest of the args
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
}

} // namespace night_voyager