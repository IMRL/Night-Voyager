/* 
 * Night-Voyager: Consistent and Efficient Nocturnal Vision-Aided State Estimation in Object Maps
 * Copyright (C) 2025 Night-Voyager Contributors
 * 
 * For technical issues and support, please contact Tianxiao Gao at <ga0.tianxiao@connect.um.edu.mo>
 * or Mingle Zhao at <zhao.mingle@connect.um.edu.mo>. For commercial use, please contact Prof. Hui Kong at <huikong@um.edu.mo>.
 * 
 * This file is subject to the terms and conditions outlined in the 'LICENSE' file,
 * which is included as part of this source code package.
 */
#ifndef NIGHT_VOYAGER_PRINT_H
#define NIGHT_VOYAGER_PRINT_H

#include <cstdarg>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>

namespace night_voyager {

/**
 * @brief Printer for open_vins that allows for various levels of printing to be done
 *
 * To set the global verbosity level one can do the following:
 * @code{.cpp}
 * night_voyager::Printer::setPrintLevel("WARNING");
 * night_voyager::Printer::setPrintLevel(night_voyager::Printer::PrintLevel::WARNING);
 * @endcode
 */
class Printer {
public:
    /**
     * @brief The different print levels possible
     *
     * - PrintLevel::ALL : All PRINT_XXXX will output to the console
     * - PrintLevel::DEBUG : "DEBUG", "INFO", "WARNING" and "ERROR" will be printed. "ALL" will be silenced
     * - PrintLevel::INFO : "INFO", "WARNING" and "ERROR" will be printed. "ALL" and "DEBUG" will be silenced
     * - PrintLevel::WARNING : "WARNING" and "ERROR" will be printed. "ALL", "DEBUG" and "INFO" will be silenced
     * - PrintLevel::ERROR : Only "ERROR" will be printed. All the rest are silenced
     * - PrintLevel::SILENT : All PRINT_XXXX will be silenced.
     */
    enum PrintLevel { ALL = 0, DEBUG = 1, INFO = 2, WARNING = 3, ERROR = 4, SILENT = 5 };

    /**
     * @brief Set the print level to use for all future printing to stdout.
     * @param level The debug level to use
     */
    static void setPrintLevel(const std::string &level);

    /**
     * @brief Set the print level to use for all future printing to stdout.
     * @param level The debug level to use
     */
    static void setPrintLevel(PrintLevel level);

    /**
     * @brief The print function that prints to stdout.
     * @param level the print level for this print call
     * @param location the location the print was made from
     * @param line the line the print was made from
     * @param format The printf format
     */
    static void debugPrint(PrintLevel level, const char location[], const char line[], const char *format, ...);

    /// The current print level
    static PrintLevel current_print_level;

private:
    /// The max length for the file path.  This is to avoid very long file paths from
    static constexpr uint32_t MAX_FILE_PATH_LEGTH = 30;
};

} /* namespace night_voyager */

/*
 * Converts anything to a string
 */
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

/*
 * The different Types of print levels
 */
#define PRINT_ALL(x...) night_voyager::Printer::debugPrint(night_voyager::Printer::PrintLevel::ALL, __FILE__, TOSTRING(__LINE__), x);
#define PRINT_DEBUG(x...) night_voyager::Printer::debugPrint(night_voyager::Printer::PrintLevel::DEBUG, __FILE__, TOSTRING(__LINE__), x);
#define PRINT_INFO(x...) night_voyager::Printer::debugPrint(night_voyager::Printer::PrintLevel::INFO, __FILE__, TOSTRING(__LINE__), x);
#define PRINT_WARNING(x...) night_voyager::Printer::debugPrint(night_voyager::Printer::PrintLevel::WARNING, __FILE__, TOSTRING(__LINE__), x);
#define PRINT_ERROR(x...) night_voyager::Printer::debugPrint(night_voyager::Printer::PrintLevel::ERROR, __FILE__, TOSTRING(__LINE__), x);

#define RESET "\033[0m"
#define BLACK "\033[30m"                /* Black */
#define RED "\033[31m"                  /* Red */
#define GREEN "\033[32m"                /* Green */
#define YELLOW "\033[33m"               /* Yellow */
#define BLUE "\033[34m"                 /* Blue */
#define MAGENTA "\033[35m"              /* Magenta */
#define CYAN "\033[36m"                 /* Cyan */
#define WHITE "\033[37m"                /* White */
#define REDPURPLE "\033[95m"            /* Red Purple */
#define BOLDBLACK "\033[1m\033[30m"     /* Bold Black */
#define BOLDRED "\033[1m\033[31m"       /* Bold Red */
#define BOLDGREEN "\033[1m\033[32m"     /* Bold Green */
#define BOLDYELLOW "\033[1m\033[33m"    /* Bold Yellow */
#define BOLDBLUE "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"   /* Bold Magenta */
#define BOLDCYAN "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE "\033[1m\033[37m"     /* Bold White */
#define BOLDREDPURPLE "\033[1m\033[95m" /* Bold Red Purple */

#endif /* NIGHT_VOYAGER_PRINT_H */
