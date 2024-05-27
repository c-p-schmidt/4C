/*----------------------------------------------------------------------*/
/*! \file
\brief Utility to demangle typeid names
\level 0
*/
/*----------------------------------------------------------------------*/

#ifndef FOUR_C_UTILS_DEMANGLE_HPP
#define FOUR_C_UTILS_DEMANGLE_HPP

#include "4C_config.hpp"

#include <cxxabi.h>

#include <memory>
#include <string>

FOUR_C_NAMESPACE_OPEN

namespace CORE::UTILS
{
  /**
   * Utility function which tries to demangle the given @p mangledString. As a fallback, it returns
   * the input string. You should typically pass the result of typeid().name to this function.
   */
  inline std::string TryDemangle(const char* mangledString)
  {
    int status;
    std::unique_ptr<char[], void (*)(void*)> result(
        abi::__cxa_demangle(mangledString, nullptr, nullptr, &status), std::free);

    return (status == 0 && result) ? std::string(result.get()) : std::string(mangledString);
  }
}  // namespace CORE::UTILS

FOUR_C_NAMESPACE_CLOSE

#endif