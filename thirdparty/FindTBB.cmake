find_path(TBB_INCLUDE_DIR tbb/tbb.h)
find_library(TBB_LIBRARY tbb)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TBB DEFAULT_MSG TBB_LIBRARY TBB_INCLUDE_DIR)

if(TBB_FOUND)
  set(TBB_LIBRARIES ${TBB_LIBRARY})
  set(TBB_INCLUDE_DIRS ${TBB_INCLUDE_DIR})
endif()