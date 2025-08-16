cmake_minimum_required(VERSION 3.15) #设置cmake最低要求
project(
    main      #项目名称
    LANGUAGES CXX C     #编程语言
    DESCRIPTION "模型推理集合"
)

set(CMAKE_CXX_EXTENSIONS OFF)  # 禁用编译器特定扩展 
set(CMAKE_CXX_STANDARD 23) # 设置为C++ 23标准
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug" )
set(CMAKE_CXX_STANDARD_REQUIRED ON) # 强制要求使用C++23
if (MSVC)  
    # 强制使用 UTF-8 编码  
    add_compile_options("/utf-8")  
endif()  
# 四个线程编译
set(CMAKE_BUILD_PARALLEL_LEVEL 8) 



##设置导出dll
set(EXCUTABLEE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
set(BUILD_SHARED_LIBS ON)



# 获取 CMakeLists.txt 文件所在目录的绝对路径
get_filename_component(PROJECT_DIR ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_DIR}/bin)




# 设定项目编写的所有代码文件路径
file(GLOB SRC_FILES 
    CMAKE_CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/include/*.h"
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
    "${PROJECT_SOURCE_DIR}/include/bytetrack/include/*.h"
    "${PROJECT_SOURCE_DIR}/include/bytetrack/src/*.cpp"
)

## 包含目录
set(CMAKE_PREFIX_PATH "D:/vcpkg/installed/x64-windows/;D:/code/Qt/Qt5.14.2/5.14.2/msvc2017_64/;D:/code/onnxruntime/onnxruntime-win-x64-1.18.1")

# 设置 ONNXRUNTIME 库的路径
set(ONNXRUNTIME_DIR "D:/code/onnxruntime/onnxruntime-win-x64-1.18.1")  
#set(ONNXRUNTIME_DIR "D:/vcpkg/installed/x64-windows")  
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_DIR}/include")  
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_DIR}/lib")
include_directories(${PROJECT_DIR}/include/)
include_directories(${PROJECT_DIR}/include/bytetrack/include/)
  
find_package(TBB REQUIRED)
find_package(OpenCV REQUIRED)



# 设置 Qt 库的路径
set(QT5_INCLUDE_DIRS 
    "D:\\code\\Qt\\Qt5.14.2\\5.14.2\\msvc2017_64\\include\\"
    "D:\\code\\Qt\\Qt5.14.2\\5.14.2\\msvc2017_64\\include\\QtWidgets\\"
    "D:\\code\\Qt\\Qt5.14.2\\5.14.2\\msvc2017_64\\include\\QtCore\\"
    "D:\\code\\Qt\\Qt5.14.2\\5.14.2\\msvc2017_64\\include\\QtGui\\"
    "D:\\code\\Qt\\Qt5.14.2\\5.14.2\\msvc2017_64\\include\\QtNetwork\\")
set(QT5_LIBRARY_DIRS 
    "D:\\code\\Qt\\Qt5.14.2\\5.14.2\\msvc2017_64\\lib")
set(QT5_LIBS
    "D:\\code\\Qt\\Qt5.14.2\\5.14.2\\msvc2017_64\\lib\\Qt5Core.lib"
    "D:\\code\\Qt\\Qt5.14.2\\5.14.2\\msvc2017_64\\lib\\Qt5Gui.lib"
    "D:\\code\\Qt\\Qt5.14.2\\5.14.2\\msvc2017_64\\lib\\Qt5Widgets.lib"
    "D:\\code\\Qt\\Qt5.14.2\\5.14.2\\msvc2017_64\\lib\\Qt5Network.lib")

    # 添加包含和库目录  
include_directories(${QT5_INCLUDE_DIRS})  
link_directories(${QT5_LIBRARY_DIRS})  


find_package(Qt5Core REQUIRED)
find_package(Qt5Network REQUIRED)
find_package(Qt5Gui REQUIRED)
find_package(Qt5Widgets REQUIRED)
find_package(spdlog REQUIRED)
# find_package(onnxruntime REQUIRED)
set(CMAKE_AUTOMOC ON)

# 设置 TBB 库的路径
set(TBB_INCLUDE_DIRS "${CMAKE_PREFIX_PATH}/include")
set(TBB_LIBRARIES "${CMAKE_PREFIX_PATH}/lib")

# 添加头文件搜索路径
include_directories(${ONNXRUNTIME_INCLUDE_DIR})  
include_directories(${TBB_INCLUDE_DIRS})
link_directories(${ONNXRUNTIME_LIB_DIR})  



add_executable(${CMAKE_PROJECT_NAME} ${SRC_FILES})

message(STATUS "ONNXRUNTIME_LIB_DIR: ${ONNXRUNTIME_DIR}")
message(STATUS "OpenCV_LIBRARIES: ${OpenCV_DIR}")
message(STATUS "TBB_LIBRARIES: ${TBB_DIR}")
message(STATUS "TBB_LIBRARIES: ${TBB_INCLUDE_DIRS}")

target_link_libraries(${CMAKE_PROJECT_NAME} TBB::tbb)
target_link_libraries(${CMAKE_PROJECT_NAME} ${ONNXRUNTIME_LIB_DIR}/onnxruntime.lib)
target_link_libraries(${CMAKE_PROJECT_NAME} ${OpenCV_LIBRARIES})
# target_link_libraries(${CMAKE_PROJECT_NAME} ${QT5_LIBS})
target_link_libraries(${CMAKE_PROJECT_NAME} Qt5::Core Qt5::Gui Qt5::Widgets Qt5::Network)
target_link_libraries(${CMAKE_PROJECT_NAME} spdlog::spdlog)

install(
    TARGETS ${CMAKE_PROJECT_NAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

#添加动态库
# add_library(sam2 SHARED  "${PROJECT_SOURCE_DIR}/src/sam2.cpp")
# target_link_libraries(sam2 TBB::tbb) 
# target_link_libraries(sam2 ${ONNXRUNTIME_LIB_DIR}/onnxruntime.lib)
# target_link_libraries(sam2 ${OpenCV_LIBRARIES})

# add_library(yolo10 SHARED  "${PROJECT_SOURCE_DIR}/src/yolov10.cpp")
# target_link_libraries(yolo10 TBB::tbb) 
# target_link_libraries(yolo10 ${ONNXRUNTIME_LIB_DIR}/onnxruntime.lib)
# target_link_libraries(yolo10 ${OpenCV_LIBRARIES})


# if(MSVC)
#     set(UV_DEBUG_DLL_FILE ${LIBRARY_OUTPUT_PATH}/Debug/jhdl.dll)
#     set(UV_RELEASE_DLL_FILE ${LIBRARY_OUTPUT_PATH}/Release/jhdl.dll)
#     add_custom_command(TARGET ${PROJECT_NAME}
#     POST_BUILD
#     COMMAND ${CMKAE_COMMAND} -E
#     copy_if_different
#     "$<$<CONFIG:Release>:${UV_RELEASE_DLL_FILE}>"
#     "$<$<CONFIG:Debug>:${UV_DEBUG_DLL_FILE}>"
#     "${EXECUTABLE_OUTPUT_PATH}/$<$<CONFIG:Release>:Release>$<$<CONFIG:Debug>:Debug>"
#     )
# endif()    
