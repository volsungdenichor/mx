add_executable(mx main.cpp)

add_custom_command(TARGET mx POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/images
    $<TARGET_FILE_DIR:mx>/images
    COMMENT "Copying images directory to build directory"
)

# Define the images path as a compile definition
target_compile_definitions(mx PRIVATE
    IMAGES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/images"
)
