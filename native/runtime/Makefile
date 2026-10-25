CXX := xcrun clang++
BUILD_DIR ?= build
empty :=
space := $(empty) $(empty)
BUILD_TARGET_DIR := $(subst $(space),\$(space),$(BUILD_DIR))
FLAGS := -std=c++17 -O2 -Wall -Wextra -Werror -pthread
SANITIZERS ?= undefined
.PHONY: all test
all: $(BUILD_TARGET_DIR)/libstrata_session.dylib
$(BUILD_TARGET_DIR):
	mkdir -p "$(BUILD_DIR)"
$(BUILD_TARGET_DIR)/libstrata_session.dylib: strata_session.cpp strata_session.h | $(BUILD_TARGET_DIR)
	$(CXX) $(FLAGS) -dynamiclib strata_session.cpp -o "$@"
$(BUILD_TARGET_DIR)/test-session: strata_session.cpp strata_session.h test_session.cpp | $(BUILD_TARGET_DIR)
	$(CXX) $(FLAGS) -fsanitize=$(SANITIZERS) strata_session.cpp test_session.cpp -o "$@"
test: all $(BUILD_TARGET_DIR)/test-session
	"$(BUILD_DIR)/test-session"
