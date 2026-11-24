CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra
BUILD = build
GATE = $(BUILD)/gate_cli

.PHONY: all test
all: $(GATE)

$(BUILD):
	mkdir -p $@

$(GATE): gate_cli.cpp upstream/judge/judge.hpp upstream/safety_gate/safety_gate.hpp upstream/safety_screen/safety_screen.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@

test: $(GATE)
	$(CXX) $(CXXFLAGS) upstream/safety_screen/test_screen.cpp -o $(BUILD)/test_screen
	$(CXX) $(CXXFLAGS) upstream/safety_gate/test_gate.cpp -o $(BUILD)/test_gate
	$(CXX) $(CXXFLAGS) upstream/judge/test_judge.cpp -o $(BUILD)/test_judge
	./$(BUILD)/test_screen upstream/safety_screen/rules.tsv
	./$(BUILD)/test_gate upstream/safety_screen/rules.tsv
	printf '%s\n' '{behavior}' '{generation}' > $(BUILD)/test_prompt.txt
	cd upstream/judge && ../../$(BUILD)/test_judge models/request.kzj models/reply.kzj ../../$(BUILD)/test_prompt.txt models/request.kze models/reply.kze
