#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <iterator>
#include <chrono>
#include <sstream>
#include <unistd.h>

#include <executor.hpp>

int main() {
    bool isStdinTerminal = isatty(0);

    if (isStdinTerminal) std::cout << "qmf, checks boolean function for monotonicity.\nFor changing amount of variables, type @n\nFor exiting, type 'exit'" << std::endl;

    Executor* executor = new Executor();

    bool inDebug = false;

    while (true) {
        if (isStdinTerminal) std::cout << "qmf> ";

        std::string input;

        std::getline(std::cin, input);

        if (input.length() == 0 || input[0] == '\n') continue;

        if (input == "exit") {
            break;
        }

        if (input == "#") {
            inDebug = !inDebug;
            std::cout << "Debug mode: " << (inDebug ? "on" : "off") << std::endl;
            continue;
        }
        
        if (input[0] == '@') {
            std::stringstream ss(input.substr(1));
            int new_size = 0;

            ss >> new_size;

            int* alpha = nullptr;

            int pos = 0;

            if (ss.rdbuf()->in_avail() > 0) {
                alpha = new int[new_size];

                for (int i = 0; i < new_size; i++) {
                    ss >> alpha[i]; 
                }
            }

            if (alpha != nullptr) {
                std::cout << "alpha_set = ";

                for (int i = 0; i < new_size; i++) {
                    std::cout << alpha[i] << " ";
                }

                std::cout << "\n";
            }

            executor->changeVectorSpaceSize(new_size, alpha);

            delete alpha;

            std::cout << "n = " << new_size << std::endl;
            continue;
        }

        if (input[0] == '$') {
            std::cout << "Total functions count to iterate: " << executor->getTotalFunctionsCount() << std::endl;

            auto begin = std::chrono::high_resolution_clock::now();

            std::size_t monotonicCount = 0;

            int lastPrintedPercent = 0;

            int percentPrecision = executor->getTotalFunctionsCount() > 65536 ? 1 : 10;

            for (std::size_t i = 0; i < executor->getTotalFunctionsCount(); i++) {
                int percent = (i * 100) / executor->getTotalFunctionsCount();

                if (percent % percentPrecision == 0 && percent != lastPrintedPercent) {
                    std::cout << percent << "% => " << i << "\n";
                    lastPrintedPercent = percent;
                }

                bool isMonotonous = executor->calculateMonotonicity(i);

                if (isMonotonous) {
                    monotonicCount++;
                }
            }

            auto end = std::chrono::high_resolution_clock::now();

            auto timeSpent = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

            std::cout << "Monotonic functions count for given vector space: " << monotonicCount << std::endl;
            std::cout << "Time spent: " << timeSpent << " ms" << std::endl;

            continue;
        }

        std::size_t functionNumber = std::stoul(input);

        bool isMonotonous = executor->calculateMonotonicity(functionNumber, inDebug);

        std::cout << (isMonotonous ? "Yes" : "No") << std::endl;
    }

    delete executor;

    return 0;
}
    