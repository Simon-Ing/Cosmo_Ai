#include <thread>
#include <random>
#include <string>
#include <opencv2/opencv.hpp>
#include "Simulator.h"


int main(int, char *argv[]) {

    Simulator simulator;

    int DATAPOINTS_TO_GENERATE = atoi(argv[1]);
    simulator.size = atoi(argv[2]);
    simulator.name = std::string(argv[3]);
    int n_params = atoi(argv[4]);

    // Generate dataset:
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> rand_lens_dist(30, 100);
    std::uniform_int_distribution<std::mt19937::result_type> rand_einsteinR(1, simulator.size/10);
    std::uniform_int_distribution<std::mt19937::result_type> rand_source_size(1, simulator.size/10);
    std::uniform_int_distribution<std::mt19937::result_type> rand_xSlider(100, simulator.size-100);
    std::uniform_int_distribution<std::mt19937::result_type> rand_ySlider(100, simulator.size-100);

    std::vector<std::vector<int>> parameters;
    for (int i = 0; i < DATAPOINTS_TO_GENERATE; i++) {
        if (n_params == 4){
            simulator.CHI_percent = 50;
        }
        else{
            simulator.CHI_percent = rand_lens_dist(rng);
        }
        simulator.einsteinR = rand_einsteinR(rng);
        simulator.sourceSize = rand_source_size(rng);
        simulator.xPosSlider = rand_xSlider(rng);
        simulator.yPosSlider = rand_ySlider(rng);

        std::vector<int> params = {simulator.CHI_percent, simulator.einsteinR, simulator.sourceSize, simulator.xPosSlider, simulator.yPosSlider };

        if ( (!std::count(parameters.begin(), parameters.end(), params)) ) { // check for duplicate
            simulator.update();
            simulator.writeToPngFiles(n_params);
            parameters.push_back({ simulator.CHI_percent, simulator.einsteinR, simulator.sourceSize, simulator.xPosSlider, simulator.yPosSlider });
        }
        else{
            i--;
        }
        if (parameters.size() % (DATAPOINTS_TO_GENERATE/10) == 0){
            std::cout << " Datapoints generated: " << parameters.size() << std::endl;
        }
    }
}