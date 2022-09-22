/* (C) 2022: Hans Georg Schaathun <hg@schaathun.net> *
 * Building on code by Simon Ingebrigtsen, Sondre Westbø Remøy,
 * Einar Leite Austnes, and Simon Nedreberg Runde
 */

#include <thread>
#include <random>
#include <string>
#include <opencv2/opencv.hpp>
#include "Simulator.h"


int main(int, char *argv[]) {

    Simulator simulator;

    int nterms = 16 ;
    int CHI_percent ;
    int CHI, einsteinR, X, Y, sourceSize ;

    int DATAPOINTS_TO_GENERATE = atoi(argv[1]);
    int imgsize = atoi(argv[2]);
    std::string simname = std::string(argv[3]);
    int n_params = atoi(argv[4]);

    simulator = PointMassSimulator(imgsize) ;

    double xyrange = imgsize/2.5 ;

    // Generate dataset:
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> rand_lens_dist(30, 100);
    std::uniform_int_distribution<std::mt19937::result_type> rand_einsteinR(1, imgsize/10);
    std::uniform_int_distribution<std::mt19937::result_type> rand_source_size(1, imgsize/10);
    std::uniform_int_distribution<std::mt19937::result_type> rand_xSlider(-xyrange, xyrange);
    std::uniform_int_distribution<std::mt19937::result_type> rand_ySlider(-xyrange, xyrange);

    std::vector<std::vector<int>> parameters;
    for (int i = 0; i < DATAPOINTS_TO_GENERATE; i++) {
        std::cout << "Iteration " << i << "\n" ;
        if (n_params == 0){
            CHI_percent = 50;
        } else {
            CHI_percent = rand_lens_dist(rng);
        }
        CHI = CHI_percent/100 ;
        einsteinR = rand_einsteinR(rng);
        sourceSize = rand_source_size(rng);
        X = rand_xSlider(rng);
        Y = rand_ySlider(rng);

        std::vector<int> params = {CHI_percent, einsteinR, sourceSize, X, Y };

        if ( (!std::count(parameters.begin(), parameters.end(), params)) ) { // check for duplicate
            simulator.updateAll( X, Y, einsteinR, sourceSize, CHI, nterms );
            std::ostringstream filename;
            filename << einsteinR << "," << sourceSize << "," << X << "," << Y << ".png";
            cv::imwrite( simname + "/images/" + filename.str(), simulator.getDistorted());
            cv::imwrite( simname + "/actual/" + filename.str(), simulator.getActual());
            parameters.push_back( params ) ;
        } else {
            i--;
        }
        if (parameters.size() % 100 == 0){
            std::cout << "Datapoints generated: " << parameters.size() << std::endl;
        }
        std::cout << "Datapoints generated: " << parameters.size() << std::endl;
        std::cout << "Done iteration " << i << "\n" ;
    }
}


