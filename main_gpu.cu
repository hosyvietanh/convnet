#include <cstdlib>
#include <exception>
#include <assert.h>
#include <sstream>
#include <fstream>
#include <iostream>
#include <cstdint>
#include <string>
#include <cmath>
#include <cfloat>
#include <vector>
#include "layers_gpu.h"
#include "mnist_parser.cpp"

using namespace std;

void initializeNet(vector<Layer*> &layers) {
  // Convolutional - Depth, spatialExtent, stride, zeroPadding
  // MaxPooling - SpatialExtent, stride
  // FullyConnected - Depth
  layers.push_back(new Input(1, 32, 32));
  layers.push_back(new ConvolutionalLayer(6, 5, 1, 0, layers.back())); // => 6 * 28 * 28
  layers.push_back(new MaxPoolingLayer(2, 2, layers.back())); // => 6 * 14 * 14 
  layers.push_back(new ConvolutionalLayer(16, 5, 1, 0, layers.back())); // => 16 * 10 * 10
  layers.push_back(new MaxPoolingLayer(2, 2, layers.back())); // => 16 * 5 * 5
  layers.push_back(new ConvolutionalLayer(100, 5, 1, 0, layers.back())); // => 120 * 1 * 1
  layers.push_back(new FullyConnectedLayer(10, layers.back())); // => 10 * 1 * 1
  layers.push_back(new OutputLayer(layers.back()));
}

void train(vector<Layer*> layers) {
  Mnist_Parser parser(".");
  vector<Sample*> input = parser.load_training();  
  int PASS = input.size();
  /*PASS = input.size();*/
  for (int test = 0; test < PASS; test++) {
    int i = test % input.size();
    if (test % 1000 == 0) {
      cout << test << endl;
    }
    ((Input*)layers[0])->setOutput(input[i]->image);
    ((OutputLayer*)layers.back())->setLabel(input[i]->label);
    //cout << test << ' ' << i << endl;
    int iter = 0;
    int M = 1;
    do {
      for (int l = 0; l < layers.size(); l++) {
        layers[l]->feedForward();
      }

      float_t *nextErrors;
      for (int l = layers.size() - 1; l >= 0; l--) {
        layers[l]->backProp(nextErrors);
        nextErrors = layers[l]->_errors;
      }
      //float_t x = ((OutputLayer*)layers.back())->getError();
      iter++;
    } while (((OutputLayer*)layers.back())->getError() > 1e-3 && iter < M);
  }

  /*return;*/
  vector<Sample*> testInput = parser.load_testing();
  int correct = 0;
  for (int i = 0; i < testInput.size(); i++) {
    ((Input*)layers[0])->setOutput(testInput[i]->image);
    ((OutputLayer*)layers.back())->setLabel(testInput[i]->label);
    for (int l = 0; l < layers.size(); l++) {
      layers[l]->feedForward();
    }
    correct += ((OutputLayer*)layers.back())->_label == ((OutputLayer*)layers.back())->getPredict();
    /*cout << ((OutputLayer*)layers.back())->_label << ' ' << ((OutputLayer*)layers.back())->getPredict() << endl;*/
  }
  cout << correct << ' ' << testInput.size() << endl;
}

int main() {
  srand(time(NULL));
  vector<Layer*> layers;
  initializeNet(layers);
  cout << "Before trained" << endl;
  train(layers);
  return 0;
}
