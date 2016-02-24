#include <iostream>
#include <vector>
#include <stdio.h>
#include "layers_gpu.h"
#include <boost/random.hpp>

using namespace std;

#define ALPHA 0.05
#define LAMBDA 0.01
float_t Layer::_alpha;
float_t Layer::_lambda;
const string SOLVE_MODE = "GPU";

inline int uniform_rand(int min, int max) {
  static boost::mt19937 gen(0);
  boost::uniform_smallint<> dst(min, max);
  return dst(gen);
}

template<typename T>
inline T uniform_rand(T min, T max) {
  static boost::mt19937 gen(0);
  boost::uniform_real<T> dst(min, max);
  return dst(gen);
}

template<typename Iter>
void uniform_rand(Iter begin, Iter end, float_t min, float_t max) {
  for (Iter it = begin; it != end; ++it)
    *it = uniform_rand(min, max);
}
float_t activationFunction(float_t v) {
  return 1.0 / (1.0 + exp(-v));
}

float_t activationDerivativeFunction(float_t v) {
  return v * (1.0 - v);
}
__device__ float_t activationFunctionDevice(float_t v) {
  return 1.0 / (1.0 + exp(-v));
}

__device__ float_t activationDerivativeFunctionDevice(float_t v) {
  return v * (1.0 - v);
}

__device__ int getIndexDevice(int depth, int height, int width, int H, int W) {
  return depth * H * W + height * W + width;
}

int getDimension(int total, int each) {
  return (total - 1)/each + 1;
}

Layer::Layer(int depth, int height, int width, int spatialExtent, int stride,
    int zeroPadding, Layer *prev) {
  _depth = depth;
  _height = height;
  _width = width;
  _spatialExtent = spatialExtent;
  _stride = stride;
  _zeroPadding = zeroPadding;
  _prev = prev;

  if (prev == NULL) {
    _weightSize = 0;
    _errorSize = 0;
  } else {
    _weightSize = spatialExtent * spatialExtent * prev->_depth * _depth;
    _errorSize = _prev->_depth * _prev->_height * _prev->_width;
  }

  _outputSize = _depth * _height * _width;

  if (SOLVE_MODE == "GPU") {
    cudaMalloc(&_weight, _weightSize * sizeof(float_t));
    cudaMalloc(&_deltaW, _weightSize * sizeof(float_t));
    cudaMalloc(&_bias, _outputSize * sizeof(float_t));
    cudaMalloc(&_errors, _errorSize * sizeof(float_t));
    cudaMalloc(&_output, _outputSize * sizeof(float_t));
  } else {
    _weight = (float_t*)malloc(_weightSize * sizeof(float_t));
    _deltaW = (float_t*)malloc(_weightSize * sizeof(float_t));
    _bias = (float_t*)malloc(_outputSize * sizeof(float_t));
    _errors = (float_t*)malloc(_errorSize * sizeof(float_t));
    _output = (float_t*)malloc(_outputSize * sizeof(float_t));
  }
}

int Layer::getIndex(int d, int h, int w) {
  return d * (_height * _width) + h * _width + w;
}

void Layer::setLearning(float_t alpha, float_t lambda) {
  _alpha = alpha;
  _lambda = lambda;
}

Input::Input(int depth, int height, int width): Layer(depth, height, width, 0, 0, 0, NULL) {}

void Input::setOutput(vector<float_t> &output) {
  float_t temp[_outputSize];
  for (int i = 0; i < _outputSize; i++) {
    temp[i] = output[i];
  }
  if (SOLVE_MODE == "GPU") {
    cudaMemcpy(_output, temp, _outputSize * sizeof(float_t), cudaMemcpyHostToDevice);
  } else {
    memcpy(_output, temp, _outputSize * sizeof(float_t));
  }
}

void Input::feedForward(){}
void Input::backProp(float_t *nextErrors){}

__global__ void convoFeedForward(float_t *prev, float_t *weight, float_t *bias, float_t *output, int outDepth, int outHeight, int outWidth,
    int inDepth, int inHeight, int inWidth, int F, int S) {
  // F = spatialExtent, S = stride
  int depth = blockDim.z * blockIdx.z + threadIdx.z;
  int height = blockDim.y * blockIdx.y + threadIdx.y;
  int width = blockDim.x * blockIdx.x + threadIdx.x;
  if (depth >= outDepth || height >= outHeight || width >= outWidth) {
    return;
  }
  int startH = height * S;
  int startW = width * S;

  float_t result = 0;
  int outIndex = getIndexDevice(depth, height, width, outHeight, outWidth);
  for (int in = 0; in < inDepth; in++) {
    for (int i = 0; i < F; i++) {
      for (int j = 0; j < F; j++) {
        int index = getIndexDevice(in, startH + i, startW + j, inHeight, inWidth);// row startH + i, col startW + j
        float_t input = prev[index];
        int indexWeight = in * outDepth * F * F + depth * F * F + i * F + j;
        result += input * weight[indexWeight];
      }
    }
  }
  output[outIndex] = activationFunctionDevice(result + bias[outIndex]);
  /*printf("Convo feedforward %d %.2f\n", outIndex, output[outIndex]);*/
}

__global__ void convoBackpropErrors(float_t *prev, float_t *weight, float_t *deltaW, float_t *bias, float_t *output, float_t *errors, float_t *nextErrors,
    int outDepth, int outHeight, int outWidth, int inDepth, int inHeight, int inWidth, int F, int S, float_t alpha, float_t lambda) {

  int inD = blockDim.z * blockIdx.z + threadIdx.z;
  int inH = blockDim.y * blockIdx.y + threadIdx.y;
  int inW = blockDim.x * blockIdx.x + threadIdx.x;
  if (inD >= inDepth || inH >= inHeight || inW >= inWidth) {
      return;
  }
  int index = getIndexDevice(inD, inH, inW, inHeight, inWidth);
  float_t derivativeInput = activationDerivativeFunctionDevice(prev[index]);
  float_t error = 0;
  for (int out = 0; out < outDepth; out++) {
    for (int y = 0; y < F; y+=S) {
      for (int x = 0; x < F; x+=S) {
        int outH = (inH - y)/S;
        int outW = (inW - x)/S;
        if (outH < 0 || outH >= outHeight || outW < 0 || outW >= outWidth) {
          continue;
        }
        int outIndex = getIndexDevice(out, outH, outW, outHeight, outWidth);
        int weightIndex = inD * outDepth * F * F + out * F * F + y * F + x;
        error += nextErrors[outIndex] * weight[weightIndex] * derivativeInput;
      }
    }
  }
  errors[index] = error;
}

__global__ void convoBackpropWeight(float_t *prev, float_t *weight, float_t *deltaW, float_t *bias, float_t *output, float_t *errors, float_t *nextErrors,
    int outDepth, int outHeight, int outWidth, int inDepth, int inHeight, int inWidth, int F, int S, float_t alpha, float_t lambda) {

  int inD = blockDim.z * blockIdx.z + threadIdx.z;
  int outD = blockDim.y * blockIdx.y + threadIdx.y;
  int FF = blockDim.x * blockIdx.x + threadIdx.x;
  if (outD >= outDepth || inD >= inDepth || FF >= F * F) {
    return;
  }
  int target = inD * outDepth * F * F + outD * F * F + FF;
  int y = FF/F, x = FF%F;
  float_t delta = deltaW[target];
  for (int h = 0; h < outHeight; h++) {
    for (int w = 0; w < outWidth; w++) {
      int outIndex = getIndexDevice(outD, h, w, outHeight, outWidth);
      int inH = h * S + y;
      int inW = w * S + x;
      float_t input = prev[getIndexDevice(inD, inH, inW, inHeight, inWidth)];
      delta = alpha * input * nextErrors[outIndex] + lambda * delta;
      weight[target] -= delta;
      if (inD == 0 && FF == 0) {
        bias[outIndex] -= alpha * nextErrors[outIndex];
      }
    }
  }
  deltaW[target] = delta;
}

ConvolutionalLayer::ConvolutionalLayer(int depth, int spatialExtent, int stride, int zeroPadding, Layer *prev):
  Layer(depth, (prev->_height - spatialExtent + 2 * zeroPadding)/stride + 1,
      (prev->_width - spatialExtent + 2 * zeroPadding)/stride + 1,
      spatialExtent, stride, zeroPadding, prev) {

    initWeight();
  }

void ConvolutionalLayer::feedForward() {
  if (SOLVE_MODE == "GPU") {
    forward_gpu();
  } else {
    forward_cpu();
  }
}

void ConvolutionalLayer::forward_cpu() {
  // CPU feedforward
  for (int out = 0; out < _depth; out++) {
    for (int h = 0; h < _height; h++) {
      for (int w = 0; w < _width; w++) {
        float_t result = 0;
        for (int in = 0; in < _prev->_depth; in++) {
          result += sumWeight(in, out, h, w);
        }
        int index = getIndex(out, h, w);
        _output[index] = activationFunction(result + _bias[index]);
      } 
    }
  }
}

void ConvolutionalLayer::forward_gpu() {
  int blockX = min(16, _width);
  int blockY = min(16, _height);
  int blockZ = min(64, min(_depth, 1024 / (blockX * blockY)));
  dim3 dimBlock(blockX, blockY, blockZ);
  dim3 dimGrid(getDimension(_width, dimBlock.x), getDimension(_height, dimBlock.y), getDimension(_depth, dimBlock.z));
  convoFeedForward<<<dimGrid, dimBlock>>>(_prev->_output, _weight, _bias, _output, _depth, _height, _width,
      _prev->_depth, _prev->_height, _prev->_width, _spatialExtent, _stride);
}

void ConvolutionalLayer::backProp(float_t *nextErrors) {
  if (SOLVE_MODE == "GPU") {
    backProp_gpu(nextErrors);  
  } else {
    backProp_cpu(nextErrors);
  }
}

void ConvolutionalLayer::backProp_gpu(float_t *nextErrors) {
  cudaMemset(_errors, 0, _errorSize * sizeof(float_t));
  int blockX = min(16, _prev->_width);
  int blockY = min(16, _prev->_height);
  int blockZ = min(64, min(_prev->_depth, 1024 / (blockX * blockY)));
  dim3 dimBlock(blockX, blockY, blockZ);
  dim3 dimGrid(getDimension(_prev->_width, dimBlock.x), getDimension(_prev->_height, dimBlock.y), getDimension(_prev->_depth, dimBlock.z));
  convoBackpropErrors<<<dimGrid, dimBlock>>>(_prev->_output, _weight, _deltaW, _bias, _output, _errors, nextErrors, _depth, _height, _width,
      _prev->_depth, _prev->_height, _prev->_width, _spatialExtent, _stride, Layer::_alpha, Layer::_lambda);

  blockX = min(16, _spatialExtent * _spatialExtent);
  blockY = min(16, _depth);
  blockZ = min(64, min(_prev->_depth, 1024 / (blockX * blockY)));
  dim3 dimBlock2(blockX, blockY, blockZ);
  dim3 dimGrid2(getDimension(_spatialExtent * _spatialExtent, blockX), getDimension(_depth, blockY), getDimension(_prev->_depth, blockZ));
  convoBackpropWeight<<<dimGrid2, dimBlock2>>>(_prev->_output, _weight, _deltaW, _bias, _output, _errors, nextErrors, _depth, _height, _width,
      _prev->_depth, _prev->_height, _prev->_width, _spatialExtent, _stride, Layer::_alpha, Layer::_lambda);
}

void ConvolutionalLayer::backProp_cpu(const float_t *nextErrors) {
  int inWidth = _prev->_width, inHeight = _prev->_height, inDepth = _prev->_depth;
  int F = _spatialExtent;

  memset(_errors, 0, _errorSize * sizeof(float_t));

  for (int out = 0; out < _depth; out++) {
    for (int h = 0; h < _height; h++) {
      for (int w = 0; w < _width; w++) {
        for (int in = 0; in < inDepth; in++) {
          for (int y = 0; y < _spatialExtent; y++) {
            for (int x = 0; x < _spatialExtent; x++) {
              int index = in * inWidth * inHeight + (h + y) * inWidth + (x + w);
              //int weightIndex = in * _depth * F * F + out * F * F + (F - 1 - y) * F + (F - 1 - x);
              int weightIndex = in * _depth * F * F + out * F * F + y * F + x;
              _errors[index] += nextErrors[out * _height * _width + h * _width + w]
              * _weight[weightIndex] * activationDerivativeFunction(_prev->_output[index]);
            }
          }
        }
      }
    }
  }
  // update weight
  for (int out = 0; out < _depth; out++) {
    for (int h = 0; h < _height; h++) {
      for (int w = 0; w < _width; w++) {
        int outIndex = out * _width * _height + h * _width + w;
        for (int in = 0; in < inDepth; in++) {
          for (int y = 0; y < F; y++) {
            for (int x = 0; x < F; x++) {
              //int target = in * _depth * F * F + out * F * F + (F - y - 1) * F + (F - x - 1);
              int target = in * _depth * F * F + out * F * F + y * F + x;
              int inH = h * _stride + y;
              int inW = w * _stride + x;
              float_t input = _prev->_output[in * inHeight * inWidth + inH * inWidth + inW];

              float_t delta = Layer::_alpha * input * nextErrors[outIndex] + Layer::_lambda * _deltaW[target];
              _weight[target] -= delta;
              // update momentum
              _deltaW[target] = delta;
            }
          }
          _bias[outIndex] -= Layer::_alpha * nextErrors[outIndex];
        }
      }
    }
  }
}

void ConvolutionalLayer::initWeight() {
  vector<float_t> weight;
  weight.resize(_weightSize);
  vector<float_t> bias;
  bias.resize(_outputSize);
  uniform_rand(weight.begin(), weight.end(), -1, 1);
  uniform_rand(bias.begin(), bias.end(), -1, 1);
  float_t tempW[_weightSize], tempBias[_outputSize];
  for (int i = 0; i < _weightSize; i++) {
    tempW[i] = 1.0 * (rand() - rand())/RAND_MAX;
    /*tempW[i] = weight[i];*/
  }
  for (int i = 0; i < _outputSize; i++) {
    tempBias[i] = 1.0 * (rand() - rand())/RAND_MAX;
    /*tempBias[i] = bias[i];*/
  }

  if (SOLVE_MODE == "GPU") {
    cudaMemcpy(_weight, tempW, sizeof(float_t) * _weightSize, cudaMemcpyHostToDevice);
    cudaMemcpy(_bias, tempBias, sizeof(float_t) * _outputSize, cudaMemcpyHostToDevice);
    cudaMemset(_deltaW, 0, sizeof(float_t) * _weightSize);
  } else {
    memcpy(_weight, tempW, sizeof(float_t) * _weightSize);
    memcpy(_bias, tempBias, sizeof(float_t) * _outputSize);
    memset(_deltaW, 0, sizeof(float_t) * _weightSize);
  }
}

float_t ConvolutionalLayer::sumWeight(int in, int out, int h, int w) {
  int startH = h * _stride;
  int startW = w * _stride;
  int inHeight = _prev->_height;
  int inWidth = _prev->_width;
  float_t result = 0;
  int F = _spatialExtent;
  for (int i = 0; i < F; i++) {
    for (int j = 0; j < F; j++) {
      int index = in * (inHeight * inWidth) + (startH + i) * inWidth + (startW + j); // row startH + i, col startW + j
      float_t input = _prev->_output[index];
      int indexWeight = in * _depth * F * F + out * F * F + i * F + j;
      result += input * _weight[indexWeight];
    }
  }
  return result;
}

__global__ void poolingFeedForward(float_t *prev, float_t *output, int *maxIndex, int outDepth, int outHeight, int outWidth,
    int inDepth, int inHeight, int inWidth, int F, int S) {

  int depth = blockDim.z * blockIdx.z + threadIdx.z;
  int height = blockDim.y * blockIdx.y + threadIdx.y;
  int width = blockDim.x * blockIdx.x + threadIdx.x;
  if (depth >= outDepth || height >= outHeight || width >= outWidth) {
    return;
  }
  int startH = height * S;
  int startW = width * S;
  int outIndex = getIndexDevice(depth, height, width, outHeight, outWidth);
  float_t result = -1000000000;
  int prevIndex = 0;
  for (int i = startH; i < startH + F; i++) {
    for (int j = startW; j < startW + F; j++) {
      int inIndex = getIndexDevice(depth, i, j, inHeight, inWidth);
      if (prev[inIndex] > result) {
        result = prev[inIndex]; 
        prevIndex = inIndex;
      }
    }
  }
  maxIndex[outIndex] = prevIndex;
  output[outIndex] = result;
  /*printf("Pooling feedforward %d %.2f\n", outIndex, result);*/
}

__global__ void poolingBackprop(float_t *nextErrors, float_t *errors, int *maxIndex, int outDepth, int outHeight, int outWidth) {
  int depth = blockDim.z * blockIdx.z + threadIdx.z;
  int height = blockDim.y * blockIdx.y + threadIdx.y;
  int width = blockDim.x * blockIdx.x + threadIdx.x;
  if (depth >= outDepth || height >= outHeight || width >= outWidth) {
    return;
  }
  int outIndex = getIndexDevice(depth, height, width, outHeight, outWidth);
  errors[maxIndex[outIndex]] = nextErrors[outIndex];
}

MaxPoolingLayer::MaxPoolingLayer(int spatialExtent, int stride, Layer *prev):
  Layer(prev->_depth, (prev->_height - spatialExtent)/stride + 1,
      (prev->_width - spatialExtent)/stride + 1,
      spatialExtent, stride, 0, prev) {

    if (SOLVE_MODE == "GPU") {
      cudaMalloc(&_maxIndex, _outputSize * sizeof(int));
    } else {
      _maxIndex = (int *) malloc(_outputSize * sizeof(int));
    }
  }

void MaxPoolingLayer::forward_cpu() {
  for (int d = 0; d < _depth; d++) {
    for (int h = 0; h < _height; h++) {
      for (int w = 0; w < _width; w++) {
        int index = getIndex(d, h, w);
        _output[index] = getMax(d, h, w, index);
      }
    }
  }
}

void MaxPoolingLayer::forward_gpu() {
  int blockX = min(16, _width);
  int blockY = min(16, _height);
  int blockZ = min(_depth, 1024 / (blockX * blockY));
  dim3 dimBlock(blockX, blockY, blockZ);
  dim3 dimGrid(getDimension(_width, dimBlock.x), getDimension(_height, dimBlock.y), getDimension(_depth, dimBlock.z));
  poolingFeedForward<<<dimGrid, dimBlock>>>(_prev->_output, _output, _maxIndex, _depth, _height, _width, _prev->_depth, _prev->_height,
      _prev->_width, _spatialExtent, _stride);
  /*cudaError_t err = cudaGetLastError();*/
  /*if (err != cudaSuccess) {*/
    /*printf("Error: %s\n", cudaGetErrorString(err));*/
  /*}*/
}

void MaxPoolingLayer::feedForward() {
  if (SOLVE_MODE == "GPU") {
    forward_gpu();

  } else {
    forward_cpu();
  }
}

void MaxPoolingLayer::backProp(float_t *nextErrors) {
  if (SOLVE_MODE == "GPU") {
    backProp_gpu(nextErrors);
  } else {
    backProp_cpu(nextErrors);
  }
}

void MaxPoolingLayer::backProp_cpu(float_t *nextErrors) {
  memset(_errors, 0, _errorSize * sizeof(float_t));
  for (int i = 0; i < _outputSize; i++) {
    _errors[_maxIndex[i]] = nextErrors[i];
  } 
}

void MaxPoolingLayer::backProp_gpu(float_t *nextErrors) {
  cudaMemset(_errors, 0, _errorSize * sizeof(float_t));
  int blockX = min(16, _width);
  int blockY = min(16, _height);
  int blockZ = min(_depth, 1024 / (blockX * blockY));
  dim3 dimBlock(blockX, blockY, blockZ);
  dim3 dimGrid(getDimension(_width, dimBlock.x), getDimension(_height, dimBlock.y), getDimension(_depth, dimBlock.z));
  poolingBackprop<<<dimGrid, dimBlock>>>(nextErrors, _errors, _maxIndex, _depth, _height, _width);
}

void MaxPoolingLayer::initWeight() {}

float_t MaxPoolingLayer::getMax(int d, int h, int w, int outIndex) {
  int startH = h * _stride;
  int startW = w * _stride;
  int H = _prev->_height;
  int W = _prev->_width;
  float_t result = -1000000000;
  for (int i = startH; i < startH + _spatialExtent; i++) {
    for (int j = startW; j < startW + _spatialExtent; j++) {
      int index = d * (H * W) + i * W + j;
      if (_prev->_output[index] > result) {
        result = _prev->_output[index]; 
        _maxIndex[outIndex] = index;
      }
    }
  }
  return result;
}

__global__ void fullFeedForward(float_t *prev, float_t *weight, float_t *bias, float_t *output, int outDepth, int inDepth) {
  int out = blockDim.x * blockIdx.x + threadIdx.x;
  if (out >= outDepth) {
    return;
  }
  float_t result = 0;
  for (int in = 0; in < inDepth; in++) {
    result += weight[out * inDepth + in] * prev[in];
  }
  output[out] = activationFunctionDevice(result + bias[out]);
}

__global__ void fullBackPropError(float_t *prev, float_t *weight, float_t *deltaW, float_t *bias, float_t *output, float_t *errors, float_t *nextErrors, 
    int outDepth, int inDepth, float_t alpha, float_t lambda) {
  
  int in = blockDim.x * blockIdx.x + threadIdx.x;
  if (in >= inDepth) {
    return;
  }

  float_t result = 0;
  float_t derivativeInput = activationDerivativeFunctionDevice(prev[in]);
  for (int out = 0; out < outDepth; out++) {
    result += nextErrors[out] * weight[out * inDepth + in] * derivativeInput;
  }
  errors[in] = result;
}

__global__ void fullBackPropWeight(float_t *prev, float_t *weight, float_t *deltaW, float_t *bias, float_t *output, float_t *errors, float_t *nextErrors, 
    int outDepth, int inDepth, float_t alpha, float_t lambda) {
  
  int out = blockDim.y * blockIdx.y + threadIdx.y;
  int in = blockDim.x * blockIdx.x + threadIdx.x;
  if (out >= outDepth || in >= inDepth) {
    return;
  }

  float_t nextE = nextErrors[out];
  int weightIndex = out * inDepth + in;
  float_t delta = alpha * prev[in] * nextE + lambda * deltaW[weightIndex];
  weight[weightIndex] -= delta;
  deltaW[weightIndex] = delta;
  if (in == 0) {
    bias[out] -= alpha * nextE;
  }
}

FullyConnectedLayer::FullyConnectedLayer(int depth, Layer *prev): Layer(depth, 1, 1, 1, 1, 0, prev) {
  initWeight();
}

void FullyConnectedLayer::forward_cpu() {
  int inDepth = _prev->_depth;
  for (int out = 0; out < _depth; out++) {
    float_t result = 0;
    for (int in = 0; in < inDepth; in++) {
      result += _weight[out * inDepth + in] * _prev->_output[in];
    }
    _output[out] = activationFunction(result + _bias[out]);
  }
}

void FullyConnectedLayer::forward_gpu() {
  int threadPerBlock = min(1024, _depth);
  int numBlock = getDimension(_depth, threadPerBlock);
  fullFeedForward<<<numBlock, threadPerBlock>>>(_prev->_output, _weight, _bias, _output, _depth, _prev->_depth);
}

void FullyConnectedLayer::feedForward() {
  if (SOLVE_MODE == "GPU") {
    forward_gpu();
  } else {
    forward_cpu();
  }
}


void FullyConnectedLayer::backProp_cpu(float_t *nextErrors) {
  memset(_errors, 0, _errorSize * sizeof(float_t));
  int inDepth = _prev->_depth;
  for (int in = 0; in < inDepth; in++) {
    float_t result = 0;
    for (int out = 0; out < _depth; out++) {
      result += nextErrors[out] * _weight[inDepth * out + in];
    }
    _errors[in] = result * activationDerivativeFunction(_prev->_output[in]);
  }

  for (int out = 0; out < _depth; out++) {
    for (int in = 0; in < inDepth; in++) {
      int index = out * inDepth + in;
      float_t delta = Layer::_alpha * _prev->_output[in] * nextErrors[out] + Layer::_lambda * _deltaW[index];
      _weight[index] -= delta;
      _deltaW[index] = delta;
    }
    _bias[out] -= Layer::_alpha * nextErrors[out];
  }
}

void FullyConnectedLayer::backProp_gpu(float_t *nextErrors) {
  cudaMemset(_errors, 0, _errorSize * sizeof(float_t));
  int threadPerBlock = min(1024, _prev->_depth);
  int numBlock = getDimension(_prev->_depth, threadPerBlock);
  fullBackPropError<<<numBlock, threadPerBlock>>>
    (_prev->_output, _weight, _deltaW, _bias, _output, _errors, nextErrors, _depth, _prev->_depth, Layer::_alpha, Layer::_lambda);

  int blockX = min(1024, _prev->_depth);
  int blockY = min(_depth, 1024/blockX);
  dim3 dimBlock(blockX, blockY);
  dim3 dimGrid(getDimension(_prev->_depth, blockX), getDimension(_depth, blockY));
  fullBackPropWeight<<<dimGrid, dimBlock>>> (_prev->_output, _weight, _deltaW, _bias, _output, _errors, nextErrors, _depth, _prev->_depth, Layer::_alpha, Layer::_lambda);
}

void FullyConnectedLayer::backProp(float_t *nextErrors) {
  if (SOLVE_MODE == "GPU") {
    backProp_gpu(nextErrors);
  } else {
    backProp_cpu(nextErrors);
  }
}

void FullyConnectedLayer::initWeight() {
  vector<float_t> weight, bias;
  weight.resize(_weightSize);
  bias.resize(_outputSize);
  uniform_rand(weight.begin(), weight.end(), -1, 1);
  uniform_rand(bias.begin(), bias.end(), -1, 1);
  float_t tempW[_weightSize], tempBias[_outputSize];
  for (int i = 0; i < _weightSize; i++) {
    tempW[i] = 1.0 * (rand() - rand())/RAND_MAX;
    /*tempW[i] = weight[i];*/
  }
  for (int i = 0; i < _outputSize; i++) {
    tempBias[i] = 1.0 * (rand() - rand())/RAND_MAX;
    /*tempBias[i] = bias[i];*/
  }

  cout << tempW[1] << ' ' << tempBias[1] << ' ' << _weightSize + _outputSize << endl;
  if (SOLVE_MODE == "GPU") {
    cudaMemcpy(_weight, tempW, sizeof(float_t) * _weightSize, cudaMemcpyHostToDevice);
    cudaMemcpy(_bias, tempBias, sizeof(float_t) * _outputSize, cudaMemcpyHostToDevice);
    cudaMemset(_deltaW, 0, sizeof(float_t) * _weightSize);
  } else {
    memcpy(_weight, tempW, sizeof(float_t) * _weightSize);
    memcpy(_bias, tempBias, sizeof(float_t) * _outputSize);
    memset(_deltaW, 0, sizeof(float_t) * _weightSize);
  }
}

__global__ void outputFeedForward(float_t *prev, float_t *output) {
  int out = blockDim.x * blockIdx.x + threadIdx.x;
  output[out] = prev[out];
}

__global__ void outputBackProp(float_t *output, float_t *errors, int label) {
  int out = blockDim.x * blockIdx.x + threadIdx.x;
  int expected = (out == label) ? 1 : 0;
  float_t predict = output[out];
  errors[out] = (predict - expected);// * activationDerivativeFunctionDevice(predict);
}

OutputLayer::OutputLayer(Layer *prev): Layer(prev->_depth, 1, 1, 0, 0, 0, prev) { }

void OutputLayer::setLabel(int label) {
  _label = label;
}

void OutputLayer::feedForward_cpu() {
  _output = _prev->_output;
}

void OutputLayer::feedForward_gpu() {
  int threadPerBlock = _depth;
  outputFeedForward<<<1, threadPerBlock>>>(_prev->_output, _output);
}

void OutputLayer::feedForward() {
  if (SOLVE_MODE == "GPU") {
    feedForward_gpu();
  } else {
    feedForward_cpu();
  }
}

float_t OutputLayer::getError() {
  float_t err = 0;
  if (SOLVE_MODE == "GPU") {
    float_t temp[_depth];
    cudaMemcpy(temp, _output, _outputSize * sizeof(float_t), cudaMemcpyDeviceToHost);
    for (int i = 0; i < _depth; i++) {
      int expected = (i == _label) ? 1 : 0;
      err += 0.5 * (temp[i] - expected) * (temp[i] - expected);
    }
  } else {
    for (int i = 0; i < _depth; i++) {
      int expected = (i == _label) ? 1 : 0;
      err += 0.5 * (_output[i] - expected) * (_output[i] - expected);
    }
  }

  return err;
}

int OutputLayer::getPredict() {
  int index = 0;
  if (SOLVE_MODE == "GPU") {
    float_t temp[_depth];
    cudaMemcpy(temp, _output, _outputSize * sizeof(float_t), cudaMemcpyDeviceToHost);
    for (int i = 1; i < _depth; i++) if (temp[i] > temp[index]) index = i;
  } else {
    for (int i = 1; i < _depth; i++) if (_output[i] > _output[index]) index = i;
  }
  //for (int i = 0; i < _depth; i++) cout << _output[i] << ' ';
  return index;
}

void OutputLayer::backProp_cpu() {
  memset(_errors, 0, _errorSize * sizeof(float_t));
  for (int i = 0; i < _depth; i++) {
    int expected = (i == _label) ? 1 : 0;
    _errors[i] = (_output[i] - expected); // * activationDerivativeFunction(_prev->_output[i]);
  }
}

void OutputLayer::backProp_gpu() {
  cudaMemset(_errors, 0, _errorSize * sizeof(float_t));
  int threadPerBlock = _depth;
  outputBackProp<<<1, threadPerBlock>>> (_output, _errors, _label);
  /*float_t temp[_errorSize];*/
  /*cudaMemcpy(temp, _errors, _errorSize * sizeof(float_t), cudaMemcpyDeviceToHost);*/
  /*for (int i = 0; i < _errorSize; i++) cout << temp[i] << ' ';*/
  /*cout << endl;*/
}
    
void OutputLayer::backProp(float_t *nextErrors) {
  if (SOLVE_MODE == "GPU") {
    backProp_gpu();
  } else {
    backProp_cpu();
  }
}

