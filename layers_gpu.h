#include <iostream>
#include <vector>

using namespace std;

class Layer {
  public:
    Layer(int depth, int height, int width, int spatialExtent, int stride,
        int zeroPadding, Layer *prev);
    virtual  void feedForward() = 0;
    virtual void backProp(float_t *nextErrors) = 0;
    int _width, _height, _depth, _spatialExtent, _stride, _zeroPadding;
    static float_t _alpha, _lambda;
    float_t* _output;
    Layer *_prev;
    float_t* _errors;

    float_t* _weight;
    float_t* _bias;
    float_t* _deltaW;
    int _weightSize;
    int _outputSize;
    int _errorSize;
    int getIndex(int d, int h, int w);
    static void setLearning(float_t alpha, float_t lambda);
};

class Input: public Layer {
  public:
    Input(int depth, int height, int width);
    void setOutput(vector<float_t> &output);
    void feedForward();
    void backProp(float_t *nextErrors);
};

class ConvolutionalLayer: public Layer {
  public:
    ConvolutionalLayer(int depth, int spatialExtent, int stride, int zeroPadding, Layer *prev);

    void feedForward();

    void forward_cpu();

    void forward_gpu();

    void backProp(float_t *nextErrors);

    void backProp_gpu(float_t *nextErrors);

    void backProp_cpu(const float_t *nextErrors);

    void initWeight();

  private:
    float_t sumWeight(int in, int out, int h, int w);
};

class MaxPoolingLayer: public Layer {
  public:
    MaxPoolingLayer(int spatialExtent, int stride, Layer *prev);

    void forward_cpu();

    void forward_gpu();

    void feedForward();

    void backProp(float_t *nextErrors);

    void backProp_cpu(float_t *nextErrors);

    void backProp_gpu(float_t *nextErrors);

    void initWeight();
    private:
      int *_maxIndex;
      float_t getMax(int d, int h, int w, int outIndex);
};

class FullyConnectedLayer: public Layer {
  public:
    FullyConnectedLayer(int depth, Layer *prev);

    void forward_cpu();

    void forward_gpu();

    void feedForward();

    void backProp_cpu(float_t *nextErrors);

    void backProp_gpu(float_t *nextErrors);

    void backProp(float_t *nextErrors);

    void initWeight();
};

class OutputLayer: public Layer {
  public:
    OutputLayer(Layer *prev);

    void setLabel(int label);

    void feedForward();

    void feedForward_cpu();

    void feedForward_gpu();

    float_t getError();

    int getPredict();

    void backProp_cpu();

    void backProp_gpu();
    
    void backProp(float_t *nextErrors);

    int _label;
};

