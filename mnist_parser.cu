#include <iostream>
#include <vector>
#include <assert.h>
#include <fstream>

using namespace std;

struct Image {
  vector< vector<float_t> > img;// a image is represented by a 2-dimension vector  
  size_t size; // width or height

  // construction
  Image(size_t size_, vector< vector<float_t> > img_) :img(img_), size(size_){}

  // display the image
  void display(){
    for (size_t i = 0; i < size; i++){
      for (size_t j = 0; j < size; j++){
        if (img[i][j] > 200)
          cout << 1;
        else
          cout << 0;
      }
      cout << endl;
    }
  }

  // up size to 32, make up with 0
  void upto_32(){
    assert(size < 32);

    vector<float_t> row(32, 0);

    for (size_t i = 0; i < size; i++){
      img[i].insert(img[i].begin(), 0);
      img[i].insert(img[i].begin(), 0);
      img[i].push_back(0);
      img[i].push_back(0);
    }
    img.insert(img.begin(), row);
    img.insert(img.begin(), row);
    img.push_back(row);
    img.push_back(row);

    size = 32;
  }

  vector<float_t> extend(){
    vector<float_t> v;
    for (size_t i = 0; i < size; i++){
      for (size_t j = 0; j < size; j++){
        v.push_back(img[i][j]);
      }
    }
    return v;
  }
};

typedef Image* Img;

struct Sample
{
  uint8_t label; // label for a specific digit
  vector<float_t> image;
  Sample(float_t label_, vector<float_t> image_) :label(label_), image(image_){}
};

class Mnist_Parser
{
  public:
    Mnist_Parser(string data_path) :
      test_img_fname(data_path + "/t10k-images-idx3-ubyte"),
      test_lbl_fname(data_path + "/t10k-labels-idx1-ubyte"),
      train_img_fname(data_path + "/train-images-idx3-ubyte"),
      train_lbl_fname(data_path + "/train-labels-idx1-ubyte"){}

    vector<Sample*> load_testing(){
      test_sample = load(test_img_fname, test_lbl_fname);
      return test_sample;
    }

    vector<Sample*> load_training(){
      train_sample = load(train_img_fname, train_lbl_fname);
      return train_sample;
    }

    void test(){
      srand((int)time(0));
      size_t i = (int)(rand());
      cout << i << endl;
      cout << (int)test_sample[i]->label << endl;
      //test_sample[i]->image->display();

      size_t j = (int)(rand() * 60000);
      cout << (int)(train_sample[i]->label) << endl;
      //train_sample[i]->image->display();

    }

    // vector for store test and train samples
    vector<Sample*> test_sample;
    vector<Sample*> train_sample;

  private:
    vector<Sample*> load(string fimage, string flabel){
      ifstream in;
      in.open(fimage, ios::binary | ios::in);
      if (!in.is_open()){
        cout << "file opened failed." << endl;
      }

      uint32_t magic = 0;
      uint32_t number = 0;
      uint32_t rows = 0;
      uint32_t cols = 0;

      in.read((char*)&magic, sizeof(uint32_t));
      in.read((char*)&number, sizeof(uint32_t));
      in.read((char*)&rows, sizeof(uint32_t));
      in.read((char*)&cols, sizeof(uint32_t));

      assert(swapEndien_32(magic) == 2051);
      cout << "number:" << swapEndien_32(number) << endl;
      assert(swapEndien_32(rows) == 28);
      assert(swapEndien_32(cols) == 28);

      vector< float_t> row;
      vector< vector<float_t> > img;
      vector<Img> images;

      uint8_t pixel = 0;
      size_t col_index = 0;
      size_t row_index = 0;
      while (!in.eof()){
        in.read((char*)&pixel, sizeof(uint8_t));
        col_index++;
        row.push_back((float_t)pixel/256.0);
        //row.push_back((float_t)pixel);
        if (col_index == 28){
          img.push_back(row);

          row.clear();
          col_index = 0;

          row_index++;
          if (row_index == 28){
            Img i = new Image(28, img);
            i->upto_32();
            //i->display();
            images.push_back(i);
            img.clear();
            row_index = 0;
          }
        }
      }

      in.close();

      assert(images.size() == swapEndien_32(number));

      //label
      in.open(flabel, ios::binary | ios::in);
      if (!in.is_open()){
        cout << "failed opened label file";
      }

      in.read((char*)&magic, sizeof(uint32_t));
      in.read((char*)&number, sizeof(uint32_t));

      assert(2049 == swapEndien_32(magic));
      assert(swapEndien_32(number) == images.size());

      vector<uint8_t>  labels;

      uint8_t label;
      while (!in.eof())
      {
        in.read((char*)&label, sizeof(uint8_t));
        //cout << (int)label << endl;
        labels.push_back(label);
      }

      vector<Sample*> samples;
      for (int i = 0; i < swapEndien_32(number); i++){
        samples.push_back(new Sample(labels[i], images[i]->extend()));
      }

      cout << "Loading complete" << endl;
      in.close();
      return samples;
    }

    // reverse endien for uint32_t
    uint32_t swapEndien_32(uint32_t value){
      return ((value & 0x000000FF) << 24) |
        ((value & 0x0000FF00) << 8) |
        ((value & 0x00FF0000) >> 8) |
        ((value & 0xFF000000) >> 24);
    }

    // filename for mnist data set
    string test_img_fname;
    string test_lbl_fname;
    string train_img_fname;
    string train_lbl_fname;
};
