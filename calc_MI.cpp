#include <iostream>
#include <cmath>
#include <vector>
using namespace std;

class Histogram {  //This is obviously just a bad vector wrapper, this is only temporary
public:
  double probabilityOf(int x);
private:
  vector<double> probabilities;
};

Histogram::Histogram() {}

Histogram::~Histogram() 
{
  delete probabilities;
}

double Histogram::probabilityOf(int x)
{
  return probabilities[x];
}

double mutual_information(const Histogram& x, const Histogram& y, const Histogram& joint)
{
  double result = 0;
  Histogram::size_type x_sz = x.size();  //wrong, these should just be iterators
  Histogram::size_type y_sz = y.size();
  for (Histogram::size_type i = 0; i < x_sz; i++) {
    for (Histogram::size_type j = 0; j < y_sz; j++) {
      result += joint.probabilityOf(i,j) * log( joint.probabilityOf(i,j) / 
						(x.probabilityOf(i) * y.probabilityOf(j)));
    }
  }
  return result;
}
